"""
Phase 4: Advanced Modeling — LightGBM + PCA Baseline

Trains LightGBM models with model serialization for downstream use
(backtesting, inference, dashboard).

Architecture:
    1. Load multimodal or text_market dataset
    2. PCA on high-dim features if needed
    3. Train LightGBM Regressor (Volatility) + Classifier (Direction)
    4. Save trained models (joblib) for walk-forward backtesting
    5. Evaluate and save results
"""

import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def train_lightgbm():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed = project_root / "data" / "processed"
    outputs = project_root / "outputs" / "models"
    outputs.mkdir(parents=True, exist_ok=True)

    # 1. Try multimodal first, fall back to text_market
    df_path = processed / "multimodal_dataset.parquet"
    source = "multimodal"
    if not df_path.exists():
        df_path = processed / "text_market_dataset.parquet"
        source = "text_market"
        if not df_path.exists():
            logger.error("No dataset found. Run multimodal_join.py first.")
            return

    df = pl.read_parquet(df_path)
    logger.info("Using %s dataset: %d rows, %d columns", source, len(df), len(df.columns))

    # 2. Identify feature groups
    meta_cols = {
        "call_id", "ticker", "call_date", "close_t0", "close_t1", "close_t5",
        "return_1d", "return_5d", "realized_vol_1d", "realized_vol_5d",
        "earnings_surprise", "n_segments", "has_real_audio", "is_aligned",
        "data_source", "return_1d_direction",
        "abnormal_return_1d", "spy_return",
    }

    if source == "multimodal":
        audio_cols = [c for c in df.columns if "_audio" in c or "wav2vec2" in c or "prosody" in c or "opensmile" in c]
        text_cols = [c for c in df.columns if any(k in c for k in ["sentiment", "uncertainty", "forward_looking", "hedging", "specificity", "linguistic"])]
        interaction_cols = [c for c in df.columns if any(k in c for k in ["divergence", "pressure", "qa_", "response_length"])]
        text_cols = [c for c in text_cols if c not in interaction_cols]
        all_feature_cols = audio_cols + text_cols + interaction_cols
    else:
        all_feature_cols = [
            c for c in df.columns
            if c not in meta_cols and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        ]
        audio_cols = []
        text_cols = all_feature_cols
        interaction_cols = []

    logger.info("Features: %d total (%d audio, %d text, %d interaction)",
                len(all_feature_cols), len(audio_cols), len(text_cols), len(interaction_cols))

    # Filter valid rows
    df = df.filter(
        pl.col("return_1d").is_not_null()
        & pl.col("realized_vol_5d").is_not_null()
        & pl.col("realized_vol_5d").is_finite()
    )

    # 3. Prepare features
    X = df.select(all_feature_cols).fill_null(0.0).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = all_feature_cols

    # Apply PCA if too many features relative to samples
    if X.shape[1] > X.shape[0] * 2:
        n_components = min(32, X.shape[0] - 1, X.shape[1])
        logger.info("Applying PCA: %d -> %d components", X.shape[1], n_components)
        pca = PCA(n_components=n_components, random_state=42)
        X = pca.fit_transform(X)
        feature_names = [f"pca_{i}" for i in range(n_components)]
        # Save PCA for inference
        joblib.dump(pca, outputs / "pca_model.joblib")

    # Targets
    y_vol = df["realized_vol_5d"].to_numpy()
    # Use abnormal returns for direction if available
    return_col = "abnormal_return_1d" if "abnormal_return_1d" in df.columns else "return_1d"
    y_ret = df[return_col].to_numpy()
    y_dir = (y_ret > 0).astype(int)
    logger.info("Direction target: %s", return_col)

    # 4. Chronological split
    df = df.with_columns(pl.col("call_date").cast(pl.Utf8))
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_idx = np.arange(train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    X_train, y_vol_train, y_dir_train = X[train_idx], y_vol[train_idx], y_dir[train_idx]
    X_val, y_vol_val, y_dir_val = X[val_idx], y_vol[val_idx], y_dir[val_idx]
    X_test, y_vol_test, y_dir_test = X[test_idx], y_vol[test_idx], y_dir[test_idx]

    logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    # 5. Train Volatility Regressor with early stopping
    logger.info("Training Volatility Regressor (LightGBM)...")
    reg = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=max(2, len(X_train) // 10),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    reg.fit(
        X_train, y_vol_train,
        eval_set=[(X_val, y_vol_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(15, verbose=False)],
    )

    # 6. Train Direction Classifier
    logger.info("Training Direction Classifier (LightGBM)...")
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=max(2, len(X_train) // 10),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(
        X_train, y_dir_train,
        eval_set=[(X_val, y_dir_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(15, verbose=False)],
    )

    # 7. Save models for downstream use
    joblib.dump(reg, outputs / "lgbm_regressor.joblib")
    joblib.dump(clf, outputs / "lgbm_classifier.joblib")
    logger.info("Saved models to: %s", outputs)

    # Also save feature names for inference
    with open(outputs / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # 8. Evaluate on Test Set
    vol_pred = reg.predict(X_test)
    dir_pred = clf.predict(X_test)
    dir_proba = clf.predict_proba(X_test)[:, 1]

    rmse = np.sqrt(mean_squared_error(y_vol_test, vol_pred))
    acc = accuracy_score(y_dir_test, dir_pred)

    ic, _ = pearsonr(vol_pred, y_vol_test) if len(y_vol_test) > 1 else (0.0, 0.0)

    results = {
        "mode": "lightgbm",
        "source_dataset": source,
        "features": len(feature_names),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "rmse": float(rmse),
        "ic": float(ic),
        "accuracy": float(acc),
        "params": {
            "n_features": len(feature_names),
            "n_audio": len(audio_cols),
            "n_text": len(text_cols),
            "n_interaction": len(interaction_cols),
        }
    }

    logger.info("=" * 40)
    logger.info("LIGHTGBM TEST RESULTS:")
    logger.info("Samples:  train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))
    logger.info("RMSE:     %.6f", rmse)
    logger.info("IC:       %.4f", ic)
    logger.info("Accuracy: %.4f", acc)
    logger.info("=" * 40)

    # Save results
    with open(outputs / "results_lightgbm.json", "w") as f:
        json.dump(results, f, indent=4)

    # Feature importance
    importance = pl.DataFrame({
        "feature": feature_names,
        "importance": reg.feature_importances_,
    }).sort("importance", descending=True)

    logger.info("Top Features:\n%s", importance.head(10))

    return results


if __name__ == "__main__":
    train_lightgbm()
