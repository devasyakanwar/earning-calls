"""
Baseline Model Comparison with Cross-Validation & Confidence Intervals

Trains and compares baselines using time-series cross-validation:
    1. Text-only:   sentiment, uncertainty, specificity, linguistic complexity
    2. Price-only:  normalized price baseline
    3. Combined:    text + structural + price
    4. Random:      shuffled labels (establishes noise floor)

Uses 5-fold expanding-window time-series CV for honest evaluation.
Reports confidence intervals on all metrics.

Outputs:
    outputs/baseline_comparison.json   — metrics with CIs for all models
    outputs/feature_importance.json    — top-20 features per model
"""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation, returning 0 on failure."""
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        ic, _ = spearmanr(y_true, y_pred)
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        return 0.0


def clean_features(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Replace NaN/Inf with 0 in feature columns."""
    for col in feature_cols:
        if col in df.columns and df[col].dtype in (pl.Float32, pl.Float64):
            df = df.with_columns(
                pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite())
                .then(0.0)
                .otherwise(pl.col(col))
                .alias(col)
            )
    return df


def get_feature_importance(model, feature_names: list[str], top_n: int = 20) -> list[dict]:
    """Extract top-N feature importances from a LightGBM model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": int(importances[i])}
        for i in indices if importances[i] > 0
    ]


def compute_confidence_interval(values: list[float], confidence: float = 0.95) -> dict:
    """Compute mean and confidence interval from a list of fold results."""
    arr = np.array(values)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean, "n_folds": len(arr)}
    
    from scipy import stats
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(len(arr))
    ci = stats.t.interval(confidence, df=len(arr)-1, loc=mean, scale=se)
    return {
        "mean": mean,
        "std": std,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "n_folds": len(arr),
    }


# ---------------------------------------------------------------------------
# Time-Series Cross-Validation
# ---------------------------------------------------------------------------

def time_series_cv_splits(n: int, n_splits: int = 5, min_train_frac: float = 0.3) -> list[tuple]:
    """
    Generate expanding-window time-series CV splits.
    
    Returns list of (train_indices, test_indices) tuples.
    """
    min_train = max(int(n * min_train_frac), 5)
    test_size = max((n - min_train) // n_splits, 1)
    
    splits = []
    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = min(test_start + test_size, n)
        if test_start >= n:
            break
        train_idx = list(range(test_start))
        test_idx = list(range(test_start, test_end))
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits


# ---------------------------------------------------------------------------
# Model training with CV
# ---------------------------------------------------------------------------

def train_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    n_splits: int = 5,
) -> dict:
    """Train LightGBM regressor with time-series CV and return metrics with CIs."""
    n = len(X)
    splits = time_series_cv_splits(n, n_splits)
    
    fold_rmse = []
    fold_ic = []
    fold_mae = []
    all_importances = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(2, len(X_train) // 10),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))
        fold_ic.append(safe_spearman(y_test, y_pred))
        fold_mae.append(float(np.mean(np.abs(y_test - y_pred))))
        all_importances.append(model.feature_importances_)

    # Aggregate importances across folds
    avg_importance = np.mean(all_importances, axis=0)
    indices = np.argsort(avg_importance)[::-1][:20]
    top_features = [
        {"feature": feature_names[i], "importance": float(avg_importance[i])}
        for i in indices if avg_importance[i] > 0
    ]

    result = {
        "model": model_name,
        "task": "volatility_regression",
        "target": "realized_vol_5d",
        "total_samples": n,
        "n_features": len(feature_names),
        "rmse": compute_confidence_interval(fold_rmse),
        "ic": compute_confidence_interval(fold_ic),
        "mae": compute_confidence_interval(fold_mae),
        "top_features": top_features,
    }

    logger.info(
        "  [%s] Regression → RMSE=%.6f±%.6f, IC=%.4f±%.4f",
        model_name, result["rmse"]["mean"], result["rmse"]["std"],
        result["ic"]["mean"], result["ic"]["std"],
    )
    return result


def train_classification_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_name: str,
    n_splits: int = 5,
) -> dict:
    """Train LightGBM classifier with time-series CV and return metrics with CIs."""
    n = len(X)
    splits = time_series_cv_splits(n, n_splits)
    
    fold_acc = []
    fold_f1 = []
    fold_auc = []
    all_importances = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(2, len(X_train) // 10),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        fold_acc.append(float(accuracy_score(y_test, y_pred)))
        fold_f1.append(float(f1_score(y_test, y_pred, zero_division=0)))
        
        if len(np.unique(y_test)) > 1:
            fold_auc.append(float(roc_auc_score(y_test, y_proba)))
        else:
            fold_auc.append(0.5)
        
        all_importances.append(model.feature_importances_)

    # Average importances
    avg_importance = np.mean(all_importances, axis=0)
    indices = np.argsort(avg_importance)[::-1][:20]
    top_features = [
        {"feature": feature_names[i], "importance": float(avg_importance[i])}
        for i in indices if avg_importance[i] > 0
    ]

    result = {
        "model": model_name,
        "task": "direction_classification",
        "target": "return_1d_direction",
        "total_samples": n,
        "n_features": len(feature_names),
        "accuracy": compute_confidence_interval(fold_acc),
        "f1": compute_confidence_interval(fold_f1),
        "auc": compute_confidence_interval(fold_auc),
        "top_features": top_features,
    }

    logger.info(
        "  [%s] Classification → Acc=%.4f±%.4f, F1=%.4f±%.4f, AUC=%.4f±%.4f",
        model_name,
        result["accuracy"]["mean"], result["accuracy"]["std"],
        result["f1"]["mean"], result["f1"]["std"],
        result["auc"]["mean"], result["auc"]["std"],
    )
    return result


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

def run_comparison(project_root: Path) -> None:
    """Run the full baseline comparison with cross-validation."""
    processed = project_root / "data" / "processed"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load text + market dataset
    tm_path = processed / "text_market_dataset.parquet"
    if not tm_path.exists():
        logger.error("text_market_dataset.parquet not found. Run multimodal_join.py first.")
        return

    df = pl.read_parquet(tm_path)
    logger.info("Loaded text_market dataset: %d rows, %d columns", len(df), len(df.columns))

    # -----------------------------------------------------------------------
    # Identify feature groups
    # -----------------------------------------------------------------------
    meta_cols = {
        "call_id", "ticker", "call_date", "close_t0", "close_t1", "close_t5",
        "return_1d", "return_5d", "realized_vol_1d", "realized_vol_5d",
        "earnings_surprise", "abnormal_return_1d", "spy_return",
    }

    text_feature_cols = [
        c for c in df.columns
        if c not in meta_cols and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    logger.info("Text feature columns (%d): %s...", len(text_feature_cols), text_feature_cols[:5])

    # -----------------------------------------------------------------------
    # Prepare targets
    # -----------------------------------------------------------------------
    reg_target = "realized_vol_5d"
    # Use abnormal returns for direction classification if available
    return_col = "abnormal_return_1d" if "abnormal_return_1d" in df.columns else "return_1d"
    logger.info("Direction target: %s", return_col)
    df = df.with_columns(
        (pl.col(return_col) > 0).cast(pl.Int32).alias("return_1d_direction")
    )
    cls_target = "return_1d_direction"

    # Filter valid rows
    df = df.filter(
        pl.col(reg_target).is_not_null()
        & pl.col(reg_target).is_finite()
        & pl.col("return_1d").is_not_null()
    )

    # Clean features
    df = clean_features(df, text_feature_cols)

    # Sort chronologically (important for time-series CV)
    df = df.sort("call_date")
    n = len(df)
    logger.info("After filtering: %d valid calls", n)

    if n < 10:
        logger.error("Too few samples (%d) for meaningful comparison. Need at least 10.", n)
        return

    # -----------------------------------------------------------------------
    # Prepare numpy arrays
    # -----------------------------------------------------------------------
    X_text = df.select(text_feature_cols).fill_null(0.0).to_numpy()
    X_text = np.nan_to_num(X_text, nan=0.0, posinf=0.0, neginf=0.0)

    y_reg = df[reg_target].to_numpy()
    y_cls = df[cls_target].to_numpy()

    n_folds = min(5, max(2, n // 5))  # Adapt folds to dataset size
    logger.info("Using %d-fold time-series CV on %d samples", n_folds, n)

    # -----------------------------------------------------------------------
    # Model 1: Text-Only
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model 1: Text-Only Features (%d features)", len(text_feature_cols))
    logger.info("=" * 60)

    results = []

    results.append(train_regression_cv(X_text, y_reg, text_feature_cols, "Text-Only", n_folds))
    results.append(train_classification_cv(X_text, y_cls, text_feature_cols, "Text-Only", n_folds))

    # -----------------------------------------------------------------------
    # Model 2: Price-Only
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model 2: Price-Only Baseline")
    logger.info("=" * 60)

    if "close_t0" in df.columns:
        df_p = df.with_columns(
            (pl.col("close_t0") / pl.col("close_t0").mean()).alias("price_norm")
        )
        X_price = df_p.select(["price_norm"]).fill_null(1.0).to_numpy()
        X_price = np.nan_to_num(X_price, nan=1.0)

        results.append(train_regression_cv(X_price, y_reg, ["price_norm"], "Price-Only", n_folds))
        results.append(train_classification_cv(X_price, y_cls, ["price_norm"], "Price-Only", n_folds))

    # -----------------------------------------------------------------------
    # Model 3: Text + Structural + Price
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model 3: Text + Structural + Price Combined")
    logger.info("=" * 60)

    combined_cols = text_feature_cols.copy()
    if "close_t0" in df.columns:
        df = df.with_columns(
            (pl.col("close_t0") / pl.col("close_t0").mean()).alias("price_norm")
        )
        combined_cols.append("price_norm")

    X_combined = df.select(combined_cols).fill_null(0.0).to_numpy()
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

    results.append(train_regression_cv(X_combined, y_reg, combined_cols, "Text+Structural+Price", n_folds))
    results.append(train_classification_cv(X_combined, y_cls, combined_cols, "Text+Structural+Price", n_folds))

    # -----------------------------------------------------------------------
    # Model 4: Random Baseline (noise floor)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model 4: Random Baseline (Noise Floor)")
    logger.info("=" * 60)

    np.random.seed(42)
    X_random = np.random.randn(n, 10)
    random_cols = [f"random_{i}" for i in range(10)]

    results.append(train_regression_cv(X_random, y_reg, random_cols, "Random-Baseline", n_folds))
    results.append(train_classification_cv(X_random, y_cls, random_cols, "Random-Baseline", n_folds))

    # -----------------------------------------------------------------------
    # Model 5: Delta Features Only (QoQ changes)
    # -----------------------------------------------------------------------
    delta_cols = [c for c in text_feature_cols if c.endswith(("_delta", "_pct_change", "_z_score"))]
    if delta_cols:
        logger.info("=" * 60)
        logger.info("Model 5: Sentiment Delta Features Only (%d features)", len(delta_cols))
        logger.info("=" * 60)

        X_delta = df.select(delta_cols).fill_null(0.0).to_numpy()
        X_delta = np.nan_to_num(X_delta, nan=0.0, posinf=0.0, neginf=0.0)

        results.append(train_regression_cv(X_delta, y_reg, delta_cols, "Delta-Only", n_folds))
        results.append(train_classification_cv(X_delta, y_cls, delta_cols, "Delta-Only", n_folds))
    else:
        logger.info("No delta features found. Skipping Delta-Only model.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY (with 95%% Confidence Intervals)")
    logger.info("=" * 60)

    for r in results:
        if r["task"] == "volatility_regression":
            m = r["rmse"]
            logger.info(
                "  %-30s RMSE=%.6f [%.6f, %.6f]",
                r["model"] + " (reg)", m["mean"], m["ci_low"], m["ci_high"],
            )
        else:
            m = r["accuracy"]
            logger.info(
                "  %-30s Acc=%.4f [%.4f, %.4f]",
                r["model"] + " (cls)", m["mean"], m["ci_low"], m["ci_high"],
            )

    # Save
    with open(output_dir / "baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results to: %s", output_dir / "baseline_comparison.json")

    # Feature importance
    importance_summary = {
        r["model"] + "_" + r["task"]: r.get("top_features", [])
        for r in results
    }
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance_summary, f, indent=2)
    logger.info("Saved feature importance to: %s", output_dir / "feature_importance.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    project_root = Path(__file__).resolve().parent.parent.parent
    run_comparison(project_root)


if __name__ == "__main__":
    main()
