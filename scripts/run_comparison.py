"""
Model Comparison Runner — Text-Only vs Audio+Hybrid

Runs training, backtesting, and visualization separately for:
  1. Text-Only model (current baseline)
  2. Multimodal model (text + audio features)

Saves results to separate directories for side-by-side comparison.
"""

import json
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def identify_feature_groups(df: pl.DataFrame):
    """Split columns into text-only vs audio vs meta."""
    meta_cols = {
        "call_id", "ticker", "call_date", "close_t0", "close_t1", "close_t5",
        "return_1d", "return_5d", "realized_vol_1d", "realized_vol_5d",
        "earnings_surprise", "n_segments", "return_1d_direction",
        "abnormal_return_1d", "spy_return", "has_real_audio", "is_aligned",
        "data_source",
    }
    numeric_types = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)

    text_cols = [
        c for c in df.columns
        if c not in meta_cols
        and df[c].dtype in numeric_types
        and not any(k in c for k in ["wav2vec2", "opensmile", "prosody", "whisper", "pitch", "energy_variance", "speech_rate", "voice_stability"])
    ]
    audio_cols = [
        c for c in df.columns
        if c not in meta_cols
        and df[c].dtype in numeric_types
        and c not in text_cols
    ]
    return text_cols, audio_cols, meta_cols


def run_single_model(df, feature_cols, model_name, output_dir, return_col="abnormal_return_1d"):
    """Train + backtest a single model variant and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.filter(
        pl.col("return_1d").is_not_null()
        & pl.col("realized_vol_5d").is_not_null()
        & pl.col("realized_vol_5d").is_finite()
    ).sort("call_date")

    n = len(df)
    if n < 20:
        logger.error("[%s] Not enough data: %d rows", model_name, n)
        return None

    X_all = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    use_abnormal = return_col in df.columns
    ret_col = return_col if use_abnormal else "return_1d"
    y_dir_all = (df[ret_col].to_numpy() > 0).astype(int)
    y_vol_all = df["realized_vol_5d"].to_numpy().astype(np.float64)
    actual_returns = df["return_1d"].to_numpy().astype(np.float64)
    tickers = df["ticker"].to_list() if "ticker" in df.columns else ["UNK"] * n
    dates = df["call_date"].to_list()
    call_ids = df["call_id"].to_list()

    # --- Static train/test split for metrics ---
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
    y_dir_train, y_dir_val, y_dir_test = y_dir_all[:train_end], y_dir_all[train_end:val_end], y_dir_all[val_end:]
    y_vol_train, y_vol_val, y_vol_test = y_vol_all[:train_end], y_vol_all[train_end:val_end], y_vol_all[val_end:]

    # Train regressor
    reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                            min_child_samples=max(2, len(X_train)//10), subsample=0.8,
                            colsample_bytree=0.8, random_state=42, verbosity=-1)
    reg.fit(X_train, y_vol_train, eval_set=[(X_val, y_vol_val)], eval_metric='rmse',
            callbacks=[lgb.early_stopping(15, verbose=False)])

    # Train classifier
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                             min_child_samples=max(2, len(X_train)//10), subsample=0.8,
                             colsample_bytree=0.8, random_state=42, verbosity=-1)
    clf.fit(X_train, y_dir_train, eval_set=[(X_val, y_dir_val)], eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(15, verbose=False)])

    vol_pred = reg.predict(X_test)
    dir_pred = clf.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_vol_test, vol_pred)))
    acc = float(accuracy_score(y_dir_test, dir_pred))
    ic, _ = pearsonr(vol_pred, y_vol_test) if len(y_vol_test) > 1 else (0.0, 0.0)

    # Feature importance
    importance = sorted(zip(feature_cols, reg.feature_importances_.tolist()),
                        key=lambda x: x[1], reverse=True)[:20]

    # --- Walk-forward backtest ---
    min_train = max(int(n * 0.5), 10)
    all_trades = []

    for t in range(min_train, n):
        bt_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=7,
                                     min_child_samples=max(2, t//10), random_state=42, verbosity=-1)
        bt_clf.fit(X_all[:t], y_dir_all[:t])
        pred_proba = bt_clf.predict_proba(X_all[t:t+1])[0, 1]

        TC = 0.001
        for thresh_name, buy_thresh, sell_thresh in [("medium", 0.55, 0.45), ("high", 0.60, 0.40), ("ultra", 0.65, 0.35)]:
            sig = 1 if pred_proba > buy_thresh else (-1 if pred_proba < sell_thresh else 0)
            pnl = sig * float(actual_returns[t]) - (TC if sig != 0 else 0)
            all_trades.append({
                "tier": thresh_name, "date": str(dates[t]), "call_id": call_ids[t],
                "ticker": tickers[t], "pred_proba": float(pred_proba),
                "signal": sig, "actual_ret": float(actual_returns[t]), "pnl": pnl,
                "actual_direction": int(y_dir_all[t]),
            })

    # Compute tier metrics
    tier_results = {}
    for tier in ["medium", "high", "ultra"]:
        tier_trades = [t for t in all_trades if t["tier"] == tier and t["signal"] != 0]
        if tier_trades:
            rets = np.array([t["pnl"] for t in tier_trades])
            correct = sum(1 for t in tier_trades if (t["signal"]>0 and t["actual_ret"]>0) or (t["signal"]<0 and t["actual_ret"]<0))
            cum = np.cumprod(1 + rets) - 1
            sharpe = (np.mean(rets) / max(np.std(rets, ddof=1), 1e-8)) * np.sqrt(252) if len(rets) > 1 else 0
            tier_results[tier] = {
                "n_trades": len(tier_trades), "hit_rate": correct / len(tier_trades),
                "total_return": float(cum[-1]), "sharpe": float(sharpe),
            }
        else:
            tier_results[tier] = {"n_trades": 0, "hit_rate": 0, "total_return": 0, "sharpe": 0}

    # Top quintile
    medium_all = [t for t in all_trades if t["tier"] == "medium"]
    sorted_by_conf = sorted(medium_all, key=lambda t: abs(t["pred_proba"] - 0.5), reverse=True)
    top_q = sorted_by_conf[:max(1, len(sorted_by_conf)//5)]
    for t in top_q:
        if t["signal"] == 0:
            t["signal"] = 1 if t["pred_proba"] > 0.5 else -1
            t["pnl"] = t["signal"] * t["actual_ret"] - 0.001
    top_q_active = [t for t in top_q if t["signal"] != 0]
    if top_q_active:
        rets = np.array([t["pnl"] for t in top_q_active])
        correct = sum(1 for t in top_q_active if (t["signal"]>0 and t["actual_ret"]>0) or (t["signal"]<0 and t["actual_ret"]<0))
        cum = np.cumprod(1 + rets) - 1
        sharpe = (np.mean(rets) / max(np.std(rets, ddof=1), 1e-8)) * np.sqrt(252) if len(rets) > 1 else 0
        tier_results["top_quintile"] = {
            "n_trades": len(top_q_active), "hit_rate": correct/len(top_q_active),
            "total_return": float(cum[-1]), "sharpe": float(sharpe),
        }

    results = {
        "model_name": model_name,
        "n_features": len(feature_cols),
        "n_samples": n,
        "rmse": rmse, "ic": float(ic), "accuracy": acc,
        "top_features": [{"feature": f, "importance": i} for f, i in importance],
        "confidence_tiers": tier_results,
        "trades": [t for t in all_trades if t["tier"] == "medium"],
    }

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save equity curve data for plotting
    medium_trades = [t for t in all_trades if t["tier"] == "medium" and t["signal"] != 0]
    if medium_trades:
        cum_pnl = np.cumprod(1 + np.array([t["pnl"] for t in medium_trades])) - 1
        cum_bench = np.cumprod(1 + np.array([t["actual_ret"] for t in medium_trades])) - 1
        curve_df = pl.DataFrame({
            "trade_idx": list(range(len(medium_trades))),
            "strategy_cum_return": cum_pnl.tolist(),
            "benchmark_cum_return": cum_bench.tolist(),
        })
        curve_df.write_parquet(output_dir / "equity_curve.parquet")

    logger.info("[%s] IC=%.4f, Acc=%.4f, Features=%d", model_name, ic, acc, len(feature_cols))
    for tier, m in tier_results.items():
        logger.info("  %-20s Trades=%d, Hit=%.1f%%, Sharpe=%.2f, Return=%.1f%%",
                    tier, m["n_trades"], m["hit_rate"]*100, m["sharpe"], m["total_return"]*100)

    return results


def main():
    project_root = Path(__file__).resolve().parent.parent
    processed = project_root / "data" / "processed"
    outputs = project_root / "outputs"

    # ===================================================================
    # MODEL A: TEXT-ONLY
    # ===================================================================
    logger.info("=" * 60)
    logger.info("MODEL A: TEXT-ONLY")
    logger.info("=" * 60)

    tm_path = processed / "text_market_dataset.parquet"
    if not tm_path.exists():
        logger.error("text_market_dataset.parquet not found!")
        return

    df_text = pl.read_parquet(tm_path)
    text_cols, _, _ = identify_feature_groups(df_text)
    text_results = run_single_model(df_text, text_cols, "Text-Only",
                                     outputs / "text_only")

    # ===================================================================
    # MODEL B: MULTIMODAL (Text + Audio)
    # ===================================================================
    mm_path = processed / "multimodal_dataset.parquet"
    if mm_path.exists():
        logger.info("=" * 60)
        logger.info("MODEL B: MULTIMODAL (Text + Audio)")
        logger.info("=" * 60)

        df_mm = pl.read_parquet(mm_path)
        text_cols_mm, audio_cols_mm, _ = identify_feature_groups(df_mm)
        all_cols = text_cols_mm + audio_cols_mm

        # Audio-only
        if audio_cols_mm:
            logger.info("=" * 60)
            logger.info("MODEL C: AUDIO-ONLY")
            logger.info("=" * 60)
            run_single_model(df_mm, audio_cols_mm, "Audio-Only",
                             outputs / "audio_only")

        # Hybrid
        logger.info("=" * 60)
        logger.info("MODEL D: HYBRID (Text + Audio)")
        logger.info("=" * 60)
        hybrid_results = run_single_model(df_mm, all_cols, "Hybrid",
                                           outputs / "hybrid")
    else:
        logger.warning("multimodal_dataset.parquet not found. Run audio pipeline first.")
        logger.info("Only Text-Only results generated.")

    # ===================================================================
    # COMPARISON SUMMARY
    # ===================================================================
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    for variant in ["text_only", "audio_only", "hybrid"]:
        res_path = outputs / variant / "results.json"
        if res_path.exists():
            with open(res_path) as f:
                r = json.load(f)
            tq = r["confidence_tiers"].get("top_quintile", {})
            logger.info(
                "  %-15s IC=%.4f  Acc=%.1f%%  TopQ_Hit=%.1f%%  TopQ_Sharpe=%.2f  Features=%d",
                r["model_name"], r["ic"], r["accuracy"]*100,
                tq.get("hit_rate", 0)*100, tq.get("sharpe", 0), r["n_features"],
            )


if __name__ == "__main__":
    main()
