"""
Walk-Forward Backtesting with Real Model Predictions + Confidence Tiers

Loads a trained LightGBM model and runs proper walk-forward backtesting:
  - Expanding training window
  - Out-of-sample predictions only
  - Transaction costs
  - Statistical significance testing
  - NEW: Tiered confidence filtering (High/Medium/All)
  - NEW: Abnormal return (market-adjusted) support

NO signal peeking — all predictions come from the model.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    """Load a saved LightGBM model from disk."""
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        return model
    except Exception:
        pass
    # Fallback: try joblib
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        logger.error("Could not load model from %s: %s", model_path, e)
        return None


def compute_tier_metrics(trades: list[dict], tier_name: str) -> dict:
    """Compute trading metrics for a given set of trades."""
    if not trades:
        return {
            "tier": tier_name,
            "n_trades": 0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "avg_pnl": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
        }

    returns = np.array([t["pnl"] for t in trades])
    correct = sum(
        1 for t in trades
        if (t["signal"] > 0 and t["actual_ret"] > 0)
        or (t["signal"] < 0 and t["actual_ret"] < 0)
    )
    hit_rate = correct / len(trades)

    # Cumulative return
    cum = np.cumprod(1 + returns) - 1
    total_return = float(cum[-1]) if len(cum) > 0 else 0.0

    # Sharpe
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-6
    sharpe = (avg_ret / max(std_ret, 1e-8)) * np.sqrt(252)

    # Max drawdown
    wealth = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Statistical significance
    if len(returns) > 2:
        t_stat, p_value = stats.ttest_1samp(returns, 0)
    else:
        t_stat, p_value = 0.0, 1.0

    return {
        "tier": tier_name,
        "n_trades": len(trades),
        "n_buy": sum(1 for t in trades if t["signal"] > 0),
        "n_sell": sum(1 for t in trades if t["signal"] < 0),
        "hit_rate": float(hit_rate),
        "sharpe": float(sharpe),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "avg_pnl": float(avg_ret),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def run_backtest():
    """
    Walk-forward backtest using real trained model predictions.

    Strategy:
        - For each call in the test window, predict direction
        - Tiered confidence thresholds:
            HIGH:   BUY if P(up) > 0.60, SELL if P(up) < 0.40
            MEDIUM: BUY if P(up) > 0.55, SELL if P(up) < 0.45
        - Apply 10bps round-trip transaction cost
        - NEW: Uses abnormal_return_1d if available
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    processed = project_root / "data" / "processed"
    models_dir = project_root / "outputs" / "models"
    outputs = project_root / "outputs" / "evaluation"
    outputs.mkdir(parents=True, exist_ok=True)

    # 1. Load the text+market dataset
    tm_path = processed / "text_market_dataset.parquet"
    if not tm_path.exists():
        logger.error("text_market_dataset.parquet not found. Run the pipeline first.")
        return

    df = pl.read_parquet(tm_path)
    logger.info("Loaded dataset: %d calls", len(df))

    # Check if abnormal returns are available
    use_abnormal = "abnormal_return_1d" in df.columns
    return_col = "abnormal_return_1d" if use_abnormal else "return_1d"
    if use_abnormal:
        logger.info("Using ABNORMAL returns (market-adjusted) as target")
    else:
        logger.info("Using RAW returns as target (no SPY data available)")

    # 2. Identify feature columns
    meta_cols = {
        "call_id", "ticker", "call_date", "close_t0", "close_t1", "close_t5",
        "return_1d", "return_5d", "realized_vol_1d", "realized_vol_5d",
        "earnings_surprise", "n_segments", "return_1d_direction",
        "abnormal_return_1d", "spy_return",
    }
    feature_cols = [
        c for c in df.columns
        if c not in meta_cols and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    # Filter valid rows
    df = df.filter(
        pl.col("return_1d").is_not_null()
        & pl.col("realized_vol_5d").is_not_null()
        & pl.col("realized_vol_5d").is_finite()
    )

    # Sort chronologically
    df = df.sort("call_date")
    n = len(df)

    if n < 10:
        logger.error("Not enough data for walk-forward backtest (have %d, need >= 10)", n)
        return

    # 3. Walk-forward: train on first 50%, test each remaining call
    import lightgbm as lgb

    min_train = max(int(n * 0.5), 10)

    all_trades = []
    all_predictions = []

    logger.info("Running walk-forward backtest: %d total calls, min_train=%d, features=%d",
                n, min_train, len(feature_cols))

    # Prepare numpy arrays
    X_all = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Use abnormal return for direction if available
    y_dir_all = (df[return_col].to_numpy() > 0).astype(int)
    actual_returns = df["return_1d"].to_numpy().astype(np.float64)  # PnL always uses raw return
    abnormal_returns = df[return_col].to_numpy().astype(np.float64) if use_abnormal else actual_returns
    tickers = df["ticker"].to_list() if "ticker" in df.columns else ["UNK"] * n
    dates = df["call_date"].to_list()
    call_ids = df["call_id"].to_list()

    # Walk-forward
    for t in range(min_train, n):
        X_train = X_all[:t]
        y_train = y_dir_all[:t]

        X_test = X_all[t:t+1]

        # Train a fresh LightGBM classifier
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=7,
            min_child_samples=max(2, len(X_train) // 10),
            random_state=42,
            verbosity=-1,
        )
        clf.fit(X_train, y_train)

        # Predict
        pred_proba = clf.predict_proba(X_test)[0, 1]  # P(up)
        pred_class = int(pred_proba > 0.5)

        # Generate signals at MULTIPLE confidence tiers
        TRANSACTION_COST = 0.001  # 10bps round-trip

        # Medium confidence (default)
        if pred_proba > 0.55:
            signal_medium = 1
        elif pred_proba < 0.45:
            signal_medium = -1
        else:
            signal_medium = 0

        # High confidence
        if pred_proba > 0.60:
            signal_high = 1
        elif pred_proba < 0.40:
            signal_high = -1
        else:
            signal_high = 0

        # Ultra-high confidence
        if pred_proba > 0.65:
            signal_ultra = 1
        elif pred_proba < 0.35:
            signal_ultra = -1
        else:
            signal_ultra = 0

        actual_ret = float(actual_returns[t])

        # PnL for each tier
        pnl_medium = signal_medium * actual_ret - (TRANSACTION_COST if signal_medium != 0 else 0)
        pnl_high = signal_high * actual_ret - (TRANSACTION_COST if signal_high != 0 else 0)
        pnl_ultra = signal_ultra * actual_ret - (TRANSACTION_COST if signal_ultra != 0 else 0)

        trade = {
            "date": str(dates[t]),
            "call_id": call_ids[t],
            "ticker": tickers[t] if t < len(tickers) else "UNK",
            "pred_proba": float(pred_proba),
            "signal_medium": signal_medium,
            "signal_high": signal_high,
            "signal_ultra": signal_ultra,
            "actual_direction": int(y_dir_all[t]),
            "actual_ret": actual_ret,
            "pnl_medium": float(pnl_medium),
            "pnl_high": float(pnl_high),
            "pnl_ultra": float(pnl_ultra),
            "train_size": t,
            # Keep backward compatibility
            "signal": signal_medium,
            "pnl": float(pnl_medium),
        }
        all_trades.append(trade)
        all_predictions.append({
            "call_id": call_ids[t],
            "ticker": tickers[t] if t < len(tickers) else "UNK",
            "date": str(dates[t]),
            "pred_proba": float(pred_proba),
            "pred_class": pred_class,
            "actual_direction": int(y_dir_all[t]),
            "signal": signal_medium,
        })

    # Save predictions for dashboard use
    pred_df = pl.DataFrame(all_predictions)
    pred_df.write_parquet(outputs / "model_predictions.parquet")
    logger.info("Saved %d model predictions to model_predictions.parquet", len(pred_df))

    # -----------------------------------------------------------------------
    # 4. Compute metrics PER CONFIDENCE TIER
    # -----------------------------------------------------------------------
    medium_trades = [t for t in all_trades if t["signal_medium"] != 0]
    high_trades = [t for t in all_trades if t["signal_high"] != 0]
    ultra_trades = [t for t in all_trades if t["signal_ultra"] != 0]

    # Adapt PnL field for each tier's metric computation
    for t in medium_trades:
        t["pnl"] = t["pnl_medium"]
        t["signal"] = t["signal_medium"]
    for t in high_trades:
        t["pnl"] = t["pnl_high"]
        t["signal"] = t["signal_high"]
    for t in ultra_trades:
        t["pnl"] = t["pnl_ultra"]
        t["signal"] = t["signal_ultra"]

    tier_results = {
        "medium_confidence": compute_tier_metrics(medium_trades, "Medium (>55%)"),
        "high_confidence": compute_tier_metrics(high_trades, "High (>60%)"),
        "ultra_confidence": compute_tier_metrics(ultra_trades, "Ultra (>65%)"),
    }

    # Also compute top-quintile metrics (rank by confidence, trade only top 20%)
    sorted_trades = sorted(all_trades, key=lambda t: abs(t["pred_proba"] - 0.5), reverse=True)
    top_quintile_n = max(1, len(sorted_trades) // 5)
    top_quintile_trades = sorted_trades[:top_quintile_n]
    # Assign signals based on the medium threshold for top-quintile
    for t in top_quintile_trades:
        t["signal"] = t["signal_medium"] if t["signal_medium"] != 0 else (1 if t["pred_proba"] > 0.5 else -1)
        t["pnl"] = t["signal"] * t["actual_ret"] - 0.001

    tier_results["top_quintile"] = compute_tier_metrics(top_quintile_trades, "Top 20% Confidence")

    # -----------------------------------------------------------------------
    # 5. Overall metrics (backward-compatible with medium tier)
    # -----------------------------------------------------------------------
    active_trades = medium_trades
    if not active_trades:
        logger.warning("No active trades generated. All predictions were HOLD.")
        return

    strategy_returns = np.array([t["pnl_medium"] for t in active_trades])
    benchmark_returns = np.array([t["actual_ret"] for t in active_trades])

    # Directional accuracy
    correct = sum(1 for t in active_trades
                  if (t["signal_medium"] > 0 and t["actual_ret"] > 0) or
                     (t["signal_medium"] < 0 and t["actual_ret"] < 0))
    hit_rate = correct / len(active_trades) if active_trades else 0

    # Overall directional accuracy (all predictions)
    all_correct = sum(1 for t in all_trades if t["pred_proba"] > 0.5 and t["actual_direction"] == 1
                      or t["pred_proba"] <= 0.5 and t["actual_direction"] == 0)
    overall_accuracy = all_correct / len(all_trades)

    # Cumulative returns
    cum_strategy = np.cumprod(1 + strategy_returns) - 1
    cum_benchmark = np.cumprod(1 + benchmark_returns) - 1

    # Sharpe ratio
    avg_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns, ddof=1) if len(strategy_returns) > 1 else 1e-6
    sharpe = (avg_ret / max(std_ret, 1e-8)) * np.sqrt(252)

    # Max drawdown
    wealth = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Statistical significance
    if len(strategy_returns) > 2:
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
    else:
        t_stat, p_value = 0.0, 1.0

    alpha = float(np.mean(strategy_returns) - np.mean(benchmark_returns))

    results = {
        "methodology": "walk_forward_backtest_v2",
        "target_variable": return_col,
        "n_total_calls": len(all_trades),
        "n_features": len(feature_cols),
        "n_active_trades": len(active_trades),
        "n_hold": len(all_trades) - len(active_trades),
        "n_buy": sum(1 for t in active_trades if t["signal_medium"] > 0),
        "n_sell": sum(1 for t in active_trades if t["signal_medium"] < 0),
        "transaction_cost_bps": 10,
        "total_strategy_return": float(cum_strategy[-1]) if len(cum_strategy) > 0 else 0,
        "total_benchmark_return": float(cum_benchmark[-1]) if len(cum_benchmark) > 0 else 0,
        "alpha": alpha,
        "hit_rate": hit_rate,
        "overall_accuracy": overall_accuracy,
        "annualized_sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "statistically_significant": bool(p_value < 0.05),
        "mean_strategy_return": float(np.mean(strategy_returns)),
        "std_strategy_return": float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0,
        "confidence_tiers": tier_results,
        "trades": [t for t in all_trades],  # Save all trades
    }

    # 6. Save
    with open(outputs / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=4, default=str)

    summary = {k: v for k, v in results.items() if k != "trades"}
    with open(outputs / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)

    logger.info("=" * 60)
    logger.info("WALK-FORWARD BACKTEST RESULTS (v2)")
    logger.info("=" * 60)
    logger.info("Target: %s | Features: %d", return_col, len(feature_cols))
    logger.info("Total calls evaluated: %d", len(all_trades))
    logger.info("Active trades:         %d (Buy=%d, Sell=%d, Hold=%d)",
                len(active_trades), results["n_buy"], results["n_sell"], results["n_hold"])
    logger.info("Hit rate (active):     %.1f%%", hit_rate * 100)
    logger.info("Overall accuracy:      %.1f%%", overall_accuracy * 100)
    logger.info("Strategy return:       %.2f%%", results["total_strategy_return"] * 100)
    logger.info("Benchmark return:      %.2f%%", results["total_benchmark_return"] * 100)
    logger.info("Alpha:                 %.2f%%", alpha * 100)
    logger.info("Sharpe ratio:          %.2f", sharpe)
    logger.info("Max drawdown:          %.2f%%", max_drawdown * 100)
    logger.info("t-statistic:           %.3f (p=%.4f)", t_stat, p_value)
    logger.info("Significant at 5%%:    %s", "YES" if p_value < 0.05 else "NO")
    logger.info("=" * 60)
    logger.info("CONFIDENCE TIER ANALYSIS")
    logger.info("=" * 60)
    for tier_name, tier in tier_results.items():
        logger.info(
            "  %-25s Trades=%d, Hit=%.1f%%, Sharpe=%.2f, Return=%.2f%%",
            tier["tier"], tier["n_trades"], tier["hit_rate"] * 100,
            tier["sharpe"], tier["total_return"] * 100,
        )
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_backtest()
