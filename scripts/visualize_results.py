"""
Visualization Suite — Publication-Quality Charts

Generates:
    1. Pressure Spike Timeline — shows executive pressure spikes per call
    2. Sector Performance Heatmap — model accuracy by sector
    3. Equity Curve — walk-forward backtest cumulative returns
    4. Feature Importance Waterfall — top contributing features
    5. Divergence Spike Plot — text-audio divergence per call
    6. Cross-Validation Stability — metric distributions across folds
    7. Dataset Coverage — calls by sector + year

All outputs saved to outputs/plots/
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
DARK_BG = "#0e1117"
CARD_BG = "#1e2227"
ACCENT_GREEN = "#2ea043"
ACCENT_RED = "#f85149"
ACCENT_BLUE = "#58a6ff"
ACCENT_YELLOW = "#d29922"
ACCENT_PURPLE = "#bc8cff"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"

SECTOR_COLORS = {
    "Technology": "#58a6ff",
    "Healthcare": "#2ea043",
    "Financials": "#d29922",
    "Energy": "#f85149",
    "Consumer": "#bc8cff",
    "Other": "#8b949e",
}

SECTOR_TICKERS = {
    "Technology": {"AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "CSCO", "ACN",
                   "ADBE", "IBM", "INTC", "INTU", "TXN", "QCOM", "AMAT", "NOW", "PANW",
                   "LRCX", "ADI", "SNPS", "KLAC", "CDNS", "MCHP", "FTNT",
                   "GOOGL", "GOOG", "META", "AMZN", "TSLA", "NFLX"},
    "Healthcare": {"JNJ", "UNH", "PFE", "ABT", "TMO", "MRK", "LLY", "ABBV", "DHR",
                   "BMY", "AMGN", "MDT", "ISRG", "GILD", "CVS", "CI", "SYK", "BSX",
                   "VRTX", "REGN", "ZTS", "BDX", "EW", "HCA", "IDXX"},
    "Financials": {"JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP",
                   "USB", "PNC", "TFC", "CB", "MMC", "AON", "ICE", "CME", "MCO",
                   "SPGI", "COF", "MET", "AIG", "PRU", "ALL", "TRV"},
    "Energy": {"XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD",
               "OXY", "WMB", "HES", "DVN", "KMI", "HAL", "FANG", "BKR",
               "TRGP", "OKE", "CTRA"},
    "Consumer": {"PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX",
                 "TGT", "LOW", "CL", "EL", "GIS", "KHC", "MDLZ", "SJM", "HSY",
                 "DG", "DLTR", "TJX", "ROST", "YUM", "DPZ", "CMG"},
}


def get_sector(ticker: str) -> str:
    for sec, ticks in SECTOR_TICKERS.items():
        if ticker in ticks:
            return sec
    return "Other"


def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
    })


# =====================================================================
# Plot 1: Pressure Spike Timeline
# =====================================================================

def plot_pressure_spikes(project_root: Path, plot_dir: Path):
    """Plot executive pressure spikes across calls — the signature visualization."""
    processed = project_root / "data" / "processed"

    # Try multimodal first (has pressure features), else use text_market
    for fname in ["multimodal_dataset.parquet", "text_market_dataset.parquet"]:
        p = processed / fname
        if p.exists():
            df = pl.read_parquet(p)
            break
    else:
        logger.warning("No dataset found for pressure spike plot")
        return

    # Compute a pressure proxy from available features
    pressure_cols = [c for c in df.columns if "pressure" in c.lower()]
    sentiment_cols = [c for c in df.columns if "sentiment" in c.lower() and "mean" in c.lower()]
    uncertainty_cols = [c for c in df.columns if "uncertainty" in c.lower() and "mean" in c.lower()]

    if pressure_cols:
        df = df.with_columns(pl.col(pressure_cols[0]).alias("_pressure"))
    elif sentiment_cols and uncertainty_cols:
        # Proxy: high uncertainty + negative sentiment = pressure
        df = df.with_columns(
            (pl.col(uncertainty_cols[0]) - pl.col(sentiment_cols[0])).alias("_pressure")
        )
    elif sentiment_cols:
        df = df.with_columns((-pl.col(sentiment_cols[0])).alias("_pressure"))
    else:
        logger.warning("No pressure-like features found, skipping pressure spike plot")
        return

    df = df.sort("call_date")

    tickers = df["ticker"].to_list() if "ticker" in df.columns else ["UNK"] * len(df)
    dates_raw = df["call_date"].to_list()
    pressure = df["_pressure"].to_numpy()
    pressure = np.nan_to_num(pressure, nan=0.0)

    # Normalize pressure to [0, 1]
    pmin, pmax = pressure.min(), pressure.max()
    if pmax - pmin > 0:
        pressure_norm = (pressure - pmin) / (pmax - pmin)
    else:
        pressure_norm = np.zeros_like(pressure)

    # Identify spike threshold (top 20%)
    threshold = np.percentile(pressure_norm, 80)

    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(pressure_norm))
    colors = [ACCENT_RED if p > threshold else ACCENT_BLUE for p in pressure_norm]

    bars = ax.bar(x, pressure_norm, color=colors, alpha=0.85, width=0.8, edgecolor="none")

    # Mark spikes with labels
    spike_indices = np.where(pressure_norm > threshold)[0]
    for idx in spike_indices:
        ax.annotate(
            tickers[idx] if idx < len(tickers) else "",
            xy=(idx, pressure_norm[idx]),
            xytext=(0, 8), textcoords="offset points",
            fontsize=7, color=ACCENT_RED, fontweight="bold",
            ha="center", rotation=45,
        )

    ax.axhline(y=threshold, color=ACCENT_YELLOW, linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Spike Threshold (p80 = {threshold:.2f})")

    # X-axis labels — show every Nth ticker
    step = max(1, len(x) // 30)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([tickers[i][:6] if i < len(tickers) else "" for i in x[::step]],
                       rotation=45, fontsize=7)

    ax.set_xlabel("Earnings Calls (Chronological)")
    ax.set_ylabel("Executive Pressure Score (normalized)")
    ax.set_title("Executive Pressure Spike Timeline", fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(plot_dir / "pressure_spikes.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: pressure_spikes.png")


# =====================================================================
# Plot 2: Backtest Equity Curve
# =====================================================================

def plot_equity_curve(project_root: Path, plot_dir: Path):
    """Plot cumulative returns: strategy vs benchmark."""
    bt_path = project_root / "outputs" / "evaluation" / "backtest_results.json"
    if not bt_path.exists():
        logger.warning("No backtest results found, skipping equity curve")
        return

    with open(bt_path) as f:
        bt = json.load(f)

    trades = bt.get("trades", [])
    if not trades:
        return

    active = [t for t in trades if t["signal"] != 0]
    if not active:
        logger.warning("No active trades, skipping equity curve")
        return

    strategy_rets = np.array([t["pnl"] for t in active])
    benchmark_rets = np.array([t["actual_ret"] for t in active])

    cum_strat = np.cumprod(1 + strategy_rets) - 1
    cum_bench = np.cumprod(1 + benchmark_rets) - 1
    trade_indices = np.arange(len(active))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # Top: equity curve
    ax1.plot(trade_indices, cum_strat * 100, color=ACCENT_GREEN, linewidth=2.5,
             label=f"Multimodal Strategy ({cum_strat[-1]*100:+.1f}%)")
    ax1.plot(trade_indices, cum_bench * 100, color="#8b949e", linewidth=1.5,
             linestyle="--", label=f"Benchmark ({cum_bench[-1]*100:+.1f}%)")
    ax1.fill_between(trade_indices, cum_strat * 100, cum_bench * 100,
                     where=cum_strat > cum_bench, alpha=0.15, color=ACCENT_GREEN)
    ax1.fill_between(trade_indices, cum_strat * 100, cum_bench * 100,
                     where=cum_strat < cum_bench, alpha=0.15, color=ACCENT_RED)

    ax1.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Walk-Forward Backtest: Strategy vs Benchmark", fontsize=16, fontweight="bold", pad=15)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(alpha=0.2)

    # Add key metrics annotation
    sharpe = bt.get("annualized_sharpe", 0)
    hit = bt.get("hit_rate", 0)
    pval = bt.get("p_value", 1)
    metrics_text = f"Sharpe: {sharpe:.2f}  |  Hit Rate: {hit:.0%}  |  p-value: {pval:.3f}"
    ax1.text(0.5, 0.02, metrics_text, transform=ax1.transAxes,
             fontsize=10, ha="center", color=TEXT_COLOR, alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_BG, edgecolor=GRID_COLOR))

    # Bottom: per-trade PnL
    colors = [ACCENT_GREEN if r > 0 else ACCENT_RED for r in strategy_rets]
    ax2.bar(trade_indices, strategy_rets * 100, color=colors, alpha=0.8)
    ax2.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("P&L (%)")
    ax2.set_title("Per-Trade Returns", fontsize=12)
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(plot_dir / "equity_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: equity_curve.png")


# =====================================================================
# Plot 3: Feature Importance Waterfall
# =====================================================================

def plot_feature_importance(project_root: Path, plot_dir: Path):
    """Horizontal bar chart of top features."""
    fi_path = project_root / "outputs" / "feature_importance.json"
    if not fi_path.exists():
        logger.warning("No feature importance data found")
        return

    with open(fi_path) as f:
        fi = json.load(f)

    # Find the best model's features (text+structural usually)
    best_key = None
    for key in fi:
        if "Text+Structural" in key and "regression" in key:
            best_key = key
            break
    if not best_key:
        best_key = list(fi.keys())[0] if fi else None
    if not best_key:
        return

    features = fi[best_key][:15]
    if not features:
        return

    names = [f["feature"][:30] for f in features][::-1]
    scores = [f["importance"] for f in features][::-1]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(names, scores, color=ACCENT_BLUE, alpha=0.85, edgecolor="none", height=0.6)

    # Color top 3 differently
    for i, bar in enumerate(bars):
        if i >= len(bars) - 3:
            bar.set_color(ACCENT_GREEN)

    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Predictive Features", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: feature_importance.png")


# =====================================================================
# Plot 4: Sector Coverage & Performance
# =====================================================================

def plot_sector_coverage(project_root: Path, plot_dir: Path):
    """Bar chart showing dataset coverage by sector + year."""
    processed = project_root / "data" / "processed"

    for fname in ["text_market_dataset.parquet", "multimodal_dataset.parquet"]:
        p = processed / fname
        if p.exists():
            df = pl.read_parquet(p)
            break
    else:
        return

    if "ticker" not in df.columns:
        return

    tickers = df["ticker"].to_list()
    sectors = [get_sector(t) for t in tickers]
    df = df.with_columns(pl.Series("sector", sectors))

    # Sector counts
    sector_counts = df.group_by("sector").len().sort("len", descending=True)
    sec_names = sector_counts["sector"].to_list()
    sec_vals = sector_counts["len"].to_list()
    sec_colors = [SECTOR_COLORS.get(s, SECTOR_COLORS["Other"]) for s in sec_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: sector distribution
    ax1.barh(sec_names[::-1], sec_vals[::-1], color=sec_colors[::-1], alpha=0.85)
    ax1.set_xlabel("Number of Calls")
    ax1.set_title("Calls by Sector", fontsize=14, fontweight="bold")
    ax1.grid(axis="x", alpha=0.2)
    for i, (name, val) in enumerate(zip(sec_names[::-1], sec_vals[::-1])):
        ax1.text(val + 0.5, i, str(val), va="center", fontsize=10, color=TEXT_COLOR)

    # Right: calls by year
    if "call_date" in df.columns:
        df2 = df.with_columns(
            pl.col("call_date").cast(pl.Utf8).str.slice(0, 4).alias("year")
        )
        year_counts = df2.group_by("year").len().sort("year")
        years = year_counts["year"].to_list()
        counts = year_counts["len"].to_list()

        ax2.bar(years, counts, color=ACCENT_BLUE, alpha=0.85)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Number of Calls")
        ax2.set_title("Calls by Year", fontsize=14, fontweight="bold")
        ax2.grid(axis="y", alpha=0.2)
        for i, (y, c) in enumerate(zip(years, counts)):
            ax2.text(i, c + 0.5, str(c), ha="center", fontsize=10, color=TEXT_COLOR)
    else:
        ax2.text(0.5, 0.5, "No date information available", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=14)

    plt.suptitle("Dataset Coverage", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(plot_dir / "sector_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: sector_coverage.png")


# =====================================================================
# Plot 5: Sentiment vs Return Scatter
# =====================================================================

def plot_sentiment_vs_return(project_root: Path, plot_dir: Path):
    """Scatter plot of sentiment score vs actual 1-day return with trend line."""
    processed = project_root / "data" / "processed"

    for fname in ["text_market_dataset.parquet", "multimodal_dataset.parquet"]:
        p = processed / fname
        if p.exists():
            df = pl.read_parquet(p)
            break
    else:
        return

    sent_cols = [c for c in df.columns if "sentiment" in c.lower() and "mean" in c.lower()]
    if not sent_cols or "return_1d" not in df.columns:
        return

    sentiment = df[sent_cols[0]].to_numpy()
    returns = df["return_1d"].to_numpy()

    # Remove NaN
    mask = np.isfinite(sentiment) & np.isfinite(returns)
    sentiment = sentiment[mask]
    returns = returns[mask]

    if len(sentiment) < 5:
        return

    tickers = [t for t, m in zip(df["ticker"].to_list(), mask) if m] if "ticker" in df.columns else None
    sectors = [get_sector(t) for t in tickers] if tickers else ["Other"] * len(sentiment)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by sector
    for sec in set(sectors):
        idx = [i for i, s in enumerate(sectors) if s == sec]
        ax.scatter(
            sentiment[idx], returns[idx] * 100,
            c=SECTOR_COLORS.get(sec, SECTOR_COLORS["Other"]),
            label=sec, alpha=0.6, s=40, edgecolors="none",
        )

    # Trend line
    z = np.polyfit(sentiment, returns * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(sentiment.min(), sentiment.max(), 100)
    ax.plot(x_line, p(x_line), color=ACCENT_YELLOW, linewidth=2, linestyle="--",
            label=f"Trend (slope={z[0]:.2f})")

    ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Average Sentiment Score")
    ax.set_ylabel("1-Day Return (%)")
    ax.set_title("Sentiment vs Post-Earnings Return", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(plot_dir / "sentiment_vs_return.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: sentiment_vs_return.png")


# =====================================================================
# Plot 6: Model Comparison Bar Chart
# =====================================================================

def plot_model_comparison(project_root: Path, plot_dir: Path):
    """Bar chart comparing baseline models with error bars (from CV CIs)."""
    bc_path = project_root / "outputs" / "baseline_comparison.json"
    if not bc_path.exists():
        return

    with open(bc_path) as f:
        results = json.load(f)

    # Extract classification results
    cls_results = [r for r in results if r["task"] == "direction_classification"]
    if not cls_results:
        return

    models = []
    accs = []
    ci_lows = []
    ci_highs = []

    for r in cls_results:
        models.append(r["model"])
        if isinstance(r["accuracy"], dict):
            accs.append(r["accuracy"]["mean"])
            ci_lows.append(r["accuracy"]["ci_low"])
            ci_highs.append(r["accuracy"]["ci_high"])
        else:
            accs.append(r["accuracy"])
            ci_lows.append(r["accuracy"])
            ci_highs.append(r["accuracy"])

    errors_low = [a - cl for a, cl in zip(accs, ci_lows)]
    errors_high = [ch - a for a, ch in zip(accs, ci_highs)]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    colors = [ACCENT_BLUE, "#8b949e", ACCENT_GREEN, ACCENT_RED][:len(models)]

    bars = ax.bar(x, [a * 100 for a in accs], color=colors, alpha=0.85, width=0.5,
                  yerr=[[e * 100 for e in errors_low], [e * 100 for e in errors_high]],
                  capsize=8, error_kw={"color": TEXT_COLOR, "linewidth": 1.5})

    ax.axhline(y=50, color=ACCENT_YELLOW, linestyle="--", alpha=0.7, label="Random Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_title("Model Comparison: Direction Prediction Accuracy (95% CI)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1%}", ha="center", fontsize=11, color=TEXT_COLOR, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: model_comparison.png")


# =====================================================================
# Plot 7: Uncertainty Spike Analysis
# =====================================================================

def plot_uncertainty_spikes(project_root: Path, plot_dir: Path):
    """Show uncertainty spikes with return overlay — key insight visualization."""
    processed = project_root / "data" / "processed"

    for fname in ["text_market_dataset.parquet", "multimodal_dataset.parquet"]:
        p = processed / fname
        if p.exists():
            df = pl.read_parquet(p)
            break
    else:
        return

    unc_cols = [c for c in df.columns if "uncertainty" in c.lower() and "mean" in c.lower()]
    if not unc_cols or "return_1d" not in df.columns:
        return

    df = df.sort("call_date")
    uncertainty = df[unc_cols[0]].to_numpy()
    returns = df["return_1d"].to_numpy()
    tickers = df["ticker"].to_list() if "ticker" in df.columns else [""] * len(df)
    uncertainty = np.nan_to_num(uncertainty, nan=0.0)
    returns = np.nan_to_num(returns, nan=0.0)

    # Normalize uncertainty
    umax = np.max(np.abs(uncertainty)) if np.max(np.abs(uncertainty)) > 0 else 1
    unc_norm = uncertainty / umax

    # Threshold for spikes
    threshold = np.percentile(unc_norm, 75)

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    x = np.arange(len(unc_norm))

    # Uncertainty bars
    unc_colors = [ACCENT_RED if u > threshold else ACCENT_BLUE for u in unc_norm]
    ax1.bar(x, unc_norm, color=unc_colors, alpha=0.6, width=0.8, label="Uncertainty Level")

    # Return line
    ax2.plot(x, returns * 100, color=ACCENT_GREEN, linewidth=1.5, alpha=0.8, label="1-Day Return (%)")
    ax2.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)

    # Mark high-uncertainty calls that predicted negative returns
    for i in range(len(unc_norm)):
        if unc_norm[i] > threshold and returns[i] < -0.01:
            ax1.annotate(
                tickers[i][:5] if i < len(tickers) else "",
                xy=(i, unc_norm[i]),
                xytext=(0, 10), textcoords="offset points",
                fontsize=7, color=ACCENT_RED, fontweight="bold",
                ha="center", rotation=45,
            )

    ax1.axhline(y=threshold, color=ACCENT_YELLOW, linestyle="--", alpha=0.7,
                label=f"Spike Threshold (p75)")

    ax1.set_xlabel("Earnings Calls (Chronological)")
    ax1.set_ylabel("Uncertainty Score (normalized)", color=ACCENT_BLUE)
    ax2.set_ylabel("1-Day Return (%)", color=ACCENT_GREEN)
    ax1.set_title("Uncertainty Spikes vs Post-Earnings Returns", fontsize=16, fontweight="bold", pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    step = max(1, len(x) // 30)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels([tickers[i][:5] if i < len(tickers) else "" for i in x[::step]],
                        rotation=45, fontsize=7)

    plt.tight_layout()
    plt.savefig(plot_dir / "uncertainty_spikes.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: uncertainty_spikes.png")


# =====================================================================
# Main
# =====================================================================

def main():
    project_root = Path(__file__).resolve().parent.parent
    plot_dir = project_root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    setup_dark_style()

    logger.info("=" * 60)
    logger.info("Generating Visualization Suite")
    logger.info("=" * 60)

    plot_pressure_spikes(project_root, plot_dir)
    plot_equity_curve(project_root, plot_dir)
    plot_feature_importance(project_root, plot_dir)
    plot_sector_coverage(project_root, plot_dir)
    plot_sentiment_vs_return(project_root, plot_dir)
    plot_model_comparison(project_root, plot_dir)
    plot_uncertainty_spikes(project_root, plot_dir)

    logger.info("=" * 60)
    logger.info("All plots saved to: %s", plot_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
