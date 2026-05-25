"""
Side-by-Side Comparison Plots — Text-Only vs Audio vs Hybrid

Generates comparison visualizations:
  1. Accuracy / IC / Sharpe bar charts
  2. Equity curves overlaid
  3. Feature importance comparison
  4. Confidence tier comparison
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

COLORS = {"Text-Only": "#4FC3F7", "Audio-Only": "#FF8A65", "Hybrid": "#81C784"}


def load_results(outputs_dir: Path) -> dict:
    """Load all available model results."""
    results = {}
    for variant, label in [("text_only", "Text-Only"), ("audio_only", "Audio-Only"), ("hybrid", "Hybrid")]:
        path = outputs_dir / variant / "results.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            r["label"] = label
            results[label] = r
    return results


def plot_metric_comparison(results: dict, plot_dir: Path):
    """Bar chart comparing IC, Accuracy, and Top-Quintile metrics."""
    labels = list(results.keys())
    colors = [COLORS.get(l, "#999") for l in labels]

    metrics = {
        "IC (Volatility Prediction)": [r["ic"] for r in results.values()],
        "Direction Accuracy": [r["accuracy"] for r in results.values()],
        "Top 20% Hit Rate": [r["confidence_tiers"].get("top_quintile", {}).get("hit_rate", 0) for r in results.values()],
        "Top 20% Sharpe": [r["confidence_tiers"].get("top_quintile", {}).get("sharpe", 0) for r in results.values()],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: Text-Only vs Audio vs Hybrid", fontsize=16, fontweight="bold", y=0.98)

    for ax, (metric_name, values) in zip(axes.flat, metrics.items()):
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
        ax.set_title(metric_name, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric_name.split("(")[0].strip())
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            fmt = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    fmt, ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Reference lines
        if "Accuracy" in metric_name or "Hit Rate" in metric_name:
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% (random)")
            ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(plot_dir / "comparison_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: comparison_metrics.png")


def plot_equity_curves(results: dict, outputs_dir: Path, plot_dir: Path):
    """Overlay equity curves from each model."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("Equity Curves — Strategy Comparison", fontsize=15, fontweight="bold")

    for label, r in results.items():
        curve_path = outputs_dir / label.lower().replace("-", "_").replace(" ", "_") / "equity_curve.parquet"
        if curve_path.exists():
            curve = pl.read_parquet(curve_path)
            ax.plot(curve["trade_idx"].to_list(), 
                    (np.array(curve["strategy_cum_return"].to_list()) * 100),
                    label=f"{label} Strategy", color=COLORS.get(label, "#999"), linewidth=2)

    # Add benchmark from the first available
    for label in results:
        curve_path = outputs_dir / label.lower().replace("-", "_").replace(" ", "_") / "equity_curve.parquet"
        if curve_path.exists():
            curve = pl.read_parquet(curve_path)
            ax.plot(curve["trade_idx"].to_list(),
                    (np.array(curve["benchmark_cum_return"].to_list()) * 100),
                    label="Benchmark (Buy & Hold)", color="gray", linewidth=1.5, linestyle="--", alpha=0.6)
            break

    ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.tight_layout()
    fig.savefig(plot_dir / "comparison_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: comparison_equity_curves.png")


def plot_feature_importance_comparison(results: dict, plot_dir: Path):
    """Side-by-side top-10 features for each model."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Top 10 Feature Importance by Model", fontsize=16, fontweight="bold", y=1.02)

    for ax, (label, r) in zip(axes, results.items()):
        features = r.get("top_features", [])[:10]
        if not features:
            ax.text(0.5, 0.5, "No features", ha="center", va="center", transform=ax.transAxes)
            continue

        names = [f["feature"][:30] for f in reversed(features)]
        values = [f["importance"] for f in reversed(features)]

        color = COLORS.get(label, "#999")
        ax.barh(names, values, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{label}\n({r['n_features']} features)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_dir / "comparison_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: comparison_feature_importance.png")


def plot_confidence_tiers(results: dict, plot_dir: Path):
    """Grouped bar chart of confidence tier performance."""
    tiers = ["medium", "high", "ultra", "top_quintile"]
    tier_labels = ["Medium\n(>55%)", "High\n(>60%)", "Ultra\n(>65%)", "Top 20%\nConfidence"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Confidence Tier Analysis — Model Comparison", fontsize=16, fontweight="bold")

    x = np.arange(len(tiers))
    width = 0.25
    models = list(results.keys())

    for ax_idx, (ax, metric, fmt) in enumerate(zip(axes, ["hit_rate", "sharpe", "total_return"],
                                                     [lambda v: f"{v*100:.1f}%", lambda v: f"{v:.2f}", lambda v: f"{v*100:.1f}%"])):
        for i, model in enumerate(models):
            r = results[model]
            values = [r["confidence_tiers"].get(t, {}).get(metric, 0) for t in tiers]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=COLORS.get(model, "#999"),
                          edgecolor="white", linewidth=0.5)

        metric_title = {"hit_rate": "Hit Rate", "sharpe": "Sharpe Ratio", "total_return": "Total Return"}
        ax.set_title(metric_title.get(metric, metric), fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        if metric == "hit_rate":
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
            ax.set_ylabel("Hit Rate")
        elif metric == "sharpe":
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.set_ylabel("Sharpe Ratio")
        else:
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.set_ylabel("Total Return")

    plt.tight_layout()
    fig.savefig(plot_dir / "comparison_confidence_tiers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: comparison_confidence_tiers.png")


def main():
    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "outputs"
    plot_dir = outputs_dir / "plots" / "comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(outputs_dir)
    if not results:
        logger.error("No model results found. Run run_comparison.py first.")
        return

    logger.info("Found %d model variants: %s", len(results), list(results.keys()))

    plot_metric_comparison(results, plot_dir)
    plot_equity_curves(results, outputs_dir, plot_dir)
    plot_feature_importance_comparison(results, plot_dir)
    plot_confidence_tiers(results, plot_dir)

    logger.info("All comparison plots saved to: %s", plot_dir)


if __name__ == "__main__":
    main()
