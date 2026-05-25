"""
Sentiment Delta Features — Quarter-over-Quarter NLP Changes

For each ticker, computes how NLP features CHANGED compared to the
previous quarter's earnings call. This captures momentum in executive tone:

    - sentiment_score_mean_delta      (raw change)
    - sentiment_score_mean_pct_change (percentage change)
    - sentiment_score_mean_z_score    (change normalized by ticker's historical std)

Repeats for: uncertainty, specificity, hedging, linguistic_complexity.

Output: data/processed/sentiment_delta_features.parquet
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Delta Features
# ---------------------------------------------------------------------------

# Key NLP features to compute deltas for
DELTA_FEATURES = [
    "sentiment_score_mean",
    "uncertainty_score_mean",
    "specificity_score_mean",
    "hedging_frequency_mean",
    "linguistic_complexity_mean",
]


def compute_sentiment_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute quarter-over-quarter changes in NLP features per ticker.

    Args:
        df: DataFrame with call_id, ticker, call_date, and NLP feature columns.

    Returns:
        DataFrame with call_id and delta feature columns.
    """
    # Ensure we have the needed columns
    available_features = [f for f in DELTA_FEATURES if f in df.columns]
    if not available_features:
        logger.warning("No delta-eligible features found in dataset.")
        return pl.DataFrame({"call_id": df["call_id"]})

    logger.info("Computing deltas for %d features: %s", len(available_features), available_features)

    # Sort by ticker then chronologically
    df = df.sort(["ticker", "call_date"])

    # For each feature, compute delta, pct_change, z_score within each ticker group
    delta_exprs = []
    for feat in available_features:
        col = pl.col(feat)
        prev = col.shift(1).over("ticker")

        # Raw delta: current - previous
        delta_exprs.append(
            (col - prev).alias(f"{feat}_delta")
        )

        # Percentage change: (current - previous) / |previous|
        # Clip to avoid division by zero
        delta_exprs.append(
            pl.when(prev.abs() > 1e-8)
            .then((col - prev) / prev.abs())
            .otherwise(0.0)
            .alias(f"{feat}_pct_change")
        )

    # Compute deltas
    result = df.select(["call_id", "ticker", "call_date"] + available_features).with_columns(delta_exprs)

    # Now compute z-scores: delta / rolling std of the feature within ticker
    # We use the historical std of the feature (all previous values for that ticker)
    z_score_exprs = []
    for feat in available_features:
        delta_col = f"{feat}_delta"
        # Historical std of the raw feature within each ticker (expanding window)
        hist_std = pl.col(feat).rolling_std(window_size=20, min_periods=2).over("ticker")

        z_score_exprs.append(
            pl.when(hist_std > 1e-8)
            .then(pl.col(delta_col) / hist_std)
            .otherwise(0.0)
            .alias(f"{feat}_z_score")
        )

    result = result.with_columns(z_score_exprs)

    # Select only call_id and delta columns
    delta_cols = [c for c in result.columns if c.endswith(("_delta", "_pct_change", "_z_score"))]
    output = result.select(["call_id"] + delta_cols)

    # Fill nulls (first quarter of each ticker will have null deltas)
    output = output.fill_null(0.0)

    n_features = len(delta_cols)
    logger.info(
        "Computed %d delta features for %d calls (%d null-filled first-quarters)",
        n_features, len(output),
        output.filter(pl.col(delta_cols[0]) == 0.0).height if delta_cols else 0,
    )

    return output


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
    processed = project_root / "data" / "processed"

    # Load the text_market dataset (has all features + ticker + call_date)
    tm_path = processed / "text_market_dataset.parquet"
    if not tm_path.exists():
        logger.error("text_market_dataset.parquet not found. Run multimodal_join.py first.")
        return

    df = pl.read_parquet(tm_path)
    logger.info("Loaded dataset: %d calls, %d columns", len(df), len(df.columns))

    # Compute deltas
    delta_df = compute_sentiment_deltas(df)

    # Save
    output_path = processed / "sentiment_delta_features.parquet"
    delta_df.write_parquet(output_path)
    logger.info("Saved sentiment delta features to: %s", output_path)

    # Preview
    logger.info("Sample delta features:")
    logger.info("\n%s", delta_df.head(5))


if __name__ == "__main__":
    main()
