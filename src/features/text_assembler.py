"""
Unified text feature assembler for the Multimodal Earnings Call Intelligence System.

Merges sentiment, uncertainty, and specificity features into a single 
text_features.parquet matching Contract B.

Contract B Schema:
    - segment_id (PK)
    - sentiment_score
    - uncertainty_score
    - forward_looking_score
    - hedging_frequency
    - specificity_score
    - linguistic_complexity
"""

import logging
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


def assemble_text_features(
    sentiment_path: Path,
    uncertainty_path: Path,
    specificity_path: Path,
    output_path: Path,
):
    """Join all text feature Parquet files and save the result."""
    
    if not sentiment_path.exists():
        logger.error("Sentiment features missing: %s", sentiment_path)
        return
    if not uncertainty_path.exists():
        logger.error("Uncertainty features missing: %s", uncertainty_path)
        return
    if not specificity_path.exists():
        logger.error("Specificity features missing: %s", specificity_path)
        return

    logger.info("Loading feature tables...")
    df_sent = pl.read_parquet(sentiment_path)
    df_unc = pl.read_parquet(uncertainty_path)
    df_spec = pl.read_parquet(specificity_path)

    logger.info("Merging tables on segment_id...")
    # Join all on segment_id
    df_merged = df_sent.join(df_unc, on="segment_id", how="inner")
    df_merged = df_merged.join(df_spec, on="segment_id", how="inner")

    # Reorder columns to match Contract B precisely
    expected_cols = [
        "segment_id",
        "sentiment_score",
        "uncertainty_score",
        "forward_looking_score",
        "hedging_frequency",
        "specificity_score",
        "linguistic_complexity",
    ]
    
    # Check if any expected columns are missing
    missing = [c for c in expected_cols if c not in df_merged.columns]
    if missing:
        logger.error("Merged table is missing expected columns: %s", missing)
        return

    df_final = df_merged.select(expected_cols)

    # Handle NaNs with column-aware imputation
    n_nan = df_final.null_count().sum().to_series()[0]
    if n_nan > 0:
        logger.warning("Found %d null values. Imputing with column median.", n_nan)
        for col in df_final.columns:
            if col == "segment_id":
                continue
            if df_final[col].dtype in (pl.Float32, pl.Float64):
                median_val = df_final[col].median()
                fill_val = median_val if median_val is not None else 0.0
                df_final = df_final.with_columns(pl.col(col).fill_null(fill_val))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(output_path)
    logger.info("Saved unified text features (%d rows) to: %s", len(df_final), output_path)


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
    data_dir = project_root / "data" / "processed"
    
    assemble_text_features(
        sentiment_path=data_dir / "text_sentiment.parquet",
        uncertainty_path=data_dir / "text_uncertainty.parquet",
        specificity_path=data_dir / "text_specificity.parquet",
        output_path=data_dir / "text_features.parquet",
    )


if __name__ == "__main__":
    main()
