"""
Chronological Train / Validation / Test Split — Phase 3, Task 3.D.3

Splits datasets into train/val/test sets using a time-based approach
to prevent data leakage (no future data in training).

Split ratios: 60% train / 20% validation / 20% test
Split method: Chronological by call_date (earliest → latest)

Outputs:
    data/processed/splits/
        train.parquet
        val.parquet
        test.parquet
        split_metadata.json
"""

import json
import logging
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def chronological_split(
    df: pl.DataFrame,
    date_col: str = "call_date",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a DataFrame chronologically by date column.

    Args:
        df: Input DataFrame with a date column.
        date_col: Name of the date column to sort by.
        train_ratio: Fraction of data for training (default 0.6).
        val_ratio: Fraction of data for validation (default 0.2).
            Test gets the remainder (1 - train_ratio - val_ratio).

    Returns:
        (train_df, val_df, test_df)
    """
    # Sort by date
    df_sorted = df.sort(date_col)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.head(train_end)
    val_df = df_sorted.slice(train_end, val_end - train_end)
    test_df = df_sorted.slice(val_end, n - val_end)

    logger.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(train_df), 100 * len(train_df) / n,
        len(val_df), 100 * len(val_df) / n,
        len(test_df), 100 * len(test_df) / n,
    )

    # Log date ranges
    if date_col in train_df.columns and len(train_df) > 0:
        logger.info("  Train: %s → %s", train_df[date_col].min(), train_df[date_col].max())
    if date_col in val_df.columns and len(val_df) > 0:
        logger.info("  Val:   %s → %s", val_df[date_col].min(), val_df[date_col].max())
    if date_col in test_df.columns and len(test_df) > 0:
        logger.info("  Test:  %s → %s", test_df[date_col].min(), test_df[date_col].max())

    return train_df, val_df, test_df


def save_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: Path,
    dataset_name: str,
    date_col: str = "call_date",
) -> None:
    """Save split DataFrames and metadata."""
    split_dir = output_dir / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df.write_parquet(split_dir / "train.parquet")
    val_df.write_parquet(split_dir / "val.parquet")
    test_df.write_parquet(split_dir / "test.parquet")

    # Metadata
    metadata = {
        "dataset_name": dataset_name,
        "split_method": "chronological",
        "date_column": date_col,
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "train": {
            "n_samples": len(train_df),
            "n_features": len(train_df.columns),
            "date_range": [
                str(train_df[date_col].min()) if len(train_df) > 0 else None,
                str(train_df[date_col].max()) if len(train_df) > 0 else None,
            ],
        },
        "val": {
            "n_samples": len(val_df),
            "n_features": len(val_df.columns),
            "date_range": [
                str(val_df[date_col].min()) if len(val_df) > 0 else None,
                str(val_df[date_col].max()) if len(val_df) > 0 else None,
            ],
        },
        "test": {
            "n_samples": len(test_df),
            "n_features": len(test_df.columns),
            "date_range": [
                str(test_df[date_col].min()) if len(test_df) > 0 else None,
                str(test_df[date_col].max()) if len(test_df) > 0 else None,
            ],
        },
        "features": train_df.columns,
        "target_columns": [
            c for c in train_df.columns
            if c.startswith("return_") or c.startswith("realized_vol_")
        ],
    }

    with open(split_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved splits to: %s/", split_dir)


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
    splits_dir = processed / "splits"

    # -----------------------------------------------------------------------
    # 1. Split Text + Market dataset
    # -----------------------------------------------------------------------
    text_market_path = processed / "text_market_dataset.parquet"
    if text_market_path.exists():
        logger.info("=" * 60)
        logger.info("Splitting Text + Market dataset...")
        logger.info("=" * 60)

        df = pl.read_parquet(text_market_path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        train, val, test = chronological_split(df, date_col="call_date")
        save_splits(train, val, test, splits_dir, "text_market", date_col="call_date")
    else:
        logger.warning("text_market_dataset.parquet not found. Run multimodal_join.py first.")

    # -----------------------------------------------------------------------
    # 2. Split Multimodal dataset
    # -----------------------------------------------------------------------
    multimodal_path = processed / "multimodal_dataset.parquet"
    if multimodal_path.exists():
        logger.info("=" * 60)
        logger.info("Splitting Multimodal dataset...")
        logger.info("=" * 60)

        df = pl.read_parquet(multimodal_path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        if "call_date" in df.columns:
            train, val, test = chronological_split(df, date_col="call_date")
            save_splits(train, val, test, splits_dir, "multimodal", date_col="call_date")
        else:
            logger.warning("No call_date column found. Skipping multimodal split.")
    else:
        logger.warning("multimodal_dataset.parquet not found. Run multimodal_join.py first.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 60)
    for split_name in ["text_market", "multimodal"]:
        meta_path = splits_dir / split_name / "split_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            logger.info(
                "  %s: %d total → train=%d, val=%d, test=%d",
                split_name,
                meta["total_samples"],
                meta["train"]["n_samples"],
                meta["val"]["n_samples"],
                meta["test"]["n_samples"],
            )


if __name__ == "__main__":
    main()
