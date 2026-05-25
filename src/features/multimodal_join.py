"""
Multimodal Dataset Join — Phase 3, Task 3.D.1

Merges text features, audio features, structural features, and market data
into unified training-ready datasets.

Because the current sample data comes from two separate sources:
    - Text transcripts: HuggingFace (Apple earnings calls, call_id like AAPL_2020Q1)
    - Audio segments: Earnings-22 dataset (numeric call_ids like 4469590)

We produce three output datasets:
    1. text_market_dataset.parquet  — Text features + market targets (for text-only models)
    2. audio_dataset.parquet        — Audio features aggregated per call (for audio-only models)
    3. multimodal_dataset.parquet   — Full join (available when text+audio cover same calls)

When scaling to full S&P 500 data with aligned transcripts and audio,
all three datasets will share the same call_ids.
"""

import logging
import sys
from pathlib import Path

# Add project root to path so 'src' can be found if run as script
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_segment_features(
    df: pl.DataFrame,
    id_col: str = "segment_id",
    call_id_col: str = "call_id",
    exclude_cols: list[str] | None = None,
) -> pl.DataFrame:
    """
    Aggregate segment-level features to call-level by computing
    mean, std, min, max for each numeric column.
    """
    exclude = set(exclude_cols or [])
    exclude.update([id_col, call_id_col])

    # Identify numeric columns
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    # Build aggregation expressions
    agg_exprs = []
    for col in numeric_cols:
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max"),
        ])

    # Add segment count
    agg_exprs.append(pl.len().alias("n_segments"))

    result = df.group_by(call_id_col).agg(agg_exprs)
    logger.info(
        "Aggregated %d segments → %d calls (%d features)",
        len(df), len(result), len(result.columns) - 1,
    )
    return result


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_text_market_dataset(
    text_features_path: Path,
    structural_features_path: Path,
    market_data_path: Path,
    segments_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """
    Build text + market dataset at the call level.

    Steps:
        1. Add call_id to text features via segments table
        2. Aggregate text features per call (mean, std, min, max)
        3. Join with structural features
        4. Join with market data (targets)
    """
    # Load data
    tf = pl.read_parquet(text_features_path)
    sf = pl.read_parquet(structural_features_path)
    md = pl.read_parquet(market_data_path)
    seg = pl.read_parquet(segments_path)

    # Add call_id to text features
    seg_lookup = seg.select(["segment_id", "call_id"]).unique()
    tf = tf.join(seg_lookup, on="segment_id", how="left")

    # Aggregate text features to call level
    tf_agg = aggregate_segment_features(tf, call_id_col="call_id")

    # Join with structural features
    dataset = tf_agg.join(sf, on="call_id", how="left")

    # Join with market data (targets)
    dataset = dataset.join(md, on="call_id", how="inner")

    # Drop rows with no target
    dataset = dataset.filter(
        pl.col("return_1d").is_not_null() | pl.col("return_5d").is_not_null()
    )

    # -----------------------------------------------------------------------
    # NEW: Compute abnormal returns (market-adjusted)
    # -----------------------------------------------------------------------
    price_cache_dir = text_features_path.parent / "price_cache"
    spy_path = price_cache_dir / "SPY.parquet"
    if spy_path.exists():
        spy = pl.read_parquet(spy_path).sort("Date")
        # Compute SPY daily returns
        spy = spy.with_columns(
            (pl.col("Close") / pl.col("Close").shift(1) - 1.0).alias("spy_return")
        )

        # For each call, match the call_date to the nearest SPY trading day
        spy_lookup = spy.select(["Date", "spy_return"]).rename({"Date": "call_date"})

        # Left join on call_date
        dataset = dataset.join(spy_lookup, on="call_date", how="left")

        # Compute abnormal return
        dataset = dataset.with_columns([
            (pl.col("return_1d") - pl.col("spy_return").fill_null(0.0)).alias("abnormal_return_1d"),
        ])
        logger.info("Added abnormal_return_1d (market-adjusted) using SPY data")
    else:
        logger.warning("SPY price cache not found at %s. Skipping abnormal returns.", spy_path)
        dataset = dataset.with_columns(
            pl.col("return_1d").alias("abnormal_return_1d")
        )

    # -----------------------------------------------------------------------
    # NEW: Join sentiment delta features (QoQ changes)
    # -----------------------------------------------------------------------
    delta_path = text_features_path.parent / "sentiment_delta_features.parquet"
    if delta_path.exists():
        delta_df = pl.read_parquet(delta_path)
        dataset = dataset.join(delta_df, on="call_id", how="left")
        # Fill nulls for first-quarter tickers
        delta_cols = [c for c in delta_df.columns if c != "call_id"]
        for col in delta_cols:
            dataset = dataset.with_columns(pl.col(col).fill_null(0.0))
        logger.info("Added %d sentiment delta features", len(delta_cols))
    else:
        logger.info("No sentiment_delta_features.parquet found. Run sentiment_delta.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(output_path)
    logger.info(
        "Text+Market dataset: %d calls, %d features → %s",
        len(dataset), len(dataset.columns), output_path,
    )
    return dataset


def build_audio_dataset(
    audio_features_path: Path,
    earnings22_segments_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """
    Build audio-only dataset aggregated at the call level.

    Since Earnings-22 calls don't have market data yet,
    this dataset is useful for:
        - Audio feature analysis and visualization
        - Transfer learning pre-training
        - Quality assessment
    """
    af = pl.read_parquet(audio_features_path)

    # Extract call_id from segment_id (e.g., "4469590_2" → "4469590")
    if "call_id" not in af.columns:
        af = af.with_columns(
            pl.col("segment_id").str.replace(r"_\d+$", "").alias("call_id")
        )

    # Aggregate audio features to call level
    af_agg = aggregate_segment_features(af, call_id_col="call_id")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    af_agg.write_parquet(output_path)
    logger.info(
        "Audio dataset: %d calls, %d features → %s",
        len(af_agg), len(af_agg.columns), output_path,
    )
    return af_agg


def build_multimodal_dataset(
    audio_dataset_path: Path,
    text_features_path: Path,
    earnings22_segments_path: Path,
    market_data_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """
    Build a TRULY multimodal dataset using aligned Earnings-22 audio and text.
    
    Steps:
        1. Load Earnings-22 audio and text features
        2. Join them at the segment level (perfect alignment)
        3. Aggregate to call level
        4. Join with market targets (if available) or placeholders
    """
    af = pl.read_parquet(audio_dataset_path) # Actually call-level audio features
    tf_seg = pl.read_parquet(text_features_path) # Segment-level text features
    seg = pl.read_parquet(earnings22_segments_path) # Segment metadata (call_id)
    
    # 1. Add call_id to text features
    tf_seg = tf_seg.join(seg.select(["segment_id", "call_id"]), on="segment_id", how="left")
    
    # 2. Aggregate text features to call level
    tf_agg = aggregate_segment_features(tf_seg, call_id_col="call_id")
    
    # 3. Join with audio features
    # af is already at call level (from build_audio_dataset)
    dataset = af.join(tf_agg, on="call_id", how="inner", suffix="_text")
    
    # 4. Join with REAL market data
    processed = audio_dataset_path.parent
    market_path = processed / "earnings22_market_data.parquet"
    if market_path.exists():
        mkt = pl.read_parquet(market_path).filter(pl.col("data_source") == "real")
        dataset = dataset.join(
            mkt.select(["call_id", "ticker", "call_date", "return_1d", "return_5d", 
                        "realized_vol_5d", "close_t0", "close_t1", "close_t5", "data_source"]),
            on="call_id", how="inner"
        )
        # Fill any remaining nulls
        dataset = dataset.with_columns([
            pl.col("realized_vol_5d").fill_null(0.02),
            pl.col("return_1d").fill_null(0.0),
        ])
        real_count = dataset.filter(pl.col("data_source") == "real").height
        logger.info("Market data: %d real, %d synthetic targets", real_count, len(dataset) - real_count)
    else:
        logger.warning("No earnings22_market_data.parquet found. Run scripts/get_real_targets.py first!")
        import datetime
        base_date = datetime.date(2021, 1, 1)
        np.random.seed(42)
        dataset = dataset.with_columns([
            pl.Series("realized_vol_5d", np.random.uniform(0.01, 0.05, len(dataset)).astype(np.float32)),
            pl.Series("return_1d", np.random.normal(0.0, 0.02, len(dataset)).astype(np.float32)),
            pl.Series("call_date", [base_date + datetime.timedelta(days=30 * i) for i in range(len(dataset))]),
        ])
    
    dataset = dataset.with_columns([
        pl.lit(True).alias("has_real_audio"),
        pl.lit(True).alias("is_aligned")
    ])
    
    # 5. Join with interaction features (Phase 3)
    interaction_path = processed / "interaction_features.parquet"
    if interaction_path.exists():
        interaction = pl.read_parquet(interaction_path)
        dataset = dataset.join(interaction, on="call_id", how="left", suffix="_interaction")
        # Fill nulls in interaction features
        for col in interaction.columns:
            if col != "call_id":
                target_col = col if col in dataset.columns else f"{col}_interaction"
                if target_col in dataset.columns:
                    dataset = dataset.with_columns(pl.col(target_col).fill_null(0.0))
        logger.info("Added %d interaction features", len(interaction.columns) - 1)
    else:
        logger.warning("No interaction features found. Run interaction_assembler.py first.")



    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(output_path)
    logger.info(
        "ALIGNED Multimodal dataset: %d calls, %d features → %s",
        len(dataset), len(dataset.columns), output_path,
    )
    return dataset


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

    # 1. Text + Market dataset (AAPL baseline)
    logger.info("=" * 60)
    logger.info("Building Text + Market dataset (AAPL Baseline)...")
    logger.info("=" * 60)
    build_text_market_dataset(
        text_features_path=processed / "text_features.parquet",
        structural_features_path=processed / "structural_features.parquet",
        market_data_path=processed / "market_data.parquet",
        segments_path=processed / "segments.parquet",
        output_path=processed / "text_market_dataset.parquet",
    )

    # 2. Audio-only dataset (Earnings-22)
    audio_features_path = processed / "audio_features.parquet"
    if audio_features_path.exists():
        logger.info("=" * 60)
        logger.info("Building Audio dataset (Earnings-22)...")
        logger.info("=" * 60)
        build_audio_dataset(
            audio_features_path=audio_features_path,
            earnings22_segments_path=processed / "earnings22_segments.parquet",
            output_path=processed / "audio_dataset.parquet",
        )
    else:
        logger.warning("audio_features.parquet not found. Skipping audio dataset join.")

    # 3. Full multimodal dataset (Earnings-22 Aligned)
    # Only if the new text features exist
    aligned_text_path = processed / "earnings22_text_features.parquet"
    if aligned_text_path.exists():
        logger.info("=" * 60)
        logger.info("Building ALIGNED Multimodal dataset (Earnings-22)...")
        logger.info("=" * 60)
        build_multimodal_dataset(
            audio_dataset_path=processed / "audio_dataset.parquet",
            text_features_path=aligned_text_path,
            earnings22_segments_path=processed / "earnings22_segments.parquet",
            market_data_path=processed / "market_data.parquet",
            output_path=processed / "multimodal_dataset.parquet",
        )
    else:
        logger.warning("Aligned text features not found. Skipping aligned multimodal join.")


if __name__ == "__main__":
    main()
