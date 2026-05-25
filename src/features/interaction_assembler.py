"""
Interaction Feature Assembler — Phase 3, Task 3.3

Orchestrates the extraction and assembly of all interaction-layer features:
    1. Text-Audio Divergence (segment-level)
    2. Q&A Pressure (call-level)

Produces: data/processed/interaction_features.parquet (call-level)
"""

import logging
from pathlib import Path

import polars as pl

from src.features.divergence import DivergenceCalculator
from src.features.qa_pressure import QAPressureExtractor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def aggregate_divergence_to_call(
    divergence_df: pl.DataFrame,
    segments_df: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate segment-level divergence features to call level."""
    # Add call_id
    div_with_call = divergence_df.join(
        segments_df.select(["segment_id", "call_id"]),
        on="segment_id", how="left"
    )
    
    # Aggregate per call
    div_cols = [c for c in divergence_df.columns if c != "segment_id"]
    agg_exprs = []
    for col in div_cols:
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).max().alias(f"{col}_max"),
            # Percentile 90 — captures extreme divergence moments
            pl.col(col).quantile(0.9).alias(f"{col}_p90"),
        ])
    
    result = div_with_call.group_by("call_id").agg(agg_exprs)
    return result


def build_interaction_features(project_root: Path):
    """Full interaction feature pipeline."""
    processed = project_root / "data" / "processed"
    
    # Load data
    segments = pl.read_parquet(processed / "earnings22_segments.parquet")
    text_features = pl.read_parquet(processed / "text_features.parquet")
    
    audio_path = processed / "audio_features.parquet"
    audio_features = pl.read_parquet(audio_path) if audio_path.exists() else None
    
    # 1. Compute Divergence Features (segment-level)
    logger.info("=" * 60)
    logger.info("Phase 3.1: Computing Text-Audio Divergence...")
    logger.info("=" * 60)
    
    if audio_features is not None:
        calculator = DivergenceCalculator()
        div_features = calculator.compute(text_features, audio_features, segments)
        div_features.write_parquet(processed / "divergence_features.parquet")
        
        # Aggregate to call level
        div_call = aggregate_divergence_to_call(div_features, segments)
        logger.info("Divergence: %d segments → %d calls, %d features",
                     len(div_features), len(div_call), len(div_call.columns) - 1)
    else:
        logger.warning("No audio features found, skipping divergence")
        div_features = None
        div_call = None
    
    # 2. Compute Q&A Pressure Features (call-level)
    logger.info("=" * 60)
    logger.info("Phase 3.2: Computing Q&A Pressure Features...")
    logger.info("=" * 60)
    
    qa_extractor = QAPressureExtractor()
    qa_features = qa_extractor.extract(segments, text_features, div_features)
    qa_features.write_parquet(processed / "qa_pressure_features.parquet")
    
    # 3. Merge into unified interaction features
    logger.info("=" * 60)
    logger.info("Phase 3.3: Assembling Interaction Features...")
    logger.info("=" * 60)
    
    if div_call is not None:
        interaction = qa_features.join(div_call, on="call_id", how="left")
    else:
        interaction = qa_features
    
    # Fill nulls with column-aware imputation (median for numeric)
    for col in interaction.columns:
        if col != "call_id" and interaction[col].dtype in (pl.Float32, pl.Float64):
            median_val = interaction[col].median()
            fill_val = median_val if median_val is not None else 0.0
            interaction = interaction.with_columns(pl.col(col).fill_null(fill_val))
    
    output_path = processed / "interaction_features.parquet"
    interaction.write_parquet(output_path)
    logger.info("Saved interaction features: %d calls, %d features → %s",
                len(interaction), len(interaction.columns) - 1, output_path)
    
    return interaction


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    build_interaction_features(project_root)
