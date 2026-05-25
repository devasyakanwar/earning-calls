"""
Acoustic feature extraction using openSMILE for earnings call segments.

Extracts the eGeMAPSv02 (Extended Geneva Minimalistic Acoustic Parameter Set) 
which contains 88 functional features including F0, jitter, shimmer, 
formants, and spectral parameters.

Outputs: data/processed/audio_opensmile.parquet
"""

import logging
from pathlib import Path

import opensmile
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# openSMILE Extractor
# ---------------------------------------------------------------------------


class OpenSmileExtractor:
    def __init__(self, sample_rate: int = 16000):
        # Initialize openSMILE with eGeMAPSv02
        logger.info("Initializing openSMILE with eGeMAPSv02 feature set...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.sample_rate = sample_rate

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run batch extraction on a Polars DataFrame of segments."""
        logger.info("Extracting openSMILE features from %d segments...", len(segments))
        
        results = []
        segment_ids = []
        
        for row in tqdm(segments.iter_rows(named=True), desc="openSMILE"):
            audio_path = row["audio_path"]
            seg_id = row["segment_id"]
            
            if not Path(audio_path).exists():
                continue
                
            try:
                # openSMILE handles loading and feature extraction
                # It returns a pandas DataFrame with one row
                df_smile = self.smile.process_file(audio_path)
                
                # Convert to dict and add to results
                smile_dict = df_smile.iloc[0].to_dict()
                results.append(smile_dict)
                segment_ids.append(seg_id)
            except Exception as e:
                logger.warning("openSMILE failed for segment %s: %s", seg_id, e)

        if not results:
            return pl.DataFrame({"segment_id": []})

        # Create Polars DataFrame from features
        df_features = pl.from_dicts(results)
        
        # Add segment_id back
        return pl.DataFrame({"segment_id": segment_ids}).with_columns(df_features)


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
    # We'll use the E22 segments as primary source for Phase 2 Path A
    segments_path = project_root / "data" / "processed" / "earnings22_segments.parquet"
    output_path = project_root / "data" / "processed" / "audio_opensmile.parquet"

    if not segments_path.exists():
        logger.error("Segments file not found: %s", segments_path)
        return

    df_segments = pl.read_parquet(segments_path)
    
    # For initial run, handle a subset to verify speed
    # df_segments = df_segments.head(50)

    extractor = OpenSmileExtractor()
    df_features = extractor.extract(df_segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved openSMILE features (%d rows) to: %s", len(df_features), output_path)


if __name__ == "__main__":
    main()
