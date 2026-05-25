"""
Audio quality assessment for earnings call segments.

Flags segments that are silent, clipped, or have low signal-to-noise ratio (SNR).
Produces a quality report used to filter unusable segments before modeling.

Outputs: data/processed/audio_quality.parquet
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torchaudio
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality Checker
# ---------------------------------------------------------------------------


class AudioQualityChecker:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def check_segment(self, audio_path: str) -> dict:
        """Analyze a single audio file for quality issues."""
        try:
            if not Path(audio_path).exists():
                return {"is_usable": False, "reason": "missing"}

            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[1] == 0:
                return {"is_usable": False, "reason": "empty"}

            # Convert to numpy for calculations
            y = waveform.squeeze().numpy()
            
            # 1. Duration check
            duration = len(y) / sr
            if duration < 0.5: # Skip segments shorter than 0.5s
                return {"is_usable": False, "reason": "too_short", "duration": duration}

            # 2. Clipping detection
            # Assuming normalized audio (-1, 1), check for values at extremes
            clipping_threshold = 0.99
            clipping_ratio = np.mean(np.abs(y) >= clipping_threshold)
            
            # 3. Silence ratio
            # Use a simple energy threshold
            energy = y**2
            silence_threshold = 0.001
            silence_ratio = np.mean(energy < silence_threshold)

            # 4. SNR Estimation (Simple approximation: peak / floor)
            # This is naive but helpful for flagging garbage
            noise_floor = np.percentile(np.abs(y), 10) + 1e-6
            peak_signal = np.max(np.abs(y))
            snr_estimate = 20 * np.log10(peak_signal / noise_floor)

            is_usable = (
                clipping_ratio < 0.05 and 
                silence_ratio < 0.8 and 
                snr_estimate > 5.0
            )

            return {
                "is_usable": bool(is_usable),
                "duration": float(duration),
                "clipping_ratio": float(clipping_ratio),
                "silence_ratio": float(silence_ratio),
                "snr_estimate": float(snr_estimate),
            }
        except Exception as e:
            logger.warning("Quality check failed for %s: %s", audio_path, e)
            return {"is_usable": False, "reason": "error"}

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run quality checks on a Polars DataFrame of segments."""
        logger.info("Running quality checks on %d segments...", len(segments))
        
        results = []
        for audio_path in tqdm(segments["audio_path"].to_list(), desc="Quality"):
            results.append(self.check_segment(audio_path))

        return segments.select("segment_id").with_columns(
            pl.from_dicts(results)
        )


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
    segments_path = project_root / "data" / "processed" / "earnings22_segments.parquet"
    output_path = project_root / "data" / "processed" / "audio_quality.parquet"

    if not segments_path.exists():
        logger.error("Segments file not found.")
        return

    df_segments = pl.read_parquet(segments_path)
    
    checker = AudioQualityChecker()
    df_quality = checker.extract(df_segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_quality.write_parquet(output_path)
    logger.info("Saved quality report to: %s", output_path)


if __name__ == "__main__":
    main()
