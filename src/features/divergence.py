"""
Text-Audio Divergence Features — Phase 3, Task 3.1

Detects misalignment between what executives SAY (text sentiment)
and how they SOUND (audio prosody). These divergence signals are
the "pressure cracks" that reveal hidden stress.

Features computed per segment:
    - sentiment_pitch_divergence: positive text + falling pitch = stress
    - sentiment_energy_divergence: positive text + low energy = doubt
    - confidence_stability_divergence: confident words + unstable voice
    - composite_divergence_score: weighted combination of all signals

Key insight from the README:
    "The strongest signals appear when a manager's narrative breaks under pressure."
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Divergence Calculator
# ---------------------------------------------------------------------------

class DivergenceCalculator:
    """
    Computes text-audio divergence features at the segment level.
    
    The core idea: when text sentiment is positive but audio features
    show stress (high pitch variance, low energy, unstable voice),
    the executive is likely under pressure and masking it.
    """
    
    def __init__(self):
        # Weights for composite score
        self.weights = {
            "sentiment_pitch": 0.30,
            "sentiment_energy": 0.25,
            "confidence_stability": 0.25,
            "hedging_prosody": 0.20,
        }
    
    def _normalize(self, series: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        mn, mx = np.nanmin(series), np.nanmax(series)
        if mx - mn < 1e-8:
            return np.zeros_like(series)
        return (series - mn) / (mx - mn)
    
    def compute(
        self,
        text_features: pl.DataFrame,
        audio_features: pl.DataFrame,
        segments: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute divergence features for all segments that have both
        text and audio features.
        
        Args:
            text_features: segment-level text features (sentiment_score, etc.)
            audio_features: segment-level audio features (pitch_mean, energy_variance, etc.)
            segments: segment metadata (segment_id, call_id)
            
        Returns:
            DataFrame with segment_id + divergence features
        """
        # Join text and audio on segment_id
        df = text_features.join(audio_features, on="segment_id", how="inner")
        
        if len(df) == 0:
            logger.warning("No overlapping segments between text and audio features")
            return pl.DataFrame({"segment_id": []})
        
        logger.info("Computing divergence features for %d segments...", len(df))
        
        # Extract arrays
        sentiment = df["sentiment_score"].to_numpy().astype(np.float64)
        
        # Audio features (may not all be present)
        pitch_mean = df["pitch_mean"].to_numpy().astype(np.float64) if "pitch_mean" in df.columns else np.zeros(len(df))
        pitch_var = df["pitch_variance"].to_numpy().astype(np.float64) if "pitch_variance" in df.columns else np.zeros(len(df))
        energy_var = df["energy_variance"].to_numpy().astype(np.float64) if "energy_variance" in df.columns else np.zeros(len(df))
        speech_rate = df["speech_rate"].to_numpy().astype(np.float64) if "speech_rate" in df.columns else np.zeros(len(df))
        voice_stab = df["voice_stability"].to_numpy().astype(np.float64) if "voice_stability" in df.columns else np.ones(len(df))
        
        # Text features
        uncertainty = df["uncertainty_score"].to_numpy().astype(np.float64) if "uncertainty_score" in df.columns else np.zeros(len(df))
        hedging = df["hedging_frequency"].to_numpy().astype(np.float64) if "hedging_frequency" in df.columns else np.zeros(len(df))
        
        # Replace NaN
        for arr in [sentiment, pitch_mean, pitch_var, energy_var, speech_rate, voice_stab, uncertainty, hedging]:
            np.nan_to_num(arr, copy=False, nan=0.0)
        
        # Normalize to [0, 1] for divergence computation
        norm_sentiment = self._normalize(sentiment)
        norm_pitch_var = self._normalize(pitch_var)
        norm_energy_var = self._normalize(energy_var)
        norm_voice_stab = self._normalize(voice_stab)
        norm_speech_rate = self._normalize(speech_rate)
        norm_hedging = self._normalize(hedging)
        
        # --- Divergence Features ---
        
        # 1. Sentiment-Pitch Divergence
        # High when: text is positive BUT pitch variance is high (stress)
        # Formula: sentiment * pitch_variance (both normalized)
        # A positive speaker with high pitch variance is "cracking"
        sent_pitch_div = np.abs(norm_sentiment - (1 - norm_pitch_var))
        
        # 2. Sentiment-Energy Divergence
        # High when: text is positive BUT energy is unstable
        sent_energy_div = np.abs(norm_sentiment - (1 - norm_energy_var))
        
        # 3. Confidence-Stability Divergence
        # High when: low uncertainty text BUT unstable voice
        confidence = 1 - self._normalize(uncertainty)  # invert: low uncertainty = high confidence
        conf_stab_div = np.abs(confidence - norm_voice_stab)
        
        # 4. Hedging-Prosody Divergence
        # High when: lots of hedging words BUT calm, steady voice (deliberate hedging)
        # OR: no hedging BUT stressed voice (hiding something)
        hedge_prosody_div = np.abs(norm_hedging - norm_pitch_var)
        
        # 5. Composite Divergence Score (weighted average)
        composite = (
            self.weights["sentiment_pitch"] * sent_pitch_div
            + self.weights["sentiment_energy"] * sent_energy_div
            + self.weights["confidence_stability"] * conf_stab_div
            + self.weights["hedging_prosody"] * hedge_prosody_div
        )
        
        # 6. Speech rate anomaly (deviation from speaker's mean)
        # Fast speech can indicate nervousness
        speech_rate_z = np.zeros_like(speech_rate)
        sr_std = np.std(speech_rate)
        if sr_std > 1e-8:
            speech_rate_z = (speech_rate - np.mean(speech_rate)) / sr_std
        
        result = pl.DataFrame({
            "segment_id": df["segment_id"],
            "sentiment_pitch_divergence": sent_pitch_div.astype(np.float32),
            "sentiment_energy_divergence": sent_energy_div.astype(np.float32),
            "confidence_stability_divergence": conf_stab_div.astype(np.float32),
            "hedging_prosody_divergence": hedge_prosody_div.astype(np.float32),
            "composite_divergence_score": composite.astype(np.float32),
            "speech_rate_anomaly": speech_rate_z.astype(np.float32),
        })
        
        logger.info("Divergence features: mean composite=%.4f, max=%.4f",
                     composite.mean(), composite.max())
        
        return result


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
    
    # Load features
    text_feat = pl.read_parquet(processed / "earnings22_text_features.parquet")
    audio_feat = pl.read_parquet(processed / "audio_features.parquet")
    segments = pl.read_parquet(processed / "earnings22_segments.parquet")
    
    calculator = DivergenceCalculator()
    result = calculator.compute(text_feat, audio_feat, segments)
    
    output_path = processed / "divergence_features.parquet"
    result.write_parquet(output_path)
    logger.info("Saved %d divergence features to %s", len(result), output_path)


if __name__ == "__main__":
    main()
