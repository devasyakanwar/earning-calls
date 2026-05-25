"""
Whisper Feature Extraction for Earnings Call Audio — GPU Accelerated

Uses OpenAI's Whisper model encoder to extract:
    - whisper_embedding_mean: Mean of encoder hidden states (768-dim → compressed)
    - whisper_embedding_std: Std of encoder hidden states
    - whisper_confidence_mean: Average token confidence (low = hesitation/mumbling)
    - whisper_confidence_min: Minimum token confidence (worst moment)
    - whisper_confidence_std: Variance in confidence (inconsistent = stressed)
    - whisper_avg_logprob: Average log probability of transcription

Outputs: data/processed/audio_whisper.parquet
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper Feature Extractor
# ---------------------------------------------------------------------------


class WhisperFeatureExtractor:
    """Extract speech features using OpenAI Whisper's encoder."""

    def __init__(self, model_name: str = "base", device: str = "auto"):
        """
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       'base' is recommended for feature extraction (balance of speed + quality)
            device: 'cuda', 'cpu', or 'auto'
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading Whisper '%s' model on %s...", model_name, self.device)

        import whisper
        self.model = whisper.load_model(model_name, device=self.device)
        self.model.eval()
        logger.info("Whisper model loaded successfully")

    @torch.no_grad()
    def extract_segment_features(self, audio_path: str) -> dict:
        """Extract features from a single audio segment using Whisper."""
        try:
            if not Path(audio_path).exists():
                return self._empty_features()

            import whisper
            import librosa

            # Load audio with librosa (bypasses ffmpeg dependency)
            y, _ = librosa.load(audio_path, sr=16000, mono=True)
            audio = torch.from_numpy(y).float().numpy()
            audio = whisper.pad_or_trim(audio)

            # Get mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            # 1. Get encoder hidden states (deep speech representation)
            encoder_output = self.model.encoder(mel)  # [1, T, D]
            hidden = encoder_output.squeeze(0).cpu().numpy()  # [T, D]

            # Compress 768/512-dim embeddings to summary statistics
            emb_mean = hidden.mean(axis=0)  # [D]
            emb_std = hidden.std(axis=0)    # [D]

            # Take top-8 PCA-like components (mean of chunks)
            n_components = 8
            chunk_size = max(1, len(emb_mean) // n_components)
            emb_compressed = [
                float(np.mean(emb_mean[i * chunk_size:(i + 1) * chunk_size]))
                for i in range(n_components)
            ]
            emb_std_compressed = [
                float(np.mean(emb_std[i * chunk_size:(i + 1) * chunk_size]))
                for i in range(n_components)
            ]

            # 2. Run transcription to get confidence metrics
            result = self.model.transcribe(
                audio,
                language="en",
                fp16=(self.device == "cuda"),
                verbose=False,
            )

            # Extract confidence from segments
            segment_probs = []
            for seg in result.get("segments", []):
                avg_logprob = seg.get("avg_logprob", -1.0)
                segment_probs.append(avg_logprob)

            if segment_probs:
                conf_mean = float(np.mean(segment_probs))
                conf_min = float(np.min(segment_probs))
                conf_std = float(np.std(segment_probs))
                avg_logprob = conf_mean
            else:
                conf_mean = -1.0
                conf_min = -1.0
                conf_std = 0.0
                avg_logprob = -1.0

            # Build feature dict
            features = {
                "whisper_confidence_mean": conf_mean,
                "whisper_confidence_min": conf_min,
                "whisper_confidence_std": conf_std,
                "whisper_avg_logprob": avg_logprob,
                "whisper_encoder_energy": float(np.mean(np.abs(hidden))),
                "whisper_encoder_variance": float(np.var(hidden)),
            }

            # Add compressed embeddings
            for i, (m, s) in enumerate(zip(emb_compressed, emb_std_compressed)):
                features[f"whisper_emb_mean_{i}"] = m
                features[f"whisper_emb_std_{i}"] = s

            return features

        except Exception as e:
            logger.warning("Failed to process %s: %s", audio_path, e)
            return self._empty_features()

    def _empty_features(self) -> dict:
        """Return empty feature dict for failed segments."""
        features = {
            "whisper_confidence_mean": 0.0,
            "whisper_confidence_min": 0.0,
            "whisper_confidence_std": 0.0,
            "whisper_avg_logprob": -1.0,
            "whisper_encoder_energy": 0.0,
            "whisper_encoder_variance": 0.0,
        }
        for i in range(8):
            features[f"whisper_emb_mean_{i}"] = 0.0
            features[f"whisper_emb_std_{i}"] = 0.0
        return features

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run extraction on a Polars DataFrame of segments."""
        logger.info("Extracting Whisper features from %d segments on %s...",
                    len(segments), self.device)

        results = []
        for audio_path in tqdm(segments["audio_path"].to_list(), desc="Whisper (GPU)"):
            results.append(self.extract_segment_features(audio_path))

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
    output_path = project_root / "data" / "processed" / "audio_whisper.parquet"

    if not segments_path.exists():
        logger.error("earnings22_segments.parquet not found. Run download_earnings22.py first.")
        return

    df_segments = pl.read_parquet(segments_path)
    logger.info("Loaded %d segments", len(df_segments))

    # Pre-filter: only keep segments where audio file exists on disk
    before = len(df_segments)
    df_segments = df_segments.filter(
        pl.col("audio_path").map_elements(lambda p: Path(p).exists(), return_dtype=pl.Boolean)
    )
    after = len(df_segments)
    if before != after:
        logger.info("Filtered %d → %d segments (skipped %d missing audio files)",
                    before, after, before - after)

    # Check if already partially computed (checkpointing)
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        done_ids = set(existing["segment_id"].to_list())
        remaining = df_segments.filter(~pl.col("segment_id").is_in(list(done_ids)))
        logger.info("Found %d already processed, %d remaining", len(done_ids), len(remaining))
        if len(remaining) == 0:
            logger.info("All segments already processed!")
            return
        df_segments = remaining
    else:
        existing = None

    extractor = WhisperFeatureExtractor(model_name="base", device="auto")
    df_features = extractor.extract(df_segments)

    # Merge with existing if checkpointing
    if existing is not None:
        df_features = pl.concat([existing, df_features])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved Whisper features (%d segments) to: %s", len(df_features), output_path)


if __name__ == "__main__":
    main()
