"""
GPU-Accelerated Prosody Feature Extraction for Earnings Call Segments.

Uses torchaudio on GPU for pitch detection (much faster than librosa.pyin)
and multiprocessing for remaining CPU operations.

Computes:
    - pitch_mean, pitch_variance (F0 via torchaudio GPU)
    - energy_variance (RMS via torch GPU)
    - speech_rate (onset detection)
    - voice_stability (jitter proxy)

Outputs: data/processed/audio_prosody.parquet
"""

import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import polars as pl
import torch
import torchaudio
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU Prosody Extractor
# ---------------------------------------------------------------------------


class ProsodyExtractor:
    def __init__(self, sample_rate: int = 16000, device: str = "auto"):
        self.sample_rate = sample_rate
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info("ProsodyExtractor using device: %s", self.device)

    def extract_segment_features(self, audio_path: str) -> dict:
        """Extract prosodic features using GPU-accelerated torchaudio."""
        try:
            if not Path(audio_path).exists():
                return self._empty()

            # Load audio with librosa (robust on Windows)
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            if len(y) == 0:
                return self._empty()

            waveform = torch.from_numpy(y).unsqueeze(0)  # [1, T]
            duration = len(y) / sr

            # 1. Pitch (F0) — GPU accelerated via torchaudio
            if self.device == "cuda":
                waveform_gpu = waveform.to(self.device)
                pitch = torchaudio.functional.detect_pitch_frequency(
                    waveform_gpu, self.sample_rate
                ).squeeze().cpu().numpy()
            else:
                pitch = torchaudio.functional.detect_pitch_frequency(
                    waveform, self.sample_rate
                ).squeeze().numpy()

            # Filter valid pitch (non-zero, reasonable range 50-600 Hz)
            f0_valid = pitch[(pitch > 50) & (pitch < 600)]
            pitch_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
            pitch_variance = float(np.var(f0_valid)) if len(f0_valid) > 0 else 0.0

            # 2. Energy (RMS) — GPU accelerated via torch
            if self.device == "cuda":
                frame_len = 2048
                hop = 512
                w_gpu = waveform_gpu.squeeze()
                n_frames = max(1, (len(w_gpu) - frame_len) // hop + 1)
                rms_vals = []
                for i in range(n_frames):
                    frame = w_gpu[i * hop: i * hop + frame_len]
                    rms_vals.append(torch.sqrt(torch.mean(frame ** 2)).item())
                rms = np.array(rms_vals)
            else:
                rms = librosa.feature.rms(y=y)[0]

            energy_variance = float(np.var(rms)) if len(rms) > 0 else 0.0

            # 3. Speech Rate (onset detection — CPU, fast)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            peaks = librosa.util.peak_pick(
                onset_env, pre_max=3, post_max=3,
                pre_avg=3, post_avg=5, delta=0.5, wait=10
            )
            speech_rate = len(peaks) / duration if duration > 0 else 0.0

            # 4. Voice Stability (Jitter)
            voice_stability = 0.0
            if len(f0_valid) > 1:
                jitter = np.abs(np.diff(f0_valid))
                voice_stability = float(np.mean(jitter))

            return {
                "pitch_mean": pitch_mean,
                "pitch_variance": pitch_variance,
                "energy_variance": energy_variance,
                "speech_rate": speech_rate,
                "voice_stability": voice_stability,
            }
        except Exception as e:
            logger.warning("Failed to process %s: %s", audio_path, e)
            return self._empty()

    def _empty(self):
        return {
            "pitch_mean": 0.0,
            "pitch_variance": 0.0,
            "energy_variance": 0.0,
            "speech_rate": 0.0,
            "voice_stability": 0.0,
        }

    def extract(self, segments: pl.DataFrame, output_path: Path) -> pl.DataFrame:
        """Run extraction with checkpointing."""
        logger.info("Extracting prosody features from %d segments on %s...",
                    len(segments), self.device)

        # Check for existing progress
        existing_ids = set()
        if output_path.exists():
            try:
                df_existing = pl.read_parquet(output_path)
                existing_ids = set(df_existing["segment_id"].to_list())
                logger.info("Found %d already processed", len(existing_ids))
            except Exception:
                pass

        df_todo = segments.filter(~pl.col("segment_id").is_in(list(existing_ids)))
        if len(df_todo) == 0:
            logger.info("All segments already processed!")
            return pl.read_parquet(output_path)

        logger.info("Processing %d remaining segments...", len(df_todo))
        results = []
        seg_ids = df_todo["segment_id"].to_list()
        audio_paths = df_todo["audio_path"].to_list()

        try:
            for i, (sid, apath) in enumerate(tqdm(
                zip(seg_ids, audio_paths), total=len(seg_ids), desc="Prosody (GPU)"
            )):
                feats = self.extract_segment_features(apath)
                feats["segment_id"] = sid
                results.append(feats)

                # Checkpoint every 500 segments
                if (i + 1) % 500 == 0 and results:
                    self._save_checkpoint(results, output_path, existing_ids)
                    logger.info("Checkpoint at %d/%d", i + 1, len(df_todo))
                    results = []
                    if output_path.exists():
                        existing_ids = set(pl.read_parquet(output_path)["segment_id"].to_list())

            if results:
                self._save_checkpoint(results, output_path, existing_ids)

            df_final = pl.read_parquet(output_path)
            logger.info("Prosody extraction complete: %d segments", len(df_final))
            return df_final

        except KeyboardInterrupt:
            logger.info("Interrupted! Saving progress...")
            if results:
                self._save_checkpoint(results, output_path, existing_ids)
            raise

    def _save_checkpoint(self, results, output_path, existing_ids):
        df_new = pl.DataFrame(results)
        if output_path.exists():
            df_old = pl.read_parquet(output_path)
            new_only = df_new.filter(~pl.col("segment_id").is_in(list(existing_ids)))
            df_final = pl.concat([df_old, new_only], how="diagonal_relaxed")
        else:
            df_final = df_new
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.write_parquet(output_path)


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
    output_path = project_root / "data" / "processed" / "audio_prosody.parquet"

    if not segments_path.exists():
        logger.error("earnings22_segments.parquet not found.")
        return

    df_segments = pl.read_parquet(segments_path)

    # Pre-filter: only keep segments where audio file exists
    before = len(df_segments)
    df_segments = df_segments.filter(
        pl.col("audio_path").map_elements(lambda p: Path(p).exists(), return_dtype=pl.Boolean)
    )
    after = len(df_segments)
    if before != after:
        logger.info("Filtered %d → %d segments (skipped %d missing)", before, after, before - after)

    extractor = ProsodyExtractor()
    extractor.extract(df_segments, output_path)


if __name__ == "__main__":
    main()
