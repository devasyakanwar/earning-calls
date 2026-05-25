"""
Vocal embedding extraction using wav2vec 2.0 for earnings call segments.

Extracts deep acoustic representations from segments using a pre-trained
wav2vec2 model. Embeddings are mean-pooled across time to produce a 
768-dimensional vector per segment.

Outputs: data/processed/audio_wav2vec2.parquet
"""

import logging
from pathlib import Path

import polars as pl
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# wav2vec2 Extractor
# ---------------------------------------------------------------------------


class Wav2Vec2Extractor:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check for Apple Silicon MPS
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"

        logger.info("Loading wav2vec2 model: %s on %s", self.model_name, self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def extract_embedding(self, audio_path: str) -> list[float] | None:
        """Extract mean-pooled embedding for a single audio file."""
        try:
            # Use librosa for robust loading on Windows (avoiding TorchCodec error)
            waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(waveform).unsqueeze(0) # [1, T]

            # Preprocess
            inputs = self.processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)

            # Inference
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
            except Exception as e:
                # Automatic fallback to CPU if MPS fails (common for long segments on Apple Silicon)
                if "MPS" in str(e) or "channels > 65536" in str(e):
                    logger.info("MPS limit reached for %s, falling back to CPU for this segment", Path(audio_path).name)
                    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                    self.model.to("cpu")
                    with torch.no_grad():
                        outputs = self.model(**inputs_cpu)
                    # Move model back to original device for next segment
                    self.model.to(self.device)
                else:
                    raise e
            
            # Mean pooling across time (dimension 1)
            # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return embeddings.cpu().tolist()
        except Exception as e:
            logger.warning("wav2vec2 failed for %s: %s", audio_path, e)
            return None

    def extract(self, df_segments: pl.DataFrame, output_path: Path) -> pl.DataFrame:
        """Extract embeddings for all segments with checkpointing and memory cleanup."""
        logger.info("Extracting wav2vec2 embeddings from %d segments...", len(df_segments))

        # Check for existing progress
        existing_ids = set()
        if output_path.exists():
            try:
                df_existing = pl.read_parquet(output_path)
                existing_ids = set(df_existing["segment_id"].to_list())
                logger.info("Found existing progress: %d segments already processed", len(existing_ids))
            except Exception as e:
                logger.warning("Could not read existing output file, starting fresh: %s", e)

        # Filter to only unprocessed segments
        df_todo = df_segments.filter(~pl.col("segment_id").is_in(list(existing_ids)))

        if len(df_todo) == 0:
            logger.info("All segments already processed.")
            return pl.read_parquet(output_path)

        logger.info("Processing %d remaining segments...", len(df_todo))
        results = []

        import gc
        try:
            for i, row in enumerate(tqdm(df_todo.iter_rows(named=True), total=len(df_todo), desc="wav2vec2")):
                emb = self.extract_embedding(row["audio_path"])
                if emb is not None:
                    # Store as segment_id + individual embedding dimensions
                    record = {"segment_id": row["segment_id"]}
                    for j, val in enumerate(emb):
                        record[f"wav2vec2_{j}"] = val
                    results.append(record)
                    del emb # Explicitly delete embedding tensor/list

                # Heavy memory cleanup every segment
                if self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect() # Force Python to clear RAM
                
                # Checkpoint every 50 segments (Save more often to clear results list)
                if (i + 1) % 50 == 0 and len(results) > 0:
                    self._save_checkpoint(results, output_path, existing_ids)
                    logger.info("Checkpoint at %d/%d", i + 1, len(df_todo))
                    results = []
                    # Refresh existing_ids
                    if output_path.exists():
                        df_check = pl.read_parquet(output_path)
                        existing_ids = set(df_check["segment_id"].to_list())

            # Final save
            if len(results) > 0:
                self._save_checkpoint(results, output_path, existing_ids)

            df_final = pl.read_parquet(output_path)
            logger.info("Extraction complete: %d total segments saved.", len(df_final))
            return df_final

        except KeyboardInterrupt:
            logger.info("Interrupted! Saving progress...")
            if len(results) > 0:
                self._save_checkpoint(results, output_path, existing_ids)
            raise

    def _save_checkpoint(self, results: list[dict], output_path: Path, existing_ids: set):
        """Save intermediate results, merging with any existing file."""
        df_new = pl.DataFrame(results)

        if output_path.exists():
            df_old = pl.read_parquet(output_path)
            # Remove duplicates
            new_only = df_new.filter(~pl.col("segment_id").is_in(list(existing_ids)))
            df_final = pl.concat([df_old, new_only], how="vertical")
        else:
            df_final = df_new

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.write_parquet(output_path)
        logger.info("Saved checkpoint: %d total segments in %s", len(df_final), output_path.name)


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
    output_path = project_root / "data" / "processed" / "audio_wav2vec2.parquet"

    if not segments_path.exists():
        logger.error("Segments file not found.")
        return

    extractor = Wav2Vec2Extractor()
    df_segments = pl.read_parquet(segments_path)

    # Pre-filter: only keep segments where audio file exists on disk
    before = len(df_segments)
    df_segments = df_segments.filter(
        pl.col("audio_path").map_elements(lambda p: Path(p).exists(), return_dtype=pl.Boolean)
    )
    after = len(df_segments)
    if before != after:
        logger.info("Filtered %d → %d segments (skipped %d missing audio files)",
                    before, after, before - after)

    extractor.extract(df_segments, output_path)


if __name__ == "__main__":
    main()
