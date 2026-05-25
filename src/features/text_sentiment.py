"""
FinBERT sentiment extraction pipeline for earnings call segments.

Loads segments.parquet, runs batch inference using ProsusAI/finbert,
and computes a continuous sentiment score: (positive_prob - negative_prob).

Outputs: data/processed/text_sentiment.parquet
"""

import logging
from pathlib import Path

import polars as pl
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment Pipeline
# ---------------------------------------------------------------------------


class SentimentExtractor:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["sentiment"]

        self.model_name = self.config["model_name"]
        self.batch_size = self.config["batch_size"]
        self.max_length = self.config["max_length"]
        self.device = self.config["device"]

        # Check for GPU/MPS if 'cpu' is not explicitly requested or if it's set to auto
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        logger.info("Loading model: %s on %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        self.nlp = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            top_k=None,  # Get all scores (pos, neg, neut)
        )

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run batch inference on a Polars DataFrame of segments."""
        texts = segments["text"].to_list()
        results = []

        logger.info("Running sentiment inference on %d segments (batch_size=%d)...", 
                    len(texts), self.batch_size)

        # Batch processing via transformers pipeline
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Sentiment"):
            batch = texts[i : i + self.batch_size]
            
            # Truncation is handled by the pipeline/tokenizer if max_length is set
            outputs = self.nlp(batch, truncation=True, max_length=self.max_length)
            
            for out in outputs:
                # out is a list of dicts like: [{'label': 'positive', 'score': 0.9}, ...]
                scores = {d["label"]: d["score"] for d in out}
                
                # Formula: Positive - Negative (Neutral is ignored in direction, but affects magnitude)
                sentiment_score = scores.get("positive", 0.0) - scores.get("negative", 0.0)
                results.append(sentiment_score)

        return segments.select("segment_id").with_columns(
            pl.Series(name="sentiment_score", values=results)
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
    config_path = project_root / "configs" / "text_config.yaml"
    segments_path = project_root / "data" / "processed" / "segments.parquet"
    output_path = project_root / "data" / "processed" / "text_sentiment.parquet"

    if not segments_path.exists():
        logger.error("segments.parquet not found. Run download_transcripts.py first.")
        return

    # Load segments
    df_segments = pl.read_parquet(segments_path)
    
    # Run extraction
    extractor = SentimentExtractor(config_path)
    df_sentiment = extractor.extract(df_segments)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sentiment.write_parquet(output_path)
    logger.info("Saved sentiment features to: %s", output_path)


if __name__ == "__main__":
    main()
