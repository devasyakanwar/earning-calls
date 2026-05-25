"""
Uncertainty and hedging detection for earnings call segments.

Computes:
    - uncertainty_score: ratio of tokens matching uncertainty lexicon
    - hedging_frequency: ratio of phrases matching hedging lexicon

Outputs: data/processed/text_uncertainty.parquet
"""

import logging
import re
from pathlib import Path

import polars as pl
import spacy
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Uncertainty Detector
# ---------------------------------------------------------------------------


class UncertaintyDetector:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
            self.config = full_config["uncertainty"]

        project_root = config_path.parent.parent
        
        # Load Lexicons
        with open(project_root / self.config["lexicon_path"], "r") as f:
            self.uncertainty_terms = set(yaml.safe_load(f)["terms"])
        
        with open(project_root / self.config["hedging_path"], "r") as f:
            self.hedging_phrases = yaml.safe_load(f)["phrases"]

        # Load spaCy for tokenization
        logger.info("Loading spaCy model: en_core_web_sm")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            logger.info("Downloading spaCy model: en_core_web_sm")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def process_segment(self, text: str) -> dict:
        """Compute uncertainty and hedging scores for a single text."""
        if not text:
            return {"uncertainty_score": 0.0, "hedging_frequency": 0.0}

        doc = self.nlp(text.lower())
        tokens = [t.text for t in doc if not t.is_punct and not t.is_space]
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return {"uncertainty_score": 0.0, "hedging_frequency": 0.0}

        # 1. Uncertainty Score (word-level)
        uncertainty_count = sum(1 for t in tokens if t in self.uncertainty_terms)
        uncertainty_score = uncertainty_count / total_tokens

        # 2. Hedging Frequency (phrase-level)
        # Use regex for phrase matching to be robust to whitespace
        hedging_count = 0
        text_lower = text.lower()
        for phrase in self.hedging_phrases:
            # Simple count of occurrences
            # Note: This might double count overlapping phrases, but good for frequency
            matches = re.findall(rf"\b{re.escape(phrase)}\b", text_lower)
            hedging_count += len(matches)
        
        hedging_frequency = hedging_count / total_tokens

        return {
            "uncertainty_score": uncertainty_score,
            "hedging_frequency": hedging_frequency,
        }

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run extraction on a Polars DataFrame of segments."""
        logger.info("Extracting uncertainty and hedging features from %d segments...", len(segments))
        
        results = []
        for text in tqdm(segments["text"].to_list(), desc="Uncertainty"):
            results.append(self.process_segment(text))

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
    config_path = project_root / "configs" / "text_config.yaml"
    segments_path = project_root / "data" / "processed" / "segments.parquet"
    output_path = project_root / "data" / "processed" / "text_uncertainty.parquet"

    if not segments_path.exists():
        logger.error("segments.parquet not found.")
        return

    df_segments = pl.read_parquet(segments_path)
    
    detector = UncertaintyDetector(config_path)
    df_features = detector.extract(df_segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved uncertainty features to: %s", output_path)


if __name__ == "__main__":
    main()
