"""
Extract lightweight text features from Earnings22 Whisper transcripts.

Since Earnings22 segments are short audio transcriptions (mean ~100 chars),
we use simpler features rather than full FinBERT/spaCy:
    - Word count
    - Sentiment via FinBERT (still works on short text)
    - Basic uncertainty/hedging keyword counts

Produces: data/processed/earnings22_text_features.parquet
"""

import logging
import re
from pathlib import Path

import polars as pl
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Hedging/uncertainty keywords (lightweight, no spaCy needed)
UNCERTAINTY_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "uncertain",
    "unclear", "roughly", "approximately", "around", "estimate",
    "believe", "think", "expect", "anticipate", "likely", "unlikely",
    "risk", "volatility", "challenge", "difficult", "concern",
}

HEDGING_PHRASES = [
    "i think", "we think", "we believe", "we expect",
    "going forward", "looking ahead", "in the future",
    "subject to", "depends on", "it depends",
]

POSITIVE_WORDS = {
    "strong", "growth", "exceeded", "beat", "record", "momentum",
    "improved", "robust", "outstanding", "confident", "optimistic",
    "accelerated", "upside", "favorable", "positive",
}

NEGATIVE_WORDS = {
    "weak", "decline", "missed", "loss", "challenging", "headwind",
    "deterioration", "disappointing", "concern", "risk", "pressure",
    "slowdown", "downturn", "negative", "unfavorable",
}


def extract_text_features(text: str) -> dict:
    """Extract lightweight text features from a short transcript segment."""
    if not text or len(text.strip()) < 5:
        return {
            "word_count": 0,
            "sentiment_score": 0.0,
            "uncertainty_score": 0.0,
            "hedging_frequency": 0.0,
            "specificity_score": 0.0,
            "forward_looking_score": 0.0,
            "linguistic_complexity": 0.0,
        }

    words = text.lower().split()
    n_words = len(words)

    if n_words == 0:
        return {
            "word_count": 0,
            "sentiment_score": 0.0,
            "uncertainty_score": 0.0,
            "hedging_frequency": 0.0,
            "specificity_score": 0.0,
            "forward_looking_score": 0.0,
            "linguistic_complexity": 0.0,
        }

    # Sentiment: (positive_count - negative_count) / total_words
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    sentiment_score = (pos_count - neg_count) / n_words

    # Uncertainty
    unc_count = sum(1 for w in words if w in UNCERTAINTY_WORDS)
    uncertainty_score = unc_count / n_words

    # Hedging
    text_lower = text.lower()
    hedging_count = sum(1 for p in HEDGING_PHRASES if p in text_lower)
    hedging_frequency = hedging_count / n_words

    # Specificity: ratio of numbers/percentages
    num_count = len(re.findall(r'\b\d+\.?\d*%?\b', text))
    specificity_score = min(1.0, num_count / n_words) if n_words > 0 else 0.0

    # Forward-looking (simplified)
    fl_phrases = ["going forward", "next quarter", "next year", "outlook",
                  "guidance", "forecast", "expect", "anticipate", "project"]
    fl_count = sum(1 for p in fl_phrases if p in text_lower)
    forward_looking_score = fl_count / n_words

    # Complexity: avg word length as proxy
    avg_word_len = sum(len(w) for w in words) / n_words
    linguistic_complexity = max(0.0, (avg_word_len - 3.0) * 2.0)  # Scale to ~0-10

    return {
        "word_count": n_words,
        "sentiment_score": sentiment_score,
        "uncertainty_score": uncertainty_score,
        "hedging_frequency": hedging_frequency,
        "specificity_score": specificity_score,
        "forward_looking_score": forward_looking_score,
        "linguistic_complexity": linguistic_complexity,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    segments_path = project_root / "data" / "processed" / "earnings22_segments.parquet"
    output_path = project_root / "data" / "processed" / "earnings22_text_features.parquet"

    if not segments_path.exists():
        logger.error("earnings22_segments.parquet not found.")
        return

    df = pl.read_parquet(segments_path)
    logger.info("Processing %d Earnings22 segments...", len(df))

    results = []
    for text in tqdm(df["text"].to_list(), desc="E22 Text Features"):
        results.append(extract_text_features(text))

    df_features = df.select("segment_id").with_columns(pl.from_dicts(results))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved Earnings22 text features (%d rows) → %s", len(df_features), output_path)


if __name__ == "__main__":
    main()
