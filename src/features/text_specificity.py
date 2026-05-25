"""
Specificity and linguistic complexity scoring for earnings call segments.

Computes:
    - specificity_score: (numeric_tokens + named_entities) / total_tokens
    - forward_looking_score: ratio of forward-looking phrases from lexicon
    - linguistic_complexity: Flesch-Kincaid Grade Level

Outputs: data/processed/text_specificity.parquet
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
# Specificity Scorer
# ---------------------------------------------------------------------------


class SpecificityScorer:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["specificity"]

        project_root = config_path.parent.parent
        
        # Load Lexicon
        with open(project_root / self.config["forward_looking_lexicon"], "r") as f:
            self.fls_phrases = yaml.safe_load(f)["phrases"]

        # Load spaCy for NER and tokenization
        logger.info("Loading spaCy model: %s", self.config["ner_model"])
        try:
            self.nlp = spacy.load(self.config["ner_model"])
        except OSError:
            logger.info("Downloading spaCy model: %s", self.config["ner_model"])
            spacy.cli.download(self.config["ner_model"])
            self.nlp = spacy.load(self.config["ner_model"])

    def _count_syllables(self, word: str) -> int:
        """Simple heuristic for counting syllables in a word."""
        word = word.lower()
        if not word:
            return 0
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    def compute_flesch_kincaid(self, text: str, tokens: list[str], sentences: list) -> float:
        """
        Compute Flesch-Kincaid Grade Level.
        Formula: 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        """
        total_words = len(tokens)
        total_sentences = len(sentences)
        
        if total_words == 0 or total_sentences == 0:
            return 0.0

        total_syllables = sum(self._count_syllables(w) for w in tokens)
        
        score = (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59
        return max(0.0, score)

    def process_segment(self, text: str) -> dict:
        """Compute specificity and complexity for a single text."""
        if not text:
            return {
                "specificity_score": 0.0, 
                "forward_looking_score": 0.0,
                "linguistic_complexity": 0.0
            }

        doc = self.nlp(text)
        
        # 1. Specificity Score
        # Tokens that are numbers (CARDINAL, MONEY, PERCENT, QUANTITY)
        # or have numerical digits
        num_tokens = sum(1 for t in doc if t.like_num or re.search(r"\d", t.text))
        
        # Named Entities (excluding numbers already counted if needed, 
        # but usually NER labels are specific enough)
        # We count unique named entity spans
        entities = len(doc.ents)
        
        total_tokens = len([t for t in doc if not t.is_punct and not t.is_space])
        if total_tokens == 0:
            return {
                "specificity_score": 0.0, 
                "forward_looking_score": 0.0,
                "linguistic_complexity": 0.0
            }
            
        specificity_score = (num_tokens + entities) / total_tokens

        # 2. Forward-Looking Score
        fls_count = 0
        text_lower = text.lower()
        for phrase in self.fls_phrases:
            matches = re.findall(rf"\b{re.escape(phrase)}\b", text_lower)
            fls_count += len(matches)
        
        forward_looking_score = fls_count / total_tokens

        # 3. Linguistic Complexity
        tokens_list = [t.text for t in doc if not t.is_punct and not t.is_space]
        sentences_list = list(doc.sents)
        complexity = self.compute_flesch_kincaid(text, tokens_list, sentences_list)

        return {
            "specificity_score": min(1.0, specificity_score),
            "forward_looking_score": forward_looking_score,
            "linguistic_complexity": complexity,
        }

    def extract(self, segments: pl.DataFrame) -> pl.DataFrame:
        """Run extraction on a Polars DataFrame of segments."""
        logger.info("Extracting specificity and complexity features from %d segments...", len(segments))
        
        results = []
        for text in tqdm(segments["text"].to_list(), desc="Specificity"):
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
    output_path = project_root / "data" / "processed" / "text_specificity.parquet"

    if not segments_path.exists():
        logger.error("segments.parquet not found.")
        return

    df_segments = pl.read_parquet(segments_path)
    
    scorer = SpecificityScorer(config_path)
    df_features = scorer.extract(df_segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    logger.info("Saved specificity features to: %s", output_path)


if __name__ == "__main__":
    main()
