"""
Transcript ingestion and cleaning for earnings call data.

Handles:
    - HuggingFace structured_content (list of {speaker, text} dicts)
    - Raw text transcripts (plain text or HTML)

Produces cleaned text with normalized whitespace and stripped boilerplate.

Phase 2 - Task 2.D.2
"""

import logging
import re
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Boilerplate patterns to strip
# ---------------------------------------------------------------------------

BOILERPLATE_PATTERNS = [
    # Legal disclaimers
    r"(?i)this\s+(?:conference\s+call|presentation|earnings\s+call)\s+(?:contains|may\s+contain|includes)\s+forward[- ]looking\s+statements.*?(?:\.|$)",
    r"(?i)safe\s+harbor\s+(?:statement|provision).*?(?:\.|$)",
    r"(?i)(?:these|such)\s+(?:forward[- ]looking\s+)?statements\s+(?:involve|are\s+subject\s+to)\s+(?:risks|uncertainties).*?(?:\.|$)",
    # Operator greetings / sign-offs
    r"(?i)^operator:\s*good\s+(?:morning|afternoon|evening).*?(?:begin|proceed)\.",
    r"(?i)this\s+concludes\s+(?:today's|the|our)\s+(?:conference|call|presentation).*$",
    r"(?i)you\s+may\s+(?:now\s+)?disconnect\s+your\s+lines?\.",
    # Copyright notices
    r"(?i)(?:©|copyright)\s+\d{4}.*$",
    # Replay instructions
    r"(?i)a\s+(?:replay|recording)\s+(?:of\s+)?(?:this\s+)?(?:conference|call)\s+(?:will\s+be|is)\s+available.*$",
]


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------


def strip_html_tags(text: str) -> str:
    """Remove HTML/XML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces, strip edges."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    """Remove known boilerplate patterns from earnings call transcripts."""
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
    return text


def normalize_encoding(text: str) -> str:
    """Fix common encoding artifacts."""
    replacements = {
        "\u2019": "'",   # Right single quote
        "\u2018": "'",   # Left single quote
        "\u201c": '"',   # Left double quote
        "\u201d": '"',   # Right double quote
        "\u2014": "—",   # Em dash (keep as-is, it's valid)
        "\u2013": "–",   # En dash
        "\u2026": "...", # Ellipsis
        "\xa0": " ",     # Non-breaking space
        "\r\n": "\n",    # Windows line endings
        "\r": "\n",      # Old Mac line endings
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_transcript_text(text: str) -> str:
    """
    Full cleaning pipeline for a single text block.

    Steps:
        1. Fix encoding artifacts
        2. Strip HTML tags
        3. Remove boilerplate
        4. Normalize whitespace
    """
    if not text:
        return ""
    text = normalize_encoding(text)
    text = strip_html_tags(text)
    text = strip_boilerplate(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Structured content processing
# ---------------------------------------------------------------------------


def process_structured_content(
    structured_content: list[dict],
    call_id: str,
) -> list[dict]:
    """
    Process HuggingFace structured_content into cleaned segment records.

    Each item in structured_content is a dict with 'speaker' and 'text' keys.
    Returns a list of cleaned segment dicts matching Contract A schema.
    """
    if not structured_content:
        return []

    segments = []
    for i, turn in enumerate(structured_content):
        speaker = turn.get("speaker", "") or ""
        text = turn.get("text", "") or ""

        cleaned = clean_transcript_text(text)
        if not cleaned:
            continue

        segments.append({
            "call_id": call_id,
            "segment_id": f"{call_id}_seg_{i:04d}",
            "speaker_name": speaker.strip(),
            "text": cleaned,
            "segment_index": i,
            "char_count": len(cleaned),
            "word_count": len(cleaned.split()),
        })

    return segments


# ---------------------------------------------------------------------------
# Raw text file processing
# ---------------------------------------------------------------------------


def process_raw_text_file(filepath: Path, call_id: str) -> list[dict]:
    """
    Read and clean a raw text transcript file.

    Splits on paragraph boundaries (double newlines) and produces
    one segment per paragraph.
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")
    text = clean_transcript_text(text)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    segments = []
    for i, para in enumerate(paragraphs):
        if len(para) < 10:  # Skip very short fragments
            continue
        segments.append({
            "call_id": call_id,
            "segment_id": f"{call_id}_seg_{i:04d}",
            "speaker_name": "",
            "text": para,
            "segment_index": i,
            "char_count": len(para),
            "word_count": len(para.split()),
        })

    return segments


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------


def ingest_transcripts(
    segments_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """
    Load existing segments.parquet, apply cleaning to all text fields,
    and save the cleaned version.
    """
    if not segments_path.exists():
        logger.error("Segments file not found: %s", segments_path)
        return pl.DataFrame()

    df = pl.read_parquet(segments_path)
    logger.info("Loaded %d segments from %s", len(df), segments_path)

    # Apply cleaning to the text column
    cleaned_texts = [clean_transcript_text(t) for t in df["text"].to_list()]

    df = df.with_columns(
        pl.Series(name="text", values=cleaned_texts)
    )

    # Filter out empty texts after cleaning
    before = len(df)
    df = df.filter(pl.col("text").str.len_chars() > 0)
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d empty segments after cleaning", dropped)

    # Add word count column for downstream use
    word_counts = [len(t.split()) for t in df["text"].to_list()]
    df = df.with_columns(
        pl.Series(name="word_count", values=word_counts)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info("Saved cleaned segments (%d rows) to: %s", len(df), output_path)

    return df


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

    # Clean the main segments.parquet (text-only pipeline)
    ingest_transcripts(
        segments_path=project_root / "data" / "processed" / "segments.parquet",
        output_path=project_root / "data" / "processed" / "segments_cleaned.parquet",
    )


if __name__ == "__main__":
    main()
