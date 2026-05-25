#!/usr/bin/env python3
"""
Download and process the FinCall-Surprise dataset from GitHub.
Parses the transcripts into segments, running role and segment type classifications,
and outputs standard parquet files matching the pipeline's schema.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
import json

import polars as pl
import requests
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.speaker_classification import (
    classify_speaker_role,
    classify_segment_type,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRANSCRIPT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/Tizzzzy/FinCall-Surprise/main/transcripts_{year}.json"
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Regex pattern to split transcripts by speaker prefixes
SPEAKER_PATTERN = r'\b(Executives:|Analysts:|Operator:)'

# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def process_call_transcript(call_id: str, raw_text: str) -> list[dict]:
    """Parse the raw transcript text into segments."""
    parts = re.split(SPEAKER_PATTERN, raw_text)
    if not parts:
        return []

    # Reassemble speaker turns
    segments = []
    i = 0
    if parts[0].strip() == "":
        i = 1

    qa_started = False
    segment_idx = 0

    while i < len(parts):
        raw_speaker = parts[i]
        text = parts[i+1].strip() if i+1 < len(parts) else ""
        i += 2

        if not text:
            continue

        # Map raw speaker prefix to speaker role and name
        if raw_speaker == "Operator:":
            speaker_role = "operator"
            speaker_name = "Operator"
        elif raw_speaker == "Analysts:":
            speaker_role = "analyst"
            speaker_name = "Analyst"
        elif raw_speaker == "Executives:":
            speaker_role = "executive"
            speaker_name = "Executive"
        else:
            speaker_role = "other"
            speaker_name = raw_speaker.replace(":", "").strip()

        # Determine segment type using canonical classification
        segment_type, qa_started = classify_segment_type(speaker_role, text, qa_started)

        segments.append({
            "call_id": call_id,
            "segment_id": f"{call_id}_seg_{segment_idx:04d}",
            "speaker_role": speaker_role,
            "speaker_name": speaker_name,
            "segment_type": segment_type,
            "text": text,
            "start_time": None,
            "end_time": None,
            "audio_path": None,
        })
        segment_idx += 1

    return segments

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def download_and_process(years: list[int], output_dir: Path, raw_dir: Path, max_calls: int | None = None):
    """Download and process FinCall-Surprise dataset for the specified years."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_segments = []
    label_records = []
    calls_processed = 0

    for year in years:
        if max_calls is not None and calls_processed >= max_calls:
            break

        logger.info(f"Processing year {year}...")
        url = TRANSCRIPT_URL_TEMPLATE.format(year=year)
        raw_path = raw_dir / f"transcripts_{year}.json"

        # Download if not already cached
        if not raw_path.exists():
            logger.info(f"Downloading from {url}...")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            raw_path.write_text(r.text, encoding="utf-8")

        # Load transcripts
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} calls for {year}")

        for call_id, call_info in tqdm(data.items(), desc=f"Parsing {year}"):
            if max_calls is not None and calls_processed >= max_calls:
                logger.info(f"Reached max-calls limit of {max_calls}. Stopping.")
                break

            input_text = call_info.get("input", "")
            mp3_id = call_info.get("mp3_id")
            ppt_id = call_info.get("ppt_id")
            label = call_info.get("label")

            # Parse segments
            call_segments = process_call_transcript(call_id, input_text)
            if call_segments:
                all_segments.extend(call_segments)
                label_records.append({
                    "call_id": call_id,
                    "year": year,
                    "mp3_id": mp3_id,
                    "ppt_id": ppt_id,
                    "surprise_label": label,
                })
                calls_processed += 1

    if not all_segments:
        logger.error("No segments were extracted.")
        sys.exit(1)

    # 1. Save segments to Parquet
    df_segments = pl.DataFrame(all_segments)
    segments_output_path = output_dir / "fincall_segments.parquet"
    df_segments.write_parquet(segments_output_path)
    logger.info(f"Saved {len(df_segments)} segments to {segments_output_path}")

    # 2. Save labels/metadata to Parquet
    df_labels = pl.DataFrame(label_records)
    labels_output_path = output_dir / "fincall_labels.parquet"
    df_labels.write_parquet(labels_output_path)
    logger.info(f"Saved {len(df_labels)} labels to {labels_output_path}")

    # 3. Print Summary Stats
    logger.info("=" * 60)
    logger.info("FINCALL INTEGRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed calls: {df_labels['call_id'].n_unique()}")
    logger.info(f"Total segments:        {len(df_segments)}")
    
    # Class labels breakdown
    label_counts = df_labels.group_by("surprise_label").len().sort("surprise_label")
    logger.info("Surprise Labels distribution:")
    for row in label_counts.iter_rows():
        logger.info(f"  Label {row[0]}: {row[1]}")

    # Segment types breakdown
    type_counts = df_segments.group_by("segment_type").len().sort("len", descending=True)
    logger.info("Segment Types distribution:")
    for row in type_counts.iter_rows():
        logger.info(f"  {row[0]:<25} {row[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process FinCall-Surprise transcripts")
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2019, 2020, 2021],
        help="Years to process (default: 2019 2020 2021)"
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Maximum number of calls to process (default: no limit)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for processed parquets (default: data/processed)"
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw/fincall",
        help="Directory to cache raw JSONs (default: data/raw/fincall)"
    )
    args = parser.parse_args()

    download_and_process(
        years=args.years,
        output_dir=PROJECT_ROOT / args.output_dir,
        raw_dir=PROJECT_ROOT / args.raw_dir,
        max_calls=args.max_calls,
    )
