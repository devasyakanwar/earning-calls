#!/usr/bin/env python3
"""
Download and process S&P 500 earnings call transcripts from HuggingFace.

Downloads the Bose345/sp500_earnings_transcripts dataset (or a filtered subset)
and converts it to the project's segments.parquet format (Contract A).

Usage:
    python scripts/download_transcripts.py                          # Full multi-sector (200+ calls)
    python scripts/download_transcripts.py --years 2023 2024        # Filter by year
    python scripts/download_transcripts.py --sectors Technology Healthcare
    python scripts/download_transcripts.py --tickers AAPL MSFT GOOGL # Filter by ticker
    python scripts/download_transcripts.py --max-calls 200          # Limit total calls
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from datasets import load_dataset
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

DATASET_ID = "Bose345/sp500_earnings_transcripts"

# ---------------------------------------------------------------------------
# Multi-sector ticker universe for credible cross-sector coverage
# ---------------------------------------------------------------------------

SECTOR_TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "CSCO", "ACN",
        "ADBE", "IBM", "INTC", "INTU", "TXN", "QCOM", "AMAT", "NOW", "PANW",
        "LRCX", "ADI", "SNPS", "KLAC", "CDNS", "MCHP", "FTNT",
        # Mega-cap often in other GICS sectors but considered tech
        "GOOGL", "GOOG", "META", "AMZN", "TSLA", "NFLX",
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "ABT", "TMO", "MRK", "LLY", "ABBV", "DHR",
        "BMY", "AMGN", "MDT", "ISRG", "GILD", "CVS", "CI", "SYK", "BSX",
        "VRTX", "REGN", "ZTS", "BDX", "EW", "HCA", "IDXX",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP",
        "USB", "PNC", "TFC", "CB", "MMC", "AON", "ICE", "CME", "MCO",
        "SPGI", "COF", "MET", "AIG", "PRU", "ALL", "TRV",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD",
        "OXY", "WMB", "HES", "DVN", "KMI", "HAL", "FANG", "BKR",
        "TRGP", "OKE", "CTRA",
    ],
    "Consumer": [
        "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX",
        "TGT", "LOW", "CL", "EL", "GIS", "KHC", "MDLZ", "SJM", "HSY",
        "DG", "DLTR", "TJX", "ROST", "YUM", "DPZ", "CMG",
    ],
}

# Flat list of ALL tickers across all sectors
ALL_TICKERS = []
for sector_tickers in SECTOR_TICKERS.values():
    ALL_TICKERS.extend(sector_tickers)
ALL_TICKERS = list(set(ALL_TICKERS))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transcript processing
# ---------------------------------------------------------------------------


def process_transcript(record: dict, min_text_length: int = 200) -> list[dict]:
    """Convert one HuggingFace dataset record to a list of segment dicts matching Contract A."""
    symbol = record.get("symbol", "UNK")
    year = record.get("year", 0)
    quarter = record.get("quarter", 0)
    call_id = f"{symbol}_{year}Q{quarter}"

    structured = record.get("structured_content")
    if not structured:
        return []

    segments = []
    qa_started = False
    total = len(structured)

    for i, turn in enumerate(structured):
        speaker = turn.get("speaker", "") or ""
        text = turn.get("text", "") or ""

        # Skip empty segments
        if not text.strip():
            continue

        speaker_role = classify_speaker_role(speaker, i, call_id=call_id)
        segment_type, qa_started = classify_segment_type(speaker_role, text, qa_started)

        segments.append({
            "call_id": call_id,
            "segment_id": f"{call_id}_seg_{i:04d}",
            "speaker_role": speaker_role,
            "speaker_name": speaker.strip(),
            "segment_type": segment_type,
            "text": text.strip(),
            "start_time": None,  # No audio timestamps in text-only MVP
            "end_time": None,
            "audio_path": None,
        })

    # Skip calls with too little text content
    total_chars = sum(len(s["text"]) for s in segments)
    if total_chars < min_text_length:
        return []

    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download S&P 500 earnings call transcripts and convert to segments.parquet",
    )
    parser.add_argument(
        "--years", nargs="*", type=int, default=None,
        help="Filter to specific years (e.g., --years 2023 2024). Default: 2020-2024.",
    )
    parser.add_argument(
        "--tickers", nargs="*", type=str, default=None,
        help="Filter to specific tickers (e.g., --tickers AAPL MSFT). Default: all.",
    )
    parser.add_argument(
        "--sectors", nargs="*", type=str, default=None,
        choices=list(SECTOR_TICKERS.keys()),
        help="Filter to specific sectors (e.g., --sectors Technology Healthcare).",
    )
    parser.add_argument(
        "--tech-only", action="store_true",
        help="Filter to Technology sector tickers only.",
    )
    parser.add_argument(
        "--all-sectors", action="store_true", default=True,
        help="Use all sector tickers (default behavior).",
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Maximum number of calls to process. Default: no limit.",
    )
    parser.add_argument(
        "--min-text-length", type=int, default=500,
        help="Minimum total character count for a call to be included. Default: 500.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for segments.parquet. Default: data/processed",
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Directory to cache the raw HuggingFace dataset. Default: data/raw",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    raw_dir = project_root / args.raw_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Download dataset
    # -----------------------------------------------------------------------
    logger.info("Downloading dataset: %s", DATASET_ID)
    logger.info("This may take a few minutes on first run (~1.8 GB)...")

    ds = load_dataset(DATASET_ID, cache_dir=str(raw_dir / "hf_cache"))
    train = ds["train"]
    logger.info("Dataset loaded: %d total records", len(train))

    # -----------------------------------------------------------------------
    # Step 2: Determine ticker universe
    # -----------------------------------------------------------------------
    tickers_filter = None
    selected_sectors = []

    if args.tickers:
        tickers_filter = set(t.upper() for t in args.tickers)
        logger.info("Filtering to %d specified tickers: %s", len(tickers_filter), tickers_filter)
    elif args.tech_only:
        tickers_filter = set(SECTOR_TICKERS["Technology"])
        selected_sectors = ["Technology"]
        logger.info("Filtering to %d Tech sector tickers", len(tickers_filter))
    elif args.sectors:
        tickers_filter = set()
        for sec in args.sectors:
            tickers_filter.update(SECTOR_TICKERS[sec])
            selected_sectors.append(sec)
        logger.info("Filtering to %d tickers across sectors: %s", len(tickers_filter), args.sectors)
    else:
        # Default: ALL sectors for maximum credibility
        tickers_filter = set(ALL_TICKERS)
        selected_sectors = list(SECTOR_TICKERS.keys())
        logger.info("Using ALL %d tickers across %d sectors", len(tickers_filter), len(SECTOR_TICKERS))

    # Default year range: 2020-2024
    years_filter = set(args.years) if args.years else {2020, 2021, 2022, 2023, 2024}
    logger.info("Filtering to years: %s", sorted(years_filter))

    # Filter the dataset
    def should_include(record):
        if tickers_filter and record.get("symbol") not in tickers_filter:
            return False
        if years_filter and record.get("year") not in years_filter:
            return False
        return True

    filtered = train.filter(should_include, desc="Filtering transcripts")
    logger.info("After filtering: %d records", len(filtered))

    if args.max_calls and len(filtered) > args.max_calls:
        filtered = filtered.select(range(args.max_calls))
        logger.info("Capped at %d calls", args.max_calls)

    # -----------------------------------------------------------------------
    # Step 3: Process transcripts into segments
    # -----------------------------------------------------------------------
    all_segments = []
    skipped = 0
    skipped_short = 0

    for record in tqdm(filtered, desc="Processing transcripts"):
        segments = process_transcript(record, min_text_length=args.min_text_length)
        if segments:
            all_segments.extend(segments)
        else:
            if record.get("structured_content"):
                skipped_short += 1
            else:
                skipped += 1

    logger.info(
        "Processed %d segments from %d calls (%d skipped no content, %d skipped too short)",
        len(all_segments), len(filtered) - skipped - skipped_short, skipped, skipped_short,
    )

    if not all_segments:
        logger.error("No segments produced. Check filters and dataset.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 4: Save as Parquet
    # -----------------------------------------------------------------------
    df = pl.DataFrame(all_segments)

    # Validate schema matches Contract A
    expected_columns = [
        "call_id", "segment_id", "speaker_role", "speaker_name",
        "segment_type", "text", "start_time", "end_time", "audio_path",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

    output_path = output_dir / "segments.parquet"
    df.write_parquet(output_path)
    logger.info("Saved segments to: %s", output_path)

    # -----------------------------------------------------------------------
    # Step 5: Summary statistics
    # -----------------------------------------------------------------------
    n_calls = df["call_id"].n_unique()
    n_segments = len(df)
    n_tickers = df["call_id"].str.extract(r"^([A-Z]+)_", 1).n_unique()

    segment_type_counts = df.group_by("segment_type").len().sort("len", descending=True)
    speaker_role_counts = df.group_by("speaker_role").len().sort("len", descending=True)

    # Sector breakdown
    tickers_in_data = set(df["call_id"].str.extract(r"^([A-Z]+)_", 1).unique().to_list())
    sector_coverage = {}
    for sec, ticks in SECTOR_TICKERS.items():
        overlap = tickers_in_data & set(ticks)
        if overlap:
            sector_coverage[sec] = len(overlap)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total calls:    %d", n_calls)
    logger.info("Total segments: %d", n_segments)
    logger.info("Unique tickers: %d", n_tickers)
    logger.info("")
    logger.info("Sector coverage:")
    for sec, count in sector_coverage.items():
        logger.info("  %-20s %d tickers", sec, count)
    logger.info("")
    logger.info("Segment types:")
    for row in segment_type_counts.iter_rows():
        logger.info("  %-25s %d", row[0], row[1])
    logger.info("")
    logger.info("Speaker roles:")
    for row in speaker_role_counts.iter_rows():
        logger.info("  %-25s %d", row[0], row[1])

    # Save a manifest CSV for downstream use
    manifest = (
        df.group_by("call_id")
        .agg([
            pl.col("segment_id").count().alias("n_segments"),
            pl.col("text").str.len_chars().sum().alias("total_chars"),
        ])
        .sort("call_id")
    )
    manifest_path = output_dir / "call_manifest.csv"
    manifest.write_csv(manifest_path)
    logger.info("Saved call manifest to: %s", manifest_path)


if __name__ == "__main__":
    main()
