#!/usr/bin/env python3
"""
Full Pipeline Orchestrator — Run Everything End-to-End

Steps:
    1. Download transcripts (200+ calls, multi-sector from HuggingFace)
    2. Extract text features (FinBERT sentiment, uncertainty, specificity)
    3. Download market data (yfinance)
    4. Build structural features
    5. Join into datasets (text_market + multimodal)
    6. Train models (LightGBM)
    7. Run baseline comparison (with CV)
    8. Walk-forward backtest
    9. Generate visualizations

Usage:
    python scripts/run_full_pipeline.py                    # Full pipeline
    python scripts/run_full_pipeline.py --skip-download    # Skip transcript download
    python scripts/run_full_pipeline.py --max-calls 50     # Quick test run
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_step(name: str, cmd: list[str], cwd: Path) -> bool:
    """Run a pipeline step, return True on success."""
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("CMD:  %s", " ".join(cmd))
    logger.info("=" * 60)
    
    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        logger.error("FAILED: %s (%.1fs, exit code %d)", name, elapsed, result.returncode)
        return False
    
    logger.info("DONE: %s (%.1fs)", name, elapsed)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full earnings call pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip transcript download")
    parser.add_argument("--skip-market", action="store_true", help="Skip market data download")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--max-calls", type=int, default=None, help="Max calls to download")
    parser.add_argument("--sectors", nargs="*", default=None, help="Sectors to download")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    python = sys.executable  # Use the same Python interpreter

    total_start = time.time()
    steps_run = 0
    steps_failed = 0

    # -----------------------------------------------------------------------
    # Step 1: Download transcripts
    # -----------------------------------------------------------------------
    if not args.skip_download:
        cmd = [python, "scripts/download_transcripts.py"]
        if args.max_calls:
            cmd += ["--max-calls", str(args.max_calls)]
        if args.sectors:
            cmd += ["--sectors"] + args.sectors
        
        if not run_step("Download Transcripts", cmd, project_root):
            steps_failed += 1
            logger.warning("Continuing despite download failure...")
        steps_run += 1
    else:
        logger.info("Skipping transcript download (--skip-download)")

    # -----------------------------------------------------------------------
    # Step 2: Extract text features
    # -----------------------------------------------------------------------
    if not args.skip_features:
        processed = project_root / "data" / "processed"
        segments_path = processed / "segments.parquet"
        
        if segments_path.exists():
            cmd = [
                python, "scripts/extract_all_text_features.py",
                "--input", str(segments_path),
                "--output_prefix", str(processed / "sp500"),
                "--final_output", str(processed / "text_features.parquet"),
            ]
            if not run_step("Extract Text Features", cmd, project_root):
                steps_failed += 1
            steps_run += 1
        else:
            logger.warning("segments.parquet not found, skipping feature extraction")

    # -----------------------------------------------------------------------
    # Step 3: Build structural features
    # -----------------------------------------------------------------------
    if not args.skip_features:
        segments_path = project_root / "data" / "processed" / "segments.parquet"
        if segments_path.exists():
            cmd = [python, "-c", """
import polars as pl
from pathlib import Path
from src.features.structural_features import compute_structural_features

project_root = Path('.').resolve()
processed = project_root / 'data' / 'processed'
seg = pl.read_parquet(processed / 'segments.parquet')
sf = compute_structural_features(seg)
sf.write_parquet(processed / 'structural_features.parquet')
print(f'Structural features: {len(sf)} calls')
"""]
            if not run_step("Build Structural Features", cmd, project_root):
                steps_failed += 1
            steps_run += 1

    # -----------------------------------------------------------------------
    # Step 4: Download market data
    # -----------------------------------------------------------------------
    if not args.skip_market:
        cmd = [python, "scripts/download_market_data.py"]
        if not run_step("Download Market Data", cmd, project_root):
            steps_failed += 1
        steps_run += 1
    else:
        logger.info("Skipping market data download (--skip-market)")

    # -----------------------------------------------------------------------
    # Step 5: Join datasets
    # -----------------------------------------------------------------------
    cmd = [python, "-m", "src.features.multimodal_join"]
    if not run_step("Join Datasets", cmd, project_root):
        steps_failed += 1
    steps_run += 1

    # -----------------------------------------------------------------------
    # Step 6: Train LightGBM
    # -----------------------------------------------------------------------
    cmd = [python, "-m", "src.modeling.train_lightgbm"]
    if not run_step("Train LightGBM", cmd, project_root):
        steps_failed += 1
    steps_run += 1

    # -----------------------------------------------------------------------
    # Step 7: Baseline comparison
    # -----------------------------------------------------------------------
    cmd = [python, "-m", "src.modeling.baseline_comparison"]
    if not run_step("Baseline Comparison", cmd, project_root):
        steps_failed += 1
    steps_run += 1

    # -----------------------------------------------------------------------
    # Step 8: Walk-forward backtest
    # -----------------------------------------------------------------------
    cmd = [python, "-m", "src.evaluation.backtesting"]
    if not run_step("Walk-Forward Backtest", cmd, project_root):
        steps_failed += 1
    steps_run += 1

    # -----------------------------------------------------------------------
    # Step 9: Generate visualizations
    # -----------------------------------------------------------------------
    cmd = [python, "scripts/visualize_results.py"]
    if not run_step("Generate Visualizations", cmd, project_root):
        steps_failed += 1
    steps_run += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Steps run:    %d", steps_run)
    logger.info("Steps failed: %d", steps_failed)
    logger.info("Total time:   %.1f seconds (%.1f minutes)", total_elapsed, total_elapsed / 60)
    logger.info("")
    logger.info("Outputs:")
    logger.info("  data/processed/text_market_dataset.parquet")
    logger.info("  outputs/baseline_comparison.json")
    logger.info("  outputs/evaluation/backtest_results.json")
    logger.info("  outputs/evaluation/model_predictions.parquet")
    logger.info("  outputs/plots/*.png")
    logger.info("")
    logger.info("To launch dashboard:")
    logger.info("  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
