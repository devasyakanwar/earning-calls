"""
Script to extract all text features (sentiment, uncertainty, specificity) 
for a given segments file.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so 'src' can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))

import polars as pl

# Import the feature extractors
from src.features.text_sentiment import SentimentExtractor
from src.features.text_uncertainty import UncertaintyDetector
from src.features.text_specificity import SpecificityScorer
from src.features.text_assembler import assemble_text_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Extract all text features for a segments file")
    parser.add_argument("--input", type=str, required=True, help="Path to segments parquet")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for intermediate files")
    parser.add_argument("--final_output", type=str, required=True, help="Path to final text_features parquet")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "text_config.yaml"
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    df = pl.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} segments from {input_path}")

    # 1. Sentiment
    sent_path = Path(f"{args.output_prefix}_sentiment.parquet")
    if sent_path.exists():
        logger.info(f"Sentiment features already exist at {sent_path}, skipping...")
    else:
        logger.info("Extracting sentiment features...")
        sent_extractor = SentimentExtractor(config_path)
        df_sent = sent_extractor.extract(df)
        df_sent.write_parquet(sent_path)
    
    # 2. Uncertainty
    unc_path = Path(f"{args.output_prefix}_uncertainty.parquet")
    if unc_path.exists():
        logger.info(f"Uncertainty features already exist at {unc_path}, skipping...")
    else:
        logger.info("Extracting uncertainty features...")
        unc_extractor = UncertaintyDetector(config_path)
        df_unc = unc_extractor.extract(df)
        df_unc.write_parquet(unc_path)
    
    # 3. Specificity
    spec_path = Path(f"{args.output_prefix}_specificity.parquet")
    if spec_path.exists():
        logger.info(f"Specificity features already exist at {spec_path}, skipping...")
    else:
        logger.info("Extracting specificity features...")
        spec_extractor = SpecificityScorer(config_path)
        df_spec = spec_extractor.extract(df)
        df_spec.write_parquet(spec_path)
    
    # 4. Assemble
    logger.info("Assembling all text features...")
    assemble_text_features(
        sentiment_path=sent_path,
        uncertainty_path=unc_path,
        specificity_path=spec_path,
        output_path=Path(args.final_output)
    )
    
    logger.info(f"Successfully completed all text feature extraction. Final output: {args.final_output}")

if __name__ == "__main__":
    main()
