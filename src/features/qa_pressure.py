"""
Q&A Pressure Features — Phase 3, Task 3.2

Captures the dynamics of how executives respond under analyst questioning.
These features model the "interaction pressure" during Q&A sessions.

Features computed per call:
    - qa_sentiment_drop: sentiment change from prepared remarks to Q&A
    - qa_uncertainty_spike: uncertainty increase during Q&A
    - qa_hedging_ratio: hedging frequency in Q&A vs prepared remarks
    - avg_response_length: average executive response length (words)
    - response_length_variance: consistency of response lengths
    - qa_divergence_spike: does divergence increase under questioning?
    - pressure_score: composite pressure metric

The hypothesis: executives under pressure give shorter, more hedged,
more uncertain responses — and their voice cracks more.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Q&A Pressure Extractor
# ---------------------------------------------------------------------------

class QAPressureExtractor:
    """
    Extracts features that capture how executive behavior changes
    between prepared remarks and Q&A sessions.
    """
    
    def extract(
        self,
        segments: pl.DataFrame,
        text_features: pl.DataFrame,
        divergence_features: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """
        Compute call-level Q&A pressure features.
        
        Args:
            segments: segment metadata with call_id, text, segment_type
            text_features: segment-level text features
            divergence_features: optional divergence features
            
        Returns:
            DataFrame with call_id + pressure features
        """
        # Join text features to segments
        df = segments.join(text_features, on="segment_id", how="inner")
        
        # Optionally add divergence features
        if divergence_features is not None and len(divergence_features) > 0:
            df = df.join(divergence_features, on="segment_id", how="left")
        
        # Estimate segment type from position in call
        # First 40% of segments ≈ prepared remarks, rest ≈ Q&A
        # (Since segment_type is "unknown" for Earnings-22)
        calls = df["call_id"].unique().to_list()
        results = []
        
        logger.info("Computing Q&A pressure features for %d calls...", len(calls))
        
        for call_id in calls:
            call_df = df.filter(pl.col("call_id") == call_id).sort("segment_id")
            n = len(call_df)
            
            if n < 4:
                # Not enough segments to split
                results.append(self._empty_result(call_id))
                continue
            
            # Split into "prepared" vs "qa" using actual segment_type when available
            if "segment_type" in call_df.columns:
                known_types = call_df.filter(
                    pl.col("segment_type").is_in(["prepared_remarks", "analyst_question", "management_answer"])
                )
                if len(known_types) > n * 0.3:
                    # segment_type labels are meaningful — use them
                    prepared = call_df.filter(pl.col("segment_type") == "prepared_remarks")
                    qa = call_df.filter(
                        pl.col("segment_type").is_in(["analyst_question", "management_answer"])
                    )
                else:
                    # Fallback: segment_type is mostly 'unknown' or unlabelled
                    split_idx = int(n * 0.4)
                    prepared = call_df.head(split_idx)
                    qa = call_df.tail(n - split_idx)
            else:
                # No segment_type column at all — use positional heuristic
                split_idx = int(n * 0.4)
                prepared = call_df.head(split_idx)
                qa = call_df.tail(n - split_idx)
            
            if len(prepared) == 0 or len(qa) == 0:
                results.append(self._empty_result(call_id))
                continue
            
            # --- Feature Computation ---
            
            # 1. Sentiment drop: prepared vs Q&A
            prep_sent = prepared["sentiment_score"].mean() if "sentiment_score" in prepared.columns else 0.0
            qa_sent = qa["sentiment_score"].mean() if "sentiment_score" in qa.columns else 0.0
            sentiment_drop = float(prep_sent - qa_sent) if prep_sent is not None and qa_sent is not None else 0.0
            
            # 2. Uncertainty spike: Q&A vs prepared
            prep_unc = prepared["uncertainty_score"].mean() if "uncertainty_score" in prepared.columns else 0.0
            qa_unc = qa["uncertainty_score"].mean() if "uncertainty_score" in qa.columns else 0.0
            uncertainty_spike = float(qa_unc - prep_unc) if qa_unc is not None and prep_unc is not None else 0.0
            
            # 3. Hedging ratio: Q&A hedging / prepared hedging
            prep_hedge = prepared["hedging_frequency"].mean() if "hedging_frequency" in prepared.columns else 0.0
            qa_hedge = qa["hedging_frequency"].mean() if "hedging_frequency" in qa.columns else 0.0
            if prep_hedge is not None and prep_hedge > 0.001:
                hedging_ratio = float(qa_hedge / prep_hedge) if qa_hedge is not None else 1.0
            else:
                hedging_ratio = 1.0
            
            # 4. Response length analysis
            qa_texts = qa["text"].to_list()
            qa_lengths = [len(t.split()) for t in qa_texts if t]
            avg_response_len = float(np.mean(qa_lengths)) if qa_lengths else 0.0
            response_len_var = float(np.std(qa_lengths)) if len(qa_lengths) > 1 else 0.0
            
            # 5. Divergence spike (if available)
            divergence_spike = 0.0
            if "composite_divergence_score" in call_df.columns:
                prep_div = prepared["composite_divergence_score"].mean()
                qa_div = qa["composite_divergence_score"].mean()
                if prep_div is not None and qa_div is not None:
                    divergence_spike = float(qa_div - prep_div)
            
            # 6. Specificity drop: do executives get vague under pressure?
            prep_spec = prepared["specificity_score"].mean() if "specificity_score" in prepared.columns else 0.0
            qa_spec = qa["specificity_score"].mean() if "specificity_score" in qa.columns else 0.0
            specificity_drop = float(prep_spec - qa_spec) if prep_spec is not None and qa_spec is not None else 0.0
            
            # 7. Composite Pressure Score
            pressure_score = (
                0.25 * max(sentiment_drop, 0)         # Positive = sentiment fell
                + 0.25 * max(uncertainty_spike, 0)     # Positive = uncertainty rose
                + 0.15 * max(hedging_ratio - 1.0, 0)   # > 1 = more hedging in Q&A
                + 0.15 * max(divergence_spike, 0)       # Positive = more divergence in Q&A
                + 0.10 * max(specificity_drop, 0)       # Positive = got more vague
                + 0.10 * (1.0 / max(avg_response_len, 1.0)) * 100  # Shorter = more pressure
            )
            
            results.append({
                "call_id": call_id,
                "qa_sentiment_drop": sentiment_drop,
                "qa_uncertainty_spike": uncertainty_spike,
                "qa_hedging_ratio": hedging_ratio,
                "qa_specificity_drop": specificity_drop,
                "avg_response_length": avg_response_len,
                "response_length_variance": response_len_var,
                "qa_divergence_spike": divergence_spike,
                "pressure_score": pressure_score,
            })
        
        result_df = pl.DataFrame(results)
        logger.info("Q&A pressure: mean pressure=%.4f, max=%.4f",
                     result_df["pressure_score"].mean(), result_df["pressure_score"].max())
        return result_df
    
    def _empty_result(self, call_id: str) -> dict:
        return {
            "call_id": call_id,
            "qa_sentiment_drop": 0.0,
            "qa_uncertainty_spike": 0.0,
            "qa_hedging_ratio": 1.0,
            "qa_specificity_drop": 0.0,
            "avg_response_length": 0.0,
            "response_length_variance": 0.0,
            "qa_divergence_spike": 0.0,
            "pressure_score": 0.0,
        }


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
    processed = project_root / "data" / "processed"
    
    segments = pl.read_parquet(processed / "earnings22_segments.parquet")
    text_features = pl.read_parquet(processed / "earnings22_text_features.parquet")
    
    # Load divergence if available
    div_path = processed / "divergence_features.parquet"
    div_features = pl.read_parquet(div_path) if div_path.exists() else None
    
    extractor = QAPressureExtractor()
    result = extractor.extract(segments, text_features, div_features)
    
    output_path = processed / "qa_pressure_features.parquet"
    result.write_parquet(output_path)
    logger.info("Saved %d Q&A pressure features to %s", len(result), output_path)


if __name__ == "__main__":
    main()
