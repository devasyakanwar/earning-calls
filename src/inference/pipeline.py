"""
Phase 5: Inference Pipeline

Takes a raw audio call (segments) and its transcript, extracts all features,
and produces a multimodal trading signal.

Usage:
    python src/inference/pipeline.py --call_id <id>
"""

import logging
import json
from pathlib import Path

import torch
import polars as pl
import numpy as np

from src.modeling.fusion_model import create_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class MultimodalInference:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.processed = project_root / "data" / "processed"
        self.model_path = project_root / "outputs" / "models" / "best_model_multimodal.pt"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained multimodal model."""
        if not self.model_path.exists():
            logger.error("Trained model not found at %s", self.model_path)
            return

        # We need to know the input dimensions
        # For simplicity in this script, we'll look them up from the latest dataset
        df = pl.read_parquet(self.processed / "multimodal_dataset.parquet")
        
        audio_cols = [c for c in df.columns if "_audio" in c or "wav2vec2" in c or "prosody" in c or "opensmile" in c]
        text_keywords = ["sentiment", "uncertainty", "forward_looking", "hedging", "specificity", "linguistic", "divergence", "pressure", "qa_", "response_length"]
        text_cols = [c for c in df.columns if any(k in c for k in text_keywords) and c not in audio_cols]
        
        logger.info("Initializing model with %d audio, %d text features", len(audio_cols), len(text_cols))
        
        self.model, _ = create_model(
            text_dim=len(text_cols),
            audio_dim=len(audio_cols),
            mode="multimodal",
            device=self.device
        )
        
        # Load the checkpoint dictionary
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract the model's state dict from the checkpoint
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        self.audio_cols = audio_cols
        self.text_cols = text_cols

    def predict_call(self, call_id: str):
        """Run end-to-end inference for a specific call ID in the processed dataset."""
        if self.model is None:
            return None
            
        # 1. Fetch features for this call
        df = pl.read_parquet(self.processed / "multimodal_dataset.parquet")
        call_data = df.filter(pl.col("call_id") == call_id)
        
        if len(call_data) == 0:
            logger.error("Call ID %s not found in multimodal dataset", call_id)
            return None
            
        # 2. Prepare tensors
        X_audio = torch.tensor(call_data.select(self.audio_cols).to_numpy(), dtype=torch.float32).to(self.device)
        X_text = torch.tensor(call_data.select(self.text_cols).to_numpy(), dtype=torch.float32).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            outputs = self.model(text_features=X_text, audio_features=X_audio)

            
        vol_pred = outputs["vol_pred"].item()
        dir_logit = outputs["dir_pred"].item()
        dir_prob = torch.sigmoid(outputs["dir_pred"]).item()
        
        # 4. Generate Signal
        signal = "HOLD"
        confidence = abs(dir_prob - 0.5) * 2 # 0 to 1
        
        if dir_prob > 0.6:
            signal = "BUY"
        elif dir_prob < 0.4:
            signal = "SELL"
            
        result = {
            "call_id": call_id,
            "ticker": call_data["ticker"][0] if "ticker" in call_data.columns else "UNKNOWN",
            "prediction": {
                "signal": signal,
                "confidence": confidence,
                "direction_prob": dir_prob,
                "volatility_score": vol_pred
            },
            "pressure_metrics": {
                "pressure_score": float(call_data["pressure_score"][0]) if "pressure_score" in call_data.columns else 0.0,
                "sentiment_divergence": float(call_data["composite_divergence_score_mean"][0]) if "composite_divergence_score_mean" in call_data.columns else 0.0
            }
        }
        
        return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    project_root = Path(__file__).resolve().parent.parent.parent
    inference = MultimodalInference(project_root)
    
    # Try a known call
    call_id = "4483296" # Imperial Oil
    result = inference.predict_call(call_id)
    
    if result:
        print("\n" + "="*40)
        print(f"TRADING SIGNAL: {result['prediction']['signal']}")
        print(f"Confidence:     {result['prediction']['confidence']:.2%}")
        print(f"Target:         {result['ticker']}")
        print(f"Pressure Score: {result['pressure_metrics']['pressure_score']:.4f}")
        print("="*40 + "\n")
        
        # Save signal
        output_dir = project_root / "outputs" / "signals"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"signal_{call_id}.json", "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()
