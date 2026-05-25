"""
Training Pipeline for Multimodal Fusion Network — Phase 3, Task 3.D.6

Handles:
    - Data loading and preparation from parquet splits
    - PyTorch Dataset / DataLoader construction
    - Training loop with early stopping
    - Learning rate scheduling (cosine annealing)
    - Evaluation and metric logging
    - Model checkpointing

Usage:
    python src/modeling/train_fusion.py --mode text_only
    python src/modeling/train_fusion.py --mode audio_only
    python src/modeling/train_fusion.py --mode multimodal
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.modeling.fusion_model import MultiTaskLoss, create_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EarningsCallDataset(Dataset):
    """
    PyTorch Dataset for earnings call features.

    Loads text and/or audio features along with regression/classification
    targets from a polars DataFrame.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        text_feature_cols: list[str],
        audio_feature_cols: list[str],
        vol_target: str = "realized_vol_5d",
        dir_target: str = "return_1d",
        mode: str = "multimodal",
    ):
        self.mode = mode

        # Extract features
        if text_feature_cols and mode in ("text_only", "multimodal"):
            self.text_features = df.select(text_feature_cols).to_numpy().astype(np.float32)
            # Replace NaN/Inf
            self.text_features = np.nan_to_num(self.text_features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self.text_features = None

        if audio_feature_cols and mode in ("audio_only", "multimodal"):
            self.audio_features = df.select(audio_feature_cols).to_numpy().astype(np.float32)
            self.audio_features = np.nan_to_num(self.audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self.audio_features = None

        # Targets
        self.vol_targets = df[vol_target].to_numpy().astype(np.float32)
        self.vol_targets = np.nan_to_num(self.vol_targets, nan=0.0)

        dir_values = df[dir_target].to_numpy().astype(np.float32)
        self.dir_targets = (dir_values > 0).astype(np.float32)

        self.n_samples = len(df)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        item = {
            "vol_target": torch.tensor(self.vol_targets[idx]),
            "dir_target": torch.tensor(self.dir_targets[idx]),
        }

        if self.text_features is not None:
            item["text_features"] = torch.tensor(self.text_features[idx])
        if self.audio_features is not None:
            item["audio_features"] = torch.tensor(self.audio_features[idx])

        return item


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    criterion: MultiTaskLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    mode: str,
) -> dict:
    """Train for one epoch, return average losses."""
    model.train()
    total_loss = 0.0
    total_vol_loss = 0.0
    total_dir_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Prepare inputs
        text_feat = batch.get("text_features", None)
        audio_feat = batch.get("audio_features", None)

        if text_feat is not None:
            text_feat = text_feat.to(device)
        if audio_feat is not None:
            audio_feat = audio_feat.to(device)

        vol_target = batch["vol_target"].to(device)
        dir_target = batch["dir_target"].to(device)

        # Forward
        outputs = model(text_features=text_feat, audio_features=audio_feat)
        losses = criterion(outputs["vol_pred"], vol_target, outputs["dir_pred"], dir_target)

        # Backward
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        total_vol_loss += losses["vol_loss"].item()
        total_dir_loss += losses["dir_loss"].item()
        n_batches += 1

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "vol_loss": total_vol_loss / max(n_batches, 1),
        "dir_loss": total_dir_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: MultiTaskLoss,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model, return losses and predictions."""
    model.eval()
    total_loss = 0.0
    total_vol_loss = 0.0
    total_dir_loss = 0.0
    n_batches = 0

    all_vol_preds = []
    all_vol_targets = []
    all_dir_preds = []
    all_dir_targets = []

    for batch in dataloader:
        text_feat = batch.get("text_features", None)
        audio_feat = batch.get("audio_features", None)

        if text_feat is not None:
            text_feat = text_feat.to(device)
        if audio_feat is not None:
            audio_feat = audio_feat.to(device)

        vol_target = batch["vol_target"].to(device)
        dir_target = batch["dir_target"].to(device)

        outputs = model(text_features=text_feat, audio_features=audio_feat)
        losses = criterion(outputs["vol_pred"], vol_target, outputs["dir_pred"], dir_target)

        total_loss += losses["total"].item()
        total_vol_loss += losses["vol_loss"].item()
        total_dir_loss += losses["dir_loss"].item()
        n_batches += 1

        all_vol_preds.append(outputs["vol_pred"].cpu().numpy())
        all_vol_targets.append(vol_target.cpu().numpy())
        all_dir_preds.append(torch.sigmoid(outputs["dir_pred"]).cpu().numpy())
        all_dir_targets.append(dir_target.cpu().numpy())

    vol_preds = np.concatenate(all_vol_preds)
    vol_targets = np.concatenate(all_vol_targets)
    dir_preds = np.concatenate(all_dir_preds)
    dir_targets = np.concatenate(all_dir_targets)

    # Compute metrics
    rmse = float(np.sqrt(np.mean((vol_targets - vol_preds) ** 2)))

    from scipy.stats import spearmanr
    if len(np.unique(vol_preds)) > 1:
        ic, _ = spearmanr(vol_targets, vol_preds)
        ic = float(ic) if np.isfinite(ic) else 0.0
    else:
        ic = 0.0

    dir_binary = (dir_preds > 0.5).astype(int)
    accuracy = float(np.mean(dir_binary == dir_targets))

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "vol_loss": total_vol_loss / max(n_batches, 1),
        "dir_loss": total_dir_loss / max(n_batches, 1),
        "rmse": rmse,
        "ic": ic,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_model(
    mode: str,
    project_root: Path,
    n_epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-3,
    embed_dim: int = 128,
    patience: int = 15,
) -> dict:
    """
    Full training pipeline for a given mode.

    Args:
        mode: 'text_only', 'audio_only', or 'multimodal'
        project_root: project root path
        n_epochs: max epochs
        batch_size: batch size
        lr: learning rate
        embed_dim: embedding dimension
        patience: early stopping patience

    Returns:
        dict of final metrics
    """
    processed = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    
    if mode == "audio_only":
        # For audio-only, use the audio dataset directly
        audio_path = processed / "audio_dataset.parquet"
        if not audio_path.exists():
            logger.error("audio_dataset.parquet not found.")
            return {}
        
        df = pl.read_parquet(audio_path)
        
        # Audio feature columns (everything except call_id)
        audio_feature_cols = [
            c for c in df.columns
            if c != "call_id" and df[c].dtype in (pl.Float32, pl.Float64)
        ]
        text_feature_cols = []
        
        # We need market targets. Try loading REAL market data first
        mkt_path = processed / "earnings22_market_data.parquet"
        if mkt_path.exists():
            mkt = pl.read_parquet(mkt_path).filter(pl.col("data_source") == "real")
            df = df.join(
                mkt.select(["call_id", "return_1d", "return_5d", "realized_vol_5d", "call_date"]),
                on="call_id", how="inner"
            )
            df = df.with_columns([
                pl.col("realized_vol_5d").fill_null(0.02),
                pl.col("return_1d").fill_null(0.0),
            ])
            # Fill missing dates
            if df["call_date"].null_count() > 0:
                import datetime
                base = datetime.date(2021, 1, 1)
                dates = [base + datetime.timedelta(days=30 * i) for i in range(len(df))]
                df = df.with_columns(pl.Series("call_date", dates))
            logger.info("Using REAL market targets for audio-only mode")
        else:
            # Fallback to synthetic
            import datetime
            np.random.seed(42)
            base = datetime.date(2021, 1, 1)
            df = df.with_columns([
                pl.Series("realized_vol_5d", np.random.uniform(0.01, 0.05, len(df)).astype(np.float64)),
                pl.Series("return_1d", np.random.normal(0.0, 0.02, len(df)).astype(np.float64)),
                pl.Series("call_date", [base + datetime.timedelta(days=30 * i) for i in range(len(df))]),
            ])
            logger.warning("Using SYNTHETIC targets (no earnings22_market_data.parquet found)")
        
        logger.info("Audio features: %d columns, %d calls", len(audio_feature_cols), len(df))
    
    elif mode == "multimodal":
        # For multimodal, use the ALIGNED multimodal dataset
        multi_path = processed / "multimodal_dataset.parquet"
        if not multi_path.exists():
            logger.error("multimodal_dataset.parquet not found. Run multimodal_join.py first.")
            return {}
        
        df = pl.read_parquet(multi_path)
        
        # Audio features have suffix _audio or are in the audio set
        # Text features are those from the text aggregation (sentiment, uncertainty, etc.)
        audio_feature_cols = [c for c in df.columns if "_audio" in c or "wav2vec2" in c or "prosody" in c or "opensmile" in c]
        
        # Capture all sentiment, uncertainty, etc. including their aggregated versions (_mean, _std)
        # Also include Phase 3 interaction features (divergence, pressure, qa_)
        text_keywords = [
            "sentiment", "uncertainty", "forward_looking", "hedging", 
            "specificity", "linguistic",
            # Phase 3: Interaction features
            "divergence", "pressure", "qa_", "response_length",
        ]

        text_feature_cols = [
            c for c in df.columns 
            if any(k in c for k in text_keywords) 
            and c not in audio_feature_cols
        ]
        
        logger.info("Multimodal features: %d audio, %d text", len(audio_feature_cols), len(text_feature_cols))
        
    else:
        # For text_only, use text_market_dataset
        tm_path = processed / "text_market_dataset.parquet"
        if not tm_path.exists():
            logger.error("text_market_dataset.parquet not found.")
            return {}
        
        df = pl.read_parquet(tm_path)
        
        # Identify text feature columns
        meta_cols = {
            "call_id", "ticker", "call_date", "close_t0", "close_t1", "close_t5",
            "return_1d", "return_5d", "realized_vol_1d", "realized_vol_5d",
            "earnings_surprise", "n_segments",
        }
        
        text_feature_cols = [
            c for c in df.columns
            if c not in meta_cols
            and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        ]
        audio_feature_cols = []

    logger.info("Text features: %d columns", len(text_feature_cols))

    # Filter valid rows
    df = df.filter(
        pl.col("realized_vol_5d").is_not_null()
        & pl.col("realized_vol_5d").is_finite()
        & pl.col("return_1d").is_not_null()
    )

    # Chronological split
    df = df.sort("call_date")
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.head(train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n - val_end)

    logger.info("Data split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # -----------------------------------------------------------------------
    # Create datasets
    # -----------------------------------------------------------------------
    active_text_cols = text_feature_cols if mode in ("text_only", "multimodal") else []
    active_audio_cols = audio_feature_cols if mode in ("audio_only", "multimodal") else []

    # Determine dimensions
    text_dim = len(active_text_cols) if active_text_cols else 1
    audio_dim = len(active_audio_cols) if active_audio_cols else 1

    train_dataset = EarningsCallDataset(train_df, active_text_cols, active_audio_cols, mode=mode)
    val_dataset = EarningsCallDataset(val_df, active_text_cols, active_audio_cols, mode=mode)
    test_dataset = EarningsCallDataset(test_df, active_text_cols, active_audio_cols, mode=mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # -----------------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------------
    model, criterion = create_model(
        text_dim=text_dim,
        audio_dim=audio_dim,
        mode=mode,
        embed_dim=embed_dim,
        device=device,
    )

    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    logger.info("Starting training: mode=%s, epochs=%d", mode, n_epochs)
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device, mode)
        val_metrics = evaluate(model, criterion, val_loader, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["total_loss"],
            "val_loss": val_metrics["total_loss"],
            "val_rmse": val_metrics["rmse"],
            "val_ic": val_metrics["ic"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d | Train Loss=%.4f | Val Loss=%.4f | RMSE=%.6f | IC=%.4f | Acc=%.4f",
                epoch,
                train_metrics["total_loss"],
                val_metrics["total_loss"],
                val_metrics["rmse"],
                val_metrics["ic"],
                val_metrics["accuracy"],
            )

        # Early stopping
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            no_improve = 0

            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, output_dir / f"best_model_{mode}.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1fs (best epoch=%d)", elapsed, best_epoch)

    # -----------------------------------------------------------------------
    # Load best model and evaluate on test set
    # -----------------------------------------------------------------------
    checkpoint = torch.load(output_dir / f"best_model_{mode}.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, criterion, test_loader, device)
    logger.info(
        "TEST Results [%s]: RMSE=%.6f | IC=%.4f | Acc=%.4f",
        mode, test_metrics["rmse"], test_metrics["ic"], test_metrics["accuracy"],
    )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "mode": mode,
        "device": device,
        "text_features": len(active_text_cols),
        "audio_features": len(active_audio_cols),
        "embed_dim": embed_dim,
        "best_epoch": best_epoch,
        "training_time_seconds": elapsed,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "test_rmse": test_metrics["rmse"],
        "test_ic": test_metrics["ic"],
        "test_accuracy": test_metrics["accuracy"],
        "test_vol_loss": test_metrics["vol_loss"],
        "test_dir_loss": test_metrics["dir_loss"],
        "history": history,
    }

    with open(output_dir / f"results_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to: %s", output_dir / f"results_{mode}.json")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    parser.add_argument(
        "--mode",
        type=str,
        default="text_only",
        choices=["text_only", "audio_only", "multimodal"],
        help="Training mode",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    train_model(
        mode=args.mode,
        project_root=project_root,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
