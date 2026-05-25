"""
Multimodal Fusion Network — Phase 3, Task 3.D.5

Architecture:
    Text Branch  → Linear → LayerNorm → ReLU → [text_embed]
    Audio Branch → Linear → LayerNorm → ReLU → [audio_embed]
                           ↓
    Cross-Attention: text attends to audio, audio attends to text
                           ↓
    Fusion MLP → Prediction Heads (volatility + direction)

Supports three training modes:
    - text_only:  uses only text branch
    - audio_only: uses only audio branch
    - multimodal: fuses both branches via cross-attention
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-Attention Module
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention where queries come from one modality
    and keys/values come from another.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:     (batch, 1, d_model)  — the modality doing the attending
            key_value: (batch, 1, d_model)  — the modality being attended to

        Returns:
            (batch, 1, d_model) — attended representation
        """
        attended, _ = self.attention(query, key_value, key_value)
        out = self.norm(query + self.dropout(attended))
        return out


# ---------------------------------------------------------------------------
# Modality Encoder
# ---------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """
    Encodes raw features from one modality into a fixed-size embedding.

    Architecture:
        Input → Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm → GELU
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        hidden_dim = max(embed_dim * 2, 128)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            (batch, embed_dim)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """
    Small MLP that maps a fused embedding to a single output.
    Used for both regression (volatility) and classification (direction).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Main Fusion Model
# ---------------------------------------------------------------------------

class MultimodalFusionNetwork(nn.Module):
    """
    End-to-end multimodal earnings call model.

    Supports three modes:
        - text_only:  only text branch is used
        - audio_only: only audio branch is used
        - multimodal: cross-attention fusion of both branches

    Produces two outputs:
        - vol_pred:   predicted 5-day realized volatility (regression)
        - dir_pred:   predicted return direction logit (classification)
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        mode: str = "multimodal",
    ):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim

        # Modality encoders
        self.text_encoder = ModalityEncoder(text_input_dim, embed_dim, dropout)
        self.audio_encoder = ModalityEncoder(audio_input_dim, embed_dim, dropout)

        # Cross-attention layers
        self.text_attends_audio = CrossAttention(embed_dim, n_heads, dropout)
        self.audio_attends_text = CrossAttention(embed_dim, n_heads, dropout)

        # Fusion dimension depends on mode
        if mode == "multimodal":
            fusion_dim = embed_dim * 2  # concatenation of both attended
        elif mode == "text_only":
            fusion_dim = embed_dim
        elif mode == "audio_only":
            fusion_dim = embed_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific prediction heads
        self.vol_head = PredictionHead(embed_dim, hidden_dim=64, dropout=dropout)
        self.dir_head = PredictionHead(embed_dim, hidden_dim=64, dropout=dropout)

    def forward(
        self,
        text_features: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            text_features:  (batch, text_input_dim) or None
            audio_features: (batch, audio_input_dim) or None

        Returns:
            dict with keys:
                vol_pred: (batch,) volatility predictions
                dir_pred: (batch,) direction logits
        """
        if self.mode == "text_only":
            assert text_features is not None
            text_embed = self.text_encoder(text_features)
            fused = text_embed

        elif self.mode == "audio_only":
            assert audio_features is not None
            audio_embed = self.audio_encoder(audio_features)
            fused = audio_embed

        elif self.mode == "multimodal":
            assert text_features is not None and audio_features is not None
            text_embed = self.text_encoder(text_features)    # (B, embed_dim)
            audio_embed = self.audio_encoder(audio_features)  # (B, embed_dim)

            # Reshape for attention: (B, 1, embed_dim)
            text_seq = text_embed.unsqueeze(1)
            audio_seq = audio_embed.unsqueeze(1)

            # Cross-attention
            text_attended = self.text_attends_audio(text_seq, audio_seq).squeeze(1)
            audio_attended = self.audio_attends_text(audio_seq, text_seq).squeeze(1)

            # Concatenate attended representations
            fused = torch.cat([text_attended, audio_attended], dim=-1)

        # Fusion MLP
        fused = self.fusion_mlp(fused)

        # Prediction heads
        vol_pred = self.vol_head(fused)
        dir_pred = self.dir_head(fused)

        return {
            "vol_pred": vol_pred,
            "dir_pred": dir_pred,
        }


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining:
        - MSE loss for volatility regression
        - BCE with logits for direction classification

    Uses learnable task weights (uncertainty weighting).
    """

    def __init__(self):
        super().__init__()
        # Learnable log-variance parameters for task weighting
        self.log_var_vol = nn.Parameter(torch.zeros(1))
        self.log_var_dir = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vol_pred: torch.Tensor,
        vol_target: torch.Tensor,
        dir_pred: torch.Tensor,
        dir_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with 'total', 'vol_loss', 'dir_loss' tensors.
        """
        # Volatility regression loss (MSE)
        vol_loss = F.mse_loss(vol_pred, vol_target)

        # Direction classification loss (BCE)
        dir_loss = F.binary_cross_entropy_with_logits(dir_pred, dir_target.float())

        # Uncertainty weighting (Kendall et al., 2018)
        precision_vol = torch.exp(-self.log_var_vol)
        precision_dir = torch.exp(-self.log_var_dir)

        total = (
            precision_vol * vol_loss + self.log_var_vol
            + precision_dir * dir_loss + self.log_var_dir
        )

        return {
            "total": total,
            "vol_loss": vol_loss.detach(),
            "dir_loss": dir_loss.detach(),
            "vol_weight": precision_vol.detach(),
            "dir_weight": precision_dir.detach(),
        }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    text_dim: int,
    audio_dim: int,
    mode: str = "multimodal",
    embed_dim: int = 128,
    device: str = "cpu",
) -> tuple[MultimodalFusionNetwork, MultiTaskLoss]:
    """Create model and loss function."""
    model = MultimodalFusionNetwork(
        text_input_dim=text_dim,
        audio_input_dim=audio_dim,
        embed_dim=embed_dim,
        mode=mode,
    ).to(device)

    criterion = MultiTaskLoss().to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in criterion.parameters() if p.requires_grad)
    logger.info(
        "Created %s model: %d trainable parameters, embed_dim=%d",
        mode, n_params, embed_dim,
    )

    return model, criterion
