"""MEG model A: spatial-temporal attention + BiGRU.

Ported and improved from the original ``recon/model/MEG_model_A.py``.

Architecture:
    Input (B, n_context, n_channels)  — multi-time-step MEG
        ↓
    ResidualProject (n_channels → n_channels + LayerNorm)
        ↓
    SpatialTemporalAttentionEncoder
        Channel projection → spatial attention → temporal attention
        (each with residual + LayerNorm)
        ↓
    BiGRU (2 layers, hidden=256, bidirectional)
        ↓
    Decoder (residual MLP)
        connect_fc → res_fc × 2 → output_fc
        ↓
    Output (B, semantic_dim=768)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..registry import register_model

logger = logging.getLogger(__name__)


# ───────────────────── Building blocks ─────────────────────


class ResidualProject(nn.Module):
    """Residual projection of MEG channels: x + LayerNorm(Linear(x))."""

    def __init__(self, n_channels: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.LayerNorm(n_channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


class SpatialTemporalAttentionEncoder(nn.Module):
    """Encode MEG: channel → spatial attention → temporal attention.

    Input: (B, T, C) — T time steps, C channels
    Output: (B, T, embed_dim)
    """

    def __init__(self, n_channels: int, embed_dim: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Project each channel to embed_dim
        self.channel_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.spatial_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape

        # Channel projection: (B*T, C, 1) → (B*T, C, embed_dim)
        x = x.reshape(B * T, C, 1)
        x = self.channel_proj(x)

        # Spatial attention: each channel attends to all channels
        spatial_out, _ = self.spatial_attn(x, x, x)
        x = self.norm1(x + spatial_out)

        # Mean pool channels → (B*T, embed_dim), reshape → (B, T, embed_dim)
        x = x.mean(dim=1).reshape(B, T, -1)

        # Temporal attention: each time step attends to all time steps
        temporal_out, _ = self.temporal_attn(x, x, x)
        x = self.norm2(x + temporal_out)

        return x


class BiGRUEncoder(nn.Module):
    """Bidirectional GRU for temporal sequence encoding."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim) → output: (B, T, hidden*2)
        output, hidden = self.gru(x)
        # Concatenate final forward and backward hidden states
        # hidden shape: (num_layers * 2, B, hidden)
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        return torch.cat([hidden_fwd, hidden_bwd], dim=1)


class ResidualFCBlock(nn.Module):
    """Residual FC block: x + LayerNorm(Linear(ReLU(Linear(x))))."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


# ───────────────────── Full model ─────────────────────


@dataclass
class MEGModelAConfig:
    """Configuration for :class:`MEGModelA`."""

    name: str = "meg_model_a"
    n_channels: int = 306
    n_context: int = 5
    embed_dim: int = 256
    gru_hidden: int = 256
    gru_layers: int = 2
    n_heads: int = 4
    semantic_dim: int = 768
    dropout: float = 0.1


class MEGModelA(nn.Module):
    """MEG model A: spatial-temporal attention + BiGRU + residual MLP.

    Args:
        config: :class:`MEGModelAConfig` (or OmegaConf DictConfig).
    """

    def __init__(self, config: MEGModelAConfig | DictConfig):
        super().__init__()
        cfg = self._normalize_config(config)
        self.cfg = cfg

        self.residual_project = ResidualProject(cfg.n_channels, dropout=cfg.dropout)
        self.encoder = SpatialTemporalAttentionEncoder(
            n_channels=cfg.n_channels,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.gru = BiGRUEncoder(
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            dropout=cfg.dropout,
        )
        # Decoder
        self.connect_fc = nn.Sequential(
            nn.Linear(self.gru.output_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.res_fc = nn.ModuleList(
            [ResidualFCBlock(512, dropout=cfg.dropout) for _ in range(2)]
        )
        self.output_fc = nn.Sequential(
            nn.Linear(512, cfg.semantic_dim),
            nn.LayerNorm(cfg.semantic_dim),
        )

    @staticmethod
    def _normalize_config(config: MEGModelAConfig | DictConfig) -> MEGModelAConfig:
        """Convert any config type to MEGModelAConfig."""
        if isinstance(config, MEGModelAConfig):
            return config
        return MEGModelAConfig(
            name=getattr(config, "name", "meg_model_a"),
            n_channels=int(getattr(config, "n_channels", 306)),
            n_context=int(getattr(config, "n_context", 5)),
            embed_dim=int(getattr(config, "embed_dim", 256)),
            gru_hidden=int(getattr(config, "gru_hidden", 256)),
            gru_layers=int(getattr(config, "gru_layers", 2)),
            n_heads=int(getattr(config, "n_heads", 4)),
            semantic_dim=int(getattr(config, "semantic_dim", 768)),
            dropout=float(getattr(config, "dropout", 0.1)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Forward pass.

        Args:
            x: MEG signal. One of:
                - (B, n_context, n_channels) — multi-time-step chunked MEG
                - (B, n_context, n_channels, n_time) — same, with a per-step
                  time window; mean-pooled over ``n_time`` internally
                - (B, n_channels) — single time step (context dim added)

        Returns:
            Tuple of (semantic_prediction, None). The None is for
            API consistency with models that have auxiliary losses
            (e.g., KL for IB).
        """
        if x.ndim == 2:
            # (B, n_channels) → (B, 1, n_channels) — single time step
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            # (B, n_context, n_channels, n_time) → (B, n_context, n_channels)
            # The original model consumes one scalar per (step, channel):
            # mean-pool the per-step time window.
            x = x.mean(dim=-1)

        x = self.residual_project(x)
        x = self.encoder(x)  # (B, T, embed_dim)
        x = self.gru(x)  # (B, gru_hidden*2)

        x = self.connect_fc(x)
        for res_block in self.res_fc:
            x = res_block(x)
        x = self.output_fc(x)
        return x, None

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aux: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss.

        Args:
            predictions: Predicted semantic vectors. Shape (B, semantic_dim).
            targets: Target semantic vectors. Shape (B, semantic_dim).
            aux: Unused for this model (no KL term).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        mse = F.mse_loss(predictions, targets)
        cos = 1 - F.cosine_similarity(predictions, targets, dim=-1).mean()
        # Original model: 0.7 * recon + 0.3 * cos (note: weight convention)
        total = 0.7 * mse + 0.3 * cos

        loss_dict = {
            "mse_loss": mse.item(),
            "cosine_loss": cos.item(),
            "total_loss": total.item(),
        }
        return total, loss_dict


# ───────────────────── Registration ─────────────────────


@register_model("meg_model_a")
def build_meg_model_a(cfg: DictConfig) -> MEGModelA:
    """Build a MEG model A from a Hydra/OmegaConf config.

    Args:
        cfg: Must have a ``name`` field plus the model hyperparameters.

    Returns:
        A constructed :class:`MEGModelA`.
    """
    logger.debug("Building MEGModelA from config: %s", cfg)
    return MEGModelA(cfg)


__all__ = [
    "BiGRUEncoder",
    "MEGModelA",
    "MEGModelAConfig",
    "ResidualFCBlock",
    "ResidualProject",
    "SpatialTemporalAttentionEncoder",
]