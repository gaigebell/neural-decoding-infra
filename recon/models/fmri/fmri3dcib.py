"""fMRI 3D CNN + Information Bottleneck alignment model.

This is the standard reference model for fMRI:
- 3D CNN backbone extracts spatial features
- Information bottleneck aligner maps features to 768-dim semantic space
- Variational inference (mu, logvar) + KL regularization

Ported and improved from the original ``recon/model/fMRI3dCIBModel.py``.

Architecture:
    Input (B, 1, X, Y, Z)
        ↓
    fMRICNNFeatureExtractor  (3D conv stack, output_dim=1000)
        ↓
    InformationBottleneckAligner
        Encoder: 1000 → 512 → 256 (mu, logvar)
        Sample: z = mu + std * eps
        Decoder: 256 → 512 → 1024 → 768
        ↓
    Output (B, 768)

Loss:
    recon = 0.7 * cosine_loss + 0.3 * mse_loss
    total = recon + beta * kl_loss
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


# ───────────────────── Backbone ─────────────────────


class FMRI3DCNNBackbone(nn.Module):
    """3D CNN backbone for fMRI feature extraction.

    Three layers of (conv3d + ReLU + maxpool). Output is a (B, output_dim)
    vector after global max pooling.
    """

    def __init__(self, output_dim: int = 1000):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 128, kernel_size=7, padding=3, bias=False),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, output_dim, kernel_size=7, padding=3, bias=False),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(-1, self.output_dim)


# ───────────────────── Information bottleneck aligner ─────────────────────


class InformationBottleneckAligner(nn.Module):
    """Variational information bottleneck aligner.

    Maps a feature vector of size ``input_dim`` to ``semantic_dim`` using
    a Gaussian latent variable. The KL term regularizes the latent to be
    close to a standard normal prior.
    """

    def __init__(
        self,
        input_dim: int,
        semantic_dim: int = 768,
        bottleneck_dim: int = 256,
        beta: float = 1e-3,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.semantic_dim = semantic_dim
        self.beta = beta

        # Encoder: produces mu and logvar
        self.encoder_mu = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )

        # Decoder: from bottleneck to semantic space
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim * 2, semantic_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(semantic_dim * 2, semantic_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from N(mu, sigma) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features. Shape (B, input_dim).

        Returns:
            Tuple of (semantic_output, kl_loss).
        """
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        z = self.reparameterize(mu, logvar)
        semantic = self.decoder(z)

        # KL divergence to standard normal
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        return semantic, kl * self.beta


# ───────────────────── Full model ─────────────────────


@dataclass
class FMRI3DCIBConfig:
    """Configuration for :class:`FMRI3DCIBModel`."""

    name: str = "fmri3dcib"
    input_shape: tuple[int, int, int] = (53, 63, 52)
    semantic_dim: int = 768
    backbone_dim: int = 1000
    bottleneck_dim: int = 256
    beta: float = 1e-3


class FMRI3DCIBModel(nn.Module):
    """fMRI brain-to-semantic model with 3D CNN + IB aligner.

    Args:
        config: :class:`FMRI3DCIBConfig` (or OmegaConf DictConfig).
    """

    def __init__(self, config: FMRI3DCIBConfig | DictConfig):
        super().__init__()
        # Support both dataclass and OmegaConf
        cfg = self._normalize_config(config)

        self.cfg = cfg
        self.backbone = FMRI3DCNNBackbone(output_dim=cfg.backbone_dim)
        self.aligner = InformationBottleneckAligner(
            input_dim=cfg.backbone_dim,
            semantic_dim=cfg.semantic_dim,
            bottleneck_dim=cfg.bottleneck_dim,
            beta=cfg.beta,
        )

    @staticmethod
    def _normalize_config(config: FMRI3DCIBConfig | DictConfig) -> FMRI3DCIBConfig:
        """Convert any config type to FMRI3DCIBConfig."""
        if isinstance(config, FMRI3DCIBConfig):
            return config
        # OmegaConf / dict
        return FMRI3DCIBConfig(
            name=getattr(config, "name", "fmri3dcib"),
            input_shape=tuple(getattr(config, "input_shape", (53, 63, 52))),
            semantic_dim=int(getattr(config, "semantic_dim", 768)),
            backbone_dim=int(getattr(config, "backbone_dim", 1000)),
            bottleneck_dim=int(getattr(config, "bottleneck_dim", 256)),
            beta=float(getattr(config, "beta", 1e-3)),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: fMRI volume. Shape (B, 1, X, Y, Z) or (B, X, Y, Z).

        Returns:
            Tuple of (semantic_prediction, kl_loss).
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)  # add channel dim
        features = self.backbone(x)
        semantic, kl_loss = self.aligner(features)
        return semantic, kl_loss

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the training loss.

        Args:
            predictions: Predicted semantic vectors. Shape (B, semantic_dim).
            targets: Target semantic vectors. Shape (B, semantic_dim).
            aux: Auxiliary outputs (e.g., ``{"kl_loss": tensor}``).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        mse = F.mse_loss(predictions, targets)
        cos = 1 - F.cosine_similarity(predictions, targets, dim=-1).mean()
        recon = 0.7 * cos + 0.3 * mse

        kl = aux.get("kl_loss", torch.tensor(0.0, device=predictions.device)) if aux else torch.tensor(0.0, device=predictions.device)
        total = recon + kl

        loss_dict = {
            "mse_loss": mse.item(),
            "cosine_loss": cos.item(),
            "recon_loss": recon.item(),
            "kl_loss": kl.item() if torch.is_tensor(kl) else float(kl),
            "total_loss": total.item(),
        }
        return total, loss_dict


# ───────────────────── Registration ─────────────────────


@register_model("fmri3dcib")
def build_fmri3dcib(cfg: DictConfig) -> FMRI3DCIBModel:
    """Build an fMRI 3D-CNN + IB model from a Hydra/OmegaConf config.

    Args:
        cfg: Must have a ``name`` field plus the model hyperparameters.

    Returns:
        A constructed :class:`FMRI3DCIBModel`.
    """
    logger.debug("Building FMRI3DCIBModel from config: %s", cfg)
    return FMRI3DCIBModel(cfg)


__all__ = ["FMRI3DCIBConfig", "FMRI3DCIBModel", "FMRI3DCNNBackbone", "InformationBottleneckAligner"]