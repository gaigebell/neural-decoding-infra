# Guide 03: Write a new model

> **Audience**: Anyone adding a new neural-network architecture.

---

## What "writing a new model" means here

Adding a new brain-to-semantic model: a new architecture (or variant) that maps brain signals to 768-dim semantic vectors.

Examples:

- New 3D CNN variant for fMRI
- New attention-based model for MEG
- New alignment head for BrainOmni features

## Steps

### 1. Choose the right sub-package

```
recon/models/
├── registry.py                # MODEL_REGISTRY — register here
├── fmri/                      # fMRI-specific models
├── meg/                       # MEG-specific models
└── brainomni/                 # BrainOmni alignment heads
```

Pick the sub-package matching your modality.

### 2. Implement the model

```python
# recon/models/fmri/fmri_attention.py
"""fMRI model with attention-based alignment.

3D CNN backbone produces a 1000-dim feature, followed by a multi-head
attention block that maps to 768-dim semantic space.
"""
import torch
from torch import nn

from ..registry import register_model
from ...config import ModelConfig


@register_model("fmri_attention")
class FmriAttentionModel(nn.Module):
    """fMRI model with 3D CNN backbone + attention aligner.

    Args:
        input_shape: Tuple (X, Y, Z) — e.g., (53, 63, 52).
        semantic_dim: Output dim, default 768.
        hidden_dim: Attention hidden dim, default 1024.
        num_heads: Number of attention heads, default 8.
        num_layers: Number of attention layers, default 3.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        semantic_dim: int = 768,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.semantic_dim = semantic_dim

        # Backbone: 3D CNN (similar to existing fMRI models)
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 128, 7, padding=3), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(128, 1000, 7, padding=3), nn.ReLU(), nn.AdaptiveMaxPool3d(1),
        )

        # Attention aligner
        self.input_proj = nn.Linear(1000, hidden_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
            ) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, semantic_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input fMRI volume. Shape (B, 1, X, Y, Z) or (B, X, Y, Z).

        Returns:
            Predicted semantic vectors. Shape (B, semantic_dim).
        """
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B, X, Y, Z) → (B, 1, X, Y, Z)

        feat = self.backbone(x).flatten(1)  # (B, 1000)
        feat = self.input_proj(feat).unsqueeze(1)  # (B, 1, hidden_dim)

        for attn, norm, ffn in zip(self.attn_layers, self.norms, self.ffns):
            attn_out, _ = attn(feat, feat, feat)
            feat = norm(feat + attn_out)
            feat = norm(feat + ffn(feat))

        return self.output_proj(feat.squeeze(1))  # (B, semantic_dim)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        beta: float = 1e-3,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss.

        Args:
            predictions: Predicted semantic vectors. Shape (B, semantic_dim).
            targets: Ground-truth semantic vectors. Shape (B, semantic_dim).
            beta: Weight for KL term (if applicable).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        recon_loss = nn.functional.mse_loss(predictions, targets)
        cos_loss = 1 - nn.functional.cosine_similarity(predictions, targets, dim=-1).mean()
        total = 0.7 * recon_loss + 0.3 * cos_loss
        return total, {"mse": recon_loss.item(), "cos": cos_loss.item(), "total": total.item()}
```

### 3. Register in the registry

Add to `recon/models/registry.py`:

```python
MODEL_REGISTRY = {
    # ...existing entries...
    "fmri_attention": build_fmri_attention,
}

def build_fmri_attention(cfg: ModelConfig) -> nn.Module:
    from .fmri.fmri_attention import FmriAttentionModel
    return FmriAttentionModel(
        input_shape=cfg.input_shape,
        semantic_dim=cfg.semantic_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
    )
```

### 4. Add Hydra config

Create `configs/model/fmri_attention.yaml`:

```yaml
name: fmri_attention
input_shape: [53, 63, 52]
semantic_dim: 768
hidden_dim: 1024
num_heads: 8
num_layers: 3
```

### 5. Write a unit test

```python
# tests/unit/test_fmri_attention.py
"""Tests for FmriAttentionModel."""
import pytest
import torch
from recon.models.fmri.fmri_attention import FmriAttentionModel


@pytest.fixture
def model():
    return FmriAttentionModel(input_shape=(53, 63, 52))


def test_forward_shape(model):
    """Forward should produce (B, 768) output."""
    x = torch.randn(2, 1, 53, 63, 52)
    y = model(x)
    assert y.shape == (2, 768)


def test_forward_4d_input(model):
    """Forward should accept 4D input and add channel dim."""
    x = torch.randn(2, 53, 63, 52)
    y = model(x)
    assert y.shape == (2, 768)


def test_compute_loss(model):
    """Compute loss should return scalar + dict."""
    pred = torch.randn(4, 768)
    target = torch.randn(4, 768)
    loss, loss_dict = model.compute_loss(pred, target)
    assert loss.ndim == 0
    assert "mse" in loss_dict
    assert "cos" in loss_dict
```

### 6. Update architecture docs

Add a section to [architecture/03-model-registry.md](../architecture/03-model-registry.md) (when written).

### 7. Run the cluster smoke

After merging:

```bash
# On a compute node
python -m recon.cli.train model=fmri_attention paths=cluster train.smoke=true
```

## Checklist

- [ ] Model class implemented with type hints + Google docstring
- [ ] `@register_model("name")` decorator
- [ ] Builder function in `recon/models/registry.py`
- [ ] Hydra config in `configs/model/<name>.yaml`
- [ ] Unit tests covering forward shape + compute_loss
- [ ] (Optional) ADR if a significant design decision was made
- [ ] Cluster smoke test passes

## Anti-patterns

- ❌ Putting training loop logic in the model (forward only)
- ❌ Hardcoded paths or device assumptions
- ❌ Print statements instead of using the logger
- ❌ Type hints missing
- ❌ Forward function that mutates input
- ❌ Not registering in MODEL_REGISTRY (then it's not usable from CLI)

## See also

- [Architecture: Model registry](../architecture/03-model-registry.md)
- [ADR-0006: Unified Trainer](../decisions/0006-unified-trainer.md)
- [Standard 02: Docstring style](../standards/02-docstring.md)
- [Guide 04: Run training](04-run-training.md)

---

Maintained by owner.