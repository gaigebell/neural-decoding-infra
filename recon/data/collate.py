"""Batch collation: list of :class:`BrainStimPair` → batched tensors.

Pydantic schemas validate *single* samples; a PyTorch ``DataLoader`` needs a
``collate_fn`` to stack them into tensors. This module is the **only** place
where samples become batches (trainer, decoder, and evaluator all consume
batches produced here).

The batch shape mirrors the schema shapes, with a leading batch dim:

    MEG          x: (B, C, T)          pos: (B, C, 6)  sensor_type: (B, C)
    MEG chunked  x: (B, ctx, C, T)     pos: (B, C, 6)  sensor_type: (B, C)
    fMRI         x: (B, X, Y, Z)       mask: (B, X, Y, Z)
    BrainOmni    x: (B, n_neurons, L, D)
    Stim         zstim: (B, semantic_dim)  (delay-weighted target)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .schema import BrainStimPair


@dataclass
class BrainBatch:
    """Batched brain data for one modality."""

    modality: str
    x: torch.Tensor  # (B, ...) — modality-specific shape, see module docstring
    pos: torch.Tensor | None = None  # (B, C, 6) for MEG
    sensor_type: torch.Tensor | None = None  # (B, C) for MEG, int
    mask: torch.Tensor | None = None  # (B, X, Y, Z) for fMRI, bool


@dataclass
class StimBatch:
    """Batched stimulus (semantic features)."""

    zstim: torch.Tensor  # (B, n_delays * semantic_dim)


@dataclass
class BrainStimBatch:
    """A batch of (brain, stimulus) pairs as tensors."""

    brain: BrainBatch
    stim: StimBatch


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr))


def collate_brain_stim_pairs(samples: list[BrainStimPair]) -> BrainStimBatch:
    """Collate validated single samples into one batched :class:`BrainStimBatch`.

    Args:
        samples: Non-empty list of schema-validated ``BrainStimPair``.
            All samples must be of the same modality.

    Returns:
        A ``BrainStimBatch`` with ``.brain.x`` / ``.brain.pos`` / ``.stim.zstim``
        tensors — the interface the trainer/decoder expect.

    Raises:
        ValueError: If ``samples`` is empty or modalities are mixed.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    modalities = {s.modality for s in samples}
    if len(modalities) != 1:
        raise ValueError(f"Cannot collate mixed modalities: {sorted(modalities)}")
    modality = samples[0].modality

    if modality in ("meg", "meg_chunked"):
        x = _to_tensor(np.stack([s.brain.x for s in samples])).float()
        pos = _to_tensor(np.stack([s.brain.pos for s in samples])).float()
        sensor_type = _to_tensor(np.stack([s.brain.sensor_type for s in samples]))
        brain = BrainBatch(
            modality=modality, x=x, pos=pos, sensor_type=sensor_type
        )
    elif modality == "fmri":
        x = _to_tensor(np.stack([s.brain.volume for s in samples])).float()
        mask = _to_tensor(np.stack([s.brain.mask for s in samples])).bool()
        brain = BrainBatch(modality=modality, x=x, mask=mask)
    elif modality == "brainomni":
        x = _to_tensor(np.stack([s.brain.features for s in samples])).float()
        brain = BrainBatch(modality=modality, x=x)
    else:  # pragma: no cover — schema already constrains modality
        raise ValueError(f"Unknown modality: {modality}")

    zstim = _to_tensor(np.stack([s.stim.zstim for s in samples])).float()
    return BrainStimBatch(brain=brain, stim=StimBatch(zstim=zstim))


__all__ = ["BrainBatch", "BrainStimBatch", "StimBatch", "collate_brain_stim_pairs"]
