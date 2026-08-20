"""PyTorch Dataset for fMRI (DRDR) data.

Returns a ``BrainStimPair`` for each ``(subject, story, time_step)``.

Cube files are opened as memmaps (one story is ~2 GB on disk; memmap pages
only the volumes actually read). The brain mask is loaded once per dataset.

Usage:
    >>> from recon.data.drdr import discover_drdr
    >>> from recon.data.datasets.fmri import fMRIDataset
    >>> index = discover_drdr("E:/results", modality="fmri")
    >>> dataset = fMRIDataset(index)
    >>> sample = dataset[0]
    >>> sample.brain.volume.shape
    (91, 109, 91)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ..drdr import (
    DEFAULT_DELAY_WEIGHTS,
    DRDRIndex,
    load_brain_mask,
    load_fmri_story,
    load_stim_story,
)
from ..schema import BrainStimPair, StimSample, fMRISample

logger = logging.getLogger(__name__)


class fMRIDataset(Dataset):
    """fMRI dataset over all (subject, story, time_step) samples.

    Args:
        index: DRDRIndex pointing to valid (subject, story) pairs.
        layer: GPT-2 layer for zstim (default 10).
        weights: Delay weights applied to zstim (default [0.1, 0.7, 0.5, 0.3]).
        mask_path: Optional explicit path to the brain mask nii.
            If None, loads ``<processed_root>/mask.nii``.
        max_steps_per_story: Optional cap on time steps per story (smoke).
    """

    def __init__(
        self,
        index: DRDRIndex,
        layer: int = 10,
        weights: tuple[float, ...] = DEFAULT_DELAY_WEIGHTS,
        mask_path: str | Path | None = None,
        max_steps_per_story: int | None = None,
    ):
        self.index = index
        self.processed_root = Path(index.processed_root)
        self.layer = layer
        self.weights = tuple(weights)

        self._items: list[tuple[int, int, int]] = []
        self._story_vol: dict[tuple[int, int], np.ndarray] = {}
        self._story_stim: dict[tuple[int, int], np.ndarray] = {}
        for sub, story in index.pairs:
            stim = self._stim(sub, story)
            n_steps = stim.shape[0]
            if max_steps_per_story is not None:
                n_steps = min(n_steps, max_steps_per_story)
            self._items.extend((sub, story, t) for t in range(n_steps))

        # Brain mask — shared across all samples
        if mask_path is not None:
            import nibabel as nib

            self.mask = nib.load(str(mask_path)).get_fdata().astype(bool)
        else:
            self.mask = load_brain_mask(self.processed_root)

        logger.info(
            "fMRIDataset: %d pairs -> %d samples (mask=%s)",
            len(index.pairs), len(self._items), self.mask.shape,
        )

    # ───────────────────── story cache ─────────────────────

    def _vol(self, sub: int, story: int) -> np.ndarray:
        key = (sub, story)
        if key not in self._story_vol:
            self._story_vol[key] = load_fmri_story(self.processed_root, sub, story)
        return self._story_vol[key]

    def _stim(self, sub: int, story: int) -> np.ndarray:
        key = (sub, story)
        if key not in self._story_stim:
            self._story_stim[key] = load_stim_story(
                self.processed_root, sub, story,
                layer=self.layer, weights=self.weights, meg=False,
            )
        return self._story_stim[key]

    # ───────────────────── Dataset API ─────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> BrainStimPair:
        sub, story, t = self._items[idx]
        brain = fMRISample(
            volume=np.asarray(self._vol(sub, story)[t]),
            mask=self.mask,
            story_id=story,
            subject_id=sub,
        )
        target = StimSample(
            zstim=self._stim(sub, story)[t],
            layer=self.layer,
            story_id=story,
            subject_id=sub,
        )
        return BrainStimPair(brain=brain, stim=target)


__all__ = ["fMRIDataset"]
