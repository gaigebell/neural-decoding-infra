"""PyTorch Dataset for MEG (DRDR) data.

Returns a ``BrainStimPair`` for each ``(subject, story, time_step)``.

Stories are opened as memmaps and cached per story, so memory grows with
the number of stories touched, not the whole dataset. Per the original
pipeline, when ``n_context > 0`` the model input at stimulus step ``t`` is
the response context window from step ``t - (n_context - 1)``, zero-padded
at the start.

Usage:
    >>> from recon.data.drdr import discover_drdr
    >>> from recon.data.datasets.meg import MEGDataset
    >>> index = discover_drdr("E:/results", modality="meg")
    >>> dataset = MEGDataset(index, n_context=5)
    >>> sample = dataset[0]
    >>> sample.brain.x.shape
    (5, 306, 1)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ..drdr import (
    DEFAULT_DELAY_WEIGHTS,
    DRDRIndex,
    load_meg_story,
    load_stim_story,
)
from ..schema import BrainStimPair, MEGChunkedSample, MEGSample, StimSample

logger = logging.getLogger(__name__)


class MEGDataset(Dataset):
    """MEG dataset over all (subject, story, time_step) samples.

    Args:
        index: DRDRIndex pointing to valid (subject, story) pairs.
        n_context: If > 0, use the pre-built context-window files
            ``zresp{sub}_{story}_context_{n_context}.npy`` (T, n_context, C)
            and apply the original pipeline's alignment shift (input at
            step ``t`` comes from response window ``t - (n_context - 1)``,
            zero-padded at the start). If 0, use the plain (T, C) files
            with no shift.
        layer: GPT-2 layer for zstim (default 10).
        weights: Delay weights applied to zstim (default [0.1, 0.7, 0.5, 0.3]).
        max_steps_per_story: Optional cap on time steps per story (smoke).
    """

    def __init__(
        self,
        index: DRDRIndex,
        n_context: int = 0,
        layer: int = 10,
        weights: tuple[float, ...] = DEFAULT_DELAY_WEIGHTS,
        max_steps_per_story: int | None = None,
    ):
        self.index = index
        self.processed_root = Path(index.processed_root)
        self.n_context = n_context
        self.layer = layer
        self.weights = tuple(weights)

        # Flat item list: (subject, story, t). Story lengths come from the
        # stim arrays (they define the sampling grid; response rows align).
        self._items: list[tuple[int, int, int]] = []
        self._story_resp: dict[tuple[int, int], np.ndarray] = {}
        self._story_stim: dict[tuple[int, int], np.ndarray] = {}
        for sub, story in index.pairs:
            stim = self._stim(sub, story)
            n_steps = stim.shape[0]
            if max_steps_per_story is not None:
                n_steps = min(n_steps, max_steps_per_story)
            self._items.extend((sub, story, t) for t in range(n_steps))

        logger.info(
            "MEGDataset: %d pairs -> %d samples (n_context=%d)",
            len(index.pairs), len(self._items), n_context,
        )

    # ───────────────────── story cache ─────────────────────

    def _resp(self, sub: int, story: int) -> np.ndarray:
        key = (sub, story)
        if key not in self._story_resp:
            self._story_resp[key] = load_meg_story(
                self.processed_root, sub, story, n_context=self.n_context
            )
        return self._story_resp[key]

    def _stim(self, sub: int, story: int) -> np.ndarray:
        key = (sub, story)
        if key not in self._story_stim:
            self._story_stim[key] = load_stim_story(
                self.processed_root, sub, story,
                layer=self.layer, weights=self.weights, meg=True,
            )
        return self._story_stim[key]

    # ───────────────────── Dataset API ─────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> BrainStimPair:
        sub, story, t = self._items[idx]
        resp = self._resp(sub, story)
        stim = self._stim(sub, story)

        n_channels = resp.shape[-1]
        pos = np.zeros((n_channels, 6), dtype=np.float32)
        sensor_type = np.array(
            [1] * (n_channels // 3) + [2] * (n_channels - n_channels // 3),
            dtype=np.int32,
        )
        if self.n_context > 0:
            # Original pipeline: input at step t is the response context
            # window from t - (n_context - 1), zero-padded at the start.
            j = t - (self.n_context - 1)
            if j < 0:
                row = np.zeros((self.n_context, n_channels), dtype=np.float32)
            else:
                row = np.asarray(resp[j])
            x = row[..., None]  # (n_context, C) -> (n_context, C, 1)
            brain = MEGChunkedSample(
                x=x,
                pos=pos,
                sensor_type=sensor_type,
                context_len=self.n_context,
                story_id=story,
                subject_id=sub,
            )
        else:
            x = np.asarray(resp[t])[:, None]  # (C,) -> (C, 1)
            brain = MEGSample(
                x=x,
                pos=pos,
                sensor_type=sensor_type,
                story_id=story,
                subject_id=sub,
            )
        target = StimSample(
            zstim=stim[t],
            layer=self.layer,
            story_id=story,
            subject_id=sub,
        )
        return BrainStimPair(brain=brain, stim=target)


__all__ = ["MEGDataset"]
