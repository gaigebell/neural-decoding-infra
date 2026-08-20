"""Unit tests for :mod:`recon.data.collate`.

Verifies that a list of schema-validated :class:`BrainStimPair` samples
collates into correctly-shaped batched tensors for every modality.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from recon.data.collate import collate_brain_stim_pairs
from recon.data.fake_data import fake_pair_fmri, fake_pair_meg


def _meg_pairs(rng, n: int = 3, n_context: int = 0) -> list:
    return [
        fake_pair_meg(rng, story_id=i + 1, subject_id=1, n_context=n_context)
        for i in range(n)
    ]


def _fmri_pairs(rng, n: int = 3) -> list:
    return [fake_pair_fmri(rng, story_id=i + 1, subject_id=1) for i in range(n)]


class TestCollateMEG:
    def test_meg_shapes(self):
        rng = np.random.default_rng(0)
        batch = collate_brain_stim_pairs(_meg_pairs(rng))
        assert batch.brain.modality == "meg"
        assert batch.brain.x.shape == (3, 306, 256)
        assert batch.brain.pos.shape == (3, 306, 6)
        assert batch.brain.sensor_type.shape == (3, 306)
        assert batch.stim.zstim.shape == (3, 768)
        assert batch.brain.x.dtype == torch.float32
        assert batch.brain.sensor_type.dtype == torch.int32

    def test_meg_chunked_shapes(self):
        rng = np.random.default_rng(0)
        batch = collate_brain_stim_pairs(_meg_pairs(rng, n_context=5))
        assert batch.brain.modality == "meg_chunked"
        assert batch.brain.x.shape == (3, 5, 306, 256)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            collate_brain_stim_pairs([])


class TestCollateFMRI:
    def test_fmri_shapes(self):
        rng = np.random.default_rng(0)
        batch = collate_brain_stim_pairs(_fmri_pairs(rng))
        assert batch.brain.modality == "fmri"
        assert batch.brain.x.shape == (3, 53, 63, 52)
        assert batch.brain.mask.shape == (3, 53, 63, 52)
        assert batch.brain.mask.dtype == torch.bool
        assert batch.stim.zstim.shape == (3, 768)


class TestCollateMixed:
    def test_mixed_modalities_raise(self):
        rng = np.random.default_rng(0)
        mixed = [_meg_pairs(rng)[0], _fmri_pairs(rng)[0]]
        with pytest.raises(ValueError, match="mixed"):
            collate_brain_stim_pairs(mixed)
