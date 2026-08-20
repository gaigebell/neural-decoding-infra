"""Integration tests against the REAL DRDR data (Tier 1, data layer).

These tests are **skipped automatically** when the preprocessed data is not
present (CI, fresh clones). The data root is resolved in this order:

1. ``RECON_PROCESSED_ROOT`` environment variable
2. ``E:/results`` (owner's dev machine)
3. ``/home/test/reconstruction/results`` (cluster)

Run explicitly on the cluster:
    RECON_PROCESSED_ROOT=/home/test/reconstruction/results pytest tests/integration/test_real_data.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from recon.data.collate import collate_brain_stim_pairs
from recon.data.drdr import (
    DEFAULT_DELAY_WEIGHTS,
    discover_drdr,
    load_brain_mask,
    weight_delays,
)
from recon.data.datasets.fmri import fMRIDataset
from recon.data.datasets.meg import MEGDataset
from recon.data.schema import MEGChunkedSample, MEGSample, StimSample, fMRISample


def _resolve_processed_root() -> Path | None:
    candidates = []
    env = os.environ.get("RECON_PROCESSED_ROOT")
    if env:
        candidates.append(Path(env))
    if sys.platform == "win32":
        candidates.append(Path("E:/results"))
    candidates.append(Path("/home/test/reconstruction/results"))
    for c in candidates:
        if c.exists():
            return c
    return None


PROCESSED_ROOT = _resolve_processed_root()

pytestmark = pytest.mark.skipif(
    PROCESSED_ROOT is None,
    reason="real DRDR data not present (set RECON_PROCESSED_ROOT to run)",
)


# ───────────────────── Discovery ─────────────────────


class TestDiscovery:
    def test_meg_discovery(self):
        index = discover_drdr(PROCESSED_ROOT, modality="meg")
        assert index.n_subjects() >= 1
        assert index.n_stories() >= 1
        assert index.n_pairs() >= 1
        # Subject 1 must exist (primary subject in the original pipeline)
        assert 1 in index.subjects

    def test_fmri_discovery(self):
        index = discover_drdr(PROCESSED_ROOT, modality="fmri")
        assert index.n_subjects() >= 1
        assert index.n_pairs() >= 1
        assert 1 in index.subjects


# ───────────────────── Delay weighting ─────────────────────


class TestWeightDelays:
    def test_known_weighting(self):
        # (2 samples, 4 delays × 2 dims): delay values 1..4
        zstim = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            ]
        )
        w = weight_delays(zstim, weights=(1.0, 0.0, 0.0, 0.0), semantic_dim=2)
        np.testing.assert_allclose(w[0], [1.0, 2.0])  # delay 1 only

    def test_default_weights_shape(self):
        zstim = np.random.randn(3, 4 * 768)
        w = weight_delays(zstim)
        assert w.shape == (3, 768)
        assert w.dtype == np.float32

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="last dim"):
            weight_delays(np.random.randn(3, 100))


# ───────────────────── MEG dataset ─────────────────────


class TestRealMEG:
    def test_context_sample_shapes(self):
        index = discover_drdr(PROCESSED_ROOT, modality="meg")
        sub, story = next((s, st) for s, st in index.pairs if s == 1 and st == 1)
        ds = MEGDataset(index, n_context=5, max_steps_per_story=8)
        sample = ds[0]
        assert isinstance(sample.brain, MEGChunkedSample)
        assert sample.brain.x.shape == (5, 306, 1)
        assert sample.brain.x.dtype == np.float32
        assert sample.stim.zstim.shape == (768,)
        assert sample.brain.story_id == story
        assert sample.brain.subject_id == sub

    def test_alignment_shift_zero_padding(self):
        """First (n_context-1) samples use zero-padded response windows."""
        index = discover_drdr(PROCESSED_ROOT, modality="meg")
        ds = MEGDataset(index, n_context=5, max_steps_per_story=8)
        for t in range(4):
            assert np.all(ds[t].brain.x == 0.0)
        assert not np.all(ds[5].brain.x == 0.0)  # after warmup: real data

    def test_plain_no_context(self):
        index = discover_drdr(PROCESSED_ROOT, modality="meg")
        ds = MEGDataset(index, n_context=0, max_steps_per_story=4)
        sample = ds[0]
        assert sample.brain.x.shape == (306, 1)
        assert not np.all(sample.brain.x == 0.0)

    def test_collate_and_model_forward(self):
        """Real batch: collate → MEGModelA forward (CPU)."""
        import torch

        from recon.models.meg.meg_model_a import MEGModelA, MEGModelAConfig

        index = discover_drdr(PROCESSED_ROOT, modality="meg")
        # Restrict to one story to keep the test fast
        sub, story = next((s, st) for s, st in index.pairs if s == 1 and st == 1)
        from recon.engine.trainer import _filter_index
        from omegaconf import OmegaConf

        filtered = _filter_index(
            index, OmegaConf.create({"subjects": [sub], "stories": [story]})
        )
        ds = MEGDataset(filtered, n_context=5, max_steps_per_story=8)
        batch = collate_brain_stim_pairs([ds[i] for i in range(min(4, len(ds)))])

        model = MEGModelA(MEGModelAConfig(n_channels=306, n_context=5))
        model.eval()
        with torch.no_grad():
            out, aux = model(batch.brain.x)
        assert out.shape == (batch.brain.x.shape[0], 768)
        # Target is already the weighted 768 vector — loss computes directly
        loss, _ = model.compute_loss(out, batch.stim.zstim)
        assert loss.ndim == 0 and loss.item() > 0


# ───────────────────── fMRI dataset ─────────────────────


class TestRealFMRI:
    def test_volume_and_mask_shapes(self):
        index = discover_drdr(PROCESSED_ROOT, modality="fmri")
        ds = fMRIDataset(index, max_steps_per_story=2)
        sample = ds[0]
        assert isinstance(sample.brain, fMRISample)
        assert sample.brain.volume.shape == (91, 109, 91)
        assert sample.brain.volume.dtype == np.float32
        assert sample.brain.mask.shape == (91, 109, 91)
        assert sample.brain.mask.dtype == bool
        assert sample.stim.zstim.shape == (768,)

    def test_mask_matches_volume_voxels(self):
        mask = load_brain_mask(PROCESSED_ROOT)
        assert mask.shape == (91, 109, 91)
        assert mask.any()

    def test_collate_fmri(self):
        index = discover_drdr(PROCESSED_ROOT, modality="fmri")
        ds = fMRIDataset(index, max_steps_per_story=2)
        batch = collate_brain_stim_pairs([ds[0], ds[1]])
        assert batch.brain.x.shape == (2, 91, 109, 91)
        assert batch.stim.zstim.shape == (2, 768)
