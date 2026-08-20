"""Unit tests for ``recon.data.schema``.

Validates that pydantic schemas catch the kinds of bugs that
caused silent training failures in the previous code:
- Wrong tensor shape
- Wrong dtype
- Out-of-range sensor type codes
- Missing fields
"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from recon.data.schema import (
    BrainOmniSample,
    BrainStimPair,
    MEGChunkedSample,
    MEGSample,
    StimSample,
    fMRISample,
)


# ───────────────────── StimSample ─────────────────────


class TestStimSample:
    """Tests for :class:`StimSample`."""

    def test_valid_1d_zstim(self):
        """Should accept a 1D zstim vector."""
        zstim = np.random.randn(4 * 768).astype(np.float32)
        s = StimSample(zstim=zstim, story_id=1, subject_id=1)
        assert s.zstim.shape == (4 * 768,)

    def test_rejects_2d_zstim(self):
        """Should reject 2D zstim (must be 1D after flattening)."""
        with pytest.raises(ValidationError, match="1D"):
            StimSample(zstim=np.zeros((4, 768), dtype=np.float32), story_id=1, subject_id=1)

    def test_auto_casts_to_float32(self):
        """Should auto-cast float64 to float32."""
        zstim = np.random.randn(4 * 768).astype(np.float64)
        s = StimSample(zstim=zstim, story_id=1, subject_id=1)
        assert s.zstim.dtype == np.float32

    def test_default_delays(self):
        """Default delays should be (1, 2, 3, 4)."""
        zstim = np.random.randn(4 * 768).astype(np.float32)
        s = StimSample(zstim=zstim, story_id=1, subject_id=1)
        assert s.delays == (1, 2, 3, 4)

    def test_default_layer(self):
        """Default layer should be 10."""
        zstim = np.random.randn(4 * 768).astype(np.float32)
        s = StimSample(zstim=zstim, story_id=1, subject_id=1)
        assert s.layer == 10


# ───────────────────── MEGSample ─────────────────────


class TestMEGSample:
    """Tests for :class:`MEGSample`."""

    def test_valid_meg(self):
        """Should accept a 2D MEG signal."""
        x = np.random.randn(306, 256).astype(np.float32)
        pos = np.random.randn(306, 6).astype(np.float32)
        st = np.array([1] * 102 + [2] * 204, dtype=np.int32)
        s = MEGSample(x=x, pos=pos, sensor_type=st, story_id=1, subject_id=1)
        assert s.x.shape == (306, 256)

    def test_rejects_3d_x(self):
        """Should reject 3D x (use MEGChunkedSample instead)."""
        x = np.random.randn(5, 306, 256).astype(np.float32)
        pos = np.random.randn(306, 6).astype(np.float32)
        st = np.zeros(306, dtype=np.int32)
        with pytest.raises(ValidationError, match="2D"):
            MEGSample(x=x, pos=pos, sensor_type=st, story_id=1, subject_id=1)

    def test_rejects_wrong_pos_shape(self):
        """Should reject pos with wrong last dim."""
        x = np.random.randn(306, 256).astype(np.float32)
        pos = np.random.randn(306, 5).astype(np.float32)  # should be 6
        st = np.zeros(306, dtype=np.int32)
        with pytest.raises(ValidationError, match="6"):
            MEGSample(x=x, pos=pos, sensor_type=st, story_id=1, subject_id=1)

    def test_rejects_invalid_sensor_type(self):
        """Should reject sensor_type values outside {0, 1, 2}."""
        x = np.random.randn(306, 256).astype(np.float32)
        pos = np.random.randn(306, 6).astype(np.float32)
        st = np.array([3] * 306, dtype=np.int32)  # 3 is invalid
        with pytest.raises(ValidationError, match="sensor_type"):
            MEGSample(x=x, pos=pos, sensor_type=st, story_id=1, subject_id=1)

    def test_accepts_valid_sensor_types(self):
        """Should accept all valid sensor types {0, 1, 2}."""
        x = np.random.randn(306, 256).astype(np.float32)
        pos = np.random.randn(306, 6).astype(np.float32)
        st = np.array([0] * 100 + [1] * 100 + [2] * 106, dtype=np.int32)
        s = MEGSample(x=x, pos=pos, sensor_type=st, story_id=1, subject_id=1)
        assert s.sensor_type.sum() == 100 * 0 + 100 * 1 + 106 * 2


# ───────────────────── MEGChunkedSample ─────────────────────


class TestMEGChunkedSample:
    """Tests for :class:`MEGChunkedSample`."""

    def test_valid_3d_x(self):
        """Should accept (context, channels, time) shape."""
        x = np.random.randn(5, 306, 256).astype(np.float32)
        pos = np.random.randn(306, 6).astype(np.float32)
        st = np.zeros(306, dtype=np.int32)
        s = MEGChunkedSample(
            x=x, pos=pos, sensor_type=st, context_len=5, story_id=1, subject_id=1
        )
        assert s.x.shape == (5, 306, 256)
        assert s.context_len == 5

    def test_rejects_2d_x(self):
        """Should reject 2D x (use MEGSample instead)."""
        x = np.random.randn(306, 256).astype(np.float32)
        pos = np.random.randn(306, 6).astype(np.float32)
        st = np.zeros(306, dtype=np.int32)
        with pytest.raises(ValidationError, match="3D"):
            MEGChunkedSample(
                x=x, pos=pos, sensor_type=st, context_len=5, story_id=1, subject_id=1
            )


# ───────────────────── fMRISample ─────────────────────


class TestFMRISample:
    """Tests for :class:`fMRISample`."""

    def test_valid_fmri(self):
        """Should accept a 3D volume + 3D mask."""
        volume = np.random.randn(53, 63, 52).astype(np.float32)
        mask = np.ones((53, 63, 52), dtype=bool)
        s = fMRISample(volume=volume, mask=mask, story_id=1, subject_id=1)
        assert s.volume.shape == (53, 63, 52)
        assert s.mask.shape == (53, 63, 52)

    def test_rejects_4d_volume(self):
        """Should reject 4D volume (no batch dim)."""
        volume = np.random.randn(10, 53, 63, 52).astype(np.float32)
        mask = np.ones((53, 63, 52), dtype=bool)
        with pytest.raises(ValidationError, match="3D"):
            fMRISample(volume=volume, mask=mask, story_id=1, subject_id=1)

    def test_mask_auto_cast_to_bool(self):
        """Should auto-cast int mask to bool."""
        volume = np.random.randn(53, 63, 52).astype(np.float32)
        mask = np.ones((53, 63, 52), dtype=np.int32)  # int, not bool
        s = fMRISample(volume=volume, mask=mask, story_id=1, subject_id=1)
        assert s.mask.dtype == bool


# ───────────────────── BrainOmniSample ─────────────────────


class TestBrainOmniSample:
    """Tests for :class:`BrainOmniSample`."""

    def test_valid_brainomni(self):
        """Should accept (n_neurons, seq_len, n_dim) shape."""
        features = np.random.randn(306, 1, 1024).astype(np.float32)
        s = BrainOmniSample(features=features, subject_id=1, story_id=1)
        assert s.features.shape == (306, 1, 1024)

    def test_rejects_2d(self):
        """Should reject 2D features."""
        features = np.random.randn(306, 1024).astype(np.float32)
        with pytest.raises(ValidationError, match="3D"):
            BrainOmniSample(features=features, subject_id=1, story_id=1)


# ───────────────────── BrainStimPair ─────────────────────


class TestBrainStimPair:
    """Tests for :class:`BrainStimPair`."""

    def test_meg_pair(self):
        """Should build a valid MEG + stim pair."""
        brain = MEGSample(
            x=np.random.randn(306, 256).astype(np.float32),
            pos=np.random.randn(306, 6).astype(np.float32),
            sensor_type=np.zeros(306, dtype=np.int32),
            story_id=1, subject_id=1,
        )
        stim = StimSample(
            zstim=np.random.randn(4 * 768).astype(np.float32),
            story_id=1, subject_id=1,
        )
        pair = BrainStimPair(brain=brain, stim=stim)
        assert pair.modality == "meg"

    def test_fmri_pair(self):
        """Should build a valid fMRI + stim pair."""
        brain = fMRISample(
            volume=np.random.randn(53, 63, 52).astype(np.float32),
            mask=np.ones((53, 63, 52), dtype=bool),
            story_id=1, subject_id=1,
        )
        stim = StimSample(
            zstim=np.random.randn(4 * 768).astype(np.float32),
            story_id=1, subject_id=1,
        )
        pair = BrainStimPair(brain=brain, stim=stim)
        assert pair.modality == "fmri"