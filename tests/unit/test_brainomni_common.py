"""Unit tests for :mod:`recon.preprocessing.brainomni` normalization helpers."""
from __future__ import annotations

import numpy as np
import pytest

from recon.preprocessing.brainomni import _normalize_pos, _sensortype_wise_normalize


class TestNormalizePos:
    def test_centers_and_scales(self):
        pos = np.random.randn(10, 6).astype(np.float32)
        eeg = np.zeros(10, dtype=bool)
        meg = np.ones(10, dtype=bool)
        out = _normalize_pos(pos.copy(), eeg, meg)
        assert out.shape == (10, 6)
        # Centered + scaled to sqrt(3) RMS (legacy formula)
        assert np.abs(out[:, :3].mean()) < 1e-5
        rms = np.sqrt(3 * np.mean(np.sum(out[:, :3] ** 2, axis=1)))
        assert abs(rms - 1.0) < 1e-4

    def test_per_group_independent(self):
        pos = np.random.randn(12, 6).astype(np.float32)
        eeg = np.zeros(12, dtype=bool)
        eeg[:4] = True
        meg = ~eeg
        out = _normalize_pos(pos.copy(), eeg, meg)
        # EEG group centered independently
        assert np.abs(out[eeg, :3].mean()) < 1e-5
        assert np.abs(out[meg, :3].mean()) < 1e-5


class TestSensortypeWiseNormalize:
    def test_groups_unit_std(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((306, 512)).astype(np.float32)
        eeg = np.zeros(306, dtype=bool)
        mag = np.zeros(306, dtype=bool)
        grad = np.zeros(306, dtype=bool)
        mag[:102] = True
        grad[102:] = True
        out = _sensortype_wise_normalize(data, eeg, mag, grad)
        assert out.dtype == np.float32
        assert np.isnan(out).sum() == 0
        # MAG group: zero mean, ~unit std (eps 1e-13)
        assert abs(out[mag].mean()) < 1e-4
        assert abs(np.std(out[mag]) - 1.0) < 1e-3
        assert abs(out[grad].mean()) < 1e-4
        assert abs(np.std(out[grad]) - 1.0) < 1e-3

    def test_no_mutation_of_input(self):
        data = np.ones((10, 4), dtype=np.float32)
        eeg = np.zeros(10, dtype=bool)
        mag = np.ones(10, dtype=bool)
        grad = np.zeros(10, dtype=bool)
        out = _sensortype_wise_normalize(data, eeg, mag, grad)
        assert np.all(data == 1.0)  # input untouched (copy semantics)
        assert np.all(out == 0.0)  # constant data -> zero after z-score