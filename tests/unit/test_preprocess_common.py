"""Unit tests for :mod:`recon.preprocessing.common`."""
from __future__ import annotations

import numpy as np
import pytest

from recon.preprocessing.common import (
    find_duplicate_indices,
    lanczosfun,
    make_delayed,
    sidecar_valid,
    write_sidecar,
    zscore_2d,
)


class TestZscore2d:
    def test_known_values(self):
        x = np.array([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])
        z = zscore_2d(x)
        np.testing.assert_allclose(z[:, 0].mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(z[:, 0].std(), 1.0, atol=1e-5)
        np.testing.assert_allclose(z[:, 1].std(), 1.0, atol=1e-5)
        # Legacy epsilon: std = std + 1e-10 in the denominator
        assert z.dtype == x.dtype

    def test_constant_column_is_zero(self):
        x = np.ones((5, 2)) * 7.0
        z = zscore_2d(x)
        assert np.allclose(z, 0.0, atol=1e-3)  # 0/(1e-10+0)


class TestMakeDelayed:
    def test_shapes_and_delays(self):
        stim = np.arange(10.0).reshape(10, 1)
        d = make_delayed(stim, [1, 2, 3, 4])
        assert d.shape == (10, 4)
        # delay=1: row t is stim[t-1]
        np.testing.assert_array_equal(d[5, 0], stim[4, 0])
        np.testing.assert_array_equal(d[0, 0], 0.0)  # zero-padded head
        # delay=2: row t is stim[t-2]
        np.testing.assert_array_equal(d[5, 1], stim[3, 0])

    def test_zero_delay_is_copy(self):
        stim = np.arange(6.0).reshape(3, 2)
        d = make_delayed(stim, [0])
        np.testing.assert_array_equal(d, stim)


class TestFindDuplicates:
    def test_duplicates(self):
        assert find_duplicate_indices([1.0, 2.0, 1.0, 3.0, 2.0]) == [2, 4]

    def test_no_duplicates(self):
        assert find_duplicate_indices([1.0, 2.0, 3.0]) == []


class TestLanczos:
    def test_kernel_peak(self):
        v = lanczosfun(2.5, np.array([0.0]))
        assert v[0] == 1.0

    def test_kernel_zero_outside_window(self):
        v = lanczosfun(2.5, np.array([5.0, -5.0]), window=3)
        assert np.all(v == 0.0)


class TestSidecar:
    def test_roundtrip_valid(self, tmp_path):
        inp = tmp_path / "in.npy"
        inp.write_bytes(b"0123456789" * 100)
        art = tmp_path / "out.npy"
        art.write_bytes(b"artifact")
        params = {"stage": "test", "x": 1}
        write_sidecar(art, {"in": inp}, params)
        assert sidecar_valid(art, {"in": inp}, params)

    def test_param_change_invalidates(self, tmp_path):
        inp = tmp_path / "in.npy"
        inp.write_bytes(b"data")
        art = tmp_path / "out.npy"
        art.write_bytes(b"a")
        write_sidecar(art, {"in": inp}, {"stage": "t", "x": 1})
        assert not sidecar_valid(art, {"in": inp}, {"stage": "t", "x": 2})

    def test_input_change_invalidates(self, tmp_path):
        inp = tmp_path / "in.npy"
        inp.write_bytes(b"v1")
        art = tmp_path / "out.npy"
        art.write_bytes(b"a")
        write_sidecar(art, {"in": inp}, {"stage": "t"})
        inp.write_bytes(b"v2-changed")
        assert not sidecar_valid(art, {"in": inp}, {"stage": "t"})

    def test_missing_artifact_invalid(self, tmp_path):
        inp = tmp_path / "in.npy"
        inp.write_bytes(b"data")
        art = tmp_path / "ghost.npy"
        art.write_bytes(b"artifact")  # caller writes the artifact
        write_sidecar(art, {"in": inp}, {"stage": "t"})
        art.unlink()  # simulate a deleted artifact
        assert not sidecar_valid(art, {"in": inp}, {"stage": "t"})
