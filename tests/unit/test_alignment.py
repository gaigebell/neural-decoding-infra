"""Unit tests for :mod:`recon.decoders.alignment` (D1-A)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from recon.decoders.alignment import char_grid, decode_input_rows


def _write_mat(tmp_path, chars: list[str], ends: list[float]) -> str:
    p = tmp_path / "story_1_char_time.mat"
    # fixed-width string array (loadmat returns plain strings, like the
    # real BIDS mats)
    cell = np.array(chars, dtype="<U10").reshape(1, -1)
    savemat(str(p), {"char": cell, "end": np.array(ends).reshape(1, -1)})
    return str(p)


class TestCharGrid:
    def test_basic_mapping(self, tmp_path):
        mat = _write_mat(tmp_path, ["我", "们", "爱"], [12.0, 12.4, 12.8])
        chars, rows = char_grid(mat)
        assert chars == ["我", "们", "爱"]
        np.testing.assert_array_equal(rows, [0, 1, 2])  # (t-12)/0.4

    def test_rounding_to_nearest_row(self, tmp_path):
        mat = _write_mat(tmp_path, ["a", "b"], [12.1, 12.3])  # 0.25 -> 0, 0.75 -> 1
        _, rows = char_grid(mat)
        np.testing.assert_array_equal(rows, [0, 1])

    def test_drops_out_of_grid(self, tmp_path):
        mat = _write_mat(tmp_path, ["a", "b", "c"], [11.0, 12.4, 100.0])
        chars, rows = char_grid(mat, n_rows=50)
        assert chars == ["b"]
        np.testing.assert_array_equal(rows, [1])

    def test_eliminate(self, tmp_path):
        mat = _write_mat(tmp_path, ["a", "eng", "b"], [12.0, 12.4, 12.8])
        chars, rows = char_grid(mat, eliminate=[1])
        assert chars == ["a", "b"]
        np.testing.assert_array_equal(rows, [0, 2])

    def test_multi_char_digit_token_kept(self, tmp_path):
        mat = _write_mat(tmp_path, ["1600", "年"], [12.0, 12.4])
        chars, _ = char_grid(mat)
        assert chars == ["1600", "年"]  # digit group is ONE token


class TestDecodeInputRows:
    def test_anchor_shift(self):
        rows = np.array([0, 5, 10])
        anchors = decode_input_rows(rows, n_context=5)
        np.testing.assert_array_equal(anchors, [-4, 1, 6])

    def test_negative_anchors_allowed(self):
        # early chars anchor before the story start (zero-padded at decode)
        rows = np.array([0, 1])
        anchors = decode_input_rows(rows, n_context=5)
        assert anchors.min() < 0
