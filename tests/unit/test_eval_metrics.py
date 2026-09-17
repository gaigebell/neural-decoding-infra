"""Unit tests for the eval CLI's metric helpers (dual 口径, completeness)."""
from __future__ import annotations

from pathlib import Path

import pytest

from recon.cli.eval import _chinese_only, _eval_one

# The tested metric subsets don't touch the LM; any path works.
_DUMMY = Path("unused")


class TestChineseOnly:
    def test_filters_punct_and_digits(self):
        assert _chinese_only("今天，天气123很好！") == "今天天气很好"

    def test_empty(self):
        assert _chinese_only("，。！123") == ""


class TestEvalOne:
    def test_dual口径_full_and_chinese(self):
        decoded = ["今天，天气好", "我 们，爱"]   # note: decoded may contain junk
        refs = ["今天，天气好", "我们爱"]
        out = _eval_one(decoded, refs, ["crr", "cer"], _DUMMY)
        assert "crr_full" in out and "cer_full" in out
        assert "crr_chinese" in out and "cer_chinese" in out
        # perfect chinese-only match
        assert out["crr_chinese"] == 1.0

    def test_perfect_full(self):
        out = _eval_one(["abc"], ["abc"], ["crr"], _DUMMY)
        assert out["crr_full"] == 1.0
        assert out["cer_full"] == 0.0

    def test_length_and_unique_ratio(self):
        decoded = ["今天今天今天"]  # 6 chars, 2 unique
        refs = ["今天"]            # 2 chars
        out = _eval_one(decoded, refs, ["length_ratio", "unique_ratio"], _DUMMY)
        assert out["length_ratio"] == pytest.approx(3.0)
        assert out["unique_ratio_decoded"] == pytest.approx(1 / 3)

    def test_empty_inputs(self):
        out = _eval_one([], [], ["crr", "length_ratio"], _DUMMY)
        assert out["crr_full"] == 0.0
        assert out["length_ratio"] == 0.0
