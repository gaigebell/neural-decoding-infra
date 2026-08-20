"""Unit tests for :mod:`recon.engine.evaluator`."""
from __future__ import annotations

import pytest

from recon.engine.evaluator import (
    Evaluator,
    character_error_rate,
    character_recognition_rate,
    perplexity,
    topk_accuracy,
)


# ───────────────────── Individual metrics ─────────────────────


class TestCharacterRecognitionRate:
    def test_perfect(self):
        assert character_recognition_rate(["abc", "de"], ["abc", "de"]) == 1.0

    def test_all_wrong(self):
        assert character_recognition_rate(["xyz", "fg"], ["abc", "de"]) == 0.0

    def test_partial(self):
        # 3 correct out of 5 characters
        assert character_recognition_rate(["abc", "fg"], ["abc", "de"]) == pytest.approx(0.6, abs=0.01)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            character_recognition_rate(["a"], ["a", "b"])

    def test_empty(self):
        assert character_recognition_rate([], []) == 0.0


class TestCharacterErrorRate:
    def test_perfect(self):
        assert character_error_rate(["abc", "de"], ["abc", "de"]) == 0.0

    def test_one_insertion(self):
        # "abc" vs "abcd" = 1 insertion / 3 ref chars = 1/3 (CER normalizes by ref length)
        assert character_error_rate(["abcd"], ["abc"]) == pytest.approx(1 / 3, abs=0.01)

    def test_one_deletion(self):
        # "abcd" vs "abc" = 1 edit / 3 ref = 0.333
        assert character_error_rate(["abcd"], ["abc"]) == pytest.approx(1 / 3, abs=0.01)

    def test_completely_different(self):
        cer = character_error_rate(["xy"], ["ab"])
        assert cer > 0.5  # 2 edits / 2 ref


class TestTopKAccuracy:
    def test_top1_match(self):
        assert topk_accuracy([["a", "b"]], ["a"], k=1) == 1.0

    def test_top1_miss_top2_hit(self):
        # Reference is 'a'; top-1 is 'b', top-2 is 'a' — k=2 should hit
        assert topk_accuracy([["b", "a"]], ["a"], k=1) == 0.0
        assert topk_accuracy([["b", "a"]], ["a"], k=2) == 1.0

    def test_topk_string_match(self):
        # Multi-character refs; 'a' in 'xyz' (string containment)
        assert topk_accuracy([["xy", "a"]], ["a"], k=2) == 1.0


class TestPerplexity:
    def test_perfect_prediction(self):
        # log_prob = 0 → perplexity = 1
        assert perplexity([0.0, 0.0]) == pytest.approx(1.0)

    def test_high_uncertainty(self):
        # log_prob = -log(0.5) ≈ -0.69 → perplexity ≈ 2
        import math
        log_half = math.log(0.5)
        assert perplexity([log_half, log_half]) == pytest.approx(2.0, abs=0.01)

    def test_empty_returns_inf(self):
        import math
        assert perplexity([]) == math.inf


# ───────────────────── Aggregator ─────────────────────


class TestEvaluator:
    """Tests for the Evaluator aggregator."""

    def test_empty_compute(self):
        ev = Evaluator()
        report = ev.compute()
        assert report.n_samples == 0
        assert report.crr == 0.0
        assert report.cer == 0.0

    def test_add_and_compute(self):
        ev = Evaluator()
        ev.add(pred="abc", ref="abc")  # perfect
        ev.add(pred="xyz", ref="abc")  # 0/3
        report = ev.compute()
        assert report.n_samples == 2
        assert report.crr == 0.5
        assert report.cer > 0  # xyz vs abc is far

    def test_reset_clears(self):
        ev = Evaluator()
        ev.add(pred="abc", ref="abc")
        ev.reset()
        report = ev.compute()
        assert report.n_samples == 0

    def test_to_dict(self):
        ev = Evaluator()
        ev.add(pred="abc", ref="abc", topk_candidates=["abc", "xyz"])
        report = ev.compute()
        d = report.to_dict()
        assert "crr" in d
        assert "cer" in d
        assert "n_samples" in d