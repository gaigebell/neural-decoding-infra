"""Evaluation metrics for the recon pipeline.

Provides standard metrics for character-level Chinese decoding:
    - CRR: Character Recognition Rate (top-1)
    - CER: Character Error Rate (Levenshtein-based)
    - Top-k: Top-k accuracy
    - Perplexity: language model perplexity (optional)

See ``docs/guides/07-run-evaluation.md`` for usage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────── Individual metrics ─────────────────────


def character_recognition_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute Character Recognition Rate (top-1, position-wise).

    Compares each predicted character at each position with the reference
    at the same position. Returns fraction of correctly matched characters
    in [0, 1].

    Args:
        predictions: List of predicted strings.
        references: List of reference (ground-truth) strings.

    Returns:
        CRR in [0, 1]. Higher is better.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    total = 0
    correct = 0
    for pred, ref in zip(predictions, references):
        for p, r in zip(pred, ref):
            total += 1
            if p == r:
                correct += 1
    return correct / total if total > 0 else 0.0


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute Character Error Rate using Levenshtein distance.

    CER = sum(edit_distance(pred, ref)) / sum(len(ref))

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        CER in [0, +inf). Lower is better. CER > 1 means more edits
        than characters in reference.
    """
    total_distance = 0
    total_ref_len = 0
    for pred, ref in zip(predictions, references):
        total_distance += _levenshtein(pred, ref)
        total_ref_len += len(ref)
    return total_distance / total_ref_len if total_ref_len > 0 else 0.0


def topk_accuracy(
    predictions: Sequence[Sequence[str]],
    references: Sequence[str],
    k: int = 5,
) -> float:
    """Compute top-k accuracy: fraction of references that appear in
    the top-k predictions for that position.

    Args:
        predictions: For each sample, a list of k candidate strings.
        references: List of ground-truth strings.

    Returns:
        Top-k accuracy in [0, 1].
    """
    if len(predictions) != len(references):
        raise ValueError("Length mismatch")
    total = 0
    correct = 0
    for cands, ref in zip(predictions, references):
        for r in ref:
            total += 1
            if any(r in cand for cand in cands[:k]):
                correct += 1
    return correct / total if total > 0 else 0.0


def perplexity(log_probs: Sequence[float]) -> float:
    """Compute perplexity from a sequence of log probabilities.

    perplexity = exp(-mean(log_probs))

    Args:
        log_probs: Per-token log probabilities (natural log).

    Returns:
        Perplexity. Lower is better.
    """
    if not log_probs:
        return float("inf")
    return float(np.exp(-np.mean(log_probs)))


# ───────────────────── Levenshtein ─────────────────────


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Pure-Python implementation, no external dependency. For long strings
    consider installing `python-Levenshtein` for speed.
    """
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# ───────────────────── Aggregator ─────────────────────


@dataclass
class EvalReport:
    """Aggregated evaluation metrics for a set of predictions."""

    crr: float = 0.0
    cer: float = 0.0
    top5: float = 0.0
    top10: float = 0.0
    perplexity: float = 0.0
    n_samples: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Convert to a flat dict for serialization."""
        d = {
            "crr": self.crr,
            "cer": self.cer,
            "top5": self.top5,
            "top10": self.top10,
            "perplexity": self.perplexity,
            "n_samples": self.n_samples,
        }
        d.update(self.extra)
        return d


class Evaluator:
    """Compute evaluation metrics for a set of predictions.

    Usage:
        >>> ev = Evaluator()
        >>> ev.add(pred="你好", ref="你号")
        >>> ev.add(pred="世界", ref="世界", topk_candidates=["世界", "世贷", "世太"])
        >>> report = ev.compute()
        >>> print(report.crr, report.cer)
    """

    def __init__(self) -> None:
        self._preds: list[str] = []
        self._refs: list[str] = []
        self._topk: list[list[str]] = []
        self._log_probs: list[float] = []

    def add(
        self,
        pred: str,
        ref: str,
        topk_candidates: list[str] | None = None,
        log_prob: float | None = None,
    ) -> None:
        """Add one (prediction, reference) pair.

        Args:
            pred: Predicted string.
            ref: Reference string.
            topk_candidates: Optional list of top-k candidate strings.
            log_prob: Optional LM log probability (for perplexity).
        """
        self._preds.append(pred)
        self._refs.append(ref)
        if topk_candidates is not None:
            self._topk.append(topk_candidates)
        if log_prob is not None:
            self._log_probs.append(log_prob)

    def compute(self) -> EvalReport:
        """Compute all metrics over accumulated samples."""
        report = EvalReport(n_samples=len(self._preds))
        if not self._preds:
            return report
        report.crr = character_recognition_rate(self._preds, self._refs)
        report.cer = character_error_rate(self._preds, self._refs)
        if self._topk:
            report.top5 = topk_accuracy(self._topk, self._refs, k=5)
            report.top10 = topk_accuracy(self._topk, self._refs, k=10)
        if self._log_probs:
            report.perplexity = perplexity(self._log_probs)
        return report

    def reset(self) -> None:
        """Clear all accumulated samples."""
        self._preds.clear()
        self._refs.clear()
        self._topk.clear()
        self._log_probs.clear()


__all__ = [
    "EvalReport",
    "Evaluator",
    "character_error_rate",
    "character_recognition_rate",
    "perplexity",
    "topk_accuracy",
]