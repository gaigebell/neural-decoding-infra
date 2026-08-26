"""Train/val/test story splitting for the DRDR dataset.

The paper needs BOTH evaluation dimensions:

- **within-subject decoding** (同被试): per subject, split STORIES into
  train/val/test by ratio. Held-out stories of the SAME subject measure
  same-subject decoding. Splitting by story (never by time step) prevents
  temporal leakage — samples of one story must not span splits.
- **cross-subject generalization** (跨被试泛化): hold out whole SUBJECTS
  as the test set (leave-one-subject-out / LOSO). The model never sees
  any data of the test subjects; decode on them measures generalization
  to new participants.

Config (``data.split``):

.. code-block:: yaml

    split:
      method: ratio        # none | ratio | holdout | explicit
      val_ratio: 0.15      # ratio mode: train gets the remainder
      test_ratio: 0.15
      seed: 42
      test_subjects: null  # holdout mode: subjects to hold out entirely
      train_stories: null  # explicit mode: exact story lists
      val_stories: null
      test_stories: null

Rules:
- ``method: none`` — everything is train (current default; backwards
  compatible with early smoke runs).
- ``method: ratio`` — every subject's stories are shuffled (fixed seed)
  and split by the ratios; ``val_ratio``/``test_ratio`` of 0 skip the
  split; tiny story counts are floored at 1.
- ``method: holdout`` — ``test_subjects`` pairs go entirely to test; the
  remaining subjects are ratio-split into train/val (``test_ratio`` is
  ignored). Re-run with each subject held out for LOSO.
- ``method: explicit`` — exact story lists (e.g. to reproduce the legacy
  ``[42, 12, 6]`` split: pass the story IDs directly).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .drdr import DRDRIndex

logger = logging.getLogger(__name__)


@dataclass
class StorySplit:
    """The three-way partition of (subject, story) pairs."""

    train: list[tuple[int, int]] = field(default_factory=list)
    val: list[tuple[int, int]] = field(default_factory=list)
    test: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "val_stories": [int(st) for st in sorted({st for _, st in self.val})],
            "test_stories": [int(st) for st in sorted({st for _, st in self.test})],
        }


def _split_stories(
    stories: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Ratio-split ONE subject's story list (story granularity)."""
    stories = sorted(set(stories))
    order = rng.permutation(stories)
    n = len(order)
    n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
    n_val = min(n_val, max(n - n_test, 0))
    test = list(order[:n_test])
    val = list(order[n_test : n_test + n_val])
    train = list(order[n_test + n_val :])
    return train, val, test


def split_index(index: DRDRIndex, split_cfg: DictConfig) -> StorySplit:
    """Partition an index's (subject, story) pairs into train/val/test.

    Args:
        index: Discovered DRDR index (already filtered by subjects/stories
            config, if any).
        split_cfg: The ``data.split`` config section.

    Returns:
        A :class:`StorySplit`. ``method=none`` puts everything in train.
    """
    method = str(split_cfg.get("method", "none"))
    pairs = list(index.pairs)

    if method == "none":
        split = StorySplit(train=pairs)
    elif method == "ratio":
        split = _ratio_split(pairs, split_cfg)
    elif method == "holdout":
        split = _holdout_split(pairs, split_cfg)
    elif method == "explicit":
        split = _explicit_split(pairs, split_cfg)
    else:
        raise ValueError(
            f"Unknown split method: {method} (expected none|ratio|holdout|explicit)"
        )

    logger.info("Story split (%s): %s", method, split.summary())
    return split


def _ratio_split(pairs: list[tuple[int, int]], cfg: DictConfig) -> StorySplit:
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    test_ratio = float(cfg.get("test_ratio", 0.15))
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio > 1.0:
        raise ValueError(
            "val_ratio + test_ratio must be within [0, 1], got "
            f"{val_ratio + test_ratio} (train gets the remainder)"
        )
    train_ratio = 1.0 - val_ratio - test_ratio

    by_subject: dict[int, list[tuple[int, int]]] = {}
    for sub, story in pairs:
        by_subject.setdefault(sub, []).append((sub, story))

    out = StorySplit()
    for sub, sub_pairs in sorted(by_subject.items()):
        t, v, te = _split_stories(
            [st for _, st in sub_pairs], train_ratio, val_ratio, test_ratio, rng
        )
        out.train.extend((sub, s) for s in t)
        out.val.extend((sub, s) for s in v)
        out.test.extend((sub, s) for s in te)
    return out


def _holdout_split(pairs: list[tuple[int, int]], cfg: DictConfig) -> StorySplit:
    test_subjects = cfg.get("test_subjects")
    if not test_subjects:
        raise ValueError("holdout split requires data.split.test_subjects")
    test_subjects = {int(s) for s in test_subjects}

    out = StorySplit()
    rest: list[tuple[int, int]] = []
    for pair in pairs:
        (out.test if pair[0] in test_subjects else rest).append(pair)
    # Remaining (train) subjects: ratio-split into train/val only
    inner = _ratio_split(
        rest,
        OmegaConf.create(
            {
                "seed": cfg.get("seed", 42),
                "train_ratio": 1.0 - float(cfg.get("val_ratio", 0.15)),
                "val_ratio": float(cfg.get("val_ratio", 0.15)),
                "test_ratio": 0.0,
            }
        ),
    )
    out.train, out.val = inner.train, inner.val
    return out


def _explicit_split(pairs: list[tuple[int, int]], cfg: DictConfig) -> StorySplit:
    def to_set(key: str) -> set[int] | None:
        val = cfg.get(key)
        return {int(s) for s in val} if val else None

    train_stories, val_stories, test_stories = (
        to_set("train_stories"),
        to_set("val_stories"),
        to_set("test_stories"),
    )
    out = StorySplit()
    for pair in pairs:
        story = pair[1]
        if test_stories is not None and story in test_stories:
            out.test.append(pair)
        elif val_stories is not None and story in val_stories:
            out.val.append(pair)
        elif train_stories is not None and story in train_stories:
            out.train.append(pair)
        else:
            out.train.append(pair)  # unspecified -> train (legacy behavior)
    return out


__all__ = ["StorySplit", "split_index"]
