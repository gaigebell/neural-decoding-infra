"""Unit tests for :mod:`recon.data.split` (story-level train/val/test split)."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from recon.data.drdr import DRDRIndex
from recon.data.split import split_index


def _index(pairs: list[tuple[int, int]]) -> DRDRIndex:
    by_subject: dict[int, list[int]] = {}
    by_story: dict[int, list[int]] = {}
    for s, st in pairs:
        by_subject.setdefault(s, []).append(st)
        by_story.setdefault(st, []).append(s)
    return DRDRIndex(
        processed_root=Path("/tmp"),
        modality="meg",
        subjects=sorted(by_subject),
        stories=sorted(by_story),
        by_subject=by_subject,
        by_story=by_story,
        pairs=pairs,
    )


def _cfg(**kw) -> dict:
    base = {
        "method": "ratio",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42,
        "test_subjects": None,
        "train_stories": None,
        "val_stories": None,
        "test_stories": None,
    }
    base.update(kw)
    return base


class TestMethodNone:
    def test_everything_train(self):
        index = _index([(1, 1), (1, 2), (2, 1)])
        split = split_index(index, OmegaConf.create({"method": "none"}))
        assert len(split.train) == 3
        assert split.val == [] and split.test == []


class TestRatio:
    def test_per_subject_ratio(self):
        # subject 1: stories 1..10; subject 2: stories 11..20
        index = _index([(1, i) for i in range(1, 11)] + [(2, i) for i in range(11, 21)])
        split = split_index(index, OmegaConf.create(_cfg()))
        # 10 stories per subject: 0.15 -> round(1.5) = 2 val + 2 test per subject
        assert len(split.val) == 4
        assert len(split.test) == 4
        assert len(split.train) == 12
        # Both subjects represented in every split
        assert {s for s, _ in split.val} == {1, 2}
        assert {s for s, _ in split.test} == {1, 2}

    def test_seed_reproducible(self):
        index = _index([(1, i) for i in range(1, 61)])
        a = split_index(index, OmegaConf.create(_cfg()))
        b = split_index(index, OmegaConf.create(_cfg()))
        assert a.train == b.train and a.val == b.val and a.test == b.test

    def test_different_seed_differs(self):
        index = _index([(1, i) for i in range(1, 61)])
        a = split_index(index, OmegaConf.create(_cfg(seed=42)))
        b = split_index(index, OmegaConf.create(_cfg(seed=7)))
        assert a.val != b.val

    def test_disjoint_and_complete(self):
        index = _index([(1, i) for i in range(1, 21)])
        split = split_index(index, OmegaConf.create(_cfg()))
        all_pairs = split.train + split.val + split.test
        assert len(set(all_pairs)) == 20  # no overlap, nothing lost

    def test_val_test_ratios_must_fit(self):
        index = _index([(1, i) for i in range(1, 11)])
        with pytest.raises(ValueError, match="within"):
            split_index(index, OmegaConf.create(_cfg(val_ratio=0.6, test_ratio=0.6)))

    def test_zero_val_ratio(self):
        index = _index([(1, i) for i in range(1, 11)])
        split = split_index(index, OmegaConf.create(_cfg(val_ratio=0.0)))
        assert split.val == []

    def test_small_story_count_floor(self):
        # 3 stories -> test_ratio 0.15 floors to 1 test story
        index = _index([(1, 1), (1, 2), (1, 3)])
        split = split_index(index, OmegaConf.create(_cfg()))
        assert len(split.test) == 1
        assert len(split.val) == 1
        assert len(split.train) == 1


class TestHoldout:
    def test_holdout_subjects(self):
        pairs = [(s, i) for s in (1, 2, 3) for i in range(1, 11)]
        index = _index(pairs)
        split = split_index(
            index,
            OmegaConf.create(_cfg(method="holdout", test_subjects=[3])),
        )
        # All of subject 3's stories in test; nothing of subject 3 in train/val
        assert {s for s, _ in split.test} == {3}
        assert len(split.test) == 10
        assert 3 not in {s for s, _ in split.train}
        assert 3 not in {s for s, _ in split.val}
        assert {s for s, _ in split.train} == {1, 2}

    def test_holdout_requires_subjects(self):
        index = _index([(1, i) for i in range(1, 11)])
        with pytest.raises(ValueError, match="test_subjects"):
            split_index(index, OmegaConf.create(_cfg(method="holdout")))


class TestExplicit:
    def test_explicit_lists(self):
        index = _index([(1, i) for i in range(1, 11)])
        split = split_index(
            index,
            OmegaConf.create(
                _cfg(
                    method="explicit",
                    train_stories=[1, 2, 3, 4, 5, 6],
                    val_stories=[7, 8],
                    test_stories=[9, 10],
                )
            ),
        )
        assert {st for _, st in split.train} == {1, 2, 3, 4, 5, 6}
        assert {st for _, st in split.val} == {7, 8}
        assert {st for _, st in split.test} == {9, 10}

    def test_unknown_method_raises(self):
        index = _index([(1, 1)])
        with pytest.raises(ValueError, match="Unknown split method"):
            split_index(index, OmegaConf.create({"method": "bogus"}))

    def test_summary(self):
        index = _index([(1, i) for i in range(1, 21)])
        split = split_index(index, OmegaConf.create(_cfg()))
        s = split.summary()
        assert s["n_train"] + s["n_val"] + s["n_test"] == 20
        assert "val_stories" in s and "test_stories" in s
