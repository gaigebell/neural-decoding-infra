"""Unit tests for ``recon.data.fake_data``.

Validates that fake data generators produce schema-valid samples and
that the CLI runs without errors.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from recon.data.fake_data import (
    fake_brainomni,
    fake_fmri,
    fake_meg,
    fake_meg_chunked,
    fake_pair_fmri,
    fake_pair_meg,
    fake_stim,
)
from recon.data.schema import (
    BrainOmniSample,
    MEGChunkedSample,
    MEGSample,
    StimSample,
    fMRISample,
)


# ───────────────────── Generator functions ─────────────────────


class TestGenerators:
    """Tests for the individual fake_* generators."""

    def test_fake_stim_shape(self):
        rng = np.random.default_rng(0)
        zstim = fake_stim(rng)
        assert zstim.shape == (4 * 768,)
        assert zstim.dtype == np.float32

    def test_fake_stim_custom_dim(self):
        rng = np.random.default_rng(0)
        zstim = fake_stim(rng, semantic_dim=128, n_delays=2)
        assert zstim.shape == (256,)

    def test_fake_meg_default(self):
        rng = np.random.default_rng(0)
        s = fake_meg(rng)
        assert isinstance(s, MEGSample)
        assert s.x.shape == (306, 256)
        assert s.pos.shape == (306, 6)
        assert s.sensor_type.shape == (306,)

    def test_fake_meg_custom_shape(self):
        rng = np.random.default_rng(0)
        s = fake_meg(rng, n_channels=64, n_time=100)
        assert s.x.shape == (64, 100)

    def test_fake_meg_chunked_default(self):
        rng = np.random.default_rng(0)
        s = fake_meg_chunked(rng)
        assert isinstance(s, MEGChunkedSample)
        assert s.x.shape == (5, 306, 256)
        assert s.context_len == 5

    def test_fake_meg_chunked_custom(self):
        rng = np.random.default_rng(0)
        s = fake_meg_chunked(rng, n_context=3, n_channels=64)
        assert s.x.shape == (3, 64, 256)
        assert s.context_len == 3

    def test_fake_fmri_default(self):
        rng = np.random.default_rng(0)
        s = fake_fmri(rng)
        assert isinstance(s, fMRISample)
        assert s.volume.shape == (53, 63, 52)
        assert s.mask.shape == (53, 63, 52)
        assert s.mask.dtype == bool

    def test_fake_fmri_custom_shape(self):
        rng = np.random.default_rng(0)
        s = fake_fmri(rng, shape=(10, 12, 8))
        assert s.volume.shape == (10, 12, 8)

    def test_fake_brainomni(self):
        rng = np.random.default_rng(0)
        s = fake_brainomni(rng)
        assert isinstance(s, BrainOmniSample)
        assert s.features.shape == (306, 1, 1024)

    def test_fake_pair_meg(self):
        rng = np.random.default_rng(0)
        pair = fake_pair_meg(rng, story_id=5, subject_id=3)
        assert pair.brain.story_id == 5
        assert pair.brain.subject_id == 3
        assert pair.stim.story_id == 5
        assert isinstance(pair.brain, MEGSample)

    def test_fake_pair_meg_chunked(self):
        rng = np.random.default_rng(0)
        pair = fake_pair_meg(rng, n_context=5)
        assert isinstance(pair.brain, MEGChunkedSample)
        assert pair.brain.context_len == 5

    def test_fake_pair_fmri(self):
        rng = np.random.default_rng(0)
        pair = fake_pair_fmri(rng, story_id=5, subject_id=3)
        assert isinstance(pair.brain, fMRISample)
        assert pair.brain.story_id == 5


# ───────────────────── Reproducibility ─────────────────────


class TestReproducibility:
    """Same seed should produce same data."""

    def test_meg_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = fake_meg(rng1)
        s2 = fake_meg(rng2)
        np.testing.assert_array_equal(s1.x, s2.x)
        np.testing.assert_array_equal(s1.pos, s2.pos)

    def test_fmri_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = fake_fmri(rng1)
        s2 = fake_fmri(rng2)
        np.testing.assert_array_equal(s1.volume, s2.volume)
        np.testing.assert_array_equal(s1.mask, s2.mask)


# ───────────────────── CLI ─────────────────────


class TestCLI:
    """Tests for the ``python -m recon.data.fake_data`` CLI."""

    def test_cli_runs_and_writes_files(self, tmp_path: Path):
        """CLI should generate files in the output dir."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "recon.data.fake_data",
                "--output",
                str(tmp_path / "fake"),
                "--n-meg",
                "5",
                "--n-fmri",
                "3",
                "--seed",
                "7",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        out_dir = tmp_path / "fake"
        assert out_dir.exists()
        # Should have 5 meg + 3 fmri files
        assert len(list(out_dir.glob("meg_*.npz"))) == 5
        assert len(list(out_dir.glob("fmri_*.npz"))) == 3
        # Should have manifest
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["seed"] == 7
        assert manifest["samples"]["meg"] == 5
        assert manifest["samples"]["fmri"] == 3

    def test_cli_generated_files_loadable(self, tmp_path: Path):
        """Files written by CLI should be loadable as npz."""
        subprocess.run(
            [
                sys.executable, "-m", "recon.data.fake_data",
                "--output", str(tmp_path / "fake"),
                "--n-meg", "2", "--n-fmri", "2",
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
        for f in (tmp_path / "fake").glob("meg_*.npz"):
            data = np.load(f)
            assert "x" in data
            assert "zstim" in data
            # x is saved channels-first: (n_channels, n_time)
            assert data["x"].shape[0] == 306