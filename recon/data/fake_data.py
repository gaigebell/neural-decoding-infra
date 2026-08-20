"""Generate fake data for tests, smoke tests, and demos.

Critical principle: ``fake_data`` is the FIRST thing that runs when
someone clones this repo. It must:

- Run on CPU (no GPU required).
- Run with no extras installed.
- Produce data that passes pydantic schema validation.
- Produce data that lets the model forward pass run end-to-end.

See:
    - docs/standards/04-testing.md
    - docs/guides/04-run-training.md (smoke tier 0)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

from .drdr import DEFAULT_DELAY_WEIGHTS, weight_delays
from .schema import (
    BrainOmniSample,
    BrainStimPair,
    Float32Array,
    MEGChunkedSample,
    MEGSample,
    StimSample,
    fMRISample,
)

logger = logging.getLogger(__name__)


# ───────────────────── Generation helpers ─────────────────────


def fake_stim(
    rng: np.random.Generator,
    semantic_dim: int = 768,
    n_delays: int = 4,
) -> Float32Array:
    """Generate a fake semantic feature vector.

    Args:
        rng: numpy random generator.
        semantic_dim: Dimension of each delay's feature.
        n_delays: Number of delay values to concatenate.

    Returns:
        1D float32 array of shape (semantic_dim * n_delays,).
    """
    # Real zstim values are roughly in [-15, 15] after layer 10 of GPT-2
    return rng.standard_normal(semantic_dim * n_delays).astype(np.float32) * 3.0


def fake_meg(
    rng: np.random.Generator,
    n_channels: int = 306,
    n_time: int = 256,
    story_id: int = 1,
    subject_id: int = 1,
) -> MEGSample:
    """Generate a fake MEG sample.

    Args:
        rng: numpy random generator.
        n_channels: Number of MEG channels.
        n_time: Time samples per window.
        story_id: Story ID for the sample.
        subject_id: Subject ID for the sample.

    Returns:
        A validated MEGSample.
    """
    # Real MEG after z-score is roughly in [-5, 5]
    x = rng.standard_normal((n_channels, n_time)).astype(np.float32)
    pos = rng.standard_normal((n_channels, 6)).astype(np.float32)
    # 102 MAG + 204 GRAD, no EEG in our cluster
    sensor_type = np.array(
        [1] * (n_channels // 3) + [2] * (n_channels - n_channels // 3),
        dtype=np.int32,
    )
    return MEGSample(
        x=x,
        pos=pos,
        sensor_type=sensor_type,
        story_id=story_id,
        subject_id=subject_id,
        word_times=np.array([0.0], dtype=np.float32),
    )


def fake_meg_chunked(
    rng: np.random.Generator,
    n_context: int = 5,
    n_channels: int = 306,
    n_time: int = 256,
    story_id: int = 1,
    subject_id: int = 1,
) -> MEGChunkedSample:
    """Generate a fake chunked MEG sample with context window.

    Args:
        rng: numpy random generator.
        n_context: Number of past time steps to include.
        n_channels: Number of MEG channels.
        n_time: Time samples per step.
        story_id: Story ID.
        subject_id: Subject ID.

    Returns:
        A validated MEGChunkedSample.
    """
    x = rng.standard_normal((n_context, n_channels, n_time)).astype(np.float32)
    pos = rng.standard_normal((n_channels, 6)).astype(np.float32)
    sensor_type = np.array(
        [1] * (n_channels // 3) + [2] * (n_channels - n_channels // 3),
        dtype=np.int32,
    )
    return MEGChunkedSample(
        x=x,
        pos=pos,
        sensor_type=sensor_type,
        context_len=n_context,
        story_id=story_id,
        subject_id=subject_id,
        word_times=np.array([0.0], dtype=np.float32),
    )


def fake_fmri(
    rng: np.random.Generator,
    shape: tuple[int, int, int] = (53, 63, 52),
    story_id: int = 1,
    subject_id: int = 1,
) -> fMRISample:
    """Generate a fake fMRI sample.

    Args:
        rng: numpy random generator.
        shape: (X, Y, Z) volume shape. Default is MNI 2mm.
        story_id: Story ID.
        subject_id: Subject ID.

    Returns:
        A validated fMRISample.
    """
    volume = rng.standard_normal(shape).astype(np.float32) * 0.5
    # A spherical-ish brain mask (same shape as volume)
    cx, cy, cz = (s // 2 for s in shape)
    xx, yy, zz = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < (min(shape) // 2) ** 2
    return fMRISample(
        volume=volume,
        mask=mask.astype(bool),
        story_id=story_id,
        subject_id=subject_id,
    )


def fake_brainomni(
    rng: np.random.Generator,
    n_neurons: int = 306,
    seq_len: int = 1,
    n_dim: int = 1024,
    story_id: int = 1,
    subject_id: int = 1,
) -> BrainOmniSample:
    """Generate a fake BrainOmni-encoded sample.

    Args:
        rng: numpy random generator.
        n_neurons: Number of "neurons" in the feature tensor.
        seq_len: Sequence length in the feature tensor.
        n_dim: Feature dimension.
        story_id: Story ID.
        subject_id: Subject ID.

    Returns:
        A validated BrainOmniSample.
    """
    features = rng.standard_normal((n_neurons, seq_len, n_dim)).astype(np.float32)
    return BrainOmniSample(
        features=features, subject_id=subject_id, story_id=story_id
    )


def fake_pair_meg(
    rng: np.random.Generator,
    story_id: int = 1,
    subject_id: int = 1,
    n_context: int = 0,
) -> BrainStimPair:
    """Generate a fake (MEG, stimulus) pair.

    Args:
        rng: numpy random generator.
        story_id: Story ID.
        subject_id: Subject ID.
        n_context: If > 0, use chunked MEG with this context length.

    Returns:
        A validated BrainStimPair.
    """
    if n_context > 0:
        brain = fake_meg_chunked(rng, n_context=n_context, story_id=story_id, subject_id=subject_id)
    else:
        brain = fake_meg(rng, story_id=story_id, subject_id=subject_id)
    raw = fake_stim(rng)
    stim = StimSample(
        zstim=weight_delays(raw)[0],
        zstim_raw=raw,
        story_id=story_id,
        subject_id=subject_id,
    )
    return BrainStimPair(brain=brain, stim=stim)


def fake_pair_fmri(
    rng: np.random.Generator,
    story_id: int = 1,
    subject_id: int = 1,
) -> BrainStimPair:
    """Generate a fake (fMRI, stimulus) pair."""
    brain = fake_fmri(rng, story_id=story_id, subject_id=subject_id)
    raw = fake_stim(rng)
    stim = StimSample(
        zstim=weight_delays(raw)[0],
        zstim_raw=raw,
        story_id=story_id,
        subject_id=subject_id,
    )
    return BrainStimPair(brain=brain, stim=stim)


# ───────────────────── CLI ─────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fake data for testing the recon pipeline."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./fake_data",
        help="Output directory for fake data files.",
    )
    parser.add_argument(
        "--n-meg",
        type=int,
        default=20,
        help="Number of fake MEG samples to generate.",
    )
    parser.add_argument(
        "--n-fmri",
        type=int,
        default=10,
        help="Number of fake fMRI samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def main() -> None:
    """CLI entry point: ``python -m recon.data.fake_data``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    summary: dict[str, int] = {}

    # MEG
    if args.n_meg > 0:
        for i in range(args.n_meg):
            pair = fake_pair_meg(rng, story_id=(i % 10) + 1, subject_id=((i // 10) % 12) + 1)
            np.savez(
                output_dir / f"meg_{i:04d}.npz",
                x=pair.brain.x,
                pos=pair.brain.pos,
                sensor_type=pair.brain.sensor_type,
                zstim=pair.stim.zstim,
                story_id=pair.brain.story_id,
                subject_id=pair.brain.subject_id,
            )
        summary["meg"] = args.n_meg

    # fMRI
    if args.n_fmri > 0:
        for i in range(args.n_fmri):
            pair = fake_pair_fmri(rng, story_id=(i % 10) + 1, subject_id=((i // 10) % 12) + 1)
            np.savez(
                output_dir / f"fmri_{i:04d}.npz",
                volume=pair.brain.volume,
                mask=pair.brain.mask,
                zstim=pair.stim.zstim,
                story_id=pair.brain.story_id,
                subject_id=pair.brain.subject_id,
            )
        summary["fmri"] = args.n_fmri

    # Write manifest
    manifest = {
        "seed": args.seed,
        "samples": summary,
        "format": "npz per sample (meg_*.npz, fmri_*.npz)",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Generated fake data in %s:", output_dir)
    for kind, n in summary.items():
        logger.info("  %s: %d samples", kind, n)
    logger.info("Manifest: %s/manifest.json", output_dir)


if __name__ == "__main__":
    main()