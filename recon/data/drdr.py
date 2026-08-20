"""Adapter for the DRDR (Decoding Reading) Chinese narrative dataset.

This is the lab's primary dataset (OpenNeuro ds004078): MEG and fMRI
recordings from 12 subjects reading 60 Chinese stories. See
``docs/research/02-data-card.md`` for the full description.

Preprocessed data layout (``processed_root``; local ``E:/results``,
cluster ``/home/test/reconstruction/results``)::

    processed_root/
    ├── MEG/
    │   ├── zresp/zresp{sub}_{story}.npy               (T, 306)        float32
    │   ├── zresp/zresp{sub}_{story}_context_5.npy     (T, 5, 306)     float32
    │   ├── zresp/zresp{sub}_{story}_chunked.npy       (T, 5, 306)     float32
    │   └── zstim/sub{sub}_zstim_{layer}_story{story}.npy  (T, 4*768)  float64
    ├── zresp/cube/zresp{sub}_{story}.npy              (T, 91, 109, 91) float32
    ├── zstim/sub{sub}_zstim_{layer}_story{story}.npy  (T, 4*768)      float64
    └── mask.nii                                       (91, 109, 91)   int16

Conventions ported verbatim from the original pipeline
(``datasetbuilderMEG.py`` / ``datasetbuilder.py``):

- ``zstim`` rows are 4 concatenated 768-dim delay vectors (GPT-2 layer 10).
  The training target is the delay-weighted combination
  ``zstim @ weights`` with ``weights = [0.1, 0.7, 0.5, 0.3]`` (see
  ``weight_delays``).
- MEG context variant: the model input at stimulus step ``t`` is the
  response context window from step ``t - (n_context - 1)``, zero-padded
  at the start (``load_story_data`` in the original code).

Files are opened with ``mmap_mode="r"`` so stories are page-loaded on
demand rather than copied into RAM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .schema import (
    MEGChunkedSample,
    MEGSample,
    StimSample,
    fMRISample,
)

logger = logging.getLogger(__name__)

# Delay weights for combining the 4 concatenated GPT-2 delays into the
# training target (original pipeline: np.array([0.1, 0.7, 0.5, 0.3])).
DEFAULT_DELAY_WEIGHTS = (0.1, 0.7, 0.5, 0.3)
SEMANTIC_DIM = 768
N_DELAYS = 4

# fMRI cube dimensions in the preprocessed data (mask.nii matches)
FMRI_CUBE_SHAPE = (91, 109, 91)


# ───────────────────── Discovery ─────────────────────


@dataclass
class DRDRIndex:
    """Index of all valid (subject, story) pairs in the dataset."""

    processed_root: Path
    modality: str
    subjects: list[int] = field(default_factory=list)
    stories: list[int] = field(default_factory=list)
    by_subject: dict[int, list[int]] = field(default_factory=dict)
    by_story: dict[int, list[int]] = field(default_factory=dict)
    pairs: list[tuple[int, int]] = field(default_factory=list)

    def n_subjects(self) -> int:
        return len(self.subjects)

    def n_stories(self) -> int:
        return len(self.stories)

    def n_pairs(self) -> int:
        return len(self.pairs)


def discover_drdr(
    processed_root: str | Path,
    modality: Literal["meg", "fmri"] = "meg",
    max_subjects: int = 12,
    max_stories: int = 60,
    layer: int = 10,
) -> DRDRIndex:
    """Discover all valid (subject, story) pairs in the preprocessed data.

    A pair is valid only if BOTH the response and the stimulus files exist
    (the original pipeline dropped stories missing one side at load time;
    we drop them at discovery).

    Args:
        processed_root: Root of the preprocessed results, e.g. ``E:/results``
            or ``/home/test/reconstruction/results``.
        modality: ``"meg"`` scans ``MEG/zresp/zresp{sub}_{story}.npy``;
            ``"fmri"`` scans ``zresp/cube/zresp{sub}_{story}.npy``.
        max_subjects: Cap on subjects to look at.
        max_stories: Cap on stories to look at.
        layer: GPT-2 layer used in the zstim filenames.

    Returns:
        A ``DRDRIndex`` with all valid (subject, story) pairs.
    """
    processed_root = Path(processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root does not exist: {processed_root}")

    if modality == "meg":
        resp_dir = processed_root / "MEG" / "zresp"
        stim_dir = processed_root / "MEG" / "zstim"
    elif modality == "fmri":
        resp_dir = processed_root / "zresp" / "cube"
        stim_dir = processed_root / "zstim"
    else:
        raise ValueError(f"Unknown modality: {modality}")

    if not resp_dir.exists():
        raise FileNotFoundError(f"Response dir does not exist: {resp_dir}")

    valid_subjects: list[int] = []
    valid_stories: list[int] = []
    pairs: list[tuple[int, int]] = []

    for sub in range(1, max_subjects + 1):
        sub_stories = []
        for story in range(1, max_stories + 1):
            has_resp = (resp_dir / f"zresp{sub}_{story}.npy").exists()
            has_stim = (
                stim_dir / f"sub{sub}_zstim_{layer}_story{story}.npy"
            ).exists()
            if has_resp and has_stim:
                sub_stories.append(story)
        if sub_stories:
            valid_subjects.append(sub)
            valid_stories.extend(sub_stories)
            pairs.extend((sub, story) for story in sub_stories)

    valid_stories = sorted(set(valid_stories))

    by_subject: dict[int, list[int]] = {s: [] for s in valid_subjects}
    by_story: dict[int, list[int]] = {s: [] for s in valid_stories}
    for sub, story in pairs:
        by_subject[sub].append(story)
        by_story[story].append(sub)

    index = DRDRIndex(
        processed_root=processed_root,
        modality=modality,
        subjects=valid_subjects,
        stories=valid_stories,
        by_subject=by_subject,
        by_story=by_story,
        pairs=pairs,
    )
    logger.info(
        "Discovered DRDR (%s): %d subjects, %d stories, %d (subject, story) pairs",
        modality, len(valid_subjects), len(valid_stories), len(pairs),
    )
    return index


# ───────────────────── Story loaders (memmap) ─────────────────────


def weight_delays(
    zstim: np.ndarray,
    weights: tuple[float, ...] = DEFAULT_DELAY_WEIGHTS,
    semantic_dim: int = SEMANTIC_DIM,
    n_delays: int = N_DELAYS,
) -> np.ndarray:
    """Combine the 4 concatenated delay vectors into one weighted vector.

    Port of the original ``process_stim_data``:
    ``(T, n_delays * D) -> (n_delays, T, D) -> (T, D, n_delays) @ weights``.

    Args:
        zstim: Stimulus array, shape (T, n_delays * semantic_dim).
        weights: Delay weights (default ``[0.1, 0.7, 0.5, 0.3]``).
        semantic_dim: Dimension of one delay vector.
        n_delays: Number of concatenated delays.

    Returns:
        Weighted target array of shape (T, semantic_dim), float32.
    """
    zstim = np.asarray(zstim)
    if zstim.ndim == 1:
        zstim = zstim[np.newaxis, :]
    if zstim.shape[-1] != n_delays * semantic_dim:
        raise ValueError(
            f"zstim last dim must be {n_delays * semantic_dim}, got {zstim.shape[-1]}"
        )
    reshaped = np.stack(
        [zstim[:, j * semantic_dim : (j + 1) * semantic_dim] for j in range(n_delays)],
        axis=0,  # (n_delays, T, D)
    )
    transposed = np.transpose(reshaped, (1, 2, 0))  # (T, D, n_delays)
    weighted = transposed @ np.asarray(weights, dtype=np.float64)  # (T, D)
    return weighted.astype(np.float32)


def load_meg_story(
    processed_root: str | Path,
    subject_id: int,
    story_id: int,
    n_context: int = 0,
) -> np.ndarray:
    """Load one story's MEG response as a memmap array.

    Args:
        n_context: If > 0, load the pre-built context-window file
            ``zresp{sub}_{story}_context_{n_context}.npy`` of shape
            (T, n_context, C); otherwise the plain (T, C) file.

    Returns:
        float32 memmap: (T, C) if n_context == 0, else (T, n_context, C).
    """
    resp_dir = Path(processed_root) / "MEG" / "zresp"
    if n_context > 0:
        fname = resp_dir / f"zresp{subject_id}_{story_id}_context_{n_context}.npy"
    else:
        fname = resp_dir / f"zresp{subject_id}_{story_id}.npy"
    if not fname.exists():
        raise FileNotFoundError(f"MEG response file not found: {fname}")
    arr = np.load(fname, mmap_mode="r")
    if arr.dtype != np.float32:
        raise ValueError(f"Unexpected dtype {arr.dtype} for {fname} (expected float32)")
    return arr


def load_fmri_story(
    processed_root: str | Path,
    subject_id: int,
    story_id: int,
) -> np.ndarray:
    """Load one story's fMRI cube response as a memmap array.

    Returns:
        float32 memmap of shape (T, 91, 109, 91).
    """
    fname = (
        Path(processed_root) / "zresp" / "cube" / f"zresp{subject_id}_{story_id}.npy"
    )
    if not fname.exists():
        raise FileNotFoundError(f"fMRI cube file not found: {fname}")
    arr = np.load(fname, mmap_mode="r")
    if arr.dtype != np.float32:
        raise ValueError(f"Unexpected dtype {arr.dtype} for {fname} (expected float32)")
    return arr


def load_stim_story(
    processed_root: str | Path,
    subject_id: int,
    story_id: int,
    layer: int = 10,
    weights: tuple[float, ...] = DEFAULT_DELAY_WEIGHTS,
    meg: bool = True,
    mmap: bool = False,
) -> np.ndarray:
    """Load one story's stimulus, delay-weighted, as (T, 768) float32.

    Args:
        meg: If True, look in ``MEG/zstim/`` (MEG sampling grid);
            otherwise in ``zstim/`` (fMRI sampling grid).
        mmap: If True, return the raw (T, 3072) memmap instead of the
            weighted (T, 768) array (used to save RAM when caching).
    """
    stim_dir = Path(processed_root) / ("MEG/zstim" if meg else "zstim")
    fname = stim_dir / f"sub{subject_id}_zstim_{layer}_story{story_id}.npy"
    if not fname.exists():
        raise FileNotFoundError(f"zstim file not found: {fname}")
    raw = np.load(fname, mmap_mode="r")
    if mmap:
        return raw
    return weight_delays(np.asarray(raw), weights=weights)


# ───────────────────── Brain mask ─────────────────────


def load_brain_mask(processed_root: str | Path) -> np.ndarray:
    """Load the fMRI brain mask (``mask.nii``) as a bool array.

    Shape (91, 109, 91) — matches the cube data.
    """
    import nibabel as nib  # lazy import; only needed for fMRI

    mask_path = Path(processed_root) / "mask.nii"
    if not mask_path.exists():
        raise FileNotFoundError(f"Brain mask not found: {mask_path}")
    data = nib.load(str(mask_path)).get_fdata()
    return data.astype(bool)


# ───────────────────── Single-sample loaders ─────────────────────
#
# Thin helpers used by tests/scripts; the Datasets read stories directly
# via the memmap loaders above.


def load_meg_sample(
    processed_root: str | Path,
    subject_id: int,
    story_id: int,
    n_context: int = 0,
    time_step: int = 0,
) -> MEGSample:
    """Load one MEG time step as a validated :class:`MEGSample`.

    Canonical schema shape: (C, 1) for plain, (n_context, C, 1) for
    context-windowed input (the trailing dim is the per-step time window,
    which the preprocessed data collapses to a single value).
    """
    resp = load_meg_story(processed_root, subject_id, story_id, n_context=n_context)
    row = resp[time_step]
    n_channels = row.shape[-1]
    pos = np.zeros((n_channels, 6), dtype=np.float32)
    sensor_type = np.array(
        [1] * (n_channels // 3) + [2] * (n_channels - n_channels // 3),
        dtype=np.int32,
    )
    if row.ndim == 1:
        return MEGSample(
            x=row[:, None],  # (C,) -> (C, 1)
            pos=pos,
            sensor_type=sensor_type,
            story_id=story_id,
            subject_id=subject_id,
        )
    return MEGChunkedSample(
        x=row[..., None],  # (n_context, C) -> (n_context, C, 1)
        pos=pos,
        sensor_type=sensor_type,
        context_len=n_context,
        story_id=story_id,
        subject_id=subject_id,
    )


def load_stim_sample(
    processed_root: str | Path,
    subject_id: int,
    story_id: int,
    layer: int = 10,
    weights: tuple[float, ...] = DEFAULT_DELAY_WEIGHTS,
    meg: bool = True,
    time_step: int = 0,
) -> StimSample:
    """Load one weighted stimulus vector as a validated :class:`StimSample`."""
    stim = load_stim_story(
        processed_root, subject_id, story_id, layer=layer, weights=weights, meg=meg
    )
    return StimSample(
        zstim=stim[time_step],
        layer=layer,
        story_id=story_id,
        subject_id=subject_id,
    )


def load_fmri_sample(
    processed_root: str | Path,
    subject_id: int,
    story_id: int,
    mask_path: str | Path | None = None,
    time_step: int = 0,
) -> fMRISample:
    """Load one fMRI volume as a validated :class:`fMRISample`.

    Args:
        mask_path: Optional path to the brain mask nii. If None, loads
            ``<processed_root>/mask.nii``.
    """
    vol = np.asarray(
        load_fmri_story(processed_root, subject_id, story_id)[time_step]
    )
    if mask_path is None:
        mask = load_brain_mask(processed_root)
    else:
        import nibabel as nib

        mask = nib.load(str(mask_path)).get_fdata().astype(bool)
    return fMRISample(
        volume=vol,
        mask=mask,
        story_id=story_id,
        subject_id=subject_id,
    )


__all__ = [
    "DEFAULT_DELAY_WEIGHTS",
    "DRDRIndex",
    "FMRI_CUBE_SHAPE",
    "N_DELAYS",
    "SEMANTIC_DIM",
    "discover_drdr",
    "load_brain_mask",
    "load_fmri_sample",
    "load_fmri_story",
    "load_meg_sample",
    "load_meg_story",
    "load_stim_sample",
    "load_stim_story",
    "weight_delays",
]
