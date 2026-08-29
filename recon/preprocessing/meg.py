"""MEG brain-side preprocessing: fif → zresp (classic 0.4 s path).

Ported verbatim from ``process_brain_feature.py:MEGFeatureProcessor.process``:

1. Read the BIDS preprocessed fif (``sub-XX_task-RDR_run-N_meg.fif``)
2. Resample to ``1/interval`` Hz (2.5 Hz for the 0.4 s grid)
3. Select channels ``channel_start:channel_end`` (12:318 = 306 ch)
4. Crop to the stimulus grid: start at ``first_char_onset_time``
   (12 s → row 30), length = rows of the semantic downsample file
   (``ds_{story}_{layer}.npy``)
5. Per-channel z-score → ``zresp{sub}_{story}.npy``

``chunk_context`` builds the (T, n_context, C) context files the model
consumes (legacy ``chunk_context``).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .common import sidecar_valid, write_sidecar, zscore_2d

logger = logging.getLogger(__name__)


def _subject_dir_name(subject_id: int) -> str:
    return f"{subject_id:02d}" if 0 < subject_id < 10 else str(subject_id)


def process_story(
    fif_path: str | Path,
    ds_path: str | Path,
    save_dir: str | Path,
    subject_id: int,
    story_id: int,
    layer: int = 10,
    interval: float = 0.4,
    first_char_onset_time: float = 12.0,
    channel_start: int = 12,
    channel_end: int = 318,
) -> Path:
    """fif → zresp for one story (legacy ``MEGFeatureProcessor.process``).

    Args:
        fif_path: BIDS preprocessed MEG fif for the story.
        ds_path: Semantic downsample file (row-count reference).
        save_dir: Directory for ``zresp{sub}_{story}.npy``.

    Returns:
        Path of the written zresp file.
    """
    import mne

    fif_path = Path(fif_path)
    ds_path = Path(ds_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_fif(str(fif_path))
    downsampled_rate = 1 / interval
    raw.resample(downsampled_rate)
    img_data = np.array(raw.get_data(), dtype="float32")
    meg_reshaped = img_data.T  # (time, channels)
    meg_resp = meg_reshaped[:, channel_start:channel_end]

    stim = np.load(ds_path)
    vol = stim.shape[0]
    b = int(first_char_onset_time / interval)
    resp = meg_resp[b : (vol + b), :]
    z_resp = zscore_2d(resp)

    out = save_dir / f"zresp{subject_id}_{story_id}.npy"
    params = {
        "stage": "meg_zresp",
        "subject_id": subject_id,
        "story_id": story_id,
        "layer": layer,
        "interval": interval,
        "first_char_onset_time": first_char_onset_time,
        "channel_start": channel_start,
        "channel_end": channel_end,
    }
    np.save(out, z_resp)
    write_sidecar(out, {"fif": fif_path, "ds": ds_path}, params)
    logger.info("MEG story %d: %s -> %s", story_id, resp.shape, z_resp.shape)
    return out


def chunk_context(
    zresp_path: str | Path,
    save_dir: str | Path,
    subject_id: int,
    story_id: int,
    n_context: int = 5,
) -> Path:
    """(T, C) → (T, n_context, C) with zero-padded warmup (legacy ``chunk_context``)."""
    zresp_path = Path(zresp_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    resp = np.load(zresp_path)  # (T, C)
    resp_list = [t for t in np.split(resp, resp.shape[0], axis=0)]
    resp_list = [np.zeros_like(resp_list[0]) for _ in range(n_context - 1)] + resp_list
    chunked = np.array(
        [np.concatenate(resp_list[j : j + n_context], axis=0) for j in range(len(resp_list) - n_context + 1)]
    )  # (T, n_context, C)

    out = save_dir / f"zresp{subject_id}_{story_id}_context_{n_context}.npy"
    params = {"stage": "meg_chunk_context", "subject_id": subject_id, "story_id": story_id, "n_context": n_context}
    np.save(out, chunked)
    write_sidecar(out, {"zresp": zresp_path}, params)
    logger.info("Chunked story %d (n_context=%d): %s -> %s", story_id, n_context, resp.shape, chunked.shape)
    return out


def needs_regeneration(artifact: str | Path, inputs: dict, params: dict) -> bool:
    """Convenience inverse of ``sidecar_valid`` (skip-if-valid semantics)."""
    return not sidecar_valid(artifact, inputs, params)


__all__ = ["chunk_context", "needs_regeneration", "process_story"]
