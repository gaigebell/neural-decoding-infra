"""fMRI brain-side preprocessing: MNI nii.gz → cube zresp.

Ported verbatim from ``brainread/v0/dataproc/fMRI/getzresp.py``:

1. Load the BIDS preprocessed MNI bold (``sub-XX_task-RDR_run-N_bold.nii.gz``)
2. Flatten to (T, n_voxels)
3. Crop to the stimulus grid: start at row 17 (≈12.07 s on the fMRI TR
   grid), length = rows of the semantic downsample file
4. Per-voxel z-score
5. Reshape back to the spatial cube (T, X, Y, Z) → ``zresp{sub}_{story}.npy``
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .common import sidecar_valid, write_sidecar, zscore_2d

logger = logging.getLogger(__name__)


def process_story(
    nii_path: str | Path,
    ds_path: str | Path,
    save_dir: str | Path,
    subject_id: int,
    story_id: int,
    layer: int = 10,
    start_row: int = 17,
) -> Path:
    """nii.gz → cube zresp for one story (legacy ``getzresp.py``)."""
    import gc

    import nibabel as nib

    nii_path = Path(nii_path)
    ds_path = Path(ds_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(nii_path))
    img_data = np.array(img.get_fdata(), dtype="float32")
    spatial_dims = img_data.shape[:3]
    time_dim = img_data.shape[3]
    fmri_reshaped = np.reshape(img_data, (*spatial_dims, time_dim))
    fmri_resp = np.transpose(
        np.reshape(fmri_reshaped, (int(np.prod(spatial_dims)), time_dim))
    )  # (T, n_voxels)
    del fmri_reshaped, img_data
    gc.collect()

    stim = np.load(ds_path)
    vol = stim.shape[0]
    resp = fmri_resp[start_row : (vol + start_row), :]
    z_resp = zscore_2d(resp)
    z_resp = np.reshape(z_resp, (resp.shape[0], *spatial_dims))  # cube

    out = save_dir / f"zresp{subject_id}_{story_id}.npy"
    params = {
        "stage": "fmri_cube",
        "subject_id": subject_id,
        "story_id": story_id,
        "layer": layer,
        "start_row": start_row,
        "cube_shape": list(spatial_dims),
    }
    np.save(out, z_resp)
    write_sidecar(out, {"nii": nii_path, "ds": ds_path}, params)
    logger.info("fMRI story %d: %s -> %s", story_id, resp.shape, z_resp.shape)
    return out


def needs_regeneration(artifact: str | Path, inputs: dict, params: dict) -> bool:
    """Convenience inverse of ``sidecar_valid`` (skip-if-valid semantics)."""
    return not sidecar_valid(artifact, inputs, params)


__all__ = ["needs_regeneration", "process_story"]
