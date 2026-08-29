"""Shared primitives for the preprocessing pipeline.

Two families of code live here, both ported VERBATIM from the legacy
pipeline so numerical outputs match the golden artifacts in
``E:/results``:

1. **Numerical kernels** — z-score, lanczos resampling, delay filtering,
   duplicate detection (from ``process_brain_feature.py`` /
   ``process_semantic_feature.py``). Do not "improve" these: any change
   breaks bit-exact equivalence with the golden zresp/zstim.

2. **Provenance** — content hashing and sidecar metadata
   (``*.meta.json`` beside every artifact), the pipeline's audit trail.
   See docs/planning/data-pipeline-design.md.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ───────────────────── Numerical kernels (legacy-verbatim) ─────────────────────


def zscore_2d(mat: np.ndarray) -> np.ndarray:
    """Z-score each COLUMN of a 2D matrix (legacy ``zscore_2d``)."""
    zmat = np.empty(mat.shape, mat.dtype)
    for ri in range(mat.shape[1]):
        std = np.std(mat[:, ri])
        mean = np.mean(mat[:, ri])
        zmat[:, ri] = (mat[:, ri] - mean) / (1e-10 + std)
    return zmat


def zscore_4d(mat_4d: np.ndarray) -> np.ndarray:
    """Z-score each voxel's time series of a 4D array (legacy ``zscore``)."""
    spatial_dims = mat_4d.shape[:3]
    time_dim = mat_4d.shape[3]
    fmri_reshaped = np.reshape(mat_4d, (*spatial_dims, time_dim))
    mat_2d = np.transpose(
        np.reshape(fmri_reshaped, (int(np.prod(spatial_dims)), time_dim))
    )  # (T, n_voxels)
    zmat_2d = zscore_2d(mat_2d)
    return np.reshape(np.transpose(zmat_2d), mat_4d.shape)


def find_duplicate_indices(lst: np.ndarray | list) -> list[int]:
    """Indices of values already seen earlier in the list (legacy)."""
    seen: set = set()
    duplicates: list[int] = []
    for i, value in enumerate(lst):
        if value in seen:
            duplicates.append(i)
        else:
            seen.add(value)
    return duplicates


def lanczosfun(cutoff: float, t: np.ndarray, window: int = 3) -> np.ndarray:
    """Lanczos filter kernel (legacy ``lanczosfun``)."""
    t = t * cutoff
    val = window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi**2 * t**2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val


def lanczosinterp2D(
    data: np.ndarray,
    oldtime: np.ndarray,
    newtime: np.ndarray,
    window: int = 3,
    cutoff_mult: float = 1.0,
    rectify: bool = False,
) -> np.ndarray:
    """Lanczos-resample rows of ``data`` from ``oldtime`` to ``newtime`` (legacy)."""
    cutoff = 1 / np.mean(np.diff(newtime)) * cutoff_mult
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi, :] = lanczosfun(cutoff, newtime[ndi] - oldtime, window)
    if rectify:
        newdata = np.hstack(
            [np.dot(sincmat, np.clip(data, -np.inf, 0)), np.dot(sincmat, np.clip(data, 0, np.inf))]
        )
    else:
        newdata = np.dot(sincmat, data)
    return newdata


def make_delayed(stim: np.ndarray, delays: list[int], circpad: bool = False) -> np.ndarray:
    """Concatenate shifted copies of ``stim`` per delay (legacy ``make_delayed``)."""
    nt, ndim = stim.shape
    dstims = []
    for d in delays:
        dstim = np.zeros((nt, ndim))
        if d < 0:
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


# ───────────────────── Provenance: hashing + sidecar ─────────────────────


def git_commit() -> str | None:
    """Best-effort git commit hash of the current checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def hash_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    """SHA-256 of a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(artifact: str | Path, inputs: dict[str, str | Path], params: dict[str, Any]) -> Path:
    """Write ``<artifact>.meta.json`` recording inputs, params, and provenance.

    Args:
        artifact: Path to the artifact (npy) just written.
        inputs: Map of input name → input file path (hashed).
        params: Arbitrary JSON-serializable parameter dict.
    """
    artifact = Path(artifact)
    meta = {
        "artifact": str(artifact),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "params": params,
        "inputs": {name: {"path": str(p), "sha256": hash_file(p)} for name, p in inputs.items()},
    }
    sidecar = artifact.with_suffix(artifact.suffix + ".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return sidecar


def sidecar_valid(artifact: str | Path, inputs: dict[str, str | Path], params: dict[str, Any]) -> bool:
    """True if the artifact exists and its sidecar matches inputs+params.

    This is the pipeline's resume/idempotence check: if the sidecar is
    valid, the artifact does not need to be regenerated.
    """
    artifact = Path(artifact)
    sidecar = artifact.with_suffix(artifact.suffix + ".meta.json")
    if not artifact.exists() or not sidecar.exists():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("params") != params:
        return False
    recorded = meta.get("inputs", {})
    if set(recorded) != set(inputs):
        return False
    for name, path in inputs.items():
        entry = recorded.get(name, {})
        if entry.get("path") != str(path):
            return False
        if not Path(path).exists():
            return False
        if entry.get("sha256") != hash_file(path):
            return False
    return True


__all__ = [
    "find_duplicate_indices",
    "git_commit",
    "hash_file",
    "lanczosfun",
    "lanczosinterp2D",
    "make_delayed",
    "sidecar_valid",
    "write_sidecar",
    "zscore_2d",
    "zscore_4d",
]
