"""BrainOmni pipeline stages: input segments (CPU) + feature encoding (GPU).

Two stages, both ported from the legacy pipeline:

1. ``extract_segments_story`` — ports
   ``2026-7-22/process_brain_feature.py:BrainOmniFeatureProcessor.brainomni_process``
   (the NEW variant used by ``prepipeline.py``): load fif → pick MEG channels
   → notch 60 Hz → bandpass 0.1-96 Hz → resample to ``sample_rate`` →
   per-char segments (``segment_length`` s BEFORE each word onset, zero-padded
   at the start) → sensor-type-wise normalization (mag/grad/eeg groups each
   z-scored separately) → ``torch.save`` a dict
   ``{x: (T, C, sr*seg), pos: (T, C, 6), sensor_type: (T, C)}`` plus the
   word-time metadata json.

2. ``encode_story`` — ports ``brainomni_encode.py:infer_story``: load the
   BrainOmni model (frozen tokenizer) from the local repo checkpoints,
   ``model.encode(**inputs)`` on GPU, save
   ``(batch, n_neurons, seq_len, n_dim)`` features.

The BrainOmni model code lives OUTSIDE this repo (owner's
``BrainOmni-main`` checkout); it is imported dynamically via
``sys.path`` so the core package does not depend on it.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from .common import sidecar_valid, write_sidecar

logger = logging.getLogger(__name__)

SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}


# ───────────────────── Segment extraction (CPU) ─────────────────────


def _extract_pos_sensor_type(info) -> tuple[np.ndarray, np.ndarray]:
    """Sensor positions (C, 6) and type codes (C,) from a fif info (legacy)."""
    pos: list[np.ndarray] = []
    sensor_type: list[int] = []
    for ch in info["chs"]:
        kind = int(ch["kind"])
        assert kind in [1, 2], f"Unknown sensor kind: {kind}"
        coil_type = str(ch["coil_type"])
        if kind == 2:  # EEG
            pos.append(np.hstack([ch["loc"][:3], np.array([0.0, 0.0, 0.0])]))
            sensor_type.append(SENSOR_TYPE_DICT["EEG"])
        else:  # MEG
            xyz = ch["loc"][:3]
            dir_idx = 1 if "PLANAR" in coil_type else 3
            direction = ch["loc"][3 * dir_idx : 3 * (dir_idx + 1)]
            pos.append(np.hstack([xyz, direction]))
            sensor_type.append(
                SENSOR_TYPE_DICT["MAG"] if "MAG" in coil_type else SENSOR_TYPE_DICT["GRAD"]
            )
    return np.stack(pos).astype(np.float32), np.array(sensor_type).astype(np.int32)


def _normalize_pos(pos: np.ndarray, eeg_mask: np.ndarray, meg_mask: np.ndarray) -> np.ndarray:
    """Per-type position centering + scaling (legacy ``normalize_pos``)."""
    if eeg_mask.any():
        pos[eeg_mask, :3] -= np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] /= np.sqrt(3 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
    if meg_mask.any():
        pos[meg_mask, :3] -= np.mean(pos[meg_mask, :3], axis=0, keepdims=True)
        pos[meg_mask, :3] /= np.sqrt(3 * np.mean(np.sum(pos[meg_mask, :3] ** 2, axis=1)))
    return pos


def _sensortype_wise_normalize(
    data: np.ndarray, eeg_mask: np.ndarray, mag_mask: np.ndarray, grad_mask: np.ndarray
) -> np.ndarray:
    """Per-sensor-type z-score of a segment.

    Ported line-for-line from the legacy ``sensortype_wise_normalize``:
    the std is computed AFTER mean removal (``mag_data = mag_data - mean``
    in place), and the eps differs per type (1e-5 EEG / 1e-13 MEG).
    """
    out = data.copy()
    if eeg_mask.any():
        eeg_data = out[eeg_mask, :]
        eeg_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
        out[eeg_mask, :] = eeg_data / (np.std(eeg_data) + 1.0e-5)
    if mag_mask.any():
        mag_data = out[mag_mask, :]
        mag_data = mag_data - np.mean(mag_data, axis=0, keepdims=True)
        out[mag_mask, :] = mag_data / (np.std(mag_data) + 1.0e-13)
    if grad_mask.any():
        grad_data = out[grad_mask, :]
        grad_data = grad_data - np.mean(grad_data, axis=0, keepdims=True)
        out[grad_mask, :] = grad_data / (np.std(grad_data) + 1.0e-13)
    return out.astype(np.float32)


def extract_segments_story(
    fif_path: str | Path,
    time_align_path: str | Path,
    save_dir: str | Path,
    subject_id: int,
    story_id: int,
    layer: int = 10,
    sample_rate: int = 256,
    segment_length: int = 2,
    x_scale: float = 1.0,
) -> Path:
    """fif → BrainOmni input segments for one story (legacy ``brainomni_process``).

    Saves ``brainomni_zresp{sub}_{story}.pt`` (dict: x/pos/sensor_type) and
    ``word_time_meta_{sub}_{story}.json`` in
    ``save_dir/brainomni_sr{sr}_seg{seg}/``.
    """
    import mne
    import torch

    fif_path = Path(fif_path)
    time_align_path = Path(time_align_path)
    # Legacy dir naming: brainomni_sample_rate_{sr}_segment_length_{seg}
    out_dir = Path(save_dir) / f"brainomni_sample_rate_{sample_rate}_segment_length_{segment_length}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    indices = mne.pick_types(raw.info, meg=True, eeg=False, ref_meg=False)
    raw.pick(indices)
    raw = raw.notch_filter(freqs=[60], verbose=False)
    raw = raw.filter(0.1, 96, verbose=False)
    raw = raw.resample(sample_rate, verbose=False)
    meg_time_secs = raw.times

    pos, sensor_type = _extract_pos_sensor_type(raw.info)
    # GOLDEN CONVENTION (verified against the golden artifacts,
    # 2026-08-29): the golden `pos` equals the RAW fif positions —
    # bit-exact. The legacy source calls `normalize_pos`, but the
    # artifact-generating code version predates it. BrainOmni was
    # trained on raw positions; we reproduce the artifact, not the
    # source. (`_normalize_pos` is kept for reference / future variants.)
    eeg_mask = sensor_type == SENSOR_TYPE_DICT["EEG"]
    mag_mask = sensor_type == SENSOR_TYPE_DICT["MAG"]
    grad_mask = sensor_type == SENSOR_TYPE_DICT["GRAD"]

    word_time_secs = np.load(time_align_path)
    if len(word_time_secs) == 0 or len(meg_time_secs) == 0:
        raise ValueError("Empty time alignment or MEG times")

    # Align word times to the MEG recording window (legacy crop logic)
    if word_time_secs[0] < meg_time_secs[0]:
        start_word_idx = int(np.where(word_time_secs >= meg_time_secs[0])[0][0])
    else:
        start_word_idx = 0
    if word_time_secs[-1] > meg_time_secs[-1]:
        end_word_idx = int(np.where(word_time_secs <= meg_time_secs[-1])[0][-1] + 1)
    else:
        end_word_idx = len(word_time_secs)
    word_time_secs_cropped = word_time_secs[start_word_idx:end_word_idx]

    meg_data = raw.get_data()  # (C, time)
    chunked_resp: list[np.ndarray] = []
    chunked_pos: list[np.ndarray] = []
    chunked_sensor_type: list[np.ndarray] = []

    for ti in word_time_secs_cropped:
        meg_idx = int(np.where(meg_time_secs >= ti)[0][0])
        ti_st = meg_idx - sample_rate * segment_length
        ti_ed = meg_idx
        if ti_st < 0:  # zero-pad at the start
            ti_resp = np.zeros(
                (meg_data.shape[0], sample_rate * segment_length), dtype=meg_data.dtype
            )
            ti_resp[:, sample_rate * segment_length - meg_idx :] = meg_data[:, 0:ti_ed]
        else:
            ti_resp = meg_data[:, ti_st:ti_ed]
        seg_data = _sensortype_wise_normalize(ti_resp, eeg_mask, mag_mask, grad_mask)
        assert not (np.isnan(seg_data).any() or np.isnan(pos).any()), "NaN in segment"
        chunked_pos.append(pos)
        chunked_sensor_type.append(sensor_type)
        chunked_resp.append(ti_resp)

    # Legacy saved float64 here (raw.get_data() default) which breaks the
    # model (conv bias is float32); the legacy OLD variant used float32 —
    # we follow it: float32 halves storage and matches the model weights.
    x = np.array(chunked_resp, dtype=np.float32)  # (T, C, sr*seg)
    if x_scale != 1.0:
        # Scale to the model's input convention. Empirical finding
        # (2026-09-11): the golden artifacts are our calibrated-T
        # segments scaled by a constant ~9.508e9 (per-segment factors
        # consistent to 0.02%, corr 0.9998) — the golden fif copy was
        # stored in different units. BrainOmni was trained on that
        # scale; set x_scale to match it when the local fif is in T.
        x = (x * x_scale).astype(np.float32)
    data_dict = {
        "x": torch.tensor(x),
        "pos": torch.tensor(np.array(chunked_pos)),  # (T, C, 6)
        "sensor_type": torch.tensor(np.array(chunked_sensor_type)),  # (T, C)
    }
    out = out_dir / f"brainomni_zresp{subject_id}_{story_id}.pt"
    torch.save(data_dict, out)
    meta = {
        "original_word_times": word_time_secs.tolist(),
        "cropped_word_times": word_time_secs_cropped.tolist(),
        "start_word_index": int(start_word_idx),
        "end_word_index": int(end_word_idx),
    }
    (out_dir / f"word_time_meta_{subject_id}_{story_id}.json").write_text(
        json.dumps(meta, indent=4)
    )
    write_sidecar(
        out,
        {"fif": fif_path, "time_align": time_align_path},
        {
            "stage": "meg_brainomni_segments",
            "subject_id": subject_id,
            "story_id": story_id,
            "layer": layer,
            "sample_rate": sample_rate,
            "segment_length": segment_length,
            "x_scale": x_scale,
        },
    )
    logger.info("BrainOmni segments story %d: x=%s", story_id, tuple(x.shape))
    return out


# ───────────────────── Feature encoding (GPU) ─────────────────────


def _ensure_deepspeed_shim() -> None:
    """Minimal no-op ``deepspeed`` shim for INFERENCE-only platforms.

    ``BrainOmni-main/model_utils/vq.py`` imports ``deepspeed.comm`` at
    module level, but every call site (all_reduce / broadcast / argmin)
    is codebook synchronization used only during multi-GPU TRAINING.
    On platforms where deepspeed is not installed (Windows local dev),
    inject no-op stubs so the model can still be imported for encoding.
    On the cluster (Linux), deepspeed is installed and this is skipped.
    """
    import importlib.util
    import types

    if "deepspeed" in sys.modules:  # already shimmed or genuinely installed
        return
    try:
        if importlib.util.find_spec("deepspeed") is not None:
            return
    except ValueError:  # a spec-less stub module in sys.modules
        pass
    logger.warning(
        "deepspeed not installed — injecting no-op shim (inference only; "
        "training-time codebook sync would be broken)"
    )
    ds = types.ModuleType("deepspeed")
    comm = types.ModuleType("deepspeed.comm")
    comm.is_initialized = lambda: False  # type: ignore[attr-defined]
    comm.get_world_size = lambda: 1  # type: ignore[attr-defined]
    comm.all_reduce = lambda *a, **k: None  # type: ignore[attr-defined]
    comm.broadcast = lambda *a, **k: None  # type: ignore[attr-defined]
    comm.argmin = lambda *a, **k: None  # type: ignore[attr-defined]
    ds.comm = comm  # type: ignore[attr-defined]
    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.comm"] = comm


def _ensure_brainomni_on_path(repo: str | Path) -> None:
    """Put the BrainOmni checkout on sys.path (idempotent)."""
    repo = str(Path(repo))
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _load_model(ckpt_dir: str | Path, cls_name: str, repo: str | Path | None = None):
    """Load a BrainOmni/BrainTokenizer model from a ckpt dir (legacy loader).

    ``repo`` (the BrainOmni checkout) is added to ``sys.path`` so the
    ``brainomni``/``braintokenizer``/``model_utils`` packages resolve —
    pass it when calling this directly; ``encode_story`` passes it.
    """
    import torch

    ckpt_dir = Path(ckpt_dir)
    model_config = json.loads((ckpt_dir / "model_cfg.json").read_text())
    if repo is not None:
        _ensure_brainomni_on_path(repo)
    _ensure_deepspeed_shim()
    if cls_name == "BrainTokenizer":
        from braintokenizer.model import BrainTokenizer
        model = BrainTokenizer(**model_config)
        state = torch.load(ckpt_dir / "BrainTokenizer.pt", map_location="cpu", weights_only=True)
    else:
        from brainomni.model import BrainOmni
        model = BrainOmni(**model_config)
        state = torch.load(ckpt_dir / "BrainOmni.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model


def encode_story(
    segments_path: str | Path,
    save_dir: str | Path,
    subject_id: int,
    story_id: int,
    brainomni_repo: str | Path,
    brainomni_ckpt: str | Path,
    tokenizer_ckpt: str | Path,
    device: str = "cuda",
) -> Path:
    """BrainOmni-encode one story's segments → ``_features.pt`` (GPU)."""
    import torch

    segments_path = Path(segments_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # The BrainOmni package lives outside this repo — add it to sys.path.
    repo = Path(brainomni_repo)
    _ensure_brainomni_on_path(repo)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"

    model = _load_model(brainomni_ckpt, "BrainOmni", repo=repo)
    for p in model.tokenizer.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    data = torch.load(segments_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in data.items()}
        outputs = model.encode(**inputs)
        features = outputs.cpu()
    out = save_dir / f"brainomni_zresp{subject_id}_{story_id}_features.pt"
    torch.save(features, out)
    write_sidecar(
        out,
        {"segments": segments_path, "brainomni_ckpt": Path(brainomni_ckpt) / "BrainOmni.pt"},
        {
            "stage": "brainomni_encode",
            "subject_id": subject_id,
            "story_id": story_id,
            "brainomni_repo": str(repo),
            "brainomni_ckpt": str(brainomni_ckpt),
            "tokenizer_ckpt": str(tokenizer_ckpt),
            "device": device,
        },
    )
    logger.info("BrainOmni features story %d: %s", story_id, tuple(features.shape))
    return out


def needs_regeneration(artifact: str | Path, inputs: dict, params: dict) -> bool:
    """Convenience inverse of ``sidecar_valid`` (skip-if-valid semantics)."""
    return not sidecar_valid(artifact, inputs, params)


__all__ = ["encode_story", "extract_segments_story", "needs_regeneration"]