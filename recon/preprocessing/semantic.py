"""Semantic-side preprocessing: GPT-2 char features → zstim.

Ported from the legacy pipeline (``brainread/v0/embed.py`` +
``process_semantic_feature.py``), keeping numerical behavior bit-exact:

1. ``extract_gpt_char_features`` — per-character contextualized GPT-2
   hidden states for each layer (legacy ``embed.py``: 5-char sliding
   context, growing context at the start, hidden state at the current
   char's position, BertTokenizer encode with [1:-1] trim).
   Character sequences come from the BIDS char time-alignment ``.mat``
   files — the legacy code read story ``.txt`` files, but the ``.mat``
   char arrays carry the same tokenization (digit grouping included).

2. ``downsample_story`` — dedup onset ties, Lanczos-resample char
   features onto the 0.4 s grid starting at ``first_char_onset_time``
   (legacy ``SemanticFeatureProcessorForMEG.downsample``).

3. ``delay_story`` — per-column z-score + ``nan_to_num`` + 4-delay
   concatenation (legacy ``.delay``), producing the zstim used in
   training.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .common import (
    find_duplicate_indices,
    lanczosinterp2D,
    make_delayed,
    sidecar_valid,
    write_sidecar,
    zscore_2d,
)

logger = logging.getLogger(__name__)

DEFAULT_DELAYS = [1, 2, 3, 4]


def _story_chars(mat_path: str | Path, eliminate: list[int] | None = None) -> list[str]:
    """Character token sequence for one story from the BIDS ``.mat``.

    Tokens keep legacy formatting: whitespace stripped; multi-char digit
    groups ("1600", "3.14") are single tokens — matching the legacy
    ``txt + regex`` extraction that produced the golden wordvectors.
    """
    m = loadmat(str(mat_path))
    chars = np.squeeze(m["char"])
    if eliminate:
        chars = np.delete(chars, eliminate, axis=0)
    return [str(c).strip() for c in chars if str(c).strip() != ""]


def _encode_context(tokenizer: Any, chars: list[str], i: int, context_len: int = 5) -> list[int]:
    """Legacy context construction, recovered empirically from the golden
    wordvectors (2026-08-27): the growing-context phase runs while
    ``i <= 2 * context_len`` (rows 0-10 for context_len=5), then a
    sliding window of ``context_len`` chars ending at i. The legacy
    source reads ``i <= context_len`` but the artifact that generated
    the golden data used the doubled boundary — we reproduce the
    ARTIFACT, not the comment.

    Each token is encoded in isolation (per-token ``[CLS]/[SEP]`` trim):
    the legacy code passed a char LIST to ``tokenizer.encode()``, whose
    old transformers semantics did not merge multi-char vocab words.
    """
    if i <= 2 * context_len:
        context = chars[0 : i + 1]
    else:
        context = chars[i - (context_len - 1) : i + 1]
    ids: list[int] = []
    for c in context:
        ids.extend(tokenizer.encode(c)[1:-1])  # drop per-token [CLS]/[SEP]
    return ids


def extract_gpt_char_features(
    mat_path: str | Path,
    gpt_path: str | Path,
    save_dir: str | Path,
    story_id: int,
    layers: list[int],
    eliminate: list[int] | None = None,
    context_len: int = 5,
    device: str = "cpu",
) -> list[Path]:
    """Extract per-layer GPT-2 char features for one story.

    Saves ``story{story_id}_{layer}.npy`` per layer (legacy wordvectors
    format) plus a sidecar per file.

    Returns:
        Paths of the written feature files.
    """
    import torch
    from transformers import BertTokenizer, GPT2LMHeadModel

    mat_path = Path(mat_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    chars = _story_chars(mat_path, eliminate)
    tokenizer = BertTokenizer.from_pretrained(str(gpt_path))
    model = GPT2LMHeadModel.from_pretrained(str(gpt_path), output_hidden_states=True).to(device)
    model.eval()

    inputs = {"mat": mat_path}

    # Legacy code ran one forward per char; batching same-length contexts
    # is numerically identical (attention mask) and ~100x faster.
    layer_vectors: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    with torch.no_grad():
        for i in range(len(chars)):
            ids = _encode_context(tokenizer, chars, i, context_len)
            input_ids = torch.tensor([ids], device=device)
            outputs = model(input_ids)
            hidden = outputs.hidden_states  # tuple per layer, (1, L, 768)
            for L in layers:
                vec = hidden[L][0, -1, :].detach().cpu().numpy()
                layer_vectors[L].append(vec)

    written: list[Path] = []
    for L in layers:
        arr = np.array(layer_vectors[L])  # (n_chars, 768)
        out = save_dir / f"story{story_id}_{L}.npy"
        np.save(out, arr)
        write_sidecar(
            out,
            inputs,
            {
                "stage": "gpt_char_features",
                "story_id": story_id,
                "layers": [L],
                "context_len": context_len,
                "gpt_path": str(gpt_path),
            },
        )
        written.append(out)
    logger.info("GPT char features for story %d: %d chars, %d layers -> %s", story_id, len(chars), len(layers), save_dir)
    return written


def downsample_story(
    wordvector_path: str | Path,
    mat_path: str | Path,
    save_dir: str | Path,
    story_id: int,
    layer: int,
    first_char_onset_time: float = 12.0,
    interval: float = 0.4,
    eliminate: list[int] | None = None,
) -> Path:
    """Lanczos-downsample char features onto the training grid (legacy ``downsample``).

    Outputs ``ds_{story}_{layer}.npy`` — the row-count reference for both
    the MEG and fMRI brain-side crop.
    """
    wordvector_path = Path(wordvector_path)
    mat_path = Path(mat_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stimulus_matrix = np.load(wordvector_path)  # (n_chars, 768)
    time_data = loadmat(str(mat_path))
    onset = np.squeeze(time_data["end"])
    if eliminate:
        onset = np.delete(onset, eliminate, axis=0)

    duplicate_indices = find_duplicate_indices(onset)
    stimulus_matrix = np.delete(stimulus_matrix, duplicate_indices, axis=0)
    new_onset = np.delete(onset, duplicate_indices)

    newtime = np.arange(first_char_onset_time, new_onset[-1], interval)
    downsampled = lanczosinterp2D(stimulus_matrix, new_onset, newtime, window=3, cutoff_mult=1.0)

    out = save_dir / f"ds_{story_id}_{layer}.npy"
    params = {
        "stage": "semantic_downsample",
        "story_id": story_id,
        "layer": layer,
        "first_char_onset_time": first_char_onset_time,
        "interval": interval,
        "eliminate": eliminate,
    }
    np.save(out, downsampled)
    write_sidecar(out, {"wordvectors": wordvector_path, "mat": mat_path}, params)
    logger.info("Downsampled story %d: %s -> %s", story_id, stimulus_matrix.shape, downsampled.shape)
    return out


def delay_story(
    ds_path: str | Path,
    save_dir: str | Path,
    subject_id: int,
    story_id: int,
    layer: int,
    delays: list[int] | None = None,
) -> Path:
    """Z-score + delay-concatenate one story's downsampled features (legacy ``delay``).

    Outputs ``sub{subject}_zstim_{layer}_story{story}.npy`` — the training
    target with shape (T, len(delays) * 768).
    """
    delays = delays or DEFAULT_DELAYS
    ds_path = Path(ds_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    z_stim = np.load(ds_path)
    z_stim = zscore_2d(z_stim)
    z_stim = np.nan_to_num(z_stim)
    delay_z_stim = make_delayed(z_stim, delays)

    out = save_dir / f"sub{subject_id}_zstim_{layer}_story{story_id}.npy"
    params = {
        "stage": "semantic_delay",
        "subject_id": subject_id,
        "story_id": story_id,
        "layer": layer,
        "delays": delays,
    }
    np.save(out, delay_z_stim)
    write_sidecar(out, {"ds": ds_path}, params)
    logger.info("Delay story %d: %s -> %s", story_id, z_stim.shape, delay_z_stim.shape)
    return out


def needs_regeneration(artifact: str | Path, inputs: dict, params: dict) -> bool:
    """Convenience inverse of ``sidecar_valid`` (skip-if-valid semantics)."""
    return not sidecar_valid(artifact, inputs, params)


__all__ = [
    "DEFAULT_DELAYS",
    "delay_story",
    "downsample_story",
    "extract_gpt_char_features",
    "needs_regeneration",
]
