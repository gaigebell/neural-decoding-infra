"""Decode-time alignment (D1-A): characters → training-grid rows.

The training pairing is ``resp_ctx[t-4] → stim[t]`` (MEGDataset). Decoding
must feed the model the SAME input. This module implements the direct
mapping approved in the plan:

- every character's onset (``end`` time from the BIDS ``.mat``) maps to
  the nearest zresp row: ``row = round((onset - first_char_onset_time) / interval)``
- characters whose row falls outside the story grid are dropped
- the model input for character ``c`` is the context window at
  ``row - (n_context - 1)`` — zero-padded when it would start before 0
  (mirroring the training warmup)
- decode length = number of valid characters ("tell the model how many
  characters to fill")

The alternative (WR model / ``LengthPredictor``) plugs in later by
replacing the ``length`` and per-char row computation — see
docs/planning/decode-eval-phase.md D1-B.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 0.4
DEFAULT_FIRST_CHAR_ONSET = 12.0


def char_grid(
    mat_path: str | Path,
    eliminate: list[int] | None = None,
    interval: float = DEFAULT_INTERVAL,
    first_char_onset_time: float = DEFAULT_FIRST_CHAR_ONSET,
    n_rows: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """Map a story's characters onto the training grid.

    Args:
        mat_path: BIDS char-time ``.mat`` (``char`` / ``start`` / ``end``).
        eliminate: Character indices to drop (legacy English elimination).
        interval: Training grid step in seconds (0.4).
        first_char_onset_time: Grid start time in seconds (12.0).
        n_rows: Story length in rows (zresp shape[0]). Rows outside
            ``[0, n_rows)`` are dropped; None keeps all mapped rows.

    Returns:
        (chars, rows): ``chars`` is the valid character token list (one
        per decode step; multi-char digit groups are single tokens, as in
        the legacy regex tokenization); ``rows`` the zresp row per
        character (same length). Characters sharing a row are allowed
        (they reuse the same brain feature).
    """
    mat_path = Path(mat_path)
    time_data = loadmat(str(mat_path))
    chars = [str(c).strip() for c in np.squeeze(time_data["char"])]
    onsets = np.squeeze(time_data["end"]).astype(np.float64)

    if eliminate:
        chars = [c for i, c in enumerate(chars) if i not in eliminate]
        onsets = np.delete(onsets, eliminate)

    rows = np.round((onsets - first_char_onset_time) / interval).astype(int)
    valid = np.ones(len(rows), dtype=bool)
    valid &= rows >= 0
    if n_rows is not None:
        valid &= rows < n_rows

    out_chars = [c for c, ok in zip(chars, valid) if ok and c != ""]
    out_rows = rows[valid]
    logger.info(
        "char_grid: %d chars -> %d valid (rows %d..%d)",
        len(chars), len(out_chars), out_rows.min() if len(out_rows) else -1,
        out_rows.max() if len(out_rows) else -1,
    )
    return out_chars, out_rows


def decode_input_rows(rows: np.ndarray, n_context: int = 5) -> np.ndarray:
    """Per-character model-input rows: the context window anchor.

    Training input for step ``t`` is ``resp_ctx[t - (n_context - 1)]``
    (zero-padded warmup before row ``n_context - 1``). So a character
    anchored at row ``r`` feeds the model the context window anchored at
    ``r - (n_context - 1)``.
    """
    return rows - (n_context - 1)


__all__ = ["char_grid", "decode_input_rows"]
