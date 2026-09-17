"""Reference text reconstruction for evaluation (D4).

The true story text is reconstructed from the BIDS char-time ``.mat``
files. The tokenization matches the legacy pipeline's regex behavior
(digit groups like "1600" are single tokens) — verified against the
golden wordvectors; the English-elimination list drops the same indices
as preprocessing. The reference for a story equals the valid character
sequence produced by ``recon.decoders.alignment.char_grid`` — the same
sequence the decoder is asked to produce, so lengths align.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..decoders.alignment import char_grid

logger = logging.getLogger(__name__)


def build_reference(
    mat_path: str | Path,
    eliminate: list[int] | None = None,
    n_rows: int | None = None,
) -> str:
    """Reference text for one story (one string, tokens concatenated)."""
    chars, _ = char_grid(mat_path, eliminate=eliminate, n_rows=n_rows)
    return "".join(chars)


def build_references(
    mat_dir: str | Path,
    out_dir: str | Path,
    stories: list[int],
    eliminate: dict[int, list[int]] | None = None,
    n_rows: int | None = None,
) -> list[Path]:
    """Build ``story{N}.txt`` references for a list of stories.

    Args:
        mat_dir: Directory with ``story_{N}_char_time.mat`` files.
        out_dir: Output directory for ``story{N}.txt``.
        stories: Story ids.
        eliminate: Legacy English-elimination dict (story → indices).
        n_rows: Optional grid length (aligns reference to decode length).

    Returns:
        Paths of the written reference files (stories with missing mats
        are skipped with a warning).
    """
    mat_dir = Path(mat_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for story in stories:
        mat = mat_dir / f"story_{story}_char_time.mat"
        if not mat.exists():
            logger.warning("mat missing for story %d — skipping reference", story)
            continue
        elim = (eliminate or {}).get(story)
        text = build_reference(mat, eliminate=elim, n_rows=n_rows)
        out = out_dir / f"story{story}.txt"
        out.write_text(text, encoding="utf-8")
        written.append(out)
        logger.info("reference story %d: %d chars -> %s", story, len(text), out)
    return written


__all__ = ["build_reference", "build_references"]
