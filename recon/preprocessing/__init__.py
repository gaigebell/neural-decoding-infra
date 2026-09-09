"""Preprocessing pipeline: raw BIDS data → training artifacts (zresp/zstim).

Design: docs/planning/data-pipeline-design.md and
docs/planning/preprocessing-phase.md. Every artifact is written with a
sidecar (``*.meta.json``: input hashes + params + git commit), enabling
idempotent re-runs and the cluster runner's skip-if-valid semantics.

Stages (via ``python -m recon.cli.preprocess``):
- ``gpt_char_features``   .mat chars → per-layer GPT-2 features (wordvectors)
- ``semantic_downsample`` wordvectors → lanczos 0.4s grid (ds_*.npy)
- ``semantic_delay``      ds → zscore + 4-delay concat (zstim)
- ``meg_zresp``           fif → zresp (classic path)
- ``meg_context``         zresp → context-windowed (T, n_context, C)
- ``fmri_cube``           nii.gz → cube zresp
"""

from . import brainomni, common, fmri, meg, semantic

__all__ = ["brainomni", "common", "fmri", "meg", "semantic"]
