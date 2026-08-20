"""Data layer for the recon package.

Components:
    - schema: Pydantic types for all data flowing through the system
    - fake_data: Synthetic data generators for tests and smoke tests
    - drdr: Adapter for the lab's primary dataset (DRDR)
    - datasets: PyTorch Dataset wrappers

Quickstart (smoke test):
    >>> from recon.data.fake_data import fake_pair_meg
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> pair = fake_pair_meg(rng, story_id=1, subject_id=1)
    >>> pair.brain.x.shape
    (306, 256)
    >>> pair.stim.zstim.shape
    (768,)

Quickstart (real data):
    >>> from recon.data.drdr import discover_drdr
    >>> index = discover_drdr("/home/test/reconstruction")
    >>> print(f"Found {index.n_subjects()} subjects, {index.n_stories()} stories")
"""

from . import datasets
from .collate import (
    BrainBatch,
    BrainStimBatch,
    StimBatch,
    collate_brain_stim_pairs,
)
from .schema import (
    BrainOmniSample,
    BrainStimPair,
    MEGChunkedSample,
    MEGSample,
    StimSample,
    fMRISample,
)

__all__ = [
    "BrainBatch",
    "BrainOmniSample",
    "BrainStimBatch",
    "BrainStimPair",
    "MEGChunkedSample",
    "MEGSample",
    "StimBatch",
    "StimSample",
    "collate_brain_stim_pairs",
    "datasets",
    "fMRISample",
]