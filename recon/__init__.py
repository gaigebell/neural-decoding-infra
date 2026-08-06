"""neural-decoding-infra: AI infrastructure for brain-to-text decoding.

A modular Python package for training and evaluating models that map
non-invasive brain signals (fMRI, MEG) to Chinese character sequences.

Importing this package is always safe; heavy integrations (BrainOmni,
vLLM, public datasets) are lazy-loaded via :mod:`recon.utils.optional`.

See:
    - docs/architecture/01-overview.md — system architecture
    - docs/standards/07-dependencies.md — dependency management
"""

__version__ = "0.0.0"