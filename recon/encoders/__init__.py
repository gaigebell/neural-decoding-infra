"""Brain encoders that convert raw signals to feature vectors.

All encoders expose a uniform interface (see :class:`BaseBrainEncoder` in
``recon.encoders.base``) so the rest of the pipeline does not depend on
the underlying modality or model.

Heavy encoders (e.g., BrainOmni) use lazy imports; see
:mod:`recon.encoders.brainomni` and ``docs/standards/07-dependencies.md``.
"""