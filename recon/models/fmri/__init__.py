"""fMRI brain-to-semantic models.

Each model:
- Takes a 4D fMRI volume (batch, 1, X, Y, Z) and produces a (batch, 768)
  semantic vector.
- Implements ``forward`` and ``compute_loss``.
- Is registered in :mod:`recon.models.registry` via ``@register_model``.

Adding a new fMRI model:
    1. Add a new file in this directory.
    2. Decorate the builder with ``@register_model("<name>")``.
    3. Add a config at ``configs/model/<name>.yaml``.

See ``docs/guides/03-write-model.md`` for a step-by-step.
"""