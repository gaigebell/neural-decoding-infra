"""Brain-to-semantic models.

All models registered in :data:`recon.models.registry.MODEL_REGISTRY`
must implement a uniform interface (forward + compute_loss) so the
training engine can use them without modification.

Adding a new model:
    1. Create a file in ``recon/models/<kind>/<name>.py``
    2. Decorate the builder with ``@register_model("<name>")``
    3. For heavy dependencies, call ``require_optional("<extra>")``
       inside the builder (see ``recon.encoders.brainomni`` for example)
    4. Add a Hydra config at ``configs/model/<name>.yaml``

See ``docs/guides/03-write-model.md`` for a step-by-step walkthrough.
"""

from .registry import (
    available_models,
    build_model,
    register_model,
)

__all__ = ["available_models", "build_model", "register_model"]