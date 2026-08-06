"""Model registry: name → builder function.

Adding a model:
    >>> from .registry import register_model
    >>> @register_model("my_model")
    ... def build_my_model(cfg):
    ...     return MyModel(...)

For models that depend on optional extras (e.g., BrainOmni), the
builder should call :func:`recon.utils.optional.require_optional`
BEFORE doing any heavy import. See ``recon.encoders.brainomni``.

Building a model from config:
    >>> from omegaconf import OmegaConf
    >>> from .registry import build_model
    >>> cfg = OmegaConf.load("configs/model/fmri3dcib.yaml")
    >>> model = build_model(cfg)
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type alias for a model builder function
ModelBuilder = Callable[..., Any]


# ───────────────────── Registry storage ─────────────────────
# Maps: model name → builder function
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str):
    """Decorator to register a model builder.

    Args:
        name: Unique model name (matches ``configs/model/<name>.yaml``).

    Returns:
        Decorator function.

    Raises:
        ValueError: If a model with this name is already registered.

    Example:
        >>> @register_model("fmri3dcib")
        ... def build_fmri3dcib(cfg):
        ...     return Fmri3dCIBModel(...)
    """
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Choose a different name or check for duplicate imports."
            )
        _MODEL_REGISTRY[name] = builder
        logger.debug("Registered model: %s", name)
        return builder
    return decorator


def available_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Return True if a model with this name is registered."""
    return name in _MODEL_REGISTRY


def build_model(cfg) -> Any:
    """Build a model from a Hydra/OmegaConf config.

    The config must have a ``name`` field naming a registered model.

    Args:
        cfg: OmegaConf config or dict with at least a ``name`` field.

    Returns:
        The constructed model.

    Raises:
        ValueError: If the model name is unknown.
        ImportError: If the model requires an optional extra that is not
            installed (raised by the builder via ``require_optional``).
    """
    # Support both OmegaConf and dict
    name = cfg["name"] if hasattr(cfg, "__getitem__") else cfg.name

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'.\n"
            f"Available models: {available_models()}\n"
            f"Check your config (configs/model/<name>.yaml) or "
            f"see docs/guides/03-write-model.md"
        )

    builder = _MODEL_REGISTRY[name]
    logger.info("Building model: %s", name)
    return builder(cfg)


def _register_builtins() -> None:
    """Import built-in model modules so their @register_model decorators run.

    Add new model modules to this function when you create them.
    Each import triggers the @register_model decorators in the module,
    which populate ``_MODEL_REGISTRY``.

    Modules that depend on optional extras (e.g., brainomni) are imported
    inside try/except so the core registry still works without those
    extras installed.
    """
    # ─────────────── Core models (always importable) ───────────────
    # Add imports here as you create the model files. For now, this is
    # a placeholder — actual fMRI/MEG model files will be added in
    # subsequent PRs.

    # from .fmri import fmri3dcib, fmri3dcib2, fmri_bnib, fmri_bnatt, fmri_catt
    # from .meg import meg_model_a

    # ─────────────── Optional models (require extras) ───────────────
    # These imports succeed regardless of whether the extra is installed,
    # because the modules themselves use lazy imports internally. Only
    # the *builder call* (build_model → builder()) will fail with a
    # friendly ImportError if the extra is missing.
    try:
        from .brainomni import align_mlp  # noqa: F401
    except ImportError:
        # Module file missing — that's OK during early bootstrap
        pass


_register_builtins()


__all__ = [
    "ModelBuilder",
    "available_models",
    "build_model",
    "is_registered",
    "register_model",
]