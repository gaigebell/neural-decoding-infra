"""BrainOmni encoder integration.

This module wraps the BrainOmni foundation model (a pretrained universal
brain feature extractor) so it can be used in ``recon``'s data pipeline.

Heavy dependency: requires BrainOmni to be installed.

Install with:
    pip install -e ".[brainomni]"

The cluster-side install of ``[all]`` includes BrainOmni.

This module uses lazy imports so the core ``recon`` package works without
BrainOmni installed — see ``docs/standards/07-dependencies.md``.

Example:
    >>> from recon.encoders.brainomni import build_brainomni_encoder, BrainOmniConfig
    >>> cfg = BrainOmniConfig(brainomni_path="/home/test/pretrained/brainomni")
    >>> encoder = build_brainomni_encoder(cfg)
    >>> features = encoder.encode({"x": ..., "pos": ..., "sensor_type": ...})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.optional import extra_installed, require_optional

logger = logging.getLogger(__name__)


# ───────────────────── Configuration ─────────────────────

@dataclass
class BrainOmniConfig:
    """Configuration for the BrainOmni encoder.

    Attributes:
        brainomni_path: Path to the BrainOmni checkpoint directory.
        cache_dir: Optional HuggingFace cache directory.
        device: Compute device, default "cuda".
        use_fp16: Whether to load in fp16 (saves memory).
    """

    brainomni_path: str | Path
    cache_dir: str | Path | None = None
    device: str = "cuda"
    use_fp16: bool = False


# ───────────────────── Public API ─────────────────────

def is_brainomni_available() -> bool:
    """Return True if BrainOmni is installed and importable."""
    return extra_installed("brainomni")


def build_brainomni_encoder(cfg: BrainOmniConfig) -> "BrainOmniEncoder":
    """Build a BrainOmni encoder from configuration.

    This is the canonical entry point. It performs the lazy import,
    raises a friendly error if BrainOmni is missing, and returns a
    wrapper that conforms to recon's encoder interface.

    Args:
        cfg: BrainOmni configuration.

    Returns:
        A configured :class:`BrainOmniEncoder`.

    Raises:
        ImportError: If BrainOmni is not installed. The error message
            includes the install command.
    """
    # Friendly error if extra not installed
    require_optional(
        "brainomni",
        hint="Or on cluster: pip install -e '.[all]'",
    )

    # ────────────── Lazy imports (only happen here) ──────────────
    # Inside the function so the module is importable without BrainOmni.
    from brainomni.model import BrainOmni  # noqa: E402
    from braintokenizer.model import BrainTokenizer  # noqa: E402

    logger.info("Loading BrainOmni from %s", cfg.brainomni_path)

    tokenizer = BrainTokenizer.from_pretrained(
        str(cfg.brainomni_path),
        cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
    )
    model = BrainOmni.from_pretrained(
        str(cfg.brainomni_path),
        cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None,
    )

    if cfg.use_fp16:
        # Cast to fp16 if requested (Pascal GPUs do not support bf16)
        model = model.half()

    encoder = BrainOmniEncoder(model=model, tokenizer=tokenizer, cfg=cfg)
    encoder.to(cfg.device)
    return encoder


# ───────────────────── Encoder wrapper ─────────────────────

class BrainOmniEncoder:
    """Wrap BrainOmni for use in recon's data pipeline.

    Conforms to the encoder interface used by the rest of the system.
    Heavy dependencies (BrainOmni model + tokenizer) are loaded once at
    construction time.

    Attributes:
        model: The loaded BrainOmni model (eval mode).
        tokenizer: The associated BrainTokenizer.
        cfg: The configuration used to build this encoder.
    """

    def __init__(self, model: Any, tokenizer: Any, cfg: BrainOmniConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model.eval()

    def encode(self, batch: dict) -> Any:
        """Encode a batch of brain signals.

        Args:
            batch: Dict with keys ``x``, ``pos``, ``sensor_type`` as
                required by BrainOmni. See BrainOmni's docs for shapes.

        Returns:
            Encoded features. Shape depends on BrainOmni's output config.
        """
        # Lazy import torch here too (in case user redefines later)
        import torch

        with torch.no_grad():
            return self.model.encode(**batch)

    def to(self, device: str) -> "BrainOmniEncoder":
        """Move model to device. Returns self for chaining."""
        self.model = self.model.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"BrainOmniEncoder(cfg={self.cfg}, "
            f"device={next(self.model.parameters()).device})"
        )