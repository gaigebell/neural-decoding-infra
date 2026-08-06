"""Helpers for handling optional dependencies.

These functions let the rest of the codebase remain importable even when
heavy optional packages (BrainOmni, vllm, etc.) are not installed.

Usage:
    from recon.utils.optional import require_optional

    def build_brainomni_encoder(cfg):
        require_optional("brainomni", hint="pip install -e '.[brainomni]'")
        # Now safe to import brainomni
        from brainomni.model import BrainOmni
        ...

See docs/standards/07-dependencies.md for the full conventions.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Final

logger = logging.getLogger(__name__)

# ───────────────────── Extra → primary package mapping ─────────────────────
# Each extra is named in pyproject.toml. We map it to a primary package
# used for the availability check (some extras wrap multiple packages).
_EXTRA_TO_PACKAGE: Final[dict[str, str]] = {
    "brainomni": "brainomni",
    "large-lm": "vllm",
    "data-public": "datasets",
    "all": "",  # special: skip check (assumes everything installed)
}


def is_installed(package: str) -> bool:
    """Check if a Python package is installed without importing it.

    Args:
        package: Name of the package (e.g., "brainomni").

    Returns:
        True if the package is importable, False otherwise.
    """
    if not package:
        return True
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def extra_installed(extra: str) -> bool:
    """Check if a named optional extra is installed.

    Args:
        extra: Name of the extra (e.g., "brainomni").

    Returns:
        True if the extra's primary package is installed.
    """
    package = _EXTRA_TO_PACKAGE.get(extra, extra)
    return is_installed(package)


def require_optional(extra: str, hint: str = "") -> None:
    """Raise ImportError if the given extra is not installed.

    Use at the entry point of any function that depends on an optional
    integration. This gives users a clear, actionable error instead of a
    bare ModuleNotFoundError.

    Args:
        extra: Name of the optional extra (e.g., "brainomni").
        hint: Optional additional install hint (e.g., "see docs/foo.md").

    Raises:
        ImportError: If the extra's primary package is not installed.
            The error message includes the install command.
    """
    if extra_installed(extra):
        return

    msg = (
        f"Optional dependency '{extra}' is not installed.\n"
        f"Install with: pip install -e '.[{extra}]'\n"
        f"(cluster users: pip install -e '.[all]')"
    )
    if hint:
        msg += f"\nOr: {hint}"
    raise ImportError(msg)


def lazy_import_check(extra: str, hint: str = "") -> None:
    """Alias for :func:`require_optional` for code that prefers this name."""
    require_optional(extra, hint=hint)