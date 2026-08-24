"""Logging utilities: standard Python logging + Weights & Biases wrapper.

Usage:
    >>> from recon.utils.logging import get_logger, WandBLogger
    >>> logger = get_logger(__name__)
    >>> logger.info("training started")
    >>> wb = WandBLogger(project="recon", config={"lr": 1e-4})
    >>> wb.log({"loss": 0.3}, step=10)
    >>> wb.finish()

Design rules:
- Use ``get_logger(__name__)`` everywhere. Never use ``print()``.
- WandB is optional: if not installed or not configured, the wrapper
  is a no-op. This keeps tests fast and lets local dev work offline.
- All log calls should go through the same interface, so swapping
  WandB for another tracker is a one-line change.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from .optional import is_installed

# ───────────────────── Standard Python logger ─────────────────────

# Default log format
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_root_logger() -> None:
    """Configure the root logger once at module import."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(os.environ.get("RECON_LOG_LEVEL", "INFO").upper())


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, using the project's format.

    Args:
        name: Logger name. Conventionally ``__name__``.

    Returns:
        A configured ``logging.Logger``.
    """
    return logging.getLogger(name)


# ───────────────────── Weights & Biases wrapper ─────────────────────


class WandBLogger:
    """Thin wrapper around Weights & Biases for experiment tracking.

    Why this exists:
    - We want a single interface for logging metrics.
    - If wandb is unavailable (offline / not installed), the wrapper
      is a no-op so the rest of the code does not need to know.
    - All wandb-specific logic is contained here, making it easy to
      swap for MLflow, TensorBoard, or a custom backend later.

    Usage:
        >>> wb = WandBLogger(project="recon", config={"lr": 1e-4})
        >>> wb.log({"loss": 0.3}, step=10)
        >>> wb.log({"val/crr": 0.12}, step=10, prefix_strip=False)
        >>> wb.finish()
    """

    def __init__(
        self,
        project: str | None = None,
        config: dict[str, Any] | None = None,
        name: str | None = None,
        run_id: str | None = None,
        resume: bool = False,
        mode: str | None = None,
        dir: str | Path | None = None,
        **kwargs: Any,
    ):
        """Initialize the WandB logger.

        Args:
            project: WandB project name. Defaults to env var WANDB_PROJECT.
            config: Config dict (hyperparameters, model info, etc.).
            name: Display name for this run.
            run_id: Resume from a specific run ID.
            resume: Whether to resume the run.
            mode: "online", "offline", or "disabled". Defaults to env var WANDB_MODE.
            dir: Where to store wandb files. Defaults to ./wandb/.
            **kwargs: Forwarded to ``wandb.init``.
        """
        # wandb is a CORE dependency (not an extra) — check the package itself.
        self._enabled = is_installed("wandb") and os.environ.get(
            "WANDB_DISABLED", ""
        ).lower() not in ("1", "true", "yes")

        if not self._enabled:
            self._run = None
            self._log_count = 0
            get_logger(__name__).warning(
                "WandBLogger initialized in DISABLED mode (no logging will occur). "
                "Reason: %s",
                "wandb not installed" if not is_installed("wandb") else "WANDB_DISABLED",
            )
            return

        # Lazy import wandb only if we know we want it
        import wandb  # noqa: PLC0415

        project = project or os.environ.get("WANDB_PROJECT", "neural-decoding-infra")
        mode = mode or os.environ.get("WANDB_MODE", "online")

        self._run = wandb.init(
            project=project,
            config=config,
            name=name,
            id=run_id,
            resume="allow" if resume else None,
            mode=mode,
            dir=str(dir) if dir else None,
            **kwargs,
        )
        self._log_count = 0
        get_logger(__name__).info(
            "WandB run started: %s (project=%s, mode=%s)",
            self._run.url if self._run else "(disabled)",
            project,
            mode,
        )

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log a dict of metrics to WandB.

        Args:
            metrics: Dict of metric name to value. Values can be scalar
                numbers or tensors / numpy arrays (will be converted).
            step: Global step. If None, WandB auto-increments.
            commit: Whether to commit the step (default True).
        """
        if not self._enabled or self._run is None:
            return

        # Lazy import
        import wandb  # noqa: PLC0415

        # Convert tensors/numpy to plain Python
        clean: dict[str, Any] = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = v.item()
            clean[k] = v

        wandb.log(clean, step=step, commit=commit)
        self._log_count += 1

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        type: str = "model",
        description: str | None = None,
    ) -> None:
        """Log a file or directory as a WandB artifact.

        Args:
            path: Path to the file or directory.
            name: Name of the artifact.
            type: Type ("model", "dataset", "code", etc.).
            description: Optional description.
        """
        if not self._enabled or self._run is None:
            return

        import wandb  # noqa: PLC0415

        artifact = wandb.Artifact(name=name, type=type, description=description)
        if Path(path).is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def watch_model(
        self,
        model: Any,
        criterion: Any | None = None,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch a PyTorch model for gradient / parameter logging.

        Args:
            model: The model to watch.
            criterion: Optional loss function.
            log: What to log: "gradients", "parameters", "all", or None.
            log_freq: How often to log (in steps).
        """
        if not self._enabled or self._run is None:
            return
        self._run.watch(model, criterion=criterion, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish the WandB run. Idempotent."""
        if not self._enabled or self._run is None:
            return
        import wandb  # noqa: PLC0415
        wandb.finish()
        get_logger(__name__).info(
            "WandB run finished. Logged %d metric groups.", self._log_count
        )
        self._run = None

    @property
    def enabled(self) -> bool:
        """Whether this logger is actually doing anything."""
        return self._enabled

    @property
    def run_id(self) -> str | None:
        """The current run's ID, or None if disabled."""
        if self._run is None:
            return None
        return self._run.id


__all__ = ["WandBLogger", "get_logger"]