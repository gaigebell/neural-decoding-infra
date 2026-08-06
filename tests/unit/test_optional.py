"""Unit tests for recon.utils.optional.

These tests do NOT require any optional extra to be installed. They
verify the helper functions behave correctly whether or not BrainOmni,
vllm, etc. are present.

To run:
    pytest tests/unit/test_optional.py -v
"""
from __future__ import annotations

import pytest

from recon.utils.optional import (
    extra_installed,
    is_installed,
    require_optional,
)


# ───────────────────── is_installed ─────────────────────

class TestIsInstalled:
    """Tests for :func:`is_installed`."""

    def test_returns_true_for_stdlib(self):
        """Standard library packages should always be detected."""
        assert is_installed("os") is True
        assert is_installed("sys") is True
        assert is_installed("json") is True

    def test_returns_true_for_recon_deps(self):
        """Recon's core dependencies should be installed in any test env."""
        # These are in pyproject.toml [project.dependencies]
        for pkg in ["numpy", "pydantic"]:
            assert is_installed(pkg) is True, f"Expected {pkg} to be installed"

    def test_returns_false_for_nonexistent(self):
        """A package that doesn't exist should return False (not raise)."""
        assert is_installed("definitely_not_a_real_package_xyz_123") is False

    def test_empty_string_returns_true(self):
        """Empty string is treated as 'always available' (used for 'all' extra)."""
        assert is_installed("") is True


# ───────────────────── extra_installed ─────────────────────

class TestExtraInstalled:
    """Tests for :func:`extra_installed`."""

    def test_brainomni_returns_bool(self):
        """Returns True or False without raising."""
        result = extra_installed("brainomni")
        assert isinstance(result, bool)

    def test_unknown_extra_returns_false(self):
        """An extra not in the mapping returns False."""
        assert extra_installed("nonexistent_extra") is False


# ───────────────────── require_optional ─────────────────────

class TestRequireOptional:
    """Tests for :func:`require_optional`."""

    def test_silent_when_installed(self):
        """Should not raise when the extra is installed."""
        # Pick an extra that is definitely installed (mapped to stdlib)
        # We can't actually map to stdlib, but we can map "brainomni"
        # which either is or isn't installed — both must not raise
        # unexpectedly.
        # Use the helper that doesn't raise:
        if extra_installed("brainomni"):
            # Should be silent
            require_optional("brainomni")
        else:
            # Should raise ImportError
            with pytest.raises(ImportError, match="brainomni"):
                require_optional("brainomni")

    def test_raises_with_install_hint(self):
        """When not installed, error message includes install hint."""
        # brainomni is unlikely to be installed in CI
        if extra_installed("brainomni"):
            pytest.skip("brainomni is installed; cannot test error case")

        with pytest.raises(ImportError) as exc_info:
            require_optional("brainomni", hint="also check docs/foo.md")

        msg = str(exc_info.value)
        assert "brainomni" in msg
        assert "pip install -e" in msg
        assert "[brainomni]" in msg
        assert "docs/foo.md" in msg  # custom hint included

    def test_no_hint_still_works(self):
        """Default error message is sufficient when no hint is provided."""
        if extra_installed("brainomni"):
            pytest.skip("brainomni is installed; cannot test error case")

        with pytest.raises(ImportError) as exc_info:
            require_optional("brainomni")

        msg = str(exc_info.value)
        assert "brainomni" in msg
        assert "[brainomni]" in msg
        # No custom hint
        assert "Or:" not in msg


# ───────────────────── Integration: registry ─────────────────────

class TestRegistryIntegration:
    """Verify MODEL_REGISTRY works regardless of extras."""

    def test_registry_importable(self):
        """Importing the registry should not require any extras."""
        from recon.models.registry import (
            available_models,
            is_registered,
            register_model,
        )
        # Functions should exist
        assert callable(available_models)
        assert callable(is_registered)
        assert callable(register_model)

    def test_build_unknown_raises_value_error(self):
        """Building an unregistered model should raise ValueError, not ImportError."""
        from omegaconf import OmegaConf
        from recon.models.registry import build_model

        cfg = OmegaConf.create({"name": "definitely_not_registered"})
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(cfg)

    def test_register_then_build(self):
        """A model registered at runtime should be buildable."""
        from omegaconf import OmegaConf
        from recon.models.registry import build_model, register_model

        @register_model("_test_dummy_model")
        def build_dummy(cfg):
            return f"dummy_with_{cfg.get('value', 'default')}"

        try:
            cfg = OmegaConf.create({"name": "_test_dummy_model", "value": 42})
            result = build_model(cfg)
            assert result == "dummy_with_42"
        finally:
            # Cleanup: remove from registry
            from recon.models.registry import _MODEL_REGISTRY
            _MODEL_REGISTRY.pop("_test_dummy_model", None)