"""Unit tests for the built-in models.

Verifies:
- Forward pass produces correct output shape
- Loss is computed and is differentiable
- Backward + optimizer step works
- Model is registered in MODEL_REGISTRY
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from recon.models.registry import available_models, build_model, is_registered
from recon.models.fmri.fmri3dcib import FMRI3DCIBConfig, FMRI3DCIBModel
from recon.models.meg.meg_model_a import MEGModelA, MEGModelAConfig


# ───────────────────── FMRI3DCIBModel ─────────────────────


class TestFMRI3DCIBModel:
    """Tests for the fMRI 3D-CNN + IB model."""

    def test_forward_shape(self):
        """Forward should produce (B, 768) output."""
        cfg = FMRI3DCIBConfig(input_shape=(53, 63, 52), semantic_dim=768)
        model = FMRI3DCIBModel(cfg)
        model.eval()
        x = torch.randn(2, 1, 53, 63, 52)
        with torch.no_grad():
            out, kl = model(x)
        assert out.shape == (2, 768)
        assert kl is not None  # IB produces a KL term

    def test_forward_4d_input(self):
        """Forward should accept (B, X, Y, Z) and add channel dim."""
        cfg = FMRI3DCIBConfig(input_shape=(53, 63, 52))
        model = FMRI3DCIBModel(cfg)
        model.eval()
        x = torch.randn(2, 53, 63, 52)
        with torch.no_grad():
            out, _ = model(x)
        assert out.shape == (2, 768)

    def test_compute_loss_returns_scalar(self):
        """compute_loss should return (loss, dict) with scalar loss."""
        cfg = FMRI3DCIBConfig()
        model = FMRI3DCIBModel(cfg)
        pred = torch.randn(4, 768)
        target = torch.randn(4, 768)
        aux = {"kl_loss": torch.tensor(0.1)}
        loss, loss_dict = model.compute_loss(pred, target, aux=aux)
        assert loss.ndim == 0  # scalar
        assert "total_loss" in loss_dict
        assert "mse_loss" in loss_dict
        assert "cosine_loss" in loss_dict
        assert "kl_loss" in loss_dict

    def test_backward_works(self):
        """Loss should be differentiable and backward should work."""
        cfg = FMRI3DCIBConfig()
        model = FMRI3DCIBModel(cfg)
        x = torch.randn(2, 1, 53, 63, 52)
        target = torch.randn(2, 768)
        out, kl = model(x)
        loss, _ = model.compute_loss(out, target, aux={"kl_loss": kl})
        loss.backward()
        # At least one parameter should have a gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g.abs().sum() > 0 for g in grads)

    def test_registered_in_registry(self):
        """FMRI3DCIB should be registered as 'fmri3dcib'."""
        assert is_registered("fmri3dcib")
        assert "fmri3dcib" in available_models()


# ───────────────────── MEGModelA ─────────────────────


class TestMEGModelA:
    """Tests for the MEG spatial-temporal model."""

    def test_forward_3d_input(self):
        """Forward should accept (B, n_context, n_channels) shape."""
        cfg = MEGModelAConfig(n_channels=306, n_context=5, embed_dim=256)
        model = MEGModelA(cfg)
        model.eval()
        x = torch.randn(2, 5, 306)
        with torch.no_grad():
            out, aux = model(x)
        assert out.shape == (2, 768)
        assert aux is None  # MEG model has no auxiliary loss

    def test_forward_2d_input(self):
        """Forward should accept (B, n_channels) and add time dim."""
        cfg = MEGModelAConfig()
        model = MEGModelA(cfg)
        model.eval()
        x = torch.randn(2, 306)
        with torch.no_grad():
            out, _ = model(x)
        assert out.shape == (2, 768)

    def test_compute_loss(self):
        """compute_loss should return (loss, dict)."""
        cfg = MEGModelAConfig()
        model = MEGModelA(cfg)
        pred = torch.randn(4, 768)
        target = torch.randn(4, 768)
        loss, loss_dict = model.compute_loss(pred, target)
        assert loss.ndim == 0
        assert "total_loss" in loss_dict

    def test_backward_works(self):
        """Backward pass should populate gradients."""
        cfg = MEGModelAConfig()
        model = MEGModelA(cfg)
        x = torch.randn(2, 5, 306)
        target = torch.randn(2, 768)
        out, _ = model(x)
        loss, _ = model.compute_loss(out, target)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_registered_in_registry(self):
        """MEG model A should be registered as 'meg_model_a'."""
        assert is_registered("meg_model_a")


# ───────────────────── Registry ─────────────────────


class TestRegistry:
    """Tests for MODEL_REGISTRY behavior."""

    def test_build_unknown_raises(self):
        """Building an unregistered model should raise ValueError."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"name": "nonexistent_model_xyz"})
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(cfg)

    def test_build_fmri3dcib(self):
        """Building 'fmri3dcib' should produce an FMRI3DCIBModel."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"name": "fmri3dcib", "input_shape": [53, 63, 52]})
        model = build_model(cfg)
        assert isinstance(model, FMRI3DCIBModel)

    def test_build_meg_model_a(self):
        """Building 'meg_model_a' should produce a MEGModelA."""
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"name": "meg_model_a", "n_channels": 306, "n_context": 5})
        model = build_model(cfg)
        assert isinstance(model, MEGModelA)


# ───────────────────── Integration smoke (no GPU) ─────────────────────


class TestSmokeCPU:
    """End-to-end smoke test on CPU. No GPU, no real data."""

    def test_full_fmri_forward_backward_on_cpu(self):
        """Full forward + backward of fMRI model on CPU."""
        torch.set_num_threads(2)
        cfg = FMRI3DCIBConfig()
        model = FMRI3DCIBModel(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # Tiny input
        x = torch.randn(1, 1, 53, 63, 52)
        target = torch.randn(1, 768)
        for _ in range(2):
            out, kl = model(x)
            loss, _ = model.compute_loss(out, target, aux={"kl_loss": kl})
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Just verify it doesn't crash
        assert loss.item() > 0

    def test_full_meg_forward_backward_on_cpu(self):
        """Full forward + backward of MEG model on CPU."""
        torch.set_num_threads(2)
        cfg = MEGModelAConfig()
        model = MEGModelA(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(1, 5, 306)
        target = torch.randn(1, 768)
        for _ in range(2):
            out, _ = model(x)
            loss, _ = model.compute_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() > 0