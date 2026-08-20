"""End-to-end smoke test: data → model → trainer (1 step, CPU only).

This is the **Tier 0 smoke** test. It must pass on any machine (no GPU,
no real data, no extras). It catches:
- Import errors
- Shape mismatches
- Schema validation errors
- Hydra config composition errors

This is the test that runs in CI (see ``.github/workflows/lint.yml``).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


# ───────────────────── Test 1: Recon imports cleanly ─────────────────────


def test_recon_imports():
    """All recon submodules should import without errors."""
    import recon
    import recon.data
    import recon.data.schema
    import recon.data.fake_data
    import recon.data.drdr
    import recon.data.datasets.meg
    import recon.data.datasets.fmri
    import recon.models
    import recon.models.fmri.fmri3dcib
    import recon.models.meg.meg_model_a
    import recon.engine
    import recon.engine.evaluator
    import recon.decoders
    import recon.decoders.beam
    import recon.cli
    import recon.cli.train
    import recon.cli.decode
    import recon.cli.eval
    import recon.utils
    import recon.utils.optional
    import recon.utils.logging
    assert recon.__version__ is not None


# ───────────────────── Test 2: Pydantic schema round-trip ─────────────────────


def test_schema_roundtrip_meg():
    """Fake data → pydantic → tensor should produce valid data."""
    import numpy as np
    from recon.data.fake_data import fake_pair_meg
    from recon.data.schema import MEGSample

    rng = np.random.default_rng(42)
    pair = fake_pair_meg(rng, story_id=1, subject_id=1)
    assert isinstance(pair.brain, MEGSample)
    # Convert to torch tensor — this is what the trainer does
    import torch
    x = torch.from_numpy(pair.brain.x).float()
    assert x.shape == (306, 256)


def test_schema_roundtrip_fmri():
    """fMRI fake data should round-trip through schema."""
    import numpy as np
    from recon.data.fake_data import fake_pair_fmri
    import torch

    rng = np.random.default_rng(42)
    pair = fake_pair_fmri(rng, story_id=1, subject_id=1)
    vol = torch.from_numpy(pair.brain.volume).float()
    assert vol.shape == (53, 63, 52)


# ───────────────────── Test 3: Model forward + loss ─────────────────────


def test_fmri_model_end_to_end():
    """FMRI model: forward → loss → backward should work on CPU."""
    import torch
    from recon.models.fmri.fmri3dcib import FMRI3DCIBConfig, FMRI3DCIBModel

    torch.set_num_threads(2)
    cfg = FMRI3DCIBConfig(input_shape=(53, 63, 52))
    model = FMRI3DCIBModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 1, 53, 63, 52)
    target = torch.randn(2, 768)
    out, kl = model(x)
    loss, _ = model.compute_loss(out, target, aux={"kl_loss": kl})

    opt.zero_grad()
    loss.backward()
    opt.step()
    assert loss.item() > 0


def test_meg_model_end_to_end():
    """MEG model: forward → loss → backward should work on CPU."""
    import torch
    from recon.models.meg.meg_model_a import MEGModelA, MEGModelAConfig

    torch.set_num_threads(2)
    cfg = MEGModelAConfig(n_channels=306, n_context=5)
    model = MEGModelA(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 5, 306)
    target = torch.randn(2, 768)
    out, _ = model(x)
    loss, _ = model.compute_loss(out, target)

    opt.zero_grad()
    loss.backward()
    opt.step()
    assert loss.item() > 0


# ───────────────────── Test 4: Trainer (no real data, no GPU) ─────────────────────


def test_trainer_runs_one_step_with_fake_data():
    """Trainer should run 1 step with fake data on CPU.

    This is the most comprehensive smoke test. It exercises:
    - Hydra config composition
    - Model building via registry
    - DataLoader construction
    - Forward + backward
    - Optimizer step
    - Logging

    We use ``max_steps_per_epoch=2`` to keep the test fast.
    """
    from omegaconf import OmegaConf

    from recon.engine.trainer import Trainer

    cfg = OmegaConf.create(
        {
            "model": {"name": "meg_model_a", "n_channels": 306, "n_context": 5},
            "data": {"name": "fake", "modality": "meg", "n_context": 5, "n_samples": 16},
            "paths": {
                "data_root": "/tmp",
                "results_dir": "/tmp/recon_results",
                "pretrained_root": "/tmp/pretrained",
            },
            "train": {
                "epochs": 1,
                "batch_size": 2,
                "amp": False,
                "num_workers": 0,
                "max_steps_per_epoch": 2,
                "log_interval": 1,
                "smoke": True,
            },
        }
    )

    trainer = Trainer(cfg, rank=0, world_size=1)
    train_loader, val_loader = trainer.build_dataloaders()
    assert len(train_loader) > 0
    assert val_loader is None  # smoke mode skips val

    # Run one training epoch
    metrics = trainer._train_epoch(train_loader, epoch=0)
    assert "total_loss" in metrics
    assert metrics["total_loss"] > 0


# ───────────────────── Test 5: Config composition ─────────────────────


def test_hydra_config_loads():
    """The default train.yaml config should compose correctly via Hydra defaults.

    ``OmegaConf.load`` alone only reads the raw file — the ``model``/``data``/
    ``paths`` groups come from Hydra ``defaults`` composition. So we use the
    real composition path (same as ``python -m recon.cli.train`` at startup).
    """
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="train")
    assert cfg.train.epochs == 100
    assert cfg.train.batch_size == 8
    assert cfg.model.name == "fmri3dcib"
    assert cfg.paths.data_root is not None


def test_path_override_local():
    """Switching to paths=local should change the data_root."""
    from omegaconf import OmegaConf

    base = OmegaConf.load("configs/train.yaml")
    local = OmegaConf.load("configs/paths/local.yaml")
    merged = OmegaConf.merge(base, {"paths": local})
    assert "E:" in merged.paths.data_root


# ───────────────────── Test 6: No GPU required ─────────────────────


def test_no_cuda_required():
    """The smoke test path should work even when CUDA is unavailable.

    This is enforced by running the entire test suite with
    ``CUDA_VISIBLE_DEVICES=""`` in CI. We just verify the imports
    and basic operations don't accidentally require CUDA.
    """
    import torch

    # Basic CPU operations
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    c = a @ b
    assert c.shape == (3, 3)