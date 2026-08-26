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
                "amp_dtype": None,
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


# ───────────────────── Test 5: Startup checks + NaN guard ─────────────────────


def _smoke_cfg(tmp_path: Path):
    """Smoke config for Trainer construction (fake data, CPU-safe)."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model": {"name": "meg_model_a", "n_channels": 306, "n_context": 5},
            "data": {"name": "fake", "modality": "meg", "n_context": 5, "n_samples": 16},
            "paths": {
                "data_root": "/tmp",
                "processed_root": "/tmp",
                "results_dir": str(tmp_path),
                "pretrained_root": "/tmp/pretrained",
            },
            "train": {
                "epochs": 1,
                "batch_size": 2,
                "amp_dtype": None,
                "num_workers": 0,
                "max_steps_per_epoch": 2,
                "log_interval": 1,
                "smoke": True,
                "save_interval": 10,
                "ckpt_dir": f"{tmp_path}/ckpt/${{run_id}}",
            },
            # run_id must exist or the ${run_id} interpolation fails on resolve
            "run_id": "test",
            # Keep wandb out of tests (mode=disabled = no network, no files)
            "logging": {"wandb_mode": "disabled"},
        }
    )


def test_fit_runs_startup_checks_and_metadata(tmp_path: Path):
    """fit() must run the fail-fast checks and write run_metadata.json."""
    import json

    from recon.engine.trainer import Trainer

    trainer = Trainer(_smoke_cfg(tmp_path), rank=0, world_size=1)
    trainer.fit()  # startup checks (device/data/dry-run) + 1 smoke epoch

    metadata_path = Path(trainer.train_cfg.ckpt_dir) / "run_metadata.json"
    assert metadata_path.exists()
    meta = json.loads(metadata_path.read_text())
    assert meta["model"]["name"] == "meg_model_a"
    assert meta["model"]["n_params"] > 0
    # smoke mode overrides fake-data size to 32 (trainer behavior)
    assert meta["data"]["n_train_samples"] == 32
    assert meta["config"]["train"]["smoke"] is True


def test_nan_loss_aborts(tmp_path: Path, monkeypatch):
    """Non-finite loss must raise with abort_on_nan=true (default)."""
    import torch
    import pytest

    from recon.engine.trainer import Trainer

    trainer = Trainer(_smoke_cfg(tmp_path), rank=0, world_size=1)
    train_loader, _ = trainer.build_dataloaders()

    def fake_loss(pred, target, aux=None):
        return torch.tensor(float("nan")), {"total_loss": float("nan")}

    monkeypatch.setattr(trainer, "_model_compute_loss", fake_loss)
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        trainer._train_epoch(train_loader, epoch=0)


def test_nan_loss_skips_when_disabled(tmp_path: Path, monkeypatch):
    """abort_on_nan=false must skip the step and finish the epoch."""
    import torch

    from recon.engine.trainer import Trainer

    cfg = _smoke_cfg(tmp_path)
    cfg.train.abort_on_nan = False
    trainer = Trainer(cfg, rank=0, world_size=1)
    train_loader, _ = trainer.build_dataloaders()

    def fake_loss(pred, target, aux=None):
        return torch.tensor(float("nan")), {"total_loss": float("nan")}

    monkeypatch.setattr(trainer, "_model_compute_loss", fake_loss)
    metrics = trainer._train_epoch(train_loader, epoch=0)
    # All steps were skipped -> no batches accumulated -> default metrics
    assert metrics["total_loss"] == 0.0


def test_fit_with_val_loop_saves_best(tmp_path: Path):
    """Non-smoke fit() must run validation and save best_val.pt."""
    from recon.engine.trainer import Trainer

    cfg = _smoke_cfg(tmp_path)
    cfg.train.smoke = False  # fake builder then creates a val set (8 pairs)
    cfg.train.eval_interval = 1
    trainer = Trainer(cfg, rank=0, world_size=1)
    trainer.fit()

    ckpt_dir = Path(trainer.train_cfg.ckpt_dir)
    assert (ckpt_dir / "best_val.pt").exists()
    assert trainer._best_val_loss is not None
    assert trainer._best_val_loss > 0


# ───────────────────── Test 6: Config composition ─────────────────────


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


# ───────────────────── Test 7: No GPU required ─────────────────────


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