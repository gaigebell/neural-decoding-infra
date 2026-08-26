"""Unified training engine.

The Trainer is the **single entry point** for training any model registered
in :data:`recon.models.registry.MODEL_REGISTRY`. It handles:

- Model construction (from config via the registry)
- Optimizer + scheduler setup
- Mixed precision (AMP)
- Gradient clipping
- Distributed coordination (DDP, single- and multi-node)
- Checkpointing (save/load)
- Logging (WandB, stdout)
- Evaluation (via :mod:`recon.engine.evaluator`)

See:
    - docs/architecture/04-training-engine.md
    - ADR-0006 (why a unified Trainer)
    - docs/guides/04-run-training.md

The Trainer is intentionally minimal but extensible. New features
(custom loss, custom checkpointing) should go in subclasses or via
hooks, not by editing the core class.
"""
from __future__ import annotations

import datetime
import itertools
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from ..data.collate import collate_brain_stim_pairs
from ..data.datasets.meg import MEGDataset
from ..data.datasets.fmri import fMRIDataset
from ..data.drdr import DRDRIndex
from ..models.registry import build_model
from ..utils.logging import WandBLogger, get_logger

logger = get_logger(__name__)


# ───────────────────── Configuration ─────────────────────


@dataclass
class TrainConfig:
    """Subset of training config used by the Trainer.

    The full Hydra config has many more fields; this is the strict subset
    the Trainer actually reads. Anything else in the config is ignored.
    """

    epochs: int = 100
    batch_size: int = 32
    grad_clip: float = 1.0
    # Automatic mixed precision: None (off), "bf16" (Ampere+ GPUs, no
    # scaler needed), or "fp16" (with GradScaler). PH402/Pascal has no
    # 16-bit hardware — keep None on the cluster.
    amp_dtype: str | None = None
    num_workers: int = 0
    pin_memory: bool = False
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"  # "cosine", "step", "constant"
    warmup_steps: int = 100
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 5
    ckpt_dir: str = "${paths.results_dir}/ckpt/${run_id}"
    keep_last_n: int = 3
    seed: int = 42
    smoke: bool = False
    max_steps_per_epoch: int | None = None  # for smoke tests
    abort_on_nan: bool = True  # stop training on non-finite loss


# ───────────────────── Trainer ─────────────────────


class Trainer:
    """Unified training loop.

    Args:
        cfg: Full Hydra/OmegaConf config (with model/data/paths sub-configs).
        rank: Global rank for DDP. Default 0 (single-GPU).
        world_size: World size for DDP. Default 1 (single-GPU).
        local_rank: Local rank within a node. Default 0.
        master_addr: Master address for DDP. Default "localhost".
        master_port: Master port for DDP. Default 29500.
    """

    def __init__(
        self,
        cfg: DictConfig,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        master_addr: str = "localhost",
        master_port: int = 29500,
    ):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port

        # Default device
        if torch.cuda.is_available() and world_size >= 1:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        # Build sub-configs (with defaults)
        self.train_cfg = self._extract_train_config(cfg)
        self.amp_dtype = self.train_cfg.amp_dtype
        self.amp_enabled = self.amp_dtype in ("bf16", "fp16") and self.device.type == "cuda"
        self._amp_torch_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        # GradScaler is only needed for fp16 (bf16 has fp32 dynamic range)
        # pyright: ignore[reportAttributeAccessIssue] — stub lag; valid in torch>=2.3
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == "fp16")
        self._best_val_loss: float | None = None
        self.split_summary: dict | None = None

        # Initialize distributed if multi-GPU / multi-node
        self.is_distributed = world_size > 1
        if self.is_distributed:
            self._init_distributed()

        # Build model, optimizer, scheduler, dataloaders
        self.model: nn.Module = build_model(cfg.model).to(self.device)
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank] if self.device.type == "cuda" else None
            )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Logging (only on rank 0). On air-gapped compute nodes set
        # WANDB_MODE=offline (or logging.wandb_mode: offline): metrics are
        # written to ./wandb/ on the shared NFS mount and later uploaded
        # from the internet-connected mgmt node via `wandb sync`
        # (scripts/sync_wandb.sh).
        self.wandb: WandBLogger | None = None
        if self.rank == 0:
            logging_cfg = cfg.get("logging", {}) if isinstance(cfg, DictConfig) else {}
            self.wandb = WandBLogger(
                project=os.environ.get(
                    "WANDB_PROJECT", logging_cfg.get("wandb_project", "neural-decoding-infra")
                ),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=os.environ.get("WANDB_RUN_NAME"),
                mode=os.environ.get("WANDB_MODE", logging_cfg.get("wandb_mode", "online")),
            )

        # Set seeds
        self._set_seed(int(self.train_cfg.seed))

    # ───────────────────── Initialization helpers ─────────────────────

    def _init_distributed(self) -> None:
        """Initialize torch.distributed for DDP.

        The 10-minute timeout matters on a cluster: if one rank dies or a
        port is blocked, the others fail fast instead of hanging for the
        default 30 minutes.
        """
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        torch.distributed.init_process_group(
            backend="nccl" if self.device.type == "cuda" else "gloo",
            rank=self.rank,
            world_size=self.world_size,
            timeout=datetime.timedelta(minutes=10),
        )
        logger.info(
            "DDP initialized: rank=%d/%d, master=%s:%d",
            self.rank, self.world_size, self.master_addr, self.master_port,
        )

    def _extract_train_config(self, cfg: DictConfig) -> TrainConfig:
        """Pull out the train sub-config, applying defaults."""
        # Resolve interpolations once (e.g. ckpt.dir's ${paths.results_dir})
        resolved = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict = resolved if isinstance(resolved, dict) else {}
        train_section = cfg_dict.get("train", {})
        # Backward compat: legacy `amp: true|false` maps to fp16|None
        if "amp_dtype" not in train_section and "amp" in train_section:
            train_section["amp_dtype"] = "fp16" if train_section.get("amp") else None
        amp_dtype = train_section.get("amp_dtype")
        return TrainConfig(
            epochs=int(train_section.get("epochs", 100)),
            batch_size=int(train_section.get("batch_size", 32)),
            grad_clip=float(train_section.get("grad_clip", 1.0)),
            amp_dtype=str(amp_dtype) if amp_dtype else None,
            num_workers=int(train_section.get("num_workers", 0)),
            pin_memory=bool(train_section.get("pin_memory", False)),
            optimizer=str(train_section.get("optimizer", "adamw")),
            lr=float(train_section.get("lr", 1e-4)),
            weight_decay=float(train_section.get("weight_decay", 0.01)),
            scheduler=str(train_section.get("scheduler", "cosine")),
            warmup_steps=int(train_section.get("warmup_steps", 100)),
            log_interval=int(train_section.get("log_interval", 10)),
            save_interval=int(train_section.get("save_interval", 10)),
            eval_interval=int(train_section.get("eval_interval", 5)),
            ckpt_dir=str(
                cfg_dict.get("ckpt", {}).get(
                    "dir", train_section.get("ckpt_dir", "./ckpt/${run_id}")
                )
            ),
            keep_last_n=int(train_section.get("keep_last_n", 3)),
            seed=int(train_section.get("seed", 42)),
            smoke=bool(train_section.get("smoke", False)),
            max_steps_per_epoch=train_section.get("max_steps_per_epoch", None),
            abort_on_nan=bool(train_section.get("abort_on_nan", True)),
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer based on config."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.train_cfg.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.train_cfg.lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        elif self.train_cfg.optimizer.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=self.train_cfg.lr,
                weight_decay=self.train_cfg.weight_decay,
            )
        elif self.train_cfg.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.train_cfg.lr,
                weight_decay=self.train_cfg.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unknown optimizer: {self.train_cfg.optimizer}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Build LR scheduler based on config.

        Returns None for constant LR.
        """
        if self.train_cfg.scheduler == "constant":
            return None
        if self.train_cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.train_cfg.epochs
            )
        if self.train_cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.train_cfg.epochs // 3, gamma=0.1
            )
        raise ValueError(f"Unknown scheduler: {self.train_cfg.scheduler}")

    def _set_seed(self, seed: int) -> None:
        """Set seeds for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ───────────────────── Data loading ─────────────────────

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Build train (and optionally val) DataLoaders from config.

        Currently supports:
            - data=drdr (MEG): uses MEGDataset
            - data=fake: generates fake data on the fly
            - data=drdr_fmri: uses fMRIDataset

        For other datasets, override this method in a subclass.
        """
        data_cfg = self.cfg.get("data", {})
        data_name = data_cfg.get("name", "drdr")

        # Windows spawn deadlocks with DataLoader workers under `python -m`
        # entry points; fall back to in-process loading.
        num_workers = int(self.train_cfg.num_workers)
        if num_workers > 0 and sys.platform == "win32":
            logger.warning(
                "DataLoader num_workers=%d is unsupported on Windows "
                "(spawn deadlock with `python -m`); forcing num_workers=0",
                num_workers,
            )
            num_workers = 0
        self._num_workers = num_workers

        if data_name == "fake":
            return self._build_fake_dataloaders()
        if data_name == "drdr":
            return self._build_drdr_meg_dataloaders(data_cfg)
        if data_name == "drdr_fmri":
            return self._build_drdr_fmri_dataloaders(data_cfg)
        raise ValueError(f"Unknown data config: {data_name}")

    def _build_fake_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """Build dataloaders from fake data generator (for smoke tests)."""
        from ..data.fake_data import fake_pair_meg, fake_pair_fmri
        import numpy as np

        rng = np.random.default_rng(42)
        n_samples = 32 if self.train_cfg.smoke else 256
        modality = self.cfg.get("data", {}).get("modality", "meg")

        def gen_fake_dataset(n: int) -> list:
            pairs = []
            for i in range(n):
                if modality == "fmri":
                    pair = fake_pair_fmri(rng, story_id=(i % 10) + 1, subject_id=((i // 10) % 12) + 1)
                else:
                    pair = fake_pair_meg(
                        rng,
                        story_id=(i % 10) + 1,
                        subject_id=((i // 10) % 12) + 1,
                        n_context=int(self.cfg.get("data", {}).get("n_context", 0)),
                    )
                pairs.append(pair)
            return pairs

        train_pairs = gen_fake_dataset(n_samples)
        val_pairs = gen_fake_dataset(8) if not self.train_cfg.smoke else None
        self.split_summary = {
            "n_train": len(train_pairs),
            "n_val": len(val_pairs) if val_pairs else 0,
            "n_test": 0,
            "val_stories": [],
            "test_stories": [],
        }

        train_loader = DataLoader(
            _PairDataset(train_pairs),
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self.train_cfg.pin_memory,
            collate_fn=collate_brain_stim_pairs,
        )
        val_loader = (
            DataLoader(
                _PairDataset(val_pairs),
                batch_size=self.train_cfg.batch_size,
                shuffle=False,
                collate_fn=collate_brain_stim_pairs,
            )
            if val_pairs
            else None
        )
        return train_loader, val_loader

    def _build_drdr_meg_dataloaders(self, data_cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
        """Build dataloaders for the DRDR MEG dataset (train + optional val)."""
        from omegaconf import OmegaConf

        from ..data.drdr import discover_drdr
        from ..data.split import split_index

        processed_root = self.cfg.paths.get("processed_root")
        if not processed_root:
            raise ValueError(
                "data=drdr requires paths.processed_root pointing at the "
                "preprocessed results (e.g. E:/results, "
                "/home/test/reconstruction/results)"
            )
        index = discover_drdr(
            Path(processed_root), modality="meg", max_stories=int(data_cfg.get("max_stories", 60))
        )
        index = _filter_index(index, data_cfg)
        split_cfg = data_cfg.get("split", OmegaConf.create({"method": "none"}))
        split = split_index(index, split_cfg)
        self.split_summary = split.summary()

        n_context = int(data_cfg.get("n_context", 5))
        weights = tuple(float(w) for w in data_cfg.get("weights", [0.1, 0.7, 0.5, 0.3]))
        max_steps = int(self.train_cfg.max_steps_per_epoch) if self.train_cfg.smoke else None
        train_ds = MEGDataset(
            _make_sub_index(index, split.train),
            n_context=n_context,
            layer=int(data_cfg.get("layer", 10)),
            weights=weights,
            max_steps_per_story=max_steps,
        )
        train_loader = self._make_loader(train_ds, shuffle=True)

        val_loader = None
        if split.val:
            val_ds = MEGDataset(
                _make_sub_index(index, split.val),
                n_context=n_context,
                layer=int(data_cfg.get("layer", 10)),
                weights=weights,
                max_steps_per_story=max_steps,
            )
            val_loader = self._make_loader(val_ds, shuffle=False)
        return train_loader, val_loader

    def _build_drdr_fmri_dataloaders(self, data_cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
        """Build dataloaders for the DRDR fMRI dataset (train + optional val)."""
        from omegaconf import OmegaConf

        from ..data.drdr import discover_drdr
        from ..data.split import split_index

        processed_root = self.cfg.paths.get("processed_root")
        if not processed_root:
            raise ValueError(
                "data=drdr_fmri requires paths.processed_root pointing at the "
                "preprocessed results (e.g. E:/results, "
                "/home/test/reconstruction/results)"
            )
        index = discover_drdr(
            Path(processed_root), modality="fmri", max_stories=int(data_cfg.get("max_stories", 60))
        )
        index = _filter_index(index, data_cfg)
        split_cfg = data_cfg.get("split", OmegaConf.create({"method": "none"}))
        split = split_index(index, split_cfg)
        self.split_summary = split.summary()

        weights = tuple(float(w) for w in data_cfg.get("weights", [0.1, 0.7, 0.5, 0.3]))
        max_steps = int(self.train_cfg.max_steps_per_epoch) if self.train_cfg.smoke else None
        train_ds = fMRIDataset(
            _make_sub_index(index, split.train),
            layer=int(data_cfg.get("layer", 10)),
            weights=weights,
            mask_path=data_cfg.get("mask_path"),
            max_steps_per_story=max_steps,
        )
        train_loader = self._make_loader(train_ds, shuffle=True)

        val_loader = None
        if split.val:
            val_ds = fMRIDataset(
                _make_sub_index(index, split.val),
                layer=int(data_cfg.get("layer", 10)),
                weights=weights,
                mask_path=data_cfg.get("mask_path"),
                max_steps_per_story=max_steps,
            )
            val_loader = self._make_loader(val_ds, shuffle=False)
        return train_loader, val_loader

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Shared DataLoader construction (sampler + collate)."""
        sampler = (
            DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
            if self.is_distributed
            else None
        )
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=self._num_workers,
            pin_memory=self.train_cfg.pin_memory,
            sampler=sampler,
            collate_fn=collate_brain_stim_pairs,
        )

    # ───────────────────── Training loop ─────────────────────

    def fit(self) -> None:
        """Run the full training loop.

        Sequence:
            0. Startup checks (devices, distributed comms, data contract,
               dry-run forward/backward) — fail fast BEFORE training
            1. Per epoch: train; (val hook reserved); checkpoint on interval
            2. Cleanup in ``finally`` — signal-safe shutdown

        SIGINT/SIGTERM set ``self._aborted``; the loop exits at the next
        batch boundary, saves a checkpoint (rank 0), and cleans up.
        """
        train_loader, val_loader = self.build_dataloaders()
        # smoke mode = 1 epoch (fast sanity run), regardless of config
        n_epochs = 1 if self.train_cfg.smoke else int(self.train_cfg.epochs)

        prev_handlers = self._install_signal_handlers()
        self._aborted = False
        self._run_startup_checks(train_loader)
        if self.rank == 0:
            self._write_run_metadata(train_loader)

        logger.info("Starting training: %d epochs, batch_size=%d", n_epochs, self.train_cfg.batch_size)
        try:
            for epoch in range(n_epochs):
                if self._aborted:
                    logger.warning("Abort signal received — stopping before epoch %d", epoch + 1)
                    break

                # DDP: re-shuffle every epoch (DistributedSampler needs this)
                sampler = getattr(train_loader, "sampler", None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)

                t0 = time.time()
                train_metrics = self._train_epoch(train_loader, epoch)
                epoch_time = time.time() - t0

                logger.info(
                    "Epoch %d/%d: loss=%.4f (cos=%.4f, mse=%.4f%s) [%.1fs]",
                    epoch + 1,
                    n_epochs,
                    train_metrics.get("total_loss", 0.0),
                    train_metrics.get("cosine_loss", 0.0),
                    train_metrics.get("mse_loss", 0.0),
                    f", kl={train_metrics.get('kl_loss', 0.0):.4f}" if "kl_loss" in train_metrics else "",
                    epoch_time,
                )

                # Log to WandB
                if self.wandb and self.wandb.enabled:
                    self.wandb.log(
                        {f"train/{k}": v for k, v in train_metrics.items()},
                        step=epoch,
                    )
                    self.wandb.log({"train/lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)

                # Validation
                if val_loader is not None and (epoch + 1) % self.train_cfg.eval_interval == 0:
                    val_metrics = self._eval_epoch(val_loader)
                    logger.info(
                        "Val epoch %d: loss=%.4f (cos=%.4f, mse=%.4f)",
                        epoch + 1,
                        val_metrics.get("total_loss", 0.0),
                        val_metrics.get("cosine_loss", 0.0),
                        val_metrics.get("mse_loss", 0.0),
                    )
                    if self.wandb and self.wandb.enabled:
                        self.wandb.log(
                            {f"val/{k}": v for k, v in val_metrics.items()}, step=epoch
                        )
                    # Best-val checkpoint
                    if self._best_val_loss is None or val_metrics["total_loss"] < self._best_val_loss:
                        self._best_val_loss = val_metrics["total_loss"]
                        if self.rank == 0:
                            self.save_checkpoint(epoch, val_metrics, name="best_val.pt")
                            logger.info(
                                "New best val loss %.4f — saved best_val.pt",
                                self._best_val_loss,
                            )

                # Save checkpoint
                if (epoch + 1) % self.train_cfg.save_interval == 0 and self.rank == 0:
                    self.save_checkpoint(epoch, train_metrics)

                # Step scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

        finally:
            if self.rank == 0 and self._aborted:
                logger.warning("Saving emergency checkpoint after abort signal...")
                try:
                    self.save_checkpoint(self.train_cfg.epochs, {"aborted": True})
                except Exception:  # pragma: no cover — best effort
                    logger.exception("Emergency checkpoint failed")
            if self.wandb and self.wandb.enabled:
                self.wandb.finish()
            if self.is_distributed and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            # Restore previous signal handlers (tests, embedding contexts)
            for sig, handler in prev_handlers.items():
                signal.signal(sig, handler)

    # ───────────────────── Startup checks ─────────────────────

    def _install_signal_handlers(self) -> dict[int, Any]:
        """Install SIGINT/SIGTERM handlers; return the previous handlers."""
        prev: dict[int, Any] = {}

        def _handle(signum: int, _frame: Any) -> None:
            logger.warning("Received signal %s — aborting after current batch", signum)
            self._aborted = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            prev[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle)
        return prev

    def _run_startup_checks(self, train_loader: DataLoader) -> None:
        """Fail fast BEFORE training: devices, comms, data, dry-run."""
        # 1. Devices
        if self.device.type == "cuda":
            n_devices = torch.cuda.device_count()
            logger.info(
                "Device check: rank=%d local_rank=%d, %d CUDA device(s) on node: %s",
                self.rank, self.local_rank, n_devices,
                [torch.cuda.get_device_name(i) for i in range(n_devices)],
            )
            if self.local_rank >= n_devices:
                raise RuntimeError(
                    f"local_rank={self.local_rank} but only {n_devices} CUDA "
                    "device(s) available — check nproc_per_node / GPU allocation"
                )

        # 2. Distributed communication test (all_reduce must round-trip)
        if self.is_distributed:
            dist = torch.distributed
            probe = torch.tensor([float(self.rank)], device=self.device)
            dist.all_reduce(probe)
            expected = float(self.world_size * (self.world_size - 1) / 2)
            if abs(probe.item() - expected) > 1e-6:
                raise RuntimeError(
                    f"Distributed all_reduce failed: got {probe.item()}, expected {expected}"
                )
            logger.info("Comm check passed: all_reduce round-trip OK (world_size=%d)", self.world_size)

        # 3. Data contract: one real batch must exist and have sane shapes
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("train_loader is empty — check subjects/stories filters") from None
        n_samples = len(train_loader.dataset)
        logger.info(
            "Data check: %d train samples, first batch brain=%s stim=%s",
            n_samples, tuple(batch.brain.x.shape), tuple(batch.stim.zstim.shape),
        )
        if batch.brain.x.shape[0] == 0 or batch.stim.zstim.shape[0] == 0:
            raise RuntimeError("First batch is empty")
        if batch.stim.zstim.shape[-1] < 1:
            raise RuntimeError("stim.zstim has no feature dim")

        # 4. Dry-run: forward + backward (no optimizer step)
        self.model.train()
        brain_x = batch.brain.x.to(self.device)
        stim_zstim = batch.stim.zstim.to(self.device)
        ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp_torch_dtype)
            if self.amp_enabled
            else nullcontext()
        )
        with ctx:
            pred, aux = self._model_forward(brain_x)
            target = stim_zstim[..., : pred.shape[-1]]
            dry_loss, _ = self._model_compute_loss(pred, target, aux)
        dry_loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            raise RuntimeError("Dry-run produced no gradients — model/optimizer wiring broken")
        self.optimizer.zero_grad()
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Dry-run passed: forward+backward OK, loss=%.4f, %d/%d params have grads, total params=%d",
            float(dry_loss), len(grads),
            sum(1 for p in self.model.parameters() if p.requires_grad), n_params,
        )

    def _write_run_metadata(self, train_loader: DataLoader) -> None:
        """Write a human-readable run_metadata.json next to checkpoints."""
        ckpt_dir = Path(self.train_cfg.ckpt_dir.replace("${run_id}", os.environ.get("RUN_ID", "run")))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        data_cfg = self.cfg.get("data", {}) if isinstance(self.cfg, DictConfig) else {}
        resolved_raw = OmegaConf.to_container(self.cfg, resolve=True)
        resolved = resolved_raw if isinstance(resolved_raw, dict) else {}
        model_cfg = resolved.get("model", {}) if isinstance(resolved.get("model"), dict) else {}
        dataset = train_loader.dataset
        n_train = len(cast(Any, dataset))
        metadata = {
            "run_id": os.environ.get("RUN_ID", None),
            "git_commit": self._git_commit(),
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "platform": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": getattr(getattr(torch, "version", {}), "cuda", None),
                "device": str(self.device),
                "device_count": torch.cuda.device_count() if self.device.type == "cuda" else 0,
                "world_size": self.world_size,
                "rank": self.rank,
            },
            "model": {
                "name": model_cfg.get("name"),
                "n_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            },
            "data": {
                "name": data_cfg.get("name") if isinstance(data_cfg, dict) else str(data_cfg),
                "subjects": data_cfg.get("subjects") if isinstance(data_cfg, dict) else None,
                "stories": data_cfg.get("stories") if isinstance(data_cfg, dict) else None,
                "split": self.split_summary,
                "n_train_samples": n_train,
                "batch_size": self.train_cfg.batch_size,
                "steps_per_epoch": len(train_loader),
            },
            "config": resolved,
        }
        path = ckpt_dir / "run_metadata.json"
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
        logger.info("Run metadata written to %s", path)

    @staticmethod
    def _git_commit() -> str | None:
        """Best-effort git commit hash of the current checkout."""
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    def _train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> dict[str, float]:
        """Train for one epoch.

        Returns a dict of metrics (averaged over the epoch).
        """
        self.model.train()
        running: dict[str, float] = {}
        n_batches = 0

        max_steps = self.train_cfg.max_steps_per_epoch
        iterator = iter(train_loader)
        if max_steps is not None and max_steps > 0:
            # Cycle the loader if it is shorter than max_steps, then take max_steps
            iterator = itertools.islice(_InfiniteIterator(train_loader), max_steps)

        # Progress bar on rank 0 only (all ranks log would interleave garbage)
        if self.rank == 0:
            iterator = tqdm(
                iterator,
                desc=f"Epoch {epoch + 1}",
                total=max_steps,
                leave=False,
                ncols=100,
            )

        for step, batch in enumerate(iterator):
            # Move brain and stim to device
            brain_x = batch.brain.x.to(self.device)
            stim_zstim = batch.stim.zstim.to(self.device)

            # Forward. The loss is computed INSIDE the autocast context:
            # model outputs are fp16 under AMP, and mixing them with the
            # fp32 target outside autocast breaks backward ("Found dtype
            # Float but expected Half").
            # stim.zstim may hold several concatenated delays (e.g. 4 × 768
            # = 3072); models predict the first delay's embedding
            # (semantic_dim = pred.shape[-1]), so slice the target.
            ctx = (
                torch.autocast(device_type="cuda", dtype=self._amp_torch_dtype)
                if self.amp_enabled
                else nullcontext()
            )
            with ctx:
                pred, aux_loss = self._model_forward(brain_x)
                target = stim_zstim[..., : pred.shape[-1]]
                loss, loss_dict = self._model_compute_loss(pred, target, aux_loss)

            # NaN / Inf guard: fail fast instead of burning cluster hours
            if not torch.isfinite(loss):
                logger.error(
                    "Non-finite loss %.4f at epoch %d step %d (rank %d)",
                    float(loss), epoch, step, self.rank,
                )
                if self.train_cfg.abort_on_nan:
                    raise RuntimeError(
                        f"Non-finite loss ({float(loss)}) at epoch {epoch} step {step}. "
                        "Training aborted — lower the LR, enable grad clipping, "
                        "or set train.abort_on_nan=false to skip non-finite steps."
                    )
                logger.warning("Skipping non-finite step (abort_on_nan=false)")
                continue

            # Backward. fp16 uses GradScaler (bf16/None don't need it).
            self.optimizer.zero_grad()
            grad_norm = float("nan")
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.train_cfg.grad_clip > 0:
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.train_cfg.grad_clip > 0:
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                    )
                self.optimizer.step()

            # Accumulate
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v
            running["grad_norm"] = running.get("grad_norm", 0.0) + grad_norm
            n_batches += 1

            # Progress bar + periodic log
            if self.rank == 0 and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    loss=f"{loss_dict.get('total_loss', float(loss)):.4f}",
                    grad=f"{grad_norm:.2f}",
                )
            if (step + 1) % self.train_cfg.log_interval == 0 and self.rank == 0:
                avg = {k: v / n_batches for k, v in running.items()}
                logger.debug("  step %d: %s", step + 1, {k: f"{v:.4f}" for k, v in avg.items()})

        if n_batches == 0:
            return {"total_loss": 0.0, "cosine_loss": 0.0, "mse_loss": 0.0}
        return {k: v / n_batches for k, v in running.items()}

    def _eval_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate one pass over the validation set (no gradients).

        Metrics are averaged over batches and all-reduced across ranks
        under DDP (each rank sees a different data shard).
        """
        self.model.eval()
        running: dict[str, float] = {}
        n_batches = 0
        ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp_torch_dtype)
            if self.amp_enabled
            else nullcontext()
        )
        with torch.no_grad():
            for batch in val_loader:
                brain_x = batch.brain.x.to(self.device)
                stim_zstim = batch.stim.zstim.to(self.device)
                with ctx:
                    pred, aux = self._model_forward(brain_x)
                    target = stim_zstim[..., : pred.shape[-1]]
                    _, loss_dict = self._model_compute_loss(pred, target, aux)
                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v
                n_batches += 1
        self.model.train()

        if n_batches == 0:
            return {"total_loss": 0.0}
        metrics = {k: v / n_batches for k, v in running.items()}

        if self.is_distributed and torch.distributed.is_initialized():
            keys = sorted(metrics)
            tens = torch.tensor([metrics[k] for k in keys], device=self.device)
            torch.distributed.all_reduce(tens)
            tens = tens / self.world_size
            metrics = {k: float(v) for k, v in zip(keys, tens.tolist())}
        return metrics

    def _model_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Forward pass through the (possibly DDP-wrapped) model.

        Handles the case where some models return (pred, aux_loss) tuples
        and others return just predictions.
        """
        out = self.model(x)
        if isinstance(out, tuple):
            pred, aux = out
        else:
            pred, aux = out, None
        return pred, aux

    def _model_compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss via the model's compute_loss method.

        Handles DDP wrapping (model.module vs model).
        """
        model_for_loss = self.model.module if hasattr(self.model, "module") else self.model
        # Some models don't have compute_loss; fall back to MSE
        if hasattr(model_for_loss, "compute_loss"):
            aux_dict = {"kl_loss": aux} if aux is not None else None
            return model_for_loss.compute_loss(pred, target, aux=aux_dict)
        # Fallback
        loss = torch.nn.functional.mse_loss(pred, target)
        return loss, {"mse_loss": loss.item(), "total_loss": loss.item()}

    # ───────────────────── Checkpointing ─────────────────────

    def save_checkpoint(
        self, epoch: int, metrics: dict[str, float], name: str | None = None
    ) -> Path:
        """Save a checkpoint to the configured directory.

        Args:
            name: Optional explicit filename (e.g. ``"best_val.pt"``);
                defaults to ``checkpoint_epoch_{epoch}.pt``.
        """
        ckpt_dir = Path(self.train_cfg.ckpt_dir.replace("${run_id}", os.environ.get("RUN_ID", f"epoch_{epoch}")))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / (name or f"checkpoint_epoch_{epoch}.pt")

        # Get state dict (unwrap DDP if needed)
        model_state = (
            self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict(),
                "metrics": metrics,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint: %s", ckpt_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints(ckpt_dir)
        return ckpt_path

    def _cleanup_old_checkpoints(self, ckpt_dir: Path) -> None:
        """Keep only the last N checkpoints."""
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if len(ckpts) > self.train_cfg.keep_last_n:
            for old in ckpts[: -self.train_cfg.keep_last_n]:
                old.unlink()
                logger.debug("Removed old checkpoint: %s", old)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint. Returns the epoch number."""
        ckpt = torch.load(str(path), map_location=self.device)
        model_for_load = self.model.module if hasattr(self.model, "module") else self.model
        model_for_load.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", 0))
        return ckpt.get("epoch", 0)


class _InfiniteIterator:
    """Helper for max_steps_per_epoch (allows iterating beyond dataset end).

    Wraps a re-iterable (e.g. a DataLoader); on exhaustion it re-creates the
    underlying iterator from the original iterable, so it never stops.
    """

    def __init__(self, base):
        self._base = base  # re-iterable
        self._it = iter(base)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._base)
            return next(self._it)


def _make_sub_index(index: DRDRIndex, pairs: list[tuple[int, int]]) -> DRDRIndex:
    """Build a DRDRIndex restricted to the given (subject, story) pairs."""
    by_subject: dict[int, list[int]] = {}
    by_story: dict[int, list[int]] = {}
    for s, st in pairs:
        by_subject.setdefault(s, []).append(st)
        by_story.setdefault(st, []).append(s)
    return DRDRIndex(
        processed_root=index.processed_root,
        modality=index.modality,
        subjects=sorted(by_subject),
        stories=sorted(by_story),
        by_subject=by_subject,
        by_story=by_story,
        pairs=pairs,
    )


def _filter_index(index: DRDRIndex, data_cfg: DictConfig) -> DRDRIndex:
    """Filter a DRDRIndex by configured subjects/stories.

    Returns a new index with ``pairs`` (and derived dicts) restricted to
    the configured subjects/stories. If neither filter is set, returns
    the index unchanged.
    """
    subjects = data_cfg.get("subjects")
    stories = data_cfg.get("stories")
    if not subjects and not stories:
        return index
    subjects = list(subjects) if subjects else None
    stories = list(stories) if stories else None
    pairs = [
        (s, st)
        for s, st in index.pairs
        if (subjects is None or s in subjects) and (stories is None or st in stories)
    ]
    by_subject: dict[int, list[int]] = {}
    by_story: dict[int, list[int]] = {}
    for s, st in pairs:
        by_subject.setdefault(s, []).append(st)
        by_story.setdefault(st, []).append(s)
    return DRDRIndex(
        processed_root=index.processed_root,
        modality=index.modality,
        subjects=sorted(by_subject),
        stories=sorted(by_story),
        by_subject=by_subject,
        by_story=by_story,
        pairs=pairs,
    )


class _PairDataset(Dataset):
    """In-memory Dataset over a list of pre-validated BrainStimPairs.

    Module-level so DataLoader workers can pickle it (Windows spawn).
    """

    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        return self.pairs[i]


__all__ = ["TrainConfig", "Trainer"]