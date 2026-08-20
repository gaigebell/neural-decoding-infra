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

import itertools
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from ..data.collate import collate_brain_stim_pairs
from ..data.datasets.meg import MEGDataset
from ..data.datasets.fmri import fMRIDataset
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
    amp: bool = True
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
        self.amp_enabled = bool(self.train_cfg.amp) and self.device.type == "cuda"

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

        # Logging (only on rank 0)
        self.wandb: WandBLogger | None = None
        if self.rank == 0:
            self.wandb = WandBLogger(
                project=os.environ.get("WANDB_PROJECT", "neural-decoding-infra"),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=os.environ.get("WANDB_RUN_NAME"),
            )

        # Set seeds
        self._set_seed(int(self.train_cfg.seed))

    # ───────────────────── Initialization helpers ─────────────────────

    def _init_distributed(self) -> None:
        """Initialize torch.distributed for DDP."""
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        torch.distributed.init_process_group(
            backend="nccl" if self.device.type == "cuda" else "gloo",
            rank=self.rank,
            world_size=self.world_size,
        )
        logger.info(
            "DDP initialized: rank=%d/%d, master=%s:%d",
            self.rank, self.world_size, self.master_addr, self.master_port,
        )

    def _extract_train_config(self, cfg: DictConfig) -> TrainConfig:
        """Pull out the train sub-config, applying defaults."""
        train_section = cfg.get("train", {})
        return TrainConfig(
            epochs=int(train_section.get("epochs", 100)),
            batch_size=int(train_section.get("batch_size", 32)),
            grad_clip=float(train_section.get("grad_clip", 1.0)),
            amp=bool(train_section.get("amp", True)),
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
            ckpt_dir=str(train_section.get("ckpt_dir", "./ckpt/${run_id}")),
            keep_last_n=int(train_section.get("keep_last_n", 3)),
            seed=int(train_section.get("seed", 42)),
            smoke=bool(train_section.get("smoke", False)),
            max_steps_per_epoch=train_section.get("max_steps_per_epoch", None),
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
        """Build dataloaders for the DRDR MEG dataset."""
        from ..data.drdr import discover_drdr

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

        n_context = int(data_cfg.get("n_context", 5))
        weights = tuple(float(w) for w in data_cfg.get("weights", [0.1, 0.7, 0.5, 0.3]))
        max_steps = int(self.train_cfg.max_steps_per_epoch) if self.train_cfg.smoke else None
        train_ds = MEGDataset(
            index,
            n_context=n_context,
            layer=int(data_cfg.get("layer", 10)),
            weights=weights,
            max_steps_per_story=max_steps,
        )
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank)
            if self.is_distributed
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self._num_workers,
            pin_memory=self.train_cfg.pin_memory,
            sampler=train_sampler,
            collate_fn=collate_brain_stim_pairs,
        )
        # No val yet — owner will add when needed
        return train_loader, None

    def _build_drdr_fmri_dataloaders(self, data_cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
        """Build dataloaders for the DRDR fMRI dataset."""
        from ..data.drdr import discover_drdr

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

        weights = tuple(float(w) for w in data_cfg.get("weights", [0.1, 0.7, 0.5, 0.3]))
        max_steps = int(self.train_cfg.max_steps_per_epoch) if self.train_cfg.smoke else None
        train_ds = fMRIDataset(
            index,
            layer=int(data_cfg.get("layer", 10)),
            weights=weights,
            mask_path=data_cfg.get("mask_path"),
            max_steps_per_story=max_steps,
        )
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.rank)
            if self.is_distributed
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self._num_workers,
            pin_memory=self.train_cfg.pin_memory,
            sampler=train_sampler,
            collate_fn=collate_brain_stim_pairs,
        )
        return train_loader, None

    # ───────────────────── Training loop ─────────────────────

    def fit(self) -> None:
        """Run the full training loop.

        For each epoch:
            1. Train one epoch (one pass through train_loader)
            2. Optionally evaluate (if val_loader and epoch % eval_interval == 0)
            3. Save checkpoint (if epoch % save_interval == 0)
        """
        train_loader, val_loader = self.build_dataloaders()
        # smoke mode = 1 epoch (fast sanity run), regardless of config
        n_epochs = 1 if self.train_cfg.smoke else int(self.train_cfg.epochs)
        logger.info("Starting training: %d epochs, batch_size=%d", n_epochs, self.train_cfg.batch_size)

        for epoch in range(n_epochs):
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

            # Save checkpoint
            if (epoch + 1) % self.train_cfg.save_interval == 0 and self.rank == 0:
                self.save_checkpoint(epoch, train_metrics)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Cleanup
        if self.wandb and self.wandb.enabled:
            self.wandb.finish()
        if self.is_distributed:
            torch.distributed.destroy_process_group()

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

        for step, batch in enumerate(iterator):
            # Move brain and stim to device
            brain_x = batch.brain.x.to(self.device)
            stim_zstim = batch.stim.zstim.to(self.device)

            # Forward
            if self.amp_enabled:
                with torch.cuda.amp.autocast():
                    pred, aux_loss = self._model_forward(brain_x)
            else:
                pred, aux_loss = self._model_forward(brain_x)

            # stim.zstim may hold several concatenated delays (e.g. 4 × 768
            # = 3072); models predict the first delay's embedding
            # (semantic_dim = pred.shape[-1]), so slice the target.
            target = stim_zstim[..., : pred.shape[-1]]
            loss, loss_dict = self._model_compute_loss(pred, target, aux_loss)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.train_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.optimizer.step()

            # Accumulate
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

            # Log
            if (step + 1) % self.train_cfg.log_interval == 0 and self.rank == 0:
                avg = {k: v / n_batches for k, v in running.items()}
                logger.debug("  step %d: %s", step + 1, {k: f"{v:.4f}" for k, v in avg.items()})

        if n_batches == 0:
            return {"total_loss": 0.0, "cosine_loss": 0.0, "mse_loss": 0.0}
        return {k: v / n_batches for k, v in running.items()}

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

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> Path:
        """Save a checkpoint to the configured directory."""
        ckpt_dir = Path(self.train_cfg.ckpt_dir.replace("${run_id}", os.environ.get("RUN_ID", f"epoch_{epoch}")))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"

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


def _filter_index(index, data_cfg: DictConfig):
    """Filter a DRDRIndex by configured subjects/stories.

    Returns the same index object with ``pairs`` (and derived dicts)
    restricted to the configured subjects/stories. If neither filter is
    set, returns the index unchanged.
    """
    from ..data.drdr import DRDRIndex

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