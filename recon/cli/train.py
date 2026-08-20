"""CLI: training entry point.

Usage:
    python -m recon.cli.train
    python -m recon.cli.train model=fmri3dcib data=fake train.smoke=true
    python -m recon.cli.train model=meg_model_a data.subjects=[1,2,3]

Distributed launch (via torchrun):
    torchrun --nproc_per_node=2 -m recon.cli.train model=fmri3dcib train.amp=true
"""
from __future__ import annotations

import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..engine.trainer import Trainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Training entry point.

    Args:
        cfg: Hydra/OmegaConf config composed from configs/ directory.
    """
    logger.info("=" * 60)
    logger.info("neural-decoding-infra — training")
    logger.info("=" * 60)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    # Detect distributed launch
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))

    if world_size > 1 and not torch.distributed.is_initialized():
        # torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; init_process_group
        # is handled by torchrun in newer versions. We still pass
        # these to Trainer for clarity.
        logger.info("Detected torchrun launch: world_size=%d, rank=%d", world_size, rank)

    # Build and run trainer
    trainer = Trainer(
        cfg=cfg,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    trainer.fit()


if __name__ == "__main__":
    main()