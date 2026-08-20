"""Engine: training loop, evaluation, distributed coordination.

This subpackage contains:
    - trainer: Unified training loop (DDP-aware)
    - evaluator: Metrics computation (CRR, CER, perplexity, etc.)

Key principles:
- All models go through the same Trainer — see ``docs/architecture/04-training-engine.md``.
- The Trainer handles device placement, optimizer, scheduler, checkpointing.
- Distributed coordination is via ``torch.distributed`` (DDP), wrapped
  transparently for single-GPU and multi-node use.
"""