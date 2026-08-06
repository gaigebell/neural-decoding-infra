# ADR-0006: Unified Trainer (over per-model scripts)

- **Status**: Accepted
- **Date**: 2026-07-28
- **Deciders**: owner

## Context and problem statement

The previous codebase (`recon/`) has **5 near-duplicate training scripts**:

- `fMRI3dCIB_train_mg.py`
- `fMRI3dCIB_train_mn.py`
- `fMRI3dCIB_train_mn_cross_sub.py`
- `fMRI3dCIB2_train_mn.py`
- `fMRI3dCIB2_train_mn_cross_sub.py`
- `MEG_model_A_train_mg.py`
- `fMRI3dBNIB_train_mg.py`
- ... (8+ total)

~95% of the code is identical (data loading, distributed init, optimizer, logging, checkpoint). Only the model class and a few hyperparameters differ.

This violates the **DRY principle** and causes:

- Bug fixes need to be replicated across 5+ files
- New model variants require copy-pasting 500+ lines
- Subtle inconsistencies between scripts (e.g., `drop_last=True` in one, `False` in another)

## Decision drivers

- Adding new model variants is a primary use case (research is about exploration)
- Bug fixes must propagate to all training scripts
- Single-owner team cannot maintain 8 duplicate files
- We need a clean way to add cross-subject / multi-node variants without forking

## Considered options

### Option A: Keep per-model scripts (status quo)

- **Pro**: Each script is self-contained; easy to "just run this one"
- **Con**: 8× duplication; bug fixes are manual; drift inevitable

### Option B: Single `train.py` with `argparse` model selection

- **Pro**: One file
- **Con**: `train.py` becomes a 1000-line god-file; argparse is ugly for hierarchical config

### Option C: Unified `Trainer` class + Hydra config selects model

- **Pro**: Model is selected via config, not code
- **Pro**: Trainer encapsulates DDP, optimizer, logging — all in one place
- **Pro**: Adding a new model = add 1 adapter file, not fork a script
- **Con**: Some indirection (model lives in registry, not in script)

### Option D: PyTorch Lightning

- **Pro**: Industry-standard; lots of callbacks
- **Con**: Heavy abstraction; harder to debug; opinionated

## Decision outcome

**Chosen option**: **Option C (Unified Trainer + MODEL_REGISTRY)**.

The Trainer class accepts a model (from registry), a dataloader, a config, and runs the full training loop. Distributed coordination is internal. Adding a new model = one file in `recon/models/{kind}/` + one YAML config + one registration line.

### Consequences

- ✅ Good: 8 training scripts → 1 Trainer + 8 model adapters
- ✅ Good: Bug fixes in Trainer propagate to all models automatically
- ✅ Good: New model variant = ~50 lines, not ~500
- ✅ Good: Cross-subject training is just a config flag (`data.cross_sub: true`)
- ❌ Bad: Slight indirection — to find "how is X trained," you read Trainer + adapter, not one file
- ❌ Bad: Some legacy script patterns (e.g., custom loss printing) require a small Trainer hook
- ❓ Risk: Trainer becomes a god-object over time. **Mitigation**: refactor into sub-modules when >500 lines.

## Implementation sketch

```python
# recon/engine/trainer.py
class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = build_model(cfg)        # from MODEL_REGISTRY
        self.optimizer = self._build_optimizer()
        self.dataloader = self._build_dataloader()
        self.logger = build_logger(cfg)       # W&B wrapper
    
    def fit(self):
        for epoch in range(self.cfg.train.epochs):
            self.train_epoch(epoch)
            if epoch % self.cfg.eval.interval == 0:
                self.evaluate(epoch)
            if epoch % self.cfg.ckpt.interval == 0:
                self.save_checkpoint(epoch)
```

```python
# recon/models/registry.py
MODEL_REGISTRY = {
    "fmri3dcib": build_fmri3dcib,
    "fmri3dcib2": build_fmri3dcib2,
    "meg_model_a": build_meg_model_a,
    "brainomni_align_mlp": build_brainomni_mlp,
    "brainomni_align_ib": build_brainomni_ib,
}
```

```yaml
# configs/model/fmri3dcib.yaml
defaults: [...]
name: fmri3dcib
backbone: cnn3d
aligner: information_bottleneck
bottleneck_dim: 256
semantic_dim: 768
beta: 0.001
```

## When to revisit

- If Trainer exceeds 1000 lines → split into `Trainer`, `DistributedMixin`, `LoggingMixin`
- If model variants exceed 20 → consider Lightning
- If training loops diverge significantly per model → consider per-model hooks (still within one Trainer)

## Links

- [Architecture: Training engine](04-training-engine.md) — full Trainer design
- [Architecture: Model registry](03-model-registry.md)
- [Guide: Write a new model](../guides/03-write-model.md)