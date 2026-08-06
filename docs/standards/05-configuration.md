# Standard 05: Configuration

> **Audience**: Anyone touching Hydra configs.
> **Status**: Active. See [ADR-0001](../decisions/0001-why-hydra.md).

---

## 1. Layout

```
configs/
├── train.yaml                       # Default entry point
├── eval.yaml                        # Evaluation entry point
├── model/
│   ├── fmri3dcib.yaml
│   ├── fmri3dcib2.yaml
│   ├── meg_model_a.yaml
│   └── brainomni_align.yaml
├── data/
│   ├── drdr.yaml
│   └── brainomni.yaml
├── decoder/
│   ├── beam.yaml
│   └── nucleus.yaml
├── paths/
│   ├── cluster.yaml                 # Production: cluster paths
│   ├── mgmt.yaml                    # Management-node paths
│   └── local.yaml                   # Dev-machine paths
└── experiment/                      # Reproducible experiment snapshots
    ├── 001-baseline-fmri-sub01.yaml
    └── 002-joint-fmri-12sub.yaml
```

## 2. File naming

- **Lowercase, kebab-case** for filenames: `fmri3dcib.yaml`, not `fmri_3dcib.yaml` or `fmri3dCIB.yaml`
- **Numbered experiment snapshots** for exact reproducibility: `001-<short-name>.yaml`

## 3. Default config

`configs/train.yaml` is the entry point. It uses Hydra's `defaults:` list:

```yaml
# configs/train.yaml
defaults:
  - model: fmri3dcib
  - data: drdr
  - decoder: beam
  - paths: cluster
  - _self_

train:
  epochs: 100
  batch_size: 32
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 0.01
  grad_clip: 1.0
  amp: true

eval:
  interval: 5
  metrics: [crr, cer, topk_5]

ckpt:
  dir: ${paths.results_dir}/ckpt/${run_id}
  save_interval: 10
  keep_last_n: 3

run_id: ${now:%Y-%m-%d_%H-%M-%S}
seed: 42
```

## 4. Override pattern

Users override via CLI:

```bash
# Switch model
python -m recon.cli.train model=meg_model_a

# Change paths for dev
python -m recon.cli.train paths=local

# Multiple overrides
python -m recon.cli.train model=fmri3dcib train.epochs=50 train.lr=3e-4

# Use a saved experiment snapshot
python -m recon.cli.train +experiment=001-baseline-fmri-sub01
```

## 5. Config conventions

### 5.1 Use snake_case keys

```yaml
# ✅ Good
train:
  batch_size: 32
  grad_clip: 1.0

# ❌ Bad
train:
  BatchSize: 32
  gradClip: 1.0
```

### 5.2 Group related keys

```yaml
train:
  epochs: 100
  batch_size: 32
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01
  scheduler:
    name: cosine
    warmup_steps: 1000
```

### 5.3 Use interpolation for derived values

```yaml
ckpt:
  dir: ${paths.results_dir}/ckpt/${run_id}
  save_interval: 10
```

### 5.4 Don't hardcode absolute paths

```yaml
# ✅ Good (relative to paths.results_dir)
results_dir: ${paths.results_dir}

# ❌ Bad (won't work on dev machines)
ckpt_dir: /home/test/reconstruction/results/ckpt
```

### 5.5 Comment non-obvious choices

```yaml
# beta=1e-3 matches original Tang et al. 2023 paper
aligner:
  beta: 1.0e-3

# chunk_length=40 covers ~10s of MEG at 4 Hz downsampling
data:
  chunk_length: 40
```

## 6. Adding a new model

When adding a new model (e.g., `fmri_attention`):

1. Create `configs/model/fmri_attention.yaml`:
   ```yaml
   name: fmri_attention
   backbone: cnn3d
   aligner: attention
   hidden_dim: 1024
   num_heads: 8
   num_layers: 3
   semantic_dim: 768
   ```

2. Update `configs/train.yaml`'s `defaults:` to use it (or let user override).

3. Register the model builder in `recon/models/registry.py`:
   ```python
   MODEL_REGISTRY["fmri_attention"] = build_fmri_attention
   ```

## 7. Adding a new dataset

Same pattern: `configs/data/<name>.yaml` + `recon/data/<name>.py` adapter.

## 8. Environment-specific paths

The `paths:` config group is the only place absolute paths live:

```yaml
# configs/paths/cluster.yaml
data_root: /home/test/reconstruction
results_root: /home/test/reconstruction/results
pretrained_root: /home/test/reconstruction/pretrained
gpt_path: ${paths.pretrained_root}/gpt2-chinese
brainomni_path: ${paths.pretrained_root}/brainomni
```

To add a new environment (e.g., a CI container):

```yaml
# configs/paths/ci.yaml
data_root: /tmp/fake_reconstruction
results_root: /tmp/results
pretrained_root: /tmp/pretrained
gpt_path: ${paths.pretrained_root}/fake_gpt
```

## 9. Experiment snapshots

When you find a config that produces a good result:

1. **Snapshot it** to `configs/experiment/NNN-<short-name>.yaml`
2. **Reference** the snapshot in the [experiment log](../research/04-experiment-log.md)
3. **Use it** for reproducibility:

```bash
python -m recon.cli.train +experiment=001-baseline-fmri-sub01
```

## 10. Anti-patterns

- ❌ Duplicating values across files (use interpolation)
- ❌ Hardcoded absolute paths outside `paths/`
- ❌ Magic numbers without comments
- ❌ Different naming conventions in different files (stick to snake_case)
- ❌ Deeply nested configs (>4 levels) — refactor

## 11. See also

- [ADR-0001: Why Hydra](../decisions/0001-why-hydra.md)
- [Architecture 01: Configuration flow](01-overview.md#5-configuration-flow)
- [Hydra docs](https://hydra.cc/)

---

Maintained by owner. Update when config conventions change.