# 01 — System Overview

> **Status**: Living document — updated as the system evolves
> **Audience**: Anyone who needs to understand the system's shape
> **Last updated**: 2026-07-28

---

## 1. Purpose

This document describes the **top-level architecture** of `neural-decoding-infra`: what modules exist, how they communicate, and what flows through them.

For the rationale behind any specific design choice, see [`docs/decisions/`](../decisions/) (ADRs).

## 2. Bird's-eye view

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          neural-decoding-infra                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────┐    │
│   │   Data     │    │   Models   │    │   Engine    │    │  Decoders    │    │
│   │  (schema,  │───▶│ (registry, │───▶│  (trainer,  │───▶│ (beam,       │    │
│   │  loaders)  │    │  fmri/meg) │    │ evaluator)  │    │  nucleus)    │    │
│   └────────────┘    └────────────┘    └─────────────┘    └──────────────┘    │
│         ▲                  ▲                  ▲                  ▲           │
│         │                  │                  │                  │           │
│         └──────────────────┴──────────────────┴──────────────────┘           │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌──────────────────┐                            │
│                              │     Configs      │                            │
│                              │     (Hydra)      │                            │
│                              └──────────────────┘                            │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌──────────────────┐                            │
│                              │   CLI + Scripts  │                            │
│                              │  (entrypoints)   │                            │
│                              └──────────────────┘                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────┐
                │       External (cluster / W&B / Git)      │
                │  • Cluster: 4 nodes × 2 PH402            │
                │  • W&B: experiment tracking               │
                │  • GitHub: code repo (private)            │
                └──────────────────────────────────────────┘
```

## 3. Module responsibilities

| Module | Path | Responsibility |
|---|---|---|
| **`recon.data`** | `recon/data/` | Define data schemas, load datasets, generate fake data for tests. **Knows nothing about models.** |
| **`recon.models`** | `recon/models/` | Define neural-network architectures. Registered in a central registry. **Knows nothing about training loops.** |
| **`recon.engine`** | `recon/engine/` | Training loop, evaluation loop, distributed coordination. **Knows about models and data, not about clusters.** |
| **`recon.decoders`** | `recon/decoders/` | Convert brain-predicted semantic vectors back to text. **Knows about language models and beam search, not about training.** |
| **`recon.cli`** | `recon/cli/` | Thin command-line entry points. Parse Hydra config, call into engine. |
| **`recon.utils`** | `recon/utils/` | Logging (W&B wrapper), I/O, metrics helpers. |
| **`configs/`** | repo root | Hydra YAML configurations, organized by concern (model / data / decoder / etc.). |

## 4. End-to-end data flow

### 4.1 Training flow

```
[Raw brain data on cluster]
  ↓
[recon.data.schema]      ← pydantic validation, type checking
  ↓
[recon.data.{drdr,...}]  ← adapter-specific loader, returns PyTorch Dataset
  ↓
[DataLoader]             ← batching, shuffling, distributed sampling
  ↓
[recon.models.{fmri,meg,brainomni}]  ← forward pass → semantic vector
  ↓
[recon.engine.trainer]   ← compute loss, backward, optimizer, log
  ↓
[W&B]                    ← metrics, hyperparameters, checkpoint
  ↓
[Checkpoint on disk]     ← /home/test/reconstruction/results/ckpt/<run_id>/
```

### 4.2 Decoding flow

```
[Trained checkpoint]
  ↓
[recon.engine.trainer.load_for_inference]  ← load model + weights
  ↓
[Brain signal at inference time]
  ↓
[recon.models.X]   ← forward → predicted semantic vector (B, T, 768)
  ↓
[recon.decoders.beam]
   • for each time step:
     1. get nucleus candidates from LM (GPT-2 with KV cache)
     2. embed candidates → semantic vectors
     3. cosine-similarity with brain-predicted vector
     4. update beam state
  ↓
[Output: best-scoring character sequence]
```

### 4.3 Evaluation flow

```
[Predicted characters]   [Ground-truth characters]
        ↓                       ↓
        └──────────┬────────────┘
                   ↓
        [recon.engine.evaluator]
          • CRR (Character Recognition Rate)
          • CER (Character Error Rate)
          • BLEU (n-gram overlap)
          • Top-k accuracy
        ↓
        [JSON + W&B log]
```

## 5. Configuration flow

```
configs/
├── train.yaml               ← default entry point
│   ├── defaults: [model/fmri3dcib, data/drdr, decoder/beam]
│   ├── paths: cluster       ← user override
│   └── train.epochs: 100
├── model/
│   ├── fmri3dcib.yaml       ← model-specific config
│   ├── meg_model_a.yaml
│   └── brainomni_align.yaml
├── data/
│   ├── drdr.yaml
│   └── brainomni.yaml
├── decoder/
│   ├── beam.yaml
│   └── nucleus.yaml
└── paths/
    ├── cluster.yaml         ← path overrides for cluster
    ├── mgmt.yaml            ← path overrides for management node
    └── local.yaml           ← path overrides for dev machine
```

Hydra composes these at runtime. The CLI is the only place that user-provided overrides enter.

See [ADR-0001: Why Hydra](../decisions/0001-why-hydra.md) and [standards/05-configuration.md](../standards/05-configuration.md).

## 6. State management

What is **stateful** vs **stateless**:

| Component | Stateful? | Where state lives |
|---|---|---|
| Models | ✅ (weights) | Checkpoint files |
| Optimizer | ✅ (state) | Checkpoint files |
| Data loaders | ❌ | Resumable via RNG seed |
| Decoders (beam) | ✅ (beam state) | In-memory only |
| Config | ❌ | Immutable per run |
| W&B run | ✅ (server-side) | W&B cloud |

**Reproducibility rule**: every run is determined by `(git SHA, config hash, data version, seed)`. See [standards/04-testing.md §Reproducibility](../standards/04-testing.md).

## 7. Failure boundaries

What fails in which module:

| Failure | Caught by | User action |
|---|---|---|
| Bad data file | `recon.data.schema` validation | Fix preprocessing |
| OOM during training | PyTorch runtime error | Lower batch size in config |
| NCCL timeout (DDP) | Distributed backend | Check cluster network |
| Wrong model output shape | Model's own assertion | Fix forward method |
| Missing config key | Hydra validation | Add to YAML |

## 8. Extension points

When adding a new feature, these are the canonical places to extend:

| You want to... | Add to... |
|---|---|
| Use a new dataset | `recon/data/{name}.py` + `configs/data/{name}.yaml` |
| Use a new model | `recon/models/{kind}/{name}.py` + register in `registry.py` + `configs/model/{name}.yaml` |
| Use a new loss | Add to `recon/engine/trainer.py` (single function) |
| Use a new decoder | `recon/decoders/{name}.py` + `configs/decoder/{name}.yaml` |
| Add a CLI subcommand | New file in `recon/cli/` |

## 9. Cross-cutting concerns

These are NOT a single module but spread across the system:

- **Logging**: `recon.utils.logging` provides a `get_logger()` and a W&B wrapper. Used everywhere.
- **Reproducibility**: RNG seeds set in `recon.engine.trainer.train_epoch`.
- **Distributed coordination**: `torch.distributed` is wrapped in `recon.engine.trainer` only.
- **Checkpointing**: Handled by `recon.engine.trainer` (consistent format across models).

## 10. Anti-patterns

Things we explicitly do NOT do:

- ❌ Hardcoded paths in module code (use configs)
- ❌ Direct `print()` for important info (use logger)
- ❌ One-off scripts outside `scripts/`
- ❌ New model variant by copy-pasting existing model file
- ❌ Training logic in `recon.models.*` (forward only)
- ❌ Data loading logic in `recon.engine.*` (use `recon.data.*`)

## 11. See also

- [02 — Data pipeline](02-data-pipeline.md)
- [03 — Model registry](03-model-registry.md)
- [04 — Training engine](04-training-engine.md)
- [05 — Decoding engine](05-decoding-engine.md)
- [06 — Evaluation](06-evaluation.md)
- [All ADRs](../decisions/)

---

Maintained by owner. Update when module boundaries change.