# Guide 04: Run training

> **Audience**: Anyone running a training experiment.

---

## Prerequisites

- Cluster access (see [Guide 01: Setup environment](01-setup-env.md))
- A model registered in `MODEL_REGISTRY` (see [Guide 03: Write a model](03-write-model.md))
- A dataset adapter (see [Guide 02: Write a data adapter](02-write-data-adapter.md))

## Run modes

| Mode | Cluster config | What it does | Duration |
|---|---|---|---|
| **Smoke** | `train.smoke=true` | 1 epoch, batch=2, fake data | ~5 min |
| **Quick** | `train.epochs=5 train.subjects=[1]` | 5 epochs, 1 subject | ~30 min |
| **Standard** | (default) | Full training as configured | hours–days |
| **Sweep** | `hydra-multirun --multirun` | Multiple configs in parallel | hours |

## Single-node training (one compute node)

```bash
# SSH to a compute node
ssh cn3

cd /home/test/reconstruction/neural-decoding-infra

# Smoke test first
python -m recon.cli.train \
    --config-path=configs \
    paths=cluster \
    model=fmri3dcib \
    data=drdr \
    train.smoke=true

# Real training (single subject)
python -m recon.cli.train \
    paths=cluster \
    model=fmri3dcib \
    data=drdr \
    data.subjects=[1] \
    train.epochs=100 \
    train.batch_size=32
```

## What you'll see

```
[2026-07-28 14:23:01] INFO     Loading model fmri3dcib
[2026-07-28 14:23:02] INFO     Loading dataset drdr (subjects=[1], stories=1-50)
[2026-07-28 14:23:05] INFO     Dataset: 12450 samples
[2026-07-28 14:23:08] INFO     Model: 12,453,201 parameters
[2026-07-28 14:23:08] INFO     W&B: https://wandb.ai/.../runs/abc123
[2026-07-28 14:23:10] INFO     Starting epoch 1/100
[2026-07-28 14:23:42] INFO     Epoch 1: loss=0.342, cosine=0.876, mse=0.012 (32s)
...
```

W&B dashboard will open in your browser (click the link printed).

## Useful overrides

```bash
# Different learning rate
python -m recon.cli.train train.lr=3e-4

# Disable AMP (for debugging)
python -m recon.cli.train train.amp=false

# Save more checkpoints
python -m recon.cli.train ckpt.save_interval=5

# Different subject
python -m recon.cli.train data.subjects=[2]

# Multiple subjects (cross-subject)
python -m recon.cli.train data.subjects=[1,2,3,4] data.cross_sub=true
```

## Background / long-running

Use `tmux` so your SSH disconnect doesn't kill training:

```bash
# On compute node
tmux new -s training
python -m recon.cli.train ...    # runs in foreground
# Ctrl-B then D to detach

# Re-attach later
tmux attach -t training

# List all tmux sessions
tmux ls
```

Or run with `nohup` and check logs:

```bash
nohup python -m recon.cli.train ... > logs/run_$(date +%s).log 2>&1 &
tail -f logs/run_*.log
```

## Where things go

| Artifact | Location |
|---|---|
| Logs | stdout + W&B run |
| Checkpoints | `${paths.results_root}/ckpt/<run_id>/` |
| Final metrics | W&B run summary |
| Config snapshot | W&B run config + local `results/<run_id>/config.yaml` |

## Canceling a training run

```bash
# Find the process
ps aux | grep "python -m recon.cli.train"

# Kill it
kill <pid>            # graceful
kill -9 <pid>         # force
```

Or in `tmux`: `Ctrl-C` inside the session.

## Common issues

| Issue | Fix |
|---|---|
| OOM | Lower `train.batch_size` or model size |
| Loss is NaN | Lower `train.lr`; enable `train.amp=true` |
| Slow IO | Check NFS; consider mmap (see [Architecture 02](../architecture/02-data-pipeline.md)) |
| W&B not connecting | Check `WANDB_API_KEY` env var |
| Stuck on data loading | `Ctrl-C` and check data paths in config |

## See also

- [Guide 05: Launch multi-node training](05-launch-multi-node.md)
- [Guide 06: Run decoding](06-run-decoding.md)
- [Guide 08: Debug checklist](08-debug-checklist.md)
- [Standard 05: Configuration](../standards/05-configuration.md)
- [ADR-0006: Unified Trainer](../decisions/0006-unified-trainer.md)

---

Maintained by owner.