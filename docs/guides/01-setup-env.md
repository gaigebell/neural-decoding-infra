# Guide 01: Set up dev / cluster environment

> **Audience**: Anyone setting up to work on this project.

---

## Overview

This project has **two environments**:

1. **Dev machine** (any team member's laptop) — for writing code only
2. **Cluster** (4-node GPU cluster) — for running experiments

We deliberately do **not** run GPU code on dev machines. See [ADR-0005](../decisions/0005-no-local-gpu-test.md).

## Dev machine setup

### Prerequisites

- Python ≥ 3.10
- Git
- One of: macOS, Linux, Windows (with WSL recommended)

### Steps

```bash
# 1. Clone the repo
git clone git@github.com:<owner>/neural-decoding-infra.git
cd neural-decoding-infra

# 2. Create a virtual env
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Verify
make test-unit
make lint
```

If `make test-unit` passes, your dev env is ready.

### What you CAN do on dev machine

- ✅ Edit code
- ✅ Run unit tests (`make test-unit`)
- ✅ Run lint (`make lint`)
- ✅ Generate fake data (`python -m recon.data.fake_data`)
- ✅ View W&B dashboards (in browser)
- ✅ Write / review docs

### What you CANNOT do on dev machine

- ❌ Train on real data
- ❌ Train on real GPU
- ❌ Decode real stories

## Cluster setup

### Prerequisites

- SSH access to mgmt.hpcc.com (ask owner)
- Your SSH key added to mgmt's `~/.ssh/authorized_keys`

### Steps

```bash
# 1. SSH into management node
ssh your_user@mgmt.hpcc.com

# 2. Clone repo (on mgmt)
cd /home/test/reconstruction
git clone git@github.com:<owner>/neural-decoding-infra.git
cd neural-decoding-infra

# 3. Install dependencies (in a venv)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 4. Verify
python -c "import torch; print(torch.cuda.is_available())"  # should be True on compute node
```

### Working on a compute node

```bash
# SSH from mgmt to a compute node
ssh cn3

# Verify GPU
nvidia-smi

# Check data access
ls /home/test/reconstruction/mydata/derivatives/preprocessed_data/sub-01/

# Run a smoke test
cd /home/test/reconstruction/neural-decoding-infra
python -m recon.cli.train paths=cluster train.smoke=true
```

### Network layout

See [Cluster card](../research/03-cluster-card.md).

## Environment variables

For W&B, set your API key:

```bash
# On cluster (after activating venv)
export WANDB_API_KEY=<your_key>

# Or put it in .env (and add .env to .gitignore!)
echo "WANDB_API_KEY=<your_key>" > .env
```

For reproducibility, optionally pin seeds:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Common issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: recon` | `pip install -e ".[dev]"` |
| `CUDA not available` | You're on dev machine, not cluster. SSH to a compute node. |
| `pre-commit` not running | Run `pre-commit install` once per clone |
| `Permission denied` on cluster | Check SSH key; ask owner |
| `wandb: ERROR ...` | Check WANDB_API_KEY env var |
| Old Python on cluster | Use `python -m venv .venv` to get a fresh one |

## See also

- [Cluster card](../research/03-cluster-card.md)
- [Standard 01: Python style](../standards/01-python-style.md)
- [ADR-0005: No local GPU testing](../decisions/0005-no-local-gpu-test.md)

---

Maintained by owner.