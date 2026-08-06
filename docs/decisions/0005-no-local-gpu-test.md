# ADR-0005: No local GPU testing

- **Status**: Accepted
- **Date**: 2026-07-28
- **Deciders**: owner

## Context and problem statement

Originally we planned to have a 4-tier testing strategy:

- Tier 0: local CPU / fake-data smoke
- Tier 1: local GPU smoke on RTX 4060 (owner's dev machine)
- Tier 2: cluster single-node smoke
- Tier 3: cluster multi-node DDP

Reality in July 2026:

1. **Development machines are personal laptops** of each team member — heterogeneous OS / Python / GPU
2. **Owner's RTX 4060 (8G)** is too weak to run fMRI 3D CNN smoke (would OOM)
3. Even if owner's laptop could run smoke, **other team members cannot replicate** because their machines differ
4. Running GPU tests on owner's machine creates a "works on my machine" class of bugs

## Decision drivers

- Reproducibility across team members
- Cluster is the only place with real GPU capacity
- Single source of truth for "what works"
- We want fast feedback, but **not at the cost of test correctness**

## Considered options

### Option A: Local smoke on RTX 4060

- **Pro**: Fast feedback (1 min)
- **Con**: Owner-specific; other members can't run; OOMs on fMRI; not representative

### Option B: Local CPU smoke (no GPU)

- **Pro**: Works on any machine; tests pure Python correctness
- **Con**: Doesn't catch CUDA errors; doesn't catch GPU-specific bugs

### Option C: Cluster-only smoke (no local GPU)

- **Pro**: Single source of truth; matches production env
- **Pro**: All team members can trigger via PR (cron on cluster)
- **Con**: Slower feedback loop (5 min vs 1 min)

### Option D: Hybrid — local for CPU unit tests, cluster for GPU smoke

- **Pro**: Best of both worlds
- **Con**: Two test environments; risk of "passes CPU, fails GPU"

## Decision outcome

**Chosen option**: **Option D (hybrid), but with a discipline rule**:

- **Tier L0 (local)**: CPU-only unit tests + lint + import smoke. **Runs on any machine.**
- **Tier L1 (cluster)**: Fake-data GPU smoke on 1 node. **Runs on cluster via cron after merge.**
- **Tier L2 (cluster)**: Real-data 1-epoch smoke on 1 node. **Owner runs manually.**
- **Tier L3 (cluster)**: Multi-node DDP smoke. **Owner runs manually.**

The rule is: **GPU-touching tests run on the cluster, not locally.**

### Consequences

- ✅ Good: Any team member can clone and run `make test` without a GPU
- ✅ Good: Cluster smoke catches GPU-specific bugs (OOM, dtype, device mismatches)
- ✅ Good: Single env = reproducible
- ❌ Bad: Slower feedback for GPU bugs (5 min instead of 1 min)
- ❌ Bad: PR review must wait for cluster cron to validate
- ❓ Risk: If cluster is busy, cron smoke delays merge. **Mitigation**: run smoke manually if needed.

## What "local" means now

```bash
# Any team member, any machine:
make lint            # ruff + mypy
make test-unit       # CPU unit tests only
make test-imports    # ensure recon.X imports cleanly
```

## What "cluster" means

```bash
# Cluster cron, runs every 5 minutes:
bash scripts/smoke.sh --tier=L1  # fake data, 1 node

# Owner-only:
bash scripts/smoke.sh --tier=L2  # real data, 1 node
bash scripts/launch_multi_node.sh baseline  # multi-node DDP
```

## When to revisit

- Cluster grows to 16+ nodes → consider adding pre-merge cluster check via GitHub Actions on cluster (if cluster has internet to GH)
- Team grows beyond 5 → consider dedicated test environment

## Links

- [Standards: Testing](../standards/04-testing.md)
- [Guide: Run training](../guides/04-run-training.md)
- [ADR-0003: Drop SLURM](./0003-drop-slurm.md)