# ADR-0004: Manual ssh for multi-node launch

- **Status**: Accepted
- **Date**: 2026-07-28
- **Deciders**: owner

## Context and problem statement

Given [ADR-0003](./0003-drop-slurm.md) (no scheduler), we need a way to launch a multi-node training job. The cluster has 4 working compute nodes (`cn3`, `gn14`, `gn15`, `gn16`), each with 2 PH402 GPUs, totaling 8 GPUs.

The naive approach — ssh to each node manually and run `python ...` on each — has been owner's working pattern. But:

- Each launch involves setting `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE` env vars correctly
- Easy to typo a rank and silently start training with broken DDP coordination
- Repeating the same shell snippet every experiment is tedious

We considered:

1. **Keep manual ssh (status quo)** — works but error-prone
2. **One bash wrapper script** — minimal, easy to inspect
3. **Python orchestration tool (Ray, etc.)** — overkill for 4 nodes

## Decision drivers

- Minimize complexity (single-owner)
- Must be inspectable line-by-line (no magic)
- Must handle the common case (4 nodes, 8 GPUs) without ceremony
- Must aggregate logs via W&B (not local files)

## Considered options

### Option A: Keep doing it manually

- **Pro**: Zero new code
- **Con**: Error-prone; typo = wasted GPU-hours

### Option B: Bash wrapper script (`scripts/launch_multi_node.sh`)

- **Pro**: ~50 lines of bash; easy to read; works today
- **Pro**: Auto-derives `MASTER_ADDR`, computes `RANK` per ssh, runs in parallel
- **Con**: Bash is fragile; harder to test

### Option C: Python orchestration (Ray / Dask / Hydra multirun)

- **Pro**: More powerful; easier to extend
- **Con**: Heavy dependency; not needed for 4 nodes

### Option D: GNU parallel / pssh

- **Pro**: Battle-tested ssh fan-out
- **Con**: Same outcome as B with one more dep

## Decision outcome

**Chosen option**: **Option B (bash wrapper script)**.

The script lives at `scripts/launch_multi_node.sh` and accepts:

- A Hydra config name (`baseline`, `meg_a`, etc.)
- Extra Hydra overrides

It auto-resolves:

- Master node IP via `ssh cn3 "hostname -I"`
- Per-node `RANK` from a known nodes list
- `WORLD_SIZE = 8` (4 nodes × 2 GPU) — configurable

It then ssh's each node in parallel and runs:

```bash
RANK=<i> MASTER_ADDR=<ip> MASTER_PORT=29500 WORLD_SIZE=8 \
    python -m recon.cli.train <config> <overrides>
```

See [Guide 05: Launch multi-node training](../guides/05-launch-multi-node.md) for full usage.

### Consequences

- ✅ Good: Launching becomes a one-liner
- ✅ Good: `MASTER_ADDR` etc. are no longer typed by hand
- ✅ Good: All logs go through W&B, not local files (no aggregation needed)
- ✅ Good: Script is small enough to read in 5 minutes
- ❌ Bad: Doesn't auto-restart failed nodes (caller must re-run)
- ❌ Bad: Doesn't detect node failures during run (PyTorch will hang on NCCL timeout)
- ❓ Risk: Beyond 8 nodes, ssh fan-out in parallel may exhaust mgmt's SSH slots (typical limit ~10 concurrent)

## Implementation sketch

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-baseline}"
shift || true

NODES=(cn3 gn14 gn15 gn16)
GPUS_PER_NODE=2
NNODES=${#NODES[@]}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

MASTER_NODE=${NODES[0]}
MASTER_ADDR=$(ssh "$MASTER_NODE" "hostname -I | awk '{print \$1}'")
MASTER_PORT=29500

for i in "${!NODES[@]}"; do
  NODE="${NODES[$i]}"
  RANK=$((i * GPUS_PER_NODE))
  echo "Launching $NODE (rank=$RANK)..."

  ssh -o StrictHostKeyChecking=no "$NODE" \
    "cd /home/test/reconstruction && \
     RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
     WORLD_SIZE=$WORLD_SIZE \
     python -m recon.cli.train paths=cluster train.nnodes=$NNODES \
     train.nproc_per_node=$GPUS_PER_NODE train.node_rank=$RANK $@" &
done

wait
echo "All nodes finished."
```

## When to revisit

- Cluster grows to >8 working nodes → consider SLURM, Ray, or pssh
- Need queue-based scheduling for multi-user → revisit [ADR-0003](./0003-drop-slurm.md)

## Links

- [ADR-0003: Drop SLURM](./0003-drop-slurm.md)
- [Cluster card](../research/03-cluster-card.md)
- [Guide: Launch multi-node training](../guides/05-launch-multi-node.md)