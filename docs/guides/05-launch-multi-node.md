# Guide 05: Launch multi-node training

> **Audience**: Anyone running a multi-node DDP experiment.

---

## Prerequisites

- Cluster access (see [Guide 01](01-setup-env.md))
- Code is up-to-date on cluster (rsync or git pull)
- A working training config that has run successfully on 1 node (see [Guide 04](04-run-training.md))

## When to use multi-node

- Single-node training takes too long (> 1 day)
- You want to sweep 4+ configs in parallel (one per node)
- Cross-subject training with 12 subjects + large batch size

## Node list

Currently working nodes:

| Node | IP | GPU |
|---|---|---|
| `cn3` | 10.0.1.3 | 2 × PH402 |
| `gn14` | 10.0.1.14 | 2 × PH402 |
| `gn15` | 10.0.1.15 | 2 × PH402 |
| `gn16` | 10.0.1.16 | 2 × PH402 |

Total: **8 GPUs**.

See [Cluster card](../research/03-cluster-card.md) for details.

## Pattern A: Manual ssh (current practice)

See [ADR-0003](../decisions/0003-drop-slurm.md) for why we do this.

```bash
# 1. SSH to mgmt
ssh your_user@mgmt.hpcc.com

# 2. Open 4 SSH sessions (one per compute node)
#    In each, run the same training command with a different RANK.
#
#    Master (rank 0) on cn3:
ssh cn3 "cd /home/test/reconstruction/neural-decoding-infra && \
  RANK=0 MASTER_ADDR=10.0.1.3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4 \
  train.nproc_per_node=2 train.node_rank=0"

#    Worker (rank 2) on gn14:
ssh gn14 "cd /home/test/reconstruction/neural-decoding-infra && \
  RANK=2 MASTER_ADDR=10.0.1.3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4 \
  train.nproc_per_node=2 train.node_rank=2"

#    Worker (rank 4) on gn15:
ssh gn15 "cd /home/test/reconstruction/neural-decoding-infra && \
  RANK=4 MASTER_ADDR=10.0.1.3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4 \
  train.nproc_per_node=2 train.node_rank=4"

#    Worker (rank 6) on gn16:
ssh gn16 "cd /home/test/reconstruction/neural-decoding-infra && \
  RANK=6 MASTER_ADDR=10.0.1.3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4 \
  train.nproc_per_node=2 train.node_rank=6"
```

**Important**: All 4 ssh sessions must run with the **same** config (model, data, hyperparams). Only `RANK`, `node_rank` differ.

## Pattern B: Use the launcher script (recommended)

```bash
# From mgmt
cd /home/test/reconstruction/neural-decoding-infra

# Launch
bash scripts/launch_multi_node.sh \
    model=fmri3dcib \
    data=drdr \
    train.epochs=100
```

The script (`scripts/launch_multi_node.sh`):

- SSHs to all 4 nodes in parallel
- Sets `MASTER_ADDR` automatically from the first node
- Sets `RANK` per node
- Forwards all Hydra overrides

See [ADR-0004](../decisions/0004-manual-ssh-launch.md).

## Pattern C: 4 parallel independent experiments (sweep)

You can also use each node for an independent experiment (no DDP coordination needed):

```bash
# Each node runs a different config
ssh cn3  "cd ... && python -m recon.cli.train train.lr=1e-4 train.beta=1e-2"
ssh gn14 "cd ... && python -m recon.cli.train train.lr=3e-4 train.beta=1e-3"
ssh gn15 "cd ... && python -m recon.cli.train train.lr=1e-3 train.beta=1e-2"
ssh gn16 "cd ... && python -m recon.cli.train train.lr=3e-3 train.beta=1e-3"
```

This is **the most efficient use of 4 nodes** for hyperparameter search.

## What you'll see

After ~30 seconds, you should see all 4 ranks appear in the master node's log:

```
[Rank 0] Initializing process group...
[Rank 0] world_size=8, master_addr=10.0.1.3
[Rank 2] Initializing process group...
[Rank 4] Initializing process group...
[Rank 6] Initializing process group...
[Rank 0] All ranks connected. Starting training.
```

If you don't see all 4 ranks within 60 seconds, check NCCL connectivity (see [Guide 08: Debug](08-debug-checklist.md)).

## NCCL debugging

If DDP hangs at startup:

```bash
# On the master node, check master_addr is reachable
ssh cn3 "ping -c 3 10.0.1.3"

# Check the master port is open (no nc, use curl)
ssh cn3 "curl -v telnet://10.0.1.3:29500 2>&1 | head -5"

# Enable verbose NCCL logging
ssh cn3 "cd ... && NCCL_DEBUG=INFO python -m recon.cli.train ..."
```

Common fixes:

- `export NCCL_SOCKET_IFNAME=p5p1` — use the right network interface
- `export NCCL_IB_DISABLE=1` — disable IB even if installed
- `export NCCL_P2P_LEVEL=PCI` — avoid NVLink if buggy

## Common issues

| Issue | Fix |
|---|---|
| NCCL timeout | Set `NCCL_SOCKET_IFNAME`; check network |
| Some ranks never connect | Check `MASTER_ADDR`; check firewall |
| OOM on one node | Lower per-GPU batch size |
| Slow all-reduce | Expected — no IB. Stay ≤ 8 ranks. |

## See also

- [ADR-0003: Drop SLURM](../decisions/0003-drop-slurm.md)
- [ADR-0004: Manual ssh launch](../decisions/0004-manual-ssh-launch.md)
- [Cluster card](../research/03-cluster-card.md)
- [Guide 04: Run training](04-run-training.md)
- [Guide 08: Debug checklist](08-debug-checklist.md)

---

Maintained by owner.