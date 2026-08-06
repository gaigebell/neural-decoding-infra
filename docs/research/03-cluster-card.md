# 03 — Cluster Card

> **Status**: Living document
> **Audience**: Anyone running experiments on the cluster
> **Last updated**: 2026-07-28

This document follows the [Model Cards / System Cards](https://arxiv.org/abs/1810.03993) convention adapted for compute infrastructure.

---

## 1. Cluster summary

| Field | Value |
|---|---|
| **Cluster name** | `sysuhpcc` (中大心理系 HPC) |
| **Location** | 中山大学心理学系 |
| **Management node** | `mgmt.hpcc.com` (10.0.1.100) |
| **Login node** | `login.hpcc.com` (10.0.1.102) |
| **OS** | CentOS 7.4 (kernel 3.10.0) |
| **Scheduler** | SLURM (controller up; **compute nodes DOWN, manual ssh used**) |
| **Effective GPU** | **8× NVIDIA PH402 SKU 200** (Pascal, 32GB each) |
| **Working nodes** | `cn3`, `gn14`, `gn15`, `gn16` |
| **Broken nodes** | `cn1`, `cn2`, `cn4`, `gn11`, `gn12`, `gn13` (hardware failures) |
| **Documentation** | `中山大学心理系高性能集群最新指南20200522（小钟）.pdf` |
| **Total storage** | 14.6 TB on `/home`, 1.1 TB on `/share` |

## 2. Per-node specs (working nodes only)

| Node | IP | CPU cores | RAM | GPU count | GPU type | Notes |
|---|---|---|---|---|---|---|
| `cn3` | 10.0.1.3 | ~40 | 62 GB | 2 | PH402 SKU 200 (Pascal) | Oldest, still works |
| `gn14` | 10.0.1.14 | ~40 | 62 GB | 2 | PH402 SKU 200 (Pascal) | |
| `gn15` | 10.0.1.15 | ~40 | 62 GB | 2 | PH402 SKU 200 (Pascal) | |
| `gn16` | 10.0.1.16 | ~40 | 62 GB | 2 | PH402 SKU 200 (Pascal) | |

> ⚠️ Specs are estimated from `nvidia-smi` + `free -h` on `cn3`. Other nodes assumed identical (owner verified).

### GPU details

- **Model**: NVIDIA PH402 SKU 200 (custom Pascal SKU; not consumer-grade)
- **Architecture**: Pascal (compute capability ~6.x)
- **Memory**: 32 GB per GPU
- **Interconnect**: NV4 between GPU pairs on same node (limited NVLink)
- **Inter-node**: **No InfiniBand.** Falls back to TCP over 10 GbE (`p5p1`, 10.0.1.x subnet)
- **bf16**: ❌ not natively supported (use fp16 for AMP)
- **NCCL**: works on TCP, but slow for large messages (>1 MB)

## 3. Network topology

```
                  ┌─────────────────────────────────────────┐
                  │         Management node                 │
                  │         mgmt.hpcc.com (10.0.1.100)      │
                  │  • slurmctld (active)                   │
                  │  • /home NFS export                     │
                  │  • /share NFS export (1.1 TB)           │
                  │  • SSH gateway                          │
                  └─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   10.0.1.x                    10.0.1.x                    10.0.1.x
   ┌──────────┐            ┌──────────┐                ┌──────────┐
   │  cn3     │            │  gn14    │                │ gn15/16  │
   │  GPU×2   │            │  GPU×2   │                │  GPU×2   │
   └──────────┘            └──────────┘                └──────────┘
   (NFS mount)             (NFS mount)                 (NFS mount)
```

- **Subnet**: 10.0.1.x/24 (private)
- **DNS**: hosts in `/etc/hosts` (no central DNS)
- **Public access**: via management node only
- **Ports**: NCCL needs 29500 + 29501 (default torchrun); SLURM needs 6817/6818

## 4. Storage

| Mount | Size | Used | Available | Purpose |
|---|---|---|---|---|
| `/home` (mgmt root) | 14.6 TB | 2.4 TB | 12 TB | User home dirs + project root |
| `/share` (mgmt) | 1.1 TB | 252 GB | 801 GB | Shared scratch space |
| `/` (compute nodes) | 494 GB | 39 GB | 456 GB | Per-node local |

**Project location**: `/home/test/reconstruction/` (NFS-mounted on all compute nodes as `/home/test/reconstruction/`).

**Data on this mount**:
- Code: `recon/` (and now `neural-decoding-infra/` after migration)
- Data: `mydata/derivatives/preprocessed_data/sub-XX/{MEG,MNI,CIFTI}/`
- Results: `results/MEG/{zresp,zstim}/`
- Pretrained: `pretrained/{gpt2-chinese,brainomni}/`

## 5. Known limitations

| Limitation | Workaround |
|---|---|
| **No InfiniBand** | NCCL uses TCP; slow for large all-reduce. Stay below 8-way DDP. |
| **Only 8 GPU total** | Cannot trigger true scaling law with current data. |
| **6 of 10 compute nodes broken** | Manual hardware triage; only cn3 + gn14-16 reliable. |
| **SLURM not functional** | `slurmctld` is up, but `slurmd` cannot reach it (munge/conf issue). **Use manual ssh.** |
| **No `nc` installed** | Use `curl -v telnet://host:port` or python sockets for port tests. |
| **No `ibstat`** | No InfiniBand to query anyway. |
| **Old kernel (3.10.0)** | Newer PyTorch wheels may have issues; pin torch<2.4 if needed. |
| **No internet from compute nodes** | Pre-download models on mgmt; copy to `/home/test/pretrained/`. |
| **Cluster shared among lab members** | Be polite: don't hog all 8 GPUs during working hours. |

## 6. SLURM status (degenerate)

```
$ sinfo -N -l
NODELIST   NODES PARTITION       STATE CPUS    ... REASON
cn3            1    debug*        down    1    ... Node unexpectedly re
gn14           1    debug*        down    1    ... Node unexpectedly re
gn15           1    debug*        down    1    ... Node unexpectedly re
gn16           1    debug*        down    1    ... Node unexpectedly re
```

- **slurmctld**: running on mgmt
- **slurmd**: started on all compute nodes but not communicating with slurmctld
- **Root cause**: likely munge key mismatch or `slurm.conf` divergence (see ADR-0003)

**Decision**: **Do NOT pursue SLURM further.** Use manual ssh for multi-node launches. See [ADR-0003](../decisions/0003-drop-slurm.md) and [ADR-0004](../decisions/0004-manual-ssh-launch.md).

## 7. Manual launch pattern

```bash
# From mgmt node, train on 4 nodes × 2 GPU = 8-way DDP

# Master node (rank 0)
ssh cn3 "cd /home/test/reconstruction && \
  RANK=0 MASTER_ADDR=cn3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4"

# Worker nodes
ssh gn14 "cd /home/test/reconstruction && \
  RANK=2 MASTER_ADDR=cn3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4"
ssh gn15 "cd /home/test/reconstruction && \
  RANK=4 MASTER_ADDR=cn3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4"
ssh gn16 "cd /home/test/reconstruction && \
  RANK=6 MASTER_ADDR=cn3 MASTER_PORT=29500 WORLD_SIZE=8 \
  python -m recon.cli.train paths=cluster train.nnodes=4"
```

For a one-liner wrapper, see [`scripts/launch_multi_node.sh`](../../scripts/launch_multi_node.sh).

## 8. Monitoring & debugging

| Question | Command |
|---|---|
| Are GPUs busy? | `ssh cn3 "nvidia-smi"` |
| Is my training running? | `ssh cn3 "ps aux \| grep python"` |
| Disk space? | `df -h /home` |
| Network test (no `nc`) | `curl -v telnet://cn3:29500` |
| Python env? | `which python && python --version` |

## 9. See also

- [Data card](02-data-card.md) — what runs on this cluster
- [ADR-0003: Drop SLURM](../decisions/0003-drop-slurm.md) — why no scheduler
- [ADR-0004: Manual ssh launch](../decisions/0004-manual-ssh-launch.md) — how we launch
- [Guide: Launch multi-node training](../guides/05-launch-multi-node.md)

---

Maintained by owner. Update when cluster topology changes.