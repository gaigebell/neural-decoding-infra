# ADR-0003: Drop SLURM as the scheduler

- **Status**: Accepted
- **Date**: 2026-07-28
- **Deciders**: owner

## Context and problem statement

The cluster has SLURM installed but it does not function:

- `slurmctld` is running on the management node
- `slurmd` is "active running" on all 4 working compute nodes
- `sinfo` still reports every node as `down*` with reason "Not responding" or "Node unexpectedly re"

Manual diagnosis in July 2026 found:

1. `slurm.conf` is the default example file from the SLURM tarball, not customized for this cluster
2. `/etc/munge/munge.key` is missing on compute nodes (the file was never copied)
3. `nc` is not installed and cannot be installed (system too old, no internet from compute nodes)
4. Even simple `sbatch` jobs hang forever

Owner spent ~3 hours attempting to repair SLURM and concluded:

- The cluster predates current owner (set up in 2020)
- Original admin (小钟) is no longer reachable for fixes
- **Repair requires deep SLURM expertise we don't have on hand**
- **The cost of repair > the cost of working around it**

## Decision drivers

- We need to train models **now**, not in 2 weeks after debugging
- We only have 4 working compute nodes, 8 GPUs — manual ssh is feasible
- Single-owner project: cannot dedicate a week to HPC sysadmin work
- Future: if cluster scales to 16+ nodes, revisit this decision

## Considered options

### Option A: Fix SLURM

- **Pro**: Standard HPC pattern, scales beyond 4 nodes
- **Con**: Requires deep SLURM expertise; no admin support; could take 1 week+ of debugging

### Option B: Switch to a different scheduler (Kubernetes / Nomad)

- **Pro**: Modern, well-documented
- **Con**: Even more setup; no container support on CentOS 7 without docker (and docker is unavailable too)

### Option C: Manual ssh launches (current fallback)

- **Pro**: Works today; no new tooling; matches owner's existing workflow
- **Con**: Verbose; error-prone (master IP, RANK, WORLD_SIZE all hand-set); doesn't scale beyond ~8 nodes

### Option D: Build a thin launcher script

- **Pro**: Same as C but scripted; less error-prone
- **Con**: Custom code; not standard

## Decision outcome

**Chosen option**: **Option D (thin ssh launcher script)**, with **Option C as the current practice** until the launcher is written.

The launcher script wraps the ssh fan-out, sets `MASTER_ADDR`, `RANK`, `WORLD_SIZE` env vars automatically, and aggregates logs via W&B. See [ADR-0004](./0004-manual-ssh-launch.md).

### Consequences

- ✅ Good: Training works today; no HPC sysadmin rabbit hole
- ✅ Good: Forces us to be explicit about master/worker setup → clearer debugging
- ✅ Good: Matches what owner was already doing manually
- ❌ Bad: 4+ nodes becomes annoying without a real scheduler
- ❌ Bad: Cannot schedule multiple users / multiple jobs gracefully
- ❓ Risk: When the lab gets a new cluster admin, they will want SLURM. Migration cost is non-zero.

## When to revisit

This ADR should be revisited if:

- Cluster grows beyond 8 working nodes
- A new admin offers to fix SLURM
- Team grows beyond 1 owner (multi-user scheduling becomes a real problem)

## Links

- [Cluster card](../research/03-cluster-card.md)
- [ADR-0004: Manual ssh launch](./0004-manual-ssh-launch.md)
- [Guide: Launch multi-node training](../guides/05-launch-multi-node.md)