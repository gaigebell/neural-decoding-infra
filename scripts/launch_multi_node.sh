#!/usr/bin/env bash
# scripts/launch_multi_node.sh — Launch DDP training across multiple compute nodes.
#
# Usage:
#   bash scripts/launch_multi_node.sh [hydra-overrides...]
#
# Examples:
#   bash scripts/launch_multi_node.sh
#   bash scripts/launch_multi_node.sh model=fmri3dcib
#   bash scripts/launch_multi_node.sh model=meg_model_a train.epochs=50 train.lr=3e-4
#
# This script:
#   1. SSH's to all working compute nodes in parallel
#   2. Sets MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE env vars
#   3. Runs `python -m recon.cli.train` on each node
#   4. Waits for all nodes to finish
#
# Logs are aggregated via W&B (one run for all ranks).
# See docs/decisions/0004-manual-ssh-launch.md for rationale.

set -euo pipefail

# ───────────────────── Config ─────────────────────
NODES=(cn3 gn14 gn15 gn16)
GPUS_PER_NODE=2
MASTER_PORT="${MASTER_PORT:-29500}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ───────────────────── Derived ─────────────────────
NNODES=${#NODES[@]}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
MASTER_NODE=${NODES[0]}

# Resolve master addr dynamically from the master node
echo "Resolving master addr from ${MASTER_NODE}..."
MASTER_ADDR=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${MASTER_NODE}" \
    "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "")

if [[ -z "${MASTER_ADDR}" ]]; then
    echo "ERROR: Could not resolve MASTER_ADDR from ${MASTER_NODE}" >&2
    echo "Check ssh connectivity: ssh ${MASTER_NODE} 'hostname -I'" >&2
    exit 1
fi

echo "Configuration:"
echo "  nodes:        ${NODES[*]}"
echo "  nnodes:       ${NNODES}"
echo "  gpus/node:    ${GPUS_PER_NODE}"
echo "  world_size:   ${WORLD_SIZE}"
echo "  master:       ${MASTER_NODE} (${MASTER_ADDR}:${MASTER_PORT})"
echo ""

# ───────────────────── Build the command ─────────────────────
# The Python command to run on every node (RANK differs per node).
build_cmd() {
    local rank=$1
    cat <<EOF
cd ${REPO_ROOT} && \
RANK=${rank} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
WORLD_SIZE=${WORLD_SIZE} \
NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-p5p1} \
python -m recon.cli.train \
  --config-path=configs \
  paths=cluster \
  train.nnodes=${NNODES} \
  train.nproc_per_node=${GPUS_PER_NODE} \
  train.node_rank=${rank} \
  $@
EOF
}

# ───────────────────── Launch on each node ─────────────────────
echo "Launching on ${NNODES} nodes..."
PIDS=()

for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    RANK=$((i * GPUS_PER_NODE))
    CMD=$(build_cmd "${RANK}" "$@")

    echo "  → ${NODE} (rank=${RANK})"

    # SSH to node, run in background
    ssh -o StrictHostKeyChecking=no "${NODE}" "${CMD}" &
    PIDS+=($!)
done

echo ""
echo "All nodes launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

# ───────────────────── Wait ─────────────────────
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        echo "  ✗ PID ${PID} failed"
        FAILED=$((FAILED + 1))
    else
        echo "  ✓ PID ${PID} succeeded"
    fi
done

if [[ ${FAILED} -gt 0 ]]; then
    echo ""
    echo "ERROR: ${FAILED} node(s) failed. Check W&B for run details."
    exit 1
fi

echo ""
echo "All nodes completed successfully."