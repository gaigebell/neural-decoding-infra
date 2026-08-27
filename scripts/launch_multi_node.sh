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

# One RUN_ID for the WHOLE job (each rank otherwise resolves
# ${now:...} independently -> inconsistent ckpt/wandb dirs)
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

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
# The Python command to run on every node (one per GPU: RANK and
# LOCAL_RANK differ per process).
build_cmd() {
    local rank=$1
    local local_rank=$2
    cat <<EOF
cd ${REPO_ROOT} && \
RANK=${rank} \
LOCAL_RANK=${local_rank} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
WORLD_SIZE=${WORLD_SIZE} \
RUN_ID=${RUN_ID} \
WANDB_MODE=${WANDB_MODE:-offline} \
NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} \
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} \
python -m recon.cli.train \
  --config-path=configs \
  paths=cluster \
  $@
EOF
}

# ───────────────────── Launch on each node ─────────────────────
# Ctrl-C on the mgmt terminal must kill the remote processes too —
# otherwise half the ranks keep running and the others hang in all_reduce.
cleanup_remote() {
    echo ""
    echo "Interrupted — killing remote training processes..."
    for NODE in "${NODES[@]}"; do
        ssh -o StrictHostKeyChecking=no "${NODE}" \
            "pkill -f 'recon.cli.train' 2>/dev/null || true" &
    done
    wait
    echo "Cleanup done."
}
trap cleanup_remote INT TERM

echo "Launching ${WORLD_SIZE} processes on ${NNODES} nodes..."
PIDS=()

for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
        RANK=$((i * GPUS_PER_NODE + gpu))
        CMD=$(build_cmd "${RANK}" "${gpu}" "$@")

        echo "  → ${NODE} (rank=${RANK}, local_rank=${gpu})"

        # SSH to node, run in background (one process per GPU)
        ssh -o StrictHostKeyChecking=no "${NODE}" "${CMD}" &
        PIDS+=($!)
    done
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