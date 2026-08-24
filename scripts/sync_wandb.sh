#!/usr/bin/env bash
# scripts/sync_wandb.sh — Upload offline W&B runs from the mgmt node.
#
# Compute nodes have no internet access, so training runs with
# WANDB_MODE=offline write metrics to ./wandb/ on the shared NFS mount.
# This script runs on the INTERNET-CONNECTED mgmt node and uploads all
# pending offline runs to the W&B server.
#
# Usage:
#   bash scripts/sync_wandb.sh             # sync everything under ./wandb
#   bash scripts/sync_wandb.sh <wandb_dir> # sync a specific directory
#
# Requires: wandb logged in on the mgmt node (`wandb login`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WANDB_DIR="${1:-${REPO_ROOT}/wandb}"

if [[ ! -d "${WANDB_DIR}" ]]; then
    echo "No wandb directory at ${WANDB_DIR} — nothing to sync."
    exit 0
fi

echo "Syncing offline W&B runs from ${WANDB_DIR} ..."
wandb sync --sync-all "${WANDB_DIR}"
echo "Done. Offline runs uploaded; pending runs stay in ${WANDB_DIR}."
