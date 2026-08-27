#!/usr/bin/env bash
# scripts/smoke.sh — Run cluster smoke tests at various tiers.
#
# Usage:
#   bash scripts/smoke.sh --tier=L0          # Tier 0: fake data, 1 node
#   bash scripts/smoke.sh --tier=L1          # Tier 1: real data (sub 1), 1 node, 1 GPU
#   bash scripts/smoke.sh --tier=L2          # Tier 2: real data, 4 nodes, 8 GPU DDP
#   bash scripts/smoke.sh --tier=all         # Run all tiers
#
# See docs/standards/04-testing.md for tier definitions.
# Modality pairing: model=meg_model_a ↔ data=drdr (MEG);
#                   model=fmri3dcib   ↔ data=drdr_fmri (fMRI cube).

set -euo pipefail

# ───────────────────── Defaults ─────────────────────
TIER="L0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

usage() {
    grep -E '^# ' "${BASH_SOURCE[0]}" | sed 's/^# //'
    exit 1
}

# ───────────────────── Parse args ─────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --tier=*) TIER="${1#*=}" ;;
        --help|-h) usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
    shift
done

cd "${REPO_ROOT}"

# ───────────────────── Tier 0: Fake data, 1 node ─────────────────────
run_tier0() {
    echo "=== Tier 0: fake data, 1 node ==="
    python -m recon.cli.train \
        paths=cluster \
        model=meg_model_a \
        data=fake \
        train.smoke=true \
        train.amp_dtype=null \
        logging.wandb_mode=offline \
        2>&1 | tee "${LOG_DIR}/smoke_tier0.log"
}

# ───────────────────── Tier 1: Real data, 1 GPU ─────────────────────
run_tier1() {
    echo "=== Tier 1: real data (sub 1 MEG), 1 node, 1 GPU ==="
    python -m recon.cli.train \
        paths=cluster \
        model=meg_model_a \
        data=drdr \
        data.subjects=[1] \
        data.split.method=ratio \
        train.epochs=1 \
        train.batch_size=64 \
        train.amp_dtype=null \
        train.eval_interval=1 \
        train.save_interval=1 \
        logging.wandb_mode=offline \
        2>&1 | tee "${LOG_DIR}/smoke_tier1.log"
}

# ───────────────────── Tier 2: Real data, multi-node DDP ─────────────────────
run_tier2() {
    echo "=== Tier 2: real data (sub 1 MEG), 4 nodes, 8 GPU DDP ==="
    bash "${SCRIPT_DIR}/launch_multi_node.sh" \
        model=meg_model_a \
        data=drdr \
        data.subjects=[1] \
        data.split.method=ratio \
        train.epochs=1 \
        train.batch_size=128 \
        train.amp_dtype=null \
        2>&1 | tee "${LOG_DIR}/smoke_tier2.log"
}

# ───────────────────── Dispatch ─────────────────────
case "${TIER}" in
    L0|0|tier0) run_tier0 ;;
    L1|1|tier1) run_tier1 ;;
    L2|2|tier2) run_tier2 ;;
    all)
        run_tier0
        run_tier1
        run_tier2
        ;;
    *) echo "Unknown tier: ${TIER}" >&2; usage ;;
esac

echo ""
echo "Smoke test ${TIER} passed."
echo "Logs in: ${LOG_DIR}/smoke_tier*.log"
