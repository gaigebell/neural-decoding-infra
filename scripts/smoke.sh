#!/usr/bin/env bash
# scripts/smoke.sh — Run cluster smoke tests at various tiers.
#
# Usage:
#   bash scripts/smoke.sh --tier=L0          # Tier 0: fake data, 1 node (CPU or 1 GPU)
#   bash scripts/smoke.sh --tier=L1          # Tier 1: real data, 1 node (1 GPU)
#   bash scripts/smoke.sh --tier=L2          # Tier 2: real data, 2 nodes (2 GPU DDP)
#   bash scripts/smoke.sh --tier=all         # Run all tiers
#
# See docs/standards/04-testing.md for tier definitions.

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

# ───────────────────── Tier 0: Fake data, 1 GPU, no cluster ─────────────────────
run_tier0() {
    echo "=== Tier 0: fake data, 1 node, 1 GPU ==="
    python -m recon.cli.train \
        --config-path=configs \
        paths=cluster \
        model=fmri3dcib \
        data=fake \
        train.smoke=true \
        train.epochs=1 \
        train.batch_size=2 \
        train.amp=false \
        2>&1 | tee "${LOG_DIR}/smoke_tier0.log"
}

# ───────────────────── Tier 1: Real data, 1 GPU ─────────────────────
run_tier1() {
    echo "=== Tier 1: real data (1 subject), 1 node, 1 GPU ==="
    python -m recon.cli.train \
        --config-path=configs \
        paths=cluster \
        model=fmri3dcib \
        data=drdr \
        data.subjects=[1] \
        train.epochs=1 \
        train.batch_size=8 \
        2>&1 | tee "${LOG_DIR}/smoke_tier1.log"
}

# ───────────────────── Tier 2: Real data, 2 nodes, DDP ─────────────────────
run_tier2() {
    echo "=== Tier 2: real data (2 subjects), 2 nodes, 4 GPU DDP ==="
    bash "${SCRIPT_DIR}/launch_multi_node.sh" \
        model=fmri3dcib \
        data=drdr \
        data.subjects=[1,2] \
        train.epochs=1 \
        train.batch_size=16 \
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