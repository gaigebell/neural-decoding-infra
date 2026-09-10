#!/usr/bin/env bash
# scripts/run_preprocess.sh — Cluster dispatcher for the preprocessing pipeline.
#
# Runs on the MGMT node. Stage-by-stage, it slices the (subject × story)
# task space into chunks and ssh-dispatches workers to the compute nodes
# (same pattern as launch_multi_node.sh). Each worker invokes
# `python -m recon.cli.preprocess` which SKIPS artifacts whose sidecar is
# valid — so re-running this script resumes an interrupted run for free
# (the filesystem is the task queue).
#
# Usage:
#   bash scripts/run_preprocess.sh                          # default stages
#   bash scripts/run_preprocess.sh meg_zresp meg_context    # specific stages
#   STAGES_OVERRIDE=1 SUBJECTS="1 2 3" bash scripts/run_preprocess.sh
#   DRY_RUN=1 bash scripts/run_preprocess.sh                # print commands only
#
# Progress (on mgmt):
#   python -m recon.cli.preprocess_run status
#
# Env overrides:
#   NODES, WORKERS_PER_NODE, GPU_NODE, GPU_WORKERS, SUBJECTS, N_STORIES,
#   CHUNK, DRY_RUN, STAGES_OVERRIDE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/preprocess"
mkdir -p "${LOG_DIR}"

# ───────────────────── Config ─────────────────────
NODES=(${NODES:-gn11 gn12 cn3 gn14 gn15 gn16})
WORKERS_PER_NODE=${WORKERS_PER_NODE:-8}
GPU_NODE=${GPU_NODE:-cn3}
GPU_WORKERS=${GPU_WORKERS:-1}
SUBJECTS=(${SUBJECTS:-1 2 3 4 5 6 7 8 9 10 11 12})
N_STORIES=${N_STORIES:-60}
CHUNK=${CHUNK:-5}                # stories per worker invocation
DRY_RUN=${DRY_RUN:-0}

# Default stage order respects dependencies (ds before delay/zresp;
# time_align before brainomni segments; segments before encode).
DEFAULT_STAGES=(gpt_char_features time_align semantic_downsample semantic_delay meg_zresp meg_context meg_brainomni_segments brainomni_encode)
STAGES=("${@}")                   # positional args override
if [[ ${#STAGES[@]} -eq 0 ]]; then STAGES=("${DEFAULT_STAGES[@]}"); fi

echo "Configuration: nodes=${NODES[*]} workers/node=${WORKERS_PER_NODE} subjects=${SUBJECTS[*]}"
echo "Stages: ${STAGES[*]}"
echo ""

# ───────────────────── Slicing ─────────────────────
# Emit "sub start end" lines: per subject, story ranges of size CHUNK.
emit_chunks() {
    local subs=("${SUBJECTS[@]}")
    if [[ "$1" == "subject-independent" ]]; then subs=(1); fi
    for sub in "${subs[@]}"; do
        for s in $(seq 1 "${CHUNK}" "${N_STORIES}"); do
            local e=$((s + CHUNK - 1))
            (( e > N_STORIES )) && e=${N_STORIES}
            echo "${sub} ${s} ${e}"
        done
    done
}

cleanup_remote() {
    echo ""
    echo "Interrupted — killing remote preprocessing workers..."
    for NODE in "${NODES[@]}"; do
        ssh -o StrictHostKeyChecking=no "${NODE}" \
            "pkill -f 'recon.cli.preprocess' 2>/dev/null || true" &
    done
    wait
    echo "Cleanup done."
}
trap cleanup_remote INT TERM

# ───────────────────── Per-stage dispatch ─────────────────────
for STAGE in "${STAGES[@]}"; do
    echo "=== Stage ${STAGE} ==="
    case "${STAGE}" in
        gpt_char_features|semantic_downsample|brainomni_encode)
            SUBJ_INDEP=0; W=$((GPU_WORKERS)); NODELIST=("${GPU_NODE}") ;;
        meg_brainomni_segments|*)
            SUBJ_INDEP=0; W=$(( ${#NODES[@]} * WORKERS_PER_NODE )); NODELIST=("${NODES[@]}") ;;
    esac
    # gpt/downsample artifacts are per-story only (subject-independent)
    if [[ "${STAGE}" == "gpt_char_features" || "${STAGE}" == "semantic_downsample" ]]; then
        SUBJ_INDEP=1
    fi

    if [[ ${SUBJ_INDEP} -eq 1 ]]; then
        mapfile -t CHUNKS < <(emit_chunks subject-independent)
    else
        mapfile -t CHUNKS < <(emit_chunks all)
    fi
    echo "  tasks=${#CHUNKS[@]} chunks, workers=${W}"

    PIDS=()
    for w in $(seq 0 $((W - 1))); do
        NODE="${NODELIST[$((w % ${#NODELIST[@]}))]}"
        # Collect this worker's chunks (round-robin by index)
        WORKER_CMDS=""
        for ((i = w; i < ${#CHUNKS[@]}; i += W)); do
            read -r sub s e <<< "${CHUNKS[$i]}"
            STORIES_LIST="[$(seq -s, "${s}" "${e}")]"
            WORKER_CMDS+="python -m recon.cli.preprocess stage=${STAGE} subject=${sub} stories='${STORIES_LIST}' paths=cluster resume=true || true; "
        done
        if [[ -z "${WORKER_CMDS}" ]]; then continue; fi
        LOG="${LOG_DIR}/${STAGE}_${NODE}_w${w}.log"
        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "  [dry] ${NODE}: ${WORKER_CMDS}"
            continue
        fi
        echo "  → ${NODE} worker ${w} (log: ${LOG})"
        ssh -o StrictHostKeyChecking=no "${NODE}" \
            "cd ${REPO_ROOT} && conda activate ndinf && bash -c '${WORKER_CMDS}'" \
            > "${LOG}" 2>&1 &
        PIDS+=($!)
    done
    [[ ${DRY_RUN} -eq 1 ]] && continue

    FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "${PID}"; then FAILED=$((FAILED + 1)); fi
    done
    echo "  stage ${STAGE} done (${FAILED} worker errors, see logs)"
    echo ""
done

echo "All stages dispatched. Check progress:"
echo "  python -m recon.cli.preprocess_run status"