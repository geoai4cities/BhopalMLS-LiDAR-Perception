#!/bin/bash
# Phase 4 — CORAL DA orchestrator. Runs stages 1-3 for both archs end-to-end.
#
# Usage:
#   ./run_all.sh           # full run (~5-6 hr CPU)
#   ./run_all.sh --smoke   # smoke test (5 scenes per split, ~1-2 min)
#   ./run_all.sh --arch concerto       # run only Concerto
#   ./run_all.sh --skip-extract        # skip stage 1 (use cached features)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PY="$REPO_ROOT/bhopalmls_venv/bin/python"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

ARCHS=("concerto" "utonia")
SMOKE_ARG=""
SKIP_EXTRACT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)         SMOKE_ARG="--smoke 5"; shift ;;
        --arch)          ARCHS=("$2"); shift 2 ;;
        --skip-extract)  SKIP_EXTRACT=1; shift ;;
        --help|-h)
            sed -n '2,7p' "${BASH_SOURCE[0]}" ; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# Pre-flight: venv + Pointcept dir
[ -x "$VENV_PY" ] || { echo "ERROR: venv python not found: $VENV_PY"; exit 1; }
[ -d "$REPO_ROOT/Pointcept" ] || {
    echo "ERROR: Pointcept not found at $REPO_ROOT/Pointcept"; exit 1; }

export PYTHONPATH="$REPO_ROOT/Pointcept:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ts() { date +%Y%m%d_%H%M%S; }

for ARCH in "${ARCHS[@]}"; do
    echo
    echo "=================================================================="
    echo "  CORAL DA — $ARCH"
    echo "=================================================================="

    if [ "$SKIP_EXTRACT" -eq 0 ]; then
        for SPLIT in bm_train pl_train pl_val; do
            LOG="$LOG_DIR/extract_${ARCH}_${SPLIT}_$(ts).log"
            echo
            echo "[stage 1] $ARCH/$SPLIT  → $LOG"
            "$VENV_PY" "$SCRIPT_DIR/extract_features.py" \
                --arch "$ARCH" --split "$SPLIT" --max-pts 2000 $SMOKE_ARG \
                2>&1 | tee "$LOG"
        done
    else
        echo "  [stage 1] skipped (--skip-extract)"
    fi

    LOG="$LOG_DIR/coral_train_${ARCH}_$(ts).log"
    echo
    echo "[stage 2] $ARCH coral-align + train  → $LOG"
    "$VENV_PY" "$SCRIPT_DIR/coral_align_and_train.py" --arch "$ARCH" \
        2>&1 | tee "$LOG"

    LOG="$LOG_DIR/eval_${ARCH}_$(ts).log"
    echo
    echo "[stage 3] $ARCH eval  → $LOG"
    "$VENV_PY" "$SCRIPT_DIR/eval_coral.py" --arch "$ARCH" \
        2>&1 | tee "$LOG"
done

echo
echo "=================================================================="
echo "  ALL CORAL stages done. See $SCRIPT_DIR/results/coral_da_supp.tex"
echo "=================================================================="
