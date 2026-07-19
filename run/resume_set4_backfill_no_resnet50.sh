#!/usr/bin/env bash

# Resume poisoned_train_set4 backfill after a pause.
#
# Defaults for this resume pass:
#   - Skip all ResNet50 BELT configs (CIFAR + Tiny)
#   - Skip non-BELT missing jobs whose arch name contains ResNet50
#   - SKIP_EXISTING=1 so finished artifacts are not redone
#   - Interrupted BELT train (no final .pt / train_results) will restart
#
# Examples:
#   # Preview only
#   DRY_RUN=1 bash run/resume_set4_backfill_no_resnet50.sh
#
#   # Single-GPU resume (default)
#   bash run/resume_set4_backfill_no_resnet50.sh
#
#   # Multi-GPU resume
#   PARALLEL=1 GPU_IDS="0 1" bash run/resume_set4_backfill_no_resnet50.sh
#
#   # Only BELT, or only non-BELT missing
#   RUN_EXISTING_MISSING=0 bash run/resume_set4_backfill_no_resnet50.sh
#   RUN_BELT=0 bash run/resume_set4_backfill_no_resnet50.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${MAIN_LOG:-${LOG_DIR}/resume_set4_backfill_no_resnet50_${STAMP}.log}"

export PHASE="${PHASE:-all}"
export DRY_RUN="${DRY_RUN:-0}"
export PARALLEL="${PARALLEL:-0}"
export GPU_IDS="${GPU_IDS:-${DEVICES:-0}}"
export MAX_JOBS="${MAX_JOBS:-}"
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
export RUN_BELT="${RUN_BELT:-1}"
export RUN_EXISTING_MISSING="${RUN_EXISTING_MISSING:-1}"
export RUN_QWEN="${RUN_QWEN:-1}"
export DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"

# Skip ResNet50 everywhere in this resume pass.
export RUN_CIFAR_RESNET50="${RUN_CIFAR_RESNET50:-0}"
export RUN_TINY_RESNET50="${RUN_TINY_RESNET50:-0}"
export RUN_CIFAR_MICROCNN="${RUN_CIFAR_MICROCNN:-1}"
export RUN_CIFAR_SMALLCNN="${RUN_CIFAR_SMALLCNN:-1}"
export RUN_TINY_RESNET34="${RUN_TINY_RESNET34:-1}"
export ARCH_EXCLUDE="${ARCH_EXCLUDE:-ResNet50}"
export BELT_SKIP_EXISTING="${BELT_SKIP_EXISTING:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "============================================================"
echo "Resume set4 backfill (no ResNet50)"
echo "============================================================"
echo "log                   : ${MAIN_LOG}"
echo "phase                 : ${PHASE}"
echo "gpu ids               : ${GPU_IDS}"
echo "parallel              : ${PARALLEL}"
echo "run BELT              : ${RUN_BELT}"
echo "run existing missing  : ${RUN_EXISTING_MISSING}"
echo "CIFAR ResNet50 BELT   : ${RUN_CIFAR_RESNET50}"
echo "Tiny ResNet50 BELT    : ${RUN_TINY_RESNET50}"
echo "arch exclude          : ${ARCH_EXCLUDE}"
echo "skip existing         : ${SKIP_EXISTING}"
echo "dry run               : ${DRY_RUN}"
echo "============================================================"
echo
echo "Interrupted note:"
echo "  Tiny ResNet34 BELT rate=0.010 alpha=0.100 was mid-train when paused."
echo "  Only *_best.pt exists; final model/train_results missing -> will retrain."
echo "============================================================"

set -o pipefail
bash run/rerun_set4_complete_missing_and_belt.sh 2>&1 | tee "$MAIN_LOG"
