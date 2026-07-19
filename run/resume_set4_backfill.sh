#!/usr/bin/env bash

# Resume poisoned_train_set4 backfill on a single GPU (includes ResNet50).
#
# Behavior:
#   - SKIP_EXISTING=1: never delete; skip finished create/train/test/defense artifacts
#   - Includes CIFAR/Tiny ResNet50 BELT configs and non-BELT ResNet50 missing jobs
#   - Interrupted incomplete trains (no final .pt / train_results) will restart
#
# Examples:
#   DRY_RUN=1 bash run/resume_set4_backfill.sh
#   bash run/resume_set4_backfill.sh
#   PARALLEL=1 GPU_IDS="0 1" bash run/resume_set4_backfill.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${MAIN_LOG:-${LOG_DIR}/resume_set4_backfill_${STAMP}.log}"

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

# Include ResNet50 (normal full resume).
export RUN_CIFAR_RESNET50="${RUN_CIFAR_RESNET50:-1}"
export RUN_TINY_RESNET50="${RUN_TINY_RESNET50:-1}"
export RUN_CIFAR_MICROCNN="${RUN_CIFAR_MICROCNN:-1}"
export RUN_CIFAR_SMALLCNN="${RUN_CIFAR_SMALLCNN:-1}"
export RUN_TINY_RESNET34="${RUN_TINY_RESNET34:-1}"
export ARCH_EXCLUDE="${ARCH_EXCLUDE:-}"
export BELT_SKIP_EXISTING="${BELT_SKIP_EXISTING:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "============================================================"
echo "Resume set4 backfill (with ResNet50)"
echo "============================================================"
echo "log                   : ${MAIN_LOG}"
echo "phase                 : ${PHASE}"
echo "gpu ids               : ${GPU_IDS}"
echo "parallel              : ${PARALLEL}"
echo "run BELT              : ${RUN_BELT}"
echo "run existing missing  : ${RUN_EXISTING_MISSING}"
echo "CIFAR ResNet50 BELT   : ${RUN_CIFAR_RESNET50}"
echo "Tiny ResNet50 BELT    : ${RUN_TINY_RESNET50}"
echo "arch exclude          : ${ARCH_EXCLUDE:-"(none)"}"
echo "skip existing         : ${SKIP_EXISTING}"
echo "dry run               : ${DRY_RUN}"
echo "============================================================"

set -o pipefail
bash run/rerun_set4_complete_missing_and_belt.sh 2>&1 | tee "$MAIN_LOG"
