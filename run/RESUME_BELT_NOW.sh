#!/usr/bin/env bash
#
# BELT-only entrypoint for poisoned_train_set4 (train + eval).
# Does NOT run non-BELT missing-artifact backfill — use RESUME_NOW.sh for that.
#
# Intended for a second machine. SKIP_EXISTING=1: never deletes; skips finished
# create/train/source/transfer/qwen/defense artifacts. Incomplete trains (no
# final .pt / train_results) will restart from scratch.
#
# Remaining heavy work (as of interruption):
#   train: Tiny ResNet34 belt 0.010 × {0.1,0.2,0.3}
#          Tiny ResNet50 belt 0.010 × {0.1,0.2,0.3}
#   eval : all 24 BELT configs (CIFAR Micro/Small/ResNet50 + Tiny R34/R50)
#          source + ImageNetV2/STL10 transfer + Qwen (tiny) + 4 defenses
#
# Usage (run from repo root, where nvidia-smi works):
#   bash run/RESUME_BELT_NOW.sh
#   PHASE=train bash run/RESUME_BELT_NOW.sh
#   PHASE=eval  bash run/RESUME_BELT_NOW.sh
#   DRY_RUN=1 bash run/RESUME_BELT_NOW.sh
#   PARALLEL=1 GPU_IDS="0 1" bash run/RESUME_BELT_NOW.sh
#
# Sync to the other machine before running:
#   - this repo
#   - poisoned_train_set4/  (poison dirs with imgs/labels + any finished .pt)
#   - data/Tiny-imagenet, data/cifar*, STL10, imagenetv2-*, tiny-target-domain-qwen-*
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${MAIN_LOG:-${LOG_DIR}/resume_set4_belt_only_${STAMP}.log}"

# Default: train + eval (skip create). Override with PHASE=train|eval|all|...
PHASE_IN="${PHASE:-train_eval}"
case "$PHASE_IN" in
  all|train|eval|evaluate|train_eval|source|transfer|qwen|defense|defenses|create)
    ;;
  *)
    echo "Unsupported PHASE=${PHASE_IN}. Use train|eval|train_eval|all|source|transfer|qwen|defense." >&2
    exit 2
    ;;
esac

export PHASE="$PHASE_IN"
export DRY_RUN="${DRY_RUN:-0}"
export PARALLEL="${PARALLEL:-0}"
export GPU_IDS="${GPU_IDS:-${DEVICES:-0}}"
export MAX_JOBS="${MAX_JOBS:-}"
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export RUN_QWEN="${RUN_QWEN:-1}"
export DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"

# Full BELT grid (same as corrected set4 schedule).
export RUN_CIFAR_MICROCNN="${RUN_CIFAR_MICROCNN:-1}"
export RUN_CIFAR_SMALLCNN="${RUN_CIFAR_SMALLCNN:-1}"
export RUN_CIFAR_RESNET50="${RUN_CIFAR_RESNET50:-1}"
export RUN_TINY_RESNET34="${RUN_TINY_RESNET34:-1}"
export RUN_TINY_RESNET50="${RUN_TINY_RESNET50:-1}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Checking GPU..."
  nvidia-smi | head -20 || echo "WARN: nvidia-smi failed; continuing anyway."
  echo
else
  echo "WARN: nvidia-smi not found; continuing anyway."
  echo
fi

echo "============================================================"
echo "Resume set4 BELT only (train / eval)"
echo "============================================================"
echo "log                   : ${MAIN_LOG}"
echo "phase                 : ${PHASE}"
echo "gpu ids               : ${GPU_IDS}"
echo "parallel              : ${PARALLEL}"
echo "skip existing         : ${SKIP_EXISTING}"
echo "CIFAR MicroCNN        : ${RUN_CIFAR_MICROCNN}"
echo "CIFAR SmallCNN        : ${RUN_CIFAR_SMALLCNN}"
echo "CIFAR ResNet50        : ${RUN_CIFAR_RESNET50}"
echo "Tiny ResNet34         : ${RUN_TINY_RESNET34}"
echo "Tiny ResNet50         : ${RUN_TINY_RESNET50}"
echo "run qwen              : ${RUN_QWEN}"
echo "defenses              : ${DEFENSE_LIST}"
echo "result root           : ${POISONED_TRAIN_SET_ROOT}"
echo "dry run               : ${DRY_RUN}"
echo "============================================================"
echo
echo "PHASE cheat-sheet:"
echo "  train      -> BELT train only"
echo "  eval       -> source + transfer + qwen + 4 defenses"
echo "  train_eval -> train + eval (default; no create)"
echo "  all        -> create + train + eval"
echo "============================================================"

set -o pipefail
bash run/rerun_set4_belt_final_checkpoint_full.sh 2>&1 | tee "$MAIN_LOG"
