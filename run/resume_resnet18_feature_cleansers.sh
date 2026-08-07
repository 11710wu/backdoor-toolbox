#!/usr/bin/env bash
# Resume AC / SS / SPECTRE for poisoned_train_set CIFAR-10 ResNet18.
# Flow per config: recreate data (if missing) -> AC/SS/SPECTRE -> delete data
# Skips configs that already have all three *_cleanser_results.json
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

export PATH="/workspace/tools/julia-1.7.2/bin:${PATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

PYTHON="${PYTHON:-/root/anaconda3/envs/backtool/bin/python}"
DEVICES="${DEVICES:-0}"
SPECTRE_JOBS="${SPECTRE_JOBS:-4}"
LOG_DIR="${REPO}/logs/recreate_and_backfill"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/resume_resnet18_feature_${STAMP}.log"

echo "[info] logging to ${LOG}"
echo "[info] devices=${DEVICES} spectre_jobs=${SPECTRE_JOBS}"

exec "${PYTHON}" -u run/recreate_and_backfill_cleanser.py \
  --roots poisoned_train_set \
  --datasets cifar10 \
  --arch ResNet18 \
  --delete-data-after \
  --devices "${DEVICES}" \
  --spectre-jobs "${SPECTRE_JOBS}" \
  "$@" 2>&1 | tee -a "${LOG}"
