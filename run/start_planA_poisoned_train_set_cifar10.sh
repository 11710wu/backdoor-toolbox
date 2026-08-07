#!/usr/bin/env bash
# Plan A: recreate poison data -> AC/SS/SPECTRE -> delete data
# Scope: poisoned_train_set / cifar10 only
set -euo pipefail
cd /workspace/backdoor-toolbox-new1
export PATH="/workspace/tools/julia-1.7.2/bin:${PATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
PYTHON="${PYTHON:-/root/anaconda3/envs/backtool/bin/python}"
DEVICES="${DEVICES:-0}"
LOG_DIR="logs/recreate_and_backfill"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/planA_poisoned_train_set_cifar10_${STAMP}.log"
echo "logging to ${LOG}"
exec "${PYTHON}" -u run/recreate_and_backfill_cleanser.py \
  --roots poisoned_train_set \
  --datasets cifar10 \
  --delete-data-after \
  --devices "${DEVICES}" \
  --spectre-jobs "${SPECTRE_JOBS:-4}" \
  "$@" 2>&1 | tee -a "${LOG}"
