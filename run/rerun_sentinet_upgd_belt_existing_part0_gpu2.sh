#!/usr/bin/env bash

# Worker 0/2 for rerunning SentiNet on existing noisy CIFAR-10 UPGD/BELT rows.
# Defaults to GPU 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/backtool/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python"
fi

DEVICES="${DEVICES:-2}"

exec "${PYTHON_BIN}" run/rerun_sentinet_upgd_belt_existing.py \
  --roots poisoned_train_set/cifar10 \
  --attacks upgd belt \
  --num-shards 2 \
  --shard-index 0 \
  --devices "${DEVICES}" \
  "$@"
