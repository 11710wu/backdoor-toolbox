#!/usr/bin/env bash
# Backfill AC / SS / SPECTRE for backdoor-toolbox-new1 CIFAR-10 only.
#
# Roots: poisoned_train_set, poisoned_train_set2, poisoned_train_set3
# Excludes: poisoned_train_set4, poisoned_train_set5
#
# Usage:
#   ./run/backfill_new1_cifar10.sh --dry-run
#   ./run/backfill_new1_cifar10.sh --devices 0
#   DEVICES=0 NUM_SHARDS=4 SHARD_INDEX=0 ./run/backfill_new1_cifar10.sh
#
# Env:
#   DEVICES        GPU id(s), default 0
#   NUM_SHARDS     parallel shards, default 1
#   SHARD_INDEX    this shard index [0, NUM_SHARDS), default 0
#   SPECTRE_JOBS   Julia concurrency for SPECTRE, default 4
#   PYTHON         python binary, default backtool env
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/workspace/tools/julia-1.7.2/bin:${PATH:-}"
PYTHON="${PYTHON:-/root/anaconda3/envs/backtool/bin/python}"
DEVICES="${DEVICES:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SPECTRE_JOBS="${SPECTRE_JOBS:-4}"

cd "${REPO}"
exec "${PYTHON}" run/backfill_cleanser_results.py \
  --roots poisoned_train_set poisoned_train_set2 poisoned_train_set3 \
  --datasets cifar10 \
  --devices "${DEVICES}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  --spectre-jobs "${SPECTRE_JOBS}" \
  "$@"
