#!/usr/bin/env bash

# ResNet50 architecture experiment, shard 8/8: UPGD
# CIFAR-10 (6) + Tiny-ImageNet (6) = 12 configs.
#
# UPGD 造毒只依赖 raw-input clean base（upgd_raw_base_*.pt）。
# 默认：两边都开启 PREPARE_UPGD_RAW_BASE；已有则跳过，不强制重训。
# normalized clean（none/*.pt）默认不准备；需要时再开 PREPARE_CLEAN_*=1。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACK="upgd"
DEVICES="${DEVICES:-0}"
export DEVICES

run_one() {
  local dataset="$1"
  local rates="$2"
  local prep_clean="$3"
  local prep_upgd="$4"
  export DATASET="${dataset}"
  export ATTACK_LIST="${ATTACK}"
  export POISON_RATES="${rates}"
  export PREPARE_CLEAN="${prep_clean}"
  export PREPARE_UPGD_RAW_BASE="${prep_upgd}"
  export FORCE_RETRAIN_CLEAN=0
  export FORCE_RETRAIN_UPGD_RAW_BASE=0
  export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_08_upgd}_${dataset}"
  export RUN_TITLE="ResNet50 architecture 08/8 UPGD [${dataset}]"
  bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
}

run_one cifar10 "0.005 0.010" \
  "${PREPARE_CLEAN_CIFAR:-0}" "${PREPARE_UPGD_CIFAR:-1}"

run_one tiny_imagenet "0.001 0.005" \
  "${PREPARE_CLEAN_TINY:-0}" "${PREPARE_UPGD_TINY:-1}"
