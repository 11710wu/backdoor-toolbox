#!/usr/bin/env bash

# ResNet50 architecture experiment, shard 1/8: Basic
# CIFAR-10 (6) + Tiny-ImageNet (6) = 12 configs.
#
# Clean model defaults:
#   CIFAR-10: prepare and train (FORCE_RETRAIN_CLEAN=1, 不跳过已有权重)
#   Tiny:     skip clean prep by default (已有 ResNet50 clean)
# Overrides:
#   PREPARE_CLEAN_CIFAR=0 / PREPARE_CLEAN_TINY=1
#   FORCE_RETRAIN_CLEAN_CIFAR=0  # 若 CIFAR clean 已存在则跳过重训

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACK="basic"
DEVICES="${DEVICES:-0}"
export DEVICES

run_one() {
  local dataset="$1"
  local rates="$2"
  local prep_clean="$3"
  local prep_upgd="$4"
  local force_retrain_clean="$5"
  export DATASET="${dataset}"
  export ATTACK_LIST="${ATTACK}"
  export POISON_RATES="${rates}"
  export PREPARE_CLEAN="${prep_clean}"
  export PREPARE_UPGD_RAW_BASE="${prep_upgd}"
  export FORCE_RETRAIN_CLEAN="${force_retrain_clean}"
  export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_01_basic}_${dataset}"
  export RUN_TITLE="ResNet50 architecture 01/8 Basic [${dataset}]"
  bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
}

# CIFAR: 默认准备并强制重训 clean；Tiny: 默认跳过 clean
run_one cifar10 "0.005 0.010" "${PREPARE_CLEAN_CIFAR:-1}" 0 "${FORCE_RETRAIN_CLEAN_CIFAR:-1}"
run_one tiny_imagenet "0.001 0.005" "${PREPARE_CLEAN_TINY:-0}" 0 0
