#!/usr/bin/env bash

# ResNet50 architecture experiment, shard 8/8: UPGD
# CIFAR-10 (6) + Tiny-ImageNet (6) = 12 configs.
# Prepares UPGD raw-input clean base for both datasets (skipped if already present).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACK="upgd"
DEVICES="${DEVICES:-0}"
export DEVICES

run_one() {
  local dataset="$1"
  local rates="$2"
  local prep_upgd="$3"
  export DATASET="${dataset}"
  export ATTACK_LIST="${ATTACK}"
  export POISON_RATES="${rates}"
  export PREPARE_CLEAN=0
  export PREPARE_UPGD_RAW_BASE="${prep_upgd}"
  export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_08_upgd}_${dataset}"
  export RUN_TITLE="ResNet50 architecture 08/8 UPGD [${dataset}]"
  bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
}

run_one cifar10 "0.005 0.010" "${PREPARE_UPGD_CIFAR:-1}"
run_one tiny_imagenet "0.001 0.005" "${PREPARE_UPGD_TINY:-1}"
