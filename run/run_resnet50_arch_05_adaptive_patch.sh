#!/usr/bin/env bash

# ResNet50 architecture experiment, shard 5/8: Adaptive-Patch
# CIFAR-10 (6) + Tiny-ImageNet (6) = 12 configs.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACK="adaptive_patch"
DEVICES="${DEVICES:-0}"
export DEVICES

run_one() {
  local dataset="$1"
  local rates="$2"
  export DATASET="${dataset}"
  export ATTACK_LIST="${ATTACK}"
  export POISON_RATES="${rates}"
  export PREPARE_CLEAN=0
  export PREPARE_UPGD_RAW_BASE=0
  export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_05_adaptive_patch}_${dataset}"
  export RUN_TITLE="ResNet50 architecture 05/8 Adaptive-Patch [${dataset}]"
  bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
}

run_one cifar10 "0.005 0.010"
run_one tiny_imagenet "0.001 0.005"
