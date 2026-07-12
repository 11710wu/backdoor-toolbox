#!/usr/bin/env bash

# ResNet50 architecture experiment: Adaptive-Patch + Adaptive-Blend
# CIFAR-10 (12) + Tiny-ImageNet (12) = 24 configs.
# (合并原 05_adaptive_patch + 06_adaptive_blend)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICES="${DEVICES:-0}"
export DEVICES

run_one() {
  local dataset="$1"
  local rates="$2"
  export DATASET="${dataset}"
  export ATTACK_LIST="${ATTACK_LIST:-adaptive_patch adaptive_blend}"
  export POISON_RATES="${rates}"
  export PREPARE_CLEAN=0
  export PREPARE_UPGD_RAW_BASE=0
  export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_04_adaptive}_${dataset}"
  export RUN_TITLE="ResNet50 architecture Adaptive-Patch+Blend [${dataset}]"
  bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
}

run_one cifar10 "0.005 0.010"
run_one tiny_imagenet "0.001 0.005"
