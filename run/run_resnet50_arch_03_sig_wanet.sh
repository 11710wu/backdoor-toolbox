#!/usr/bin/env bash

# ResNet50 architecture experiment: SIG + WaNet
# CIFAR-10 (12) + Tiny-ImageNet (12) = 24 configs.
# (合并原 03_sig + 04_wanet)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICES="${DEVICES:-0}"
export DEVICES

run_one() {
  local dataset="$1"
  local rates="$2"
  export DATASET="${dataset}"
  export ATTACK_LIST="${ATTACK_LIST:-SIG WaNet}"
  export POISON_RATES="${rates}"
  export PREPARE_CLEAN=0
  export PREPARE_UPGD_RAW_BASE=0
  export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_03_sig_wanet}_${dataset}"
  export RUN_TITLE="ResNet50 architecture SIG+WaNet [${dataset}]"
  bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
}

run_one cifar10 "0.005 0.010"
run_one tiny_imagenet "0.001 0.005"
