#!/usr/bin/env bash

# ResNet50 architecture experiment: SIG + WaNet
# Tiny-ImageNet only, poison_rate=0.005 → 6 configs.
# (合并原 03_sig + 04_wanet)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICES="${DEVICES:-0}"
export DEVICES

export DATASET="tiny_imagenet"
export ATTACK_LIST="${ATTACK_LIST:-SIG WaNet}"
export POISON_RATES="${POISON_RATES:-0.005}"
export PREPARE_CLEAN=0
export PREPARE_UPGD_RAW_BASE=0
export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_03_sig_wanet_tiny}"
export RUN_TITLE="ResNet50 architecture SIG+WaNet [tiny_imagenet]"
bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
