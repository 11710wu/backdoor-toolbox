#!/usr/bin/env bash

# ResNet50 architecture experiment: Basic + Blend
# Tiny-ImageNet only, poison_rate=0.005 → 6 configs.
# (合并原 01_basic + 02_blend)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICES="${DEVICES:-0}"
export DEVICES

export DATASET="tiny_imagenet"
export ATTACK_LIST="${ATTACK_LIST:-basic blend}"
export POISON_RATES="${POISON_RATES:-0.005}"
export PREPARE_CLEAN=0
export PREPARE_UPGD_RAW_BASE=0
export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_01_basic_blend_tiny}"
export RUN_TITLE="ResNet50 architecture Basic+Blend [tiny_imagenet]"
bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
