#!/usr/bin/env bash

# Tiny-ImageNet ResNet50 architecture experiment, shard 2/3.
# Runs WaNet, Adaptive-Patch and Adaptive-Blend.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="tiny_imagenet"
export RUN_NAME="${RUN_NAME:-run_tiny_imagenet_resnet50_arch_02_wanet_adaptive}"
export RUN_TITLE="${RUN_TITLE:-Tiny-ImageNet ResNet50 architecture experiment 02: WaNet/Adaptive attacks}"
export ATTACK_LIST="${ATTACK_LIST:-WaNet adaptive_patch adaptive_blend}"
export POISON_RATES="${POISON_RATES:-0.001 0.005}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-0}"
export PREPARE_UPGD_RAW_BASE="${PREPARE_UPGD_RAW_BASE:-0}"

bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
