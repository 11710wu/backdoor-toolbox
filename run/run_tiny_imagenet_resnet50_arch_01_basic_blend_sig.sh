#!/usr/bin/env bash

# Tiny-ImageNet ResNet50 architecture experiment, shard 1/3.
# Runs Basic, Blend and SIG. This script prepares the normalized clean model.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="tiny_imagenet"
export RUN_NAME="${RUN_NAME:-run_tiny_imagenet_resnet50_arch_01_basic_blend_sig}"
export RUN_TITLE="${RUN_TITLE:-Tiny-ImageNet ResNet50 architecture experiment 01: Basic/Blend/SIG}"
export ATTACK_LIST="${ATTACK_LIST:-basic blend SIG}"
export POISON_RATES="${POISON_RATES:-0.001 0.005}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-1}"
export PREPARE_UPGD_RAW_BASE="${PREPARE_UPGD_RAW_BASE:-0}"

bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
