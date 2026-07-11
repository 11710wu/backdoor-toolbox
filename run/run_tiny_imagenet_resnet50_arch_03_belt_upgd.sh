#!/usr/bin/env bash

# Tiny-ImageNet ResNet50 architecture experiment, legacy shard 3/3.
# Prefer the 8-way split: run/run_resnet50_arch_01_basic.sh ... _08_upgd.sh
# Runs BELT and UPGD. This script prepares the raw-input clean model needed by UPGD.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="tiny_imagenet"
export RUN_NAME="${RUN_NAME:-run_tiny_imagenet_resnet50_arch_03_belt_upgd}"
export RUN_TITLE="${RUN_TITLE:-Tiny-ImageNet ResNet50 architecture experiment 03: BELT/UPGD}"
export ATTACK_LIST="${ATTACK_LIST:-belt upgd}"
export POISON_RATES="${POISON_RATES:-0.001 0.005}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-0}"
export PREPARE_UPGD_RAW_BASE="${PREPARE_UPGD_RAW_BASE:-1}"

bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
