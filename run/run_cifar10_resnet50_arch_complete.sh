#!/usr/bin/env bash

# CIFAR10 ResNet50 architecture experiment (legacy single script).
# Prefer the 8-way split: run/run_resnet50_arch_01_basic.sh ... _08_upgd.sh
# or: bash run/run_resnet50_arch_all_8.sh
# Grid: 8 attacks x 2 poison rates x 3 strengths, matching the SmallCNN set4 grid.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATASET="cifar10"
export RUN_NAME="${RUN_NAME:-run_cifar10_resnet50_arch_complete}"
export RUN_TITLE="${RUN_TITLE:-CIFAR10 ResNet50 architecture experiment}"
export ATTACK_LIST="${ATTACK_LIST:-basic blend SIG WaNet adaptive_patch adaptive_blend belt upgd}"
export POISON_RATES="${POISON_RATES:-0.005 0.010}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-1}"
export PREPARE_UPGD_RAW_BASE="${PREPARE_UPGD_RAW_BASE:-1}"

bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
