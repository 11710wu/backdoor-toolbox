#!/usr/bin/env bash

# ResNet50 architecture experiment: BELT + UPGD
# Tiny-ImageNet only, poison_rate=0.005 → 6 configs.
# (合并原 07_belt + 08_upgd)
#
# Clean / UPGD raw-base 已训好，默认不再 prepare。
# 如需重训：PREPARE_UPGD_TINY=1 或 PREPARE_CLEAN_TINY=1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICES="${DEVICES:-0}"
export DEVICES

export DATASET="tiny_imagenet"
export ATTACK_LIST="${ATTACK_LIST:-belt upgd}"
export POISON_RATES="${POISON_RATES:-0.005}"
export PREPARE_CLEAN="${PREPARE_CLEAN_TINY:-0}"
export PREPARE_UPGD_RAW_BASE="${PREPARE_UPGD_TINY:-0}"
export FORCE_RETRAIN_CLEAN=0
export FORCE_RETRAIN_UPGD_RAW_BASE=0
export RUN_NAME="${RUN_NAME_PREFIX:-run_resnet50_arch_05_belt_upgd_tiny}"
export RUN_TITLE="ResNet50 architecture BELT+UPGD [tiny_imagenet]"
bash "${SCRIPT_DIR}/run_resnet50_arch_common.sh"
