#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-run_tiny_imagenet_densenet121_arch_04_belt_upgd}"
export RUN_TITLE="${RUN_TITLE:-Tiny-ImageNet DenseNet121 architecture part 04: BELT + UPGD}"
export ATTACK_LIST="${ATTACK_LIST:-belt upgd}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-0}"
export DEVICES="${DEVICES:-3}"

exec bash "${SCRIPT_DIR}/run_tiny_imagenet_densenet121_arch_common.sh" "$@"
