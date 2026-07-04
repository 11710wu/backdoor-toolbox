#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-run_tiny_imagenet_densenet121_arch_01_basic_blend}"
export RUN_TITLE="${RUN_TITLE:-Tiny-ImageNet DenseNet121 architecture part 01: basic + blend}"
export ATTACK_LIST="${ATTACK_LIST:-basic blend}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-1}"
export DEVICES="${DEVICES:-0}"

exec bash "${SCRIPT_DIR}/run_tiny_imagenet_densenet121_arch_common.sh" "$@"
