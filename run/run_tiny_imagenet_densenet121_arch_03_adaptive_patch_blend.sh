#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-run_tiny_imagenet_densenet121_arch_03_adaptive_patch_blend}"
export RUN_TITLE="${RUN_TITLE:-Tiny-ImageNet DenseNet121 architecture part 03: adaptive_patch + adaptive_blend}"
export ATTACK_LIST="${ATTACK_LIST:-adaptive_patch adaptive_blend}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-0}"
export DEVICES="${DEVICES:-2}"

exec bash "${SCRIPT_DIR}/run_tiny_imagenet_densenet121_arch_common.sh" "$@"
