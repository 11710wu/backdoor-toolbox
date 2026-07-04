#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-run_tiny_imagenet_densenet121_arch_02_sig_wanet}"
export RUN_TITLE="${RUN_TITLE:-Tiny-ImageNet DenseNet121 architecture part 02: SIG + WaNet}"
export ATTACK_LIST="${ATTACK_LIST:-SIG WaNet}"
export PREPARE_CLEAN="${PREPARE_CLEAN:-0}"
export DEVICES="${DEVICES:-1}"

exec bash "${SCRIPT_DIR}/run_tiny_imagenet_densenet121_arch_common.sh" "$@"
