#!/usr/bin/env bash
# Run Tiny-ImageNet UPGD all-to-one experiments for ResNet18.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set2}"
export DATASETS="${DATASETS:-tiny_imagenet}"
export TINY_IMAGENET_MODELS="${TINY_IMAGENET_MODELS:-resnet18}"
export RUN_NAME="${RUN_NAME:-run_resnet18_upgd_all2one_raw_base_to_poisoned_train_set2}"

exec bash "${SCRIPT_DIR}/run_upgd_all2one_raw_base_to_poisoned_train_set2.sh" "$@"
