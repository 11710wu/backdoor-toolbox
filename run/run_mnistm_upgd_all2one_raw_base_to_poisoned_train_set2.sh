#!/usr/bin/env bash
# Run MNIST-M UPGD all-to-one experiments for the three baseline models.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set2}"
export DATASETS="mnistm"
export MNISTM_MODELS="${MNISTM_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
export RUN_NAME="${RUN_NAME:-run_mnistm_upgd_all2one_raw_base_to_poisoned_train_set2}"

exec bash "${SCRIPT_DIR}/run_upgd_all2one_raw_base_to_poisoned_train_set2.sh" "$@"
