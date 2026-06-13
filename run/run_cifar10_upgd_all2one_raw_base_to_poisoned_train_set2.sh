#!/usr/bin/env bash
# Run CIFAR-10 UPGD all-to-one experiments for the three baseline models.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set2}"
export DATASETS="cifar10"
export CIFAR10_MODELS="${CIFAR10_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
export RUN_NAME="${RUN_NAME:-run_cifar10_upgd_all2one_raw_base_to_poisoned_train_set2}"

exec bash "${SCRIPT_DIR}/run_upgd_all2one_raw_base_to_poisoned_train_set2.sh" "$@"
