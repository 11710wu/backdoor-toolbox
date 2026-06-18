#!/usr/bin/env bash
# Backfill CIFAR-10 and MNIST-M UPGD all-to-one experiments.
#
# Defaults:
# - datasets: cifar10 mnistm
# - models: resnet18 mobilenetv2 vgg19_bn
# - poison rates: cifar10/mnistm defaults from the generic UPGD script
# - eps values: 4 6 8 10 12 16 20 24
# - output root: poisoned_train_set2
#
# Useful overrides:
#   DEVICES=1 bash run/backfill_cifar10_mnistm_upgd_all2one.sh
#   DRY_RUN=1 bash run/backfill_cifar10_mnistm_upgd_all2one.sh
#   RUN_DEFENSES=0 bash run/backfill_cifar10_mnistm_upgd_all2one.sh
#   CIFAR10_MODELS="resnet18" MNISTM_MODELS="resnet18" bash run/backfill_cifar10_mnistm_upgd_all2one.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERIC_SCRIPT="${SCRIPT_DIR}/run_upgd_all2one_raw_base_to_poisoned_train_set2.sh"

if [ ! -f "$GENERIC_SCRIPT" ]; then
  echo "Missing generic UPGD all2one script: ${GENERIC_SCRIPT}" >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-backfill_cifar10_mnistm_upgd_all2one}"
export DATASETS="${DATASETS:-cifar10 mnistm}"
export CIFAR10_MODELS="${CIFAR10_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
export MNISTM_MODELS="${MNISTM_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set2}"
export RUN_QWEN_TRANSFER=0

echo "Backfill UPGD all2one for CIFAR-10 and MNIST-M"
echo "DATASETS=${DATASETS}"
echo "CIFAR10_MODELS=${CIFAR10_MODELS}"
echo "MNISTM_MODELS=${MNISTM_MODELS}"
echo "POISONED_TRAIN_SET_ROOT=${POISONED_TRAIN_SET_ROOT}"
echo "Generic script: ${GENERIC_SCRIPT}"

exec bash "$GENERIC_SCRIPT" "$@"
