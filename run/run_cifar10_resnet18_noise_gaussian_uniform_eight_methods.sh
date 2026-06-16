#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-gaussian uniform}"
export SIG_UPGD_LABEL_MODE="${SIG_UPGD_LABEL_MODE:-clean}"
export SKIP_UPGD_PREP="${SKIP_UPGD_PREP:-0}"
export RUN_NAME="${RUN_NAME:-run_cifar10_resnet18_noise_gaussian_uniform_eight_methods}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 ResNet18 gaussian+uniform input-noise eight-method experiment}"

echo "output: poisoned_train_set/cifar10/*_noise={gaussian,uniform}_level={0.030,0.060,0.100}_poison_seed=2333_arch=ResNet18_cifar10"
echo "SIG/UPGD label_mode: ${SIG_UPGD_LABEL_MODE}"
echo "SKIP_UPGD_PREP: ${SKIP_UPGD_PREP}"

exec bash "$SCRIPT_DIR/run_cifar10_resnet18_noise_eight_methods.sh" "$@"
