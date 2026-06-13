#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-gaussian}"
export RUN_NAME="${RUN_NAME:-run_cifar10_resnet18_noise_gaussian_eight_methods}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 ResNet18 gaussian input-noise eight-method experiment}"

echo "output: poisoned_train_set/cifar10/*_noise=gaussian_level={0.030,0.060,0.100}_poison_seed=2333_arch=ResNet18_cifar10"

exec bash "$SCRIPT_DIR/run_cifar10_resnet18_noise_eight_methods.sh" "$@"
