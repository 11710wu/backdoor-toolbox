#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-uniform}"
export RUN_NAME="${RUN_NAME:-run_cifar10_small_cnn_noise_sig_upgd_clean_label_uniform}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN uniform input-noise clean-label SIG/UPGD experiment}"

echo "output: poisoned_train_set/cifar10/{SIG,upgd}_*_mode=clean_noise=uniform_level={0.030,0.060,0.100}_poison_seed=2333_arch=SmallCNN_cifar10"

exec bash "$SCRIPT_DIR/run_cifar10_small_cnn_noise_sig_upgd_clean_label.sh" "$@"
