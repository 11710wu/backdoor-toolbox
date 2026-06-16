#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-speckle}"
export RUN_NAME="${RUN_NAME:-run_cifar10_small_cnn_noise_speckle_subset}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN speckle input-noise subset experiment}"

exec bash "$SCRIPT_DIR/run_cifar10_small_cnn_noise_difficulty_subset.sh" "$@"
