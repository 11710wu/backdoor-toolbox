#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-speckle}"
export SKIP_UPGD_PREP="${SKIP_UPGD_PREP:-1}"
export WAIT_FOR_UPGD_BASE="${WAIT_FOR_UPGD_BASE:-1}"
export RUN_NAME="${RUN_NAME:-run_cifar10_small_cnn_noise_speckle_subset}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN speckle input-noise subset experiment}"

echo "SKIP_UPGD_PREP: ${SKIP_UPGD_PREP}"
echo "WAIT_FOR_UPGD_BASE: ${WAIT_FOR_UPGD_BASE}"

exec bash "$SCRIPT_DIR/run_cifar10_small_cnn_noise_difficulty_subset.sh" "$@"
