#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-small_cnn}"
export NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-salt_pepper speckle}"
export RUN_NAME="${RUN_NAME:-backfill_cifar10_smallcnn_noise_saltpepper_speckle_unmatched_only}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN salt-pepper+speckle unmatched-only noise backfill}"

exec bash "${SCRIPT_DIR}/backfill_cifar10_noise_matched_common.sh" "$@"
