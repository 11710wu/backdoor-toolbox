#!/usr/bin/env bash

# Backfill CIFAR-10 SmallCNN clean-label SIG/UPGD with salt-pepper input noise.
# By default this script reuses the UPGD raw-base model prepared by the gaussian
# script, and waits for it if both scripts are launched at the same time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOISE_TYPE_FILTER="salt_pepper"
export RUN_NAME="${RUN_NAME:-backfill_cifar10_smallcnn_sig_upgd_clean_label_salt_pepper}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN clean-label SIG/UPGD salt-pepper-noise backfill}"
export SKIP_UPGD_PREP="${SKIP_UPGD_PREP:-1}"
export WAIT_FOR_UPGD_BASE="${WAIT_FOR_UPGD_BASE:-1}"

exec bash "${SCRIPT_DIR}/run_cifar10_small_cnn_noise_sig_upgd_clean_label.sh" "$@"
