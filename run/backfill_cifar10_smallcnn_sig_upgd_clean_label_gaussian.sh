#!/usr/bin/env bash

# Backfill CIFAR-10 SmallCNN clean-label SIG/UPGD with gaussian input noise.
# This script is the default owner of UPGD raw-base preparation, so it can be
# launched first or together with the other three noise-specific scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NOISE_TYPE_FILTER="gaussian"
export RUN_NAME="${RUN_NAME:-backfill_cifar10_smallcnn_sig_upgd_clean_label_gaussian}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN clean-label SIG/UPGD gaussian-noise backfill}"
export SKIP_UPGD_PREP="${SKIP_UPGD_PREP:-0}"
export WAIT_FOR_UPGD_BASE="${WAIT_FOR_UPGD_BASE:-0}"

exec bash "${SCRIPT_DIR}/run_cifar10_small_cnn_noise_sig_upgd_clean_label.sh" "$@"
