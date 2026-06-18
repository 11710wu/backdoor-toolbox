#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-resnet18}"
export NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-gaussian uniform}"
export RUN_NAME="${RUN_NAME:-backfill_cifar10_resnet18_noise_gaussian_uniform_unmatched_only}"
export RUN_TITLE="${RUN_TITLE:-CIFAR-10 ResNet18 gaussian+uniform unmatched-only noise backfill}"

exec bash "${SCRIPT_DIR}/backfill_cifar10_noise_matched_common.sh" "$@"
