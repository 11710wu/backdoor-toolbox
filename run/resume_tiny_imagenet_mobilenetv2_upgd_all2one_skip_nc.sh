#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL="${MODEL:-mobilenetv2}"
export DEVICES="${DEVICES:-2}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-/workspace/backdoor-toolbox/poisoned_train_set2}"
export START_RATE="${START_RATE:-0.001}"
export START_EPS="${START_EPS:-18}"
export START_AFTER="${START_AFTER:-0}"
exec bash "${SCRIPT_DIR}/resume_tiny_imagenet_upgd_all2one_skip_nc.sh"
