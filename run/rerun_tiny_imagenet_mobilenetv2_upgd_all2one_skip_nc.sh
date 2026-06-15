#!/usr/bin/env bash

# Full rerun for Tiny-ImageNet MobileNetV2 UPGD all-to-one.
# Rebuilds raw base, poison sets, trained models, tests, transfers, and non-NC defenses.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-mobilenetv2}"
export DEVICES="${DEVICES:-2}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-/workspace/backdoor-toolbox/poisoned_train_set2}"

export RUN_PREP="${RUN_PREP:-1}"
export RUN_CREATE="${RUN_CREATE:-1}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_TEST="${RUN_TEST:-1}"
export RUN_TRANSFER="${RUN_TRANSFER:-1}"
export RUN_QWEN_TRANSFER="${RUN_QWEN_TRANSFER:-1}"
export RUN_DEFENSES="${RUN_DEFENSES:-1}"

export FORCE="${FORCE:-1}"
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
export DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

exec bash "${SCRIPT_DIR}/resume_tiny_imagenet_upgd_all2one_skip_nc.sh"
