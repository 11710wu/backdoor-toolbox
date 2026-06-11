#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET="tiny_imagenet"
export MODEL="vgg19_bn"
exec bash "${SCRIPT_DIR}/run_upgd_clean_label_raw_base.sh" "$@"
