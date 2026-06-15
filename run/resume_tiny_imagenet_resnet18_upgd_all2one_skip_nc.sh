#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL="${MODEL:-resnet18}"
export DEVICES="${DEVICES:-0}"
exec bash "${SCRIPT_DIR}/resume_tiny_imagenet_upgd_all2one_skip_nc.sh"
