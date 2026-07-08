#!/usr/bin/env bash
#
# Adaptive-Patch alpha=0.2 rerun — Script 2/5 (GPU1)
# Workload: 5 result dirs (~15 GPU-hours @ 3h/dir)
#   - ResNet18: all 3 pr/cover pairs
#   - MobileNetV2: pr/cover 0.05/0.1 and 0.01/0.02
#
# Suggested launch:
#   DEVICES=1 bash run/rerun_tiny_imagenet_adaptive_patch_alpha_0_2_part0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PART_TITLE="Tiny-ImageNet part0 (2/5)"
PART_SLUG="part0"
EXPECTED_DIRS=5
TINY_JOBS=(
  "resnet18|0.05|0.1"
  "resnet18|0.01|0.02"
  "resnet18|0.005|0.01"
  "mobilenetv2|0.05|0.1"
  "mobilenetv2|0.01|0.02"
)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_rerun_tiny_imagenet_adaptive_patch_alpha_0_2_common.sh"
