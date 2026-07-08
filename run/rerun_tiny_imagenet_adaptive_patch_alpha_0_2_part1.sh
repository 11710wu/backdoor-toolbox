#!/usr/bin/env bash
#
# Adaptive-Patch alpha=0.2 rerun — Script 3/5 (GPU2)
# Workload: 4 result dirs (~12 GPU-hours @ 3h/dir)
#   - MobileNetV2: pr/cover 0.005/0.01
#   - VGG19: all 3 pr/cover pairs
#
# Suggested launch:
#   DEVICES=2 bash run/rerun_tiny_imagenet_adaptive_patch_alpha_0_2_part1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PART_TITLE="Tiny-ImageNet part1 (3/5)"
PART_SLUG="part1"
EXPECTED_DIRS=4
TINY_JOBS=(
  "mobilenetv2|0.005|0.01"
  "vgg19_bn|0.05|0.1"
  "vgg19_bn|0.01|0.02"
  "vgg19_bn|0.005|0.01"
)

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_rerun_tiny_imagenet_adaptive_patch_alpha_0_2_common.sh"
