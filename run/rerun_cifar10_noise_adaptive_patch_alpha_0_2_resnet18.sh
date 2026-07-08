#!/usr/bin/env bash
#
# Adaptive-Patch alpha=0.2 noise rerun — Script 4/5 (GPU3)
# Workload: 24 result dirs (~24 GPU-hours @ 1h/dir)
#   - ResNet18, 2 pr/cover x 4 noise x 3 levels
#
# Suggested launch:
#   DEVICES=3 bash run/rerun_cifar10_noise_adaptive_patch_alpha_0_2_resnet18.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="resnet18"
ARCH_NAME="ResNet18_cifar10"
PART_TITLE="Noise ResNet18 (4/5)"
PART_SLUG="resnet18"
EXPECTED_DIRS=24
MOVE_TO_SET1=0

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_rerun_cifar10_noise_adaptive_patch_alpha_0_2_common.sh"
