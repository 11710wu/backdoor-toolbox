#!/usr/bin/env bash
#
# Adaptive-Patch alpha=0.2 noise rerun — Script 5/5 (GPU4)
# Workload: 24 result dirs (~24 GPU-hours @ 1h/dir)
#   - SmallCNN, 2 pr/cover x 4 noise x 3 levels
#
# Pipeline writes to poisoned_train_set/cifar10; by default moves finished dirs
# to poisoned_train_set1/cifar10 for analysis consistency.
#
# Suggested launch:
#   DEVICES=4 MOVE_TO_SET1=1 bash run/rerun_cifar10_noise_adaptive_patch_alpha_0_2_small_cnn.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="small_cnn"
ARCH_NAME="SmallCNN_cifar10"
PART_TITLE="Noise SmallCNN (5/5)"
PART_SLUG="small_cnn"
EXPECTED_DIRS=24
MOVE_TO_SET1="${MOVE_TO_SET1:-1}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_rerun_cifar10_noise_adaptive_patch_alpha_0_2_common.sh"
