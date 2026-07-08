#!/usr/bin/env bash
#
# Deprecated wrapper — use the five balanced scripts instead:
#   backdoor-toolbox-new1:
#     run/rerun_adaptive_patch_alpha_0_2_cifar10_mnistm.sh          (GPU0, ~18h)
#     run/rerun_tiny_imagenet_adaptive_patch_alpha_0_2_part0.sh       (GPU1, ~15h)
#     run/rerun_tiny_imagenet_adaptive_patch_alpha_0_2_part1.sh       (GPU2, ~12h)
#   backdoor-toolbox-noise:
#     run/rerun_cifar10_noise_adaptive_patch_alpha_0_2_resnet18.sh    (GPU3, ~24h)
#     run/rerun_cifar10_noise_adaptive_patch_alpha_0_2_small_cnn.sh   (GPU4, ~24h)
#
# This entry point remains for backward compatibility and runs both noise models
# sequentially on one GPU (~48h).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "NOTE: prefer the split scripts for 5-GPU parallel execution." >&2
echo "Running ResNet18 then SmallCNN sequentially on DEVICES=${DEVICES:-0}." >&2

DEVICES="${DEVICES:-0}" CLEAN_OLD="${CLEAN_OLD:-1}" bash "${SCRIPT_DIR}/rerun_cifar10_noise_adaptive_patch_alpha_0_2_resnet18.sh"
DEVICES="${DEVICES:-0}" CLEAN_OLD=0 MOVE_TO_SET1="${MOVE_TO_SET1:-1}" bash "${SCRIPT_DIR}/rerun_cifar10_noise_adaptive_patch_alpha_0_2_small_cnn.sh"
