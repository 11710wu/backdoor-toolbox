#!/usr/bin/env bash

# Run the same fixed six-GPU Tiny-ImageNet BELT grid for two architectures.
# The models run sequentially so that each GPU owns at most one training job:
#   round 1: MobileNetV2 on GPUs 0..5
#   round 2: VGG19-BN on GPUs 0..5
# Within each round, every poison rate is statically assigned to two GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_MODEL_RUNNER="${SCRIPT_DIR}/run_tiny_imagenet_resnet18_belt_final_grid_6gpu.sh"

if [ ! -x "$ONE_MODEL_RUNNER" ]; then
  echo "Single-model runner is missing or not executable: ${ONE_MODEL_RUNNER}" >&2
  exit 2
fi

echo "============================================================"
echo "Round 1/2: Tiny-ImageNet BELT / MobileNetV2"
echo "============================================================"
MODEL=mobilenetv2 bash "$ONE_MODEL_RUNNER"

echo
echo "============================================================"
echo "Round 2/2: Tiny-ImageNet BELT / VGG19-BN"
echo "============================================================"
MODEL=vgg19_bn bash "$ONE_MODEL_RUNNER"

echo
echo "MobileNetV2 and VGG19-BN BELT grids completed successfully."
