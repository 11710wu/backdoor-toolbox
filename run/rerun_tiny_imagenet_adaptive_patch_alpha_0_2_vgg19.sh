#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="vgg19_bn" bash "${SCRIPT_DIR}/rerun_tiny_imagenet_adaptive_patch_alpha_0_2_resnet18.sh"
