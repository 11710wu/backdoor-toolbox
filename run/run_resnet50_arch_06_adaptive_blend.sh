#!/usr/bin/env bash

# Legacy: Adaptive-Blend only. Prefer run_resnet50_arch_04_adaptive.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[warn] ${BASH_SOURCE[0]} is legacy; redirecting to run_resnet50_arch_04_adaptive.sh (ATTACK_LIST=adaptive_blend)"
ATTACK_LIST=adaptive_blend bash "${SCRIPT_DIR}/run_resnet50_arch_04_adaptive.sh"
