#!/usr/bin/env bash

# Legacy: Blend only. Prefer run_resnet50_arch_01_basic_blend.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[warn] ${BASH_SOURCE[0]} is legacy; redirecting to run_resnet50_arch_01_basic_blend.sh (ATTACK_LIST=blend)"
ATTACK_LIST=blend bash "${SCRIPT_DIR}/run_resnet50_arch_01_basic_blend.sh"
