#!/usr/bin/env bash

# Legacy: BELT only. Prefer run_resnet50_arch_05_belt_upgd.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[warn] ${BASH_SOURCE[0]} is legacy; redirecting to run_resnet50_arch_05_belt_upgd.sh (ATTACK_LIST=belt)"
ATTACK_LIST=belt PREPARE_UPGD_CIFAR=0 PREPARE_UPGD_TINY=0 bash "${SCRIPT_DIR}/run_resnet50_arch_05_belt_upgd.sh"
