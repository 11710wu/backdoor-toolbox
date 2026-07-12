#!/usr/bin/env bash

# Legacy: UPGD only. Prefer run_resnet50_arch_05_belt_upgd.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[warn] ${BASH_SOURCE[0]} is legacy; redirecting to run_resnet50_arch_05_belt_upgd.sh (ATTACK_LIST=upgd)"
ATTACK_LIST=upgd bash "${SCRIPT_DIR}/run_resnet50_arch_05_belt_upgd.sh"
