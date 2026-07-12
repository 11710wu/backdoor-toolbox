#!/usr/bin/env bash

# Legacy: WaNet only. Prefer run_resnet50_arch_03_sig_wanet.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[warn] ${BASH_SOURCE[0]} is legacy; redirecting to run_resnet50_arch_03_sig_wanet.sh (ATTACK_LIST=WaNet)"
ATTACK_LIST=WaNet bash "${SCRIPT_DIR}/run_resnet50_arch_03_sig_wanet.sh"
