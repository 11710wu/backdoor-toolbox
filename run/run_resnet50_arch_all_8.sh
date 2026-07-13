#!/usr/bin/env bash

# Launch all ResNet50 architecture shards (4-way split).
# Tiny-ImageNet only, poison_rate=0.005.
# Default GPUs: 0..3. Override: DEVICES_LIST="0 1 2 3"
#
# Usage:
#   bash run/run_resnet50_arch_all_8.sh
#   DRY_RUN=1 bash run/run_resnet50_arch_all_8.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

SHARDS=(
  run_resnet50_arch_01_basic_blend.sh
  run_resnet50_arch_03_sig_wanet.sh
  run_resnet50_arch_04_adaptive.sh
  run_resnet50_arch_05_belt_upgd.sh
)

read -r -a DEVICES_ARR <<< "${DEVICES_LIST:-0 1 2 3}"

echo "Launching ${#SHARDS[@]} ResNet50 shards"
echo "DEVICES_LIST=${DEVICES_ARR[*]}"
echo "DRY_RUN=${DRY_RUN:-0}"
echo

pids=()
for i in "${!SHARDS[@]}"; do
  shard="${SHARDS[$i]}"
  if [ "$i" -ge "${#DEVICES_ARR[@]}" ]; then
    echo "Skip ${shard}: no GPU assigned (DEVICES_LIST shorter than ${#SHARDS[@]})"
    continue
  fi
  gpu="${DEVICES_ARR[$i]}"
  log="logs/${shard%.sh}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p logs
  echo ">>> GPU ${gpu}: ${shard}  (log: ${log})"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    DEVICES="${gpu}" DRY_RUN=1 bash "${SCRIPT_DIR}/${shard}" | tee "${log}"
  else
    (
      DEVICES="${gpu}" bash "${SCRIPT_DIR}/${shard}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
  fi
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN finished."
  exit 0
fi

echo
echo "Background PIDs: ${pids[*]}"
echo "Wait with: wait ${pids[*]}"
echo "Or: tail -f logs/run_resnet50_arch_*_gpu*.log"
