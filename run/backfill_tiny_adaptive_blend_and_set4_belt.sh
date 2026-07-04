#!/usr/bin/env bash

# Run the remaining Tiny-ImageNet adaptive_blend backfill together with the
# poisoned_train_set4 matched BELT backfill.
#
# Usage:
#   cd /workspace/backdoor-toolbox-new1
#   DEVICES=0 bash run/backfill_tiny_adaptive_blend_and_set4_belt.sh
#
# Useful overrides:
#   DRY_RUN=1 DEVICES=0 bash run/backfill_tiny_adaptive_blend_and_set4_belt.sh
#   STOP_ON_FAIL=1 DEVICES=0 bash run/backfill_tiny_adaptive_blend_and_set4_belt.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/tiny-target-domain-qwen-full-organized}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${LOG_DIR}/backfill_tiny_adaptive_blend_and_set4_belt_${TIMESTAMP}.log"

run_script() {
  local script_path="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_script_$$_${RANDOM}.out")"

  echo
  echo ">>> ${description}"
  echo "DEVICES=${DEVICES} DRY_RUN=${DRY_RUN} STOP_ON_FAIL=${STOP_ON_FAIL} QWEN_TARGET_DOMAIN_DIR=${QWEN_TARGET_DOMAIN_DIR} bash ${script_path}"

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] executing child script in dry-run mode"
  fi

  DEVICES="${DEVICES}" \
  DRY_RUN="${DRY_RUN}" \
  STOP_ON_FAIL="${STOP_ON_FAIL}" \
  QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR}" \
    bash "${script_path}" 2>&1 | tee "${tmp_out}"

  local exit_code="${PIPESTATUS[0]}"
  if [ "${exit_code}" -ne 0 ]; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] script failed with exit code ${exit_code}"
      echo "script: ${script_path}"
      echo "description: ${description}"
      echo "--- stdout/stderr ---"
      cat "${tmp_out}" 2>/dev/null
      echo "---"
    } >> "${ERROR_LOG}"

    if [ "${STOP_ON_FAIL}" = "1" ]; then
      rm -f "${tmp_out}"
      exit "${exit_code}"
    fi
  fi

  rm -f "${tmp_out}"
  return "${exit_code}"
}

echo "============================================================"
echo "Backfill Tiny adaptive_blend and poisoned_train_set4 BELT"
echo "============================================================"
echo "repo root    : ${REPO_ROOT}"
echo "devices      : ${DEVICES}"
echo "qwen domain  : ${QWEN_TARGET_DOMAIN_DIR}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

run_script \
  "run/backfill_tiny_imagenet_adaptive_blend_rate0005_alpha001_all_arch.sh" \
  "Tiny-ImageNet adaptive_blend rate=0.005 alpha=0.01 all baseline arch"

run_script \
  "run/backfill_set4_matched_belt_smallcnn_resnet34.sh" \
  "poisoned_train_set4 matched BELT SmallCNN/ResNet34"

echo
echo "============================================================"
echo "Combined backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
