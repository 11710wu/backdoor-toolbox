#!/usr/bin/env bash

# Backfill the missing Tiny-ImageNet adaptive_blend baseline rows:
#   poison_rate=0.005, cover_rate=0.005, alpha=0.01
# for the three baseline architectures:
#   resnet18, mobilenetv2, vgg19_bn
#
# This writes to the default poisoned_train_set root.
#
# Usage:
#   DEVICES=0 bash run/backfill_tiny_imagenet_adaptive_blend_rate0005_alpha001_all_arch.sh
#
# Useful overrides:
#   DRY_RUN=1 DEVICES=0 bash run/backfill_tiny_imagenet_adaptive_blend_rate0005_alpha001_all_arch.sh
#   STOP_ON_FAIL=1 DEVICES=0 bash run/backfill_tiny_imagenet_adaptive_blend_rate0005_alpha001_all_arch.sh
#   RUN_QWEN=0 bash run/backfill_tiny_imagenet_adaptive_blend_rate0005_alpha001_all_arch.sh
#   RUN_CORRUPTION=0 bash run/backfill_tiny_imagenet_adaptive_blend_rate0005_alpha001_all_arch.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_CORRUPTION="${RUN_CORRUPTION:-1}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-/workspace/data/tiny-target-domain-qwen-full-organized}"
DEFENSES_STR="${DEFENSES_STR:-SentiNet STRIP ScaleUp IBD_PSC}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${LOG_DIR}/backfill_tiny_imagenet_adaptive_blend_rate0005_alpha001_${TIMESTAMP}.log"

read -r -a DEFENSES <<< "${DEFENSES_STR}"

DATASET="tiny_imagenet"
ATTACK="adaptive_blend"
POISON_RATE="0.005"
COVER_RATE="0.005"
ALPHA="0.01"
TRIGGER="hellokitty_64.png"
MODELS=("resnet18" "mobilenetv2" "vgg19_bn")

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"

  echo
  echo ">>> ${description}"
  echo "${cmd}"

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] skipped"
    rm -f "${tmp_out}"
    return 0
  fi

  eval "${cmd}" 2>&1 | tee "${tmp_out}"
  local exit_code="${PIPESTATUS[0]}"
  if [ "${exit_code}" -ne 0 ]; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed with exit code ${exit_code}"
      echo "command: ${cmd}"
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

base_args() {
  local model="$1"
  echo "-dataset=${DATASET} -model=${model} -devices=${DEVICES}"
}

attack_args() {
  echo "-poison_type=${ATTACK} -poison_rate=${POISON_RATE} -cover_rate=${COVER_RATE} -alpha=${ALPHA} -trigger=${TRIGGER}"
}

run_one_model() {
  local model="$1"
  local label="${DATASET} ${model} ${ATTACK} rate=${POISON_RATE} cover=${COVER_RATE} alpha=${ALPHA}"

  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args "${model}") $(attack_args)" \
    "Create: ${label}"

  run_command \
    "${PYTHON_BIN} train_on_poisoned_set.py $(base_args "${model}") $(attack_args)" \
    "Train: ${label}"

  run_command \
    "${PYTHON_BIN} test_model.py $(base_args "${model}") $(attack_args)" \
    "Source test: ${label}"

  run_command \
    "${PYTHON_BIN} test_tiny_target_domain.py $(base_args "${model}") -source_dataset=${DATASET} $(attack_args)" \
    "ImageNetV2-tiny transfer: ${label}"

  if [ "${RUN_QWEN}" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(base_args "${model}") -source_dataset=${DATASET} $(attack_args) -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}" \
      "Qwen target-domain transfer: ${label}"
  fi

  if [ "${RUN_CORRUPTION}" = "1" ]; then
    for severity in 2 3; do
      run_command \
        "${PYTHON_BIN} test_tiny_imagenet.py $(base_args "${model}") -source_dataset=${DATASET} $(attack_args) -corruption_type=frost -severity=${severity}" \
        "Tiny-ImageNet-C frost severity=${severity}: ${label}"
    done
  fi

  for defense in "${DEFENSES[@]}"; do
    run_command \
      "${PYTHON_BIN} other_defense.py $(base_args "${model}") -defense=${defense} $(attack_args)" \
      "Defense ${defense}: ${label}"
  done
}

echo "============================================================"
echo "Backfill Tiny-ImageNet adaptive_blend rate=0.005 alpha=0.01"
echo "============================================================"
echo "repo root      : ${REPO_ROOT}"
echo "python         : ${PYTHON_BIN}"
echo "devices        : ${DEVICES}"
echo "models         : ${MODELS[*]}"
echo "run qwen       : ${RUN_QWEN}"
echo "run corruption : ${RUN_CORRUPTION}"
echo "qwen domain    : ${QWEN_TARGET_DOMAIN_DIR}"
echo "defenses       : ${DEFENSES[*]}"
echo "dry run        : ${DRY_RUN}"
echo "stop on fail   : ${STOP_ON_FAIL}"
echo "error log      : ${ERROR_LOG}"
echo "============================================================"

for model in "${MODELS[@]}"; do
  run_one_model "${model}"
done

echo
echo "============================================================"
echo "Backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
