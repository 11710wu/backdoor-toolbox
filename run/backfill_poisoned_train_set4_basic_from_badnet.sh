#!/usr/bin/env bash

# Backfill the Basic/BadNet-style patch attack rows that correspond to the
# mistaken badnet rows in poisoned_train_set4.
#
# Scope:
#   CIFAR-10 SmallCNN:
#     rates 0.005, 0.010; alpha 0.2, 0.5, 1.0
#   Tiny-ImageNet ResNet34:
#     rates 0.001, 0.005; alpha 0.2, 0.5, 1.0
#
# The output directories use poison_type=basic, so they include the trigger
# name in the directory:
#   basic_..._trigger=badnet_patch_32.png...
#   basic_..._trigger=badnet_patch_64.png...
#
# Usage:
#   bash run/backfill_poisoned_train_set4_basic_from_badnet.sh
#
# Useful overrides:
#   DEVICES=0 DRY_RUN=1 bash run/backfill_poisoned_train_set4_basic_from_badnet.sh
#   RUN_TINY=0 DEVICES=0 bash run/backfill_poisoned_train_set4_basic_from_badnet.sh
#   RUN_CIFAR=0 DEVICES=1 bash run/backfill_poisoned_train_set4_basic_from_badnet.sh
#   STOP_ON_FAIL=1 bash run/backfill_poisoned_train_set4_basic_from_badnet.sh
#   RUN_QWEN=0 bash run/backfill_poisoned_train_set4_basic_from_badnet.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_CIFAR="${RUN_CIFAR:-1}"
RUN_TINY="${RUN_TINY:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-/workspace/data/tiny-target-domain-qwen-full-organized}"
DEFENSES_STR="${DEFENSES_STR:-SentiNet STRIP ScaleUp IBD_PSC}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${LOG_DIR}/backfill_poisoned_train_set4_basic_from_badnet_${TIMESTAMP}.log"

read -r -a DEFENSES <<< "${DEFENSES_STR}"

CIFAR_RATES=("0.005" "0.010")
TINY_RATES=("0.001" "0.005")
ALPHAS=("0.2" "0.5" "1.0")

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
  local dataset="$1"
  local model="$2"
  echo "-dataset=${dataset} -model=${model} -devices=${DEVICES}"
}

run_basic_config() {
  local dataset="$1"
  local model="$2"
  local rate="$3"
  local alpha="$4"
  local trigger="$5"
  local transfer_script="$6"
  local label="${dataset} ${model} basic rate=${rate} alpha=${alpha}"
  local args="-poison_type=basic -poison_rate=${rate} -alpha ${alpha} -trigger=${trigger}"

  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args "${dataset}" "${model}") ${args}" \
    "Create: ${label}"

  run_command \
    "${PYTHON_BIN} train_on_poisoned_set.py $(base_args "${dataset}" "${model}") ${args}" \
    "Train: ${label}"

  run_command \
    "${PYTHON_BIN} test_model.py $(base_args "${dataset}" "${model}") ${args}" \
    "Source test: ${label}"

  if [ "${transfer_script}" = "test_tiny_target_domain.py" ]; then
    run_command \
      "${PYTHON_BIN} ${transfer_script} $(base_args "${dataset}" "${model}") -source_dataset=${dataset} ${args}" \
      "ImageNetV2-tiny transfer: ${label}"

    if [ "${RUN_QWEN}" = "1" ]; then
      run_command \
        "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(base_args "${dataset}" "${model}") -source_dataset=${dataset} ${args} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}" \
        "Qwen transfer: ${label}"
    fi
  else
    run_command \
      "${PYTHON_BIN} ${transfer_script} $(base_args "${dataset}" "${model}") ${args}" \
      "STL10 transfer: ${label}"
  fi

  for defense in "${DEFENSES[@]}"; do
    run_command \
      "${PYTHON_BIN} other_defense.py $(base_args "${dataset}" "${model}") -defense=${defense} ${args}" \
      "Defense ${defense}: ${label}"
  done
}

echo "============================================================"
echo "Backfill poisoned_train_set4 basic rows from mistaken badnet rows"
echo "============================================================"
echo "repo root    : ${REPO_ROOT}"
echo "python       : ${PYTHON_BIN}"
echo "devices      : ${DEVICES}"
echo "result root  : ${POISONED_TRAIN_SET_ROOT}"
echo "run cifar    : ${RUN_CIFAR}"
echo "run tiny     : ${RUN_TINY}"
echo "run qwen     : ${RUN_QWEN}"
echo "qwen domain  : ${QWEN_TARGET_DOMAIN_DIR}"
echo "defenses     : ${DEFENSES[*]}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "${RUN_CIFAR}" = "1" ]; then
  echo
  echo "----- CIFAR-10 SmallCNN basic backfill -----"
  for rate in "${CIFAR_RATES[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      run_basic_config "cifar10" "small_cnn" "${rate}" "${alpha}" "badnet_patch_32.png" "test_stl10.py"
    done
  done
fi

if [ "${RUN_TINY}" = "1" ]; then
  echo
  echo "----- Tiny-ImageNet ResNet34 basic backfill -----"
  for rate in "${TINY_RATES[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      run_basic_config "tiny_imagenet" "resnet34" "${rate}" "${alpha}" "badnet_patch_64.png" "test_tiny_target_domain.py"
    done
  done
fi

echo
echo "============================================================"
echo "Backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
