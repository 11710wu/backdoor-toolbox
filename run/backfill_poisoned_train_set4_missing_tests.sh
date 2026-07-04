#!/usr/bin/env bash

# Test-only backfill for poisoned_train_set4.
#
# Scope:
#   CIFAR-10 SmallCNN:
#     - SIG: source test + STL10 transfer
#     - WaNet: source test + STL10 transfer
#   Tiny-ImageNet ResNet34:
#     - SIG: source test + ImageNetV2-tiny transfer + Qwen transfer
#     - WaNet: source test + ImageNetV2-tiny transfer + Qwen transfer
#     - adaptive_blend/adaptive_patch/belt/blend: Qwen transfer only
#
# This script does not recreate poisoned sets and does not retrain models.
#
# Usage:
#   cd /workspace/backdoor-toolbox-new1
#   DEVICES=0 bash run/backfill_poisoned_train_set4_missing_tests.sh
#
# Useful overrides:
#   DRY_RUN=1 bash run/backfill_poisoned_train_set4_missing_tests.sh
#   RUN_CIFAR=0 DEVICES=1 bash run/backfill_poisoned_train_set4_missing_tests.sh
#   RUN_TINY=0 DEVICES=0 bash run/backfill_poisoned_train_set4_missing_tests.sh
#   RUN_QWEN=0 bash run/backfill_poisoned_train_set4_missing_tests.sh
#   STOP_ON_FAIL=1 bash run/backfill_poisoned_train_set4_missing_tests.sh

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
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${LOG_DIR}/backfill_poisoned_train_set4_missing_tests_${TIMESTAMP}.log"

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

cifar_base_args() {
  echo "-dataset=cifar10 -model=small_cnn -devices=${DEVICES}"
}

tiny_base_args() {
  echo "-dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES}"
}

run_cifar_source_and_transfer() {
  local attack="$1"
  local rate="$2"
  local args="$3"
  local label="cifar10 SmallCNN ${attack} rate=${rate} ${args}"

  run_command \
    "${PYTHON_BIN} test_model.py $(cifar_base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
    "CIFAR source test: ${label}"

  run_command \
    "${PYTHON_BIN} test_stl10.py $(cifar_base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
    "STL10 transfer: ${label}"
}

run_tiny_source_imagenetv2_qwen() {
  local attack="$1"
  local rate="$2"
  local args="$3"
  local label="tiny_imagenet ResNet34 ${attack} rate=${rate} ${args}"

  run_command \
    "${PYTHON_BIN} test_model.py $(tiny_base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
    "Tiny source test: ${label}"

  run_command \
    "${PYTHON_BIN} test_tiny_target_domain.py $(tiny_base_args) -source_dataset=tiny_imagenet -poison_type=${attack} -poison_rate=${rate} ${args}" \
    "ImageNetV2-tiny transfer: ${label}"

  if [ "${RUN_QWEN}" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(tiny_base_args) -source_dataset=tiny_imagenet -poison_type=${attack} -poison_rate=${rate} ${args} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}" \
      "Qwen transfer: ${label}"
  fi
}

run_tiny_qwen_only() {
  local attack="$1"
  local rate="$2"
  local args="$3"
  local label="tiny_imagenet ResNet34 ${attack} rate=${rate} ${args}"

  if [ "${RUN_QWEN}" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(tiny_base_args) -source_dataset=tiny_imagenet -poison_type=${attack} -poison_rate=${rate} ${args} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}" \
      "Qwen transfer: ${label}"
  fi
}

echo "============================================================"
echo "Backfill poisoned_train_set4 missing test/transfer files"
echo "============================================================"
echo "repo root    : ${REPO_ROOT}"
echo "python       : ${PYTHON_BIN}"
echo "devices      : ${DEVICES}"
echo "result root  : ${POISONED_TRAIN_SET_ROOT}"
echo "run cifar    : ${RUN_CIFAR}"
echo "run tiny     : ${RUN_TINY}"
echo "run qwen     : ${RUN_QWEN}"
echo "qwen domain  : ${QWEN_TARGET_DOMAIN_DIR}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "${RUN_CIFAR}" = "1" ]; then
  echo
  echo "----- CIFAR-10 SmallCNN: SIG source + STL10 -----"
  for rate in 0.005 0.010; do
    for delta in 20 30 40; do
      run_cifar_source_and_transfer "SIG" "${rate}" "-f 6 -delta ${delta} -label_mode clean"
    done
  done

  echo
  echo "----- CIFAR-10 SmallCNN: WaNet source + STL10 -----"
  for rate_cover in "0.005 0.010" "0.010 0.020"; do
    read -r rate cover <<< "${rate_cover}"
    for s in 0.4 0.5 0.8; do
      run_cifar_source_and_transfer "WaNet" "${rate}" "-cover_rate ${cover} -s ${s} -k 4"
    done
  done
fi

if [ "${RUN_TINY}" = "1" ]; then
  echo
  echo "----- Tiny-ImageNet ResNet34: SIG source + ImageNetV2 + Qwen -----"
  for rate in 0.001 0.005; do
    for delta in 20 30 40; do
      run_tiny_source_imagenetv2_qwen "SIG" "${rate}" "-f 6 -delta ${delta} -label_mode clean"
    done
  done

  echo
  echo "----- Tiny-ImageNet ResNet34: WaNet source + ImageNetV2 + Qwen -----"
  for rate_cover in "0.001 0.002" "0.005 0.010"; do
    read -r rate cover <<< "${rate_cover}"
    for s in 0.4 0.5 0.8; do
      run_tiny_source_imagenetv2_qwen "WaNet" "${rate}" "-cover_rate ${cover} -s ${s} -k 4"
    done
  done

  echo
  echo "----- Tiny-ImageNet ResNet34: Qwen-only for attacks with source/ImageNetV2 already present -----"
  for rate_cover in "0.001 0.001" "0.005 0.005"; do
    read -r rate cover <<< "${rate_cover}"
    for alpha in 0.05 0.15 0.25; do
      run_tiny_qwen_only "adaptive_blend" "${rate}" "-cover_rate ${cover} -alpha ${alpha} -trigger=hellokitty_64.png"
    done
  done

  for rate_cover in "0.001 0.002" "0.005 0.010"; do
    read -r rate cover <<< "${rate_cover}"
    for alpha in 0.1 0.2 0.3; do
      run_tiny_qwen_only "adaptive_patch" "${rate}" "-cover_rate ${cover} -alpha ${alpha}"
    done
  done

  for rate in 0.001 0.005; do
    for mask in 0.1 0.2 0.3; do
      run_tiny_qwen_only "belt" "${rate}" "-cover_rate 0.5 -mask_rate ${mask} -alpha 1.0"
    done
  done

  for rate in 0.001 0.005; do
    for alpha in 0.05 0.15 0.3; do
      run_tiny_qwen_only "blend" "${rate}" "-alpha ${alpha} -trigger=hellokitty_64.png"
    done
  done
fi

echo
echo "============================================================"
echo "Backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
