#!/usr/bin/env bash
# Backfill Tiny-ImageNet ResNet34 clean-label SIG/UPGD target-domain transfer tests only.
# This script does not create poison sets, train models, source-test, or run defenses.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"
export POISONED_TRAIN_SET_ROOT

DATASET="tiny_imagenet"
MODEL="resnet34"
ARCH_NAME="ResNet34_tiny_imagenet"
POISON_RATES="${POISON_RATES:-0.001 0.005}"
SIG_DELTAS="${SIG_DELTAS:-20 30 40}"
SIG_F="${SIG_F:-6}"
UPGD_EPS_VALUES="${UPGD_EPS_VALUES:-4 8 12}"
UPGD_CONSTRAINT="${UPGD_CONSTRAINT:-Linf}"
UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
LABEL_MODE="${LABEL_MODE:-clean}"

TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/data/imagenetv2-matched-frequency-tiny-organized}"
TARGET_DOMAIN_QWEN_DIR="${TARGET_DOMAIN_QWEN_DIR:-/workspace/backdoor-toolbox-new1/data/tiny-target-domain-qwen-full-organized}"
RUN_TARGET_DOMAIN="${RUN_TARGET_DOMAIN:-1}"
RUN_QWEN_TRANSFER="${RUN_QWEN_TRANSFER:-1}"

RUN_SIG="${RUN_SIG:-1}"
RUN_UPGD="${RUN_UPGD:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/backfill_tiny_imagenet_resnet34_sig_upgd_clean_transfer_${TIMESTAMP}.log}"

rate_dir_value() {
  printf "%.3f" "$1"
}

eps_dir_value() {
  printf "%.1f" "$1"
}

sig_dir() {
  local rate="$1"
  local delta="$2"
  echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/SIG_$(rate_dir_value "$rate")_delta=${delta}_f=${SIG_F}_mode=${LABEL_MODE}_poison_seed=2333_arch=${ARCH_NAME}"
}

upgd_dir() {
  local rate="$1"
  local eps="$2"
  echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_$(rate_dir_value "$rate")_eps=$(eps_dir_value "$eps")_constraint=${UPGD_CONSTRAINT}_steps=${UPGD_STEPS}_mode=${LABEL_MODE}_mult=${UPGD_STEPS_MULTIPLIER}_poison_seed=2333_arch=${ARCH_NAME}"
}

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  local exit_code

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${description}"
  echo "$cmd"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] skipped"
    return 0
  fi

  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"
  eval "$cmd" 2>&1 | tee "$tmp_out"
  exit_code="${PIPESTATUS[0]}"

  if [ "$exit_code" -ne 0 ]; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed with exit code ${exit_code}"
      echo "command: ${cmd}"
      echo "description: ${description}"
      echo "--- stdout/stderr ---"
      cat "$tmp_out" 2>/dev/null
      echo "---"
    } >> "$ERROR_LOG"

    if [ "$STOP_ON_FAIL" = "1" ]; then
      rm -f "$tmp_out"
      exit "$exit_code"
    fi
  fi

  rm -f "$tmp_out"
  return "$exit_code"
}

skip_file() {
  local path="$1"
  local description="$2"
  if [ "$FORCE" != "1" ] && [ -s "$path" ]; then
    echo "[SKIP] ${description}: ${path}"
    return 0
  fi
  return 1
}

run_transfer_pair() {
  local dir="$1"
  local args="$2"
  local description="$3"
  local target_result="$4"
  local qwen_result="$5"

  echo
  echo "========== ${description} =========="
  echo "dir: ${dir}"

  if [ ! -d "$dir" ]; then
    echo "[WARN] poison dir not found, command may fail if model is missing: ${dir}"
  fi

  if [ "$RUN_TARGET_DOMAIN" = "1" ]; then
    if skip_file "${dir}/${target_result}" "target-domain transfer results"; then
      :
    else
      run_command \
        "${PYTHON_BIN} test_tiny_target_domain.py ${args} -source_dataset=${DATASET} -target_domain_dir=${TARGET_DOMAIN_DIR}" \
        "Tiny target-domain transfer ${description}"
    fi
  fi

  if [ "$RUN_QWEN_TRANSFER" = "1" ]; then
    if skip_file "${dir}/${qwen_result}" "Qwen target-domain transfer results"; then
      :
    else
      run_command \
        "${PYTHON_BIN} test_tiny_target_domain_qwen.py ${args} -source_dataset=${DATASET} -target_domain_dir=${TARGET_DOMAIN_QWEN_DIR}" \
        "Tiny Qwen target-domain transfer ${description}"
    fi
  fi
}

echo "============================================================"
echo "Backfill Tiny-ImageNet ResNet34 SIG/UPGD clean transfer tests"
echo "============================================================"
echo "output_root       : ${POISONED_TRAIN_SET_ROOT}"
echo "poison_rates      : ${POISON_RATES}"
echo "sig_deltas        : ${SIG_DELTAS}"
echo "upgd_eps_values   : ${UPGD_EPS_VALUES}"
echo "target_domain     : ${RUN_TARGET_DOMAIN} (${TARGET_DOMAIN_DIR})"
echo "qwen_transfer     : ${RUN_QWEN_TRANSFER} (${TARGET_DOMAIN_QWEN_DIR})"
echo "force             : ${FORCE}"
echo "dry_run           : ${DRY_RUN}"
echo "error_log         : ${ERROR_LOG}"
echo "============================================================"

if [ "$RUN_SIG" = "1" ]; then
  for rate in ${POISON_RATES}; do
    for delta in ${SIG_DELTAS}; do
      dir="$(sig_dir "$rate" "$delta")"
      args="-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES} -poison_type=SIG -poison_rate=${rate} -f=${SIG_F} -delta=${delta} -label_mode=${LABEL_MODE}"
      run_transfer_pair \
        "$dir" \
        "$args" \
        "ResNet34 SIG rate=${rate} delta=${delta}" \
        "test_tiny_target_domain_results_delta=${delta}.txt" \
        "test_tiny_target_domain_qwen_results_delta=${delta}.txt"
    done
  done
fi

if [ "$RUN_UPGD" = "1" ]; then
  for rate in ${POISON_RATES}; do
    for eps in ${UPGD_EPS_VALUES}; do
      dir="$(upgd_dir "$rate" "$eps")"
      args="-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES} -poison_type=upgd -poison_rate=${rate} -eps=${eps} -constraint=${UPGD_CONSTRAINT} -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER} -label_mode=${LABEL_MODE}"
      run_transfer_pair \
        "$dir" \
        "$args" \
        "ResNet34 UPGD rate=${rate} eps=${eps}" \
        "test_tiny_target_domain_results.txt" \
        "test_tiny_target_domain_qwen_results.txt"
    done
  done
fi

echo
echo "============================================================"
echo "Backfill finished. Failed commands, if any: ${ERROR_LOG}"
echo "============================================================"
