#!/usr/bin/env bash

# CIFAR-10 SmallCNN architecture backfill with baseline-matched configs.
#
# This script only reruns architecture configs that can be strictly matched to
# the original CIFAR-10/STL10 ResNet18 baseline. It replaces old unmatched
# rows such as badnet, off-grid SIG/WaNet strengths, and BELT mask-axis rows.
#
# Usage:
#   bash run/backfill_cifar10_smallcnn_arch_matched.sh
#
# Useful overrides:
#   PYTHON_BIN=/root/anaconda3/envs/backtool/bin/python DEVICES=1 bash run/backfill_cifar10_smallcnn_arch_matched.sh
#   DRY_RUN=1 bash run/backfill_cifar10_smallcnn_arch_matched.sh
#   STOP_ON_FAIL=1 bash run/backfill_cifar10_smallcnn_arch_matched.sh

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="cifar10"
MODEL="small_cnn"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="$LOG_DIR/backfill_cifar10_smallcnn_arch_matched_${TIMESTAMP}.log"

DEFENSES=(
  "SentiNet"
  "STRIP"
  "ScaleUp"
  "IBD_PSC"
)

CONFIGS=(
  "basic|0.005|alpha=0.2|-alpha 0.2"
  "basic|0.005|alpha=0.5|-alpha 0.5"
  "basic|0.005|alpha=1.0|-alpha 1.0"
  "basic|0.010|alpha=0.2|-alpha 0.2"
  "basic|0.010|alpha=0.5|-alpha 0.5"
  "basic|0.010|alpha=1.0|-alpha 1.0"
  "SIG|0.005|delta=28|-f 6 -delta 28 -label_mode clean"
  "SIG|0.005|delta=36|-f 6 -delta 36 -label_mode clean"
  "SIG|0.010|delta=28|-f 6 -delta 28 -label_mode clean"
  "SIG|0.010|delta=36|-f 6 -delta 36 -label_mode clean"
  "WaNet|0.005|s=0.6|-cover_rate 0.010 -s 0.6 -k 4"
  "WaNet|0.010|s=0.6|-cover_rate 0.020 -s 0.6 -k 4"
  "belt|0.010|alpha=0.10|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.10"
  "belt|0.010|alpha=0.20|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.20"
  "belt|0.010|alpha=0.30|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.30"
  "belt|0.020|alpha=0.10|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.10"
  "belt|0.020|alpha=0.20|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.20"
  "belt|0.020|alpha=0.30|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.30"
)

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"

  echo
  echo ">>> ${description}"
  echo "${cmd}"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] skipped"
    rm -f "$tmp_out"
    return 0
  fi

  eval "$cmd" 2>&1 | tee "$tmp_out"
  local exit_code="${PIPESTATUS[0]}"
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

echo "============================================================"
echo "CIFAR-10 SmallCNN unmatched-only architecture backfill"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "dataset      : ${DATASET}"
echo "model        : ${MODEL}"
echo "devices      : ${DEVICES}"
echo "result root  : ${POISONED_TRAIN_SET_ROOT}"
echo "configs      : ${#CONFIGS[@]} unmatched replacement configs"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

echo
echo "----- clean model preparation -----"
run_command \
  "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
  "Create clean set/model dir for architecture backfill"
run_command \
  "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
  "Train clean model for architecture backfill"

for phase in create train source transfer defenses; do
  echo
  echo "----- ${phase} -----"
  for item in "${CONFIGS[@]}"; do
    IFS="|" read -r attack rate label args <<< "$item"
    case "$phase" in
      create)
        run_command \
          "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
          "Create: ${attack}, rate=${rate}, ${label}"
        ;;
      train)
        run_command \
          "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
          "Train: ${attack}, rate=${rate}, ${label}"
        ;;
      source)
        run_command \
          "${PYTHON_BIN} test_model.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
          "Source test: ${attack}, rate=${rate}, ${label}"
        ;;
      transfer)
        run_command \
          "${PYTHON_BIN} test_stl10.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
          "STL10 transfer: ${attack}, rate=${rate}, ${label}"
        ;;
      defenses)
        for defense in "${DEFENSES[@]}"; do
          run_command \
            "${PYTHON_BIN} other_defense.py $(base_args) -defense=${defense} -poison_type=${attack} -poison_rate=${rate} ${args}" \
            "Defense ${defense}: ${attack}, rate=${rate}, ${label}"
        done
        ;;
    esac
  done
done

echo
echo "============================================================"
echo "CIFAR-10 SmallCNN architecture backfill finished."
echo "Check ${ERROR_LOG} for failures."
echo "============================================================"
