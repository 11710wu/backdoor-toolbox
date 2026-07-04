#!/usr/bin/env bash
# Backfill Tiny-ImageNet baseline UPGD target-domain transfer tests only.
# This script does not create poison sets, train models, source-test, or run defenses.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-/workspace/backdoor-toolbox/poisoned_train_set}"
export POISONED_TRAIN_SET_ROOT

DATASET="tiny_imagenet"
MODELS="${MODELS:-resnet18 mobilenetv2 vgg19_bn}"
POISON_RATES="${POISON_RATES:-0.001 0.005}"
EPS_VALUES="${EPS_VALUES:-4 6 8 10 12 16 20 24}"
UPGD_CONSTRAINT="${UPGD_CONSTRAINT:-Linf}"
UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
LABEL_MODE="${LABEL_MODE:-clean}"

TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/backdoor-toolbox-new1/data/imagenetv2-matched-frequency-tiny-organized}"
TARGET_DOMAIN_QWEN_DIR="${TARGET_DOMAIN_QWEN_DIR:-/workspace/backdoor-toolbox-new1/data/tiny-target-domain-qwen-full-organized}"
RUN_TARGET_DOMAIN="${RUN_TARGET_DOMAIN:-1}"
RUN_QWEN_TRANSFER="${RUN_QWEN_TRANSFER:-1}"

FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/backfill_tiny_imagenet_baseline_upgd_transfer_${TIMESTAMP}.log}"

arch_name() {
  case "$1" in
    resnet18) echo "ResNet18_tiny_imagenet" ;;
    mobilenetv2) echo "mobilenetv2_tiny_imagenet" ;;
    vgg19_bn) echo "vgg19_bn_tiny_imagenet" ;;
    *)
      echo "[ERROR] Unsupported model: $1" >&2
      return 1
      ;;
  esac
}

rate_dir_value() {
  printf "%.3f" "$1"
}

eps_dir_value() {
  printf "%.1f" "$1"
}

poison_dir() {
  local model="$1"
  local rate="$2"
  local eps="$3"
  local arch
  arch="$(arch_name "$model")" || return 1
  echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_$(rate_dir_value "$rate")_eps=$(eps_dir_value "$eps")_constraint=${UPGD_CONSTRAINT}_steps=${UPGD_STEPS}_mode=${LABEL_MODE}_mult=${UPGD_STEPS_MULTIPLIER}_poison_seed=2333_arch=${arch}"
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

shared_args() {
  local model="$1"
  local rate="$2"
  local eps="$3"
  echo "-dataset=${DATASET} -model=${model} -devices=${DEVICES} -poison_type=upgd -poison_rate=${rate} -eps=${eps} -constraint=${UPGD_CONSTRAINT} -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER} -label_mode=${LABEL_MODE}"
}

echo "============================================================"
echo "Backfill Tiny-ImageNet baseline UPGD transfer tests"
echo "============================================================"
echo "output_root       : ${POISONED_TRAIN_SET_ROOT}"
echo "models            : ${MODELS}"
echo "poison_rates      : ${POISON_RATES}"
echo "eps_values        : ${EPS_VALUES}"
echo "target_domain     : ${RUN_TARGET_DOMAIN} (${TARGET_DOMAIN_DIR})"
echo "qwen_transfer     : ${RUN_QWEN_TRANSFER} (${TARGET_DOMAIN_QWEN_DIR})"
echo "force             : ${FORCE}"
echo "dry_run           : ${DRY_RUN}"
echo "error_log         : ${ERROR_LOG}"
echo "============================================================"

for model in ${MODELS}; do
  for rate in ${POISON_RATES}; do
    for eps in ${EPS_VALUES}; do
      dir="$(poison_dir "$model" "$rate" "$eps")" || exit 2
      args="$(shared_args "$model" "$rate" "$eps")"

      echo
      echo "========== ${model}: rate=${rate}, eps=${eps} =========="
      echo "dir: ${dir}"

      if [ ! -d "$dir" ]; then
        echo "[WARN] poison dir not found, command may fail if model is missing: ${dir}"
      fi

      if [ "$RUN_TARGET_DOMAIN" = "1" ]; then
        if skip_file "${dir}/test_tiny_target_domain_results.txt" "target-domain transfer results"; then
          :
        else
          run_command \
            "${PYTHON_BIN} test_tiny_target_domain.py ${args} -source_dataset=${DATASET} -target_domain_dir=${TARGET_DOMAIN_DIR}" \
            "Tiny target-domain transfer ${model} rate=${rate} eps=${eps}"
        fi
      fi

      if [ "$RUN_QWEN_TRANSFER" = "1" ]; then
        if skip_file "${dir}/test_tiny_target_domain_qwen_results.txt" "Qwen target-domain transfer results"; then
          :
        else
          run_command \
            "${PYTHON_BIN} test_tiny_target_domain_qwen.py ${args} -source_dataset=${DATASET} -target_domain_dir=${TARGET_DOMAIN_QWEN_DIR}" \
            "Tiny Qwen target-domain transfer ${model} rate=${rate} eps=${eps}"
        fi
      fi
    done
  done
done

echo
echo "============================================================"
echo "Backfill finished. Failed commands, if any: ${ERROR_LOG}"
echo "============================================================"
