#!/usr/bin/env bash

# Rerun CIFAR-10 SmallCNN input-noise UPGD all-to-one experiments with a
# raw-input clean base. This replaces the old UPGD delta generation that used
# the normal clean baseline trained with Normalize.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-run_cifar10_small_cnn_noise_upgd_all2one_raw_base}"
DATASET="cifar10"
MODEL="small_cnn"
DEVICES="${DEVICES:-0}"
LABEL_MODE="all2one"
INPUT_NOISE_SEED="${INPUT_NOISE_SEED:-2333}"

DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SKIP_UPGD_PREP="${SKIP_UPGD_PREP:-0}"
FORCE_RAW_BASE="${FORCE_RAW_BASE:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

POISON_RATES="${POISON_RATES:-0.01 0.005}"
EPS_VALUES="${EPS_VALUES:-4 8 12}"
NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-gaussian uniform}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
UPGD_RAW_BASE_DIR="${UPGD_RAW_BASE_DIR:-poisoned_train_set/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10}"
UPGD_CLEAN_MODEL_PATH="${UPGD_CLEAN_MODEL_PATH:-${UPGD_RAW_BASE_DIR}/upgd_raw_base_SmallCNN_cifar10.pt}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log}"

read -r -a POISON_RATE_LIST <<< "$POISON_RATES"
read -r -a EPS_LIST <<< "$EPS_VALUES"
read -r -a NOISE_TYPES <<< "$NOISE_TYPE_FILTER"

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

noise_levels() {
  case "$1" in
    "gaussian") echo "${GAUSSIAN_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    "uniform") echo "${UNIFORM_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    *)
      echo "Unsupported noise type: $1" >&2
      return 1
      ;;
  esac
}

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

upgd_args() {
  echo "-poison_type=upgd -label_mode=${LABEL_MODE} -constraint=Linf -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER}"
}

noise_args() {
  echo "-input_noise_type=$1 -input_noise_level=$2 -input_noise_seed=${INPUT_NOISE_SEED}"
}

echo "============================================================"
echo "CIFAR-10 SmallCNN noise UPGD all-to-one raw-base rerun"
echo "============================================================"
echo "poison rates : ${POISON_RATE_LIST[*]}"
echo "eps values   : ${EPS_LIST[*]}"
echo "noise types  : ${NOISE_TYPES[*]}"
echo "label_mode   : ${LABEL_MODE}"
echo "raw base     : ${UPGD_CLEAN_MODEL_PATH}"
echo "defenses     : ${DEFENSES}"
echo "dry run      : ${DRY_RUN}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "$SKIP_UPGD_PREP" != "1" ]; then
  echo
  echo "----- 0. UPGD raw clean base preparation -----"
  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
    "Create clean set for raw UPGD base"

  if [ "$FORCE_RAW_BASE" = "1" ] || [ ! -f "$UPGD_CLEAN_MODEL_PATH" ]; then
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${UPGD_CLEAN_MODEL_PATH}" \
      "Train raw-input clean base model for UPGD"
  else
    echo "[SKIP] Raw clean base exists: ${UPGD_CLEAN_MODEL_PATH}"
  fi
fi

for noise_type in "${NOISE_TYPES[@]}"; do
  for noise_level in $(noise_levels "$noise_type"); do
    noise="$(noise_args "$noise_type" "$noise_level")"
    for rate in "${POISON_RATE_LIST[@]}"; do
      for eps in "${EPS_LIST[@]}"; do
        shared="$(base_args) $(upgd_args) -poison_rate=${rate} -eps=${eps} ${noise}"

        if [ "$RUN_CREATE" = "1" ]; then
          run_command \
            "${PYTHON_BIN} create_poisoned_set.py ${shared} -upgd_model_path=${UPGD_CLEAN_MODEL_PATH}" \
            "Create UPGD all-to-one raw-base set: noise=${noise_type}/${noise_level} rate=${rate} eps=${eps}"
        fi

        if [ "$RUN_TRAIN" = "1" ]; then
          run_command \
            "${PYTHON_BIN} train_on_poisoned_set.py ${shared}" \
            "Train UPGD all-to-one model: noise=${noise_type}/${noise_level} rate=${rate} eps=${eps}"
        fi

        if [ "$RUN_TEST" = "1" ]; then
          run_command \
            "${PYTHON_BIN} test_model.py ${shared}" \
            "Source test UPGD all-to-one: noise=${noise_type}/${noise_level} rate=${rate} eps=${eps}"
        fi

        if [ "$RUN_TRANSFER" = "1" ]; then
          run_command \
            "${PYTHON_BIN} test_stl10.py ${shared}" \
            "STL-10 transfer UPGD all-to-one: noise=${noise_type}/${noise_level} rate=${rate} eps=${eps}"
        fi

        if [ "$RUN_DEFENSES" = "1" ]; then
          for defense in ${DEFENSES}; do
            run_command \
              "${PYTHON_BIN} other_defense.py -defense=${defense} ${shared}" \
              "Defense ${defense} UPGD all-to-one: noise=${noise_type}/${noise_level} rate=${rate} eps=${eps}"
          done
        fi
      done
    done
  done
done

echo
echo "============================================================"
echo "Noise UPGD all-to-one raw-base rerun finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
