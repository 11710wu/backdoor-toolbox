#!/usr/bin/env bash

# CIFAR-10 SmallCNN input-noise clean-label SIG/UPGD experiment.
#
# This script is intentionally limited to SIG and UPGD because label_mode only
# changes the training-set semantics for these two attacks.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-run_cifar10_small_cnn_noise_sig_upgd_clean_label}"
TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN input-noise clean-label SIG/UPGD experiment}"
DATASET="cifar10"
MODEL="small_cnn"
TRANSFER_SCRIPT="test_stl10.py"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SKIP_UPGD_PREP="${SKIP_UPGD_PREP:-0}"
INPUT_NOISE_SEED="${INPUT_NOISE_SEED:-2333}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

POISON_RATES=("0.01" "0.005")
ATTACKS=("SIG" "upgd")
NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-gaussian uniform}"
read -r -a NOISE_TYPES <<< "$NOISE_TYPE_FILTER"
DEFENSES=("SentiNet" "STRIP" "ScaleUp" "IBD_PSC")

UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
# UPGD uses a raw-input base model for delta generation, not the normal clean baseline.
UPGD_RAW_BASE_DIR="${UPGD_RAW_BASE_DIR:-poisoned_train_set/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10}"
UPGD_CLEAN_MODEL_PATH="${UPGD_CLEAN_MODEL_PATH:-${UPGD_RAW_BASE_DIR}/upgd_raw_base_SmallCNN_cifar10.pt}"

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
    "gaussian") echo "0.030 0.060 0.100" ;;
    "uniform") echo "0.030 0.060 0.100" ;;
    *)
      echo "Unsupported noise type: $1" >&2
      return 1
      ;;
  esac
}

strength_values() {
  case "$1" in
    "SIG") echo "20 30 40" ;;
    "upgd") echo "4 8 12" ;;
    *)
      echo "Unsupported attack: $1" >&2
      return 1
      ;;
  esac
}

attack_args() {
  local attack="$1"
  local strength="$2"

  case "$attack" in
    "SIG")
      echo "-f 6 -delta ${strength} -label_mode clean"
      ;;
    "upgd")
      echo "-eps ${strength} -constraint Linf -upgd_steps ${UPGD_STEPS} -upgd_steps_multiplier ${UPGD_STEPS_MULTIPLIER} -label_mode clean"
      ;;
    *)
      echo "Unsupported attack: $attack" >&2
      return 1
      ;;
  esac
}

strength_label() {
  case "$1" in
    "SIG") echo "delta=$2" ;;
    "upgd") echo "eps=$2" ;;
    *) echo "strength=$2" ;;
  esac
}

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

noise_args() {
  echo "-input_noise_type=$1 -input_noise_level=$2 -input_noise_seed=${INPUT_NOISE_SEED}"
}

transfer_command() {
  local attack="$1"
  local rate="$2"
  local args="$3"
  local noise="$4"

  echo "${PYTHON_BIN} ${TRANSFER_SCRIPT} $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}"
}

echo "============================================================"
echo "${TITLE}"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "dataset      : ${DATASET}"
echo "model        : ${MODEL}"
echo "devices      : ${DEVICES}"
echo "poison rates : ${POISON_RATES[*]}"
echo "attacks      : ${ATTACKS[*]}"
echo "noise types  : ${NOISE_TYPES[*]}"
echo "defenses     : ${DEFENSES[*]}"
echo "transfer     : ${TRANSFER_SCRIPT}"
echo "upgd clean   : ${UPGD_CLEAN_MODEL_PATH}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "$SKIP_UPGD_PREP" != "1" ]; then
  echo
  echo "----- 0. UPGD clean base model preparation -----"
  if [ -f "$UPGD_CLEAN_MODEL_PATH" ]; then
    echo "UPGD clean base model already exists: ${UPGD_CLEAN_MODEL_PATH}"
  else
    run_command \
      "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
      "Create clean set for UPGD base model"
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${UPGD_CLEAN_MODEL_PATH}" \
      "Train raw-input clean base model for UPGD"
  fi
fi

echo
echo "----- 1. Create poisoned datasets -----"
for noise_type in "${NOISE_TYPES[@]}"; do
  for noise_level in $(noise_levels "$noise_type"); do
    noise="$(noise_args "$noise_type" "$noise_level")"
    for attack in "${ATTACKS[@]}"; do
      for rate in "${POISON_RATES[@]}"; do
        for strength in $(strength_values "$attack"); do
          args="$(attack_args "$attack" "$strength")"
          if [ "$attack" = "upgd" ]; then
            args="${args} -upgd_model_path ${UPGD_CLEAN_MODEL_PATH}"
          fi
          label="$(strength_label "$attack" "$strength")"
          run_command \
            "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
            "Create clean-label poisoned set: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
        done
      done
    done
  done
done

echo
echo "----- 2. Train poisoned models -----"
for noise_type in "${NOISE_TYPES[@]}"; do
  for noise_level in $(noise_levels "$noise_type"); do
    noise="$(noise_args "$noise_type" "$noise_level")"
    for attack in "${ATTACKS[@]}"; do
      for rate in "${POISON_RATES[@]}"; do
        for strength in $(strength_values "$attack"); do
          args="$(attack_args "$attack" "$strength")"
          label="$(strength_label "$attack" "$strength")"
          run_command \
            "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
            "Train clean-label model: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
        done
      done
    done
  done
done

echo
echo "----- 3. Source-domain testing -----"
for noise_type in "${NOISE_TYPES[@]}"; do
  for noise_level in $(noise_levels "$noise_type"); do
    noise="$(noise_args "$noise_type" "$noise_level")"
    for attack in "${ATTACKS[@]}"; do
      for rate in "${POISON_RATES[@]}"; do
        for strength in $(strength_values "$attack"); do
          args="$(attack_args "$attack" "$strength")"
          label="$(strength_label "$attack" "$strength")"
          run_command \
            "${PYTHON_BIN} test_model.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
            "Source test: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
        done
      done
    done
  done
done

echo
echo "----- 4. Transfer testing -----"
for noise_type in "${NOISE_TYPES[@]}"; do
  for noise_level in $(noise_levels "$noise_type"); do
    noise="$(noise_args "$noise_type" "$noise_level")"
    for attack in "${ATTACKS[@]}"; do
      for rate in "${POISON_RATES[@]}"; do
        for strength in $(strength_values "$attack"); do
          args="$(attack_args "$attack" "$strength")"
          label="$(strength_label "$attack" "$strength")"
          run_command \
            "$(transfer_command "$attack" "$rate" "$args" "$noise")" \
            "Transfer test: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
        done
      done
    done
  done
done

echo
echo "----- 5. Stealth/detection defenses -----"
for defense in "${DEFENSES[@]}"; do
  echo
  echo "----- Defense: ${defense} -----"
  for noise_type in "${NOISE_TYPES[@]}"; do
    for noise_level in $(noise_levels "$noise_type"); do
      noise="$(noise_args "$noise_type" "$noise_level")"
      for attack in "${ATTACKS[@]}"; do
        for rate in "${POISON_RATES[@]}"; do
          for strength in $(strength_values "$attack"); do
            args="$(attack_args "$attack" "$strength")"
            label="$(strength_label "$attack" "$strength")"
            run_command \
              "${PYTHON_BIN} other_defense.py $(base_args) -defense=${defense} -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
              "Defense ${defense}: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
          done
        done
      done
    done
  done
done

echo
echo "============================================================"
echo "Input-noise clean-label SIG/UPGD pipeline finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
