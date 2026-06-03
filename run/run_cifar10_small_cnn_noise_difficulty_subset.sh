#!/usr/bin/env bash

# CIFAR-10 SmallCNN input-noise difficulty experiment.
#
# Complete but non-exhaustive pipeline across eight attack methods and four
# stealth/detection defenses. Noise is applied before the trigger when creating
# poisoned CIFAR-10 training sets.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-run_cifar10_small_cnn_noise_difficulty_subset}"
TITLE="${RUN_TITLE:-CIFAR-10 SmallCNN gaussian/uniform input-noise difficulty subset experiment}"
DATASET="cifar10"
MODEL="small_cnn"
TRANSFER_SCRIPT="test_stl10.py"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
INPUT_NOISE_SEED="${INPUT_NOISE_SEED:-2333}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

POISON_RATES=("0.01" "0.005")

ATTACKS=(
  "badnet"
  "blend"
  "SIG"
  "WaNet"
  "adaptive_patch"
  "adaptive_blend"
  "belt"
  "upgd"
)

NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-gaussian uniform}"
read -r -a NOISE_TYPES <<< "$NOISE_TYPE_FILTER"

DEFENSES=(
  "SentiNet"
  "STRIP"
  "ScaleUp"
  "IBD_PSC"
)

UPGD_STEPS="100"
UPGD_STEPS_MULTIPLIER="5"
UPGD_CLEAN_MODEL_PATH="poisoned_train_set/${DATASET}/none_0.000_poison_seed=2333_arch=SmallCNN_cifar10/SmallCNN_cifar10.pt"

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

double_cover_rate() {
  case "$1" in
    "0.05") echo "0.1" ;;
    "0.01") echo "0.02" ;;
    "0.005") echo "0.01" ;;
    "0.001") echo "0.002" ;;
    *)
      echo "Unsupported poison rate for fixed subset cover-rate mapping: $1" >&2
      return 1
      ;;
  esac
}

noise_levels() {
  case "$1" in
    "none") echo "0.000" ;;
    "gaussian") echo "0.030 0.060 0.100" ;;
    "uniform") echo "0.030 0.060 0.100" ;;
    "salt_pepper") echo "0.010 0.030 0.050" ;;
    "speckle") echo "0.030 0.060 0.100" ;;
    *)
      echo "Unsupported noise type: $1" >&2
      return 1
      ;;
  esac
}

strength_values() {
  case "$1" in
    "badnet") echo "0.2 0.5 1.0" ;;
    "blend") echo "0.05 0.15 0.3" ;;
    "SIG") echo "20 30 40" ;;
    "WaNet") echo "0.4 0.5 0.8" ;;
    "adaptive_patch") echo "0.1 0.2 0.3" ;;
    "adaptive_blend") echo "0.05 0.15 0.25" ;;
    "belt") echo "0.1 0.2 0.3" ;;
    "upgd") echo "4 8 12" ;;
    *)
      echo "Unsupported attack: $1" >&2
      return 1
      ;;
  esac
}

attack_args() {
  local attack="$1"
  local rate="$2"
  local strength="$3"

  case "$attack" in
    "badnet"|"blend")
      echo "-alpha ${strength}"
      ;;
    "SIG")
      echo "-f 6 -delta ${strength}"
      ;;
    "WaNet")
      echo "-cover_rate $(double_cover_rate "$rate") -s ${strength} -k 4"
      ;;
    "adaptive_patch")
      echo "-cover_rate $(double_cover_rate "$rate") -alpha ${strength}"
      ;;
    "adaptive_blend")
      echo "-cover_rate $rate -alpha ${strength}"
      ;;
    "belt")
      echo "-cover_rate 0.5 -mask_rate ${strength} -alpha 1.0"
      ;;
    "upgd")
      echo "-eps ${strength} -constraint Linf -upgd_steps ${UPGD_STEPS} -upgd_steps_multiplier ${UPGD_STEPS_MULTIPLIER}"
      ;;
    *)
      echo "Unsupported attack: $attack" >&2
      return 1
      ;;
  esac
}

strength_label() {
  case "$1" in
    "badnet"|"blend"|"adaptive_patch"|"adaptive_blend") echo "alpha=$2" ;;
    "SIG") echo "delta=$2" ;;
    "WaNet") echo "s=$2" ;;
    "belt") echo "mask_rate=$2" ;;
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
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

echo
echo "----- 0. UPGD clean base model preparation -----"
run_command \
  "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
  "Create clean set for UPGD base model"
run_command \
  "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
  "Train clean base model for UPGD"

echo
echo "----- 1. Create poisoned datasets -----"
for noise_type in "${NOISE_TYPES[@]}"; do
  for noise_level in $(noise_levels "$noise_type"); do
    noise="$(noise_args "$noise_type" "$noise_level")"
    for attack in "${ATTACKS[@]}"; do
      for rate in "${POISON_RATES[@]}"; do
        for strength in $(strength_values "$attack"); do
          args="$(attack_args "$attack" "$rate" "$strength")"
          if [ "$attack" = "upgd" ]; then
            args="${args} -upgd_model_path ${UPGD_CLEAN_MODEL_PATH}"
          fi
          label="$(strength_label "$attack" "$strength")"
          run_command \
            "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
            "Create poisoned set: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
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
          args="$(attack_args "$attack" "$rate" "$strength")"
          label="$(strength_label "$attack" "$strength")"
          run_command \
            "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
            "Train model: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
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
          args="$(attack_args "$attack" "$rate" "$strength")"
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
          args="$(attack_args "$attack" "$rate" "$strength")"
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
            args="$(attack_args "$attack" "$rate" "$strength")"
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
echo "Noise difficulty subset pipeline finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
