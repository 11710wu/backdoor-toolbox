#!/usr/bin/env bash

# CIFAR-10 ResNet18 input-noise experiment.
#
# This combines eight attack methods into one ResNet18-only pipeline. Use the
# per-noise wrapper scripts when you want the four noise types to run separately.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-run_cifar10_resnet18_noise_eight_methods}"
TITLE="${RUN_TITLE:-CIFAR-10 ResNet18 input-noise eight-method experiment}"
DATASET="cifar10"
MODEL="resnet18"
ARCH_NAME="ResNet18_cifar10"
TRANSFER_SCRIPT="test_stl10.py"
DEVICES="${DEVICES:-0}"
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
NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:-gaussian uniform salt_pepper speckle}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

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

UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
UPGD_RAW_BASE_DIR="${UPGD_RAW_BASE_DIR:-poisoned_train_set/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=${ARCH_NAME}}"
UPGD_CLEAN_MODEL_PATH="${UPGD_CLEAN_MODEL_PATH:-${UPGD_RAW_BASE_DIR}/upgd_raw_base_${ARCH_NAME}.pt}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log}"

read -r -a POISON_RATE_LIST <<< "$POISON_RATES"
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

double_cover_rate() {
  case "$1" in
    "0.05") echo "0.1" ;;
    "0.01") echo "0.02" ;;
    "0.005") echo "0.01" ;;
    "0.001") echo "0.002" ;;
    *)
      echo "Unsupported poison rate for cover-rate mapping: $1" >&2
      return 1
      ;;
  esac
}

noise_levels() {
  case "$1" in
    "gaussian") echo "${GAUSSIAN_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    "uniform") echo "${UNIFORM_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    "salt_pepper") echo "${SALT_PEPPER_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    "speckle") echo "${SPECKLE_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    *)
      echo "Unsupported noise type: $1" >&2
      return 1
      ;;
  esac
}

strength_values() {
  case "$1" in
    "badnet") echo "${BADNET_ALPHAS:-0.2 0.5 1.0}" ;;
    "blend") echo "${BLEND_ALPHAS:-0.05 0.15 0.3}" ;;
    "SIG") echo "${SIG_DELTAS:-20 30 40}" ;;
    "WaNet") echo "${WANET_S_VALUES:-0.4 0.5 0.8}" ;;
    "adaptive_patch") echo "${ADAPTIVE_PATCH_ALPHAS:-0.1 0.2 0.3}" ;;
    "adaptive_blend") echo "${ADAPTIVE_BLEND_ALPHAS:-0.05 0.15 0.25}" ;;
    "belt") echo "${BELT_MASK_RATES:-0.1 0.2 0.3}" ;;
    "upgd") echo "${UPGD_EPS_VALUES:-4 8 12}" ;;
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
      echo "-f 6 -delta ${strength} -label_mode all2one"
      ;;
    "WaNet")
      echo "-cover_rate $(double_cover_rate "$rate") -s ${strength} -k 4"
      ;;
    "adaptive_patch")
      echo "-cover_rate $(double_cover_rate "$rate") -alpha ${strength}"
      ;;
    "adaptive_blend")
      echo "-cover_rate ${rate} -alpha ${strength}"
      ;;
    "belt")
      echo "-cover_rate 0.5 -mask_rate ${strength} -alpha 1.0"
      ;;
    "upgd")
      echo "-eps ${strength} -constraint Linf -upgd_steps ${UPGD_STEPS} -upgd_steps_multiplier ${UPGD_STEPS_MULTIPLIER} -label_mode all2one"
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

echo "============================================================"
echo "${TITLE}"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "dataset      : ${DATASET}"
echo "model        : ${MODEL}"
echo "devices      : ${DEVICES}"
echo "output root  : poisoned_train_set/${DATASET}"
echo "poison rates : ${POISON_RATE_LIST[*]}"
echo "attacks      : ${ATTACKS[*]}"
echo "noise types  : ${NOISE_TYPES[*]}"
echo "noise levels : gaussian=${GAUSSIAN_NOISE_LEVELS:-0.030 0.060 0.100}; uniform=${UNIFORM_NOISE_LEVELS:-0.030 0.060 0.100}; salt_pepper=${SALT_PEPPER_NOISE_LEVELS:-0.030 0.060 0.100}; speckle=${SPECKLE_NOISE_LEVELS:-0.030 0.060 0.100}"
echo "noise seed   : ${INPUT_NOISE_SEED}"
echo "label mode   : all2one for SIG/UPGD"
echo "transfer     : ${TRANSFER_SCRIPT}"
echo "defenses     : ${DEFENSES}"
echo "upgd raw base: ${UPGD_CLEAN_MODEL_PATH}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "$SKIP_UPGD_PREP" != "1" ]; then
  echo
  echo "----- 0. UPGD raw clean base preparation -----"
  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
    "Create clean set for UPGD raw base"

  if [ "$FORCE_RAW_BASE" = "1" ] || [ ! -f "$UPGD_CLEAN_MODEL_PATH" ]; then
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${UPGD_CLEAN_MODEL_PATH}" \
      "Train raw-input clean base model for UPGD"
  else
    echo "[SKIP] Raw clean base exists: ${UPGD_CLEAN_MODEL_PATH}"
  fi
fi

echo
echo "----- 1. Create poisoned datasets -----"
if [ "$RUN_CREATE" = "1" ]; then
  for noise_type in "${NOISE_TYPES[@]}"; do
    for noise_level in $(noise_levels "$noise_type"); do
      noise="$(noise_args "$noise_type" "$noise_level")"
      for attack in "${ATTACKS[@]}"; do
        for rate in "${POISON_RATE_LIST[@]}"; do
          for strength in $(strength_values "$attack"); do
            args="$(attack_args "$attack" "$rate" "$strength")"
            if [ "$attack" = "upgd" ]; then
              args="${args} -upgd_model_path=${UPGD_CLEAN_MODEL_PATH}"
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
fi

echo
echo "----- 2. Train poisoned models -----"
if [ "$RUN_TRAIN" = "1" ]; then
  for noise_type in "${NOISE_TYPES[@]}"; do
    for noise_level in $(noise_levels "$noise_type"); do
      noise="$(noise_args "$noise_type" "$noise_level")"
      for attack in "${ATTACKS[@]}"; do
        for rate in "${POISON_RATE_LIST[@]}"; do
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
fi

echo
echo "----- 3. Source-domain testing -----"
if [ "$RUN_TEST" = "1" ]; then
  for noise_type in "${NOISE_TYPES[@]}"; do
    for noise_level in $(noise_levels "$noise_type"); do
      noise="$(noise_args "$noise_type" "$noise_level")"
      for attack in "${ATTACKS[@]}"; do
        for rate in "${POISON_RATE_LIST[@]}"; do
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
fi

echo
echo "----- 4. STL-10 transfer testing -----"
if [ "$RUN_TRANSFER" = "1" ]; then
  for noise_type in "${NOISE_TYPES[@]}"; do
    for noise_level in $(noise_levels "$noise_type"); do
      noise="$(noise_args "$noise_type" "$noise_level")"
      for attack in "${ATTACKS[@]}"; do
        for rate in "${POISON_RATE_LIST[@]}"; do
          for strength in $(strength_values "$attack"); do
            args="$(attack_args "$attack" "$rate" "$strength")"
            label="$(strength_label "$attack" "$strength")"
            run_command \
              "${PYTHON_BIN} ${TRANSFER_SCRIPT} $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
              "STL-10 transfer test: noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
          done
        done
      done
    done
  done
fi

echo
echo "----- 5. Stealth/detection defenses -----"
if [ "$RUN_DEFENSES" = "1" ]; then
  for defense in ${DEFENSES}; do
    echo
    echo "----- Defense: ${defense} -----"
    for noise_type in "${NOISE_TYPES[@]}"; do
      for noise_level in $(noise_levels "$noise_type"); do
        noise="$(noise_args "$noise_type" "$noise_level")"
        for attack in "${ATTACKS[@]}"; do
          for rate in "${POISON_RATE_LIST[@]}"; do
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
fi

echo
echo "============================================================"
echo "CIFAR-10 ResNet18 noise eight-method pipeline finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
