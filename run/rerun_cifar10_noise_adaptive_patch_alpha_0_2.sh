#!/usr/bin/env bash

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DATASET="cifar10"
ALPHA="${ALPHA:-0.2}"
INPUT_NOISE_SEED="${INPUT_NOISE_SEED:-2333}"
CLEAN_OLD="${CLEAN_OLD:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

MODELS="${MODELS:-resnet18 small_cnn}"
NOISE_TYPES="${NOISE_TYPES:-gaussian salt_pepper speckle uniform}"
NOISE_LEVELS="${NOISE_LEVELS:-0.030 0.060 0.100}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_cifar10_noise_adaptive_patch_alpha_0_2_${TIMESTAMP}.log}"

read -r -a MODEL_LIST <<< "$MODELS"
read -r -a NOISE_TYPE_LIST <<< "$NOISE_TYPES"
read -r -a NOISE_LEVEL_LIST <<< "$NOISE_LEVELS"
read -r -a DEFENSE_LIST <<< "$DEFENSES"

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

poison_cover_pairs() {
  echo "0.01 0.02"
  echo "0.005 0.01"
}

safe_cleanup_old_alpha() {
  echo
  echo "----- Safe cleanup: CIFAR-10 noise adaptive_patch alpha=0.200 -----"
  local base="poisoned_train_set/${DATASET}"
  local paths=()
  local path name

  [ -d "$base" ] || return 0
  for path in "$base"/adaptive_patch_*_alpha=0.200_*_noise=*; do
    [ -d "$path" ] || continue
    name="$(basename "$path")"
    case "$path" in
      poisoned_train_set/${DATASET}/adaptive_patch_*_alpha=0.200_*_noise=*) ;;
      *)
        echo "Refusing suspicious path: ${path}" >&2
        return 1
        ;;
    esac
    case "$name" in
      adaptive_patch_*_alpha=0.200_*_noise=*) paths+=("$path") ;;
      *)
        echo "Refusing suspicious directory name: ${name}" >&2
        return 1
        ;;
    esac
  done

  echo "Matched ${#paths[@]} directories:"
  printf '  %s\n' "${paths[@]}"

  if [ "$CLEAN_OLD" != "1" ]; then
    echo "[SKIP] CLEAN_OLD=${CLEAN_OLD}"
    return 0
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] cleanup skipped"
    return 0
  fi

  for path in "${paths[@]}"; do
    rm -rf -- "$path"
  done
}

base_args() {
  local model="$1"
  local rate="$2"
  local cover="$3"
  echo "-dataset=${DATASET} -poison_type=adaptive_patch -poison_rate=${rate} -cover_rate=${cover} -alpha=${ALPHA} -model=${model} -devices=${DEVICES}"
}

noise_args() {
  local noise_type="$1"
  local noise_level="$2"
  echo "-input_noise_type=${noise_type} -input_noise_level=${noise_level} -input_noise_seed=${INPUT_NOISE_SEED}"
}

echo "============================================================"
echo "CIFAR-10 input-noise Adaptive-Patch alpha=0.2 rerun"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "dataset      : ${DATASET}"
echo "models       : ${MODEL_LIST[*]}"
echo "alpha        : ${ALPHA}"
echo "devices      : ${DEVICES}"
echo "noise types  : ${NOISE_TYPE_LIST[*]}"
echo "noise levels : ${NOISE_LEVEL_LIST[*]}"
echo "noise seed   : ${INPUT_NOISE_SEED}"
echo "defenses     : ${DEFENSE_LIST[*]}"
echo "clean old    : ${CLEAN_OLD}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

safe_cleanup_old_alpha || exit 1

for model in "${MODEL_LIST[@]}"; do
  for noise_type in "${NOISE_TYPE_LIST[@]}"; do
    for noise_level in "${NOISE_LEVEL_LIST[@]}"; do
      noise="$(noise_args "$noise_type" "$noise_level")"
      echo
      echo "===== Model=${model}, noise=${noise_type}/${noise_level} ====="

      if [ "$RUN_CREATE" = "1" ]; then
        echo "----- 1. Creation -----"
        while read -r rate cover; do
          args="$(base_args "$model" "$rate" "$cover")"
          run_command "${PYTHON_BIN} create_poisoned_set.py ${args} ${noise}" \
            "Create: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model}) noise=${noise_type}/${noise_level}"
        done < <(poison_cover_pairs)
      fi

      if [ "$RUN_TRAIN" = "1" ]; then
        echo "----- 2. Training -----"
        while read -r rate cover; do
          args="$(base_args "$model" "$rate" "$cover")"
          run_command "${PYTHON_BIN} train_on_poisoned_set.py ${args} ${noise}" \
            "Train: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model}) noise=${noise_type}/${noise_level}"
        done < <(poison_cover_pairs)
      fi

      if [ "$RUN_TEST" = "1" ]; then
        echo "----- 3. Local Testing -----"
        while read -r rate cover; do
          args="$(base_args "$model" "$rate" "$cover")"
          run_command "${PYTHON_BIN} test_model.py ${args} ${noise}" \
            "Test: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model}) noise=${noise_type}/${noise_level}"
        done < <(poison_cover_pairs)
      fi

      if [ "$RUN_TRANSFER" = "1" ]; then
        echo "----- 4. Transfer Testing -----"
        while read -r rate cover; do
          args="$(base_args "$model" "$rate" "$cover")"
          run_command "${PYTHON_BIN} test_stl10.py ${args} ${noise}" \
            "Transfer: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model}) noise=${noise_type}/${noise_level}"
        done < <(poison_cover_pairs)
      fi

      if [ "$RUN_DEFENSES" = "1" ]; then
        echo "----- 5. Defenses -----"
        for defense in "${DEFENSE_LIST[@]}"; do
          while read -r rate cover; do
            args="$(base_args "$model" "$rate" "$cover")"
            run_command "${PYTHON_BIN} other_defense.py -defense=${defense} ${args} ${noise}" \
              "Defense ${defense}: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model}) noise=${noise_type}/${noise_level}"
          done < <(poison_cover_pairs)
        done
      fi
    done
  done
done

echo
echo "CIFAR-10 input-noise Adaptive-Patch alpha=0.2 rerun script finished."
