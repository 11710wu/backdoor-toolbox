#!/usr/bin/env bash

# CIFAR-10 input-noise backfill with baseline-matched attack names/strengths.
#
# This script intentionally uses strict baseline-compatible configs only:
# - old noise `badnet` should be rerun as `basic`;
# - old unmatched strengths are replaced by strengths that exist in the
#   original CIFAR-10 ResNet18 baseline.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-backfill_cifar10_noise_matched_common}"
TITLE="${RUN_TITLE:-CIFAR-10 matched input-noise backfill}"
DATASET="cifar10"
MODEL="${MODEL:?Set MODEL to small_cnn or resnet18}"
TRANSFER_SCRIPT="${TRANSFER_SCRIPT:-test_stl10.py}"
DEVICES="${DEVICES:-0}"
INPUT_NOISE_SEED="${INPUT_NOISE_SEED:-2333}"
NOISE_TYPE_FILTER="${NOISE_TYPE_FILTER:?Set NOISE_TYPE_FILTER, e.g. 'gaussian uniform'}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log}"

read -r -a NOISE_TYPES <<< "$NOISE_TYPE_FILTER"

# Format: attack|poison_rate|extra_args|label
# Unmatched-only replacement set:
# - old noise `badnet` is rerun as baseline-compatible `basic`;
# - old off-grid blend/adaptive_blend/WaNet strengths are replaced by matched
#   baseline-grid strengths;
# - old BELT mask-axis rows are replaced by alpha-axis rows with mask_rate=0.2.
CONFIGS=(
  "basic|0.005|-alpha 0.2|alpha=0.2"
  "basic|0.005|-alpha 0.5|alpha=0.5"
  "basic|0.005|-alpha 1.0|alpha=1.0"
  "basic|0.01|-alpha 0.2|alpha=0.2"
  "basic|0.01|-alpha 0.5|alpha=0.5"
  "basic|0.01|-alpha 1.0|alpha=1.0"

  "blend|0.005|-alpha 0.10|alpha=0.10"

  "adaptive_blend|0.01|-cover_rate 0.01 -alpha 0.10|alpha=0.10,cover=0.01"

  "WaNet|0.005|-cover_rate 0.01 -s 0.6 -k 4|s=0.6,cover=0.01"
  "WaNet|0.005|-cover_rate 0.01 -s 1.0 -k 4|s=1.0,cover=0.01"
  "WaNet|0.01|-cover_rate 0.02 -s 0.6 -k 4|s=0.6,cover=0.02"

  "belt|0.01|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.10|alpha=0.10,mask_rate=0.2,cover=0.5"
  "belt|0.01|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.20|alpha=0.20,mask_rate=0.2,cover=0.5"
  "belt|0.01|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.30|alpha=0.30,mask_rate=0.2,cover=0.5"
  "belt|0.02|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.10|alpha=0.10,mask_rate=0.2,cover=0.5"
  "belt|0.02|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.20|alpha=0.20,mask_rate=0.2,cover=0.5"
  "belt|0.02|-cover_rate 0.5 -mask_rate 0.2 -alpha 0.30|alpha=0.30,mask_rate=0.2,cover=0.5"
)

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
    "salt_pepper") echo "${SALT_PEPPER_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    "speckle") echo "${SPECKLE_NOISE_LEVELS:-0.030 0.060 0.100}" ;;
    *)
      echo "Unsupported noise type: $1" >&2
      return 1
      ;;
  esac
}

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

noise_args() {
  echo "-input_noise_type=$1 -input_noise_level=$2 -input_noise_seed=${INPUT_NOISE_SEED}"
}

run_config_phase() {
  local phase_name="$1"
  local script_name="$2"
  local desc_prefix="$3"
  local extra_prefix="$4"

  echo
  echo "----- ${phase_name} -----"
  for noise_type in "${NOISE_TYPES[@]}"; do
    for noise_level in $(noise_levels "$noise_type"); do
      noise="$(noise_args "$noise_type" "$noise_level")"
      for config in "${CONFIGS[@]}"; do
        IFS='|' read -r attack rate args label <<< "$config"
        run_command \
          "${PYTHON_BIN} ${script_name} $(base_args) ${extra_prefix} -poison_type=${attack} -poison_rate=${rate} ${args} ${noise}" \
          "${desc_prefix}: model=${MODEL}, noise=${noise_type}/${noise_level}, ${attack}, poison_rate=${rate}, ${label}"
      done
    done
  done
}

echo "============================================================"
echo "${TITLE}"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "repo         : ${REPO_ROOT}"
echo "dataset      : ${DATASET}"
echo "model        : ${MODEL}"
echo "devices      : ${DEVICES}"
echo "noise types  : ${NOISE_TYPES[*]}"
echo "noise levels : gaussian=${GAUSSIAN_NOISE_LEVELS:-0.030 0.060 0.100}; uniform=${UNIFORM_NOISE_LEVELS:-0.030 0.060 0.100}; salt_pepper=${SALT_PEPPER_NOISE_LEVELS:-0.030 0.060 0.100}; speckle=${SPECKLE_NOISE_LEVELS:-0.030 0.060 0.100}"
echo "configs      : ${#CONFIGS[@]} unmatched replacement configs per noise level"
echo "defenses     : ${DEFENSES}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "$RUN_CREATE" = "1" ]; then
  run_config_phase "1. Create poisoned datasets" "create_poisoned_set.py" "Create poisoned set" ""
fi

if [ "$RUN_TRAIN" = "1" ]; then
  run_config_phase "2. Train poisoned models" "train_on_poisoned_set.py" "Train model" ""
fi

if [ "$RUN_TEST" = "1" ]; then
  run_config_phase "3. Source-domain testing" "test_model.py" "Source test" ""
fi

if [ "$RUN_TRANSFER" = "1" ]; then
  run_config_phase "4. STL-10 transfer testing" "${TRANSFER_SCRIPT}" "STL-10 transfer test" ""
fi

if [ "$RUN_DEFENSES" = "1" ]; then
  echo
  echo "----- 5. Stealth/detection defenses -----"
  for defense in ${DEFENSES}; do
    run_config_phase "Defense: ${defense}" "other_defense.py" "Defense ${defense}" "-defense=${defense}"
  done
fi

echo
echo "============================================================"
echo "Matched CIFAR-10 noise backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
