#!/usr/bin/env bash

# CIFAR-10 ResNet18 noise symmetry patch.
#
# Goal: align the noise grid (pr=0.005/0.01) with baseline-matched configs.
#
# Phase A - delete asymmetric configs (36 dirs = 3 configs x 12 noise conditions):
#   - WaNet      pr=0.005, s=1.0,  cover=0.01
#   - blend      pr=0.005, alpha=0.1
#   - adaptive_blend pr=0.01, alpha=0.1, cover=0.01
#
# Phase B - supplement missing baseline-aligned configs (72 dirs = 6 configs x 12):
#   - SIG            pr=0.005/0.01, delta=28
#   - SIG            pr=0.005/0.01, delta=36
#   - adaptive_patch pr=0.005, alpha=0.0, cover=0.01
#   - adaptive_patch pr=0.01,  alpha=0.0, cover=0.02
#
# Expected final grid: 48 attack configs x 4 noise types x 3 levels = 576 noise rows.
#
# Usage examples:
#   # Preview only
#   DRY_RUN=1 bash run/run_cifar10_resnet18_noise_symmetry_patch.sh
#
#   # Delete old asymmetric dirs, then run supplements
#   bash run/run_cifar10_resnet18_noise_symmetry_patch.sh
#
#   # Skip delete (already cleaned) and only supplement
#   RUN_DELETE=0 bash run/run_cifar10_resnet18_noise_symmetry_patch.sh
#
#   # Only delete, no training
#   RUN_SUPPLEMENT=0 bash run/run_cifar10_resnet18_noise_symmetry_patch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="cifar10"
MODEL="resnet18"
ARCH_NAME="ResNet18_cifar10"
TRANSFER_SCRIPT="test_stl10.py"
DEVICES="${DEVICES:-0}"
INPUT_NOISE_SEED="${INPUT_NOISE_SEED:-2333}"
SIG_UPGD_LABEL_MODE="${SIG_UPGD_LABEL_MODE:-clean}"
OUTPUT_ROOT="${OUTPUT_ROOT:-poisoned_train_set/${DATASET}}"

DRY_RUN="${DRY_RUN:-0}"
RUN_DELETE="${RUN_DELETE:-1}"
RUN_SUPPLEMENT="${RUN_SUPPLEMENT:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

NOISE_TYPES="${NOISE_TYPES:-gaussian uniform salt_pepper speckle}"
NOISE_LEVELS="${NOISE_LEVELS:-0.030 0.060 0.100}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
UPGD_RAW_BASE_DIR="${UPGD_RAW_BASE_DIR:-poisoned_train_set/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=${ARCH_NAME}}"
UPGD_CLEAN_MODEL_PATH="${UPGD_CLEAN_MODEL_PATH:-${UPGD_RAW_BASE_DIR}/upgd_raw_base_${ARCH_NAME}.pt}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/run_cifar10_resnet18_noise_symmetry_patch_${TIMESTAMP}.log}"

read -r -a NOISE_TYPE_LIST <<< "$NOISE_TYPES"
read -r -a NOISE_LEVEL_LIST <<< "$NOISE_LEVELS"

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

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

noise_args() {
  echo "-input_noise_type=$1 -input_noise_level=$2 -input_noise_seed=${INPUT_NOISE_SEED}"
}

result_exists() {
  local result_dir="$1"
  [ -d "${result_dir}" ] && [ -f "${result_dir}/train_results_seed=2333.json" ]
}

delete_dir_if_matches() {
  local dir_path="$1"
  local reason="$2"
  local name
  name="$(basename "${dir_path}")"

  case "${name}" in
    WaNet_0.005_cover=0.010_s=1_k=4_noise=*|blend_0.005_alpha=0.100_*_noise=*|adaptive_blend_0.010_alpha=0.100_cover=0.010_*_noise=*_arch=ResNet18_cifar10) ;;
    *)
      echo "[DELETE-SKIP] pattern mismatch: ${name}"
      return 0
      ;;
  esac

  if [[ "${name}" != *"_noise="* ]] || [[ "${name}" != *"arch=ResNet18"* ]]; then
    echo "[DELETE-SKIP] safety check failed: ${name}"
    return 0
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN DELETE] ${name} (${reason})"
    return 0
  fi

  rm -rf "${dir_path}"
  echo "[DELETED] ${name} (${reason})"
}

phase_delete_asymmetric_configs() {
  echo
  echo "============================================================"
  echo "Phase A: delete asymmetric ResNet18 noise configs"
  echo "============================================================"

  local deleted=0
  local candidate=0
  local dir_path

  shopt -s nullglob
  for dir_path in \
    "${OUTPUT_ROOT}"/WaNet_0.005_cover=0.010_s=1_k=4_noise=*_arch=ResNet18_cifar10 \
    "${OUTPUT_ROOT}"/blend_0.005_alpha=0.100_*_noise=*_arch=ResNet18_cifar10 \
    "${OUTPUT_ROOT}"/adaptive_blend_0.010_alpha=0.100_cover=0.010_*_noise=*_arch=ResNet18_cifar10
  do
    [ -d "${dir_path}" ] || continue
    candidate=$((candidate + 1))
    if [ "$DRY_RUN" = "1" ]; then
      delete_dir_if_matches "${dir_path}" "asymmetric config"
    else
      delete_dir_if_matches "${dir_path}" "asymmetric config"
      deleted=$((deleted + 1))
    fi
  done
  shopt -u nullglob

  if [ "$DRY_RUN" = "1" ]; then
    echo "Phase A preview: ${candidate} directories matched delete patterns (expected 36)."
  else
    echo "Phase A done: deleted ${deleted} directories (expected 36)."
  fi
}

# Each supplement job is encoded as:
#   attack|poison_rate|extra_args|dir_prefix_without_noise_suffix
SUPPLEMENT_JOBS=(
  "SIG|0.005|-f 6 -delta 28 -label_mode ${SIG_UPGD_LABEL_MODE}|SIG_0.005_delta=28_f=6_mode=${SIG_UPGD_LABEL_MODE}"
  "SIG|0.01|-f 6 -delta 28 -label_mode ${SIG_UPGD_LABEL_MODE}|SIG_0.010_delta=28_f=6_mode=${SIG_UPGD_LABEL_MODE}"
  "SIG|0.005|-f 6 -delta 36 -label_mode ${SIG_UPGD_LABEL_MODE}|SIG_0.005_delta=36_f=6_mode=${SIG_UPGD_LABEL_MODE}"
  "SIG|0.01|-f 6 -delta 36 -label_mode ${SIG_UPGD_LABEL_MODE}|SIG_0.010_delta=36_f=6_mode=${SIG_UPGD_LABEL_MODE}"
  "adaptive_patch|0.005|-cover_rate 0.01 -alpha 0.0|adaptive_patch_0.005_alpha=0.000_cover=0.010"
  "adaptive_patch|0.01|-cover_rate 0.02 -alpha 0.0|adaptive_patch_0.010_alpha=0.000_cover=0.020"
)

expected_result_dir() {
  local dir_prefix="$1"
  local noise_type="$2"
  local noise_level="$3"
  echo "${OUTPUT_ROOT}/${dir_prefix}_noise=${noise_type}_level=${noise_level}_poison_seed=${INPUT_NOISE_SEED}_arch=${ARCH_NAME}"
}

run_supplement_job() {
  local attack="$1"
  local rate="$2"
  local extra_args="$3"
  local dir_prefix="$4"
  local noise_type="$5"
  local noise_level="$6"
  local noise
  local result_dir

  noise="$(noise_args "${noise_type}" "${noise_level}")"
  result_dir="$(expected_result_dir "${dir_prefix}" "${noise_type}" "${noise_level}")"

  if [ "${SKIP_EXISTING}" = "1" ] && result_exists "${result_dir}"; then
    echo "[SKIP] existing result: $(basename "${result_dir}")"
    return 0
  fi

  if [ "$RUN_CREATE" = "1" ]; then
    run_command \
      "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${extra_args} ${noise}" \
      "Create: ${attack}, pr=${rate}, ${extra_args}, noise=${noise_type}/${noise_level}"
  fi

  if [ "$RUN_TRAIN" = "1" ]; then
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${extra_args} ${noise}" \
      "Train: ${attack}, pr=${rate}, ${extra_args}, noise=${noise_type}/${noise_level}"
  fi

  if [ "$RUN_TEST" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_model.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${extra_args} ${noise}" \
      "Source test: ${attack}, pr=${rate}, ${extra_args}, noise=${noise_type}/${noise_level}"
  fi

  if [ "$RUN_TRANSFER" = "1" ]; then
    run_command \
      "${PYTHON_BIN} ${TRANSFER_SCRIPT} $(base_args) -poison_type=${attack} -poison_rate=${rate} ${extra_args} ${noise}" \
      "STL-10 transfer: ${attack}, pr=${rate}, ${extra_args}, noise=${noise_type}/${noise_level}"
  fi

  if [ "$RUN_DEFENSES" = "1" ]; then
    for defense in ${DEFENSES}; do
      run_command \
        "${PYTHON_BIN} other_defense.py $(base_args) -defense=${defense} -poison_type=${attack} -poison_rate=${rate} ${extra_args} ${noise}" \
        "Defense ${defense}: ${attack}, pr=${rate}, ${extra_args}, noise=${noise_type}/${noise_level}"
    done
  fi
}

phase_supplement_missing_configs() {
  echo
  echo "============================================================"
  echo "Phase B: supplement baseline-aligned ResNet18 noise configs"
  echo "============================================================"

  local job
  local attack rate extra_args dir_prefix
  local noise_type noise_level

  for job in "${SUPPLEMENT_JOBS[@]}"; do
    IFS='|' read -r attack rate extra_args dir_prefix <<< "${job}"
    echo
    echo "----- ${attack} pr=${rate} ${extra_args} -----"
    for noise_type in "${NOISE_TYPE_LIST[@]}"; do
      for noise_level in "${NOISE_LEVEL_LIST[@]}"; do
        run_supplement_job "${attack}" "${rate}" "${extra_args}" "${dir_prefix}" "${noise_type}" "${noise_level}"
      done
    done
  done
}

echo "============================================================"
echo "CIFAR-10 ResNet18 noise symmetry patch"
echo "============================================================"
echo "repo root     : ${REPO_ROOT}"
echo "output root   : ${OUTPUT_ROOT}"
echo "devices       : ${DEVICES}"
echo "noise types   : ${NOISE_TYPE_LIST[*]}"
echo "noise levels  : ${NOISE_LEVEL_LIST[*]}"
echo "run delete    : ${RUN_DELETE}"
echo "run supplement: ${RUN_SUPPLEMENT}"
echo "dry run       : ${DRY_RUN}"
echo "skip existing : ${SKIP_EXISTING}"
echo "error log     : ${ERROR_LOG}"
echo "============================================================"

if [ "$RUN_DELETE" = "1" ]; then
  phase_delete_asymmetric_configs
fi

if [ "$RUN_SUPPLEMENT" = "1" ]; then
  phase_supplement_missing_configs
fi

echo
echo "============================================================"
echo "Symmetry patch finished. Check ${ERROR_LOG} for failures."
echo "After completion, regenerate master_results in backdoor-toolbox-new1:"
echo "  cd analysis-transfer-asr2/paper_analysis && python3 build_master_table.py"
echo "============================================================"
