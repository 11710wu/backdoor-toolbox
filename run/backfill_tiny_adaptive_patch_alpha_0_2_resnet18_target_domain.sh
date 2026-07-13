#!/usr/bin/env bash

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${PROJECT_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_DIR"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/backfill_tiny_adaptive_patch_alpha_0_2_resnet18_target_domain.log}"

cd "$PROJECT_ROOT" || exit 1

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
      echo "description: ${description}"
      echo "command: ${cmd}"
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

check_dir() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo "Missing expected result directory: $dir" | tee -a "$ERROR_LOG"
    exit 1
  fi
}

check_model() {
  local dir="$1"
  local model_path="${dir}/ResNet18_tiny_imagenet.pt"
  if [ ! -f "$model_path" ]; then
    echo "Missing expected model file: $model_path" | tee -a "$ERROR_LOG"
    exit 1
  fi
}

echo "============================================================"
echo "Backfill Tiny-ImageNet Adaptive-Patch alpha=0.2 ResNet18 target-domain transfer"
echo "============================================================"
echo "python            : ${PYTHON_BIN}"
echo "devices           : ${DEVICES}"
echo "target domain dir : ${TARGET_DOMAIN_DIR}"
echo "dry run           : ${DRY_RUN}"
echo "stop on fail      : ${STOP_ON_FAIL}"
echo "error log         : ${ERROR_LOG}"
echo "============================================================"

if [ ! -d "$TARGET_DOMAIN_DIR" ]; then
  echo "Target domain directory does not exist: $TARGET_DOMAIN_DIR" | tee -a "$ERROR_LOG"
  exit 1
fi

DIR_005="poisoned_train_set/tiny_imagenet/adaptive_patch_0.005_alpha=0.200_cover=0.010_poison_seed=2333_arch=ResNet18_tiny_imagenet"
DIR_010="poisoned_train_set/tiny_imagenet/adaptive_patch_0.010_alpha=0.200_cover=0.020_poison_seed=2333_arch=ResNet18_tiny_imagenet"

check_dir "$DIR_005"
check_dir "$DIR_010"
check_model "$DIR_005"
check_model "$DIR_010"

run_command "${PYTHON_BIN} test_tiny_target_domain.py -source_dataset=tiny_imagenet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.010 -alpha=0.2 -model=resnet18 -devices=${DEVICES} -target_domain_dir=${TARGET_DOMAIN_DIR}" \
  "Target-domain transfer: adaptive_patch rate=0.005 cover=0.010 alpha=0.2 (resnet18)"

run_command "${PYTHON_BIN} test_tiny_target_domain.py -source_dataset=tiny_imagenet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.010 -cover_rate=0.020 -alpha=0.2 -model=resnet18 -devices=${DEVICES} -target_domain_dir=${TARGET_DOMAIN_DIR}" \
  "Target-domain transfer: adaptive_patch rate=0.010 cover=0.020 alpha=0.2 (resnet18)"

echo
echo "Done. Expected output files:"
echo "  ${DIR_005}/test_tiny_target_domain_results.txt"
echo "  ${DIR_010}/test_tiny_target_domain_results.txt"
