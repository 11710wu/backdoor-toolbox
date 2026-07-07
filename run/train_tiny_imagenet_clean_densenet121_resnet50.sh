#!/usr/bin/env bash

# Tiny-ImageNet clean-model pilot for DenseNet121 and ResNet50.
#
# This intentionally trains only clean none_0.000 models. It is meant as a
# cheap first pass for checking whether either architecture creates a useful
# clean-ACC gap before launching poisoned large-scale architecture experiments.
#
# Usage:
#   cd /workspace/backdoor-toolbox-new1
#   DEVICES=0 bash run/train_tiny_imagenet_clean_densenet121_resnet50.sh
#
# Useful overrides:
#   MODELS="densenet121" DEVICES=0 bash run/train_tiny_imagenet_clean_densenet121_resnet50.sh
#   MODELS="resnet50" DEVICES=1 bash run/train_tiny_imagenet_clean_densenet121_resnet50.sh
#   DRY_RUN=1 bash run/train_tiny_imagenet_clean_densenet121_resnet50.sh
#   FORCE_RETRAIN=1 bash run/train_tiny_imagenet_clean_densenet121_resnet50.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/root/anaconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
  elif [ -x "${HOME}/miniconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/backtool/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="conda run -n backtool python"
  else
    PYTHON_BIN="python"
  fi
fi
DATASET="${DATASET:-tiny_imagenet}"
MODELS="${MODELS:-densenet121 resnet50}"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
RUN_SOURCE_TEST="${RUN_SOURCE_TEST:-1}"
RUN_TARGET_TEST="${RUN_TARGET_TEST:-1}"
RUN_QWEN_TEST="${RUN_QWEN_TEST:-1}"
TRAIN_UPGD_RAW_BASE="${TRAIN_UPGD_RAW_BASE:-0}"
PREPARE_CLEAN_SPLIT="${PREPARE_CLEAN_SPLIT:-auto}"
CLEAN_BUDGET="${CLEAN_BUDGET:-2000}"

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/tiny-target-domain-qwen-full-organized}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/train_tiny_imagenet_clean_densenet121_resnet50_${TIMESTAMP}.log}"

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

arch_name_for_model() {
  case "$1" in
    densenet121) echo "densenet121_tiny_imagenet" ;;
    resnet50) echo "ResNet50_tiny_imagenet" ;;
    *)
      echo "Unsupported model for this clean pilot: $1" >&2
      return 1
      ;;
  esac
}

base_args() {
  local model="$1"
  echo "-dataset=${DATASET} -model=${model} -devices=${DEVICES}"
}

maybe_prepare_clean_split() {
  local labels_path="clean_set/${DATASET}/test_split/labels"

  if [ "$PREPARE_CLEAN_SPLIT" = "0" ]; then
    return 0
  fi

  if [ "$PREPARE_CLEAN_SPLIT" = "auto" ] && [ -f "$labels_path" ]; then
    echo "Clean test split already exists: ${labels_path}"
    return 0
  fi

  run_command \
    "${PYTHON_BIN} create_clean_set.py -dataset=${DATASET} -clean_budget=${CLEAN_BUDGET}" \
    "Prepare clean split for ${DATASET}"
}

echo "============================================================"
echo "Tiny-ImageNet clean-model pilot: DenseNet121 + ResNet50"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "repo         : ${REPO_ROOT}"
echo "dataset      : ${DATASET}"
echo "models       : ${MODELS}"
echo "devices      : ${DEVICES}"
echo "result root  : ${POISONED_TRAIN_SET_ROOT}"
echo "source test  : ${RUN_SOURCE_TEST}"
echo "target test  : ${RUN_TARGET_TEST}"
echo "qwen test    : ${RUN_QWEN_TEST}"
echo "raw UPGD base: ${TRAIN_UPGD_RAW_BASE}"
echo "dry run      : ${DRY_RUN}"
echo "force create : ${FORCE_RECREATE}"
echo "force retrain: ${FORCE_RETRAIN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

maybe_prepare_clean_split

overall_status=0
for model in ${MODELS}; do
  arch_name="$(arch_name_for_model "$model")" || exit 1
  clean_dir="${POISONED_TRAIN_SET_ROOT}/${DATASET}/none_0.000_poison_seed=2333_arch=${arch_name}"
  model_path="${clean_dir}/${arch_name}.pt"
  raw_base_dir="${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=${arch_name}"
  raw_base_path="${raw_base_dir}/upgd_raw_base_${arch_name}.pt"

  echo
  echo "============================================================"
  echo "Clean pilot model: ${model} (${arch_name})"
  echo "clean dir : ${clean_dir}"
  echo "model path: ${model_path}"
  echo "============================================================"

  if [ "$FORCE_RECREATE" = "1" ] || [ ! -f "${clean_dir}/labels" ]; then
    run_command \
      "${PYTHON_BIN} create_poisoned_set.py $(base_args "$model") -poison_type=none -poison_rate=0.0" \
      "Create clean training directory for ${arch_name}"
    status=$?
    [ "$status" -ne 0 ] && overall_status="$status"
  else
    echo "Clean training directory already exists: ${clean_dir}"
  fi

  if [ "$FORCE_RETRAIN" = "1" ] || [ ! -f "$model_path" ]; then
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args "$model") -poison_type=none -poison_rate=0.0" \
      "Train normalized clean model for ${arch_name}"
    status=$?
    [ "$status" -ne 0 ] && overall_status="$status"
  else
    echo "Clean model already exists, skip training: ${model_path}"
  fi

  if [ "$TRAIN_UPGD_RAW_BASE" = "1" ]; then
    if [ "$FORCE_RETRAIN" = "1" ] || [ ! -f "$raw_base_path" ]; then
      run_command \
        "${PYTHON_BIN} train_on_poisoned_set.py $(base_args "$model") -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${raw_base_path}" \
        "Train raw-input clean base for future UPGD ${arch_name}"
      status=$?
      [ "$status" -ne 0 ] && overall_status="$status"
    else
      echo "UPGD raw-input clean base already exists, skip: ${raw_base_path}"
    fi
  fi

  if [ "$RUN_SOURCE_TEST" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_model.py $(base_args "$model") -poison_type=none -poison_rate=0.0" \
      "Source-domain clean test for ${arch_name}"
    status=$?
    [ "$status" -ne 0 ] && overall_status="$status"
  fi

  if [ "$RUN_TARGET_TEST" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_tiny_target_domain.py $(base_args "$model") -source_dataset=${DATASET} -target_domain_dir=${TARGET_DOMAIN_DIR} -poison_type=none -poison_rate=0.0" \
      "ImageNetV2-tiny clean target-domain test for ${arch_name}"
    status=$?
    [ "$status" -ne 0 ] && overall_status="$status"
  fi

  if [ "$RUN_QWEN_TEST" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(base_args "$model") -source_dataset=${DATASET} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR} -poison_type=none -poison_rate=0.0" \
      "Qwen clean target-domain test for ${arch_name}"
    status=$?
    [ "$status" -ne 0 ] && overall_status="$status"
  fi
done

echo
echo "============================================================"
echo "Clean-model pilot finished with status=${overall_status}."
echo "Check ${ERROR_LOG} for failures."
echo "============================================================"
exit "$overall_status"
