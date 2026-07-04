#!/usr/bin/env bash

# Backfill matched BELT rows for poisoned_train_set4.
#
# Why this exists:
#   The old set4 BELT rows varied mask_rate with alpha=1.0, which does not
#   strictly match the original ResNet18 baseline grid used by paper_analysis.
#   The matched grid fixes mask_rate=0.2 and varies alpha in {0.10, 0.20, 0.30}.
#
# Scope:
#   CIFAR-10 SmallCNN:
#     poison_rate = 0.005, 0.010
#     alpha       = 0.10, 0.20, 0.30
#     cover_rate  = 0.5
#     mask_rate   = 0.2
#
#   Tiny-ImageNet ResNet34:
#     poison_rate = 0.001, 0.005
#     alpha       = 0.10, 0.20, 0.30
#     cover_rate  = 0.5
#     mask_rate   = 0.2
#
# Usage:
#   cd /workspace/backdoor-toolbox-new1
#   DEVICES=0 bash run/backfill_set4_matched_belt_smallcnn_resnet34.sh
#
# Useful overrides:
#   DRY_RUN=1 DEVICES=0 bash run/backfill_set4_matched_belt_smallcnn_resnet34.sh
#   RUN_CIFAR_SMALLCNN=0 DEVICES=1 bash run/backfill_set4_matched_belt_smallcnn_resnet34.sh
#   RUN_TINY_RESNET34=0 DEVICES=0 bash run/backfill_set4_matched_belt_smallcnn_resnet34.sh
#   RUN_QWEN=0 bash run/backfill_set4_matched_belt_smallcnn_resnet34.sh
#   STOP_ON_FAIL=1 bash run/backfill_set4_matched_belt_smallcnn_resnet34.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_CIFAR_SMALLCNN="${RUN_CIFAR_SMALLCNN:-1}"
RUN_TINY_RESNET34="${RUN_TINY_RESNET34:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/tiny-target-domain-qwen-full-organized}"
DEFENSES_STR="${DEFENSES_STR:-SentiNet STRIP ScaleUp IBD_PSC}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${LOG_DIR}/backfill_set4_matched_belt_smallcnn_resnet34_${TIMESTAMP}.log"

read -r -a DEFENSES <<< "${DEFENSES_STR}"

CIFAR_BELT_RATES=("0.005" "0.010")
TINY_BELT_RATES=("0.001" "0.005")
BELT_ALPHAS=("0.10" "0.20" "0.30")
COVER_RATE="0.5"
MASK_RATE="0.2"

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"

  echo
  echo ">>> ${description}"
  echo "${cmd}"

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] skipped"
    rm -f "${tmp_out}"
    return 0
  fi

  eval "${cmd}" 2>&1 | tee "${tmp_out}"
  local exit_code="${PIPESTATUS[0]}"
  if [ "${exit_code}" -ne 0 ]; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed with exit code ${exit_code}"
      echo "command: ${cmd}"
      echo "description: ${description}"
      echo "--- stdout/stderr ---"
      cat "${tmp_out}" 2>/dev/null
      echo "---"
    } >> "${ERROR_LOG}"

    if [ "${STOP_ON_FAIL}" = "1" ]; then
      rm -f "${tmp_out}"
      exit "${exit_code}"
    fi
  fi

  rm -f "${tmp_out}"
  return "${exit_code}"
}

base_args() {
  local dataset="$1"
  local model="$2"
  echo "-dataset=${dataset} -model=${model} -devices=${DEVICES}"
}

belt_args() {
  local rate="$1"
  local alpha="$2"
  echo "-poison_type=belt -poison_rate=${rate} -cover_rate ${COVER_RATE} -mask_rate ${MASK_RATE} -alpha ${alpha}"
}

run_cifar_smallcnn_belt() {
  local rate="$1"
  local alpha="$2"
  local label="cifar10 SmallCNN BELT rate=${rate} alpha=${alpha} cover=${COVER_RATE} mask=${MASK_RATE}"

  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args cifar10 small_cnn) $(belt_args "${rate}" "${alpha}")" \
    "Create: ${label}"

  run_command \
    "${PYTHON_BIN} train_on_poisoned_set.py $(base_args cifar10 small_cnn) $(belt_args "${rate}" "${alpha}")" \
    "Train: ${label}"

  run_command \
    "${PYTHON_BIN} test_model.py $(base_args cifar10 small_cnn) $(belt_args "${rate}" "${alpha}")" \
    "Source test: ${label}"

  run_command \
    "${PYTHON_BIN} test_stl10.py $(base_args cifar10 small_cnn) $(belt_args "${rate}" "${alpha}")" \
    "STL10 transfer: ${label}"

  for defense in "${DEFENSES[@]}"; do
    run_command \
      "${PYTHON_BIN} other_defense.py $(base_args cifar10 small_cnn) -defense=${defense} $(belt_args "${rate}" "${alpha}")" \
      "Defense ${defense}: ${label}"
  done
}

run_tiny_resnet34_belt() {
  local rate="$1"
  local alpha="$2"
  local label="tiny_imagenet ResNet34 BELT rate=${rate} alpha=${alpha} cover=${COVER_RATE} mask=${MASK_RATE}"

  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args tiny_imagenet resnet34) $(belt_args "${rate}" "${alpha}")" \
    "Create: ${label}"

  run_command \
    "${PYTHON_BIN} train_on_poisoned_set.py $(base_args tiny_imagenet resnet34) $(belt_args "${rate}" "${alpha}")" \
    "Train: ${label}"

  run_command \
    "${PYTHON_BIN} test_model.py $(base_args tiny_imagenet resnet34) $(belt_args "${rate}" "${alpha}")" \
    "Source test: ${label}"

  run_command \
    "${PYTHON_BIN} test_tiny_target_domain.py $(base_args tiny_imagenet resnet34) -source_dataset=tiny_imagenet $(belt_args "${rate}" "${alpha}")" \
    "ImageNetV2-tiny transfer: ${label}"

  if [ "${RUN_QWEN}" = "1" ]; then
    run_command \
      "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(base_args tiny_imagenet resnet34) -source_dataset=tiny_imagenet $(belt_args "${rate}" "${alpha}") -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}" \
      "Qwen transfer: ${label}"
  fi

  for defense in "${DEFENSES[@]}"; do
    run_command \
      "${PYTHON_BIN} other_defense.py $(base_args tiny_imagenet resnet34) -defense=${defense} $(belt_args "${rate}" "${alpha}")" \
      "Defense ${defense}: ${label}"
  done
}

echo "============================================================"
echo "Backfill matched BELT rows for poisoned_train_set4"
echo "============================================================"
echo "repo root          : ${REPO_ROOT}"
echo "python             : ${PYTHON_BIN}"
echo "devices            : ${DEVICES}"
echo "result root        : ${POISONED_TRAIN_SET_ROOT}"
echo "run cifar smallcnn : ${RUN_CIFAR_SMALLCNN}"
echo "run tiny resnet34  : ${RUN_TINY_RESNET34}"
echo "run qwen           : ${RUN_QWEN}"
echo "qwen domain        : ${QWEN_TARGET_DOMAIN_DIR}"
echo "cifar rates        : ${CIFAR_BELT_RATES[*]}"
echo "tiny rates         : ${TINY_BELT_RATES[*]}"
echo "alphas             : ${BELT_ALPHAS[*]}"
echo "cover/mask         : ${COVER_RATE}/${MASK_RATE}"
echo "defenses           : ${DEFENSES[*]}"
echo "dry run            : ${DRY_RUN}"
echo "stop on fail       : ${STOP_ON_FAIL}"
echo "error log          : ${ERROR_LOG}"
echo "============================================================"

if [ "${RUN_CIFAR_SMALLCNN}" = "1" ]; then
  echo
  echo "----- CIFAR-10 SmallCNN matched BELT -----"
  for rate in "${CIFAR_BELT_RATES[@]}"; do
    for alpha in "${BELT_ALPHAS[@]}"; do
      run_cifar_smallcnn_belt "${rate}" "${alpha}"
    done
  done
fi

if [ "${RUN_TINY_RESNET34}" = "1" ]; then
  echo
  echo "----- Tiny-ImageNet ResNet34 matched BELT -----"
  for rate in "${TINY_BELT_RATES[@]}"; do
    for alpha in "${BELT_ALPHAS[@]}"; do
      run_tiny_resnet34_belt "${rate}" "${alpha}"
    done
  done
fi

echo
echo "============================================================"
echo "Matched BELT backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
