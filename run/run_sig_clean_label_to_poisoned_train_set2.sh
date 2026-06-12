#!/usr/bin/env bash
# Supplement explicit clean-label SIG experiments for the clean-vs-dirty study.
# Results are written to poisoned_train_set2 by default.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-run_sig_clean_label_to_poisoned_train_set2}"
DEVICES="${DEVICES:-0}"
LABEL_MODE="clean"
POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set2}"
export POISONED_TRAIN_SET_ROOT

DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_QWEN_TRANSFER="${RUN_QWEN_TRANSFER:-0}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

SIG_DELTAS="${SIG_DELTAS:-4 12 20 28 36 44 56}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC NC}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/data/imagenetv2-matched-frequency-tiny-organized}"
TARGET_DOMAIN_QWEN_DIR="${TARGET_DOMAIN_QWEN_DIR:-/workspace/data/tiny-target-domain-qwen-full-organized}"

DATASETS="${DATASETS:-cifar10 tiny_imagenet mnistm}"
CIFAR10_MODELS="${CIFAR10_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
TINY_IMAGENET_MODELS="${TINY_IMAGENET_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
MNISTM_MODELS="${MNISTM_MODELS:-resnet18 mobilenetv2 vgg19_bn}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log}"

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

models_for_dataset() {
  case "$1" in
    "cifar10") echo "$CIFAR10_MODELS" ;;
    "tiny_imagenet") echo "$TINY_IMAGENET_MODELS" ;;
    "mnistm") echo "$MNISTM_MODELS" ;;
    *)
      echo "Unsupported dataset: $1" >&2
      return 1
      ;;
  esac
}

rates_for_dataset() {
  case "$1" in
    "cifar10") echo "${CIFAR10_POISON_RATES:-0.005 0.01 0.05}" ;;
    "tiny_imagenet") echo "${TINY_IMAGENET_POISON_RATES:-0.001 0.005}" ;;
    "mnistm") echo "${MNISTM_POISON_RATES:-0.005 0.01 0.05}" ;;
    *)
      echo "Unsupported dataset: $1" >&2
      return 1
      ;;
  esac
}

base_args() {
  echo "-dataset=$1 -model=$2 -devices=${DEVICES}"
}

sig_args() {
  echo "-poison_type=SIG -label_mode=${LABEL_MODE} -f=6 -delta=$1"
}

transfer_test() {
  local dataset="$1"
  local model="$2"
  local rate="$3"
  local delta="$4"
  local shared
  shared="$(base_args "$dataset" "$model") $(sig_args "$delta") -poison_rate=${rate}"

  case "$dataset" in
    "cifar10")
      run_command \
        "${PYTHON_BIN} test_stl10.py ${shared}" \
        "STL-10 transfer SIG clean: ${dataset}/${model} rate=${rate} delta=${delta}"
      ;;
    "tiny_imagenet")
      run_command \
        "${PYTHON_BIN} test_tiny_target_domain.py ${shared} -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" \
        "Tiny target-domain transfer SIG clean: ${model} rate=${rate} delta=${delta}"
      if [ "$RUN_QWEN_TRANSFER" = "1" ]; then
        run_command \
          "${PYTHON_BIN} test_tiny_target_domain_qwen.py ${shared} -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_QWEN_DIR}" \
          "Tiny Qwen target-domain transfer SIG clean: ${model} rate=${rate} delta=${delta}"
      fi
      ;;
    "mnistm")
      run_command \
        "${PYTHON_BIN} test_mnist.py ${shared}" \
        "MNIST transfer SIG clean: ${model} rate=${rate} delta=${delta}"
      ;;
  esac
}

echo "============================================================"
echo "SIG clean-label supplement"
echo "============================================================"
echo "output root: ${POISONED_TRAIN_SET_ROOT}"
echo "output note : each poisoned-set dir under this root stores poisoned data, model checkpoints, test/transfer files, and defense outputs"
echo "datasets   : ${DATASETS}"
echo "deltas     : ${SIG_DELTAS}"
echo "defenses   : ${DEFENSES}"
echo "dry run    : ${DRY_RUN}"
echo "error log  : ${ERROR_LOG}"
echo "============================================================"

for dataset in ${DATASETS}; do
  for model in $(models_for_dataset "$dataset"); do
    echo
    echo "========== ${dataset} / ${model} =========="
    for rate in $(rates_for_dataset "$dataset"); do
      for delta in ${SIG_DELTAS}; do
        shared="$(base_args "$dataset" "$model") $(sig_args "$delta") -poison_rate=${rate}"

        if [ "$RUN_CREATE" = "1" ]; then
          run_command \
            "${PYTHON_BIN} create_poisoned_set.py ${shared}" \
            "Create SIG clean poison set: ${dataset}/${model} rate=${rate} delta=${delta}"
        fi

        if [ "$RUN_TRAIN" = "1" ]; then
          run_command \
            "${PYTHON_BIN} train_on_poisoned_set.py ${shared}" \
            "Train SIG clean model: ${dataset}/${model} rate=${rate} delta=${delta}"
        fi

        if [ "$RUN_TEST" = "1" ]; then
          run_command \
            "${PYTHON_BIN} test_model.py ${shared}" \
            "Source test SIG clean: ${dataset}/${model} rate=${rate} delta=${delta}"
        fi

        if [ "$RUN_TRANSFER" = "1" ]; then
          transfer_test "$dataset" "$model" "$rate" "$delta"
        fi

        if [ "$RUN_DEFENSES" = "1" ]; then
          for defense in ${DEFENSES}; do
            if [ "$dataset" = "tiny_imagenet" ] && [ "$defense" = "NC" ]; then
              continue
            fi
            run_command \
              "${PYTHON_BIN} other_defense.py -defense=${defense} ${shared}" \
              "Defense ${defense} SIG clean: ${dataset}/${model} rate=${rate} delta=${delta}"
          done
        fi
      done
    done
  done
done

echo
echo "============================================================"
echo "SIG clean-label supplement finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
