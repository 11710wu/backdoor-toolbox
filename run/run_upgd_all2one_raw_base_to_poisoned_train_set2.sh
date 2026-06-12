#!/usr/bin/env bash

# Supplement UPGD all-to-one experiments for the clean-vs-dirty label study.
# Results are written to poisoned_train_set2 by default.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-run_upgd_all2one_raw_base_to_poisoned_train_set2}"
DEVICES="${DEVICES:-0}"
LABEL_MODE="all2one"
POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set2}"
export POISONED_TRAIN_SET_ROOT

DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_PREP="${RUN_PREP:-1}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_QWEN_TRANSFER="${RUN_QWEN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

UPGD_CONSTRAINT="${UPGD_CONSTRAINT:-Linf}"
UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
EPS_VALUES="${EPS_VALUES:-4 6 8 10 12 16 20 24}"
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

arch_name() {
  case "$1:$2" in
    "cifar10:resnet18") echo "ResNet18_cifar10" ;;
    "cifar10:mobilenetv2") echo "mobilenetv2_cifar10" ;;
    "cifar10:vgg19_bn") echo "vgg19_bn_cifar10" ;;
    "cifar10:small_cnn") echo "SmallCNN_cifar10" ;;
    "tiny_imagenet:resnet18") echo "ResNet18_tiny_imagenet" ;;
    "tiny_imagenet:resnet34") echo "ResNet34_tiny_imagenet" ;;
    "tiny_imagenet:mobilenetv2") echo "mobilenetv2_tiny_imagenet" ;;
    "tiny_imagenet:vgg19_bn") echo "vgg19_bn_tiny_imagenet" ;;
    "mnistm:resnet18") echo "ResNet18_mnistm" ;;
    "mnistm:mobilenetv2") echo "mobilenetv2_mnistm" ;;
    "mnistm:vgg19_bn") echo "vgg19_bn_mnistm" ;;
    *)
      echo "Unsupported DATASET/MODEL pair: $1/$2" >&2
      return 1
      ;;
  esac
}

base_args() {
  echo "-dataset=$1 -model=$2 -devices=${DEVICES}"
}

upgd_args() {
  echo "-poison_type=upgd -label_mode=${LABEL_MODE} -constraint=${UPGD_CONSTRAINT} -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER}"
}

raw_base_path() {
  local dataset="$1"
  local arch="$2"
  echo "${POISONED_TRAIN_SET_ROOT}/${dataset}/upgd_raw_base_0.000_poison_seed=2333_arch=${arch}/upgd_raw_base_${arch}.pt"
}

prepare_raw_base() {
  local dataset="$1"
  local model="$2"
  local arch="$3"
  local raw_base
  raw_base="$(raw_base_path "$dataset" "$arch")"

  if [ "$RUN_PREP" != "1" ]; then
    return 0
  fi

  if [ -f "$raw_base" ]; then
    echo "[SKIP] Raw UPGD base exists: ${raw_base}"
    return 0
  fi

  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args "$dataset" "$model") -poison_type=none -poison_rate=0.0" \
    "Create clean set for raw UPGD base: ${dataset}/${model}"
  run_command \
    "${PYTHON_BIN} train_on_poisoned_set.py $(base_args "$dataset" "$model") -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${raw_base}" \
    "Train raw-input clean base for UPGD: ${dataset}/${model}"
}

transfer_test() {
  local dataset="$1"
  local model="$2"
  local rate="$3"
  local eps="$4"
  local shared
  shared="$(base_args "$dataset" "$model") $(upgd_args) -poison_rate=${rate} -eps=${eps}"

  case "$dataset" in
    "cifar10")
      run_command \
        "${PYTHON_BIN} test_stl10.py ${shared}" \
        "STL-10 transfer: ${dataset}/${model} rate=${rate} eps=${eps}"
      ;;
    "tiny_imagenet")
      run_command \
        "${PYTHON_BIN} test_tiny_target_domain.py ${shared} -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" \
        "Tiny target-domain transfer: ${model} rate=${rate} eps=${eps}"
      if [ "$RUN_QWEN_TRANSFER" = "1" ]; then
        run_command \
          "${PYTHON_BIN} test_tiny_target_domain_qwen.py ${shared} -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_QWEN_DIR}" \
          "Tiny Qwen target-domain transfer: ${model} rate=${rate} eps=${eps}"
      fi
      ;;
    "mnistm")
      run_command \
        "${PYTHON_BIN} test_mnist.py ${shared}" \
        "MNIST cross-domain transfer: ${model} rate=${rate} eps=${eps}"
      ;;
  esac
}

echo "============================================================"
echo "UPGD all-to-one raw-base supplement"
echo "============================================================"
echo "output root : ${POISONED_TRAIN_SET_ROOT}"
echo "output note : each poisoned-set dir under this root stores poisoned data, model checkpoints, test/transfer files, and defense outputs"
echo "datasets     : ${DATASETS}"
echo "eps values   : ${EPS_VALUES}"
echo "label_mode   : ${LABEL_MODE}"
echo "defenses     : ${DEFENSES}"
echo "dry run      : ${DRY_RUN}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

for dataset in ${DATASETS}; do
  for model in $(models_for_dataset "$dataset"); do
    arch="$(arch_name "$dataset" "$model")" || exit 2
    raw_base="$(raw_base_path "$dataset" "$arch")"

    echo
    echo "========== ${dataset} / ${model} (${arch}) =========="
    prepare_raw_base "$dataset" "$model" "$arch"

    for rate in $(rates_for_dataset "$dataset"); do
      for eps in ${EPS_VALUES}; do
        shared="$(base_args "$dataset" "$model") $(upgd_args) -poison_rate=${rate} -eps=${eps}"

        if [ "$RUN_CREATE" = "1" ]; then
          run_command \
            "${PYTHON_BIN} create_poisoned_set.py ${shared} -upgd_model_path=${raw_base}" \
            "Create UPGD all-to-one poison set: ${dataset}/${model} rate=${rate} eps=${eps}"
        fi

        if [ "$RUN_TRAIN" = "1" ]; then
          run_command \
            "${PYTHON_BIN} train_on_poisoned_set.py ${shared}" \
            "Train UPGD all-to-one model: ${dataset}/${model} rate=${rate} eps=${eps}"
        fi

        if [ "$RUN_TEST" = "1" ]; then
          run_command \
            "${PYTHON_BIN} test_model.py ${shared}" \
            "Source test UPGD all-to-one: ${dataset}/${model} rate=${rate} eps=${eps}"
        fi

        if [ "$RUN_TRANSFER" = "1" ]; then
          transfer_test "$dataset" "$model" "$rate" "$eps"
        fi

        if [ "$RUN_DEFENSES" = "1" ]; then
          for defense in ${DEFENSES}; do
            run_command \
              "${PYTHON_BIN} other_defense.py -defense=${defense} ${shared}" \
              "Defense ${defense}: ${dataset}/${model} rate=${rate} eps=${eps}"
          done
        fi
      done
    done
  done
done

echo
echo "============================================================"
echo "UPGD all-to-one supplement finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
