#!/usr/bin/env bash
#
# Rerun MNIST-M clean-label UPGD (72 dirs).
# Also rebuilds BOTH clean models per arch:
#   1) none_0.000_...           (Normalize clean baseline)
#   2) upgd_raw_base_0.000_...  (raw [0,1] base for UPGD delta)
#
# Does NOT delete anything — delete old dirs yourself first.
#
# Suggested launch:
#   DEVICES=0 bash run/rerun_mnistm_clean_upgd_raw_base.sh

set +e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

RUN_CLEAN_BASELINE="${RUN_CLEAN_BASELINE:-1}"
# Always retrain raw-input clean bases for UPGD.
FORCE_RAW_BASE=1
RUN_PREP=1
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"
MODELS="${MODELS:-resnet18 mobilenetv2 vgg19_bn}"
POISON_RATES="${POISON_RATES:-0.05 0.01 0.005}"
EPS_VALUES="${EPS_VALUES:-4 6 8 10 12 16 20 24}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_mnistm_clean_upgd_raw_base_${TIMESTAMP}.log}"

INNER_SCRIPT="run/run_upgd_clean_label_raw_base.sh"
read -r -a MODEL_LIST <<< "$MODELS"

arch_name() {
  case "$1" in
    resnet18) echo "ResNet18_mnistm" ;;
    mobilenetv2) echo "mobilenetv2_mnistm" ;;
    vgg19_bn) echo "vgg19_bn_mnistm" ;;
    *) echo "${1}_mnistm" ;;
  esac
}

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out exit_code

  echo
  echo ">>> ${description}"
  echo "${cmd}"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] skipped"
    return 0
  fi

  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"
  eval "$cmd" 2>&1 | tee "$tmp_out"
  exit_code=${PIPESTATUS[0]}

  if [ "$exit_code" -ne 0 ]; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed (exit=${exit_code})"
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

echo "=========================================="
echo "Rerun MNIST-M clean UPGD with raw_base"
echo "root=${ROOT_DIR}"
echo "output=${POISONED_TRAIN_SET_ROOT}/mnistm"
echo "models=${MODELS}"
echo "run_clean_baseline=${RUN_CLEAN_BASELINE}  # none_ Normalize clean"
echo "force_raw_base=1                         # upgd_raw_base"
echo "defenses=${DEFENSES}"
echo "dry_run=${DRY_RUN}"
echo "error_log=${ERROR_LOG}"
echo "=========================================="

if [ ! -f "$INNER_SCRIPT" ]; then
  echo "[ERROR] missing ${INNER_SCRIPT}"
  exit 2
fi

for model in "${MODEL_LIST[@]}"; do
  arch="$(arch_name "$model")"
  echo
  echo "########## MODEL=${model} (${arch}) ##########"

  if [ "$RUN_CLEAN_BASELINE" = "1" ]; then
    echo
    echo "----- Rebuild Normalize clean baseline (none_) -----"
    run_command \
      "${PYTHON_BIN} create_poisoned_set.py -dataset=mnistm -model=${model} -devices=${DEVICES} -poison_type=none -poison_rate=0.0" \
      "Create none_ clean set: mnistm ${model}"

    # Default save path: poisoned_train_set/mnistm/none_0.000_.../${arch}.pt
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py -dataset=mnistm -model=${model} -devices=${DEVICES} -poison_type=none -poison_rate=0.0" \
      "Train none_ Normalize clean model: mnistm ${model}"
  fi

  cmd="DATASET=mnistm MODEL=${model} DEVICES=${DEVICES} LABEL_MODE=clean"
  cmd="${cmd} POISONED_TRAIN_SET_ROOT=${POISONED_TRAIN_SET_ROOT}"
  cmd="${cmd} POISON_RATES='${POISON_RATES}' EPS_VALUES='${EPS_VALUES}'"
  cmd="${cmd} DEFENSES='${DEFENSES}'"
  cmd="${cmd} RUN_PREP=${RUN_PREP} FORCE_RAW_BASE=${FORCE_RAW_BASE}"
  cmd="${cmd} RUN_CREATE=${RUN_CREATE} RUN_TRAIN=${RUN_TRAIN} RUN_TEST=${RUN_TEST}"
  cmd="${cmd} RUN_TRANSFER=${RUN_TRANSFER} RUN_DEFENSES=${RUN_DEFENSES}"
  cmd="${cmd} DRY_RUN=${DRY_RUN} STOP_ON_FAIL=${STOP_ON_FAIL}"
  cmd="${cmd} PYTHON_BIN=${PYTHON_BIN} ERROR_LOG=${ERROR_LOG}"
  cmd="${cmd} bash ${INNER_SCRIPT}"

  echo
  echo "----- UPGD raw_base + 72-grid pipeline -----"
  echo "$cmd"
  eval "$cmd"
  exit_code=$?
  if [ "$exit_code" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    echo "[ERROR] model=${model} failed (exit=${exit_code})"
    exit "$exit_code"
  fi
done

echo
echo "=========================================="
echo "Done. Per model this script rebuilds:"
echo "  ${POISONED_TRAIN_SET_ROOT}/mnistm/none_0.000_poison_seed=2333_arch=<arch>/"
echo "  ${POISONED_TRAIN_SET_ROOT}/mnistm/upgd_raw_base_0.000_poison_seed=2333_arch=<arch>/"
echo "  + 24 clean UPGD dirs"
echo "error_log=${ERROR_LOG}"
echo "=========================================="
