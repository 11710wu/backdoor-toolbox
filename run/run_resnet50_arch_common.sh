#!/usr/bin/env bash

# Shared runner for ResNet50 architecture backfill experiments.
# Entry scripts set DATASET, ATTACK_LIST and poison-rate defaults.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/root/anaconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
  elif [ -x "${HOME}/miniconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/backtool/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DATASET="${DATASET:-tiny_imagenet}"
MODEL="resnet50"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

PREPARE_CLEAN="${PREPARE_CLEAN:-0}"
PREPARE_UPGD_RAW_BASE="${PREPARE_UPGD_RAW_BASE:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SOURCE="${RUN_SOURCE:-1}"
RUN_TARGET="${RUN_TARGET:-1}"
RUN_QWEN_TRANSFER="${RUN_QWEN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"
FORCE_RECREATE_CLEAN="${FORCE_RECREATE_CLEAN:-0}"
FORCE_RETRAIN_CLEAN="${FORCE_RETRAIN_CLEAN:-0}"
FORCE_RETRAIN_UPGD_RAW_BASE="${FORCE_RETRAIN_UPGD_RAW_BASE:-0}"

ATTACK_LIST="${ATTACK_LIST:?Set ATTACK_LIST, e.g. 'basic blend SIG'}"
read -r -a ATTACKS <<< "$ATTACK_LIST"
read -r -a DEFENSES <<< "${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

if [ "$DATASET" = "tiny_imagenet" ]; then
  ARCH_NAME="ResNet50_tiny_imagenet"
  # Keep a single poison rate for the ResNet50 arch grid.
  DEFAULT_POISON_RATES="0.005"
  TARGET_PHASE_NAME="ImageNetV2-tiny transfer testing"
  TARGET_SCRIPT="test_tiny_target_domain.py"
  TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"
  QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/tiny-target-domain-qwen-full-organized}"
  TARGET_EXTRA_PREFIX="-source_dataset=${DATASET} -target_domain_dir=${TARGET_DOMAIN_DIR}"
else
  echo "Unsupported DATASET for ResNet50 architecture backfill: ${DATASET}" >&2
  echo "This runner is Tiny-ImageNet only (cifar10 removed)." >&2
  exit 1
fi

POISON_RATES="${POISON_RATES:-${DEFAULT_POISON_RATES}}"
UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
UPGD_RAW_BASE_DIR="${UPGD_RAW_BASE_DIR:-${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=${ARCH_NAME}}"
UPGD_CLEAN_MODEL_PATH="${UPGD_CLEAN_MODEL_PATH:-${UPGD_RAW_BASE_DIR}/upgd_raw_base_${ARCH_NAME}.pt}"
CLEAN_DIR="${POISONED_TRAIN_SET_ROOT}/${DATASET}/none_0.000_poison_seed=2333_arch=${ARCH_NAME}"
CLEAN_MODEL_PATH="${CLEAN_DIR}/${ARCH_NAME}.pt"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_NAME="${RUN_NAME:-run_${DATASET}_resnet50_arch}"
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

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

double_cover_rate() {
  case "$1" in
    "0.001") echo "0.002" ;;
    "0.005") echo "0.010" ;;
    "0.010") echo "0.020" ;;
    *)
      echo "Unsupported poison rate for cover-rate mapping: $1" >&2
      return 1
      ;;
  esac
}

strength_values() {
  case "$1" in
    "basic") echo "0.2 0.5 1.0" ;;
    "blend") echo "0.05 0.15 0.3" ;;
    "SIG") echo "20 30 40" ;;
    "WaNet") echo "0.4 0.5 0.8" ;;
    "adaptive_patch") echo "0.1 0.2 0.3" ;;
    "adaptive_blend") echo "0.05 0.15 0.25" ;;
    "belt") echo "0.10 0.20 0.30" ;;
    "upgd") echo "4 8 12" ;;
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
    "basic"|"blend")
      echo "-alpha ${strength}"
      ;;
    "SIG")
      echo "-f 6 -delta ${strength} -label_mode clean"
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
      echo "-cover_rate 0.5 -mask_rate 0.2 -alpha ${strength}"
      ;;
    "upgd")
      echo "-eps ${strength} -constraint Linf -upgd_steps ${UPGD_STEPS} -upgd_steps_multiplier ${UPGD_STEPS_MULTIPLIER} -label_mode clean"
      ;;
    *)
      echo "Unsupported attack: $attack" >&2
      return 1
      ;;
  esac
}

strength_label() {
  case "$1" in
    "SIG") echo "delta=$2" ;;
    "WaNet") echo "s=$2" ;;
    "upgd") echo "eps=$2" ;;
    *) echo "alpha=$2" ;;
  esac
}

run_config_phase() {
  local phase="$1"
  local script_name="$2"
  local desc_prefix="$3"
  local extra_prefix="$4"

  echo
  echo "----- ${phase} -----"
  for attack in "${ATTACKS[@]}"; do
    for rate in ${POISON_RATES}; do
      for strength in $(strength_values "$attack"); do
        args="$(attack_args "$attack" "$rate" "$strength")"
        # create_poisoned_set.py 不接受 -no_normalize；只给 train/test/defense 加。
        if { [ "$attack" = "upgd" ] || [ "$attack" = "belt" ]; } && [ "$script_name" != "create_poisoned_set.py" ]; then
          args="${args} -no_normalize"
        fi
        if [ "$script_name" = "create_poisoned_set.py" ] && [ "$attack" = "upgd" ]; then
          args="${args} -upgd_model_path ${UPGD_CLEAN_MODEL_PATH}"
        fi
        label="$(strength_label "$attack" "$strength")"
        run_command \
          "${PYTHON_BIN} ${script_name} $(base_args) ${extra_prefix} -poison_type=${attack} -poison_rate=${rate} ${args}" \
          "${desc_prefix}: ${attack}, poison_rate=${rate}, ${label}"
      done
    done
  done
}

echo "============================================================"
echo "${RUN_TITLE:-ResNet50 architecture backfill}"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "repo         : ${REPO_ROOT}"
echo "dataset      : ${DATASET}"
echo "model        : ${MODEL}"
echo "arch name    : ${ARCH_NAME}"
echo "devices      : ${DEVICES}"
echo "result root  : ${POISONED_TRAIN_SET_ROOT}"
echo "attacks      : ${ATTACKS[*]}"
echo "poison rates : ${POISON_RATES}"
echo "defenses     : ${DEFENSES[*]}"
echo "prepare clean: ${PREPARE_CLEAN}"
echo "prepare upgd : ${PREPARE_UPGD_RAW_BASE}"
echo "clean model  : ${CLEAN_MODEL_PATH}"
echo "upgd raw base: ${UPGD_CLEAN_MODEL_PATH}"
if [ "$DATASET" = "tiny_imagenet" ]; then
  echo "target domain: ${TARGET_DOMAIN_DIR}"
  echo "qwen domain  : ${QWEN_TARGET_DOMAIN_DIR}"
fi
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "$PREPARE_CLEAN" = "1" ]; then
  echo
  echo "----- 0. Clean model preparation -----"
  if [ "$FORCE_RECREATE_CLEAN" = "1" ] || [ ! -f "${CLEAN_DIR}/labels" ]; then
    run_command \
      "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
      "Create clean set/model dir"
  else
    echo "Clean set/model dir already exists: ${CLEAN_DIR}"
  fi
  if [ "$FORCE_RETRAIN_CLEAN" = "1" ] || [ ! -f "$CLEAN_MODEL_PATH" ]; then
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
      "Train normalized clean ResNet50 model"
  else
    echo "Clean model already exists, skip training: ${CLEAN_MODEL_PATH}"
  fi
fi

if [ "$PREPARE_UPGD_RAW_BASE" = "1" ]; then
  echo
  echo "----- 0b. UPGD raw-input clean base preparation -----"
  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
    "Ensure clean set/model dir exists for raw UPGD base"
  if [ "$FORCE_RETRAIN_UPGD_RAW_BASE" = "1" ] || [ ! -f "$UPGD_CLEAN_MODEL_PATH" ]; then
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${UPGD_CLEAN_MODEL_PATH}" \
      "Train raw-input clean ResNet50 model for UPGD"
  else
    echo "UPGD raw-input clean base already exists, skip training: ${UPGD_CLEAN_MODEL_PATH}"
  fi
fi

if [ "$RUN_CREATE" = "1" ]; then
  run_config_phase "1. Create poisoned datasets" "create_poisoned_set.py" "Create poisoned set" ""
fi

if [ "$RUN_TRAIN" = "1" ]; then
  run_config_phase "2. Train poisoned models" "train_on_poisoned_set.py" "Train model" ""
fi

if [ "$RUN_SOURCE" = "1" ]; then
  run_config_phase "3. Source-domain testing" "test_model.py" "Source test" ""
fi

if [ "$RUN_TARGET" = "1" ]; then
  run_config_phase "4. ${TARGET_PHASE_NAME}" "${TARGET_SCRIPT}" "${TARGET_PHASE_NAME}" "${TARGET_EXTRA_PREFIX}"

  if [ "$DATASET" = "tiny_imagenet" ] && [ "$RUN_QWEN_TRANSFER" = "1" ]; then
    run_config_phase \
      "5. Qwen target-domain transfer testing" \
      "test_tiny_target_domain_qwen.py" \
      "Qwen transfer" \
      "-source_dataset=${DATASET} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}"
  fi
fi

if [ "$RUN_DEFENSES" = "1" ]; then
  echo
  echo "----- 6. Stealth/detection defenses -----"
  for defense in "${DEFENSES[@]}"; do
    run_config_phase "Defense: ${defense}" "other_defense.py" "Defense ${defense}" "-defense=${defense}"
  done
fi

echo
echo "============================================================"
echo "ResNet50 architecture backfill finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
