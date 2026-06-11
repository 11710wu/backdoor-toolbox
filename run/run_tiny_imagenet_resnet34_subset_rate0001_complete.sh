#!/usr/bin/env bash

# Tiny-ImageNet ResNet34 subset experiment — poison_rate=0.001 only
#
# Split from run_tiny_imagenet_resnet34_subset_complete.sh
#   8 attacks × 3 trigger strengths × 1 poison rate (0.001) = 24 configs
#
# Usage:
#   bash run/run_tiny_imagenet_resnet34_subset_rate0001_complete.sh
#
# Useful overrides:
#   PYTHON_BIN=/root/anaconda3/envs/backtool/bin/python bash run/run_tiny_imagenet_resnet34_subset_rate0001_complete.sh
#   DEVICES=2 DRY_RUN=1 bash run/run_tiny_imagenet_resnet34_subset_rate0001_complete.sh
#   SKIP_UPGD_PREP=1 bash run/run_tiny_imagenet_resnet34_subset_rate0001_complete.sh

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
TITLE="Tiny-ImageNet ResNet34 subset (poison_rate=0.001)"
DATASET="tiny_imagenet"
MODEL="resnet34"
TRANSFER_SCRIPT="test_tiny_target_domain.py"
QWEN_TRANSFER_SCRIPT="test_tiny_target_domain_qwen.py"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-/workspace/data/tiny-target-domain-qwen-full-organized}"
DEVICES="${DEVICES:-2}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SKIP_UPGD_PREP="${SKIP_UPGD_PREP:-0}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="$LOG_DIR/run_tiny_imagenet_resnet34_subset_rate0001_${TIMESTAMP}.log"

POISON_RATES=("0.001")

ATTACKS=(
  "badnet"
  "blend"
  "SIG"
  "WaNet"
  "adaptive_patch"
  "adaptive_blend"
  "belt"
  "upgd"
)

DEFENSES=(
  "SentiNet"
  "STRIP"
  "ScaleUp"
  "IBD_PSC"
)

UPGD_STEPS="100"
UPGD_STEPS_MULTIPLIER="5"
UPGD_CLEAN_MODEL_PATH="poisoned_train_set/${DATASET}/none_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/ResNet34_tiny_imagenet.pt"

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

double_cover_rate() {
  case "$1" in
    "0.05") echo "0.1" ;;
    "0.01") echo "0.02" ;;
    "0.005") echo "0.01" ;;
    "0.001") echo "0.002" ;;
    *)
      echo "Unsupported poison rate for fixed subset cover-rate mapping: $1" >&2
      return 1
      ;;
  esac
}

strength_values() {
  local attack="$1"

  case "$attack" in
    "badnet") echo "0.2 0.5 1.0" ;;
    "blend") echo "0.05 0.15 0.3" ;;
    "SIG") echo "20 30 40" ;;
    "WaNet") echo "0.4 0.5 0.8" ;;
    "adaptive_patch") echo "0.1 0.2 0.3" ;;
    "adaptive_blend") echo "0.05 0.15 0.25" ;;
    "belt") echo "0.1 0.2 0.3" ;;
    "upgd") echo "4 8 12" ;;
    *)
      echo "Unsupported attack: $attack" >&2
      return 1
      ;;
  esac
}

attack_args() {
  local attack="$1"
  local rate="$2"
  local strength="$3"

  case "$attack" in
    "badnet")
      echo "-alpha ${strength}"
      ;;
    "blend")
      echo "-alpha ${strength}"
      ;;
    "SIG")
      echo "-f 6 -delta ${strength} -label_mode all2one"
      ;;
    "WaNet")
      echo "-cover_rate $(double_cover_rate "$rate") -s ${strength} -k 4"
      ;;
    "adaptive_patch")
      echo "-cover_rate $(double_cover_rate "$rate") -alpha ${strength}"
      ;;
    "adaptive_blend")
      echo "-cover_rate $rate -alpha ${strength}"
      ;;
    "belt")
      echo "-cover_rate 0.5 -mask_rate ${strength} -alpha 1.0"
      ;;
    "upgd")
      echo "-eps ${strength} -constraint Linf -upgd_steps ${UPGD_STEPS} -upgd_steps_multiplier ${UPGD_STEPS_MULTIPLIER} -label_mode all2one"
      ;;
    *)
      echo "Unsupported attack: $attack" >&2
      return 1
      ;;
  esac
}

strength_label() {
  local attack="$1"
  local strength="$2"

  case "$attack" in
    "badnet"|"blend"|"adaptive_patch"|"adaptive_blend") echo "alpha=${strength}" ;;
    "SIG") echo "delta=${strength}" ;;
    "WaNet") echo "s=${strength}" ;;
    "belt") echo "mask_rate=${strength}" ;;
    "upgd") echo "eps=${strength}" ;;
    *) echo "strength=${strength}" ;;
  esac
}

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

transfer_command() {
  local attack="$1"
  local rate="$2"
  local args="$3"

  if [ "$TRANSFER_SCRIPT" = "test_tiny_target_domain.py" ]; then
    echo "${PYTHON_BIN} ${TRANSFER_SCRIPT} $(base_args) -source_dataset=${DATASET} -poison_type=${attack} -poison_rate=${rate} ${args}"
  else
    echo "${PYTHON_BIN} ${TRANSFER_SCRIPT} $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}"
  fi
}

qwen_transfer_command() {
  local attack="$1"
  local rate="$2"
  local args="$3"

  echo "${PYTHON_BIN} ${QWEN_TRANSFER_SCRIPT} $(base_args) -source_dataset=${DATASET} -poison_type=${attack} -poison_rate=${rate} ${args} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR}"
}

echo "============================================================"
echo "${TITLE}"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "dataset      : ${DATASET}"
echo "model        : ${MODEL}"
echo "devices      : ${DEVICES}"
echo "poison rates : ${POISON_RATES[*]}"
echo "attacks      : ${ATTACKS[*]}"
echo "defenses     : ${DEFENSES[*]}"
echo "transfer     : ${TRANSFER_SCRIPT}"
echo "qwen transfer: ${QWEN_TRANSFER_SCRIPT}"
echo "qwen domain  : ${QWEN_TARGET_DOMAIN_DIR}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "skip upgd prep: ${SKIP_UPGD_PREP}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

if [ "$SKIP_UPGD_PREP" != "1" ]; then
  echo
  echo "----- 0. UPGD clean base model preparation -----"
  run_command \
    "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
    "Create clean set for UPGD base model"
  run_command \
    "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
    "Train clean base model for UPGD"
fi

echo
echo "----- 1. Create poisoned datasets -----"
for attack in "${ATTACKS[@]}"; do
  for rate in "${POISON_RATES[@]}"; do
    for strength in $(strength_values "$attack"); do
      args="$(attack_args "$attack" "$rate" "$strength")"
      if [ "$attack" = "upgd" ]; then
        args="${args} -upgd_model_path ${UPGD_CLEAN_MODEL_PATH}"
      fi
      label="$(strength_label "$attack" "$strength")"
      run_command \
        "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
        "Create poisoned set: ${attack}, poison_rate=${rate}, ${label}"
    done
  done
done

echo
echo "----- 2. Train poisoned models -----"
for attack in "${ATTACKS[@]}"; do
  for rate in "${POISON_RATES[@]}"; do
    for strength in $(strength_values "$attack"); do
      args="$(attack_args "$attack" "$rate" "$strength")"
      label="$(strength_label "$attack" "$strength")"
      run_command \
        "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
        "Train model: ${attack}, poison_rate=${rate}, ${label}"
    done
  done
done

echo
echo "----- 3. Source-domain testing -----"
for attack in "${ATTACKS[@]}"; do
  for rate in "${POISON_RATES[@]}"; do
    for strength in $(strength_values "$attack"); do
      args="$(attack_args "$attack" "$rate" "$strength")"
      label="$(strength_label "$attack" "$strength")"
      run_command \
        "${PYTHON_BIN} test_model.py $(base_args) -poison_type=${attack} -poison_rate=${rate} ${args}" \
        "Source test: ${attack}, poison_rate=${rate}, ${label}"
    done
  done
done

echo
echo "----- 4. Transfer testing -----"
for attack in "${ATTACKS[@]}"; do
  for rate in "${POISON_RATES[@]}"; do
    for strength in $(strength_values "$attack"); do
      args="$(attack_args "$attack" "$rate" "$strength")"
      label="$(strength_label "$attack" "$strength")"
      run_command \
        "$(transfer_command "$attack" "$rate" "$args")" \
        "Transfer test: ${attack}, poison_rate=${rate}, ${label}"
    done
  done
done

echo
echo "----- 4b. Qwen target-domain transfer testing -----"
for attack in "${ATTACKS[@]}"; do
  for rate in "${POISON_RATES[@]}"; do
    for strength in $(strength_values "$attack"); do
      args="$(attack_args "$attack" "$rate" "$strength")"
      label="$(strength_label "$attack" "$strength")"
      run_command \
        "$(qwen_transfer_command "$attack" "$rate" "$args")" \
        "Qwen transfer test: ${attack}, poison_rate=${rate}, ${label}"
    done
  done
done

echo
echo "----- 5. Stealth/detection defenses -----"
for defense in "${DEFENSES[@]}"; do
  echo
  echo "----- Defense: ${defense} -----"
  for attack in "${ATTACKS[@]}"; do
    for rate in "${POISON_RATES[@]}"; do
      for strength in $(strength_values "$attack"); do
        args="$(attack_args "$attack" "$rate" "$strength")"
        label="$(strength_label "$attack" "$strength")"
        run_command \
          "${PYTHON_BIN} other_defense.py $(base_args) -defense=${defense} -poison_type=${attack} -poison_rate=${rate} ${args}" \
          "Defense ${defense}: ${attack}, poison_rate=${rate}, ${label}"
      done
    done
  done
done

echo
echo "============================================================"
echo "Subset pipeline (rate=0.001) finished. Check ${ERROR_LOG} for failures."
echo "============================================================"
