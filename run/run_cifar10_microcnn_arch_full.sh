#!/usr/bin/env bash

# CIFAR-10 MicroCNN architecture supplement.
#
# Full grid: 8 attacks x 2 poison rates x 3 strengths = 48 configs.
# This runner is conservative: it never deletes result folders and, by default,
# stops if a selected result folder already existed before this run but is not
# fully complete.
#
# Usage:
#   bash run/run_cifar10_microcnn_arch_full.sh
#   DRY_RUN=1 bash run/run_cifar10_microcnn_arch_full.sh
#   PHASE=create ATTACK_LIST=basic,blend bash run/run_cifar10_microcnn_arch_full.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/root/anaconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DATASET="cifar10"
MODEL="micro_cnn"
ARCH_NAME="MicroCNN_cifar10"
DEVICES="${DEVICES:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
ALLOW_RESUME_PARTIAL="${ALLOW_RESUME_PARTIAL:-0}"
PHASE="${PHASE:-all}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"

ATTACK_LIST="${ATTACK_LIST:-basic blend adaptive_blend adaptive_patch WaNet SIG upgd belt}"
ATTACK_LIST="${ATTACK_LIST//,/ }"
read -r -a ATTACKS <<< "$ATTACK_LIST"
read -r -a DEFENSES <<< "${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"

PREPARE_CLEAN="${PREPARE_CLEAN:-1}"
PREPARE_UPGD_RAW_BASE="${PREPARE_UPGD_RAW_BASE:-1}"
RUN_CREATE=0
RUN_TRAIN=0
RUN_SOURCE=0
RUN_TRANSFER=0
RUN_DEFENSES=0

case "$PHASE" in
  all)
    RUN_CREATE=1
    RUN_TRAIN=1
    RUN_SOURCE=1
    RUN_TRANSFER=1
    RUN_DEFENSES=1
    ;;
  create) RUN_CREATE=1 ;;
  train) RUN_TRAIN=1 ;;
  source|test) RUN_SOURCE=1 ;;
  transfer|target) RUN_TRANSFER=1 ;;
  defense|defenses) RUN_DEFENSES=1 ;;
  *)
    echo "Unsupported PHASE=${PHASE}. Use create|train|source|transfer|defense|all." >&2
    exit 2
    ;;
esac

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/run_cifar10_microcnn_arch_full_${TIMESTAMP}.log}"

CLEAN_DIR="${POISONED_TRAIN_SET_ROOT}/${DATASET}/none_0.000_poison_seed=2333_arch=${ARCH_NAME}"
CLEAN_MODEL_PATH="${CLEAN_DIR}/${ARCH_NAME}.pt"
UPGD_RAW_BASE_DIR="${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=${ARCH_NAME}"
UPGD_RAW_BASE_PATH="${UPGD_RAW_BASE_DIR}/upgd_raw_base_${ARCH_NAME}.pt"

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

rate_fmt() {
  printf "%.3f" "$1"
}

alpha_fmt() {
  printf "%.3f" "$1"
}

cover_for_rate() {
  case "$1" in
    0.005) echo "0.010" ;;
    0.010) echo "0.020" ;;
    *)
      echo "Unsupported poison rate for double cover mapping: $1" >&2
      return 1
      ;;
  esac
}

strength_values() {
  case "$1" in
    basic) echo "0.2 0.5 1.0" ;;
    blend) echo "0.05 0.15 0.30" ;;
    adaptive_blend) echo "0.05 0.15 0.25" ;;
    adaptive_patch) echo "0.1 0.2 0.3" ;;
    WaNet) echo "0.4 0.6 0.8" ;;
    SIG) echo "20 28 36" ;;
    upgd) echo "4.0 8.0 12.0" ;;
    belt) echo "0.10 0.20 0.30" ;;
    *)
      echo "Unsupported attack in ATTACK_LIST: $1" >&2
      return 1
      ;;
  esac
}

poison_rates() {
  case "$1" in
    belt) echo "0.010 0.020" ;;
    *) echo "0.005 0.010" ;;
  esac
}

attack_args() {
  local attack="$1"
  local rate="$2"
  local strength="$3"
  case "$attack" in
    basic|blend)
      echo "-alpha ${strength}"
      ;;
    adaptive_blend)
      echo "-cover_rate ${rate} -alpha ${strength}"
      ;;
    adaptive_patch)
      echo "-cover_rate $(cover_for_rate "$rate") -alpha ${strength}"
      ;;
    WaNet)
      echo "-cover_rate $(cover_for_rate "$rate") -s ${strength} -k 4"
      ;;
    SIG)
      echo "-f 6 -delta ${strength} -label_mode clean"
      ;;
    upgd)
      echo "-eps ${strength} -constraint Linf -upgd_steps 100 -upgd_steps_multiplier 5 -label_mode clean"
      ;;
    belt)
      echo "-cover_rate 0.5 -mask_rate 0.2 -alpha ${strength}"
      ;;
  esac
}

poison_dir() {
  local attack="$1"
  local rate="$2"
  local strength="$3"
  local rate_s
  rate_s="$(rate_fmt "$rate")"
  case "$attack" in
    basic)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/basic_${rate_s}_alpha=$(alpha_fmt "$strength")_trigger=badnet_patch_32.png_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
    blend)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/blend_${rate_s}_alpha=$(alpha_fmt "$strength")_trigger=hellokitty_32.png_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
    adaptive_blend)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/adaptive_blend_${rate_s}_alpha=$(alpha_fmt "$strength")_cover=$(rate_fmt "$rate")_trigger=hellokitty_32.png_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
    adaptive_patch)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/adaptive_patch_${rate_s}_alpha=$(alpha_fmt "$strength")_cover=$(cover_for_rate "$rate")_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
    WaNet)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/WaNet_${rate_s}_cover=$(cover_for_rate "$rate")_s=${strength}_k=4_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
    SIG)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/SIG_${rate_s}_delta=${strength}_f=6_mode=clean_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
    upgd)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_${rate_s}_eps=${strength}_constraint=Linf_steps=100_mode=clean_mult=5_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
    belt)
      echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/belt_${rate_s}_alpha=$(alpha_fmt "$strength")_cover=0.500_mask=0.200_poison_seed=2333_arch=${ARCH_NAME}"
      ;;
  esac
}

model_file_for_attack() {
  local attack="$1"
  if [ "$attack" = "belt" ]; then
    echo "${ARCH_NAME}_belt_aug_model_seed=2333.pt"
  else
    echo "${ARCH_NAME}.pt"
  fi
}

is_complete_dir() {
  local dir="$1"
  local attack="$2"
  local model_file
  model_file="$(model_file_for_attack "$attack")"
  [ -f "${dir}/labels" ] || return 1
  [ -f "${dir}/poison_indices" ] || return 1
  [ -f "${dir}/${model_file}" ] || return 1
  [ -f "${dir}/train_results_seed=2333.json" ] || return 1
  compgen -G "${dir}/test_results_seed=2333*.json" >/dev/null || return 1
  compgen -G "${dir}/test_stl10_results*.txt" >/dev/null || return 1
  [ -f "${dir}/sentinet_defense_results.json" ] || return 1
  [ -f "${dir}/strip_defense_results.json" ] || return 1
  [ -f "${dir}/scaleup_defense_results.json" ] || return 1
  [ -f "${dir}/ibd_psc_defense_results.json" ] || return 1
}

phase_done() {
  local phase="$1"
  local dir="$2"
  local attack="$3"
  local model_file
  model_file="$(model_file_for_attack "$attack")"
  case "$phase" in
    create) [ -f "${dir}/labels" ] && [ -f "${dir}/poison_indices" ] ;;
    train) [ -f "${dir}/${model_file}" ] && [ -f "${dir}/train_results_seed=2333.json" ] ;;
    source) compgen -G "${dir}/test_results_seed=2333*.json" >/dev/null ;;
    transfer) compgen -G "${dir}/test_stl10_results*.txt" >/dev/null ;;
    defense:SentiNet) [ -f "${dir}/sentinet_defense_results.json" ] ;;
    defense:STRIP) [ -f "${dir}/strip_defense_results.json" ] ;;
    defense:ScaleUp) [ -f "${dir}/scaleup_defense_results.json" ] ;;
    defense:IBD_PSC) [ -f "${dir}/ibd_psc_defense_results.json" ] ;;
    *) return 1 ;;
  esac
}

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/microcnn_run_$$_${RANDOM}.out")"

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
      cat "$tmp_out" 2>/dev/null
      echo "---"
    } >> "$ERROR_LOG"
    rm -f "$tmp_out"
    if [ "$STOP_ON_FAIL" = "1" ]; then
      exit "$exit_code"
    fi
    return "$exit_code"
  fi

  rm -f "$tmp_out"
  return 0
}

selected_configs() {
  for attack in "${ATTACKS[@]}"; do
    for rate in $(poison_rates "$attack"); do
      for strength in $(strength_values "$attack"); do
        echo "${attack}|${rate}|${strength}|$(poison_dir "$attack" "$rate" "$strength")|$(attack_args "$attack" "$rate" "$strength")"
      done
    done
  done
}

preflight_existing_partial() {
  if [ "$ALLOW_RESUME_PARTIAL" = "1" ] || [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  local bad=0
  while IFS="|" read -r attack rate strength dir args; do
    if [ -d "$dir" ] && ! is_complete_dir "$dir" "$attack"; then
      echo "Existing incomplete result folder: ${dir}" >&2
      bad=1
    fi
  done < <(selected_configs)
  if [ "$bad" = "1" ]; then
    echo "Stop before running. Set ALLOW_RESUME_PARTIAL=1 only if you intentionally want to resume partial folders." >&2
    exit 3
  fi
}

run_config_phase() {
  local phase="$1"
  local script="$2"
  local desc="$3"
  local extra="$4"
  echo
  echo "----- ${desc} -----"
  while IFS="|" read -r attack rate strength dir args; do
    if phase_done "$phase" "$dir" "$attack"; then
      echo "Skip ${phase}: already complete: ${dir}"
      continue
    fi
    if { [ "$attack" = "upgd" ] || [ "$attack" = "belt" ]; } && [ "$script" != "create_poisoned_set.py" ]; then
      args="${args} -no_normalize"
    fi
    if [ "$script" = "create_poisoned_set.py" ] && [ "$attack" = "upgd" ]; then
      args="${args} -upgd_model_path ${UPGD_RAW_BASE_PATH}"
    fi
    run_command \
      "${PYTHON_BIN} ${script} $(base_args) ${extra} -poison_type=${attack} -poison_rate=${rate} ${args}" \
      "${desc}: ${attack}, poison_rate=${rate}, strength=${strength}"
  done < <(selected_configs)
}

echo "============================================================"
echo "CIFAR-10 MicroCNN architecture supplement"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "repo         : ${REPO_ROOT}"
echo "model        : ${MODEL} (${ARCH_NAME})"
echo "devices      : ${DEVICES}"
echo "result root  : ${POISONED_TRAIN_SET_ROOT}"
echo "phase        : ${PHASE}"
echo "attacks      : ${ATTACKS[*]}"
echo "defenses     : ${DEFENSES[*]}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

config_count=0
echo
echo "Selected configs:"
while IFS="|" read -r attack rate strength dir args; do
  config_count=$((config_count + 1))
  echo "CONFIG ${config_count}: ${attack} rate=${rate} strength=${strength} -> ${dir}"
done < <(selected_configs)
echo "Total configs: ${config_count}"

preflight_existing_partial

if [ "$PREPARE_CLEAN" = "1" ]; then
  echo
  echo "----- Clean model preparation -----"
  if [ ! -f "${CLEAN_DIR}/labels" ]; then
    run_command "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" "Create clean MicroCNN directory"
  else
    echo "Clean labels already exist: ${CLEAN_DIR}/labels"
  fi
  if [ ! -f "$CLEAN_MODEL_PATH" ]; then
    run_command "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" "Train normalized clean MicroCNN"
  else
    echo "Clean MicroCNN already exists: ${CLEAN_MODEL_PATH}"
  fi
  if [ "$DRY_RUN" != "1" ] && [ -f "${CLEAN_DIR}/train_results_seed=2333.json" ]; then
    "${PYTHON_BIN}" - <<PY
import json, sys
path = "${CLEAN_DIR}/train_results_seed=2333.json"
acc = json.load(open(path, "r", encoding="utf-8")).get("clean_acc")
if acc is None:
    print(f"Clean ACC missing in {path}", file=sys.stderr)
    sys.exit(4)
print(f"Clean ACC gate: {acc:.6f}")
if not (0.55 <= float(acc) <= 0.65):
    print("Clean ACC is outside the required [0.55, 0.65] range; stop before attack grid.", file=sys.stderr)
    sys.exit(5)
PY
    gate_status=$?
    if [ "$gate_status" -ne 0 ]; then
      exit "$gate_status"
    fi
  fi
fi

if [ "$PREPARE_UPGD_RAW_BASE" = "1" ]; then
  echo
  echo "----- UPGD raw-input clean base preparation -----"
  if [ ! -f "$UPGD_RAW_BASE_PATH" ]; then
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${UPGD_RAW_BASE_PATH}" \
      "Train raw-input clean MicroCNN for UPGD"
  else
    echo "UPGD raw base already exists: ${UPGD_RAW_BASE_PATH}"
  fi
fi

if [ "$RUN_CREATE" = "1" ]; then
  run_config_phase "create" "create_poisoned_set.py" "Create poisoned set" ""
fi
if [ "$RUN_TRAIN" = "1" ]; then
  run_config_phase "train" "train_on_poisoned_set.py" "Train model" ""
fi
if [ "$RUN_SOURCE" = "1" ]; then
  run_config_phase "source" "test_model.py" "Source-domain test" ""
fi
if [ "$RUN_TRANSFER" = "1" ]; then
  run_config_phase "transfer" "test_stl10.py" "STL10 transfer test" ""
fi
if [ "$RUN_DEFENSES" = "1" ]; then
  for defense in "${DEFENSES[@]}"; do
    run_config_phase "defense:${defense}" "other_defense.py" "Defense ${defense}" "-defense=${defense}"
  done
fi

echo
echo "============================================================"
echo "CIFAR-10 MicroCNN architecture supplement finished."
echo "============================================================"
