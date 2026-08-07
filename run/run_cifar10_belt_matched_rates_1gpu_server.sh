#!/usr/bin/env bash

# Single-GPU server: the remaining four CIFAR-10 BELT configurations.
# This assignment is disjoint from run_cifar10_belt_matched_rates_8gpu_server.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-0}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"
LOG_DIR="${LOG_DIR:-logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export PHASE DRY_RUN SKIP_EXISTING DEFENSE_LIST
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/root/anaconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/cifar10_belt_matched_1gpu_gpu${DEVICE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

arch_name() {
  case "$1" in
    resnet18) echo "ResNet18_cifar10" ;;
    mobilenetv2) echo "mobilenetv2_cifar10" ;;
    vgg19_bn) echo "vgg19_bn_cifar10" ;;
    *) echo "Unsupported model: $1" >&2; return 2 ;;
  esac
}

poison_dir() {
  local model="$1"
  local rate="$2"
  local alpha="$3"
  local arch
  arch="$(arch_name "$model")"
  printf '%s/cifar10/belt_%.3f_alpha=%.3f_cover=0.500_mask=0.200_poison_seed=2333_arch=%s' \
    "$POISONED_TRAIN_SET_ROOT" "$rate" "$alpha" "$arch"
}

phase_complete() {
  local phase="$1"
  local dir="$2"
  local arch="$3"
  local defense="${4:-}"
  local model_file="${dir}/${arch}_belt_aug_model_seed=2333.pt"
  local result_file

  [ "$SKIP_EXISTING" = "1" ] || return 1
  case "$phase" in
    create)
      [ -d "${dir}/data" ] && [ -f "${dir}/labels" ] \
        && [ -f "${dir}/poison_indices" ] && [ -f "${dir}/cover_indices" ] \
        && [ -f "${dir}/pmarks" ]
      ;;
    train)
      result_file="${dir}/train_results_seed=2333.json"
      [ -f "$model_file" ] && [ -f "$result_file" ] \
        && [ -f "${dir}/poison_indices" ] && [ "$model_file" -nt "${dir}/poison_indices" ] \
        && grep -Eq '"checkpoint_selection"[[:space:]]*:[[:space:]]*"final_epoch"' "$result_file"
      ;;
    source)
      result_file="${dir}/test_results_seed=2333.json"
      [ -f "$model_file" ] && [ -f "$result_file" ] && [ "$result_file" -nt "$model_file" ]
      ;;
    transfer)
      result_file="${dir}/test_stl10_results.txt"
      [ -f "$model_file" ] && [ -f "$result_file" ] && [ "$result_file" -nt "$model_file" ]
      ;;
    defense)
      case "$defense" in
        SentiNet) result_file="${dir}/sentinet_defense_results.json" ;;
        STRIP) result_file="${dir}/strip_defense_results.json" ;;
        ScaleUp) result_file="${dir}/scaleup_defense_results.json" ;;
        IBD_PSC) result_file="${dir}/ibd_psc_defense_results.json" ;;
        *) return 1 ;;
      esac
      [ -f "$model_file" ] && [ -f "$result_file" ] && [ "$result_file" -nt "$model_file" ]
      ;;
    *) return 1 ;;
  esac
}

run_command() {
  local description="$1"
  shift
  echo
  echo ">>> ${description}"
  printf '    '
  printf '%q ' "$@"
  printf '\n'
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

requested_phases() {
  case "$PHASE" in
    all) echo "create train source transfer defense" ;;
    test) echo "source" ;;
    target) echo "transfer" ;;
    defenses) echo "defense" ;;
    *) echo "$PHASE" ;;
  esac
}

run_config_phase() {
  local phase="$1"
  local model="$2"
  local rate="$3"
  local alpha="$4"
  local arch
  local dir
  local defense
  arch="$(arch_name "$model")"
  dir="$(poison_dir "$model" "$rate" "$alpha")"
  local common_args=(
    -dataset=cifar10 -model="$model" -devices="$DEVICE" -poison_type=belt
    -poison_rate="$rate" -cover_rate=0.5 -mask_rate=0.2 -alpha="$alpha"
  )
  local label="model=${model}, rate=${rate}, alpha=${alpha}, GPU=${DEVICE}"

  case "$phase" in
    create)
      phase_complete create "$dir" "$arch" \
        && echo "Skip create: ${dir}" \
        || run_command "Create ${label}" "$PYTHON_BIN" create_poisoned_set.py "${common_args[@]}"
      ;;
    train)
      phase_complete train "$dir" "$arch" \
        && echo "Skip train: ${dir}" \
        || run_command "Train ${label}" "$PYTHON_BIN" train_on_poisoned_set.py "${common_args[@]}"
      ;;
    source)
      phase_complete source "$dir" "$arch" \
        && echo "Skip source test: ${dir}" \
        || run_command "Source test ${label}" "$PYTHON_BIN" test_model.py "${common_args[@]}"
      ;;
    transfer)
      phase_complete transfer "$dir" "$arch" \
        && echo "Skip STL-10 test: ${dir}" \
        || run_command "STL-10 transfer ${label}" "$PYTHON_BIN" test_stl10.py "${common_args[@]}"
      ;;
    defense)
      for defense in $DEFENSE_LIST; do
        phase_complete defense "$dir" "$arch" "$defense" \
          && echo "Skip ${defense}: ${dir}" \
          || run_command "Defense ${defense}, ${label}" \
            "$PYTHON_BIN" other_defense.py "${common_args[@]}" -defense="$defense"
      done
      ;;
  esac
}

run_chunk() {
  local model="$1"
  local rate="$2"
  local alpha_values="$3"
  local -a alphas
  local phase
  local alpha
  read -r -a alphas <<< "$alpha_values"

  for phase in $(requested_phases); do
    for alpha in "${alphas[@]}"; do
      run_config_phase "$phase" "$model" "$rate" "$alpha"
    done
  done
}

echo "============================================================"
echo "CIFAR-10 BELT matched rates: single-GPU server"
echo "Server share    : 4 configurations"
echo "GPU             : ${DEVICE}"
echo "Assignments     : mobilenetv2|0.005|0.30 0.35"
echo "                  vgg19_bn|0.002|0.30 0.35"
echo "Phase           : ${PHASE}"
echo "Skip existing   : ${SKIP_EXISTING}"
echo "Defenses        : ${DEFENSE_LIST}"
echo "Dry run         : ${DRY_RUN}"
echo "Log             : ${LOG_FILE}"
echo "============================================================"

run_chunk mobilenetv2 0.005 "0.30 0.35"
run_chunk vgg19_bn 0.002 "0.30 0.35"

echo "Single-GPU server assignment completed successfully."
