#!/usr/bin/env bash

# Eight-GPU server: 32 of the 36 CIFAR-10 BELT matched-rate configurations.
# Each fixed GPU receives four configurations. The remaining four configurations
# are assigned to run_cifar10_belt_matched_rates_1gpu_server.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
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

read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ARRAY[@]}" -ne 8 ]; then
  echo "GPU_IDS must contain exactly eight GPU ids; got ${#GPU_ARRAY[@]}: ${GPU_IDS}" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

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
  local gpu="$2"
  local model="$3"
  local rate="$4"
  local alpha="$5"
  local arch
  local dir
  local defense
  arch="$(arch_name "$model")"
  dir="$(poison_dir "$model" "$rate" "$alpha")"
  local common_args=(
    -dataset=cifar10 -model="$model" -devices="$gpu" -poison_type=belt
    -poison_rate="$rate" -cover_rate=0.5 -mask_rate=0.2 -alpha="$alpha"
  )
  local label="model=${model}, rate=${rate}, alpha=${alpha}, GPU=${gpu}"

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
  local gpu="$3"
  local alpha_values="$4"
  local -a alphas
  local phase
  local alpha
  read -r -a alphas <<< "$alpha_values"
  for phase in $(requested_phases); do
    for alpha in "${alphas[@]}"; do
      run_config_phase "$phase" "$gpu" "$model" "$rate" "$alpha"
    done
  done
}

run_part() {
  local part="$1"
  local gpu="$2"
  shift 2
  local -a chunks=("$@")
  local chunk
  local model
  local rate
  local alpha_values

  echo "============================================================"
  echo "Eight-GPU server part ${part}, fixed GPU ${gpu}"
  echo "Assignments: ${chunks[*]}"
  echo "============================================================"

  for chunk in "${chunks[@]}"; do
    IFS='|' read -r model rate alpha_values <<< "$chunk"
    run_chunk "$model" "$rate" "$gpu" "$alpha_values"
  done
}

declare -a PIDS=()
declare -a PART_LOGS=()

launch_part() {
  local part="$1"
  local gpu="$2"
  shift 2
  local log_file="${LOG_DIR}/cifar10_belt_matched_8gpu_part${part}_gpu${gpu}_${TIMESTAMP}.log"
  PART_LOGS+=("$log_file")
  run_part "$part" "$gpu" "$@" >"$log_file" 2>&1 &
  PIDS+=("$!")
  echo "Launched part ${part} on fixed GPU ${gpu}; log: ${log_file}"
}

echo "============================================================"
echo "CIFAR-10 BELT matched rates: eight-GPU server"
echo "Server share    : 32 configurations"
echo "Rates           : 0.002 0.005"
echo "Alpha grid      : 0.10 0.15 0.20 0.25 0.30 0.35"
echo "Architectures   : resnet18 mobilenetv2 vgg19_bn"
echo "GPU ids         : ${GPU_IDS}"
echo "Phase           : ${PHASE}"
echo "Skip existing   : ${SKIP_EXISTING}"
echo "Defenses        : ${DEFENSE_LIST}"
echo "Dry run         : ${DRY_RUN}"
echo "============================================================"

# Each quoted chunk is MODEL|RATE|ALPHA_VALUES.
launch_part 0 "${GPU_ARRAY[0]}" \
  "resnet18|0.002|0.10 0.15" \
  "mobilenetv2|0.005|0.10 0.15"

launch_part 1 "${GPU_ARRAY[1]}" \
  "resnet18|0.002|0.20 0.25" \
  "vgg19_bn|0.005|0.10 0.15"

launch_part 2 "${GPU_ARRAY[2]}" \
  "resnet18|0.002|0.30 0.35" \
  "mobilenetv2|0.005|0.20 0.25"

launch_part 3 "${GPU_ARRAY[3]}" \
  "resnet18|0.005|0.10 0.15" \
  "vgg19_bn|0.002|0.10 0.15"

launch_part 4 "${GPU_ARRAY[4]}" \
  "resnet18|0.005|0.20 0.25" \
  "mobilenetv2|0.002|0.10 0.15"

launch_part 5 "${GPU_ARRAY[5]}" \
  "resnet18|0.005|0.30 0.35" \
  "vgg19_bn|0.002|0.20 0.25"

launch_part 6 "${GPU_ARRAY[6]}" \
  "mobilenetv2|0.002|0.20 0.25" \
  "vgg19_bn|0.005|0.20 0.25"

launch_part 7 "${GPU_ARRAY[7]}" \
  "mobilenetv2|0.002|0.30 0.35" \
  "vgg19_bn|0.005|0.30 0.35"

failed=0
for index in 0 1 2 3 4 5 6 7; do
  if wait "${PIDS[$index]}"; then
    echo "Part ${index} completed: ${PART_LOGS[$index]}"
  else
    echo "Part ${index} FAILED: ${PART_LOGS[$index]}" >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "One or more eight-GPU server parts failed. Inspect the part logs above." >&2
  exit 1
fi

echo "All eight-GPU server parts completed successfully."
