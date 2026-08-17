#!/usr/bin/env bash

# Tiny-ImageNet / BELT missing final-checkpoint grid on eight fixed GPUs.
#
# This script follows the style and command sequence of:
#   run/run_tiny_imagenet_resnet18_belt_final_grid_6gpu.sh
#
# Scope: internal rates 0.020 and 0.100 (paper rates 1% and 5%), three
# architectures, and alpha 0.10..0.35: 2 x 3 x 6 = 36 configurations.
# Tiny internal rate 0.010, CIFAR-10, Qwen, and corruption are not run.
#
# Each part owns a disjoint static configuration list and never changes GPU:
# parts 0..3 have five configurations; parts 4..7 have four configurations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_INPUT="${REPO_ROOT:-/workspace/backdoor-toolbox}"
if [ ! -d "$REPO_ROOT_INPUT" ]; then
  echo "Repository root not found: ${REPO_ROOT_INPUT}" >&2
  exit 2
fi
REPO_ROOT="$(cd "$REPO_ROOT_INPUT" && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/root/anaconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set}"

PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/run_logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

case "$PHASE" in
  all|create|train|source|test|imgv2|transfer|target|defense|defenses|eval|train_eval)
    ;;
  *)
    echo "Unsupported PHASE=${PHASE}." >&2
    echo "Use all|create|train|source|imgv2|defense|eval|train_eval." >&2
    exit 2
    ;;
esac

read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ARRAY[@]}" -ne 8 ]; then
  echo "GPU_IDS must contain exactly eight GPU ids; got ${#GPU_ARRAY[@]}: ${GPU_IDS}" >&2
  exit 2
fi

declare -A SEEN_GPU=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [ -n "${SEEN_GPU[$gpu]:-}" ]; then
    echo "GPU_IDS contains duplicate id ${gpu}; each part requires a distinct GPU." >&2
    exit 2
  fi
  SEEN_GPU["$gpu"]=1
done

for required_script in \
  create_poisoned_set.py \
  train_on_poisoned_set.py \
  test_model.py \
  test_tiny_target_domain.py \
  other_defense.py \
  train_belt.py; do
  if [ ! -f "${REPO_ROOT}/${required_script}" ]; then
    echo "Required repository file not found: ${REPO_ROOT}/${required_script}" >&2
    exit 2
  fi
done

if ! grep -Eq "['\"]checkpoint_selection['\"][[:space:]]*:[[:space:]]*['\"]final_epoch['\"]" \
  "${REPO_ROOT}/train_belt.py"; then
  echo "Refusing to run: train_belt.py lacks the corrected final-epoch signature." >&2
  exit 3
fi

if [ "$DRY_RUN" != "1" ]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
  fi
  if [ ! -d "${REPO_ROOT}/data/Tiny-imagenet/tiny-imagenet-200" ]; then
    echo "Tiny-ImageNet training data not found under ${REPO_ROOT}/data/Tiny-imagenet." >&2
    exit 2
  fi
  case "$PHASE" in
    all|imgv2|transfer|target|eval|train_eval)
      if [ ! -d "$TARGET_DOMAIN_DIR" ]; then
        echo "ImageNetV2-Tiny directory not found: ${TARGET_DOMAIN_DIR}" >&2
        exit 2
      fi
      ;;
  esac
fi

arch_name_for_model() {
  case "$1" in
    resnet18) printf '%s' "ResNet18_tiny_imagenet" ;;
    mobilenetv2) printf '%s' "mobilenetv2_tiny_imagenet" ;;
    vgg19_bn) printf '%s' "vgg19_bn_tiny_imagenet" ;;
    *)
      echo "Unsupported model: $1" >&2
      return 2
      ;;
  esac
}

poison_dir() {
  local model="$1"
  local rate="$2"
  local alpha="$3"
  local arch_name
  arch_name="$(arch_name_for_model "$model")"
  printf '%s/tiny_imagenet/belt_%.3f_alpha=%.3f_cover=0.500_mask=0.200_poison_seed=2333_arch=%s' \
    "$POISONED_TRAIN_SET_ROOT" "$rate" "$alpha" "$arch_name"
}

phase_complete() {
  local phase="$1"
  local dir="$2"
  local arch_name="$3"
  local defense="${4:-}"
  local model_file="${dir}/${arch_name}_belt_aug_model_seed=2333.pt"

  if [ "$SKIP_EXISTING" != "1" ]; then
    return 1
  fi

  case "$phase" in
    create)
      [ -f "${dir}/labels" ] \
        && [ -f "${dir}/poison_indices" ] \
        && [ -f "${dir}/cover_indices" ] \
        && [ -f "${dir}/pmarks" ] \
        && [ -f "${dir}/belt_trigger.pt" ]
      ;;
    train)
      corrected_train_complete "$dir" "$arch_name"
      ;;
    source)
      [ -f "$model_file" ] \
        && [ -f "${dir}/test_results_seed=2333.json" ] \
        && [ "${dir}/test_results_seed=2333.json" -nt "$model_file" ]
      ;;
    imgv2)
      [ -f "$model_file" ] \
        && [ -f "${dir}/test_tiny_target_domain_results.txt" ] \
        && [ "${dir}/test_tiny_target_domain_results.txt" -nt "$model_file" ]
      ;;
    defense)
      local output_file
      case "$defense" in
        SentiNet) output_file="sentinet_defense_results.json" ;;
        STRIP) output_file="strip_defense_results.json" ;;
        ScaleUp) output_file="scaleup_defense_results.json" ;;
        IBD_PSC) output_file="ibd_psc_defense_results.json" ;;
        *) return 1 ;;
      esac
      [ -f "$model_file" ] \
        && [ -f "${dir}/${output_file}" ] \
        && [ "${dir}/${output_file}" -nt "$model_file" ]
      ;;
    *)
      return 1
      ;;
  esac
}

corrected_train_complete() {
  local dir="$1"
  local arch_name="$2"
  local train_json="${dir}/train_results_seed=2333.json"
  local model_file="${dir}/${arch_name}_belt_aug_model_seed=2333.pt"
  local best_model_file="${dir}/${arch_name}_belt_aug_model_seed=2333_best.pt"

  [ -f "$model_file" ] \
    && [ -f "$best_model_file" ] \
    && [ -f "$train_json" ] \
    && [ -f "${dir}/poison_indices" ] \
    && [ "$model_file" -nt "${dir}/poison_indices" ] \
    && grep -Eq '"checkpoint_selection"[[:space:]]*:[[:space:]]*"final_epoch"' "$train_json" \
    && "$PYTHON_BIN" - "$train_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
raise SystemExit(0 if record.get("final_epoch") == record.get("epochs") else 1)
PY
}

run_command() {
  local description="$1"
  shift

  echo
  echo ">>> ${description}"
  printf '    '
  printf '%q ' "$@"
  printf '\n'

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] skipped"
    return 0
  fi

  "$@"
}

run_config_phase() {
  local phase="$1"
  local gpu="$2"
  local model="$3"
  local rate="$4"
  local alpha="$5"
  local arch_name
  local dir
  local defense
  arch_name="$(arch_name_for_model "$model")"
  dir="$(poison_dir "$model" "$rate" "$alpha")"

  local common_args=(
    -dataset=tiny_imagenet
    -model="$model"
    -devices="$gpu"
    -poison_type=belt
    -poison_rate="$rate"
    -cover_rate=0.5
    -mask_rate=0.2
    -alpha="$alpha"
  )
  local label="GPU=${gpu}: model=${model}, rate=${rate}, alpha=${alpha}"

  if [ "$DRY_RUN" != "1" ]; then
    case "$phase" in
      source|imgv2|defense)
        if ! corrected_train_complete "$dir" "$arch_name"; then
          echo "Refusing ${phase}: corrected final-epoch training is incomplete: ${dir}" >&2
          echo "Run PHASE=train (or PHASE=all/train_eval) first." >&2
          return 4
        fi
        ;;
    esac
  fi

  case "$phase" in
    create)
      if phase_complete create "$dir" "$arch_name"; then
        echo "Skip create: ${dir}"
      else
        run_command "Create ${label}" \
          "$PYTHON_BIN" create_poisoned_set.py "${common_args[@]}"
      fi
      ;;
    train)
      if phase_complete train "$dir" "$arch_name"; then
        echo "Skip corrected final-epoch train: ${dir}"
      else
        run_command "Train final epoch ${label}" \
          "$PYTHON_BIN" train_on_poisoned_set.py "${common_args[@]}" -no_normalize
      fi
      ;;
    source)
      if phase_complete source "$dir" "$arch_name"; then
        echo "Skip source test: ${dir}"
      else
        run_command "Source test ${label}" \
          "$PYTHON_BIN" test_model.py "${common_args[@]}" -no_normalize
      fi
      ;;
    imgv2)
      if phase_complete imgv2 "$dir" "$arch_name"; then
        echo "Skip ImageNetV2-Tiny test: ${dir}"
      else
        run_command "ImageNetV2-Tiny transfer ${label}" \
          "$PYTHON_BIN" test_tiny_target_domain.py \
          "${common_args[@]}" \
          -source_dataset=tiny_imagenet \
          -target_domain_dir="$TARGET_DOMAIN_DIR" \
          -no_normalize
      fi
      ;;
    defense)
      local defense_failed=0
      for defense in $DEFENSE_LIST; do
        if phase_complete defense "$dir" "$arch_name" "$defense"; then
          echo "Skip ${defense}: ${dir}"
        else
          if ! run_command "Defense ${defense} ${label}" \
              "$PYTHON_BIN" other_defense.py \
              "${common_args[@]}" \
              -defense="$defense" \
              -no_normalize; then
            defense_failed=1
            if [ "$STOP_ON_FAIL" = "1" ]; then
              return 1
            fi
          fi
        fi
      done
      return "$defense_failed"
      ;;
  esac
}

requested_phases() {
  case "$PHASE" in
    all) echo "create train source imgv2 defense" ;;
    eval) echo "source imgv2 defense" ;;
    train_eval) echo "train source imgv2 defense" ;;
    test) echo "source" ;;
    transfer|target) echo "imgv2" ;;
    defenses) echo "defense" ;;
    *) echo "$PHASE" ;;
  esac
}

run_part() {
  local part="$1"
  local gpu="$2"
  shift 2
  local -a configs=("$@")
  local phase
  local config
  local model
  local rate
  local alpha
  local failed=0

  echo "============================================================"
  echo "Part ${part}: fixed GPU ${gpu}; ${#configs[@]} configurations"
  printf '  %s\n' "${configs[@]}"
  echo "============================================================"

  for phase in $(requested_phases); do
    echo
    echo "----- Part ${part} / GPU ${gpu} / phase ${phase} -----"
    for config in "${configs[@]}"; do
      IFS='|' read -r model rate alpha <<< "$config"
      if ! run_config_phase "$phase" "$gpu" "$model" "$rate" "$alpha"; then
        echo "Part ${part} failed: phase=${phase}, config=${config}" >&2
        failed=1
        if [ "$STOP_ON_FAIL" = "1" ]; then
          return 1
        fi
      fi
    done
  done
  return "$failed"
}

# Explicit static assignment. A config token is MODEL|INTERNAL_RATE|ALPHA.
# The mix keeps all three architectures reasonably balanced across GPUs.
PART0=(
  'resnet18|0.020|0.100'
  'mobilenetv2|0.020|0.200'
  'vgg19_bn|0.020|0.300'
  'mobilenetv2|0.100|0.100'
  'vgg19_bn|0.100|0.200'
)
PART1=(
  'resnet18|0.020|0.150'
  'mobilenetv2|0.020|0.250'
  'vgg19_bn|0.020|0.350'
  'mobilenetv2|0.100|0.150'
  'vgg19_bn|0.100|0.250'
)
PART2=(
  'resnet18|0.020|0.200'
  'mobilenetv2|0.020|0.300'
  'resnet18|0.100|0.100'
  'mobilenetv2|0.100|0.200'
  'vgg19_bn|0.100|0.300'
)
PART3=(
  'resnet18|0.020|0.250'
  'mobilenetv2|0.020|0.350'
  'resnet18|0.100|0.150'
  'mobilenetv2|0.100|0.250'
  'vgg19_bn|0.100|0.350'
)
PART4=(
  'resnet18|0.020|0.300'
  'vgg19_bn|0.020|0.100'
  'resnet18|0.100|0.200'
  'mobilenetv2|0.100|0.300'
)
PART5=(
  'resnet18|0.020|0.350'
  'vgg19_bn|0.020|0.150'
  'resnet18|0.100|0.250'
  'mobilenetv2|0.100|0.350'
)
PART6=(
  'mobilenetv2|0.020|0.100'
  'vgg19_bn|0.020|0.200'
  'resnet18|0.100|0.300'
  'vgg19_bn|0.100|0.100'
)
PART7=(
  'mobilenetv2|0.020|0.150'
  'vgg19_bn|0.020|0.250'
  'resnet18|0.100|0.350'
  'vgg19_bn|0.100|0.150'
)

validate_static_grid() {
  local -a all_configs=(
    "${PART0[@]}" "${PART1[@]}" "${PART2[@]}" "${PART3[@]}"
    "${PART4[@]}" "${PART5[@]}" "${PART6[@]}" "${PART7[@]}"
  )
  local config
  local model
  local rate
  local alpha
  local extra
  local key
  declare -A seen_config=()
  declare -A cell_count=()

  if [ "${#all_configs[@]}" -ne 36 ]; then
    echo "Internal error: static grid has ${#all_configs[@]} configs; expected 36." >&2
    return 2
  fi
  for config in "${all_configs[@]}"; do
    IFS='|' read -r model rate alpha extra <<< "$config"
    if [ -n "${extra:-}" ]; then
      echo "Internal error: malformed config token: ${config}" >&2
      return 2
    fi
    case "$model" in resnet18|mobilenetv2|vgg19_bn) ;; *) return 2 ;; esac
    case "$rate" in 0.020|0.100) ;; *) return 2 ;; esac
    case "$alpha" in 0.100|0.150|0.200|0.250|0.300|0.350) ;; *) return 2 ;; esac
    if [ -n "${seen_config[$config]:-}" ]; then
      echo "Internal error: duplicate config: ${config}" >&2
      return 2
    fi
    seen_config["$config"]=1
    key="${model}|${rate}"
    cell_count["$key"]=$(( ${cell_count[$key]:-0} + 1 ))
  done
  for model in resnet18 mobilenetv2 vgg19_bn; do
    for rate in 0.020 0.100; do
      key="${model}|${rate}"
      if [ "${cell_count[$key]:-0}" -ne 6 ]; then
        echo "Internal error: ${key} has ${cell_count[$key]:-0} alphas; expected 6." >&2
        return 2
      fi
    done
  done
}

validate_static_grid

echo "============================================================"
echo "Tiny-ImageNet BELT missing final grid: fixed eight-part layout"
echo "============================================================"
echo "repo              : ${REPO_ROOT}"
echo "python            : ${PYTHON_BIN}"
echo "result root       : ${POISONED_TRAIN_SET_ROOT}"
echo "phase             : ${PHASE}"
echo "GPU ids           : ${GPU_IDS}"
echo "skip existing     : ${SKIP_EXISTING}"
echo "defenses          : ${DEFENSE_LIST}"
echo "ImageNetV2 target : ${TARGET_DOMAIN_DIR}"
echo "dry run           : ${DRY_RUN}"
echo "logs              : ${LOG_DIR}"
echo "============================================================"
PART_COUNTS=(
  "${#PART0[@]}" "${#PART1[@]}" "${#PART2[@]}" "${#PART3[@]}"
  "${#PART4[@]}" "${#PART5[@]}" "${#PART6[@]}" "${#PART7[@]}"
)
for part in 0 1 2 3 4 5 6 7; do
  echo "Part ${part} -> GPU ${GPU_ARRAY[$part]} -> ${PART_COUNTS[$part]} configs"
done

declare -a PIDS=()
declare -a PART_LOGS=()

launch_part() {
  local part="$1"
  local gpu="$2"
  shift 2
  local log_file="${LOG_DIR}/tiny_belt_missing_part${part}_gpu${gpu}_${TIMESTAMP}.log"
  PART_LOGS+=("$log_file")
  run_part "$part" "$gpu" "$@" >"$log_file" 2>&1 &
  PIDS+=("$!")
  echo "Launched part ${part} on fixed GPU ${gpu}; log: ${log_file}"
}

launch_part 0 "${GPU_ARRAY[0]}" "${PART0[@]}"
launch_part 1 "${GPU_ARRAY[1]}" "${PART1[@]}"
launch_part 2 "${GPU_ARRAY[2]}" "${PART2[@]}"
launch_part 3 "${GPU_ARRAY[3]}" "${PART3[@]}"
launch_part 4 "${GPU_ARRAY[4]}" "${PART4[@]}"
launch_part 5 "${GPU_ARRAY[5]}" "${PART5[@]}"
launch_part 6 "${GPU_ARRAY[6]}" "${PART6[@]}"
launch_part 7 "${GPU_ARRAY[7]}" "${PART7[@]}"

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
  echo "One or more fixed GPU parts failed. Inspect the part logs above." >&2
  exit 1
fi

echo "All eight fixed GPU parts completed successfully."
echo "Validate with:"
echo "  ${PYTHON_BIN} ${SCRIPT_DIR}/validate_belt_tiny_results.py --repo-root ${REPO_ROOT} --include-existing-0p5"
