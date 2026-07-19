#!/usr/bin/env bash

# Tiny-ImageNet / BELT final 3 x 6 factorial grid for one architecture.
# MODEL defaults to resnet18 and may also be mobilenetv2 or vgg19_bn.
#
# The six workers are statically assigned. There is no round-robin scheduler:
#   part 0 / GPU 0: rate=0.002, alpha=0.10,0.15,0.20
#   part 1 / GPU 1: rate=0.002, alpha=0.25,0.30,0.35
#   part 2 / GPU 2: rate=0.005, alpha=0.10,0.15,0.20
#   part 3 / GPU 3: rate=0.005, alpha=0.25,0.30,0.35
#   part 4 / GPU 4: rate=0.010, alpha=0.10,0.15,0.20
#   part 5 / GPU 5: rate=0.010, alpha=0.25,0.30,0.35
#
# Each worker processes only its three configurations and never changes GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/root/anaconda3/envs/backtool/bin/python" ]; then
    PYTHON_BIN="/root/anaconda3/envs/backtool/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set}"

MODEL="${MODEL:-resnet18}"
case "$MODEL" in
  resnet18) ARCH_NAME="ResNet18_tiny_imagenet" ;;
  mobilenetv2) ARCH_NAME="mobilenetv2_tiny_imagenet" ;;
  vgg19_bn) ARCH_NAME="vgg19_bn_tiny_imagenet" ;;
  *)
    echo "Unsupported MODEL=${MODEL}; use resnet18|mobilenetv2|vgg19_bn." >&2
    exit 2
    ;;
esac

PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/tiny-target-domain-qwen-full-organized}"
LOG_DIR="${LOG_DIR:-logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

case "$PHASE" in
  all|create|train|source|test|imgv2|transfer|target|qwen|defense|defenses)
    ;;
  *)
    echo "Unsupported PHASE=${PHASE}." >&2
    echo "Use all|create|train|source|imgv2|qwen|defense." >&2
    exit 2
    ;;
esac

read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ARRAY[@]}" -ne 6 ]; then
  echo "GPU_IDS must contain exactly six GPU ids; got ${#GPU_ARRAY[@]}: ${GPU_IDS}" >&2
  exit 2
fi

if [ "$DRY_RUN" != "1" ]; then
  case "$PHASE" in
    all|imgv2|transfer|target)
      if [ ! -d "$TARGET_DOMAIN_DIR" ]; then
        echo "ImageNetV2-Tiny directory not found: ${TARGET_DOMAIN_DIR}" >&2
        exit 2
      fi
      ;;
  esac
  if [ "$RUN_QWEN" = "1" ]; then
    case "$PHASE" in
      all|qwen)
        if [ ! -d "$QWEN_TARGET_DOMAIN_DIR" ]; then
          echo "Qwen target-domain directory not found: ${QWEN_TARGET_DOMAIN_DIR}" >&2
          exit 2
        fi
        ;;
    esac
  fi
fi

poison_dir() {
  local rate="$1"
  local alpha="$2"
  printf '%s/tiny_imagenet/belt_%.3f_alpha=%.3f_cover=0.500_mask=0.200_poison_seed=2333_arch=%s' \
    "$POISONED_TRAIN_SET_ROOT" "$rate" "$alpha" "$ARCH_NAME"
}

phase_complete() {
  local phase="$1"
  local dir="$2"
  local defense="${3:-}"
  local train_json="${dir}/train_results_seed=2333.json"
  local model_file="${dir}/${ARCH_NAME}_belt_aug_model_seed=2333.pt"

  if [ "$SKIP_EXISTING" != "1" ]; then
    return 1
  fi

  case "$phase" in
    create)
      [ -d "${dir}/imgs" ] \
        && [ -f "${dir}/labels" ] \
        && [ -f "${dir}/poison_indices" ] \
        && [ -f "${dir}/cover_indices" ] \
        && [ -f "${dir}/pmarks" ]
      ;;
    train)
      [ -f "$model_file" ] \
        && [ -f "$train_json" ] \
        && [ -f "${dir}/poison_indices" ] \
        && [ "$model_file" -nt "${dir}/poison_indices" ] \
        && grep -Eq '"checkpoint_selection"[[:space:]]*:[[:space:]]*"final_epoch"' "$train_json"
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
    qwen)
      [ -f "$model_file" ] \
        && [ -f "${dir}/test_tiny_target_domain_qwen_results.txt" ] \
        && [ "${dir}/test_tiny_target_domain_qwen_results.txt" -nt "$model_file" ]
      ;;
    defense)
      case "$defense" in
        SentiNet)
          [ -f "$model_file" ] \
            && [ -f "${dir}/sentinet_defense_results.json" ] \
            && [ "${dir}/sentinet_defense_results.json" -nt "$model_file" ]
          ;;
        STRIP)
          [ -f "$model_file" ] \
            && [ -f "${dir}/strip_defense_results.json" ] \
            && [ "${dir}/strip_defense_results.json" -nt "$model_file" ]
          ;;
        ScaleUp)
          [ -f "$model_file" ] \
            && [ -f "${dir}/scaleup_defense_results.json" ] \
            && [ "${dir}/scaleup_defense_results.json" -nt "$model_file" ]
          ;;
        IBD_PSC)
          [ -f "$model_file" ] \
            && [ -f "${dir}/ibd_psc_defense_results.json" ] \
            && [ "${dir}/ibd_psc_defense_results.json" -nt "$model_file" ]
          ;;
        *) return 1 ;;
      esac
      ;;
    *)
      return 1
      ;;
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

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] skipped"
    return 0
  fi

  "$@"
}

run_config_phase() {
  local phase="$1"
  local gpu="$2"
  local rate="$3"
  local alpha="$4"
  local dir
  local defense
  dir="$(poison_dir "$rate" "$alpha")"

  local common_args=(
    -dataset=tiny_imagenet
    -model="$MODEL"
    -devices="$gpu"
    -poison_type=belt
    -poison_rate="$rate"
    -cover_rate=0.5
    -mask_rate=0.2
    -alpha="$alpha"
  )
  local label="part GPU=${gpu}: rate=${rate}, alpha=${alpha}"

  case "$phase" in
    create)
      if phase_complete create "$dir"; then
        echo "Skip create: ${dir}"
      else
        run_command "Create ${label}" \
          "$PYTHON_BIN" create_poisoned_set.py "${common_args[@]}"
      fi
      ;;
    train)
      if phase_complete train "$dir"; then
        echo "Skip corrected final-epoch train: ${dir}"
      else
        run_command "Train final epoch ${label}" \
          "$PYTHON_BIN" train_on_poisoned_set.py "${common_args[@]}" -no_normalize
      fi
      ;;
    source)
      if phase_complete source "$dir"; then
        echo "Skip source test: ${dir}"
      else
        run_command "Source test ${label}" \
          "$PYTHON_BIN" test_model.py "${common_args[@]}" -no_normalize
      fi
      ;;
    imgv2)
      if phase_complete imgv2 "$dir"; then
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
    qwen)
      if [ "$RUN_QWEN" != "1" ]; then
        echo "Skip Qwen by RUN_QWEN=0: ${dir}"
      elif phase_complete qwen "$dir"; then
        echo "Skip Qwen test: ${dir}"
      else
        run_command "Qwen transfer ${label}" \
          "$PYTHON_BIN" test_tiny_target_domain_qwen.py \
          "${common_args[@]}" \
          -source_dataset=tiny_imagenet \
          -target_domain_dir="$QWEN_TARGET_DOMAIN_DIR" \
          -no_normalize
      fi
      ;;
    defense)
      for defense in $DEFENSE_LIST; do
        if phase_complete defense "$dir" "$defense"; then
          echo "Skip ${defense}: ${dir}"
        else
          run_command "Defense ${defense} ${label}" \
            "$PYTHON_BIN" other_defense.py \
            "${common_args[@]}" \
            -defense="$defense" \
            -no_normalize
        fi
      done
      ;;
  esac
}

requested_phases() {
  case "$PHASE" in
    all)
      echo "create train source imgv2 qwen defense"
      ;;
    test)
      echo "source"
      ;;
    transfer|target)
      echo "imgv2"
      ;;
    defenses)
      echo "defense"
      ;;
    *)
      echo "$PHASE"
      ;;
  esac
}

run_part() {
  local part="$1"
  local gpu="$2"
  local rate="$3"
  shift 3
  local -a alphas=("$@")
  local phase
  local alpha

  echo "============================================================"
  echo "Part ${part}: fixed GPU ${gpu}, poison rate ${rate}"
  echo "Alphas: ${alphas[*]}"
  echo "============================================================"

  for phase in $(requested_phases); do
    echo
    echo "----- Part ${part} / GPU ${gpu} / phase ${phase} -----"
    for alpha in "${alphas[@]}"; do
      if ! run_config_phase "$phase" "$gpu" "$rate" "$alpha"; then
        echo "Part ${part} failed: phase=${phase}, rate=${rate}, alpha=${alpha}" >&2
        if [ "$STOP_ON_FAIL" = "1" ]; then
          return 1
        fi
      fi
    done
  done
}

echo "============================================================"
echo "Tiny-ImageNet ${MODEL} BELT final grid: fixed six-part layout"
echo "============================================================"
echo "repo              : ${REPO_ROOT}"
echo "python            : ${PYTHON_BIN}"
echo "model             : ${MODEL} (${ARCH_NAME})"
echo "result root       : ${POISONED_TRAIN_SET_ROOT}"
echo "phase             : ${PHASE}"
echo "GPU ids           : ${GPU_IDS}"
echo "skip existing     : ${SKIP_EXISTING}"
echo "run Qwen          : ${RUN_QWEN}"
echo "defenses          : ${DEFENSE_LIST}"
echo "ImageNetV2 target : ${TARGET_DOMAIN_DIR}"
echo "Qwen target       : ${QWEN_TARGET_DOMAIN_DIR}"
echo "dry run           : ${DRY_RUN}"
echo "============================================================"
echo "Part 0 -> GPU ${GPU_ARRAY[0]} -> rate 0.002 -> alpha 0.10 0.15 0.20"
echo "Part 1 -> GPU ${GPU_ARRAY[1]} -> rate 0.002 -> alpha 0.25 0.30 0.35"
echo "Part 2 -> GPU ${GPU_ARRAY[2]} -> rate 0.005 -> alpha 0.10 0.15 0.20"
echo "Part 3 -> GPU ${GPU_ARRAY[3]} -> rate 0.005 -> alpha 0.25 0.30 0.35"
echo "Part 4 -> GPU ${GPU_ARRAY[4]} -> rate 0.010 -> alpha 0.10 0.15 0.20"
echo "Part 5 -> GPU ${GPU_ARRAY[5]} -> rate 0.010 -> alpha 0.25 0.30 0.35"

declare -a PIDS=()
declare -a PART_LOGS=()

launch_part() {
  local part="$1"
  local gpu="$2"
  local rate="$3"
  shift 3
  local log_file="${LOG_DIR}/tiny_${MODEL}_belt_grid_part${part}_gpu${gpu}_${TIMESTAMP}.log"
  PART_LOGS+=("$log_file")
  run_part "$part" "$gpu" "$rate" "$@" >"$log_file" 2>&1 &
  PIDS+=("$!")
  echo "Launched part ${part} on fixed GPU ${gpu}; log: ${log_file}"
}

# Explicit static assignment: one poison rate owns exactly two GPUs.
launch_part 0 "${GPU_ARRAY[0]}" 0.002 0.10 0.15 0.20
launch_part 1 "${GPU_ARRAY[1]}" 0.002 0.25 0.30 0.35
launch_part 2 "${GPU_ARRAY[2]}" 0.005 0.10 0.15 0.20
launch_part 3 "${GPU_ARRAY[3]}" 0.005 0.25 0.30 0.35
launch_part 4 "${GPU_ARRAY[4]}" 0.010 0.10 0.15 0.20
launch_part 5 "${GPU_ARRAY[5]}" 0.010 0.25 0.30 0.35

failed=0
for index in 0 1 2 3 4 5; do
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

echo "All six fixed GPU parts completed successfully."
