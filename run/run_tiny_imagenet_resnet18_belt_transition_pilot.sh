#!/usr/bin/env bash

# Tiny-ImageNet / ResNet18 BELT transition-region pilot.
#
# Fixed grid (8 configurations):
#   poison_rate = 0.002, 0.005
#   alpha       = 0.100, 0.200, 0.300, 0.400
#   cover_rate  = 0.500
#   mask_rate   = 0.200
#
# The pilot intentionally runs only create, train, source test, and
# ImageNetV2-Tiny transfer test. Defenses should be run only after this grid
# identifies non-saturated, source-valid configurations.

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
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL="${PARALLEL:-0}"
GPU_IDS="${GPU_IDS:-${DEVICES:-0}}"
MAX_JOBS="${MAX_JOBS:-}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"

case "$PHASE" in
  all|create|train|source|test|transfer|target)
    ;;
  *)
    echo "Unsupported PHASE=${PHASE}. Use all|create|train|source|transfer." >&2
    exit 2
    ;;
esac

read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 2
fi

poison_dir() {
  local rate="$1"
  local alpha="$2"
  printf '%s/tiny_imagenet/belt_%.3f_alpha=%.3f_cover=0.500_mask=0.200_poison_seed=2333_arch=ResNet18_tiny_imagenet' \
    "$POISONED_TRAIN_SET_ROOT" "$rate" "$alpha"
}

selected_configs() {
  local rate
  local alpha
  for rate in 0.002 0.005; do
    for alpha in 0.100 0.200 0.300 0.400; do
      echo "${rate}|${alpha}|$(poison_dir "$rate" "$alpha")"
    done
  done
}

gpu_for_index() {
  local index="$1"
  echo "${GPU_ARRAY[$((index % ${#GPU_ARRAY[@]}))]}"
}

parallel_jobs() {
  if [ "$PARALLEL" != "1" ]; then
    echo 1
  elif [ -n "$MAX_JOBS" ]; then
    echo "$MAX_JOBS"
  else
    echo "${#GPU_ARRAY[@]}"
  fi
}

phase_done() {
  local phase="$1"
  local dir="$2"
  local model_file="${dir}/ResNet18_tiny_imagenet_belt_aug_model_seed=2333.pt"
  local result_file="${dir}/train_results_seed=2333.json"

  case "$phase" in
    create)
      [ -d "${dir}/imgs" ] && [ -f "${dir}/labels" ] && \
        [ -f "${dir}/poison_indices" ] && [ -f "${dir}/pmarks" ]
      ;;
    train)
      [ -f "$model_file" ] && [ -f "$result_file" ] && \
        grep -q '"checkpoint_selection": "final_epoch"' "$result_file"
      ;;
    source)
      compgen -G "${dir}/test_results_seed=2333*.json" >/dev/null
      ;;
    transfer)
      [ -f "${dir}/test_tiny_target_domain_results.txt" ]
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
  local rate="$2"
  local alpha="$3"
  local dir="$4"
  local gpu="$5"
  local label="Tiny-ImageNet ResNet18 BELT rate=${rate} alpha=${alpha}"
  local common_args=(
    -dataset=tiny_imagenet
    -model=resnet18
    -devices="$gpu"
    -poison_type=belt
    -poison_rate="$rate"
    -cover_rate=0.5
    -mask_rate=0.2
    -alpha="$alpha"
  )

  if [ "$SKIP_EXISTING" = "1" ] && phase_done "$phase" "$dir"; then
    echo "Skip ${phase}: already complete: ${dir}"
    return 0
  fi

  case "$phase" in
    create)
      run_command "Create: ${label}" \
        "$PYTHON_BIN" create_poisoned_set.py "${common_args[@]}"
      ;;
    train)
      run_command "Train final checkpoint: ${label}" \
        "$PYTHON_BIN" train_on_poisoned_set.py "${common_args[@]}" -no_normalize
      ;;
    source)
      run_command "Source test: ${label}" \
        "$PYTHON_BIN" test_model.py "${common_args[@]}" -no_normalize
      ;;
    transfer)
      run_command "ImageNetV2-Tiny transfer: ${label}" \
        "$PYTHON_BIN" test_tiny_target_domain.py \
        "${common_args[@]}" \
        -source_dataset=tiny_imagenet \
        -target_domain_dir="$TARGET_DOMAIN_DIR" \
        -no_normalize
      ;;
  esac
}

wait_batch() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
  return "$failed"
}

run_phase_group() {
  local phase="$1"
  local index=0
  local active=0
  local max_jobs
  local rate alpha dir gpu
  local -a pids=()
  max_jobs="$(parallel_jobs)"

  while IFS='|' read -r rate alpha dir; do
    gpu="$(gpu_for_index "$index")"
    index=$((index + 1))

    if [ "$max_jobs" -gt 1 ] && [ "$DRY_RUN" != "1" ]; then
      (run_config_phase "$phase" "$rate" "$alpha" "$dir" "$gpu") &
      pids+=("$!")
      active=$((active + 1))
      if [ "$active" -ge "$max_jobs" ]; then
        wait_batch "${pids[@]}"
        pids=()
        active=0
      fi
    else
      run_config_phase "$phase" "$rate" "$alpha" "$dir" "$gpu"
    fi
  done < <(selected_configs)

  if [ "$active" -gt 0 ]; then
    wait_batch "${pids[@]}"
  fi
}

echo "============================================================"
echo "Tiny-ImageNet ResNet18 BELT transition-region pilot"
echo "============================================================"
echo "repo          : ${REPO_ROOT}"
echo "python        : ${PYTHON_BIN}"
echo "result root   : ${POISONED_TRAIN_SET_ROOT}"
echo "phase         : ${PHASE}"
echo "gpu ids       : ${GPU_IDS}"
echo "parallel      : ${PARALLEL}"
echo "skip existing : ${SKIP_EXISTING}"
echo "target domain : ${TARGET_DOMAIN_DIR}"
echo "dry run       : ${DRY_RUN}"
echo "============================================================"

config_count=0
echo
echo "Selected configs:"
while IFS='|' read -r rate alpha dir; do
  gpu="$(gpu_for_index "$config_count")"
  config_count=$((config_count + 1))
  echo "CONFIG ${config_count}: rate=${rate} alpha=${alpha} gpu=${gpu} -> ${dir}"
done < <(selected_configs)
echo "Total configs: ${config_count}"

if [ "$PHASE" = "all" ]; then
  for phase in create train source transfer; do
    echo
    echo "----- ${phase} -----"
    run_phase_group "$phase"
  done
else
  case "$PHASE" in
    test) phase=source ;;
    target) phase=transfer ;;
    *) phase="$PHASE" ;;
  esac
  echo
  echo "----- ${phase} -----"
  run_phase_group "$phase"
fi

echo
echo "Finished Tiny-ImageNet ResNet18 BELT transition-region pilot."
