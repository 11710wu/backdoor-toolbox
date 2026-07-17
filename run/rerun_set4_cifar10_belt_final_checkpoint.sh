#!/usr/bin/env bash

# Rerun CIFAR-10 set4 BELT with final-epoch checkpoints.
#
# Defaults:
#   - MicroCNN: poison_rate 0.010 and 0.020
#   - SmallCNN: poison_rate 0.010 and 0.020
#   - ResNet50: poison_rate 0.010 only
#   - alpha: 0.100, 0.200, 0.300
#
# The script does not delete existing results. Before a rerun overwrites the
# standard output files, it copies the old files into a timestamped backup
# directory inside each result folder.
#
# Usage:
#   DRY_RUN=1 bash run/rerun_set4_cifar10_belt_final_checkpoint.sh
#   PHASE=train bash run/rerun_set4_cifar10_belt_final_checkpoint.sh
#   PHASE=all PARALLEL=1 GPU_IDS="0 1 2 3" bash run/rerun_set4_cifar10_belt_final_checkpoint.sh

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

DATASET="cifar10"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
BACKUP_OLD_RESULTS="${BACKUP_OLD_RESULTS:-1}"
GPU_IDS="${GPU_IDS:-${DEVICES:-0}}"
PARALLEL="${PARALLEL:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"
BACKUP_TAG="${BACKUP_TAG:-belt_best_ckpt_backup_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_set4_cifar10_belt_final_checkpoint_$(date +%Y%m%d_%H%M%S).log}"

RUN_TRAIN=0
RUN_SOURCE=0
RUN_TRANSFER=0
RUN_DEFENSES=0

case "$PHASE" in
  all)
    RUN_TRAIN=1
    RUN_SOURCE=1
    RUN_TRANSFER=1
    RUN_DEFENSES=1
    ;;
  train) RUN_TRAIN=1 ;;
  source|test) RUN_SOURCE=1 ;;
  transfer|target) RUN_TRANSFER=1 ;;
  defense|defenses) RUN_DEFENSES=1 ;;
  *)
    echo "Unsupported PHASE=${PHASE}. Use train|source|transfer|defense|all." >&2
    exit 2
    ;;
esac

model_to_arch() {
  case "$1" in
    micro_cnn) echo "MicroCNN_cifar10" ;;
    small_cnn) echo "SmallCNN_cifar10" ;;
    resnet50) echo "ResNet50_cifar10" ;;
    *)
      echo "Unsupported model: $1" >&2
      return 1
      ;;
  esac
}

rates_for_model() {
  case "$1" in
    micro_cnn|small_cnn)
      if [ "$1" = "micro_cnn" ]; then
        echo "${MICROCNN_RATES:-0.010 0.020}"
      else
        echo "${SMALLCNN_RATES:-0.010 0.020}"
      fi
      ;;
    resnet50)
      echo "${RESNET50_RATES:-0.010}"
      ;;
  esac
}

alphas() {
  echo "${ALPHAS:-0.100 0.200 0.300}"
}

rate_fmt() {
  printf "%.3f" "$1"
}

poison_dir() {
  local model="$1"
  local rate="$2"
  local alpha="$3"
  local arch
  arch="$(model_to_arch "$model")"
  echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/belt_$(rate_fmt "$rate")_alpha=$(rate_fmt "$alpha")_cover=0.500_mask=0.200_poison_seed=2333_arch=${arch}"
}

model_file() {
  local model="$1"
  local arch
  arch="$(model_to_arch "$model")"
  echo "${arch}_belt_aug_model_seed=2333.pt"
}

selected_configs() {
  local models="${MODEL_LIST:-micro_cnn small_cnn resnet50}"
  models="${models//,/ }"
  for model in $models; do
    for rate in $(rates_for_model "$model"); do
      for alpha in $(alphas); do
        echo "${model}|${rate}|${alpha}|$(poison_dir "$model" "$rate" "$alpha")"
      done
    done
  done
}

gpu_for_index() {
  local index="$1"
  local ids=($GPU_IDS)
  local n="${#ids[@]}"
  if [ "$n" -eq 0 ]; then
    echo "0"
  else
    echo "${ids[$((index % n))]}"
  fi
}

max_jobs() {
  local ids=($GPU_IDS)
  if [ "${PARALLEL}" = "1" ]; then
    echo "${MAX_JOBS:-${#ids[@]}}"
  else
    echo "1"
  fi
}

backup_existing_outputs() {
  local dir="$1"
  local model="$2"
  local model_file
  local backup_dir
  model_file="$(model_file "$model")"
  backup_dir="${dir}/${BACKUP_TAG}"

  if [ "$BACKUP_OLD_RESULTS" != "1" ]; then
    return 0
  fi
  if [ ! -d "$dir" ]; then
    return 0
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] backup existing outputs under ${backup_dir}"
    return 0
  fi

  mkdir -p "$backup_dir"
  for path in \
    "${dir}/${model_file}" \
    "${dir}/${model_file%.pt}_best.pt" \
    "${dir}/train_results_seed=2333.json" \
    "${dir}/test_results_seed=2333.json" \
    "${dir}/test_stl10_results.txt" \
    "${dir}/sentinet_defense_results.json" \
    "${dir}/strip_defense_results.json" \
    "${dir}/scaleup_defense_results.json" \
    "${dir}/ibd_psc_defense_results.json"; do
    if [ -e "$path" ]; then
      cp -a "$path" "$backup_dir/"
    fi
  done
}

prepare_backups() {
  if [ "$BACKUP_OLD_RESULTS" != "1" ]; then
    return 0
  fi
  echo
  echo "----- backup existing outputs -----"
  while IFS="|" read -r model rate alpha dir; do
    echo "Backup: model=${model}, poison_rate=${rate}, alpha=${alpha}, dir=${dir}"
    backup_existing_outputs "$dir" "$model"
  done < <(selected_configs)
}

run_command() {
  local cmd="$1"
  local desc="$2"
  echo
  echo ">>> ${desc}"
  echo "$cmd"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  if ! eval "$cmd"; then
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed"
      echo "description: ${desc}"
      echo "command: ${cmd}"
      echo "---"
    } >> "$ERROR_LOG"
    if [ "$STOP_ON_FAIL" = "1" ]; then
      exit 1
    fi
    return 1
  fi
}

run_phase_for_config() {
  local phase="$1"
  local model="$2"
  local rate="$3"
  local alpha="$4"
  local dir="$5"
  local gpu="$6"
  local base_args="-dataset=${DATASET} -model=${model} -devices=${gpu} -poison_type=belt -poison_rate=${rate} -cover_rate 0.5 -mask_rate 0.2 -alpha ${alpha} -no_normalize"

  if [ ! -d "$dir" ]; then
    echo "Missing poisoned-set directory: ${dir}" >&2
    echo "Run create_poisoned_set.py first if this config is intentional." >&2
    return 3
  fi

  case "$phase" in
    train)
      run_command "${PYTHON_BIN} train_on_poisoned_set.py ${base_args}" \
        "Train BELT final checkpoint: model=${model}, poison_rate=${rate}, alpha=${alpha}, gpu=${gpu}"
      ;;
    source)
      run_command "${PYTHON_BIN} test_model.py ${base_args}" \
        "Source test BELT final checkpoint: model=${model}, poison_rate=${rate}, alpha=${alpha}, gpu=${gpu}"
      ;;
    transfer)
      run_command "${PYTHON_BIN} test_stl10.py ${base_args}" \
        "STL10 transfer test BELT final checkpoint: model=${model}, poison_rate=${rate}, alpha=${alpha}, gpu=${gpu}"
      ;;
    defense)
      local defense
      for defense in $DEFENSE_LIST; do
        run_command "${PYTHON_BIN} other_defense.py ${base_args} -defense=${defense}" \
          "Defense ${defense}: model=${model}, poison_rate=${rate}, alpha=${alpha}, gpu=${gpu}"
      done
      ;;
  esac
}

run_phase_group() {
  local phase="$1"
  local idx=0
  local active=0
  local max
  local pids=()
  max="$(max_jobs)"

  while IFS="|" read -r model rate alpha dir; do
    gpu="$(gpu_for_index "$idx")"
    idx=$((idx + 1))
    if [ "$max" -gt 1 ] && [ "$DRY_RUN" != "1" ]; then
      (
        run_phase_for_config "$phase" "$model" "$rate" "$alpha" "$dir" "$gpu"
      ) &
      pids+=("$!")
      active=$((active + 1))
      if [ "$active" -ge "$max" ]; then
        wait_phase_batch "${pids[@]}"
        pids=()
        active=0
      fi
    else
      run_phase_for_config "$phase" "$model" "$rate" "$alpha" "$dir" "$gpu"
    fi
  done < <(selected_configs)

  if [ "$active" -gt 0 ]; then
    wait_phase_batch "${pids[@]}"
  fi
}

wait_phase_batch() {
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

echo "============================================================"
echo "Rerun CIFAR-10 set4 BELT with final-epoch checkpoints"
echo "============================================================"
echo "repo              : ${REPO_ROOT}"
echo "python            : ${PYTHON_BIN}"
echo "result root       : ${POISONED_TRAIN_SET_ROOT}"
echo "phase             : ${PHASE}"
echo "models            : ${MODEL_LIST:-micro_cnn small_cnn resnet50}"
echo "gpu ids           : ${GPU_IDS}"
echo "parallel          : ${PARALLEL}"
echo "backup old result : ${BACKUP_OLD_RESULTS}"
echo "backup tag        : ${BACKUP_TAG}"
echo "dry run           : ${DRY_RUN}"
echo "error log         : ${ERROR_LOG}"
echo "============================================================"

config_count=0
echo
echo "Selected configs:"
while IFS="|" read -r model rate alpha dir; do
  gpu="$(gpu_for_index "$config_count")"
  config_count=$((config_count + 1))
  echo "CONFIG ${config_count}: model=${model} rate=${rate} alpha=${alpha} gpu=${gpu} -> ${dir}"
done < <(selected_configs)
echo "Total configs: ${config_count}"

prepare_backups

if [ "$RUN_TRAIN" = "1" ]; then
  echo
  echo "----- train -----"
  run_phase_group train
fi

if [ "$RUN_SOURCE" = "1" ]; then
  echo
  echo "----- source -----"
  run_phase_group source
fi

if [ "$RUN_TRANSFER" = "1" ]; then
  echo
  echo "----- transfer -----"
  run_phase_group transfer
fi

if [ "$RUN_DEFENSES" = "1" ]; then
  echo
  echo "----- defense -----"
  run_phase_group defense
fi

echo
echo "Finished set4 CIFAR-10 BELT rerun script."
