#!/usr/bin/env bash

# Full poisoned_train_set4 BELT rerun with final-epoch checkpoints.
#
# Default grid follows the corrected BELT poison-rate rule:
#   BELT poison_rate = 2x the normal attack poison rate.
#
# Defaults:
#   CIFAR-10 MicroCNN : 0.010, 0.020
#   CIFAR-10 SmallCNN : 0.010, 0.020
#   CIFAR-10 ResNet50 : 0.010
#   Tiny ResNet34     : 0.002, 0.010
#   Tiny ResNet50     : 0.010
#   alpha             : 0.100, 0.200, 0.300
#
# Phases:
#   create, train, source, transfer, qwen, defense, all
#
# Examples:
#   DRY_RUN=1 bash run/rerun_set4_belt_final_checkpoint_full.sh
#   PHASE=all PARALLEL=1 GPU_IDS="0 1 2 3" bash run/rerun_set4_belt_final_checkpoint_full.sh
#   RUN_TINY_RESNET50=0 PHASE=all PARALLEL=1 GPU_IDS="0 1 2 3" bash run/rerun_set4_belt_final_checkpoint_full.sh
#   PHASE=defense DEFENSE_LIST="SentiNet STRIP ScaleUp IBD_PSC" bash run/rerun_set4_belt_final_checkpoint_full.sh

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

export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL="${PARALLEL:-0}"
GPU_IDS="${GPU_IDS:-${DEVICES:-0}}"
MAX_JOBS="${MAX_JOBS:-}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_CIFAR_MICROCNN="${RUN_CIFAR_MICROCNN:-1}"
RUN_CIFAR_SMALLCNN="${RUN_CIFAR_SMALLCNN:-1}"
RUN_CIFAR_RESNET50="${RUN_CIFAR_RESNET50:-1}"
RUN_TINY_RESNET34="${RUN_TINY_RESNET34:-1}"
RUN_TINY_RESNET50="${RUN_TINY_RESNET50:-1}"
DEFENSE_LIST="${DEFENSE_LIST:-SentiNet STRIP ScaleUp IBD_PSC}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/imagenetv2-matched-frequency-tiny-organized}"
QWEN_TARGET_DOMAIN_DIR="${QWEN_TARGET_DOMAIN_DIR:-${REPO_ROOT}/data/tiny-target-domain-qwen-full-organized}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_set4_belt_final_checkpoint_full_$(date +%Y%m%d_%H%M%S).log}"

CIFAR_MICROCNN_RATES="${CIFAR_MICROCNN_RATES:-0.010 0.020}"
CIFAR_SMALLCNN_RATES="${CIFAR_SMALLCNN_RATES:-0.010 0.020}"
CIFAR_RESNET50_RATES="${CIFAR_RESNET50_RATES:-0.010}"
TINY_RESNET34_RATES="${TINY_RESNET34_RATES:-0.002 0.010}"
TINY_RESNET50_RATES="${TINY_RESNET50_RATES:-0.010}"
ALPHAS="${ALPHAS:-0.100 0.200 0.300}"

RUN_CREATE=0
RUN_TRAIN=0
RUN_SOURCE=0
RUN_TRANSFER=0
RUN_QWEN_PHASE=0
RUN_DEFENSES=0

case "$PHASE" in
  all)
    RUN_CREATE=1
    RUN_TRAIN=1
    RUN_SOURCE=1
    RUN_TRANSFER=1
    RUN_QWEN_PHASE="$RUN_QWEN"
    RUN_DEFENSES=1
    ;;
  create) RUN_CREATE=1 ;;
  train) RUN_TRAIN=1 ;;
  source|test) RUN_SOURCE=1 ;;
  transfer|target) RUN_TRANSFER=1 ;;
  qwen) RUN_QWEN_PHASE=1 ;;
  defense|defenses) RUN_DEFENSES=1 ;;
  *)
    echo "Unsupported PHASE=${PHASE}. Use create|train|source|transfer|qwen|defense|all." >&2
    exit 2
    ;;
esac

arch_name() {
  case "$1:$2" in
    cifar10:micro_cnn) echo "MicroCNN_cifar10" ;;
    cifar10:small_cnn) echo "SmallCNN_cifar10" ;;
    cifar10:resnet50) echo "ResNet50_cifar10" ;;
    tiny_imagenet:resnet34) echo "ResNet34_tiny_imagenet" ;;
    tiny_imagenet:resnet50) echo "ResNet50_tiny_imagenet" ;;
    *)
      echo "Unsupported dataset/model pair: $1/$2" >&2
      return 1
      ;;
  esac
}

rate_fmt() {
  printf "%.3f" "$1"
}

poison_dir() {
  local dataset="$1"
  local model="$2"
  local rate="$3"
  local alpha="$4"
  local arch
  arch="$(arch_name "$dataset" "$model")"
  echo "${POISONED_TRAIN_SET_ROOT}/${dataset}/belt_$(rate_fmt "$rate")_alpha=$(rate_fmt "$alpha")_cover=0.500_mask=0.200_poison_seed=2333_arch=${arch}"
}

model_file() {
  local dataset="$1"
  local model="$2"
  local arch
  arch="$(arch_name "$dataset" "$model")"
  echo "${arch}_belt_aug_model_seed=2333.pt"
}

selected_configs() {
  local rate
  local alpha
  if [ "$RUN_CIFAR_MICROCNN" = "1" ]; then
    for rate in $CIFAR_MICROCNN_RATES; do
      for alpha in $ALPHAS; do
        echo "cifar10|micro_cnn|${rate}|${alpha}|$(poison_dir cifar10 micro_cnn "$rate" "$alpha")"
      done
    done
  fi
  if [ "$RUN_CIFAR_SMALLCNN" = "1" ]; then
    for rate in $CIFAR_SMALLCNN_RATES; do
      for alpha in $ALPHAS; do
        echo "cifar10|small_cnn|${rate}|${alpha}|$(poison_dir cifar10 small_cnn "$rate" "$alpha")"
      done
    done
  fi
  if [ "$RUN_CIFAR_RESNET50" = "1" ]; then
    for rate in $CIFAR_RESNET50_RATES; do
      for alpha in $ALPHAS; do
        echo "cifar10|resnet50|${rate}|${alpha}|$(poison_dir cifar10 resnet50 "$rate" "$alpha")"
      done
    done
  fi
  if [ "$RUN_TINY_RESNET34" = "1" ]; then
    for rate in $TINY_RESNET34_RATES; do
      for alpha in $ALPHAS; do
        echo "tiny_imagenet|resnet34|${rate}|${alpha}|$(poison_dir tiny_imagenet resnet34 "$rate" "$alpha")"
      done
    done
  fi
  if [ "$RUN_TINY_RESNET50" = "1" ]; then
    for rate in $TINY_RESNET50_RATES; do
      for alpha in $ALPHAS; do
        echo "tiny_imagenet|resnet50|${rate}|${alpha}|$(poison_dir tiny_imagenet resnet50 "$rate" "$alpha")"
      done
    done
  fi
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
  if [ "$PARALLEL" = "1" ]; then
    if [ -n "$MAX_JOBS" ]; then
      echo "$MAX_JOBS"
    else
      echo "${#ids[@]}"
    fi
  else
    echo "1"
  fi
}

base_args() {
  local dataset="$1"
  local model="$2"
  local gpu="$3"
  echo "-dataset=${dataset} -model=${model} -devices=${gpu}"
}

belt_args_create() {
  local rate="$1"
  local alpha="$2"
  echo "-poison_type=belt -poison_rate=${rate} -cover_rate 0.5 -mask_rate 0.2 -alpha ${alpha}"
}

belt_args_raw() {
  local rate="$1"
  local alpha="$2"
  echo "$(belt_args_create "$rate" "$alpha") -no_normalize"
}

phase_done() {
  local phase="$1"
  local dataset="$2"
  local model="$3"
  local dir="$4"
  local defense="${5:-}"
  local mf
  mf="$(model_file "$dataset" "$model")"
  case "$phase" in
    create)
      [ -f "${dir}/labels" ] && [ -f "${dir}/poison_indices" ] && [ -f "${dir}/pmarks" ]
      ;;
    train)
      [ -f "${dir}/${mf}" ] && [ -f "${dir}/train_results_seed=2333.json" ]
      ;;
    source)
      compgen -G "${dir}/test_results_seed=2333*.json" >/dev/null
      ;;
    transfer)
      if [ "$dataset" = "cifar10" ]; then
        compgen -G "${dir}/test_stl10_results*.txt" >/dev/null
      else
        [ -f "${dir}/test_tiny_target_domain_results.txt" ]
      fi
      ;;
    qwen)
      [ "$dataset" = "tiny_imagenet" ] && [ -f "${dir}/test_tiny_target_domain_qwen_results.txt" ]
      ;;
    defense)
      case "$defense" in
        SentiNet) [ -f "${dir}/sentinet_defense_results.json" ] ;;
        STRIP) [ -f "${dir}/strip_defense_results.json" ] ;;
        ScaleUp) [ -f "${dir}/scaleup_defense_results.json" ] ;;
        IBD_PSC) [ -f "${dir}/ibd_psc_defense_results.json" ] ;;
        *) return 1 ;;
      esac
      ;;
    *)
      return 1
      ;;
  esac
}

run_command() {
  local cmd="$1"
  local desc="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/set4_belt_$$_${RANDOM}.out")"
  echo
  echo ">>> ${desc}"
  echo "$cmd"
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
      echo "description: ${desc}"
      echo "command: ${cmd}"
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
}

run_config_phase() {
  local phase="$1"
  local dataset="$2"
  local model="$3"
  local rate="$4"
  local alpha="$5"
  local dir="$6"
  local gpu="$7"
  local label="${dataset} ${model} BELT rate=${rate} alpha=${alpha}"

  if [ "$SKIP_EXISTING" = "1" ] && phase_done "$phase" "$dataset" "$model" "$dir"; then
    echo "Skip ${phase}: already complete: ${dir}"
    return 0
  fi

  case "$phase" in
    create)
      run_command "${PYTHON_BIN} create_poisoned_set.py $(base_args "$dataset" "$model" "$gpu") $(belt_args_create "$rate" "$alpha")" \
        "Create: ${label}"
      ;;
    train)
      run_command "${PYTHON_BIN} train_on_poisoned_set.py $(base_args "$dataset" "$model" "$gpu") $(belt_args_raw "$rate" "$alpha")" \
        "Train final checkpoint: ${label}"
      ;;
    source)
      run_command "${PYTHON_BIN} test_model.py $(base_args "$dataset" "$model" "$gpu") $(belt_args_raw "$rate" "$alpha")" \
        "Source test: ${label}"
      ;;
    transfer)
      if [ "$dataset" = "cifar10" ]; then
        run_command "${PYTHON_BIN} test_stl10.py $(base_args "$dataset" "$model" "$gpu") $(belt_args_raw "$rate" "$alpha")" \
          "STL10 transfer: ${label}"
      else
        run_command "${PYTHON_BIN} test_tiny_target_domain.py $(base_args "$dataset" "$model" "$gpu") -source_dataset=${dataset} -target_domain_dir=${TARGET_DOMAIN_DIR} $(belt_args_raw "$rate" "$alpha")" \
          "ImageNetV2-tiny transfer: ${label}"
      fi
      ;;
    qwen)
      if [ "$dataset" = "tiny_imagenet" ]; then
        run_command "${PYTHON_BIN} test_tiny_target_domain_qwen.py $(base_args "$dataset" "$model" "$gpu") -source_dataset=${dataset} -target_domain_dir=${QWEN_TARGET_DOMAIN_DIR} $(belt_args_raw "$rate" "$alpha")" \
          "Qwen transfer: ${label}"
      fi
      ;;
  esac
}

run_defenses_for_config() {
  local dataset="$1"
  local model="$2"
  local rate="$3"
  local alpha="$4"
  local dir="$5"
  local gpu="$6"
  local defense
  local label="${dataset} ${model} BELT rate=${rate} alpha=${alpha}"

  for defense in $DEFENSE_LIST; do
    if [ "$SKIP_EXISTING" = "1" ] && phase_done defense "$dataset" "$model" "$dir" "$defense"; then
      echo "Skip defense ${defense}: already complete: ${dir}"
      continue
    fi
    run_command "${PYTHON_BIN} other_defense.py $(base_args "$dataset" "$model" "$gpu") -defense=${defense} $(belt_args_raw "$rate" "$alpha")" \
      "Defense ${defense}: ${label}"
  done
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

run_phase_group() {
  local phase="$1"
  local idx=0
  local active=0
  local max
  local pids=()
  max="$(max_jobs)"

  while IFS="|" read -r dataset model rate alpha dir; do
    gpu="$(gpu_for_index "$idx")"
    idx=$((idx + 1))
    if [ "$max" -gt 1 ] && [ "$DRY_RUN" != "1" ]; then
      if [ "$phase" = "defense" ]; then
        (run_defenses_for_config "$dataset" "$model" "$rate" "$alpha" "$dir" "$gpu") &
      else
        (run_config_phase "$phase" "$dataset" "$model" "$rate" "$alpha" "$dir" "$gpu") &
      fi
      pids+=("$!")
      active=$((active + 1))
      if [ "$active" -ge "$max" ]; then
        wait_phase_batch "${pids[@]}"
        pids=()
        active=0
      fi
    else
      if [ "$phase" = "defense" ]; then
        run_defenses_for_config "$dataset" "$model" "$rate" "$alpha" "$dir" "$gpu"
      else
        run_config_phase "$phase" "$dataset" "$model" "$rate" "$alpha" "$dir" "$gpu"
      fi
    fi
  done < <(selected_configs)

  if [ "$active" -gt 0 ]; then
    wait_phase_batch "${pids[@]}"
  fi
}

echo "============================================================"
echo "Full poisoned_train_set4 BELT rerun with final checkpoints"
echo "============================================================"
echo "repo              : ${REPO_ROOT}"
echo "python            : ${PYTHON_BIN}"
echo "result root       : ${POISONED_TRAIN_SET_ROOT}"
echo "phase             : ${PHASE}"
echo "gpu ids           : ${GPU_IDS}"
echo "parallel          : ${PARALLEL}"
echo "skip existing     : ${SKIP_EXISTING}"
echo "run qwen          : ${RUN_QWEN}"
echo "defenses          : ${DEFENSE_LIST}"
echo "target domain     : ${TARGET_DOMAIN_DIR}"
echo "qwen domain       : ${QWEN_TARGET_DOMAIN_DIR}"
echo "dry run           : ${DRY_RUN}"
echo "error log         : ${ERROR_LOG}"
echo "============================================================"

config_count=0
echo
echo "Selected configs:"
while IFS="|" read -r dataset model rate alpha dir; do
  gpu="$(gpu_for_index "$config_count")"
  config_count=$((config_count + 1))
  echo "CONFIG ${config_count}: dataset=${dataset} model=${model} rate=${rate} alpha=${alpha} gpu=${gpu} -> ${dir}"
done < <(selected_configs)
echo "Total configs: ${config_count}"

if [ "$RUN_CREATE" = "1" ]; then
  echo
  echo "----- create -----"
  run_phase_group create
fi

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

if [ "$RUN_QWEN_PHASE" = "1" ]; then
  echo
  echo "----- qwen -----"
  run_phase_group qwen
fi

if [ "$RUN_DEFENSES" = "1" ]; then
  echo
  echo "----- defense -----"
  run_phase_group defense
fi

echo
echo "Finished full poisoned_train_set4 BELT rerun script."
