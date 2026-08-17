#!/usr/bin/env bash

# Shards 0--7 of the nine-way Adaptive-Patch evaluation split.
# One configuration folder is the indivisible scheduling unit.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${REPO_ROOT:-$SCRIPT_DIR/..}" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR}"
ROOTS=("$RESULT_ROOT/poisoned_train_set" "$RESULT_ROOT/poisoned_train_set3")
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
SEED="${SEED:-2333}"
DRY_RUN="${DRY_RUN:-0}"
TINY_TARGET_DIR="${TINY_TARGET_DIR:-$REPO_ROOT/data/imagenetv2-matched-frequency-tiny-organized}"

read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -ne 8 ]]; then
  echo "GPU_IDS must contain exactly eight GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
for required in test_model.py test_stl10.py test_tiny_target_domain.py other_defense.py; do
  [[ -f "$REPO_ROOT/$required" ]] || { echo "Missing $REPO_ROOT/$required" >&2; exit 2; }
done
if [[ "$DRY_RUN" != 1 && ! -d "$TINY_TARGET_DIR" ]]; then
  echo "Missing ImageNetV2 target directory: $TINY_TARGET_DIR" >&2
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/adaptive_patch_matched_retest/eight_gpu/$RUN_ID}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ap-eight-gpu.XXXXXX")"
trap 'rm -rf -- "$TMP_ROOT"' EXIT
mkdir -p "$LOG_ROOT"

parse_config() {
  local name="$1"
  [[ "$name" =~ ^adaptive_patch_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)_poison_seed=2333_arch=(ResNet18|mobilenetv2|vgg19_bn)_(cifar10|tiny_imagenet)$ ]] || return 1
  RATE="${BASH_REMATCH[1]}"
  TRAIN_ALPHA="${BASH_REMATCH[2]}"
  COVER="${BASH_REMATCH[3]}"
  ARCH="${BASH_REMATCH[4]}"
  DATASET="${BASH_REMATCH[5]}"
  case "$ARCH" in
    ResNet18) MODEL=resnet18 ;;
    mobilenetv2) MODEL=mobilenetv2 ;;
    vgg19_bn) MODEL=vgg19_bn ;;
  esac
  TEST_ALPHA="$(awk -v alpha="$TRAIN_ALPHA" 'BEGIN { printf "%.1f", alpha + 0.5 }')"
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  (
    printf 'Command:'
    printf ' %q' "$@"
    printf '\nStarted: %s\n' "$(date --iso-8601=seconds)"
    "$@"
    status=$?
    printf 'Finished: %s\nExit status: %d\n' "$(date --iso-8601=seconds)" "$status"
    exit "$status"
  ) >"$log_file" 2>&1
}

run_plain_stage() {
  local gpu="$1" root="$2" log_file="$3"
  shift 3
  if [[ "$DRY_RUN" == 1 ]]; then
    mkdir -p "$(dirname "$log_file")"
    printf 'DRY RUN: inherited CUDA_VISIBLE_DEVICES=%q; env POISONED_TRAIN_SET_ROOT=%q ' "${CUDA_VISIBLE_DEVICES:-unset}" "$root" >"$log_file"
    printf '%q ' "$@" >>"$log_file"
    printf '\n' >>"$log_file"
    return 0
  fi
  run_logged "$log_file" env POISONED_TRAIN_SET_ROOT="$root" "$@"
}

run_defense_stage() {
  local gpu="$1" root="$2" dir="$3" defense="$4" test_alpha="$5" log_file="$6"
  shift 6
  local stem base_file suffix_file backup had_base=0 status
  case "$defense" in
    SentiNet) stem=sentinet_defense_results ;;
    ScaleUp) stem=scaleup_defense_results ;;
    STRIP) stem=strip_defense_results ;;
    IBD_PSC) stem=ibd_psc_defense_results ;;
    *) return 2 ;;
  esac
  base_file="$dir/$stem.json"
  suffix_file="$dir/${stem}_test_alpha=${test_alpha}.json"
  backup="$(mktemp "$TMP_ROOT/${stem}.XXXXXX")"
  if [[ -f "$base_file" ]]; then
    cp -p -- "$base_file" "$backup"
    had_base=1
  fi
  run_plain_stage "$gpu" "$root" "$log_file" "$@"
  status=$?
  if [[ "$had_base" == 1 ]]; then
    cp -p -- "$backup" "$base_file"
  else
    rm -f -- "$base_file"
  fi
  rm -f -- "$backup"
  [[ "$DRY_RUN" == 1 ]] && return 0
  [[ "$status" -eq 0 && -f "$suffix_file" ]]
}

run_config() {
  local gpu="$1" dir="$2" failure_file="$3"
  local root config_tag root_tag log_dir status failed=0 output defense
  root="$(dirname "$(dirname "$dir")")"
  config_tag="$(basename "$dir")"
  root_tag="$(basename "$root")"
  parse_config "$config_tag" || { printf '%s\tparse\n' "$dir" >>"$failure_file"; return 1; }
  [[ -f "$dir/${ARCH}_${DATASET}.pt" ]] || { printf '%s\tcheckpoint\n' "$dir" >>"$failure_file"; return 1; }
  log_dir="$LOG_ROOT/gpu${gpu}/${root_tag}/${DATASET}/${config_tag}"
  common=(
    -dataset="$DATASET" -poison_type=adaptive_patch -poison_rate="$RATE"
    -cover_rate="$COVER" -alpha="$TRAIN_ALPHA" -test_alpha="$TEST_ALPHA"
    -model="$MODEL" -devices="$gpu" -seed="$SEED"
  )
  echo "[GPU $gpu] $DATASET $root_tag/$config_tag (test alpha=$TEST_ALPHA)"

  run_plain_stage "$gpu" "$root" "$log_dir/local.log" \
    "$PYTHON_BIN" "$REPO_ROOT/test_model.py" "${common[@]}"
  status=$?
  output="$dir/test_results_seed=${SEED}_test_alpha=${TEST_ALPHA}.json"
  if [[ "$status" -ne 0 || ( "$DRY_RUN" != 1 && ! -f "$output" ) ]]; then
    printf '%s\tlocal\n' "$dir" >>"$failure_file"; failed=1
  fi

  if [[ "$DATASET" == cifar10 ]]; then
    run_plain_stage "$gpu" "$root" "$log_dir/cross_stl10.log" \
      "$PYTHON_BIN" "$REPO_ROOT/test_stl10.py" "${common[@]}"
    status=$?
    output="$dir/test_stl10_results_test_alpha=${TEST_ALPHA}.txt"
  else
    run_plain_stage "$gpu" "$root" "$log_dir/cross_imagenetv2.log" \
      "$PYTHON_BIN" "$REPO_ROOT/test_tiny_target_domain.py" "${common[@]}" -target_domain_dir="$TINY_TARGET_DIR"
    status=$?
    output="$dir/test_tiny_target_domain_results_test_alpha=${TEST_ALPHA}.txt"
  fi
  if [[ "$status" -ne 0 || ( "$DRY_RUN" != 1 && ! -f "$output" ) ]]; then
    printf '%s\tcross_domain\n' "$dir" >>"$failure_file"; failed=1
  fi

  for defense in SentiNet ScaleUp STRIP IBD_PSC; do
    if ! run_defense_stage "$gpu" "$root" "$dir" "$defense" "$TEST_ALPHA" "$log_dir/defense_${defense}.log" \
      "$PYTHON_BIN" "$REPO_ROOT/other_defense.py" "${common[@]}" -defense="$defense"; then
      printf '%s\tdefense_%s\n' "$dir" "$defense" >>"$failure_file"; failed=1
    fi
  done
  return "$failed"
}

run_gpu_shard() {
  local worker="$1" gpu="${GPUS[$1]}" failed=0
  local failure_file="$LOG_ROOT/failures_gpu${gpu}.tsv"
  local -a configs
  local dataset arch dataset_index selected
  : >"$failure_file"
  configs=()
  for dataset in cifar10 tiny_imagenet; do
    dataset_index=0
    selected=0
    for arch in ResNet18 mobilenetv2 vgg19_bn; do
      while IFS= read -r config_dir; do
        if (( dataset_index % 8 == worker )); then
          configs+=("$config_dir")
          selected=$((selected + 1))
        fi
        dataset_index=$((dataset_index + 1))
      done < <(
        for root in "${ROOTS[@]}"; do
          find "$root/$dataset" -maxdepth 1 -type d \
            -name "adaptive_patch_*_arch=${arch}_${dataset}" 2>/dev/null
        done | sort
      )
    done
    if [[ "$dataset_index" -ne 56 || "$selected" -ne 7 ]]; then
      echo "GPU shard $worker: expected 56 total and 7 selected $dataset folders; got $dataset_index and $selected." >&2
      return 1
    fi
  done
  if [[ "${#configs[@]}" -ne 14 ]]; then
    echo "Shard $worker must contain 14 configuration folders; found ${#configs[@]}." >&2
    return 1
  fi
  cifar_count="$(printf '%s\n' "${configs[@]}" | grep -c '/cifar10/')"
  tiny_count="$(printf '%s\n' "${configs[@]}" | grep -c '/tiny_imagenet/')"
  if [[ "$cifar_count" -ne 7 || "$tiny_count" -ne 7 ]]; then
    echo "Shard $worker is unbalanced: CIFAR=$cifar_count Tiny=$tiny_count" >&2
    return 1
  fi
  for dir in "${configs[@]}"; do
    run_config "$gpu" "$dir" "$failure_file" || failed=1
  done
  return "$failed"
}

cd "$REPO_ROOT"
echo "Eight-GPU portion: 8 shards x 14 folders x 6 evaluations = 672 jobs"
echo "Result roots: ${ROOTS[*]}"
echo "Logs: $LOG_ROOT"
pids=()
for worker in {0..7}; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPUS[$worker]}" run_gpu_shard "$worker" &
  pids+=("$!")
done
overall=0
for pid in "${pids[@]}"; do wait "$pid" || overall=1; done
find "$LOG_ROOT" -name 'failures_gpu*.tsv' -type f -exec cat {} + >"$LOG_ROOT/failures.tsv"
if [[ "$overall" -ne 0 || -s "$LOG_ROOT/failures.tsv" ]]; then
  echo "Eight-GPU run has failures; see $LOG_ROOT/failures.tsv" >&2
  exit 1
fi
echo "All 112 configuration folders completed successfully."
