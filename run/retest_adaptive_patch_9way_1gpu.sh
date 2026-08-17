#!/usr/bin/env bash

# Shard 8 of the nine-way Adaptive-Patch evaluation split.
# It selects 14 folders from the two local result roots: seven CIFAR-10 and
# seven Tiny-ImageNet, balanced across the three architectures.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${REPO_ROOT:-$SCRIPT_DIR/..}" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"
SEED="${SEED:-2333}"
DRY_RUN="${DRY_RUN:-0}"
TINY_TARGET_DIR="${TINY_TARGET_DIR:-$REPO_ROOT/data/imagenetv2-matched-frequency-tiny-organized}"
if [[ -n "${RESULT_ROOTS:-}" ]]; then
  read -r -a ROOTS <<< "$RESULT_ROOTS"
else
  ROOTS=("$REPO_ROOT/poisoned_train_set" "$REPO_ROOT/poisoned_train_set3")
fi

for required in test_model.py test_stl10.py test_tiny_target_domain.py other_defense.py; do
  [[ -f "$REPO_ROOT/$required" ]] || { echo "Missing $REPO_ROOT/$required" >&2; exit 2; }
done
if [[ "$DRY_RUN" != 1 && ! -d "$TINY_TARGET_DIR" ]]; then
  echo "Missing ImageNetV2 target directory: $TINY_TARGET_DIR" >&2
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/adaptive_patch_matched_retest/single_gpu/$RUN_ID}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ap-single-gpu.XXXXXX")"
trap 'rm -rf -- "$TMP_ROOT"' EXIT
mkdir -p "$LOG_ROOT"

declare -a CONFIGS
for dataset in cifar10 tiny_imagenet; do
  dataset_selected=0
  for arch in ResNet18 mobilenetv2 vgg19_bn; do
    while IFS= read -r config_dir; do
      CONFIGS+=("$config_dir")
      dataset_selected=$((dataset_selected + 1))
    done < <(
      for root in "${ROOTS[@]}"; do
        find "$root/$dataset" -maxdepth 1 -type d -name "adaptive_patch_*_arch=${arch}_${dataset}" 2>/dev/null
      done | sort
    )
  done
  if [[ "$dataset_selected" -ne 7 ]]; then
    echo "Expected 7 remaining $dataset folders; got $dataset_selected." >&2
    exit 2
  fi
done
if [[ "${#CONFIGS[@]}" -ne 14 ]]; then
  echo "Single-GPU shard must contain 14 folders; found ${#CONFIGS[@]}." >&2
  exit 2
fi

parse_config() {
  local name="$1"
  [[ "$name" =~ ^adaptive_patch_([0-9.]+)_alpha=([0-9.]+)_cover=([0-9.]+)_poison_seed=2333_arch=(ResNet18|mobilenetv2|vgg19_bn)_(cifar10|tiny_imagenet)$ ]] || return 1
  RATE="${BASH_REMATCH[1]}"; TRAIN_ALPHA="${BASH_REMATCH[2]}"; COVER="${BASH_REMATCH[3]}"
  ARCH="${BASH_REMATCH[4]}"; DATASET="${BASH_REMATCH[5]}"
  case "$ARCH" in ResNet18) MODEL=resnet18 ;; mobilenetv2) MODEL=mobilenetv2 ;; vgg19_bn) MODEL=vgg19_bn ;; esac
  TEST_ALPHA="$(awk -v alpha="$TRAIN_ALPHA" 'BEGIN { printf "%.1f", alpha + 0.5 }')"
}

run_logged() {
  local log_file="$1"; shift
  mkdir -p "$(dirname "$log_file")"
  (
    printf 'Command:'; printf ' %q' "$@"; printf '\nStarted: %s\n' "$(date --iso-8601=seconds)"
    "$@"; status=$?
    printf 'Finished: %s\nExit status: %d\n' "$(date --iso-8601=seconds)" "$status"
    exit "$status"
  ) >"$log_file" 2>&1
}

run_plain_stage() {
  local root="$1" log_file="$2"; shift 2
  if [[ "$DRY_RUN" == 1 ]]; then
    mkdir -p "$(dirname "$log_file")"
    printf 'DRY RUN: env CUDA_VISIBLE_DEVICES=%q POISONED_TRAIN_SET_ROOT=%q ' "$SINGLE_GPU_ID" "$root" >"$log_file"
    printf '%q ' "$@" >>"$log_file"; printf '\n' >>"$log_file"; return 0
  fi
  run_logged "$log_file" env CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" POISONED_TRAIN_SET_ROOT="$root" "$@"
}

run_defense_stage() {
  local root="$1" dir="$2" defense="$3" test_alpha="$4" log_file="$5"; shift 5
  local stem base_file suffix_file backup had_base=0 status
  case "$defense" in
    SentiNet) stem=sentinet_defense_results ;; ScaleUp) stem=scaleup_defense_results ;;
    STRIP) stem=strip_defense_results ;; IBD_PSC) stem=ibd_psc_defense_results ;; *) return 2 ;;
  esac
  base_file="$dir/$stem.json"; suffix_file="$dir/${stem}_test_alpha=${test_alpha}.json"
  backup="$(mktemp "$TMP_ROOT/${stem}.XXXXXX")"
  if [[ -f "$base_file" ]]; then cp -p -- "$base_file" "$backup"; had_base=1; fi
  run_plain_stage "$root" "$log_file" "$@"; status=$?
  if [[ "$had_base" == 1 ]]; then cp -p -- "$backup" "$base_file"; else rm -f -- "$base_file"; fi
  rm -f -- "$backup"
  [[ "$DRY_RUN" == 1 ]] && return 0
  [[ "$status" -eq 0 && -f "$suffix_file" ]]
}

failure_file="$LOG_ROOT/failures.tsv"
: >"$failure_file"
cd "$REPO_ROOT"
echo "Single-GPU portion: 14 folders x 6 evaluations = 84 jobs"
echo "GPU: $SINGLE_GPU_ID"
for dir in "${CONFIGS[@]}"; do
  root="$(dirname "$(dirname "$dir")")"; config_tag="$(basename "$dir")"; root_tag="$(basename "$root")"
  parse_config "$config_tag" || { printf '%s\tparse\n' "$dir" >>"$failure_file"; continue; }
  if [[ ! -f "$dir/${ARCH}_${DATASET}.pt" ]]; then printf '%s\tcheckpoint\n' "$dir" >>"$failure_file"; continue; fi
  log_dir="$LOG_ROOT/${root_tag}/${DATASET}/${config_tag}"
  common=(
    -dataset="$DATASET" -poison_type=adaptive_patch -poison_rate="$RATE"
    -cover_rate="$COVER" -alpha="$TRAIN_ALPHA" -test_alpha="$TEST_ALPHA"
    -model="$MODEL" -devices=0 -seed="$SEED"
  )
  echo "[GPU $SINGLE_GPU_ID] $DATASET $root_tag/$config_tag (test alpha=$TEST_ALPHA)"

  run_plain_stage "$root" "$log_dir/local.log" "$PYTHON_BIN" "$REPO_ROOT/test_model.py" "${common[@]}"; status=$?
  output="$dir/test_results_seed=${SEED}_test_alpha=${TEST_ALPHA}.json"
  [[ "$status" -eq 0 && ( "$DRY_RUN" == 1 || -f "$output" ) ]] || printf '%s\tlocal\n' "$dir" >>"$failure_file"

  if [[ "$DATASET" == cifar10 ]]; then
    run_plain_stage "$root" "$log_dir/cross_stl10.log" "$PYTHON_BIN" "$REPO_ROOT/test_stl10.py" "${common[@]}"; status=$?
    output="$dir/test_stl10_results_test_alpha=${TEST_ALPHA}.txt"
  else
    run_plain_stage "$root" "$log_dir/cross_imagenetv2.log" "$PYTHON_BIN" "$REPO_ROOT/test_tiny_target_domain.py" "${common[@]}" -target_domain_dir="$TINY_TARGET_DIR"; status=$?
    output="$dir/test_tiny_target_domain_results_test_alpha=${TEST_ALPHA}.txt"
  fi
  [[ "$status" -eq 0 && ( "$DRY_RUN" == 1 || -f "$output" ) ]] || printf '%s\tcross_domain\n' "$dir" >>"$failure_file"

  for defense in SentiNet ScaleUp STRIP IBD_PSC; do
    run_defense_stage "$root" "$dir" "$defense" "$TEST_ALPHA" "$log_dir/defense_${defense}.log" \
      "$PYTHON_BIN" "$REPO_ROOT/other_defense.py" "${common[@]}" -defense="$defense" \
      || printf '%s\tdefense_%s\n' "$dir" "$defense" >>"$failure_file"
  done
done
if [[ -s "$failure_file" ]]; then echo "Single-GPU run has failures; see $failure_file" >&2; exit 1; fi
echo "All 14 single-GPU configuration folders completed successfully."
