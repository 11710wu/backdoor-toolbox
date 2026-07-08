#!/usr/bin/env bash
#
# Shared driver for Tiny-ImageNet adaptive-patch alpha=0.2 reruns.
# Set PART_TITLE, EXPECTED_DIRS, and TINY_JOBS before sourcing this file.
#
# Each TINY_JOBS entry: model|poison_rate|cover_rate

set +e

: "${PART_TITLE:?Set PART_TITLE}"
: "${EXPECTED_DIRS:?Set EXPECTED_DIRS}"
: "${TINY_JOBS:?Set TINY_JOBS}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DATASET="tiny_imagenet"
ALPHA="${ALPHA:-0.2}"
CLEAN_OLD="${CLEAN_OLD:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"
CORRUPTION_TYPES="${CORRUPTION_TYPES:-frost}"
SEVERITIES="${SEVERITIES:-2 3}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
PART_SLUG="${PART_SLUG:-tiny_part}"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_tiny_imagenet_adaptive_patch_alpha_0_2_${PART_SLUG}_${TIMESTAMP}.log}"

read -r -a CORRUPTION_LIST <<< "$CORRUPTION_TYPES"
read -r -a SEVERITY_LIST <<< "$SEVERITIES"
read -r -a DEFENSE_LIST <<< "$DEFENSES"

arch_name_for_model() {
  case "$1" in
    resnet18) echo "ResNet18_tiny_imagenet" ;;
    mobilenetv2) echo "mobilenetv2_tiny_imagenet" ;;
    vgg19_bn) echo "vgg19_bn_tiny_imagenet" ;;
    *)
      echo "Unsupported Tiny-ImageNet model: $1" >&2
      return 1
      ;;
  esac
}

format_rate() {
  printf '%.3f' "$1"
}

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"

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
      echo "--- stdout/stderr ---"
      cat "$tmp_out" 2>/dev/null
      echo "---"
    } >> "$ERROR_LOG"

    if [ "$STOP_ON_FAIL" = "1" ]; then
      rm -f "$tmp_out"
      exit "$exit_code"
    fi
  fi

  rm -f "$tmp_out"
  return "$exit_code"
}

job_matches_cleanup() {
  local name="$1"
  local job model rate cover rate_fmt cover_fmt arch_name
  for job in "${TINY_JOBS[@]}"; do
    IFS='|' read -r model rate cover <<< "$job"
    rate_fmt="$(format_rate "$rate")"
    cover_fmt="$(format_rate "$cover")"
    arch_name="$(arch_name_for_model "$model")" || return 1
    case "$name" in
      adaptive_patch_${rate_fmt}_alpha=0.200_cover=${cover_fmt}_poison_seed=*_arch=${arch_name}) return 0 ;;
    esac
  done
  return 1
}

safe_cleanup_old_alpha() {
  echo
  echo "----- Safe cleanup: adaptive_patch alpha=0.200 for ${PART_TITLE} -----"
  local base="poisoned_train_set/${DATASET}"
  local paths=()
  local path name

  [ -d "$base" ] || return 0
  for path in "$base"/adaptive_patch_*_alpha=0.200_*; do
    [ -d "$path" ] || continue
    name="$(basename "$path")"
    job_matches_cleanup "$name" || continue
    case "$path" in
      poisoned_train_set/${DATASET}/adaptive_patch_*_alpha=0.200_*) paths+=("$path") ;;
      *)
        echo "Refusing suspicious path: ${path}" >&2
        return 1
        ;;
    esac
  done

  echo "Matched ${#paths[@]} directories (expected ${EXPECTED_DIRS}):"
  printf '  %s\n' "${paths[@]}"

  if [ "$CLEAN_OLD" != "1" ]; then
    echo "[SKIP] CLEAN_OLD=${CLEAN_OLD}"
    return 0
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] cleanup skipped"
    return 0
  fi

  for path in "${paths[@]}"; do
    rm -rf -- "$path"
  done
}

base_args() {
  local model="$1"
  local rate="$2"
  local cover="$3"
  echo "-dataset=${DATASET} -poison_type=adaptive_patch -poison_rate=${rate} -cover_rate=${cover} -alpha=${ALPHA} -model=${model} -devices=${DEVICES}"
}

echo "============================================================"
echo "Adaptive-Patch alpha=0.2 rerun: ${PART_TITLE}"
echo "============================================================"
echo "workload     : ${EXPECTED_DIRS} result dirs (~$((EXPECTED_DIRS * 3)) GPU-hours @ 3h/dir)"
echo "python       : ${PYTHON_BIN}"
echo "dataset      : ${DATASET}"
echo "alpha        : ${ALPHA}"
echo "devices      : ${DEVICES}"
echo "transfer     : test_tiny_imagenet.py"
echo "corruptions  : ${CORRUPTION_LIST[*]}"
echo "severities   : ${SEVERITY_LIST[*]}"
echo "defenses     : ${DEFENSE_LIST[*]}"
echo "clean old    : ${CLEAN_OLD}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "jobs         :"
printf '  %s\n' "${TINY_JOBS[@]}"
echo "============================================================"

safe_cleanup_old_alpha || exit 1

for job in "${TINY_JOBS[@]}"; do
  IFS='|' read -r model rate cover <<< "$job"
  args="$(base_args "$model" "$rate" "$cover")"
  echo
  echo "===== model=${model}, pr=${rate}, cover=${cover} ====="

  if [ "$RUN_CREATE" = "1" ]; then
    run_command "${PYTHON_BIN} create_poisoned_set.py ${args}" \
      "Create: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
  fi

  if [ "$RUN_TRAIN" = "1" ]; then
    run_command "${PYTHON_BIN} train_on_poisoned_set.py ${args}" \
      "Train: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
  fi

  if [ "$RUN_TEST" = "1" ]; then
    run_command "${PYTHON_BIN} test_model.py ${args}" \
      "Test: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
  fi

  if [ "$RUN_TRANSFER" = "1" ]; then
    for corruption in "${CORRUPTION_LIST[@]}"; do
      for severity in "${SEVERITY_LIST[@]}"; do
        run_command "${PYTHON_BIN} test_tiny_imagenet.py ${args} -corruption_type=${corruption} -severity=${severity}" \
          "Transfer: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model}) ${corruption}/s${severity}"
      done
    done
  fi

  if [ "$RUN_DEFENSES" = "1" ]; then
    for defense in "${DEFENSE_LIST[@]}"; do
      run_command "${PYTHON_BIN} other_defense.py -defense=${defense} ${args}" \
        "Defense ${defense}: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
    done
  fi
done

echo
echo "Adaptive-Patch alpha=0.2 Tiny-ImageNet ${PART_TITLE} rerun script finished."
