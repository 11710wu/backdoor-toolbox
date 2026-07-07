#!/usr/bin/env bash

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DATASET="tiny_imagenet"
MODEL="${MODEL:-resnet18}"
case "$MODEL" in
  resnet18) ARCH_NAME="ResNet18_tiny_imagenet" ;;
  mobilenetv2) ARCH_NAME="mobilenetv2_tiny_imagenet" ;;
  vgg19_bn) ARCH_NAME="vgg19_bn_tiny_imagenet" ;;
  *)
    echo "Unsupported Tiny-ImageNet model: ${MODEL}" >&2
    exit 1
    ;;
esac
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
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_tiny_imagenet_adaptive_patch_alpha_0_2_${MODEL}_${TIMESTAMP}.log}"

read -r -a CORRUPTION_LIST <<< "$CORRUPTION_TYPES"
read -r -a SEVERITY_LIST <<< "$SEVERITIES"
read -r -a DEFENSE_LIST <<< "$DEFENSES"

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

poison_cover_pairs() {
  echo "0.05 0.1"
  echo "0.01 0.02"
  echo "0.005 0.01"
}

safe_cleanup_old_alpha() {
  echo
  echo "----- Safe cleanup: adaptive_patch alpha=0.200 for ${DATASET}/${MODEL} -----"
  local base="poisoned_train_set/${DATASET}"
  local paths=()
  local path name

  [ -d "$base" ] || return 0
  for path in "$base"/adaptive_patch_*_alpha=0.200_*_arch="${ARCH_NAME}"; do
    [ -d "$path" ] || continue
    name="$(basename "$path")"
    case "$path" in
      poisoned_train_set/${DATASET}/adaptive_patch_*_alpha=0.200_*_arch=${ARCH_NAME}) ;;
      *)
        echo "Refusing suspicious path: ${path}" >&2
        return 1
        ;;
    esac
    case "$name" in
      adaptive_patch_*_alpha=0.200_*_arch=${ARCH_NAME}) paths+=("$path") ;;
      *)
        echo "Refusing suspicious directory name: ${name}" >&2
        return 1
        ;;
    esac
  done

  echo "Matched ${#paths[@]} directories:"
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
  local rate="$1"
  local cover="$2"
  echo "-dataset=${DATASET} -poison_type=adaptive_patch -poison_rate=${rate} -cover_rate=${cover} -alpha=${ALPHA} -model=${MODEL} -devices=${DEVICES}"
}

echo "============================================================"
echo "Adaptive-Patch alpha=0.2 rerun: Tiny-ImageNet ${MODEL}"
echo "============================================================"
echo "python       : ${PYTHON_BIN}"
echo "dataset      : ${DATASET}"
echo "model        : ${MODEL}"
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
echo "============================================================"

safe_cleanup_old_alpha || exit 1

if [ "$RUN_CREATE" = "1" ]; then
  echo "----- 1. Creation -----"
  while read -r rate cover; do
    args="$(base_args "$rate" "$cover")"
    run_command "${PYTHON_BIN} create_poisoned_set.py ${args}" \
      "Create: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${MODEL})"
  done < <(poison_cover_pairs)
fi

if [ "$RUN_TRAIN" = "1" ]; then
  echo "----- 2. Training -----"
  while read -r rate cover; do
    args="$(base_args "$rate" "$cover")"
    run_command "${PYTHON_BIN} train_on_poisoned_set.py ${args}" \
      "Train: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${MODEL})"
  done < <(poison_cover_pairs)
fi

if [ "$RUN_TEST" = "1" ]; then
  echo "----- 3. Local Testing -----"
  while read -r rate cover; do
    args="$(base_args "$rate" "$cover")"
    run_command "${PYTHON_BIN} test_model.py ${args}" \
      "Test: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${MODEL})"
  done < <(poison_cover_pairs)
fi

if [ "$RUN_TRANSFER" = "1" ]; then
  echo "----- 4. Transfer Testing -----"
  while read -r rate cover; do
    args="$(base_args "$rate" "$cover")"
    for corruption in "${CORRUPTION_LIST[@]}"; do
      for severity in "${SEVERITY_LIST[@]}"; do
        run_command "${PYTHON_BIN} test_tiny_imagenet.py ${args} -corruption_type=${corruption} -severity=${severity}" \
          "Transfer: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${MODEL}) ${corruption}/s${severity}"
      done
    done
  done < <(poison_cover_pairs)
fi

if [ "$RUN_DEFENSES" = "1" ]; then
  echo "----- 5. Defenses -----"
  for defense in "${DEFENSE_LIST[@]}"; do
    while read -r rate cover; do
      args="$(base_args "$rate" "$cover")"
      run_command "${PYTHON_BIN} other_defense.py -defense=${defense} ${args}" \
        "Defense ${defense}: ${DATASET} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${MODEL})"
    done < <(poison_cover_pairs)
  done
fi

echo
echo "Adaptive-Patch alpha=0.2 Tiny-ImageNet ${MODEL} rerun script finished."
