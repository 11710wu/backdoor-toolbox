#!/usr/bin/env bash
#
# Adaptive-Patch alpha=0.2 rerun — Script 1/5 (GPU0)
# Workload: 2 datasets x 3 models x 3 pr/cover = 18 result dirs (~18h @ 1h/dir)
#
# Suggested launch:
#   DEVICES=0 bash run/rerun_adaptive_patch_alpha_0_2_cifar10_mnistm.sh

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
ALPHA="${ALPHA:-0.2}"
CLEAN_OLD="${CLEAN_OLD:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

DATASETS="${DATASETS:-cifar10 mnistm}"
MODELS="${MODELS:-resnet18 mobilenetv2 vgg19_bn}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_adaptive_patch_alpha_0_2_cifar10_mnistm_${TIMESTAMP}.log}"

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a MODEL_LIST <<< "$MODELS"
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

transfer_script() {
  case "$1" in
    cifar10) echo "test_stl10.py" ;;
    mnistm) echo "test_mnist.py" ;;
    *)
      echo "Unsupported dataset for transfer: $1" >&2
      return 1
      ;;
  esac
}

safe_cleanup_old_alpha() {
  echo
  echo "----- Safe cleanup: adaptive_patch alpha=0.200 for CIFAR-10/MNIST-M -----"
  local paths=()
  local dataset base path name

  for dataset in "${DATASET_LIST[@]}"; do
    case "$dataset" in
      cifar10|mnistm) ;;
      *)
        echo "Refusing cleanup for non-whitelisted dataset: ${dataset}" >&2
        return 1
        ;;
    esac

    base="poisoned_train_set/${dataset}"
    [ -d "$base" ] || continue
    for path in "$base"/adaptive_patch_*_alpha=0.200_*; do
      [ -d "$path" ] || continue
      name="$(basename "$path")"
      case "$path" in
        poisoned_train_set/${dataset}/adaptive_patch_*_alpha=0.200_*) ;;
        *)
          echo "Refusing suspicious path: ${path}" >&2
          return 1
          ;;
      esac
      case "$name" in
        adaptive_patch_*_alpha=0.200_*) paths+=("$path") ;;
        *)
          echo "Refusing suspicious directory name: ${name}" >&2
          return 1
          ;;
      esac
    done
  done

  echo "Matched ${#paths[@]} directories (expected 18):"
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
  local dataset="$1"
  local model="$2"
  local rate="$3"
  local cover="$4"
  echo "-dataset=${dataset} -poison_type=adaptive_patch -poison_rate=${rate} -cover_rate=${cover} -alpha=${ALPHA} -model=${model} -devices=${DEVICES}"
}

echo "============================================================"
echo "Adaptive-Patch alpha=0.2 rerun: CIFAR-10 + MNIST-M (1/5)"
echo "============================================================"
echo "workload     : 18 result dirs (~18 GPU-hours @ 1h/dir)"
echo "python       : ${PYTHON_BIN}"
echo "datasets     : ${DATASET_LIST[*]}"
echo "models       : ${MODEL_LIST[*]}"
echo "alpha        : ${ALPHA}"
echo "devices      : ${DEVICES}"
echo "defenses     : ${DEFENSE_LIST[*]}"
echo "clean old    : ${CLEAN_OLD}"
echo "dry run      : ${DRY_RUN}"
echo "stop on fail : ${STOP_ON_FAIL}"
echo "error log    : ${ERROR_LOG}"
echo "============================================================"

safe_cleanup_old_alpha || exit 1

for dataset in "${DATASET_LIST[@]}"; do
  transfer="$(transfer_script "$dataset")" || exit 1
  for model in "${MODEL_LIST[@]}"; do
    echo
    echo "===== Dataset=${dataset}, model=${model} ====="

    if [ "$RUN_CREATE" = "1" ]; then
      echo "----- 1. Creation -----"
      while read -r rate cover; do
        args="$(base_args "$dataset" "$model" "$rate" "$cover")"
        run_command "${PYTHON_BIN} create_poisoned_set.py ${args}" \
          "Create: ${dataset} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
      done < <(poison_cover_pairs)
    fi

    if [ "$RUN_TRAIN" = "1" ]; then
      echo "----- 2. Training -----"
      while read -r rate cover; do
        args="$(base_args "$dataset" "$model" "$rate" "$cover")"
        run_command "${PYTHON_BIN} train_on_poisoned_set.py ${args}" \
          "Train: ${dataset} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
      done < <(poison_cover_pairs)
    fi

    if [ "$RUN_TEST" = "1" ]; then
      echo "----- 3. Local Testing -----"
      while read -r rate cover; do
        args="$(base_args "$dataset" "$model" "$rate" "$cover")"
        run_command "${PYTHON_BIN} test_model.py ${args}" \
          "Test: ${dataset} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
      done < <(poison_cover_pairs)
    fi

    if [ "$RUN_TRANSFER" = "1" ]; then
      echo "----- 4. Transfer Testing (${transfer}) -----"
      while read -r rate cover; do
        args="$(base_args "$dataset" "$model" "$rate" "$cover")"
        run_command "${PYTHON_BIN} ${transfer} ${args}" \
          "Transfer: ${transfer} ${dataset} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
      done < <(poison_cover_pairs)
    fi

    if [ "$RUN_DEFENSES" = "1" ]; then
      echo "----- 5. Defenses -----"
      for defense in "${DEFENSE_LIST[@]}"; do
        while read -r rate cover; do
          args="$(base_args "$dataset" "$model" "$rate" "$cover")"
          run_command "${PYTHON_BIN} other_defense.py -defense=${defense} ${args}" \
            "Defense ${defense}: ${dataset} adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
        done < <(poison_cover_pairs)
      done
    fi
  done
done

echo
echo "Adaptive-Patch alpha=0.2 CIFAR-10/MNIST-M rerun script finished."
