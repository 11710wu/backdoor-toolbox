#!/usr/bin/env bash
#
# Adaptive-Patch alpha=0.2 unified rerun (CIFAR-10 + MNIST-M + Tiny-ImageNet)
#
# Merges the previous three scripts into one sequential run:
#   1) CIFAR-10 + MNIST-M: 2 datasets x 3 models x 3 pr/cover = 18 dirs (~18h @ 1h/dir)
#   2) Tiny-ImageNet: 3 models x 3 pr/cover = 9 dirs (~27h @ 3h/dir)
# Total: 27 result dirs, ~45 GPU-hours serial
#
# Suggested launch:
#   DEVICES=0 bash run/rerun_adaptive_patch_alpha_0_2_all.sh
#
# Useful overrides:
#   DRY_RUN=1 bash run/rerun_adaptive_patch_alpha_0_2_all.sh
#   RUN_TINY=0 bash run/rerun_adaptive_patch_alpha_0_2_all.sh          # only CIFAR/MNIST-M
#   RUN_CIFAR_MNISTM=0 bash run/rerun_adaptive_patch_alpha_0_2_all.sh  # only Tiny-ImageNet
#   CLEAN_OLD=0 bash run/rerun_adaptive_patch_alpha_0_2_all.sh

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

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
RUN_CIFAR_MNISTM="${RUN_CIFAR_MNISTM:-1}"
RUN_TINY="${RUN_TINY:-1}"

DATASETS="${DATASETS:-cifar10 mnistm}"
MODELS="${MODELS:-resnet18 mobilenetv2 vgg19_bn}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"
CORRUPTION_TYPES="${CORRUPTION_TYPES:-frost}"
SEVERITIES="${SEVERITIES:-2 3}"

# Tiny-ImageNet: all 9 jobs from former part0 + part1
TINY_JOBS=(
  "resnet18|0.05|0.1"
  "resnet18|0.01|0.02"
  "resnet18|0.005|0.01"
  "mobilenetv2|0.05|0.1"
  "mobilenetv2|0.01|0.02"
  "mobilenetv2|0.005|0.01"
  "vgg19_bn|0.05|0.1"
  "vgg19_bn|0.01|0.02"
  "vgg19_bn|0.005|0.01"
)

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/rerun_adaptive_patch_alpha_0_2_all_${TIMESTAMP}.log}"

read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a MODEL_LIST <<< "$MODELS"
read -r -a DEFENSE_LIST <<< "$DEFENSES"
read -r -a CORRUPTION_LIST <<< "$CORRUPTION_TYPES"
read -r -a SEVERITY_LIST <<< "$SEVERITIES"

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

arch_name_for_tiny_model() {
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

base_args() {
  local dataset="$1"
  local model="$2"
  local rate="$3"
  local cover="$4"
  echo "-dataset=${dataset} -poison_type=adaptive_patch -poison_rate=${rate} -cover_rate=${cover} -alpha=${ALPHA} -model=${model} -devices=${DEVICES}"
}

safe_cleanup_cifar_mnistm() {
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

tiny_job_matches_cleanup() {
  local name="$1"
  local job model rate cover rate_fmt cover_fmt arch_name
  for job in "${TINY_JOBS[@]}"; do
    IFS='|' read -r model rate cover <<< "$job"
    rate_fmt="$(format_rate "$rate")"
    cover_fmt="$(format_rate "$cover")"
    arch_name="$(arch_name_for_tiny_model "$model")" || return 1
    case "$name" in
      adaptive_patch_${rate_fmt}_alpha=0.200_cover=${cover_fmt}_poison_seed=*_arch=${arch_name}) return 0 ;;
    esac
  done
  return 1
}

safe_cleanup_tiny() {
  echo
  echo "----- Safe cleanup: adaptive_patch alpha=0.200 for Tiny-ImageNet -----"
  local base="poisoned_train_set/tiny_imagenet"
  local paths=()
  local path name

  [ -d "$base" ] || return 0
  for path in "$base"/adaptive_patch_*_alpha=0.200_*; do
    [ -d "$path" ] || continue
    name="$(basename "$path")"
    tiny_job_matches_cleanup "$name" || continue
    case "$path" in
      poisoned_train_set/tiny_imagenet/adaptive_patch_*_alpha=0.200_*) paths+=("$path") ;;
      *)
        echo "Refusing suspicious path: ${path}" >&2
        return 1
        ;;
    esac
  done

  echo "Matched ${#paths[@]} directories (expected 9):"
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

run_cifar_mnistm_phase() {
  echo
  echo "============================================================"
  echo "Phase A: CIFAR-10 + MNIST-M (18 configs, ~18h @ 1h/dir)"
  echo "============================================================"

  safe_cleanup_cifar_mnistm || exit 1

  local dataset model transfer rate cover args defense
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
}

run_tiny_phase() {
  echo
  echo "============================================================"
  echo "Phase B: Tiny-ImageNet (9 configs, ~27h @ 3h/dir)"
  echo "============================================================"

  safe_cleanup_tiny || exit 1

  local job model rate cover args corruption severity defense
  for job in "${TINY_JOBS[@]}"; do
    IFS='|' read -r model rate cover <<< "$job"
    args="$(base_args "tiny_imagenet" "$model" "$rate" "$cover")"
    echo
    echo "===== Tiny-ImageNet model=${model}, pr=${rate}, cover=${cover} ====="

    if [ "$RUN_CREATE" = "1" ]; then
      run_command "${PYTHON_BIN} create_poisoned_set.py ${args}" \
        "Create: tiny_imagenet adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
    fi

    if [ "$RUN_TRAIN" = "1" ]; then
      run_command "${PYTHON_BIN} train_on_poisoned_set.py ${args}" \
        "Train: tiny_imagenet adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
    fi

    if [ "$RUN_TEST" = "1" ]; then
      run_command "${PYTHON_BIN} test_model.py ${args}" \
        "Test: tiny_imagenet adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
    fi

    if [ "$RUN_TRANSFER" = "1" ]; then
      for corruption in "${CORRUPTION_LIST[@]}"; do
        for severity in "${SEVERITY_LIST[@]}"; do
          run_command "${PYTHON_BIN} test_tiny_imagenet.py ${args} -corruption_type=${corruption} -severity=${severity}" \
            "Transfer: tiny_imagenet adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model}) ${corruption}/s${severity}"
        done
      done
    fi

    if [ "$RUN_DEFENSES" = "1" ]; then
      for defense in "${DEFENSE_LIST[@]}"; do
        run_command "${PYTHON_BIN} other_defense.py -defense=${defense} ${args}" \
          "Defense ${defense}: tiny_imagenet adaptive_patch rate=${rate} cover=${cover} alpha=${ALPHA} (${model})"
      done
    fi
  done
}

echo "============================================================"
echo "Adaptive-Patch alpha=0.2 unified rerun"
echo "============================================================"
echo "python            : ${PYTHON_BIN}"
echo "devices           : ${DEVICES}"
echo "alpha             : ${ALPHA}"
echo "run cifar/mnistm  : ${RUN_CIFAR_MNISTM} (18 dirs, ~18h)"
echo "run tiny          : ${RUN_TINY} (9 dirs, ~27h)"
echo "total estimate    : ~45 GPU-hours serial if both phases enabled"
echo "clean old         : ${CLEAN_OLD}"
echo "dry run           : ${DRY_RUN}"
echo "stop on fail      : ${STOP_ON_FAIL}"
echo "error log         : ${ERROR_LOG}"
echo "============================================================"

if [ "$RUN_CIFAR_MNISTM" = "1" ]; then
  run_cifar_mnistm_phase
else
  echo "[SKIP] CIFAR-10/MNIST-M phase (RUN_CIFAR_MNISTM=0)"
fi

if [ "$RUN_TINY" = "1" ]; then
  run_tiny_phase
else
  echo "[SKIP] Tiny-ImageNet phase (RUN_TINY=0)"
fi

echo
echo "============================================================"
echo "Adaptive-Patch alpha=0.2 unified rerun finished."
echo "Check ${ERROR_LOG} for failures."
echo "============================================================"
