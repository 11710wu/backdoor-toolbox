#!/usr/bin/env bash

# Resume/backfill Tiny-ImageNet UPGD all-to-one runs without NC.
# It skips completed artifacts and only reruns missing non-NC stages.

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DATASET="tiny_imagenet"
MODEL="${MODEL:-resnet18}"
POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-/workspace/backdoor-toolbox/poisoned_train_set2}"
export POISONED_TRAIN_SET_ROOT

LABEL_MODE="all2one"
POISON_RATES="${POISON_RATES:-0.001 0.005}"
EPS_VALUES="${EPS_VALUES:-4 6 8 10 12 16 20 24}"
START_RATE="${START_RATE:-}"
START_EPS="${START_EPS:-}"
START_AFTER="${START_AFTER:-1}"
UPGD_CONSTRAINT="${UPGD_CONSTRAINT:-Linf}"
UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

# Resume mode is conservative by default: do not regenerate raw bases, poison
# tensors, poison indices, or trained checkpoints. Set these to 1 explicitly
# only when you really want to rebuild missing upstream artifacts.
RUN_PREP="${RUN_PREP:-0}"
RUN_CREATE="${RUN_CREATE:-0}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_QWEN_TRANSFER="${RUN_QWEN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"

FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/data/imagenetv2-matched-frequency-tiny-organized}"
TARGET_DOMAIN_QWEN_DIR="${TARGET_DOMAIN_QWEN_DIR:-/workspace/backdoor-toolbox/data/tiny-target-domain-qwen-full-organized}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/resume_tiny_imagenet_${MODEL}_upgd_all2one_skip_nc_${TIMESTAMP}.log}"

case "${MODEL}" in
  resnet18) ARCH_NAME="ResNet18_tiny_imagenet" ;;
  mobilenetv2) ARCH_NAME="mobilenetv2_tiny_imagenet" ;;
  vgg19_bn) ARCH_NAME="vgg19_bn_tiny_imagenet" ;;
  *)
    echo "[ERROR] Unsupported MODEL=${MODEL}. Use resnet18, mobilenetv2, or vgg19_bn."
    exit 2
    ;;
esac

RAW_BASE_DIR="${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=${ARCH_NAME}"
RAW_BASE_PATH="${RAW_BASE_DIR}/upgd_raw_base_${ARCH_NAME}.pt"

base_args() {
  echo "-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
}

upgd_args() {
  echo "$(base_args) -poison_type=upgd -label_mode=${LABEL_MODE} -constraint=${UPGD_CONSTRAINT} -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER}"
}

rate_dir_value() {
  printf "%.3f" "$1"
}

eps_dir_value() {
  printf "%.1f" "$1"
}

poison_dir() {
  local rate_fmt
  local eps_fmt
  rate_fmt="$(rate_dir_value "$1")"
  eps_fmt="$(eps_dir_value "$2")"
  echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/upgd_${rate_fmt}_eps=${eps_fmt}_constraint=${UPGD_CONSTRAINT}_steps=${UPGD_STEPS}_mode=${LABEL_MODE}_mult=${UPGD_STEPS_MULTIPLIER}_poison_seed=2333_arch=${ARCH_NAME}"
}

model_path_for() {
  echo "$(poison_dir "$1" "$2")/${ARCH_NAME}.pt"
}

num_lt() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !((a + 0) < (b + 0)) }'
}

num_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !((a + 0) > (b + 0)) }'
}

num_le() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !((a + 0) <= (b + 0)) }'
}

should_skip_combo_for_resume_start() {
  local rate="$1"
  local eps="$2"

  if [ -z "$START_RATE" ]; then
    return 1
  fi

  if num_lt "$rate" "$START_RATE"; then
    return 0
  fi
  if num_gt "$rate" "$START_RATE"; then
    return 1
  fi

  if [ -z "$START_EPS" ]; then
    return 1
  fi

  if [ "$START_AFTER" = "1" ]; then
    if num_le "$eps" "$START_EPS"; then
      return 0
    fi
  else
    if num_lt "$eps" "$START_EPS"; then
      return 0
    fi
  fi

  return 1
}

defense_result_file() {
  case "$1" in
    SentiNet) echo "sentinet_defense_results.json" ;;
    STRIP) echo "strip_defense_results.json" ;;
    ScaleUp) echo "scaleup_defense_results.json" ;;
    IBD_PSC) echo "ibd_psc_defense_results.json" ;;
    NC) echo "nc_defense_results.json" ;;
    *) echo "" ;;
  esac
}

filtered_defenses() {
  local defense
  for defense in ${DEFENSES}; do
    if [ "$defense" = "NC" ]; then
      continue
    fi
    echo "$defense"
  done
}

run_command() {
  local cmd="$1"
  local description="$2"
  local tmp_out
  local exit_code

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${description}"
  echo "$cmd"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] skipped"
    return 0
  fi

  tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"
  eval "$cmd" 2>&1 | tee "$tmp_out"
  exit_code="${PIPESTATUS[0]}"

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

skip_file() {
  local path="$1"
  local description="$2"
  if [ "$FORCE" != "1" ] && [ -s "$path" ]; then
    echo "[SKIP] ${description}: ${path}"
    return 0
  fi
  return 1
}

skip_poison_creation() {
  local dir="$1"
  if [ "$FORCE" = "1" ]; then
    return 1
  fi

  # Existing Tiny-ImageNet poison sets may store images under either data/ or
  # imgs/. Treat downstream artifacts as a hard skip too, so resume mode never
  # overwrites poison indices for a model that was already trained/tested.
  if { [ -d "$dir/data" ] || [ -d "$dir/imgs" ]; } \
    && [ -e "$dir/labels" ] \
    && [ -e "$dir/poison_indices" ] \
    && compgen -G "${dir}/upgd_*.pth" >/dev/null; then
    echo "[SKIP] poison set exists: ${dir}"
    return 0
  fi

  if [ -s "${dir}/${ARCH_NAME}.pt" ] \
    || [ -s "${dir}/train_results_seed=2333.json" ] \
    || [ -s "${dir}/test_results_seed=2333.json" ] \
    || compgen -G "${dir}/*_defense_results.json" >/dev/null; then
    echo "[SKIP] poison creation skipped because downstream artifacts exist: ${dir}"
    return 0
  fi

  return 1
}

echo "============================================================"
echo "Resume Tiny-ImageNet UPGD all-to-one without NC"
echo "============================================================"
echo "output_root       : ${POISONED_TRAIN_SET_ROOT}"
echo "dataset/model     : ${DATASET}/${MODEL}"
echo "arch              : ${ARCH_NAME}"
echo "label_mode        : ${LABEL_MODE}"
echo "poison_rates      : ${POISON_RATES}"
echo "eps_values        : ${EPS_VALUES}"
echo "resume_start      : rate=${START_RATE:-none}, eps=${START_EPS:-none}, start_after=${START_AFTER}"
echo "defenses          : $(filtered_defenses | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
echo "qwen_transfer     : ${RUN_QWEN_TRANSFER}"
echo "run_prep/create/train: ${RUN_PREP}/${RUN_CREATE}/${RUN_TRAIN}"
echo "force             : ${FORCE}"
echo "dry_run           : ${DRY_RUN}"
echo "error_log         : ${ERROR_LOG}"
if [[ " ${DEFENSES} " == *" NC "* ]]; then
  echo "tiny note         : NC was requested but is skipped for tiny_imagenet"
fi
echo "============================================================"

if [ "$RUN_PREP" = "1" ]; then
  if skip_file "$RAW_BASE_PATH" "raw UPGD base"; then
    :
  else
    run_command \
      "${PYTHON_BIN} create_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0" \
      "Create clean set for raw UPGD base (${MODEL})"
    run_command \
      "${PYTHON_BIN} train_on_poisoned_set.py $(base_args) -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${RAW_BASE_PATH}" \
      "Train raw-input clean base for UPGD (${MODEL})"
  fi
fi

for rate in ${POISON_RATES}; do
  for eps in ${EPS_VALUES}; do
    if should_skip_combo_for_resume_start "$rate" "$eps"; then
      echo "[SKIP] before resume start: ${MODEL} rate=${rate} eps=${eps}"
      continue
    fi

    dir="$(poison_dir "$rate" "$eps")"
    model_path="$(model_path_for "$rate" "$eps")"
    shared="$(upgd_args) -poison_rate=${rate} -eps=${eps}"

    echo
    echo "========== ${MODEL}: rate=${rate}, eps=${eps} =========="
    echo "dir: ${dir}"

    if [ "$RUN_CREATE" = "1" ]; then
      if skip_poison_creation "$dir"; then
        :
      else
        run_command \
          "${PYTHON_BIN} create_poisoned_set.py ${shared} -upgd_model_path=${RAW_BASE_PATH}" \
          "Create UPGD all-to-one poison set ${MODEL} rate=${rate} eps=${eps}"
      fi
    fi

    if [ "$RUN_TRAIN" = "1" ]; then
      if skip_file "$model_path" "trained model"; then
        :
      else
        run_command \
          "${PYTHON_BIN} train_on_poisoned_set.py ${shared}" \
          "Train UPGD all-to-one model ${MODEL} rate=${rate} eps=${eps}"
      fi
    fi

    if [ "$RUN_TEST" = "1" ]; then
      if skip_file "${dir}/test_results_seed=2333.json" "source test results"; then
        :
      else
        run_command \
          "${PYTHON_BIN} test_model.py ${shared}" \
          "Source test ${MODEL} rate=${rate} eps=${eps}"
      fi
    fi

    if [ "$RUN_TRANSFER" = "1" ]; then
      if skip_file "${dir}/test_tiny_target_domain_results.txt" "target-domain transfer results"; then
        :
      else
        run_command \
          "${PYTHON_BIN} test_tiny_target_domain.py ${shared} -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" \
          "Tiny target-domain transfer ${MODEL} rate=${rate} eps=${eps}"
      fi
    fi

    if [ "$RUN_QWEN_TRANSFER" = "1" ]; then
      if skip_file "${dir}/test_tiny_target_domain_qwen_results.txt" "Qwen target-domain transfer results"; then
        :
      else
        run_command \
          "${PYTHON_BIN} test_tiny_target_domain_qwen.py ${shared} -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_QWEN_DIR}" \
          "Tiny Qwen target-domain transfer ${MODEL} rate=${rate} eps=${eps}"
      fi
    fi

    if [ "$RUN_DEFENSES" = "1" ]; then
      for defense in $(filtered_defenses); do
        result_file="$(defense_result_file "$defense")"
        if [ -n "$result_file" ] && skip_file "${dir}/${result_file}" "defense ${defense} results"; then
          :
        else
          run_command \
            "${PYTHON_BIN} other_defense.py -defense=${defense} ${shared}" \
            "Defense ${defense} ${MODEL} rate=${rate} eps=${eps}"
        fi
      done
    fi
  done
done

echo
echo "============================================================"
echo "Resume script finished: ${MODEL}. Failed commands, if any: ${ERROR_LOG}"
echo "============================================================"
