#!/bin/bash

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/backfill_cifar10_upgd_missing_results_${TIMESTAMP}.log"

set +e

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
DATASET="cifar10"
POISON_TYPE="upgd"
CONSTRAINT="Linf"
UPGD_STEPS="100"
UPGD_STEPS_MULTIPLIER="5"
POISON_SEED="2333"
DRY_RUN="${DRY_RUN:-0}"
RETRAIN_RAW_BASE="${RETRAIN_RAW_BASE:-1}"
RECREATE_UPGD="${RECREATE_UPGD:-1}"
RERUN_RESULTS="${RERUN_RESULTS:-1}"

# This script now defaults to a full rerun for the two incomplete CIFAR10 UPGD
# models. Command-line model args must stay lowercase; supervisor maps resnet18
# to the ResNet18_cifar10 architecture used in directory names.
MODELS="${MODELS:-resnet18 vgg19_bn}"
RATES="${RATES:-0.05 0.01 0.005}"
EPSES="${EPSES:-4 6 8 10 12 16 20 24}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC NC}"

run_command() {
    local original_cmd="$1"
    local description="$2"
    local tmp_out
    tmp_out=$(mktemp 2>/dev/null || echo "/tmp/backfill_cmd_$$_${RANDOM}.out")

    echo
    echo ">>> $description"
    echo "$original_cmd"
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    eval "$original_cmd" 2>&1 | tee "$tmp_out"
    local exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -ne 0 ]; then
        {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed (exit: $exit_code)"
            echo "command: $original_cmd"
            echo "description: $description"
            echo "--- stdout+stderr ---"
            cat "$tmp_out"
            echo "---"
        } >> "$ERROR_LOG"
    fi
    rm -f "$tmp_out"
    return "$exit_code"
}

arch_name() {
    case "$1" in
        resnet18) echo "ResNet18_cifar10" ;;
        mobilenetv2) echo "mobilenetv2_cifar10" ;;
        vgg19_bn) echo "vgg19_bn_cifar10" ;;
        *) echo "${1}_cifar10" ;;
    esac
}

model_file_name() {
    case "$1" in
        resnet18) echo "ResNet18_cifar10.pt" ;;
        mobilenetv2) echo "mobilenetv2_cifar10.pt" ;;
        vgg19_bn) echo "vgg19_bn_cifar10.pt" ;;
        *) echo "${1}_cifar10.pt" ;;
    esac
}

raw_base_dir() {
    local model="$1"
    local arch
    arch=$(arch_name "$model")
    echo "poisoned_train_set/${DATASET}/upgd_raw_base_0.000_poison_seed=${POISON_SEED}_arch=${arch}"
}

raw_base_path() {
    local model="$1"
    local arch
    arch=$(arch_name "$model")
    echo "$(raw_base_dir "$model")/upgd_raw_base_${arch}.pt"
}

rate_dir() {
    python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.3f}")
PY
}

eps_dir() {
    python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.1f}")
PY
}

poison_dir() {
    local model="$1"
    local rate="$2"
    local eps="$3"
    local arch
    arch=$(arch_name "$model")
    echo "poisoned_train_set/${DATASET}/upgd_$(rate_dir "$rate")_eps=$(eps_dir "$eps")_constraint=${CONSTRAINT}_steps=${UPGD_STEPS}_mode=clean_mult=${UPGD_STEPS_MULTIPLIER}_poison_seed=${POISON_SEED}_arch=${arch}"
}

has_nc_result() {
    compgen -G "$1/nc_detection_*.json" >/dev/null
}

remove_path() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "remove: $path"
        if [ "$DRY_RUN" != "1" ]; then
            rm -rf "$path"
        fi
    fi
}

echo "=========================================="
echo "Rerun CIFAR10 UPGD results"
echo "models: $MODELS"
echo "rates: $RATES"
echo "eps: $EPSES"
echo "defenses: $DEFENSES"
echo "devices: $DEVICES"
echo "retrain_raw_base: $RETRAIN_RAW_BASE"
echo "recreate_upgd: $RECREATE_UPGD"
echo "rerun_results: $RERUN_RESULTS"
echo "dry_run: $DRY_RUN"
echo "error log: $ERROR_LOG"
echo "=========================================="

for model in $MODELS; do
    raw_dir=$(raw_base_dir "$model")
    raw_model_path=$(raw_base_path "$model")

    echo
    echo "----- Raw clean base: ${DATASET} ${model} -----"
    run_command "${PYTHON_BIN} create_poisoned_set.py -dataset=${DATASET} -model=${model} -devices=${DEVICES} -poison_type=none -poison_rate=0.0" "Create clean set for raw UPGD base: ${DATASET} ${model}"
    if [ "$RETRAIN_RAW_BASE" = "1" ]; then
        remove_path "$raw_dir"
    fi
    if [ "$RETRAIN_RAW_BASE" = "1" ] || [ ! -f "$raw_model_path" ]; then
        run_command "${PYTHON_BIN} train_on_poisoned_set.py -dataset=${DATASET} -model=${model} -devices=${DEVICES} -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${raw_model_path}" "Train raw-input clean base: ${DATASET} ${model}"
    else
        echo "skip raw base: $raw_model_path"
    fi

    if [ ! -f "$raw_model_path" ] && [ "$DRY_RUN" != "1" ]; then
        echo "!!! Raw base missing, skip model: $raw_model_path"
        continue
    fi

    for rate in $RATES; do
        for eps in $EPSES; do
            dir=$(poison_dir "$model" "$rate" "$eps")
            model_path="$dir/$(model_file_name "$model")"

            if [ "$RECREATE_UPGD" = "1" ]; then
                remove_path "$dir"
            fi

            if [ "$RECREATE_UPGD" = "1" ] || [ ! -f "$dir/labels" ] || [ ! -f "$dir/poison_indices" ] || [ ! -f "$dir/upgd_0.pth" ]; then
                run_command "${PYTHON_BIN} create_poisoned_set.py -dataset=${DATASET} -model=${model} -devices=${DEVICES} -poison_type=${POISON_TYPE} -poison_rate=${rate} -eps=${eps} -constraint=${CONSTRAINT} -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER} -label_mode=clean -upgd_model_path=${raw_model_path}" "Create UPGD poison set: ${DATASET} ${model} rate=${rate} eps=${eps}"
            fi

            if [ "$DRY_RUN" != "1" ] && { [ ! -f "$dir/labels" ] || [ ! -f "$dir/poison_indices" ] || [ ! -f "$dir/upgd_0.pth" ]; }; then
                echo "!!! UPGD poisoned set still incomplete, skip: $dir"
                continue
            fi

            base_args="-dataset=${DATASET} -model=${model} -devices=${DEVICES} -poison_type=${POISON_TYPE} -poison_rate=${rate} -eps=${eps} -constraint=${CONSTRAINT} -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER} -label_mode=clean"

            if [ "$RERUN_RESULTS" = "1" ]; then
                remove_path "$model_path"
                remove_path "$dir/meta_info_$(model_file_name "$model")"
                remove_path "$dir/train_results_seed=${POISON_SEED}.json"
                remove_path "$dir/test_results_seed=${POISON_SEED}.json"
                remove_path "$dir/test_stl10_results.txt"
                remove_path "$dir/sentinet_defense_results.json"
                remove_path "$dir/strip_defense_results.json"
                remove_path "$dir/scaleup_defense_results.json"
                remove_path "$dir/ibd_psc_defense_results.json"
                if [ "$DRY_RUN" = "1" ]; then
                    echo "remove: $dir/nc_detection_*.json"
                else
                    rm -f "$dir"/nc_detection_*.json
                fi
            fi

            if [ "$RERUN_RESULTS" = "1" ] || [ ! -f "$model_path" ] || [ ! -f "$dir/train_results_seed=${POISON_SEED}.json" ]; then
                run_command "${PYTHON_BIN} train_on_poisoned_set.py $base_args" "Train: ${DATASET} ${POISON_TYPE} ${model} rate=${rate} eps=${eps}"
            else
                echo "skip train: $dir"
            fi

            if [ "$DRY_RUN" != "1" ] && [ ! -f "$model_path" ]; then
                echo "!!! Model still missing after train attempt, skip test/defense: $model_path"
                continue
            fi

            if [ "$RERUN_RESULTS" = "1" ] || [ ! -f "$dir/test_results_seed=${POISON_SEED}.json" ]; then
                run_command "${PYTHON_BIN} test_model.py $base_args" "Local test: ${DATASET} ${POISON_TYPE} ${model} rate=${rate} eps=${eps}"
            else
                echo "skip local test: $dir"
            fi

            if [ "$RERUN_RESULTS" = "1" ] || [ ! -f "$dir/test_stl10_results.txt" ]; then
                run_command "${PYTHON_BIN} test_stl10.py $base_args" "STL10 transfer: ${DATASET} ${POISON_TYPE} ${model} rate=${rate} eps=${eps}"
            else
                echo "skip STL10 transfer: $dir"
            fi

            for defense in $DEFENSES; do
                case "$defense" in
                    SentiNet)
                        [ "$RERUN_RESULTS" != "1" ] && [ -f "$dir/sentinet_defense_results.json" ] && { echo "skip SentiNet: $dir"; continue; }
                        ;;
                    STRIP)
                        [ "$RERUN_RESULTS" != "1" ] && [ -f "$dir/strip_defense_results.json" ] && { echo "skip STRIP: $dir"; continue; }
                        ;;
                    ScaleUp)
                        [ "$RERUN_RESULTS" != "1" ] && [ -f "$dir/scaleup_defense_results.json" ] && { echo "skip ScaleUp: $dir"; continue; }
                        ;;
                    IBD_PSC)
                        [ "$RERUN_RESULTS" != "1" ] && [ -f "$dir/ibd_psc_defense_results.json" ] && { echo "skip IBD_PSC: $dir"; continue; }
                        ;;
                    NC)
                        [ "$RERUN_RESULTS" != "1" ] && has_nc_result "$dir" && { echo "skip NC: $dir"; continue; }
                        ;;
                esac

                run_command "${PYTHON_BIN} other_defense.py -defense=${defense} $base_args" "Defense: ${defense} ${DATASET} ${POISON_TYPE} ${model} rate=${rate} eps=${eps}"
            done
        done
    done
done

echo
echo "Rerun finished. Check failures in: $ERROR_LOG"
