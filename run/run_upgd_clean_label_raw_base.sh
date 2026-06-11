#!/usr/bin/env bash
# Rerun UPGD for one existing dataset/model pair with a raw-input clean base.
# Default label mode is clean-label; set LABEL_MODE=all2one to reproduce old labels.

set +e

SCRIPT_NAME="$(basename "$0")"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/${SCRIPT_NAME%.sh}_${TIMESTAMP}.log}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet18}"
DEVICES="${DEVICES:-0}"
LABEL_MODE="${LABEL_MODE:-clean}"
UPGD_CONSTRAINT="${UPGD_CONSTRAINT:-Linf}"
UPGD_STEPS="${UPGD_STEPS:-100}"
UPGD_STEPS_MULTIPLIER="${UPGD_STEPS_MULTIPLIER:-5}"

RUN_PREP="${RUN_PREP:-1}"
FORCE_RAW_BASE="${FORCE_RAW_BASE:-0}"
RUN_CREATE="${RUN_CREATE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_TRANSFER="${RUN_TRANSFER:-1}"
RUN_DEFENSES="${RUN_DEFENSES:-1}"
# UPGD Tiny-ImageNet transfer defaults to the two target-domain suites only.
# Set RUN_TINY_CORRUPTION=1 explicitly if you also want Tiny-ImageNet-C.
RUN_TINY_CORRUPTION="${RUN_TINY_CORRUPTION:-0}"
RUN_TINY_TARGET_DOMAIN="${RUN_TINY_TARGET_DOMAIN:-1}"

DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC NC}"
CORRUPTION_TYPES="${CORRUPTION_TYPES:-frost}"
CORRUPTION_SEVERITIES="${CORRUPTION_SEVERITIES:-2 3}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/data/imagenetv2-matched-frequency-tiny-organized}"
TARGET_DOMAIN_QWEN_DIR="${TARGET_DOMAIN_QWEN_DIR:-/workspace/data/tiny-target-domain-qwen-full-organized}"

case "${DATASET}:${MODEL}" in
    cifar10:resnet18)
        ARCH_NAME="ResNet18_cifar10"
        ;;
    cifar10:mobilenetv2)
        ARCH_NAME="mobilenetv2_cifar10"
        ;;
    cifar10:vgg19_bn)
        ARCH_NAME="vgg19_bn_cifar10"
        ;;
    tiny_imagenet:resnet18)
        ARCH_NAME="ResNet18_tiny_imagenet"
        ;;
    tiny_imagenet:mobilenetv2)
        ARCH_NAME="mobilenetv2_tiny_imagenet"
        ;;
    tiny_imagenet:vgg19_bn)
        ARCH_NAME="vgg19_bn_tiny_imagenet"
        ;;
    *)
        echo "[ERROR] Unsupported DATASET/MODEL pair: DATASET=${DATASET}, MODEL=${MODEL}"
        echo "        Supported models: resnet18, mobilenetv2, vgg19_bn"
        echo "        Supported datasets: cifar10, tiny_imagenet"
        exit 2
        ;;
esac

if [ -n "${POISON_RATES:-}" ]; then
    read -r -a POISON_RATE_LIST <<< "$POISON_RATES"
elif [ "$DATASET" = "cifar10" ]; then
    POISON_RATE_LIST=(0.05 0.01 0.005)
else
    POISON_RATE_LIST=(0.005 0.001)
fi

if [ -n "${EPS_VALUES:-}" ]; then
    read -r -a EPS_LIST <<< "$EPS_VALUES"
else
    EPS_LIST=(4 6 8 10 12 16 20 24)
fi

# UPGD delta generation feeds raw [0,1] tensors to the base model, so keep this
# checkpoint separate from the normal clean baseline trained with Normalize.
RAW_BASE_DIR="${RAW_BASE_DIR:-poisoned_train_set/${DATASET}/upgd_raw_base_0.000_poison_seed=2333_arch=${ARCH_NAME}}"
UPGD_CLEAN_MODEL_PATH="${UPGD_CLEAN_MODEL_PATH:-${RAW_BASE_DIR}/upgd_raw_base_${ARCH_NAME}.pt}"

BASE_ARGS="-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES}"
UPGD_SHARED_ARGS="-dataset=${DATASET} -model=${MODEL} -devices=${DEVICES} -poison_type=upgd -label_mode=${LABEL_MODE} -constraint=${UPGD_CONSTRAINT} -upgd_steps=${UPGD_STEPS} -upgd_steps_multiplier=${UPGD_STEPS_MULTIPLIER}"

run_command() {
    local cmd="$1"
    local description="$2"
    local tmp_out
    local exit_code

    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${description}"
    echo "$cmd"

    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi

    tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"
    eval "$cmd" 2>&1 | tee "$tmp_out"
    exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -ne 0 ]; then
        {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command failed (exit=${exit_code})"
            echo "Command: ${cmd}"
            echo "Description: ${description}"
            echo "--- stdout+stderr ---"
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

run_stage_for_grid() {
    local stage_name="$1"
    local command_template="$2"
    local description_template="$3"
    local rate
    local eps
    local cmd
    local description

    echo
    echo "----- ${stage_name}: ${DATASET} ${MODEL} -----"
    for rate in "${POISON_RATE_LIST[@]}"; do
        for eps in "${EPS_LIST[@]}"; do
            cmd="${command_template//__RATE__/${rate}}"
            cmd="${cmd//__EPS__/${eps}}"
            description="${description_template//__RATE__/${rate}}"
            description="${description//__EPS__/${eps}}"
            run_command "$cmd" "$description"
        done
    done
}

echo "=========================================="
echo "UPGD raw-base rerun"
echo "dataset=${DATASET}"
echo "model=${MODEL}"
echo "arch=${ARCH_NAME}"
echo "label_mode=${LABEL_MODE}"
echo "raw_base=${UPGD_CLEAN_MODEL_PATH}"
echo "error_log=${ERROR_LOG}"
echo "=========================================="

if [ "$RUN_PREP" = "1" ]; then
    echo
    echo "----- 0. Raw clean base preparation: ${DATASET} ${MODEL} -----"
    run_command "${PYTHON_BIN} create_poisoned_set.py ${BASE_ARGS} -poison_type=none -poison_rate=0.0" \
        "Create clean set for raw UPGD base (${DATASET}, ${MODEL})"

    if [ "$FORCE_RAW_BASE" = "1" ] || [ ! -f "$UPGD_CLEAN_MODEL_PATH" ]; then
        run_command "${PYTHON_BIN} train_on_poisoned_set.py ${BASE_ARGS} -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${UPGD_CLEAN_MODEL_PATH}" \
            "Train raw-input clean base for UPGD (${DATASET}, ${MODEL})"
    else
        echo "[SKIP] Raw clean base exists: ${UPGD_CLEAN_MODEL_PATH}"
        echo "       Set FORCE_RAW_BASE=1 to retrain it."
    fi
fi

if [ "$RUN_CREATE" = "1" ]; then
    run_stage_for_grid "1. UPGD poison-set creation" \
        "${PYTHON_BIN} create_poisoned_set.py ${UPGD_SHARED_ARGS} -poison_rate=__RATE__ -eps=__EPS__ -upgd_model_path=${UPGD_CLEAN_MODEL_PATH}" \
        "Create UPGD poison set: ${DATASET} ${MODEL} rate=__RATE__ eps=__EPS__"
fi

if [ "$RUN_TRAIN" = "1" ]; then
    run_stage_for_grid "2. Backdoored training" \
        "${PYTHON_BIN} train_on_poisoned_set.py ${UPGD_SHARED_ARGS} -poison_rate=__RATE__ -eps=__EPS__" \
        "Train UPGD model: ${DATASET} ${MODEL} rate=__RATE__ eps=__EPS__"
fi

if [ "$RUN_TEST" = "1" ]; then
    run_stage_for_grid "3. Local testing" \
        "${PYTHON_BIN} test_model.py ${UPGD_SHARED_ARGS} -poison_rate=__RATE__ -eps=__EPS__" \
        "Local test UPGD model: ${DATASET} ${MODEL} rate=__RATE__ eps=__EPS__"
fi

if [ "$RUN_TRANSFER" = "1" ]; then
    if [ "$DATASET" = "cifar10" ]; then
        run_stage_for_grid "4. Transfer testing on STL-10" \
            "${PYTHON_BIN} test_stl10.py ${UPGD_SHARED_ARGS} -poison_rate=__RATE__ -eps=__EPS__" \
            "STL-10 transfer test: ${DATASET} ${MODEL} rate=__RATE__ eps=__EPS__"
    else
        if [ "$RUN_TINY_CORRUPTION" = "1" ]; then
            echo
            echo "----- 4a. Tiny-ImageNet-C transfer testing: ${DATASET} ${MODEL} -----"
            for rate in "${POISON_RATE_LIST[@]}"; do
                for eps in "${EPS_LIST[@]}"; do
                    for corruption_type in ${CORRUPTION_TYPES}; do
                        for severity in ${CORRUPTION_SEVERITIES}; do
                            run_command "${PYTHON_BIN} test_tiny_imagenet.py ${UPGD_SHARED_ARGS} -poison_rate=${rate} -eps=${eps} -corruption_type=${corruption_type} -severity=${severity}" \
                                "Tiny-ImageNet-C transfer test: ${MODEL} rate=${rate} eps=${eps} ${corruption_type} severity=${severity}"
                        done
                    done
                done
            done
        fi

        if [ "$RUN_TINY_TARGET_DOMAIN" = "1" ]; then
            run_stage_for_grid "4b. Tiny target-domain transfer testing" \
                "${PYTHON_BIN} test_tiny_target_domain.py ${UPGD_SHARED_ARGS} -source_dataset=tiny_imagenet -poison_rate=__RATE__ -eps=__EPS__ -target_domain_dir=${TARGET_DOMAIN_DIR}" \
                "Tiny target-domain transfer test: ${MODEL} rate=__RATE__ eps=__EPS__"
            run_stage_for_grid "4c. Tiny target-domain Qwen transfer testing" \
                "${PYTHON_BIN} test_tiny_target_domain_qwen.py ${UPGD_SHARED_ARGS} -source_dataset=tiny_imagenet -poison_rate=__RATE__ -eps=__EPS__ -target_domain_dir=${TARGET_DOMAIN_QWEN_DIR}" \
                "Tiny target-domain Qwen transfer test: ${MODEL} rate=__RATE__ eps=__EPS__"
        fi
    fi
fi

if [ "$RUN_DEFENSES" = "1" ]; then
    echo
    echo "----- 5. Defenses: ${DATASET} ${MODEL} -----"
    for defense in ${DEFENSES}; do
        run_stage_for_grid "Defense ${defense}" \
            "${PYTHON_BIN} other_defense.py -defense=${defense} ${UPGD_SHARED_ARGS} -poison_rate=__RATE__ -eps=__EPS__" \
            "Defense ${defense}: ${DATASET} ${MODEL} rate=__RATE__ eps=__EPS__"
    done
fi

echo
echo "=========================================="
echo "Done: ${DATASET} ${MODEL}"
echo "Log for failed commands: ${ERROR_LOG}"
echo "=========================================="
