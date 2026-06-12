#!/usr/bin/env bash
# Supplement explicit clean-label UPGD experiments for the clean-vs-dirty study.
# Results are written to poisoned_train_set2 by default.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_upgd_clean_label_raw_base.sh"

POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set2}"
export POISONED_TRAIN_SET_ROOT

LABEL_MODE="clean"
DATASETS="${DATASETS:-cifar10 tiny_imagenet mnistm}"
CIFAR10_MODELS="${CIFAR10_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
TINY_IMAGENET_MODELS="${TINY_IMAGENET_MODELS:-resnet18 mobilenetv2 vgg19_bn}"
MNISTM_MODELS="${MNISTM_MODELS:-resnet18 mobilenetv2 vgg19_bn}"

DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC NC}"
RUN_TINY_CORRUPTION="${RUN_TINY_CORRUPTION:-0}"
RUN_TINY_TARGET_DOMAIN="${RUN_TINY_TARGET_DOMAIN:-1}"

models_for_dataset() {
    case "$1" in
        "cifar10") echo "$CIFAR10_MODELS" ;;
        "tiny_imagenet") echo "$TINY_IMAGENET_MODELS" ;;
        "mnistm") echo "$MNISTM_MODELS" ;;
        *)
            echo "Unsupported dataset: $1" >&2
            return 1
            ;;
    esac
}

rates_for_dataset() {
    case "$1" in
        "cifar10") echo "${CIFAR10_POISON_RATES:-0.005 0.01 0.05}" ;;
        "tiny_imagenet") echo "${TINY_IMAGENET_POISON_RATES:-0.001 0.005}" ;;
        "mnistm") echo "${MNISTM_POISON_RATES:-0.005 0.01 0.05}" ;;
        *)
            echo "Unsupported dataset: $1" >&2
            return 1
            ;;
    esac
}

echo "=========================================="
echo "UPGD clean-label raw-base supplement"
echo "output_root=${POISONED_TRAIN_SET_ROOT}"
echo "output_note=each poisoned-set dir under this root stores poisoned data, model checkpoints, test/transfer files, and defense outputs"
echo "datasets=${DATASETS}"
echo "label_mode=${LABEL_MODE}"
echo "defenses=${DEFENSES}"
echo "=========================================="

overall_status=0
for dataset in ${DATASETS}; do
    for model in $(models_for_dataset "$dataset"); do
        echo
        echo "========== ${dataset} / ${model} =========="
        DATASET="${dataset}" \
        MODEL="${model}" \
        LABEL_MODE="${LABEL_MODE}" \
        POISON_RATES="$(rates_for_dataset "$dataset")" \
        DEFENSES="${DEFENSES}" \
        RUN_TINY_CORRUPTION="${RUN_TINY_CORRUPTION}" \
        RUN_TINY_TARGET_DOMAIN="${RUN_TINY_TARGET_DOMAIN}" \
        bash "${RUNNER}"
        status=$?

        if [ "${status}" -ne 0 ]; then
            overall_status="${status}"
            echo "[ERROR] ${dataset} ${model} failed with exit=${status}"
            if [ "${STOP_ON_FAIL:-0}" = "1" ]; then
                exit "${status}"
            fi
        fi
    done
done

exit "${overall_status}"
