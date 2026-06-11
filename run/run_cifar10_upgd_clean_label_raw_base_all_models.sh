#!/usr/bin/env bash
# Rerun CIFAR-10 UPGD clean-label experiments for the three baseline models.
# CIFAR-10 keeps NC in the default defense list.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_upgd_clean_label_raw_base.sh"

MODELS="${MODELS:-resnet18 mobilenetv2 vgg19_bn}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC NC}"
LABEL_MODE="${LABEL_MODE:-clean}"

echo "=========================================="
echo "CIFAR-10 UPGD raw-base all-model rerun"
echo "models=${MODELS}"
echo "label_mode=${LABEL_MODE}"
echo "defenses=${DEFENSES}"
echo "=========================================="

overall_status=0
for model in ${MODELS}; do
    echo
    echo "========== CIFAR-10 / ${model} =========="
    DATASET="cifar10" \
    MODEL="${model}" \
    LABEL_MODE="${LABEL_MODE}" \
    DEFENSES="${DEFENSES}" \
    bash "${RUNNER}"
    status=$?

    if [ "${status}" -ne 0 ]; then
        overall_status="${status}"
        echo "[ERROR] CIFAR-10 ${model} failed with exit=${status}"
        if [ "${STOP_ON_FAIL:-0}" = "1" ]; then
            exit "${status}"
        fi
    fi
done

exit "${overall_status}"
