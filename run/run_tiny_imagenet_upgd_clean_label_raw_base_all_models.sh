#!/usr/bin/env bash
# Rerun Tiny-ImageNet UPGD clean-label experiments for the three baseline models.
# Tiny-ImageNet intentionally excludes NC by default because it is expensive here.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_upgd_clean_label_raw_base.sh"

MODELS="${MODELS:-resnet18 mobilenetv2 vgg19_bn}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"
LABEL_MODE="${LABEL_MODE:-clean}"
# Tiny UPGD transfer defaults to the two target-domain suites, not Tiny-C.
RUN_TINY_CORRUPTION="${RUN_TINY_CORRUPTION:-0}"
RUN_TINY_TARGET_DOMAIN="${RUN_TINY_TARGET_DOMAIN:-1}"

echo "=========================================="
echo "Tiny-ImageNet UPGD raw-base all-model rerun"
echo "models=${MODELS}"
echo "label_mode=${LABEL_MODE}"
echo "defenses=${DEFENSES}"
echo "run_tiny_corruption=${RUN_TINY_CORRUPTION}"
echo "run_tiny_target_domain=${RUN_TINY_TARGET_DOMAIN}"
echo "=========================================="

overall_status=0
for model in ${MODELS}; do
    echo
    echo "========== Tiny-ImageNet / ${model} =========="
    DATASET="tiny_imagenet" \
    MODEL="${model}" \
    LABEL_MODE="${LABEL_MODE}" \
    DEFENSES="${DEFENSES}" \
    RUN_TINY_CORRUPTION="${RUN_TINY_CORRUPTION}" \
    RUN_TINY_TARGET_DOMAIN="${RUN_TINY_TARGET_DOMAIN}" \
    bash "${RUNNER}"
    status=$?

    if [ "${status}" -ne 0 ]; then
        overall_status="${status}"
        echo "[ERROR] Tiny-ImageNet ${model} failed with exit=${status}"
        if [ "${STOP_ON_FAIL:-0}" = "1" ]; then
            exit "${status}"
        fi
    fi
done

exit "${overall_status}"
