#!/bin/bash

# Backfill stealth/detection defenses for CIFAR-10 WaNet same-warp cover ablation.
#
# Reference pipelines:
#   - run/run_cifar10_cover_rate_ablation_complete.sh   (old noise-cover + defenses)
#   - run/run_cifar10_wanet_same_cover_resnet18_cover_ablation.sh
#     (branch wanet-same-cover-cifar10-resnet18; create/train/test only)
#
# This script ONLY runs defenses against already-trained models under
# poisoned_train_set5. It does not recreate or retrain.
#
# Stealthiness uses: SentiNet, STRIP, ScaleUp, IBD_PSC
#
# Usage:
#   cd /workspace/backdoor-toolbox-new1
#   DEVICES=0 bash run/backfill_cifar10_wanet_same_cover_set5_defenses.sh
#
# Optional env:
#   DEVICES=0
#   PYTHON_BIN=python
#   POISONED_TRAIN_SET_ROOT=poisoned_train_set5
#   POISON_RATES="0.05"            # default: 0.005 0.01 0.05
#   COVER_RATES="0.0 0.002 ..."    # default: full same-warp grid
#   DEFENSES="SentiNet STRIP ScaleUp IBD_PSC"
#   SKIP_EXISTING=1                # skip if defense json already exists
#   DRY_RUN=1

set +e

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="${ERROR_LOG:-${LOG_DIR}/backfill_cifar10_wanet_same_cover_set5_defenses_${TIMESTAMP}.log}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="cifar10"
POISON_TYPE="WaNet"
MODEL="resnet18"
S_VALUE="0.5"
K_VALUE="4"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set5}"

read -r -a POISON_RATES <<< "${POISON_RATES:-0.005 0.01 0.05}"
read -r -a COVER_RATES <<< "${COVER_RATES:-0.000000 0.002000 0.005000 0.010000 0.020000 0.050000 0.100000}"
read -r -a DEFENSES <<< "${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [ -n "${DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVICES"
fi

declare -A DEFENSE_JSON=(
    [SentiNet]="sentinet_defense_results.json"
    [STRIP]="strip_defense_results.json"
    [ScaleUp]="scaleup_defense_results.json"
    [IBD_PSC]="ibd_psc_defense_results.json"
)

fmt_ratio() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.3f}")
PY
}

fmt_s() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
print(("%g" % float(sys.argv[1])))
PY
}

poison_dir() {
    local rate="$1"
    local cover="$2"
    local ratio cover3 s_param
    ratio="$(fmt_ratio "$rate")"
    cover3="$(fmt_ratio "$cover")"
    s_param="$(fmt_s "$S_VALUE")"
    echo "${POISONED_TRAIN_SET_ROOT}/${DATASET}/${POISON_TYPE}_${ratio}_cover=${cover3}_s=${s_param}_k=${K_VALUE}_poison_seed=2333_arch=ResNet18_cifar10"
}

model_path() {
    echo "$(poison_dir "$1" "$2")/ResNet18_cifar10.pt"
}

defense_json_path() {
    local defense="$1"
    local rate="$2"
    local cover="$3"
    local fname="${DEFENSE_JSON[$defense]:-}"
    if [ -z "$fname" ]; then
        echo ""
        return
    fi
    echo "$(poison_dir "$rate" "$cover")/${fname}"
}

run_command() {
    local original_cmd="$1"
    local description="$2"
    local tmp_out
    tmp_out=$(mktemp 2>/dev/null || echo "/tmp/backfill_cmd_$$_${RANDOM}.out")

    echo
    echo ">>> ${description}"
    echo "${original_cmd}"

    if [ "$DRY_RUN" = "1" ]; then
        rm -f "$tmp_out"
        return 0
    fi

    eval "$original_cmd" 2>&1 | tee "$tmp_out"
    local exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] failed (exit code: $exit_code)"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] command: $original_cmd"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] description: $description"
            cat "$tmp_out"
            echo "----------------------------------------"
        } >> "$ERROR_LOG"
    fi
    rm -f "$tmp_out"
    return "$exit_code"
}

TOTAL=$((${#POISON_RATES[@]} * ${#COVER_RATES[@]} * ${#DEFENSES[@]}))
DONE=0
SKIPPED=0
MISSING_MODEL=0
FAILED=0

echo "=========================================="
echo "Backfill: CIFAR-10 WaNet same-warp defenses"
echo "root         : ${POISONED_TRAIN_SET_ROOT}"
echo "model        : ${MODEL}"
echo "s/k          : ${S_VALUE}/${K_VALUE}"
echo "poison_rates : ${POISON_RATES[*]}"
echo "cover_rates  : ${COVER_RATES[*]}"
echo "defenses     : ${DEFENSES[*]}"
echo "skip_existing: ${SKIP_EXISTING}"
echo "dry_run      : ${DRY_RUN}"
echo "total jobs   : ${TOTAL}"
echo "error log    : ${ERROR_LOG}"
echo "=========================================="

# Quick inventory
echo
echo "----- Inventory -----"
for POISON_RATE in "${POISON_RATES[@]}"; do
    for COVER_RATE in "${COVER_RATES[@]}"; do
        mp="$(model_path "$POISON_RATE" "$COVER_RATE")"
        if [ -f "$mp" ]; then
            echo "[ok]   poison=${POISON_RATE} cover=${COVER_RATE}"
        else
            echo "[miss] poison=${POISON_RATE} cover=${COVER_RATE} -> ${mp}"
            MISSING_MODEL=$((MISSING_MODEL + 1))
        fi
    done
done

echo
echo "----- Defenses -----"
for DEFENSE in "${DEFENSES[@]}"; do
    echo
    echo "===== Defense: ${DEFENSE} ====="
    for POISON_RATE in "${POISON_RATES[@]}"; do
        for COVER_RATE in "${COVER_RATES[@]}"; do
            MP="$(model_path "$POISON_RATE" "$COVER_RATE")"
            JP="$(defense_json_path "$DEFENSE" "$POISON_RATE" "$COVER_RATE")"

            if [ ! -f "$MP" ]; then
                echo "[skip-missing-model] ${DEFENSE} poison=${POISON_RATE} cover=${COVER_RATE}"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            if [ "$SKIP_EXISTING" = "1" ] && [ -n "$JP" ] && [ -f "$JP" ]; then
                echo "[skip-existing] ${DEFENSE} poison=${POISON_RATE} cover=${COVER_RATE}"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            CMD="${PYTHON_BIN} other_defense.py -defense=${DEFENSE} -dataset=${DATASET} -poison_type=${POISON_TYPE} -poison_rate=${POISON_RATE} -cover_rate=${COVER_RATE} -s=${S_VALUE} -k=${K_VALUE} -model=${MODEL}"
            DESC="Defense ${DEFENSE}: ${DATASET} ${POISON_TYPE} ${MODEL} poison=${POISON_RATE} cover=${COVER_RATE} s=${S_VALUE}"
            if run_command "$CMD" "$DESC"; then
                DONE=$((DONE + 1))
            else
                FAILED=$((FAILED + 1))
            fi
        done
    done
done

echo
echo "=========================================="
echo "Finished."
echo "ran_ok=${DONE} skipped=${SKIPPED} failed=${FAILED} missing_model_cfgs=${MISSING_MODEL}"
echo "Errors (if any): ${ERROR_LOG}"
echo "=========================================="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
exit 0
