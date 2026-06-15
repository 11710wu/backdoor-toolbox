#!/usr/bin/env bash

# Supplement MNIST-M UPGD baseline results.
# Default grid:
#   models: resnet18, mobilenetv2, vgg19_bn
#   poison rates: 0.005, 0.01, 0.05
#   eps: 4, 6, 8, 10, 12, 16, 20, 24
#
# The script is resumable by default. It skips a step if the expected output
# file already exists. Use FORCE=1 to rerun every step.

set +e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

DEVICES="${DEVICES:-0}"
POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set}"
MODELS="${MODELS:-resnet18 mobilenetv2 vgg19_bn}"
POISON_RATES="${POISON_RATES:-0.005 0.01 0.05}"
EPS_LIST="${EPS_LIST:-4 6 8 10 12 16 20 24}"
DEFENSES="${DEFENSES:-SentiNet STRIP ScaleUp IBD_PSC}"
FORCE="${FORCE:-0}"

export POISONED_TRAIN_SET_ROOT

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ERROR_LOG="$LOG_DIR/run_mnistm_upgd_baseline_supplement_${TIMESTAMP}.log"

run_command() {
    local cmd="$1"
    local description="$2"
    local marker="${3:-}"
    local tmp_out

    if [[ -n "$marker" && "$FORCE" != "1" && -e "$marker" ]]; then
        echo "[SKIP] $description"
        echo "       exists: $marker"
        return 0
    fi

    tmp_out="$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")"
    echo "[RUN ] $description"
    echo "       $cmd"
    eval "$cmd" 2>&1 | tee "$tmp_out"
    local exit_code=${PIPESTATUS[0]}

    if [[ "$exit_code" -ne 0 ]]; then
        {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] command failed, exit code: $exit_code"
            echo "command: $cmd"
            echo "description: $description"
            echo "--- stdout+stderr ---"
            cat "$tmp_out" 2>/dev/null
            echo "---"
        } >> "$ERROR_LOG"
    fi

    rm -f "$tmp_out"
    return "$exit_code"
}

arch_name() {
    case "$1" in
        resnet18) echo "ResNet18_mnistm" ;;
        mobilenetv2) echo "mobilenetv2_mnistm" ;;
        vgg19_bn) echo "vgg19_bn_mnistm" ;;
        *)
            echo "Unsupported model: $1" >&2
            return 1
            ;;
    esac
}

rate_fmt() {
    python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.3f}")
PY
}

eps_fmt() {
    python - "$1" <<'PY'
import sys
print(str(float(sys.argv[1])))
PY
}

clean_model_path() {
    local arch="$1"
    echo "${POISONED_TRAIN_SET_ROOT}/mnistm/none_0.000_poison_seed=2333_arch=${arch}/${arch}.pt"
}

poison_dir() {
    local rate="$1"
    local eps="$2"
    local arch="$3"
    local r
    local e
    r="$(rate_fmt "$rate")"
    e="$(eps_fmt "$eps")"
    echo "${POISONED_TRAIN_SET_ROOT}/mnistm/upgd_${r}_eps=${e}_constraint=Linf_steps=100_mode=clean_mult=5_poison_seed=2333_arch=${arch}"
}

defense_marker() {
    case "$1" in
        SentiNet) echo "sentinet_defense_results.json" ;;
        STRIP) echo "strip_defense_results.json" ;;
        ScaleUp) echo "scaleup_defense_results.json" ;;
        IBD_PSC) echo "ibd_psc_defense_results.json" ;;
        *)
            echo "Unsupported defense: $1" >&2
            return 1
            ;;
    esac
}

echo "=========================================="
echo "MNIST-M UPGD baseline supplement"
echo "root: ${POISONED_TRAIN_SET_ROOT}"
echo "devices: ${DEVICES}"
echo "models: ${MODELS}"
echo "rates: ${POISON_RATES}"
echo "eps: ${EPS_LIST}"
echo "force: ${FORCE}"
echo "error log: ${ERROR_LOG}"
echo "=========================================="

for model in $MODELS; do
    arch="$(arch_name "$model")" || exit 1
    clean_path="$(clean_model_path "$arch")"

    echo
    echo "----- 0. Clean base preparation: ${model} (${arch}) -----"
    run_command \
        "python create_poisoned_set.py -dataset=mnistm -poison_type=none -poison_rate=0.0 -model=${model} -devices=${DEVICES}" \
        "Create clean MNIST-M set for ${model}" \
        "${POISONED_TRAIN_SET_ROOT}/mnistm/none_0.000_poison_seed=2333_arch=${arch}/labels"

    run_command \
        "python train_on_poisoned_set.py -dataset=mnistm -poison_type=none -poison_rate=0.0 -model=${model} -devices=${DEVICES}" \
        "Train clean MNIST-M base model for ${model}" \
        "$clean_path"

    if [[ ! -e "$clean_path" ]]; then
        echo "[ERROR] Missing clean base model for ${model}: ${clean_path}" | tee -a "$ERROR_LOG"
        echo "        Skip UPGD generation for this model."
        continue
    fi

    for rate in $POISON_RATES; do
        for eps in $EPS_LIST; do
            dir="$(poison_dir "$rate" "$eps" "$arch")"
            common_args="-dataset=mnistm -poison_type=upgd -poison_rate=${rate} -eps=${eps} -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -model=${model} -devices=${DEVICES}"

            echo
            echo "----- UPGD ${model}: rate=${rate}, eps=${eps} -----"

            run_command \
                "python create_poisoned_set.py ${common_args} -upgd_model_path=${clean_path}" \
                "Create MNIST-M UPGD set (${model}, rate=${rate}, eps=${eps})" \
                "${dir}/labels"

            run_command \
                "python train_on_poisoned_set.py ${common_args}" \
                "Train MNIST-M UPGD model (${model}, rate=${rate}, eps=${eps})" \
                "${dir}/train_results_seed=2333.json"

            run_command \
                "python test_model.py ${common_args}" \
                "Source test MNIST-M UPGD (${model}, rate=${rate}, eps=${eps})" \
                "${dir}/test_results_seed=2333.json"

            run_command \
                "python test_mnist.py ${common_args}" \
                "Transfer test MNIST-M -> MNIST UPGD (${model}, rate=${rate}, eps=${eps})" \
                "${dir}/test_mnist_cross_results.txt"

            for defense in $DEFENSES; do
                marker="$(defense_marker "$defense")" || exit 1
                run_command \
                    "python other_defense.py -defense=${defense} ${common_args}" \
                    "Defense ${defense} MNIST-M UPGD (${model}, rate=${rate}, eps=${eps})" \
                    "${dir}/${marker}"
            done
        done
    done
done

echo
echo "=========================================="
echo "MNIST-M UPGD supplement finished."
echo "Check failures in: ${ERROR_LOG}"
echo "=========================================="
