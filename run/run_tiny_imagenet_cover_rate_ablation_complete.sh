#!/bin/bash
# Cover-rate 影响实验（tiny_imagenet）
# - adaptive_patch: alpha=0, poison_rate=0.005
# - adaptive_blend: alpha=0.15, poison_rate=0.005
# - WaNet: s=0.5, k=4, poison_rate=0.01
# cover_rate = poison_rate * {0, 0.2, 0.5, 1, 2, 5, 10}

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/data/tiny-target-domain-qwen-full-organized}"
cd "$PROJECT_ROOT" || exit 1

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
ERROR_LOG="$LOG_DIR/run_tiny_imagenet_cover_rate_ablation_complete.log"

MODELS=("resnet18" "mobilenetv2" "vgg19_bn")
COVER_MULTIPLIERS=("0" "0.2" "0.5" "1" "2" "5" "10")
DEFENSES=("SentiNet" "STRIP" "ScaleUp" "IBD_PSC")
DATASET="tiny_imagenet"

run_command() {
    local original_cmd="$1"
    local description="$2"
    local TMP_OUT
    TMP_OUT=$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")
    eval "$original_cmd" 2>&1 | tee "$TMP_OUT"
    local exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 失败 (退出码: $exit_code)" >> "$ERROR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 命令: $original_cmd" >> "$ERROR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 描述: $description" >> "$ERROR_LOG"
        cat "$TMP_OUT" >> "$ERROR_LOG" 2>/dev/null
        echo "----------------------------------------" >> "$ERROR_LOG"
    fi
    rm -f "$TMP_OUT"
    return "$exit_code"
}

calc_cover_rate() {
    local poison_rate="$1"
    local multiplier="$2"
    awk "BEGIN { printf \"%.6f\", $poison_rate * $multiplier }"
}

run_one_setting() {
    local model="$1"
    local poison_type="$2"
    local poison_rate="$3"
    local alpha="$4"
    local s="$5"
    local cover_rate="$6"

    local extra_args=""
    if [[ "$poison_type" == "adaptive_patch" || "$poison_type" == "adaptive_blend" ]]; then
        extra_args="$extra_args -alpha=$alpha"
    fi
    if [[ "$poison_type" == "WaNet" ]]; then
        extra_args="$extra_args -s=$s -k=4"
    fi

    local common_args="-dataset=${DATASET} -poison_type=${poison_type} -poison_rate=${poison_rate} -cover_rate=${cover_rate} -model=${model}${extra_args}"

    run_command "python create_poisoned_set.py ${common_args}" "Create: ${DATASET} ${poison_type} model=${model} cover=${cover_rate}"
    run_command "python train_on_poisoned_set.py ${common_args}" "Train: ${DATASET} ${poison_type} model=${model} cover=${cover_rate}"
    run_command "python test_model.py ${common_args}" "Test: ${DATASET} ${poison_type} model=${model} cover=${cover_rate}"
    run_command "python test_tiny_target_domain.py -source_dataset=tiny_imagenet ${common_args} -target_domain_dir=${TARGET_DOMAIN_DIR}" "Cross Test: test_tiny_target_domain.py ${DATASET} ${poison_type} model=${model} cover=${cover_rate}"

    for defense in "${DEFENSES[@]}"; do
        run_command "python other_defense.py -defense=${defense} ${common_args}" "Defense: ${defense} ${DATASET} ${poison_type} model=${model} cover=${cover_rate}"
    done
}

echo "=========================================="
echo "Tiny-ImageNet Cover-rate Ablation (adaptive_patch/adaptive_blend/WaNet)"
echo "=========================================="

if [ ! -d "$TARGET_DOMAIN_DIR" ] || [ ! -d "$TARGET_DOMAIN_DIR/images" ]; then
    echo "目标域目录不可用: $TARGET_DOMAIN_DIR" | tee -a "$ERROR_LOG"
    exit 1
fi

for model in "${MODELS[@]}"; do
    echo "----- Model: ${model} -----"

    # adaptive_patch: alpha=0, poison_rate=0.005
    for m in "${COVER_MULTIPLIERS[@]}"; do
        cover=$(calc_cover_rate "0.005" "$m")
        run_one_setting "$model" "adaptive_patch" "0.005" "0" "0.5" "$cover"
    done

    # adaptive_blend: alpha=0.15, poison_rate=0.005
    for m in "${COVER_MULTIPLIERS[@]}"; do
        cover=$(calc_cover_rate "0.005" "$m")
        run_one_setting "$model" "adaptive_blend" "0.005" "0.15" "0.5" "$cover"
    done

    # WaNet: s=0.5, poison_rate=0.01
    for m in "${COVER_MULTIPLIERS[@]}"; do
        cover=$(calc_cover_rate "0.01" "$m")
        run_one_setting "$model" "WaNet" "0.01" "0" "0.5" "$cover"
    done
done

echo "=========================================="
echo "Tiny-ImageNet cover-rate 消融实验完成"
echo "错误日志: $ERROR_LOG"
echo "=========================================="
