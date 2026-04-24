#!/bin/bash

# 重新完整跑一遍缺失的 tiny_imagenet / mobilenetv2 / belt 配置：
# poison_rate=0.1, cover_rate=0.5, mask_rate=0.2, alpha=0.1
# 该脚本按照 run/ 目录现有 complete 脚本风格编写，包含：
# 1) poisoned set creation
# 2) training
# 3) local testing
# 4) tiny-imagenet-c cross testing
# 5) target-domain transfer testing
# 6) four defenses: SentiNet / STRIP / ScaleUp / IBD_PSC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/data/tiny-target-domain-qwen-full-organized}"

cd "$PROJECT_ROOT" || exit 1

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
ERROR_LOG="$LOG_DIR/run_tiny_imagenet_belt_mobilenetv2_rate0p1_alpha0p1_rerun.log"

set +e

run_command() {
    local original_cmd="$1"
    local description="$2"
    local TMP_OUT
    TMP_OUT=$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")

    eval "$original_cmd" 2>&1 | tee "$TMP_OUT"
    local exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 命令执行失败 (退出码: $exit_code)" >> "$ERROR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 命令: $original_cmd" >> "$ERROR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 描述: $description" >> "$ERROR_LOG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- 命令输出 (stdout+stderr) ---" >> "$ERROR_LOG"
        cat "$TMP_OUT" >> "$ERROR_LOG" 2>/dev/null
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ---" >> "$ERROR_LOG"
    fi

    rm -f "$TMP_OUT"
    return "$exit_code"
}

echo "=========================================="
echo "Rerun Missing Tiny-ImageNet BELT Config - Model: mobilenetv2"
echo "poison_rate=0.1, cover_rate=0.5, mask_rate=0.2, alpha=0.1"
echo "=========================================="

echo '----- 0. Check Target Domain Dataset -----'
if [ ! -d "$TARGET_DOMAIN_DIR" ]; then
    echo "目标域目录不存在: $TARGET_DOMAIN_DIR" | tee -a "$ERROR_LOG"
    exit 1
fi

if [ ! -d "$TARGET_DOMAIN_DIR/images" ]; then
    echo "目标域目录缺少 images/ 子目录: $TARGET_DOMAIN_DIR/images" | tee -a "$ERROR_LOG"
    exit 1
fi

echo "Using target domain dataset: $TARGET_DOMAIN_DIR"

echo '----- 1. Creation -----'
run_command \
    "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2" \
    "Create: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo '----- 2. Training -----'
run_command \
    "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2" \
    "Train: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo '----- 3. Local Testing -----'
run_command \
    "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2" \
    "Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo '----- 4. Cross Testing -----'
run_command \
    "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=2" \
    "Cross Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2) frost s=2"
run_command \
    "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=3" \
    "Cross Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2) frost s=3"

echo '----- 5. Target-Domain Transfer Testing -----'
run_command \
    "python test_tiny_target_domain.py -source_dataset=tiny_imagenet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2 -target_domain_dir=${TARGET_DOMAIN_DIR}" \
    "Transfer: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo '----- 6. Defenses -----'

echo '----- Defense: SentiNet -----'
run_command \
    "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2" \
    "Defense: SentiNet tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo '----- Defense: STRIP -----'
run_command \
    "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2" \
    "Defense: STRIP tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo '----- Defense: ScaleUp -----'
run_command \
    "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2" \
    "Defense: ScaleUp tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo '----- Defense: IBD_PSC -----'
run_command \
    "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=mobilenetv2" \
    "Defense: IBD_PSC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (mobilenetv2)"

echo "=========================================="
echo "Rerun finished."
echo "If any command failed, check: $ERROR_LOG"
echo "=========================================="
