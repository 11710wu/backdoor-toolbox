#!/bin/bash

# 设置错误日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/run_defenses_mobilenet_resnet_explicit.log"

# 执行命令并记录失败的命令
set +e

run_command() {
    local original_cmd="$1"
    local description="$2"
    local run_in_background=false
    local cmd="$original_cmd"
    local TMP_OUT
    TMP_OUT=$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${RANDOM}.out")

    if [[ "$cmd" == *" &" ]]; then
        run_in_background=true
        cmd="${cmd% &}"
    fi

    if [ "$run_in_background" = true ]; then
        eval "$cmd" > "$TMP_OUT" 2>&1 &
        local pid=$!
        wait $pid
        local exit_code=$?
    else
        eval "$cmd" 2>&1 | tee "$TMP_OUT"
        local exit_code=${PIPESTATUS[0]}
    fi

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
echo "TINY_IMAGENET BELT Attack Experiment Script - Model: vgg19_bn"
echo "mask_rate=0.2 fixed, alpha in {0.1, 0.15, 0.2, 0.25, 0.3, 0.35} (centered on 0.2)"
echo "=========================================="

# ==============================================================================
# Model: vgg19_bn
# ==============================================================================

echo '----- 1. Creation (vgg19_bn) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

echo '----- 2. Training (vgg19_bn) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

echo '----- 3. Local Testing (vgg19_bn) -----'
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

echo '----- 4. Cross Testing (vgg19_bn) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn) frost s=3"

echo '----- 5. Defenses (vgg19_bn) -----'
echo '----- Defense: SentiNet (vgg19_bn) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

echo '----- Defense: STRIP (vgg19_bn) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

echo '----- Defense: ScaleUp (vgg19_bn) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

echo '----- Defense: IBD_PSC (vgg19_bn) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

echo '----- Defense: NC (vgg19_bn) -----'
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"

# echo '----- Defense: FeatureRE (vgg19_bn) -----'
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.1 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.1 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.15 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.15 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.2 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.2 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.25 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.25 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (vgg19_bn)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.35 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.35 (vgg19_bn)"
