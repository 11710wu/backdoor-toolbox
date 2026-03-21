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
echo "TINY_IMAGENET Adaptive Patch Attack Experiment Script - Model: mobilenetv2"
echo "=========================================="

# ==============================================================================
# Model: mobilenetv2
# ==============================================================================

echo '----- 1. Creation (mobilenetv2) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Create: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

echo '----- 2. Training (mobilenetv2) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Train: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

echo '----- 3. Local Testing (mobilenetv2) -----'
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Test: tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

echo '----- 4. Cross Testing (mobilenetv2) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2) frost s=3"

echo '----- 5. Defenses (mobilenetv2) -----'

echo '----- Defense: SentiNet (mobilenetv2) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

echo '----- Defense: ScaleUp (mobilenetv2) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

echo '----- Defense: IBD_PSC (mobilenetv2) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

echo '----- Defense: NC (mobilenetv2) -----'
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: NC tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"

# echo '----- Defense: FeatureRE (mobilenetv2) -----'
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"
