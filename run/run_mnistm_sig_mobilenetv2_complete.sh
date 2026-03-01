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
echo "MNISTM SIG Attack Experiment Script - Model: mobilenetv2"
echo "=========================================="

# ==============================================================================
# Model: mobilenetv2
# ==============================================================================

echo '----- 1. Creation (mobilenetv2) -----'
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Create: mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Create: mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Create: mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Create: mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Create: mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Create: mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Create: mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Create: mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Create: mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Create: mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Create: mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Create: mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Create: mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Create: mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Create: mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Create: mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Create: mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Create: mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Create: mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Create: mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Create: mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- 2. Training (mobilenetv2) -----'
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Train: mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Train: mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Train: mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Train: mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Train: mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Train: mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Train: mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Train: mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Train: mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Train: mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Train: mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Train: mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Train: mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Train: mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Train: mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Train: mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Train: mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Train: mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Train: mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Train: mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Train: mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- 3. Local Testing (mobilenetv2) -----'
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Test: mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Test: mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Test: mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Test: mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Test: mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Test: mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Test: mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Test: mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Test: mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Test: mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Test: mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Test: mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Test: mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Test: mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Test: mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Test: mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Test: mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Test: mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Test: mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Test: mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Test: mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- 4. Cross Testing (mobilenetv2) -----'
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Cross Test: test_mnist.py SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- 5. Defenses (mobilenetv2) -----'

echo '----- Defense: SentiNet (mobilenetv2) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Defense: SentiNet mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Defense: STRIP mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- Defense: ScaleUp (mobilenetv2) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Defense: ScaleUp mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- Defense: IBD_PSC (mobilenetv2) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Defense: IBD_PSC mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- Defense: NC (mobilenetv2) -----'
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Defense: NC mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"

echo '----- Defense: FeatureRE (mobilenetv2) -----'
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.05 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.05 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.05 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.05 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.05 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.05 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.05 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.01 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.01 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.01 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.01 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.01 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.01 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.01 f=6 delta=56 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.005 f=6 delta=4 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.005 f=6 delta=12 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.005 f=6 delta=20 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.005 f=6 delta=28 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.005 f=6 delta=36 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.005 f=6 delta=44 (mobilenetv2)"
run_command "python other_defense.py -defense=FeatureRE -dataset=mnistm -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=mobilenetv2" "Defense: FeatureRE mnistm SIG rate=0.005 f=6 delta=56 (mobilenetv2)"
