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
echo "TINY_IMAGENET SIG Attack Experiment Script - Model: vgg19"
echo "=========================================="

# ==============================================================================
# Model: vgg19
# ==============================================================================

echo '----- 1. Creation (vgg19) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Create: tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

echo '----- 2. Training (vgg19) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Train: tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

echo '----- 3. Local Testing (vgg19) -----'
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Test: tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

echo '----- 4. Cross Testing (vgg19) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=4 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=4 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=12 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=12 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=20 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=20 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=28 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=28 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=36 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=36 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=44 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=44 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=56 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.005 f=6 delta=56 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=4 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=4 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=12 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=12 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=20 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=20 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=28 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=28 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=36 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=36 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=44 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=44 (vgg19) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=56 (vgg19) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py SIG rate=0.001 f=6 delta=56 (vgg19) frost s=3"

echo '----- 5. Defenses (vgg19) -----'

echo '----- Defense: SentiNet (vgg19) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Defense: SentiNet tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

echo '----- Defense: STRIP (vgg19) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Defense: STRIP tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

echo '----- Defense: ScaleUp (vgg19) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Defense: ScaleUp tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

echo '----- Defense: IBD_PSC (vgg19) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Defense: IBD_PSC tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

echo '----- Defense: NC (vgg19) -----'
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=4 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.001 f=6 delta=4 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=12 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.001 f=6 delta=12 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.001 f=6 delta=20 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=28 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.001 f=6 delta=28 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=36 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.001 f=6 delta=36 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=44 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.001 f=6 delta=44 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.001 -f=6 -delta=56 -model=vgg19_bn" "Defense: NC tiny_imagenet SIG rate=0.001 f=6 delta=56 (vgg19)"

# echo '----- Defense: FeatureRE (vgg19) -----'
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.05 -f=6 -delta=4 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.05 f=6 delta=4 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.05 -f=6 -delta=12 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.05 f=6 delta=12 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.05 -f=6 -delta=20 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.05 f=6 delta=20 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.05 -f=6 -delta=28 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.05 f=6 delta=28 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.05 -f=6 -delta=36 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.05 f=6 delta=36 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.05 -f=6 -delta=44 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.05 f=6 delta=44 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.05 -f=6 -delta=56 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.05 f=6 delta=56 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.01 -f=6 -delta=4 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.01 f=6 delta=4 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.01 -f=6 -delta=12 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.01 f=6 delta=12 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.01 f=6 delta=20 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.01 -f=6 -delta=28 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.01 f=6 delta=28 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.01 -f=6 -delta=36 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.01 f=6 delta=36 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.01 -f=6 -delta=44 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.01 f=6 delta=44 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.01 -f=6 -delta=56 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.01 f=6 delta=56 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=4 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.005 f=6 delta=4 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=12 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.005 f=6 delta=12 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.005 f=6 delta=20 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=28 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.005 f=6 delta=28 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=36 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.005 f=6 delta=36 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=44 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.005 f=6 delta=44 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.005 -f=6 -delta=56 -model=vgg19_bn" "Defense: FeatureRE tiny_imagenet SIG rate=0.005 f=6 delta=56 (vgg19)"
