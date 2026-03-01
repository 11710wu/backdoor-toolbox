#!/bin/bash

# 设置错误日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# User requested specific log name
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
echo "CIFAR10 Blend Attack Experiment Script - Model: resnet18"
echo "=========================================="

# ==============================================================================
# Model: resnet18
# ==============================================================================

echo '----- 1. Creation (resnet18) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Create: cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Create: cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Create: cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Create: cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Create: cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Create: cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Create: cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Create: cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Create: cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Create: cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Create: cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Create: cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Create: cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Create: cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Create: cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Create: cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Create: cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Create: cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Create: cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Create: cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Create: cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- 2. Training (resnet18) -----'
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Train: cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Train: cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Train: cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Train: cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Train: cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Train: cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Train: cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Train: cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Train: cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Train: cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Train: cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Train: cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Train: cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Train: cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Train: cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Train: cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Train: cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Train: cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Train: cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Train: cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Train: cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- 3. Local Testing (resnet18) -----'
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Test: cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Test: cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Test: cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Test: cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Test: cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Test: cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Test: cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Test: cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Test: cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Test: cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Test: cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Test: cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Test: cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Test: cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Test: cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Test: cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Test: cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Test: cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Test: cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Test: cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Test: cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- 4. Cross Testing (resnet18) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Cross Test: test_stl10.py blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- 5. Defenses (resnet18) -----'

echo '----- Defense: SentiNet (resnet18) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: SentiNet cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- Defense: STRIP (resnet18) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- Defense: ScaleUp (resnet18) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: ScaleUp cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- Defense: IBD_PSC (resnet18) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: IBD_PSC cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- Defense: NC (resnet18) -----'
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Defense: NC cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Defense: NC cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Defense: NC cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Defense: NC cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: NC cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Defense: NC cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: NC cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Defense: NC cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Defense: NC cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: NC cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Defense: NC cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: NC cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Defense: NC cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: NC cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Defense: NC cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Defense: NC cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Defense: NC cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Defense: NC cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: NC cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Defense: NC cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: NC cifar10 blend rate=0.005 alpha=0.3 (resnet18)"

echo '----- Defense: FeatureRE (resnet18) -----'
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.05 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.05 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.05 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.05 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.05 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.01 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.01 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.01 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.01 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.005 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.005 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.005 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.005 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.005 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: FeatureRE cifar10 blend rate=0.005 alpha=0.3 (resnet18)"
