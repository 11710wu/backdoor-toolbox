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
echo "CIFAR10 Adaptive Patch Attack Experiment Script - Model: vgg19"
echo "=========================================="

# ==============================================================================
# Model: vgg19
# ==============================================================================

echo '----- 1. Creation (vgg19) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Create: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- 2. Training (vgg19) -----'
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Train: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- 3. Local Testing (vgg19) -----'
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Test: cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- 4. Cross Testing (vgg19) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Cross Test: test_stl10.py adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- 5. Defenses (vgg19) -----'

echo '----- Defense: SentiNet (vgg19) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Defense: SentiNet cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- Defense: STRIP (vgg19) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- Defense: ScaleUp (vgg19) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Defense: ScaleUp cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- Defense: IBD_PSC (vgg19) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Defense: IBD_PSC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

echo '----- Defense: NC (vgg19) -----'
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Defense: NC cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"

# echo '----- Defense: FeatureRE (vgg19) -----'
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (vgg19)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=vgg19_bn" "Defense: FeatureRE cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (vgg19)"
