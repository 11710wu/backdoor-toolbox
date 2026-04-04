#!/bin/bash

# 设置错误日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/run_tiny_imagenet_belt_resnet18_part2.log"

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
echo "本脚本为 2/2 部分：rate=0.02（alpha 0.8~1.0） + rate=0.01（全 alpha）"
echo "TINY_IMAGENET BELT Attack Experiment Script - Model: resnet18"
echo "mask_rate=0.2 fixed, alpha in {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}"
echo "=========================================="
echo "本脚本为 2/2 部分：rate=0.02（alpha 0.8~1.0） + rate=0.01（全 alpha）"

# ==============================================================================
# Model: resnet18
# ==============================================================================

echo '----- 1. Creation (resnet18) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Create: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Create: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

echo '----- 2. Training (resnet18) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Train: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Train: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

echo '----- 3. Local Testing (resnet18) -----'
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Test: tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Test: tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

echo '----- 4. Cross Testing (resnet18) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18) frost s=3"

echo '----- 5. Defenses (resnet18) -----'
echo '----- Defense: SentiNet (resnet18) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: SentiNet tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

echo '----- Defense: STRIP (resnet18) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: STRIP tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

echo '----- Defense: ScaleUp (resnet18) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: ScaleUp tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

echo '----- Defense: IBD_PSC (resnet18) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: IBD_PSC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

echo '----- Defense: NC (resnet18) -----'
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: NC tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"

# echo '----- Defense: FeatureRE (resnet18) -----'
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.1 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.02 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.02 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.3 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.3 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.4 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.4 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.5 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.5 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.6 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.6 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.7 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.7 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.8 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.8 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=0.9 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=0.9 (resnet18)"
# run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.01 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=resnet18" "Defense: FeatureRE tiny_imagenet belt rate=0.01 cover=0.5 mask=0.2 alpha=1.0 (resnet18)"
