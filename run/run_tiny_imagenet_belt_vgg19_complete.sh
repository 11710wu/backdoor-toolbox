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
echo "TINY_IMAGENET BELT Attack Experiment Script - Model: vgg19"
echo "=========================================="

# ==============================================================================
# Model: vgg19
# ==============================================================================

echo '----- 1. Creation (vgg19) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Create: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- 2. Training (vgg19) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Train: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- 3. Local Testing (vgg19) -----'
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Test: tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- 4. Cross Testing (vgg19) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Cross Test: test_tiny_imagenet.py belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- 5. Defenses (vgg19) -----'

echo '----- Defense: SentiNet (vgg19) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: SentiNet tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- Defense: STRIP (vgg19) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: STRIP tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- Defense: ScaleUp (vgg19) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: ScaleUp tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- Defense: IBD_PSC (vgg19) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: IBD_PSC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- Defense: NC (vgg19) -----'
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: NC tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"

echo '----- Defense: FeatureRE (vgg19) -----'
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.005 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.005 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.005 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.005 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.005 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.005 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.005 cover=0.5 mask=0.3 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.05 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.001 cover=0.5 mask=0.05 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.1 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.001 cover=0.5 mask=0.1 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.15 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.001 cover=0.5 mask=0.15 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.2 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.001 cover=0.5 mask=0.2 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.25 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.001 cover=0.5 mask=0.25 (vgg19)"
run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.001 -cover_rate=0.5 -mask_rate=0.3 -alpha=1.0 -model=vgg19" "Defense: FeatureRE tiny_imagenet belt rate=0.001 cover=0.5 mask=0.3 (vgg19)"
