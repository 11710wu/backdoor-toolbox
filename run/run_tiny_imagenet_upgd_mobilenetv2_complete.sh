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
echo "TINY_IMAGENET UPGD Attack Experiment Script - Model: mobilenetv2"
echo "=========================================="

# ==============================================================================
# Model: mobilenetv2
# ==============================================================================

echo '----- 0. Clean Model Preparation (mobilenetv2) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "Create Clean: tiny_imagenet mobilenetv2"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "Train Clean: tiny_imagenet mobilenetv2"

echo '----- 1. Creation (mobilenetv2) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt -model=mobilenetv2" "Create: tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- 2. Training (mobilenetv2) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- 3. Local Testing (mobilenetv2) -----'
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- 4. Cross Testing (mobilenetv2) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- 5. Defenses (mobilenetv2) -----'

echo '----- Defense: SentiNet (mobilenetv2) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- Defense: ScaleUp (mobilenetv2) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- Defense: IBD_PSC (mobilenetv2) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

echo '----- Defense: NC (mobilenetv2) -----'
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"

#echo '----- Defense: FeatureRE (mobilenetv2) -----'
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=4 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=6 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=8 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=10 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=12 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=16 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=20 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.005 eps=24 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=4 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=6 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=8 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=10 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=12 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=16 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=20 (mobilenetv2)"
#run_command "python other_defense.py -defense=FeatureRE -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: FeatureRE tiny_imagenet upgd rate=0.001 eps=24 (mobilenetv2)"
