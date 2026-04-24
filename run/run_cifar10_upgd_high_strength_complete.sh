#!/bin/bash
# Mode: all-to-one poisoning for SIG/UPGD (labels of poisoned samples are reassigned to target_class)

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
echo "CIFAR10 UPGD Attack HIGH Strength Experiment Script"
echo "=========================================="

# ==============================================================================
# Model: mobilenetv2
# ==============================================================================

echo '----- 0. Clean Model Preparation (mobilenetv2) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "Create Clean: cifar10 mobilenetv2"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "Train Clean: cifar10 mobilenetv2"

echo '----- 1. Creation (mobilenetv2) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt -model=mobilenetv2" "Create: cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- 2. Training (mobilenetv2) -----'
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Train: cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- 3. Local Testing (mobilenetv2) -----'
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Test: cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- 4. Cross Testing (mobilenetv2) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- 5. Defenses (mobilenetv2) -----'

echo '----- Defense: SentiNet (mobilenetv2) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: SentiNet cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: STRIP cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- Defense: ScaleUp (mobilenetv2) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- Defense: IBD_PSC (mobilenetv2) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

echo '----- Defense: NC (mobilenetv2) -----'
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.05 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.05 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.05 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.05 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.01 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.01 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.01 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.01 eps=48 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.005 eps=28 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.005 eps=32 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.005 eps=40 (mobilenetv2)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Defense: NC cifar10 upgd rate=0.005 eps=48 (mobilenetv2)"

# ==============================================================================
# Model: resnet18
# ==============================================================================

echo '----- 0. Clean Model Preparation (resnet18) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=resnet18" "Create Clean: cifar10 resnet18"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=resnet18" "Train Clean: cifar10 resnet18"

echo '----- 1. Creation (resnet18) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt -model=resnet18" "Create: cifar10 upgd rate=0.005 eps=48 (resnet18)"

echo '----- 2. Training (resnet18) -----'
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Train: cifar10 upgd rate=0.005 eps=48 (resnet18)"

echo '----- 3. Local Testing (resnet18) -----'
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Test: cifar10 upgd rate=0.005 eps=48 (resnet18)"

echo '----- 4. Cross Testing (resnet18) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=28 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=32 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=40 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=48 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=28 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=32 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=40 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=48 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=28 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=32 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=40 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=48 (resnet18)"

echo '----- 5. Defenses (resnet18) -----'

echo '----- Defense: SentiNet (resnet18) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: SentiNet cifar10 upgd rate=0.005 eps=48 (resnet18)"

echo '----- Defense: STRIP (resnet18) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: STRIP cifar10 upgd rate=0.005 eps=48 (resnet18)"

echo '----- Defense: ScaleUp (resnet18) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=48 (resnet18)"

echo '----- Defense: IBD_PSC (resnet18) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=48 (resnet18)"

echo '----- Defense: NC (resnet18) -----'
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.05 eps=28 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.05 eps=32 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.05 eps=40 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.05 eps=48 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.01 eps=28 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.01 eps=32 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.01 eps=40 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.01 eps=48 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.005 eps=28 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.005 eps=32 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.005 eps=40 (resnet18)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Defense: NC cifar10 upgd rate=0.005 eps=48 (resnet18)"

# ==============================================================================
# Model: vgg19_bn
# ==============================================================================

echo '----- 0. Clean Model Preparation (vgg19_bn) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=vgg19_bn" "Create Clean: cifar10 vgg19_bn"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=vgg19_bn" "Train Clean: cifar10 vgg19_bn"

echo '----- 1. Creation (vgg19_bn) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=vgg19_bn_cifar10/vgg19_bn_cifar10.pt -model=vgg19_bn" "Create: cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- 2. Training (vgg19_bn) -----'
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Train: cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- 3. Local Testing (vgg19_bn) -----'
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Test: cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- 4. Cross Testing (vgg19_bn) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- 5. Defenses (vgg19_bn) -----'

echo '----- Defense: SentiNet (vgg19_bn) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: SentiNet cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- Defense: STRIP (vgg19_bn) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: STRIP cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- Defense: ScaleUp (vgg19_bn) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: ScaleUp cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- Defense: IBD_PSC (vgg19_bn) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: IBD_PSC cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"

echo '----- Defense: NC (vgg19_bn) -----'
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.05 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.05 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.05 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.05 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.01 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.01 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.01 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.01 eps=48 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=28 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.005 eps=28 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=32 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.005 eps=32 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=40 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.005 eps=40 (vgg19_bn)"
run_command "python other_defense.py -defense=NC -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=48 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Defense: NC cifar10 upgd rate=0.005 eps=48 (vgg19_bn)"
