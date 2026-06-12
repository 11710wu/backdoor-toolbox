#!/bin/bash

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/run_cifar10_small_cnn_sig_upgd_clean_label_explicit_${TIMESTAMP}.log"

set +e

DEVICES="${DEVICES:-0}"
export POISONED_TRAIN_SET_ROOT="${POISONED_TRAIN_SET_ROOT:-poisoned_train_set4}"
TARGET_DOMAIN_DIR="${TARGET_DOMAIN_DIR:-/workspace/data/imagenetv2-matched-frequency-tiny-organized}"

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
echo "CIFAR10 SmallCNN SIG/UPGD clean-label explicit -> poisoned_train_set4"
echo "=========================================="
echo "devices: ${DEVICES}"
echo "error log: ${ERROR_LOG}"
echo "output root: ${POISONED_TRAIN_SET_ROOT}"
echo "output note: poisoned sets, raw-base models, trained models, test results, transfer results, and defense outputs are written below this root"
echo "=========================================="


echo '----- 0. Raw Clean Base (small_cnn) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=none -poison_rate=0.0" "Create clean set for raw UPGD base: cifar10 small_cnn"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${POISONED_TRAIN_SET_ROOT}/cifar10/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10/upgd_raw_base_SmallCNN_cifar10.pt" "Train raw-input clean base: cifar10 small_cnn"

echo '----- 1. Creation (small_cnn) -----'
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Create: cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Create: cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Create: cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Create: cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Create: cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Create: cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/cifar10/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10/upgd_raw_base_SmallCNN_cifar10.pt" "Create: cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/cifar10/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10/upgd_raw_base_SmallCNN_cifar10.pt" "Create: cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/cifar10/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10/upgd_raw_base_SmallCNN_cifar10.pt" "Create: cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/cifar10/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10/upgd_raw_base_SmallCNN_cifar10.pt" "Create: cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/cifar10/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10/upgd_raw_base_SmallCNN_cifar10.pt" "Create: cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python create_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/cifar10/upgd_raw_base_0.000_poison_seed=2333_arch=SmallCNN_cifar10/upgd_raw_base_SmallCNN_cifar10.pt" "Create: cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"

echo '----- 2. Training (small_cnn) -----'
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Train: cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Train: cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Train: cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Train: cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Train: cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Train: cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"

echo '----- 3. Local Testing (small_cnn) -----'
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Test: cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Test: cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Test: cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Test: cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Test: cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Test: cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python test_model.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"

echo '----- 4. Cross Testing (small_cnn) -----'
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Cross Test: test_stl10.py cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Cross Test: test_stl10.py cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Cross Test: test_stl10.py cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Cross Test: test_stl10.py cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Cross Test: test_stl10.py cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Cross Test: test_stl10.py cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Cross Test: test_stl10.py cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Cross Test: test_stl10.py cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Cross Test: test_stl10.py cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Cross Test: test_stl10.py cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Cross Test: test_stl10.py cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python test_stl10.py -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Cross Test: test_stl10.py cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"

echo '----- 5. Defenses (small_cnn) -----'

echo '----- Defense: SentiNet (small_cnn) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Defense: SentiNet cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Defense: SentiNet cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Defense: SentiNet cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: SentiNet cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: SentiNet cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: SentiNet cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"

echo '----- Defense: STRIP (small_cnn) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Defense: STRIP cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Defense: STRIP cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Defense: STRIP cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: STRIP cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: STRIP cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: STRIP cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"

echo '----- Defense: ScaleUp (small_cnn) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Defense: ScaleUp cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Defense: ScaleUp cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Defense: ScaleUp cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: ScaleUp cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: ScaleUp cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: ScaleUp cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"

echo '----- Defense: IBD_PSC (small_cnn) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=20 -label_mode=clean" "Defense: IBD_PSC cifar10 SIG clean rate=0.01 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=30 -label_mode=clean" "Defense: IBD_PSC cifar10 SIG clean rate=0.01 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.01 -f=6 -delta=40 -label_mode=clean" "Defense: IBD_PSC cifar10 SIG clean rate=0.01 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: IBD_PSC cifar10 SIG clean rate=0.005 delta=20 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: IBD_PSC cifar10 SIG clean rate=0.005 delta=30 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: IBD_PSC cifar10 SIG clean rate=0.005 delta=40 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC cifar10 upgd clean rate=0.01 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC cifar10 upgd clean rate=0.01 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.01 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC cifar10 upgd clean rate=0.01 eps=12 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC cifar10 upgd clean rate=0.005 eps=4 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC cifar10 upgd clean rate=0.005 eps=8 (small_cnn)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -model=small_cnn -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC cifar10 upgd clean rate=0.005 eps=12 (small_cnn)"
