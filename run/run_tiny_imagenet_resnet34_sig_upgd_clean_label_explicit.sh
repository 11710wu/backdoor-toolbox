#!/bin/bash

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/run_tiny_imagenet_resnet34_sig_upgd_clean_label_explicit_${TIMESTAMP}.log"

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
echo "Tiny-ImageNet ResNet34 SIG/UPGD clean-label explicit -> poisoned_train_set4"
echo "=========================================="
echo "devices: ${DEVICES}"
echo "error log: ${ERROR_LOG}"
echo "output root: ${POISONED_TRAIN_SET_ROOT}"
echo "output note: poisoned sets, raw-base models, trained models, test results, transfer results, and defense outputs are written below this root"
echo "=========================================="


echo '----- 0. Raw Clean Base (resnet34) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=none -poison_rate=0.0" "Create clean set for raw UPGD base: tiny_imagenet resnet34"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Train raw-input clean base: tiny_imagenet resnet34"

echo '----- 1. Creation (resnet34) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean" "Create: tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean" "Create: tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean" "Create: tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Create: tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Create: tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Create: tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 2. Training (resnet34) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean" "Train: tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean" "Train: tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean" "Train: tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Train: tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Train: tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Train: tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 3. Local Testing (resnet34) -----'
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean" "Test: tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean" "Test: tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean" "Test: tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Test: tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Test: tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Test: tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 4. Cross Testing (resnet34) -----'
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 5. Defenses (resnet34) -----'

echo '----- Defense: SentiNet (resnet34) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean" "Defense: SentiNet tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean" "Defense: SentiNet tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean" "Defense: SentiNet tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: SentiNet tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: SentiNet tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: SentiNet tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- Defense: STRIP (resnet34) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean" "Defense: STRIP tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean" "Defense: STRIP tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean" "Defense: STRIP tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: STRIP tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: STRIP tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: STRIP tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- Defense: ScaleUp (resnet34) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean" "Defense: ScaleUp tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean" "Defense: ScaleUp tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean" "Defense: ScaleUp tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: ScaleUp tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: ScaleUp tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: ScaleUp tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- Defense: IBD_PSC (resnet34) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=20 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet SIG clean rate=0.001 delta=20 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=30 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet SIG clean rate=0.001 delta=30 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.001 -f=6 -delta=40 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet SIG clean rate=0.001 delta=40 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=20 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet SIG clean rate=0.005 delta=20 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=30 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet SIG clean rate=0.005 delta=30 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=SIG -poison_rate=0.005 -f=6 -delta=40 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet SIG clean rate=0.005 delta=40 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"
