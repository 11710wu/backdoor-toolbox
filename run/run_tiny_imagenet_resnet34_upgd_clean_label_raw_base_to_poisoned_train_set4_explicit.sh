#!/bin/bash

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/run_tiny_imagenet_resnet34_upgd_clean_label_raw_base_to_poisoned_train_set4_explicit_${TIMESTAMP}.log"

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
echo "Tiny-ImageNet ResNet34 UPGD clean-label raw-base explicit -> poisoned_train_set4"
echo "=========================================="
echo "devices: ${DEVICES}"
echo "error log: ${ERROR_LOG}"
echo "output root: ${POISONED_TRAIN_SET_ROOT}"
echo "output dataset dir: ${POISONED_TRAIN_SET_ROOT}/tiny_imagenet"
echo "raw-base model: ${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt"
echo "poisoned-set dirs: ${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_{0.001,0.005}_eps={4,8,12}_constraint=Linf_steps=100_mode=clean_mult=5_arch=ResNet34_tiny_imagenet"
echo "trained models, test results, transfer results, and defense outputs are written under each poisoned-set dir"
echo "=========================================="

echo '----- 0. Raw Clean Base: UPGD clean-label (resnet34) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=none -poison_rate=0.0" "Create clean set for raw UPGD base: tiny_imagenet resnet34"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=none -poison_rate=0.0 -no_normalize -model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Train raw-input clean base: tiny_imagenet resnet34"

echo '----- 1. Creation: UPGD clean-label (resnet34) -----'
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -upgd_model_path=${POISONED_TRAIN_SET_ROOT}/tiny_imagenet/upgd_raw_base_0.000_poison_seed=2333_arch=ResNet34_tiny_imagenet/upgd_raw_base_ResNet34_tiny_imagenet.pt" "Create: tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 2. Training: UPGD clean-label (resnet34) -----'
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Train: tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 3. Local Testing: UPGD clean-label (resnet34) -----'
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python test_model.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Test: tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 4. Cross Testing: UPGD clean-label (resnet34) -----'
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python test_tiny_target_domain.py -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean -source_dataset=tiny_imagenet -target_domain_dir=${TARGET_DOMAIN_DIR}" "Transfer: tiny target-domain tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- 5. Defenses: UPGD clean-label (resnet34) -----'

echo '----- Defense: SentiNet UPGD clean-label (resnet34) -----'
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: SentiNet tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- Defense: STRIP UPGD clean-label (resnet34) -----'
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: STRIP tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- Defense: ScaleUp UPGD clean-label (resnet34) -----'
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: ScaleUp tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"

echo '----- Defense: IBD_PSC UPGD clean-label (resnet34) -----'
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.001 eps=4 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.001 eps=8 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.001 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.001 eps=12 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=4 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.005 eps=4 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=8 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.005 eps=8 (resnet34)"
run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -model=resnet34 -devices=${DEVICES} -poison_type=upgd -poison_rate=0.005 -eps=12 -constraint=Linf -upgd_steps=100 -upgd_steps_multiplier=5 -label_mode=clean" "Defense: IBD_PSC tiny_imagenet upgd clean rate=0.005 eps=12 (resnet34)"
