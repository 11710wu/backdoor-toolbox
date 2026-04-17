#!/bin/bash

# 设置错误日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/run_vgg19_complete_errors_${TIMESTAMP}.log"

# 执行命令并记录失败的命令（失败时保存 stdout+stderr 到错误日志，便于排查）
# 使用 set +e 确保某条失败后继续执行后续所有命令
set +e

run_command() {
    local original_cmd="$1"
    local description="$2"
    local run_in_background=false
    local cmd="$original_cmd"
    local TMP_OUT
    TMP_OUT=$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_$RANDOM.out")

    # 检查是否是后台运行（命令末尾有 &）
    if [[ "$cmd" == *" &" ]]; then
        run_in_background=true
        cmd="${cmd% &}"  # 移除末尾的 & 用于执行
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

    # 如果命令失败，记录到错误日志：元信息 + 完整命令输出（stdout+stderr）
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

echo '----- 4. Cross Testing (mobilenetv2) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=4 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=4 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=6 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=6 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=8 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=8 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=10 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=10 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=12 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=12 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=16 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=16 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=20 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=20 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=24 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=24 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=4 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=4 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=6 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=6 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=8 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=8 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=10 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=10 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=12 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=12 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=16 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=16 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=20 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=20 (mobilenetv2) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=24 (mobilenetv2) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=24 (mobilenetv2) frost s=3"



echo '----- 4. Cross Testing (resnet18) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=4 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=4 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=6 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=6 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=8 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=8 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=10 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=10 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=12 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=12 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=16 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=16 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=20 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=20 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=24 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=24 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=4 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=4 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=6 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=6 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=8 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=8 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=10 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=10 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=12 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=12 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=16 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=16 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=20 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=20 (resnet18) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=24 (resnet18) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18 -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=24 (resnet18) frost s=3"



echo '----- 4. Cross Testing (vgg19_bn) -----'
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=4 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=4 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=6 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=6 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=8 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=8 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=10 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=10 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=12 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=12 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=16 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=16 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=20 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=20 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=24 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.005 eps=24 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=4 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=4 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=6 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=6 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=8 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=8 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=10 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=10 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=12 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=12 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=16 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=16 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=20 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=20 (vgg19_bn) frost s=3"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=2" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=24 (vgg19_bn) frost s=2"
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn -corruption_type=frost -severity=3" "Cross Test: test_tiny_imagenet.py upgd rate=0.001 eps=24 (vgg19_bn) frost s=3"



echo '----- 4. Cross Testing (mobilenetv2) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=4 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=6 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=8 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=10 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=12 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=16 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=20 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.05 eps=24 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=4 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=6 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=8 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=10 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=12 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=16 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=20 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.01 eps=24 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=4 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=6 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=8 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=10 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=12 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=16 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=20 (mobilenetv2)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "Cross Test: test_stl10.py upgd rate=0.005 eps=24 (mobilenetv2)"



echo '----- 4. Cross Testing (resnet18) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=4 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=6 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=8 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=10 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=12 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=16 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=20 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.05 eps=24 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=4 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=6 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=8 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=10 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=12 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=16 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=20 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.01 eps=24 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=4 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=6 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=8 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=10 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=12 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=16 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=20 (resnet18)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=resnet18" "Cross Test: test_stl10.py upgd rate=0.005 eps=24 (resnet18)"




echo '----- 4. Cross Testing (vgg19_bn) -----'
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=4 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=6 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=8 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=10 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=12 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=16 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=20 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.05 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.05 eps=24 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=4 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=6 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=8 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=10 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=12 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=16 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=20 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.01 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.01 eps=24 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=4 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=4 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=6 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=6 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=8 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=8 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=10 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=10 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=12 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=12 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=16 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=16 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=20 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=20 (vgg19_bn)"
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.005 -eps=24 -upgd_steps=100 -upgd_steps_multiplier=5 -model=vgg19_bn" "Cross Test: test_stl10.py upgd rate=0.005 eps=24 (vgg19_bn)"







echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "错误日志位置: $ERROR_LOG"
echo ""
if [ -s "$ERROR_LOG" ]; then
    # 统计失败的命令数量（每条含：时间戳、命令、描述、命令输出 stdout+stderr、分隔线）
    error_count=$(grep -c "命令执行失败" "$ERROR_LOG" 2>/dev/null || echo "0")
    echo "⚠️  发现 $error_count 个命令执行失败！"
    echo ""
    echo "失败的命令列表："
    echo "----------------------------------------"
    grep "命令:" "$ERROR_LOG" | sed 's/.*命令: //' | nl
    echo "----------------------------------------"
    echo ""
    echo "查看完整错误日志："
    echo "  cat $ERROR_LOG           # 查看所有错误详情"
    echo "  tail -n 50 $ERROR_LOG    # 查看最后50行错误"
else
    echo "✅ 所有命令执行成功，没有发现错误！"
fi
echo "=========================================="
