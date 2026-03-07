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



echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.01 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.01 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.05 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.05 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.15 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.15 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.25 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.25 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.01 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.01 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.05 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.05 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.15 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.15 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.25 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.25 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.01 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.01 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.05 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.05 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.15 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.25 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.25 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.3 (mobilenetv2)"


echo '----- Defense: STRIP (resnet18) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.01 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.05 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.15 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.25 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.01 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.05 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.15 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.25 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.01 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.01 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.05 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.05 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.15 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.25 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.25 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.3 (resnet18)"



echo '----- Defense: STRIP (vgg19) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.01 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.01 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.05 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.05 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.15 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.15 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.25 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.25 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.05 -cover_rate=0.05 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.05 cover=0.05 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.01 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.01 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.05 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.05 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.15 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.15 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.25 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.25 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.01 -cover_rate=0.01 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.01 cover=0.01 alpha=0.3 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.01 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.01 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.05 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.05 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.1 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.1 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.15 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.2 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.2 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.25 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.25 (vgg19)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.3 -model=vgg19_bn" "Defense: STRIP cifar10 adaptive_blend rate=0.005 cover=0.005 alpha=0.3 (vgg19)"



echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (mobilenetv2)"




echo '----- Defense: STRIP (resnet18) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.05 -cover_rate=0.1 -alpha=0.4 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.05 cover=0.1 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.01 -cover_rate=0.02 -alpha=0.4 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.01 cover=0.02 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.1 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.1 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.005 -cover_rate=0.01 -alpha=0.4 -model=resnet18" "Defense: STRIP cifar10 adaptive_patch rate=0.005 cover=0.01 alpha=0.4 (resnet18)"



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




echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.5 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.5 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.6 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.6 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.7 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.7 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.8 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.8 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.9 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.9 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=1.0 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.05 alpha=1.0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.5 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.5 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.6 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.6 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.7 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.7 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.8 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.8 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.9 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.9 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=1.0 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.01 alpha=1.0 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.4 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.4 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.5 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.5 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.6 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.6 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.7 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.7 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.8 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.8 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.9 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.9 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=1.0 -model=mobilenetv2" "Defense: STRIP cifar10 basic rate=0.005 alpha=1.0 (mobilenetv2)"




echo '----- Defense: STRIP (resnet18) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.4 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.5 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.6 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.7 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.8 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=0.9 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.05 -alpha=1.0 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.05 alpha=1.0 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.4 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.5 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.6 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.7 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.8 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=0.9 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.01 -alpha=1.0 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.01 alpha=1.0 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.2 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.2 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.3 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.3 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.4 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.4 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.5 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.5 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.6 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.6 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.7 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.7 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.8 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.8 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=0.9 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=0.9 (resnet18)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.005 -alpha=1.0 -model=resnet18" "Defense: STRIP cifar10 basic rate=0.005 alpha=1.0 (resnet18)"



echo '----- Defense: STRIP (mobilenetv2) -----'
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.01 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.01 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.05 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.05 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.15 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.15 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.25 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.25 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.05 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.05 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.01 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.01 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.05 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.05 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.15 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.15 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.25 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.25 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.01 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.01 alpha=0.3 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.01 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.01 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.05 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.05 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.1 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.1 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.15 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.15 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.2 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.2 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.25 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.25 (mobilenetv2)"
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.005 -alpha=0.3 -model=mobilenetv2" "Defense: STRIP cifar10 blend rate=0.005 alpha=0.3 (mobilenetv2)"

