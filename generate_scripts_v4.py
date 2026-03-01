import os

# Configuration
datasets = ["cifar10", "mnistm", "tiny_imagenet"]
# Strictly following user request for model names (including case)
models = ["mobilenetv2", "resnet18", "vgg19"] 
alphas = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]  # Alpha 0.2-1.0
poison_rates = ["0.05", "0.01", "0.005"]
poison_type = "basic"
defenses = ["SentiNet", "STRIP", "ScaleUp", "IBD_PSC", "NC", "FeatureRE"]

# Log file name as requested
error_log_name = "run_defenses_mobilenet_resnet_explicit.log"

output_dir = "run"
os.makedirs(output_dir, exist_ok=True)

def generate_script(dataset, model):
    # Determine poison rates based on dataset
    if dataset == "tiny_imagenet":
        current_poison_rates = ["0.005", "0.001"]
    else:
        current_poison_rates = poison_rates

    # Cross domain logic
    cross_script = None
    if dataset == "cifar10":
        cross_script = "test_stl10.py"
    elif dataset == "mnistm":
        cross_script = "test_mnist.py" # Reasonable cross for MNIST-M
    elif dataset == "tiny_imagenet":
        cross_script = "test_tiny_imagenet.py"  # Tiny ImageNet -> Tiny ImageNet-C 跨域测试

    script_content = f"""#!/bin/bash

# 设置错误日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# User requested specific log name
ERROR_LOG="$LOG_DIR/{error_log_name}"

# 执行命令并记录失败的命令
set +e

run_command() {{
    local original_cmd="$1"
    local description="$2"
    local run_in_background=false
    local cmd="$original_cmd"
    local TMP_OUT
    TMP_OUT=$(mktemp 2>/dev/null || echo "/tmp/run_cmd_$$_${{RANDOM}}.out")

    if [[ "$cmd" == *" &" ]]; then
        run_in_background=true
        cmd="${{cmd% &}}"
    fi

    if [ "$run_in_background" = true ]; then
        eval "$cmd" > "$TMP_OUT" 2>&1 &
        local pid=$!
        wait $pid
        local exit_code=$?
    else
        eval "$cmd" 2>&1 | tee "$TMP_OUT"
        local exit_code=${{PIPESTATUS[0]}}
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
}}

echo "=========================================="
echo "{dataset.upper()} Basic Attack Experiment Script - Model: {model}"
echo "=========================================="
"""

    script_content += f"\n# ==============================================================================\n"
    script_content += f"# Model: {model}\n"
    script_content += f"# ==============================================================================\n"

    # 1. Creation Loop (Separate)
    script_content += f"\necho '----- 1. Creation ({model}) -----'\n"
    for rate in current_poison_rates:
        for alpha in alphas:
            cmd = f"python create_poisoned_set.py -dataset={dataset} -poison_type={poison_type} -poison_rate={rate} -alpha={alpha} -model={model}"
            desc = f"Create: {dataset} {poison_type} rate={rate} alpha={alpha} ({model})"
            script_content += f'run_command "{cmd}" "{desc}"\n'

    # 2. Training Loop (Separate)
    script_content += f"\necho '----- 2. Training ({model}) -----'\n"
    for rate in current_poison_rates:
        for alpha in alphas:
            cmd = f"python train_on_poisoned_set.py -dataset={dataset} -poison_type={poison_type} -poison_rate={rate} -alpha={alpha} -model={model}"
            desc = f"Train: {dataset} {poison_type} rate={rate} alpha={alpha} ({model})"
            script_content += f'run_command "{cmd}" "{desc}"\n'

    # 3. Local Testing Loop
    script_content += f"\necho '----- 3. Local Testing ({model}) -----'\n"
    for rate in current_poison_rates:
        for alpha in alphas:
            cmd = f"python test_model.py -dataset={dataset} -poison_type={poison_type} -poison_rate={rate} -alpha={alpha} -model={model}"
            desc = f"Test: {dataset} {poison_type} rate={rate} alpha={alpha} ({model})"
            script_content += f'run_command "{cmd}" "{desc}"\n'

    # 4. Cross Testing Loop (if applicable)
    if cross_script:
        script_content += f"\necho '----- 4. Cross Testing ({model}) -----'\n"
        for rate in current_poison_rates:
            for alpha in alphas:
                cmd = f"python {cross_script} -dataset={dataset} -poison_type={poison_type} -poison_rate={rate} -alpha={alpha} -model={model}"
                desc = f"Cross Test: {cross_script} {poison_type} rate={rate} alpha={alpha} ({model})"
                script_content += f'run_command "{cmd}" "{desc}"\n'

    # 5. Defenses Loop
    script_content += f"\necho '----- 5. Defenses ({model}) -----'\n"
    for defense in defenses:
        script_content += f"\necho '----- Defense: {defense} ({model}) -----'\n"
        for rate in current_poison_rates:
            for alpha in alphas:
                cmd = f"python other_defense.py -defense={defense} -dataset={dataset} -poison_type={poison_type} -poison_rate={rate} -alpha={alpha} -model={model}"
                desc = f"Defense: {defense} {dataset} {poison_type} rate={rate} alpha={alpha} ({model})"
                script_content += f'run_command "{cmd}" "{desc}"\n'

    return script_content

for dataset in datasets:
    for model in models:
        script = generate_script(dataset, model)
        filename = f"{output_dir}/run_{dataset}_basic_{model}_complete.sh"
        with open(filename, "w") as f:
            f.write(script)
        print(f"Generated {filename}")
