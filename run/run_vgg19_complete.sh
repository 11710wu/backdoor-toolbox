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

echo "=========================================="
echo "VGG19_BN 后门攻击完整实验 (按数据集分组)"
echo "=========================================="
echo "模型: VGG19_BN"
echo "攻击: Basic, Blend, Adaptive-Patch, SIG, WaNet, UPGD, BELT, Adaptive-Blend"
echo "数据集: CIFAR-10→STL-10, Tiny ImageNet→Tiny ImageNet-C, MNIST-M→MNIST"
echo "防御: AC, STRIP, SentiNet, IBD-PSC, ScaleUp"
echo "=========================================="
echo "错误日志: $ERROR_LOG"
echo "=========================================="

# ==========================================
# 0. 干净模型训练 (基准模型)
# ==========================================

echo "=========================================="
echo "0. 干净数据集和模型准备"
echo "=========================================="

echo "0.1 创建干净的CIFAR-10数据集..."
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "创建干净的CIFAR-10数据集"

echo "0.2 训练干净的MobileNetV2模型 (CIFAR-10)..."
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "训练干净的MobileNetV2模型 (CIFAR-10)"

echo "干净模型训练完成！"
echo ""


echo "=========================================="
echo "0. 干净数据集和模型准备"
echo "=========================================="

echo "0.1 创建干净的tiny_imagenet数据集..."
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "创建干净的tiny_imagenet数据集"

echo "0.2 训练干净的MobileNetV2模型 (tiny_imagenet)..."
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "训练干净的MobileNetV2模型 (tiny_imagenet)"

echo "干净模型训练完成！"
echo ""



echo "=========================================="
echo "0. 干净数据集和模型准备"
echo "=========================================="

echo "0.1 创建干净的mnistm数据集..."
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "创建干净的mnistm数据集"

echo "0.2 训练干净的MobileNetV2模型 (mnistm)..."
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=none -poison_rate=0.0 -model=mobilenetv2" "训练干净的MobileNetV2模型 (mnistm)"

echo "干净模型训练完成！"
echo ""

# ==========================================
# 命令分组说明：
# - 按数据集分组：每个数据集作为一个大组
# - 每个数据集组内包含所有攻击方法的完整流程
# - 每个阶段（创建/训练/测试/防御）：三个数据集同时运行
# ==========================================

echo "=========================================="
echo "1. CIFAR-10 → STL-10 实验 (VGG19-BN) [大部分跳过，UPGD部分仍在运行]"
echo "=========================================="

echo "===== 创建投毒数据集 ====="
echo "1.1 创建投毒数据集 - 第1组 (Basic, Blend, Adaptive-Patch)..."
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "创建投毒数据集 - CIFAR-10 Basic"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "创建投毒数据集 - CIFAR-10 Blend"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "创建投毒数据集 - CIFAR-10 Adaptive-Patch"


echo "1.2 创建投毒数据集 - 第2组 (UPGD，SIG和WaNet数据集已存在)..."
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "创建投毒数据集 - CIFAR-10 SIG"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "创建投毒数据集 - CIFAR-10 WaNet"
run_command "python create_poisoned_set.py -dataset=cifar10  -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -upgd_model_path=poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=mobilenetv2_cifar10/mobilenetv2_cifar10.pt" "创建投毒数据集 - CIFAR-10 UPGD"


echo "1.3 创建投毒数据集 - 第3组 (BELT, Adaptive-Blend)..."
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "创建投毒数据集 - CIFAR-10 BELT"
run_command "python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "创建投毒数据集 - CIFAR-10 Adaptive-Blend"


echo "1.4 创建投毒数据集 - 第4组 (Basic, Blend, Adaptive-Patch)..."
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"


echo "1.5 创建投毒数据集 - 第5组 (SIG, WaNet, UPGD)..."
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -upgd_model_path=poisoned_train_set/tiny_imagenet/none_0.000_poison_seed=2333_arch=mobilenetv2_tiny_imagenet/mobilenetv2_tiny_imagenet.pt" "tiny_imagenet upgd"


echo "1.6 创建投毒数据集 - 第6组 (BELT, Adaptive-Blend)..."
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"
run_command "python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"


echo "1.7 创建投毒数据集 - 第7组 (Basic, Blend, Adaptive-Patch)..."
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"


echo "1.8 创建投毒数据集 - 第8组 (SIG, WaNet, UPGD)..."
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 -upgd_model_path=poisoned_train_set/mnistm/none_0.000_poison_seed=2333_arch=mobilenetv2_mnistm/mobilenetv2_mnistm.pt" "mnistm upgd"
# [注释：MNIST-M UPGD跳过 - 没有对应的干净MNIST-M模型]


echo "1.9 创建投毒数据集 - 第9组 (BELT, Adaptive-Blend)..."
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"
run_command "python create_poisoned_set.py -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"
    

echo "===== 训练后门模型 ====="
echo "2.1 训练后门模型 - CIFAR-10 第1组..."
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "cifar10 basic"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "cifar10 blend"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "cifar10 adaptive_patch"


echo "2.2 训练后门模型 - CIFAR-10 第2组 (仅UPGD，SIG和WaNet已训练)..."
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "cifar10 SIG"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "cifar10 WaNet"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2 &" "训练后门模型 - CIFAR-10 UPGD"


echo "2.3 训练后门模型 - CIFAR-10 第3组..."
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "cifar10 belt"
run_command "python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "cifar10 adaptive_blend"


echo "2.4 训练后门模型 - Tiny ImageNet 第1组..."
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"


echo "2.5 训练后门模型 - Tiny ImageNet 第2组..."
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "tiny_imagenet upgd"


echo "2.6 训练后门模型 - Tiny ImageNet 第3组..."
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"
run_command "python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"


echo "2.7 训练后门模型 - MNIST-M 第1组..."
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"


echo "2.8 训练后门模型 - MNIST-M 第2组..."
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "mnistm upgd"


echo "2.9 训练后门模型 - MNIST-M 第3组..."
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"
run_command "python train_on_poisoned_set.py -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"


echo "===== 本地测试 ====="
echo "3.1.1 本地测试 - CIFAR-10 Basic..."
run_command "python test_model.py -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "cifar10 basic"

echo "3.1.2 本地测试 - CIFAR-10 Blend..."
run_command "python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "cifar10 blend"

echo "3.1.3 本地测试 - CIFAR-10 Adaptive-Patch..."
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "cifar10 adaptive_patch"

echo "3.2.1 本地测试 - CIFAR-10 SIG..."
run_command "python test_model.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "cifar10 SIG"

echo "3.2.2 本地测试 - CIFAR-10 WaNet..."
run_command "python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "cifar10 WaNet"

echo "3.2.3 本地测试 - CIFAR-10 UPGD..."
run_command "python test_model.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "cifar10 upgd"

echo "3.3.1 本地测试 - CIFAR-10 BELT..."
run_command "python test_model.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "cifar10 belt"

echo "3.3.2 本地测试 - CIFAR-10 Adaptive-Blend..."
run_command "python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "cifar10 adaptive_blend"

echo "3.4.1 本地测试 - Tiny ImageNet Basic..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"

echo "3.4.2 本地测试 - Tiny ImageNet Blend..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"

echo "3.4.3 本地测试 - Tiny ImageNet Adaptive-Patch..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"

echo "3.5.1 本地测试 - Tiny ImageNet SIG..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"

echo "3.5.2 本地测试 - Tiny ImageNet WaNet..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"

echo "3.5.3 本地测试 - Tiny ImageNet UPGD..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "tiny_imagenet upgd"

echo "3.6.1 本地测试 - Tiny ImageNet BELT..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"

echo "3.6.2 本地测试 - Tiny ImageNet Adaptive-Blend..."
run_command "python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"

echo "3.7.1 本地测试 - MNIST-M Basic..."
run_command "python test_model.py -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"

echo "3.7.2 本地测试 - MNIST-M Blend..."
run_command "python test_model.py -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"

echo "3.7.3 本地测试 - MNIST-M Adaptive-Patch..."
run_command "python test_model.py -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"

echo "3.8.1 本地测试 - MNIST-M SIG..."
run_command "python test_model.py -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"

echo "3.8.2 本地测试 - MNIST-M WaNet..."
run_command "python test_model.py -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"

echo "3.8.3 本地测试 - MNIST-M UPGD..."
run_command "python test_model.py -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "mnistm upgd"

echo "3.9.1 本地测试 - MNIST-M BELT..."
run_command "python test_model.py -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"

echo "3.9.2 本地测试 - MNIST-M Adaptive-Blend..."
run_command "python test_model.py -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"

# ===== 跨域测试 =====
echo "4.1.1 跨域测试 - CIFAR-10 → STL-10 Basic..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "cifar10 basic"

echo "4.1.2 跨域测试 - CIFAR-10 → STL-10 Blend..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "cifar10 blend"

echo "4.1.3 跨域测试 - CIFAR-10 → STL-10 Adaptive-Patch..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "cifar10 adaptive_patch"

echo "4.2.1 跨域测试 - CIFAR-10 → STL-10 SIG..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "cifar10 SIG"

echo "4.2.2 跨域测试 - CIFAR-10 → STL-10 WaNet..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "cifar10 WaNet"

echo "4.2.3 跨域测试 - CIFAR-10 → STL-10 UPGD..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "cifar10 upgd"

echo "4.3.1 跨域测试 - CIFAR-10 → STL-10 BELT..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "cifar10 belt"

echo "4.3.2 跨域测试 - CIFAR-10 → STL-10 Adaptive-Blend..."
run_command "python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "cifar10 adaptive_blend"

echo "4.4.1 跨域测试 - Tiny ImageNet → Tiny ImageNet-C Basic..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"

echo "4.4.2 跨域测试 - Tiny ImageNet → Tiny ImageNet-C Blend..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"

echo "4.4.3 跨域测试 - Tiny ImageNet → Tiny ImageNet-C Adaptive-Patch..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"

echo "4.5.1 跨域测试 - Tiny ImageNet → Tiny ImageNet-C SIG..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"

echo "4.5.2 跨域测试 - Tiny ImageNet → Tiny ImageNet-C WaNet..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"

echo "4.5.3 跨域测试 - Tiny ImageNet → Tiny ImageNet-C UPGD..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "tiny_imagenet upgd"

echo "4.6.1 跨域测试 - Tiny ImageNet → Tiny ImageNet-C BELT..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"

echo "4.6.2 跨域测试 - Tiny ImageNet → Tiny ImageNet-C Adaptive-Blend..."
run_command "python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"

echo "4.7.1 跨域测试 - MNIST-M → MNIST Basic..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"

echo "4.7.2 跨域测试 - MNIST-M → MNIST Blend..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"

echo "4.7.3 跨域测试 - MNIST-M → MNIST Adaptive-Patch..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"

echo "4.8.1 跨域测试 - MNIST-M → MNIST SIG..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"

echo "4.8.2 跨域测试 - MNIST-M → MNIST WaNet..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"

echo "4.8.3 跨域测试 - MNIST-M → MNIST UPGD..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "mnistm upgd"

echo "4.9.1 跨域测试 - MNIST-M → MNIST BELT..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"

echo "4.9.2 跨域测试 - MNIST-M → MNIST Adaptive-Blend..."
run_command "python test_mnist.py -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"



===== SentiNet防御 =====
echo "5.10.1 SentiNet防御 - CIFAR-10 Basic..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "cifar10 basic"

echo "5.10.2 SentiNet防御 - Tiny ImageNet Basic..."
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"

echo "5.10.3 SentiNet防御 - MNIST-M Basic..."
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"

echo "5.11.1 SentiNet防御 - CIFAR-10 Blend..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "cifar10 blend"

echo "5.11.2 SentiNet防御 - Tiny ImageNet Blend..."
run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"

echo "5.11.3 SentiNet防御 - MNIST-M Blend..."
run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"

echo "5.12.1 SentiNet防御 - CIFAR-10 Adaptive-Patch..."
echo "5.12.2 SentiNet防御 - Tiny ImageNet Adaptive-Patch..."
echo "5.12.3 SentiNet防御 - MNIST-M Adaptive-Patch..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "cifar10 adaptive_patch"

run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"

run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"


echo "5.13.1 SentiNet防御 - CIFAR-10 SIG..."
echo "5.13.2 SentiNet防御 - Tiny ImageNet SIG..."
echo "5.13.3 SentiNet防御 - MNIST-M SIG..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "cifar10 SIG"

run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"

run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"


echo "5.WaNet.1 SentiNet防御 - CIFAR-10 WaNet..."
echo "5.WaNet.2 SentiNet防御 - Tiny ImageNet WaNet..."
echo "5.WaNet.3 SentiNet防御 - MNIST-M WaNet..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "cifar10 WaNet"

run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"

run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"


echo "5.UPGD.1 SentiNet防御 - CIFAR-10 UPGD..."
echo "5.UPGD.2 SentiNet防御 - Tiny ImageNet UPGD..."
echo "5.UPGD.3 SentiNet防御 - MNIST-M UPGD..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "cifar10 upgd"

run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "tiny_imagenet upgd"

run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "mnistm upgd"


echo "5.BELT.1 SentiNet防御 - CIFAR-10 BELT..."
echo "5.BELT.2 SentiNet防御 - Tiny ImageNet BELT..."
echo "5.BELT.3 SentiNet防御 - MNIST-M BELT..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "cifar10 belt"

run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"

run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"


echo "5.Adaptive-Blend.1 SentiNet防御 - CIFAR-10 Adaptive-Blend..."
echo "5.Adaptive-Blend.2 SentiNet防御 - Tiny ImageNet Adaptive-Blend..."
echo "5.Adaptive-Blend.3 SentiNet防御 - MNIST-M Adaptive-Blend..."
run_command "python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "cifar10 adaptive_blend"

run_command "python other_defense.py -defense=SentiNet -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"

run_command "python other_defense.py -defense=SentiNet -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"


# ===== ScaleUp防御 =====
echo "5.18.1 ScaleUp防御 - CIFAR-10 Basic..."
echo "5.18.2 ScaleUp防御 - Tiny ImageNet Basic..."
echo "5.18.3 ScaleUp防御 - MNIST-M Basic..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "cifar10 basic"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"


echo "5.19.1 ScaleUp防御 - CIFAR-10 Blend..."
echo "5.19.2 ScaleUp防御 - Tiny ImageNet Blend..."
echo "5.19.3 ScaleUp防御 - MNIST-M Blend..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "cifar10 blend"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"


echo "5.20.1 ScaleUp防御 - CIFAR-10 Adaptive-Patch..."
echo "5.20.2 ScaleUp防御 - Tiny ImageNet Adaptive-Patch..."
echo "5.20.3 ScaleUp防御 - MNIST-M Adaptive-Patch..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "cifar10 adaptive_patch"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"


echo "5.21.1 ScaleUp防御 - CIFAR-10 SIG..."
echo "5.21.2 ScaleUp防御 - Tiny ImageNet SIG..."
echo "5.21.3 ScaleUp防御 - MNIST-M SIG..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "cifar10 SIG"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"


echo "5.22.1 ScaleUp防御 - CIFAR-10 WaNet..."
echo "5.22.2 ScaleUp防御 - Tiny ImageNet WaNet..."
echo "5.22.3 ScaleUp防御 - MNIST-M WaNet..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "cifar10 WaNet"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"


echo "5.23.1 ScaleUp防御 - CIFAR-10 UPGD..."
echo "5.23.2 ScaleUp防御 - Tiny ImageNet UPGD..."
echo "5.23.3 ScaleUp防御 - MNIST-M UPGD..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "cifar10 upgd"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "tiny_imagenet upgd"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "mnistm upgd"


echo "5.24.1 ScaleUp防御 - CIFAR-10 BELT..."
echo "5.24.2 ScaleUp防御 - Tiny ImageNet BELT..."
echo "5.24.3 ScaleUp防御 - MNIST-M BELT..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "cifar10 belt"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"


echo "5.25.1 ScaleUp防御 - CIFAR-10 Adaptive-Blend..."
echo "5.25.2 ScaleUp防御 - Tiny ImageNet Adaptive-Blend..."
echo "5.25.3 ScaleUp防御 - MNIST-M Adaptive-Blend..."
run_command "python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "cifar10 adaptive_blend"

run_command "python other_defense.py -defense=ScaleUp -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"

run_command "python other_defense.py -defense=ScaleUp -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"


# ===== STRIP防御 =====
echo "5.26 防御测试 - STRIP防御 Basic..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "cifar10 basic"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"


echo "5.27 防御测试 - STRIP防御 Blend..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "cifar10 blend"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"


echo "5.28 防御测试 - STRIP防御 Adaptive-Patch..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "cifar10 adaptive_patch"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"


echo "5.29 防御测试 - STRIP防御 SIG..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "cifar10 SIG"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"


echo "5.30 防御测试 - STRIP防御 WaNet..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "cifar10 WaNet"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"


echo "5.31 防御测试 - STRIP防御 UPGD..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "cifar10 upgd"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "tiny_imagenet upgd"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "mnistm upgd"


echo "5.32 防御测试 - STRIP防御 BELT..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "cifar10 belt"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"


echo "5.33 防御测试 - STRIP防御 Adaptive-Blend..."
run_command "python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "cifar10 adaptive_blend"

run_command "python other_defense.py -defense=STRIP -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"

run_command "python other_defense.py -defense=STRIP -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"


# ===== IBD-PSC防御 =====
echo "5.34.1 IBD-PSC防御 - Basic CIFAR-10..."
echo "5.34.2 IBD-PSC防御 - Basic Tiny ImageNet..."
echo "5.34.3 IBD-PSC防御 - Basic MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "cifar10 basic"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "tiny_imagenet basic"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=basic -poison_rate=0.1 -model=mobilenetv2" "mnistm basic"


echo "5.35.1 IBD-PSC防御 - Blend CIFAR-10..."
echo "5.35.2 IBD-PSC防御 - Blend Tiny ImageNet..."
echo "5.35.3 IBD-PSC防御 - Blend MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "cifar10 blend"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet blend"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=blend -poison_rate=0.1 -alpha=0.2 -model=mobilenetv2" "mnistm blend"


echo "5.36.1 IBD-PSC防御 - Adaptive-Patch CIFAR-10..."
echo "5.36.2 IBD-PSC防御 - Adaptive-Patch Tiny ImageNet..."
echo "5.36.3 IBD-PSC防御 - Adaptive-Patch MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "cifar10 adaptive_patch"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "tiny_imagenet adaptive_patch"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=adaptive_patch -poison_rate=0.03 -cover_rate=0.06 -model=mobilenetv2" "mnistm adaptive_patch"


echo "5.37.1 IBD-PSC防御 - SIG CIFAR-10..."
echo "5.37.2 IBD-PSC防御 - SIG Tiny ImageNet..."
echo "5.37.3 IBD-PSC防御 - SIG MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "cifar10 SIG"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "tiny_imagenet SIG"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=SIG -poison_rate=0.1 -delta=30 -f=6 -model=mobilenetv2" "mnistm SIG"


echo "5.38.1 IBD-PSC防御 - WaNet CIFAR-10..."
echo "5.38.2 IBD-PSC防御 - WaNet Tiny ImageNet..."
echo "5.38.3 IBD-PSC防御 - WaNet MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "cifar10 WaNet"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "tiny_imagenet WaNet"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.06 -s=0.5 -k=4 -model=mobilenetv2" "mnistm WaNet"


echo "5.39.1 IBD-PSC防御 - UPGD CIFAR-10..."
echo "5.39.2 IBD-PSC防御 - UPGD Tiny ImageNet..."
echo "5.39.3 IBD-PSC防御 - UPGD MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "cifar10 upgd"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "tiny_imagenet upgd"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=upgd -poison_rate=0.001 -eps=8.0 -upgd_steps=100 -upgd_steps_multiplier=5 -model=mobilenetv2" "mnistm upgd"


echo "5.40.1 IBD-PSC防御 - BELT CIFAR-10..."
echo "5.40.2 IBD-PSC防御 - BELT Tiny ImageNet..."
echo "5.40.3 IBD-PSC防御 - BELT MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "cifar10 belt"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "tiny_imagenet belt"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=belt -poison_rate=0.1 -cover_rate=0.5 -model=mobilenetv2" "mnistm belt"


echo "5.41.1 IBD-PSC防御 - Adaptive-Blend CIFAR-10..."
echo "5.41.2 IBD-PSC防御 - Adaptive-Blend Tiny ImageNet..."
echo "5.41.3 IBD-PSC防御 - Adaptive-Blend MNIST-M..."
run_command "python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "cifar10 adaptive_blend"

run_command "python other_defense.py -defense=IBD_PSC -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "tiny_imagenet adaptive_blend"

run_command "python other_defense.py -defense=IBD_PSC -dataset=mnistm -poison_type=adaptive_blend -poison_rate=0.1 -cover_rate=0.01 -alpha=0.2 -model=mobilenetv2" "mnistm adaptive_blend"


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
