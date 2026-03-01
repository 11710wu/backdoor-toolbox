#!/bin/bash

# 设置错误日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ERROR_LOG="$LOG_DIR/run_resnet18_complete_errors_${TIMESTAMP}.log"

# 只重定向错误输出到日志文件，同时显示在终端
exec 2> >(tee -a "$ERROR_LOG" >&2)

echo "=========================================="
echo "ResNet18 后门攻击完整实验 (按数据集分组)"
echo "=========================================="
echo "模型: ResNet18"
echo "攻击: BadNet, Blend, Adaptive-Patch, SIG, WaNet, Adaptive-Blend"
echo "数据集: CIFAR-10→STL-10"
echo "防御: AC, STRIP, SentiNet, IBD-PSC, ScaleUp"
echo "=========================================="
echo "错误日志: $ERROR_LOG"
echo "=========================================="

# ==========================================
# 0. 干净模型训练 (基准模型)
# ==========================================



# ==========================================
# 命令分组说明：
# - 只针对CIFAR-10→STL-10数据集进行实验
# - 包含所有攻击方法的完整流程
# - 按阶段顺序执行：创建投毒数据集 → 训练后门模型 → 本地测试 → 跨域测试 → 防御测试
# ==========================================

echo "=========================================="
echo "1. CIFAR-10 → STL-10 实验 (ResNet18)"
echo "=========================================="

===== 创建投毒数据集 =====
echo "1.1 创建投毒数据集 - 第1组 (Blend, Adaptive-Patch)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001


echo "1.2 创建投毒数据集 - 第2组 (SIG和WaNet数据集已存在)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18


echo "1.3 创建投毒数据集 - 第3组 (Adaptive-Blend)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha=0.21 -model=resnet18




    

===== 训练后门模型 =====
echo "2.1 训练后门模型 - CIFAR-10 第1组..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001


echo "2.2 训练后门模型 - CIFAR-10 第2组 (SIG和WaNet已训练)..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18


echo "2.3 训练后门模型 - CIFAR-10 第3组..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18






===== 本地测试 =====
echo "3.1.2 本地测试 - CIFAR-10 Blend..."
python test_model.py -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18

echo "3.1.3 本地测试 - CIFAR-10 Adaptive-Patch..."
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001

echo "3.2.1 本地测试 - CIFAR-10 SIG..."
python test_model.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18

echo "3.2.2 本地测试 - CIFAR-10 WaNet..."
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18


echo "3.3.1 本地测试 - CIFAR-10 Adaptive-Blend..."
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18



# ===== 跨域测试 =====
echo "4.1.2 跨域测试 - CIFAR-10 → STL-10 Blend..."
python test_stl10.py -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18

echo "4.1.3 跨域测试 - CIFAR-10 → STL-10 Adaptive-Patch..."
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001

echo "4.2.1 跨域测试 - CIFAR-10 → STL-10 SIG..."
python test_stl10.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18

echo "4.2.2 跨域测试 - CIFAR-10 → STL-10 WaNet..."
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18


echo "4.3.1 跨域测试 - CIFAR-10 → STL-10 Adaptive-Blend..."
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18





# ===== SentiNet防御 =====
echo "5.11.1 SentiNet防御 - CIFAR-10 Blend..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18


echo "5.12.1 SentiNet防御 - CIFAR-10 Adaptive-Patch..."
echo "5.12.2 SentiNet防御 - Tiny ImageNet Adaptive-Patch..."
echo "5.12.3 SentiNet防御 - MNIST-M Adaptive-Patch..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001


echo "5.13.1 SentiNet防御 - CIFAR-10 SIG..."
echo "5.13.2 SentiNet防御 - Tiny ImageNet SIG..."
echo "5.13.3 SentiNet防御 - MNIST-M SIG..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18


echo "5.WaNet.1 SentiNet防御 - CIFAR-10 WaNet..."
echo "5.WaNet.2 SentiNet防御 - Tiny ImageNet WaNet..."
echo "5.WaNet.3 SentiNet防御 - MNIST-M WaNet..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18






echo "5.Adaptive-Blend.1 SentiNet防御 - CIFAR-10 Adaptive-Blend..."
echo "5.Adaptive-Blend.2 SentiNet防御 - Tiny ImageNet Adaptive-Blend..."
echo "5.Adaptive-Blend.3 SentiNet防御 - MNIST-M Adaptive-Blend..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18


# ===== ScaleUp防御 =====
echo "5.19.1 ScaleUp防御 - CIFAR-10 Blend..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18


echo "5.20.1 ScaleUp防御 - CIFAR-10 Adaptive-Patch..."
echo "5.20.2 ScaleUp防御 - Tiny ImageNet Adaptive-Patch..."
echo "5.20.3 ScaleUp防御 - MNIST-M Adaptive-Patch..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001


echo "5.21.1 ScaleUp防御 - CIFAR-10 SIG..."
echo "5.21.2 ScaleUp防御 - Tiny ImageNet SIG..."
echo "5.21.3 ScaleUp防御 - MNIST-M SIG..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18


echo "5.22.1 ScaleUp防御 - CIFAR-10 WaNet..."
echo "5.22.2 ScaleUp防御 - Tiny ImageNet WaNet..."
echo "5.22.3 ScaleUp防御 - MNIST-M WaNet..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18






echo "5.25.1 ScaleUp防御 - CIFAR-10 Adaptive-Blend..."
echo "5.25.2 ScaleUp防御 - Tiny ImageNet Adaptive-Blend..."
echo "5.25.3 ScaleUp防御 - MNIST-M Adaptive-Blend..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18


# ===== STRIP防御 =====



echo "5.27 防御测试 - STRIP防御 Blend..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18


echo "5.28 防御测试 - STRIP防御 Adaptive-Patch..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001


echo "5.29 防御测试 - STRIP防御 SIG..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18


echo "5.30 防御测试 - STRIP防御 WaNet..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18
  

python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18





echo "5.33 防御测试 - STRIP防御 Adaptive-Blend..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18


# ===== IBD-PSC防御 =====
echo "5.35.1 IBD-PSC防御 - Blend CIFAR-10..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=blend -poison_rate=0.03 -alpha=0.151 -model=resnet18


echo "5.36.1 IBD-PSC防御 - Adaptive-Patch CIFAR-10..."
echo "5.36.2 IBD-PSC防御 - Adaptive-Patch Tiny ImageNet..."
echo "5.36.3 IBD-PSC防御 - Adaptive-Patch MNIST-M..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.003 -cover_rate=0.006 -model=resnet18 -alpha=0.001


echo "5.37.1 IBD-PSC防御 - SIG CIFAR-10..."
echo "5.37.2 IBD-PSC防御 - SIG Tiny ImageNet..."
echo "5.37.3 IBD-PSC防御 - SIG MNIST-M..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=SIG -poison_rate=0.02 -delta=33 -f=6 -model=resnet18


echo "5.38.1 IBD-PSC防御 - WaNet CIFAR-10..."
echo "5.38.2 IBD-PSC防御 - WaNet Tiny ImageNet..."
echo "5.38.3 IBD-PSC防御 - WaNet MNIST-M..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.1 -cover_rate=0.2 -s=0.51 -k=4 -model=resnet18






echo "5.41.1 IBD-PSC防御 - Adaptive-Blend CIFAR-10..."
echo "5.41.2 IBD-PSC防御 - Adaptive-Blend Tiny ImageNet..."
echo "5.41.3 IBD-PSC防御 - Adaptive-Blend MNIST-M..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.21 -model=resnet18


echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "错误日志位置: $ERROR_LOG"
echo ""
echo "查看错误日志命令："
echo "  cat $ERROR_LOG           # 查看所有错误"
echo "  tail -n 50 $ERROR_LOG    # 查看最后50行错误"
echo ""
if [ -s "$ERROR_LOG" ]; then
    echo "⚠️  发现错误！错误数量: $(wc -l < "$ERROR_LOG")"
    echo "最近的错误："
    tail -n 10 "$ERROR_LOG"
else
    echo "✅ 没有发现错误！"
fi
echo "=========================================="
