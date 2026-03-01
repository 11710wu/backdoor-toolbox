
echo "=========================================="
echo "4. Adaptive Patch 攻击测试 (alpha: 0.1-0.9)"
echo "=========================================="

echo "启动第一组创建中毒数据集 (alpha: 0.1, 0.2, 0.3)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.01 -cover_rate=0.06&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第一组训练..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第一组STL-10测试..."
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第二组创建中毒数据集 (alpha: 0.4, 0.5, 0.6)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait

echo "启动第二组训练..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait

echo "启动第二组STL-10测试..."
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait


echo "开始五种防御方法测试 - Adaptive Patch..."
echo ""
echo "==========================================="
echo "AC检测模式 - Adaptive Patch"
echo "==========================================="

echo "启动第1组AC检测 (alpha: 0.1, 0.2, 0.3)..."
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第2组AC检测 (alpha: 0.4, 0.5, 0.6)..."
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait



echo "==========================================="
echo "STRIP检测模式 - Adaptive Patch"
echo "==========================================="

echo "启动第1组STRIP检测 (alpha: 0.1, 0.2, 0.3)..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第2组STRIP检测 (alpha: 0.4, 0.5, 0.6)..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait



echo "==========================================="
echo "SCaLe-Up检测模式 - Adaptive Patch"
echo "==========================================="

echo "启动第1组SCaLe-Up检测 (alpha: 0.1, 0.2, 0.3)..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第2组SCaLe-Up检测 (alpha: 0.4, 0.5, 0.6)..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait



echo "==========================================="
echo "SentiNet检测模式 - Adaptive Patch"
echo "==========================================="

echo "启动第1组SentiNet检测 (alpha: 0.1, 0.2, 0.3)..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第2组SentiNet检测 (alpha: 0.4, 0.5, 0.6)..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait


echo "==========================================="
echo "IBD_PSC检测模式 - Adaptive Patch"
echo "==========================================="

echo "启动第1组IBD_PSC检测 (alpha: 0.1, 0.2, 0.3)..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第2组IBD_PSC检测 (alpha: 0.4, 0.5, 0.6)..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait



echo "Adaptive Patch 攻击测试完成（含五种防御方法：AC, STRIP, SCaLe-Up, SentiNet, IBD_PSC）"
echo ""

echo "=========================================="
echo "模型测试 - Adaptive Patch攻击"
echo "=========================================="

echo "启动第1组模型测试 (alpha: 0.1, 0.2, 0.3)..."
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.1 -cover_rate=0.06&
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.2 -cover_rate=0.06&
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.3 -cover_rate=0.06&
wait

echo "启动第2组模型测试 (alpha: 0.4, 0.5, 0.0)..."
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.4 -cover_rate=0.06&
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.5 -cover_rate=0.06&
python test_model.py -dataset=cifar10 -poison_type=adaptive_patch -poison_rate=0.03 -alpha=0.0 -cover_rate=0.06&
wait

echo "Adaptive Patch攻击模型测试完成！"
echo ""


echo "=========================================="
echo "5. Adaptive Blend 攻击测试 (alpha: 0.05-0.3)"
echo "=========================================="

echo "启动第一组创建中毒数据集 (alpha: 0.05, 0.1, 0.3)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第一组训练..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第一组STL-10测试..."
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第二组创建中毒数据集 (alpha: 0.15, 0.2, 0.25)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "启动第二组训练..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "启动第二组STL-10测试..."
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python test_stl10.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "开始五种防御方法测试 - Adaptive Blend..."
echo ""
echo "==========================================="
echo "AC检测模式 - Adaptive Blend"
echo "==========================================="

echo "启动第1组AC检测 (alpha: 0.05, 0.1, 0.3)..."
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第2组AC检测 (alpha: 0.15, 0.2, 0.25)..."
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "==========================================="
echo "STRIP检测模式 - Adaptive Blend"
echo "==========================================="

echo "启动第1组STRIP检测 (alpha: 0.05, 0.1, 0.3)..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第2组STRIP检测 (alpha: 0.15, 0.2, 0.25)..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "==========================================="
echo "SCaLe-Up检测模式 - Adaptive Blend"
echo "==========================================="

echo "启动第1组SCaLe-Up检测 (alpha: 0.05, 0.1, 0.3)..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第2组SCaLe-Up检测 (alpha: 0.15, 0.2, 0.25)..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "==========================================="
echo "SentiNet检测模式 - Adaptive Blend"
echo "==========================================="

echo "启动第1组SentiNet检测 (alpha: 0.05, 0.1, 0.3)..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第2组SentiNet检测 (alpha: 0.15, 0.2, 0.25)..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "==========================================="
echo "IBD_PSC检测模式 - Adaptive Blend"
echo "==========================================="

echo "启动第1组IBD_PSC检测 (alpha: 0.05, 0.1, 0.3)..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第2组IBD_PSC检测 (alpha: 0.15, 0.2, 0.25)..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait


echo "Adaptive Blend 攻击测试完成（含五种防御方法：AC, STRIP, SCaLe-Up, SentiNet, IBD_PSC）"
echo ""

echo "=========================================="
echo "模型测试 - Adaptive Blend攻击"
echo "=========================================="

echo "启动第1组模型测试 (alpha: 0.05, 0.1, 0.3)..."
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.05 -trigger=hellokitty_32.png&
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.1 -trigger=hellokitty_32.png&
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.3 -trigger=hellokitty_32.png&
wait

echo "启动第2组模型测试 (alpha: 0.15, 0.2, 0.25)..."
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.15 -trigger=hellokitty_32.png&
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.2 -trigger=hellokitty_32.png&
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.03 -cover_rate=0.03 -alpha=0.25 -trigger=hellokitty_32.png&
wait

echo "Adaptive Blend攻击模型测试完成！"
echo ""

# ==================== WaNet 攻击测试 ====================
echo "=========================================="
echo "6. WaNet 攻击测试 (s: 0.1-0.5, poison_rate=0.03)"
echo "=========================================="

echo "启动第一组创建中毒数据集 (s: 0.1, 0.15, 0.2)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第一组训练..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第一组STL-10测试..."
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第二组创建中毒数据集 (s: 0.25, 0.3, 0.35)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第二组训练..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第二组STL-10测试..."
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第三组创建中毒数据集 (s: 0.4, 0.45, 0.5)..."
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "启动第三组训练..."
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "启动第三组STL-10测试..."
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python test_stl10.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "开始后门检测 - WaNet攻击..."
echo ""
echo "=========================================="
echo "AC检测模式 - WaNet攻击"
echo "=========================================="

echo "启动第一组AC检测 (s: 0.1, 0.15, 0.2)..."
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第二组AC检测 (s: 0.25, 0.3, 0.35)..."
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第三组AC检测 (s: 0.4, 0.45, 0.5)..."
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python other_defense.py -defense=AC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "AC检测模式完成！"
echo ""
echo "=========================================="
echo "STRIP检测模式 - WaNet攻击"
echo "=========================================="

echo "启动第一组STRIP检测 (s: 0.1, 0.15, 0.2)..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第二组STRIP检测 (s: 0.25, 0.3, 0.35)..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第三组STRIP检测 (s: 0.4, 0.45, 0.5)..."
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python other_defense.py -defense=STRIP -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "STRIP检测模式完成！"
echo ""
echo "=========================================="
echo "SCaLe-Up检测模式 - WaNet攻击"
echo "=========================================="

echo "启动第一组SCaLe-Up检测 (s: 0.1, 0.15, 0.2)..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第二组SCaLe-Up检测 (s: 0.25, 0.3, 0.35)..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第三组SCaLe-Up检测 (s: 0.4, 0.45, 0.5)..."
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python other_defense.py -defense=ScaleUp -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "SCaLe-Up检测模式完成！"
echo ""
echo "=========================================="
echo "SentiNet检测模式 - WaNet攻击"
echo "=========================================="

echo "启动第一组SentiNet检测 (s: 0.1, 0.15, 0.2)..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第二组SentiNet检测 (s: 0.25, 0.3, 0.35)..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第三组SentiNet检测 (s: 0.4, 0.45, 0.5)..."
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python other_defense.py -defense=SentiNet -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "SentiNet检测模式完成！"
echo ""
echo "=========================================="
echo "IBD_PSC检测模式 - WaNet攻击"
echo "=========================================="

echo "启动第一组IBD_PSC检测 (s: 0.1, 0.15, 0.2)..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第二组IBD_PSC检测 (s: 0.25, 0.3, 0.35)..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第三组IBD_PSC检测 (s: 0.4, 0.45, 0.5)..."
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python other_defense.py -defense=IBD_PSC -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "IBD_PSC检测模式完成！"
echo ""


echo "WaNet 攻击测试完成（含五种防御方法：AC, STRIP, SCaLe-Up, SentiNet, IBD_PSC）"
echo ""

echo "=========================================="
echo "模型测试 - WaNet攻击"
echo "=========================================="

echo "启动第一组模型测试 (s: 0.1, 0.15, 0.2)..."
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.1 -k=4&
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.15 -k=4&
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.2 -k=4&
wait

echo "启动第二组模型测试 (s: 0.25, 0.3, 0.35)..."
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.25 -k=4&
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.3 -k=4&
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.35 -k=4&
wait

echo "启动第三组模型测试 (s: 0.4, 0.45, 0.5)..."
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.4 -k=4&
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.45 -k=4&
python test_model.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.03 -cover_rate=0.06 -s=0.5 -k=4&
wait

echo "WaNet攻击模型测试完成！"
echo ""

echo "=========================================="
echo "所有攻击方法测试完成！"
echo "完成时间: $(date)"
echo "=========================================="