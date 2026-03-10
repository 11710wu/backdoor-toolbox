#!/bin/bash

# ==============================================================================
# Tiny ImageNet-C 跨域测试脚本
# 测试所有 15 种损坏类型 × 5 种严重程度 = 75 种组合
# ==============================================================================

# 攻击配置
DATASET="tiny_imagenet"
POISON_TYPE="adaptive_blend"
POISON_RATE="0.005"
COVER_RATE="0.005"
ALPHA="0.15"

# ==============================================================================
# 1. 创建投毒数据集
# ==============================================================================
# echo "===== 1. 创建投毒数据集 ====="

# python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18
# python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn
# python create_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2

# # ==============================================================================
# # 2. 训练后门模型
# # ==============================================================================
# echo "===== 2. 训练后门模型 ====="

# python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18
# python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn
# python train_on_poisoned_set.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2

# ==============================================================================
# 3. 本地测试（干净 Tiny ImageNet）
# ==============================================================================
echo "===== 3. 本地测试 ====="

python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18
python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn
python test_model.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2

# ==============================================================================
# 4. Tiny ImageNet-C 跨域测试
# 损坏类型: brightness, contrast, defocus_blur, elastic_transform, fog, frost,
#          gaussian_noise, glass_blur, impulse_noise, jpeg_compression,
#          motion_blur, pixelate, shot_noise, snow, zoom_blur
# 严重程度: 1, 2, 3, 4, 5
# ==============================================================================
echo "===== 4. Tiny ImageNet-C 跨域测试 ====="

# ---------- ResNet18 ----------
echo "----- ResNet18: Tiny ImageNet-C 测试 -----"

# brightness
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=brightness -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=brightness -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=brightness -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=brightness -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=brightness -severity=5

# contrast
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=contrast -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=contrast -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=contrast -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=contrast -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=contrast -severity=5

# defocus_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=defocus_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=defocus_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=defocus_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=defocus_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=defocus_blur -severity=5

# elastic_transform
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=elastic_transform -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=elastic_transform -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=elastic_transform -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=elastic_transform -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=elastic_transform -severity=5

# fog
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=fog -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=fog -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=fog -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=fog -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=fog -severity=5

# frost
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=frost -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=frost -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=frost -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=frost -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=frost -severity=5

# gaussian_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=gaussian_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=gaussian_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=gaussian_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=gaussian_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=gaussian_noise -severity=5

# glass_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=glass_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=glass_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=glass_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=glass_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=glass_blur -severity=5

# impulse_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=impulse_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=impulse_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=impulse_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=impulse_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=impulse_noise -severity=5

# jpeg_compression
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=jpeg_compression -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=jpeg_compression -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=jpeg_compression -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=jpeg_compression -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=jpeg_compression -severity=5

# motion_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=motion_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=motion_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=motion_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=motion_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=motion_blur -severity=5

# pixelate
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=pixelate -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=pixelate -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=pixelate -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=pixelate -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=pixelate -severity=5

# shot_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=shot_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=shot_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=shot_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=shot_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=shot_noise -severity=5

# snow
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=snow -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=snow -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=snow -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=snow -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=snow -severity=5

# zoom_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=zoom_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=zoom_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=zoom_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=zoom_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=resnet18 -corruption_type=zoom_blur -severity=5

# ---------- VGG19_BN ----------
echo "----- VGG19_BN: Tiny ImageNet-C 测试 -----"

# brightness
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=brightness -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=brightness -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=brightness -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=brightness -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=brightness -severity=5

# contrast
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=contrast -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=contrast -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=contrast -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=contrast -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=contrast -severity=5

# defocus_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=defocus_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=defocus_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=defocus_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=defocus_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=defocus_blur -severity=5

# elastic_transform
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=elastic_transform -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=elastic_transform -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=elastic_transform -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=elastic_transform -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=elastic_transform -severity=5

# fog
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=fog -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=fog -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=fog -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=fog -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=fog -severity=5

# frost
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=frost -severity=5

# gaussian_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=gaussian_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=gaussian_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=gaussian_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=gaussian_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=gaussian_noise -severity=5

# glass_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=glass_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=glass_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=glass_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=glass_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=glass_blur -severity=5

# impulse_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=impulse_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=impulse_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=impulse_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=impulse_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=impulse_noise -severity=5

# jpeg_compression
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=jpeg_compression -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=jpeg_compression -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=jpeg_compression -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=jpeg_compression -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=jpeg_compression -severity=5

# motion_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=motion_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=motion_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=motion_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=motion_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=motion_blur -severity=5

# pixelate
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=pixelate -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=pixelate -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=pixelate -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=pixelate -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=pixelate -severity=5

# shot_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=shot_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=shot_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=shot_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=shot_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=shot_noise -severity=5

# snow
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=snow -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=snow -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=snow -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=snow -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=snow -severity=5

# zoom_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=zoom_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=zoom_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=zoom_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=zoom_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=vgg19_bn -corruption_type=zoom_blur -severity=5

# ---------- MobileNetV2 ----------
echo "----- MobileNetV2: Tiny ImageNet-C 测试 -----"

# brightness
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=brightness -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=brightness -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=brightness -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=brightness -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=brightness -severity=5

# contrast
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=contrast -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=contrast -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=contrast -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=contrast -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=contrast -severity=5

# defocus_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=defocus_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=defocus_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=defocus_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=defocus_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=defocus_blur -severity=5

# elastic_transform
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=elastic_transform -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=elastic_transform -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=elastic_transform -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=elastic_transform -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=elastic_transform -severity=5

# fog
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=fog -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=fog -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=fog -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=fog -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=fog -severity=5

# frost
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=frost -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=frost -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=frost -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=frost -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=frost -severity=5

# gaussian_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=gaussian_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=gaussian_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=gaussian_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=gaussian_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=gaussian_noise -severity=5

# glass_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=glass_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=glass_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=glass_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=glass_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=glass_blur -severity=5

# impulse_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=impulse_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=impulse_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=impulse_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=impulse_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=impulse_noise -severity=5

# jpeg_compression
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=jpeg_compression -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=jpeg_compression -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=jpeg_compression -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=jpeg_compression -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=jpeg_compression -severity=5

# motion_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=motion_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=motion_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=motion_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=motion_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=motion_blur -severity=5

# pixelate
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=pixelate -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=pixelate -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=pixelate -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=pixelate -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=pixelate -severity=5

# shot_noise
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=shot_noise -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=shot_noise -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=shot_noise -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=shot_noise -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=shot_noise -severity=5

# snow
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=snow -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=snow -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=snow -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=snow -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=snow -severity=5

# zoom_blur
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=zoom_blur -severity=1
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=zoom_blur -severity=2
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=zoom_blur -severity=3
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=zoom_blur -severity=4
python test_tiny_imagenet.py -dataset=tiny_imagenet -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -alpha=0.15 -model=mobilenetv2 -corruption_type=zoom_blur -severity=5

echo "=========================================="
echo "Tiny ImageNet-C 测试完成！"
echo "共测试: 3 模型 × 15 损坏类型 × 5 严重程度 = 225 组实验"
echo "=========================================="
