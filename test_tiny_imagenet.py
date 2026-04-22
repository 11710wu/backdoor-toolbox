import numpy as np
import torch
import os
from torchvision import transforms, datasets
import argparse
import random
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from PIL import Image
from utils import supervisor, tools, default_args
from utils.densenet import DenseNetWrapper
import config
import time
from torchvision.utils import save_image

# =============================================================================
# Tiny ImageNet 后门攻击测试函数
# =============================================================================

def test_tiny_imagenet_model(model, test_loader, poison_transform, poison_type, num_classes, 
                             target_class, source_dataset='cifar10', save_example=True, model_dir=None,
                             corruption_type=None, severity=None, test_alpha=None, test_s=None, test_delta=None):
    """
    在 Tiny ImageNet-C 数据集上测试后门攻击效果（跨数据集迁移）
    
    处理流程：
    1. Tiny ImageNet-C 图像原始是 64×64（与 Tiny ImageNet 训练集一致）
    2. 如果源数据集是 tiny_imagenet，直接使用 64×64 图像（无需 resize）
    3. 如果源数据集是其他数据集（如 CIFAR-10），将图像 resize 到 32×32
    4. 在图像上添加触发器
    5. 测试模型性能和攻击成功率
    
    Args:
        model (torch.nn.Module): 已训练的目标模型（在源数据集上训练）
        test_loader (torch.utils.data.DataLoader): Tiny ImageNet-C 测试数据加载器
        poison_transform: 后门攻击变换器
        poison_type (str): 攻击类型名称
        num_classes (int): 源数据集的类别数量（如 CIFAR-10 为 10）
        target_class (int): 后门攻击的目标类别
        source_dataset (str): 源数据集名称（如 'cifar10'），用于确定输入尺寸
        save_example (bool, optional): 是否保存示例图片，默认为True
        model_dir (str, optional): 模型目录路径，用于保存结果文件
        
    Returns:
        tuple: (准确率, 攻击成功率, 输入尺寸)
    """
    print(f"执行 Tiny ImageNet-C 测试: 攻击类型={poison_type}, 源数据集={source_dataset}")
    
    # 根据源数据集确定输入尺寸
    # 注意：Tiny ImageNet-C 图像原始是 64×64
    # 如果源数据集是 tiny_imagenet，模型期望 64×64 输入（与 Tiny ImageNet-C 一致，无需 resize）
    # 其他数据集（如 CIFAR-10）期望 32×32 输入，需要将 Tiny ImageNet-C 的 64×64 图像 resize 到 32×32
    if source_dataset == 'tiny_imagenet':
        input_size = 64  # Tiny ImageNet 使用 64×64 尺寸（与 Tiny ImageNet-C 一致）
    elif source_dataset in ['cifar10', 'cifar100', 'gtsrb']:
        input_size = 32  # CIFAR-10 等使用 32×32（需要从 64×64 resize）
    elif source_dataset in ['imagenette', 'imagenet']:
        input_size = 224  # ImageNet 使用 224×224（需要从 64×64 resize）
    else:
        input_size = 32  # 默认使用 32x32（需要从 64×64 resize）
    
    print(f"输入尺寸: {input_size}x{input_size} (源数据集: {source_dataset}, Tiny ImageNet-C 原始: 64×64)")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 计算准确率（不加触发器）
    # 注意：本脚本主要用于在 Tiny ImageNet 上训练的模型在 Tiny ImageNet-C 上测试
    # 类别空间匹配（都是 200 类），准确率计算有意义
    print(f"\n=== 计算准确率（Tiny ImageNet-C，不加触发器）===")
    print(f"模型在 {source_dataset} 上训练（{num_classes} 个类别，输出范围: 0-{num_classes-1}）")
    print(f"Tiny ImageNet-C 有 200 个类别（标签范围: 0-199）")
    
    if source_dataset == 'tiny_imagenet' and num_classes == 200:
        print(f"类别空间匹配（都是 200 类），准确率计算有意义")
    elif source_dataset != 'tiny_imagenet':
        print(f"注意: 这是跨数据集迁移测试，类别空间可能不匹配")
    
    acc_correct = 0
    acc_total = 0
    # 统计预测类别分布（用于调试）
    pred_class_counts = {}
    target_class_counts = {}

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            # 注意：图像尺寸已在数据变换阶段处理（transforms.Resize）
            # 如果 source_dataset == 'tiny_imagenet'，数据保持 64×64
            # 如果 source_dataset 是其他数据集，数据已在 transforms 中 resize 到 32×32
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            # 统计预测和真实类别
            for p in pred.squeeze().cpu().numpy():
                pred_class_counts[p] = pred_class_counts.get(p, 0) + 1
            for t in target.cpu().numpy():
                target_class_counts[t] = target_class_counts.get(t, 0) + 1

            acc_correct += pred.eq(target.view_as(pred)).sum().item()
            acc_total += target.size(0)

    # 计算准确率
    acc = acc_correct / acc_total if acc_total > 0 else 0
    print(f"Tiny ImageNet-C 总体准确率: {acc:.6f} ({acc*100:.2f}%)")
    print(f"正确预测: {acc_correct} / {acc_total}")
    
    # 详细的类别分布统计
    if pred_class_counts:
        print(f"预测类别范围: {min(pred_class_counts.keys())} - {max(pred_class_counts.keys())} (模型输出 {num_classes} 个类别)")
        print(f"预测类别分布（前10个）: {dict(sorted(pred_class_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
    if target_class_counts:
        print(f"真实类别范围: {min(target_class_counts.keys())} - {max(target_class_counts.keys())} (Tiny ImageNet-C 有 200 个类别)")
        print(f"真实类别分布（前10个）: {dict(sorted(target_class_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
    
    # 检查是否有类别映射问题
    if source_dataset == 'tiny_imagenet' and num_classes == 200:
        # 检查预测类别是否都在有效范围内
        invalid_preds = [p for p in pred_class_counts.keys() if p < 0 or p >= num_classes]
        if invalid_preds:
            print(f"警告: 发现无效的预测类别: {invalid_preds}")
        # 检查真实类别是否都在有效范围内
        invalid_targets = [t for t in target_class_counts.keys() if t < 0 or t >= 200]
        if invalid_targets:
            print(f"警告: 发现无效的真实类别: {invalid_targets}")
    
    # 测试攻击成功率
    print(f"\n=== 测试攻击成功率 ===")
    
    # 初始化攻击成功率统计变量
    attack_success = 0  # 攻击成功的样本数
    attack_total = 0    # 非目标类别的总样本数
    
    # 使用torch.no_grad()禁用梯度计算
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # 将数据移动到GPU上
            data, target = data.cuda(), target.cuda()
            
            # 注意：图像尺寸已在数据变换阶段处理（transforms.Resize）
            # 如果 source_dataset == 'tiny_imagenet'，数据保持 64×64
            # 如果 source_dataset 是其他数据集，数据已在 transforms 中 resize 到 32×32
            
            # 添加触发器（poison_transform 会处理归一化后的数据）
            data_poisoned, _ = poison_transform.transform(data, target)
            
            # 保存示例图片（仅第一个batch）
            if save_example and batch_idx == 0:
                original_img = data[0].clone()
                poisoned_img = data_poisoned[0].clone()
                
                if model_dir:
                    # 构建文件名，包含测试参数信息以区分不同的测试运行
                    method_name_parts = [f"tiny_imagenet_test_{source_dataset}_{input_size}x{input_size}"]
                    if corruption_type:
                        method_name_parts.append(f"corruption={corruption_type}")
                    if severity is not None:
                        method_name_parts.append(f"severity={severity}")
                    if test_alpha is not None:
                        method_name_parts.append(f"test_alpha={test_alpha}")
                    elif test_s is not None:
                        method_name_parts.append(f"test_s={test_s}")
                    elif test_delta is not None:
                        method_name_parts.append(f"test_delta={test_delta}")
                    
                    method_name = "_".join(method_name_parts)
                    save_poisoned_example_to_dir(original_img, poisoned_img, 
                                                poison_type, 
                                                method_name, 
                                                model_dir, source_dataset)
            
            # 预测
            output = model(data_poisoned)
            pred = output.argmax(dim=1, keepdim=True)
            
            # 计算攻击成功率（只计算非目标类别的样本）
            non_target_mask = (target != target_class)
            if non_target_mask.sum() > 0:
                attack_success += (pred.squeeze() == target_class)[non_target_mask].sum().item()
                attack_total += non_target_mask.sum().item()
    
    # 计算攻击成功率
    asr = attack_success / attack_total if attack_total > 0 else 0
    
    print(f"攻击成功率: {asr:.6f} ({asr*100:.2f}%)")
    
    return acc, asr, input_size


def save_poisoned_example_to_dir(original_img, poisoned_img, attack_type, method_name, model_dir, source_dataset='cifar10'):
    """
    保存中毒示例图片到指定模型目录
    
    Args:
        original_img (torch.Tensor): 原始图像tensor
        poisoned_img (torch.Tensor): 中毒后的图像tensor
        attack_type (str): 攻击类型名称
        method_name (str): 测试方法名称
        model_dir (str): 模型目录路径
        source_dataset (str): 源数据集名称，用于确定归一化参数
    """
    # 创建保存目录
    save_dir = os.path.join(model_dir, "tiny_imagenet_examples")
    os.makedirs(save_dir, exist_ok=True)
    
    # UPGD/BELT examples are already in raw [0,1] space.
    # Applying an extra denormalization here causes shifted colors.
    if attack_type in ('upgd', 'belt'):
        original_denorm = torch.clamp(original_img, 0, 1)
        poisoned_denorm = torch.clamp(poisoned_img, 0, 1)
    else:
        # 根据源数据集选择归一化参数
        # 注意: Tiny ImageNet 使用 ImageNet 归一化（语义上更接近 ImageNet）
        if source_dataset in ['cifar10', 'cifar100', 'gtsrb']:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).cuda()
            std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1).cuda()
        elif source_dataset in ['imagenette', 'imagenet', 'tiny_imagenet']:
            # Tiny ImageNet 使用 ImageNet 归一化（是 ImageNet 的子集）
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
        else:
            # 默认使用 ImageNet 的归一化参数
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()

        original_denorm = torch.clamp(original_img * std + mean, 0, 1)
        poisoned_denorm = torch.clamp(poisoned_img * std + mean, 0, 1)
    
    # 计算差异图
    diff = torch.abs(poisoned_denorm - original_denorm)
    diff = diff / (diff.max() + 1e-8)
    
    # 保存图片
    save_image(original_denorm, os.path.join(save_dir, f"{attack_type}_{method_name}_original.png"))
    save_image(poisoned_denorm, os.path.join(save_dir, f"{attack_type}_{method_name}_poisoned.png"))
    save_image(diff, os.path.join(save_dir, f"{attack_type}_{method_name}_diff.png"))
    
    print(f"示例图片已保存到模型目录: {save_dir}")


# =============================================================================
# 创建poison transform
# =============================================================================

def create_poison_transform(args, data_transform, source_dataset='cifar10'):
    """
    创建poison transform（根据源数据集创建对应的变换器）
    
    Args:
        args: 命令行参数
        data_transform: 数据变换
        source_dataset: 源数据集名称
        
    Returns:
        poison_transform: poison transform对象
    """
    # ========== [修改] 在创建 poison_transform 之前，如果指定了 test_s 或 test_delta，临时修改 args 中的值 ==========
    # 保存原始值到 args 对象中（用于后续结果记录）
    if args.poison_type == 'WaNet':
        args.original_s = getattr(args, 's', None)  # 保存训练时的 s 值
        # 如果指定了 test_s，临时修改 args.s（用于测试）
        if args.test_s is not None:
            args.s = args.test_s
    elif args.poison_type == 'SIG':
        args.original_delta = getattr(args, 'delta', None)  # 保存训练时的 delta 值
        # 如果指定了 test_delta，临时修改 args.delta（用于测试）
        if args.test_delta is not None:
            args.delta = args.test_delta
    # ========== [修改结束] ==========
    
    # 获取触发器名称
    trigger_name = args.trigger if args.trigger is not None else args.trigger_path
    
    # ========== [UPGD/BELT 特殊处理] 强制不使用归一化（与原始代码一致）==========
    # UPGD: 原始代码（parameter_backdoor）全程不使用 Normalize
    # BELT: 原始代码（BadNet_BELT.py）全程不使用 Normalize
    if args.poison_type == 'upgd':
        is_normalized = False  # UPGD 强制不归一化
    elif args.poison_type == 'belt':
        is_normalized = False  # BELT 强制不归一化
    else:
        is_normalized = not args.no_normalize if hasattr(args, 'no_normalize') else True
    
    # 创建 poison transform（使用源数据集的配置）
    poison_transform = supervisor.get_poison_transform(
        poison_type=args.poison_type, 
        dataset_name=source_dataset,  # 使用源数据集名称
        target_class=config.target_class[source_dataset],  # 使用源数据集的目标类
        trigger_transform=data_transform,
        is_normalized_input=is_normalized,
        alpha=args.alpha if args.test_alpha is None else args.test_alpha,
        trigger_name=trigger_name, 
        args=args
    )
    
    return poison_transform


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    主函数：Tiny ImageNet-C 后门攻击测试程序入口（跨数据集迁移）
    
    本函数负责：
    1. 解析命令行参数
    2. 加载预训练模型（在源数据集上训练）
    3. 准备 Tiny ImageNet-C 测试数据集
    4. 创建后门攻击变换器（基于源数据集）
    5. 执行测试并保存结果
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Tiny ImageNet-C 后门攻击测试工具（跨数据集迁移）')
    
    # ===== 基本参数 =====
    # 工程改动：不再强制要求 -source_dataset
    # - 本脚本常见用法是 tiny_imagenet -> tiny_imagenet-c，因此默认值设为 tiny_imagenet
    # - 若你要做 cifar10 -> tiny_imagenet-c 等迁移测试，仍可显式传入 -source_dataset
    parser.add_argument('-source_dataset', type=str, required=False, default='tiny_imagenet',
                        choices=['cifar10', 'cifar100', 'gtsrb', 'imagenette', 'tiny_imagenet'],
                        help='源数据集名称（模型训练的数据集）。默认 tiny_imagenet')
    parser.add_argument('-dataset', type=str, required=False,
                        default='tiny_imagenet',
                        choices=default_args.parser_choices['dataset'],
                        help='目标数据集名称（测试的数据集）')
    parser.add_argument('-poison_type', type=str, required=False,
                        choices=default_args.parser_choices['poison_type'],
                        default=default_args.parser_default['poison_type'],
                        help='后门攻击类型')
    parser.add_argument('-poison_rate', type=float, required=False,
                        choices=default_args.parser_choices['poison_rate'],
                        default=default_args.parser_default['poison_rate'],
                        help='中毒样本比例')
    parser.add_argument('-cover_rate', type=float, required=False,
                        choices=default_args.parser_choices['cover_rate'],
                        default=default_args.parser_default['cover_rate'],
                        help='触发器覆盖率')
    parser.add_argument('-alpha', type=float, required=False,
                        default=default_args.parser_default['alpha'],
                        help='攻击强度参数')
    parser.add_argument('-test_alpha', type=float, required=False, default=None,
                        help='测试时的alpha参数（覆盖训练时的alpha）')
    # ========== [修改] 添加 test_s 和 test_delta 参数 ==========
    parser.add_argument('-test_s', type=float, required=False, default=None,
                        help='测试时的WaNet s参数（覆盖训练时的s）')
    parser.add_argument('-test_delta', type=float, required=False, default=None,
                        help='测试时的SIG delta参数（覆盖训练时的delta）')
    # ========== [修改结束] ==========
    
    # ===== 触发器参数 =====
    parser.add_argument('-trigger', type=str, required=False, default=None,
                        help='触发器名称')
    parser.add_argument('-trigger_path', type=str, default=None,
                        help='自定义触发器文件路径')
    
    # ===== 模型参数 =====
    parser.add_argument('-model', type=str, required=False, default=None,
                        choices=['resnet18', 'vgg19_bn', 'mobilenetv2'],
                        help='模型架构选择（覆盖config.py中的默认设置）')
    parser.add_argument('-model_path', required=False, default=None,
                        help='模型文件路径')
    parser.add_argument('-cleanser', type=str, required=False, default=None,
                        choices=default_args.parser_choices['cleanser'],
                        help='数据清洗方法')
    parser.add_argument('-defense', type=str, required=False, default=None,
                        choices=default_args.parser_choices['defense'],
                        help='防御方法')
    
    # ===== 数据处理参数 =====
    parser.add_argument('-no_normalize', default=False, action='store_true',
                        help='禁用数据标准化')
    parser.add_argument('-no_aug', default=False, action='store_true',
                        help='禁用数据增强')
    
    # ===== Tiny ImageNet-C 参数 =====
    # ========== [强制使用 Tiny ImageNet-C] 只测试 Tiny ImageNet-C ==========
    # 默认使用 gaussian_noise（高斯噪声），严重程度 4，这是最简单直观的损坏类型
    parser.add_argument('-corruption_type', type=str, required=False, default='gaussian_noise',
                        choices=['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 
                                'fog', 'frost', 'gaussian_noise', 'glass_blur', 'impulse_noise',
                                'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise',
                                'snow', 'zoom_blur'],
                        help='Tiny ImageNet-C 损坏类型（默认: gaussian_noise）')
    parser.add_argument('-severity', type=int, required=False, default=4,
                        choices=[1, 2, 3, 4, 5],
                        help='Tiny ImageNet-C 损坏严重程度（1-5，默认4）')
    # ========== [强制使用 Tiny ImageNet-C] 结束 ==========
    
    # ===== 系统参数 =====
    parser.add_argument('-devices', type=str, default='0',
                        help='使用的GPU设备ID')
    parser.add_argument('-seed', type=int, required=False, default=default_args.seed,
                        help='随机种子')
    
    # ===== 攻击特定参数 =====
    parser.add_argument('-delta', type=float, default=30,
                        help='SIG攻击delta参数，会自动除以255 (默认30，即30/255)')
    parser.add_argument('-f', type=float, default=6,
                        help='SIG攻击频率参数 (默认6)')
    parser.add_argument('-s', type=float, default=0.5,
                        help='WaNet攻击s参数 (默认0.5)')
    parser.add_argument('-k', type=int, default=4,
                        help='WaNet攻击k参数 (默认4)')

    # ========== [UPGD 参数] 开始 ==========
    # 用于定位 poison_set_dir / model_path（目录名包含 eps/constraint/steps）
    parser.add_argument('-eps', type=float, required=False, default=8.0,
                        help='UPGD eps（与 create_poisoned_set.py 保持一致）')
    parser.add_argument('-constraint', type=str, required=False, default='Linf',
                        choices=['Linf', 'L2'], help='UPGD 约束类型（与 create_poisoned_set.py 保持一致）')
    parser.add_argument('-upgd_steps', type=int, required=False, default=100,
                        help='UPGD steps（与 create_poisoned_set.py 保持一致，用于定位数据/模型目录）')
    parser.add_argument('-upgd_steps_multiplier', type=int, required=False, default=5,
                        help='UPGD steps_multiplier（与 create_poisoned_set.py 保持一致，用于定位数据/模型目录）')
    # ========== [UPGD 参数] 结束 ==========
    # ========== [BELT 参数] 开始 ==========
    parser.add_argument('-mask_rate', type=float, required=False, default=0.2,
                        help='BELT cover samples 的 mask 比例（默认 0.2）')
    # ========== [BELT 参数] 结束 ==========
    
    # 解析命令行参数
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
    
    # 设置默认触发器（使用源数据集的触发器配置）
    source_dataset = args.source_dataset
    if args.trigger is None and args.trigger_path is None:
        args.trigger = config.trigger_default[source_dataset][args.poison_type]
    elif args.trigger is None and args.trigger_path is not None:
        args.trigger = os.path.basename(args.trigger_path)
    
    # 设置数据集参数
    if source_dataset == 'cifar10' or source_dataset == 'cifar100':
        num_classes = 10 if source_dataset == 'cifar10' else 100
        batch_size = 128
    elif source_dataset == 'gtsrb':
        num_classes = 43
        batch_size = 128
    elif source_dataset == 'tiny_imagenet':
        num_classes = 200
        batch_size = 64  # Tiny ImageNet 图像较大，减小 batch size
    elif source_dataset == 'imagenette':
        num_classes = 10
        batch_size = 64
    else:
        print(f'不支持的源数据集: {source_dataset}')
        raise NotImplementedError(f'不支持的源数据集: {source_dataset}')
    
    # 保存原始参数并准备获取模型路径
    original_dataset = args.dataset
    original_alpha = args.alpha
    if args.poison_type == 'WaNet':
        original_s = getattr(args, 's', None)
        args.original_s = original_s
        # 如果指定了 test_s，临时恢复原始值以获取正确的模型路径
        if args.test_s is not None:
            args.s = original_s
    elif args.poison_type == 'SIG':
        original_delta = getattr(args, 'delta', None)
        args.original_delta = original_delta
        # 如果指定了 test_delta，临时恢复原始值以获取正确的模型路径
        if args.test_delta is not None:
            args.delta = original_delta
    
    # 如果指定了 test_alpha，临时恢复原始值以获取正确的模型路径
    if args.test_alpha is not None:
        args.alpha = original_alpha
    
    # 临时修改 args.dataset 为源数据集，以获取正确的模型路径和架构
    args.dataset = source_dataset
    
    # 获取模型路径（基于源数据集的参数）
    model_path = supervisor.get_model_dir(args, cleanse=(args.cleanser is not None), 
                                         defense=(args.defense is not None))
    
    # 获取模型架构（基于源数据集）
    arch = supervisor.get_arch(args)
    model = arch(num_classes=num_classes)
    
    # 设置数据变换（使用源数据集的归一化参数）
    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)
    
    # 恢复 args.dataset 和测试参数
    args.dataset = original_dataset
    if args.test_alpha is not None:
        args.alpha = args.test_alpha
    if args.poison_type == 'WaNet' and args.test_s is not None:
        args.s = args.test_s
    if args.poison_type == 'SIG' and args.test_delta is not None:
        args.delta = args.test_delta
    
    # BELT特殊处理：使用增强模型
    if args.poison_type == 'belt':
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        # 替换为belt_aug_model
        base_name = model_name.replace('.pt', '').replace('.pth', '')
        belt_aug_model_path = os.path.join(model_dir, f"{base_name}_belt_aug_model_seed={args.seed}.pt")
        if os.path.exists(belt_aug_model_path):
            print(f"[BELT] 使用增强模型: {belt_aug_model_path}")
            model_path = belt_aug_model_path
        else:
            print(f"[BELT] 警告: 增强模型不存在，尝试使用基础模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"请确保模型已经在 {source_dataset} 上训练并保存到该路径")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 调试：打印前几个键名
    print(f"加载的 state_dict 键名示例（前5个）: {list(state_dict.keys())[:5]}")
    
    # 处理 DataParallel 保存的模型（键名带有 'module.' 前缀）
    if any(key.startswith('module.') for key in state_dict.keys()):
        # 移除 'module.' 前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        print(f"移除 'module.' 前缀后的键名示例（前5个）: {list(state_dict.keys())[:5]}")
    
    # 处理 DenseNetWrapper 的情况
    # DenseNetWrapper 的结构包含：
    # - self.densenet: 原始 densenet 模型
    # - self.features: 从 densenet 提取的 features（引用，不是副本）
    # - self.classifier: 从 densenet 提取的 classifier（引用，不是副本）
    # 
    # DenseNetWrapper 的 state_dict() 会包含所有子模块的键：
    # - densenet.features.xxx (来自 self.densenet)
    # - features.xxx (来自 self.features，但实际上是同一个对象)
    # - classifier.xxx (来自 self.classifier)
    #
    # 关键：如果保存的模型键名是 'densenet.features.xxx'，而模型期望的也是 'densenet.features.xxx'
    # 那么不需要移除前缀，直接使用即可！
    if isinstance(model, DenseNetWrapper):
        # 检查模型期望的键名格式
        model_keys = list(model.state_dict().keys())
        has_densenet_prefix = any('densenet.' in key for key in model_keys)
        state_dict_has_densenet_prefix = any('densenet.' in key for key in state_dict.keys())
        
        print(f"模型期望的键名是否有 'densenet.' 前缀: {has_densenet_prefix}")
        print(f"state_dict 的键名是否有 'densenet.' 前缀: {state_dict_has_densenet_prefix}")
        
        # 如果两者都有前缀，直接使用（不需要移除）
        if has_densenet_prefix and state_dict_has_densenet_prefix:
            print("键名格式匹配（都有 'densenet.' 前缀），直接使用")
            # 不需要做任何转换
        # 如果 state_dict 有前缀但模型期望没有，需要移除
        elif not has_densenet_prefix and state_dict_has_densenet_prefix:
            new_state_dict = {}
            for key, value in state_dict.items():
                # 移除 'densenet.' 前缀
                new_key = key.replace('densenet.', '') if 'densenet.' in key else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            print(f"已移除 'densenet.' 前缀从 state_dict 键名")
        # 如果 state_dict 没有前缀但模型期望有，需要添加
        elif has_densenet_prefix and not state_dict_has_densenet_prefix:
            new_state_dict = {}
            for key, value in state_dict.items():
                # 添加 'densenet.' 前缀
                if key.startswith('features.') or key.startswith('classifier.'):
                    new_key = 'densenet.' + key
                else:
                    new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            print(f"已添加 'densenet.' 前缀到 state_dict 键名")
        # 如果两者都没有前缀，直接使用
        else:
            print("键名格式匹配（都没有 'densenet.' 前缀），直接使用")
    
    # 调试：打印模型期望的键名
    print(f"模型期望的键名示例（前5个）: {list(model.state_dict().keys())[:5]}")
    print(f"state_dict 中的键名示例（前5个）: {list(state_dict.keys())[:5]}")
    
    # 尝试加载 state_dict
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"警告: 以下键在模型中缺失: {missing_keys[:5]}... (共 {len(missing_keys)} 个)")
            # 如果缺失键太多，可能是键名映射问题
            if len(missing_keys) > 100:
                print(f"⚠️  警告: 缺失键过多（{len(missing_keys)} 个），可能是键名映射问题！")
                print(f"   建议检查模型保存和加载时的结构是否一致")
        if unexpected_keys:
            print(f"警告: 以下键在 state_dict 中多余: {unexpected_keys[:5]}... (共 {len(unexpected_keys)} 个)")
    except RuntimeError as e:
        print(f"错误: 加载 state_dict 失败: {e}")
        raise
    
    print(f"模型加载成功: {model_path}")
    print(f"源数据集: {source_dataset}, 目标数据集: {args.dataset}")
    print(f"模型类别数: {num_classes}")
    
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()  # 确保模型处于评估模式
    print("Evaluating model '{}'...".format(model_path))
    
    # 加载 Tiny ImageNet-C 数据集（强制使用）
    print("加载 Tiny ImageNet-C 数据集...")
    
    # ========== [强制使用 Tiny ImageNet-C] 只测试 Tiny ImageNet-C ==========
    # ========== [默认配置] 如果未指定 corruption_type，使用默认值 ==========
    if args.corruption_type is None:
        args.corruption_type = 'gaussian_noise'  # 默认使用 gaussian_noise
        print(f"[提示] 未指定 corruption_type，使用默认值: {args.corruption_type}")
    # ========== [默认配置] 结束 ==========
    
    # Tiny ImageNet-C 路径结构: Tiny-ImageNet-C/corruption_type/severity/
    test_set_path = os.path.join(config.tiny_imagenet_c_dir, args.corruption_type, str(args.severity))
    
    if not os.path.exists(test_set_path):
        print(f"错误: Tiny ImageNet-C 路径不存在: {test_set_path}")
        raise FileNotFoundError(f"Tiny ImageNet-C 路径不存在: {test_set_path}")
    
    print(f"使用 Tiny ImageNet-C: {args.corruption_type}, 严重程度: {args.severity}")
    print(f"数据路径: {test_set_path}")
    # ========== [强制使用 Tiny ImageNet-C] 结束 ==========
    
    # 创建数据变换（根据源数据集调整尺寸和归一化）
    # 注意：Tiny ImageNet-C 图像原始是 64×64（与 Tiny ImageNet 训练集一致）
    if source_dataset == 'tiny_imagenet':
        if args.poison_type in ('upgd', 'belt'):
            # UPGD/BELT 训练时不使用 Normalize，测试时也必须保持一致
            tiny_imagenet_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            tiny_imagenet_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
            ])
    else:
        # 其他数据集：需要将 64×64 resize 到 32×32 + 对应的归一化参数
        if args.poison_type in ('upgd', 'belt'):
            tiny_imagenet_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        else:
            tiny_imagenet_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
    
    test_set = datasets.ImageFolder(test_set_path, transform=tiny_imagenet_transform)
    print(f"Tiny ImageNet-C 数据集加载完成: {len(test_set)} 张图像")
    print(f"Tiny ImageNet-C 类别数: {len(test_set.classes)}")
    print(f"数据变换: {tiny_imagenet_transform}")
    
    # 验证第一个样本的数据范围（用于调试）
    if len(test_set) > 0:
        sample_img, sample_label = test_set[0]
        print(f"样本数据范围: min={sample_img.min():.4f}, max={sample_img.max():.4f}, mean={sample_img.mean():.4f}")
        print(f"样本标签: {sample_label} (类别: {test_set.classes[sample_label] if sample_label < len(test_set.classes) else 'N/A'})")
    
    # 创建数据加载器
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, 
        num_workers=4, pin_memory=True)
    
    # 创建poison transform（基于源数据集）
    poison_transform = create_poison_transform(args, data_transform, source_dataset=source_dataset)
    
    # 执行 Tiny ImageNet-C 测试
    print("开始执行 Tiny ImageNet-C 测试...")
    target_class = config.target_class[source_dataset]  # 使用源数据集的目标类
    
    # 获取模型目录路径
    model_dir = os.path.dirname(model_path)
    
    acc, asr, input_size = test_tiny_imagenet_model(
        model=model, 
        test_loader=test_set_loader, 
        poison_transform=poison_transform,
        poison_type=args.poison_type, 
        num_classes=num_classes, 
        target_class=target_class,
        source_dataset=source_dataset,
        model_dir=model_dir,
        corruption_type=args.corruption_type,
        severity=args.severity,
        test_alpha=args.test_alpha,
        test_s=args.test_s,
        test_delta=args.test_delta
    )
    
    # ========== 按损坏类型、强度及攻击参数，单独保存结果文件 ==========
    # 文件名始终包含 corruption_type 和 severity，便于不同损坏/强度单独保存
    suffix = f"corruption={args.corruption_type}_severity={args.severity}"
    if args.poison_type == 'SIG':
        if args.test_delta is not None:
            delta_value = args.test_delta
            base_name = f'test_tiny_imagenet_results_test_delta={delta_value}_{suffix}.txt'
        else:
            delta_value = args.delta if hasattr(args, 'delta') else 30
            base_name = f'test_tiny_imagenet_results_delta={delta_value}_{suffix}.txt'
    elif args.poison_type == 'WaNet':
        if args.test_s is not None:
            s_value = args.test_s
            base_name = f'test_tiny_imagenet_results_test_s={s_value}_{suffix}.txt'
        else:
            s_value = args.s if hasattr(args, 's') else 0.5
            base_name = f'test_tiny_imagenet_results_s={s_value}_{suffix}.txt'
    elif args.poison_type in ['blend', 'adaptive_blend', 'adaptive_patch', 'basic', 'clean_label']:
        if args.test_alpha is not None:
            base_name = f'test_tiny_imagenet_results_test_alpha={args.test_alpha}_{suffix}.txt'
        else:
            base_name = f'test_tiny_imagenet_results_{suffix}.txt'
    else:
        base_name = f'test_tiny_imagenet_results_{suffix}.txt'
    test_result_path = os.path.join(model_dir, base_name)
    # ========== [修改结束] ==========
    
    with open(test_result_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Tiny ImageNet-C 测试结果（跨数据集迁移）===\n")
        f.write(f"源数据集: {source_dataset}\n")
        f.write(f"目标数据集: {args.dataset}\n")
        f.write(f"测试数据集: Tiny ImageNet-C\n")
        f.write(f"损坏类型: {args.corruption_type}\n")
        f.write(f"严重程度: {args.severity}\n")
        f.write(f"攻击类型: {args.poison_type}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试方法: Tiny ImageNet-C 图像原始是 64×64\n")
        if source_dataset == 'tiny_imagenet':
            f.write(f"输入尺寸: {input_size}×{input_size}（保持 64×64，与训练集一致）\n")
        elif input_size != 64:
            f.write(f"输入尺寸: {input_size}×{input_size}（从 64×64 resize 到 {input_size}×{input_size}）\n")
            f.write(f"插值方法: bicubic\n")
        else:
            f.write(f"输入尺寸: {input_size}×{input_size}（保持 64×64）\n")
        f.write(f"目标类别: {target_class}\n")
        
        # 保存攻击参数
        if hasattr(args, 'alpha') and args.alpha is not None:
            f.write(f"训练时Alpha: {args.alpha}\n")
        if hasattr(args, 'test_alpha') and args.test_alpha is not None:
            f.write(f"测试时Alpha (test_alpha): {args.test_alpha}\n")
        elif hasattr(args, 'alpha') and args.alpha is not None:
            f.write(f"测试时Alpha: {args.alpha} (使用训练时的alpha)\n")
        if hasattr(args, 'trigger') and args.trigger is not None:
            f.write(f"触发器: {args.trigger}\n")
        
        # 保存特定攻击类型参数
        if args.poison_type == 'WaNet':
            if args.test_s is not None:
                train_s = getattr(args, 'original_s', None)
                if train_s is not None:
                    f.write(f"训练时WaNet s参数: {train_s}\n")
                f.write(f"测试时WaNet s参数 (test_s): {args.test_s}\n")
            else:
                if hasattr(args, 's') and args.s is not None:
                    f.write(f"WaNet s参数: {args.s}\n")
        elif args.poison_type == 'SIG':
            if args.test_delta is not None:
                train_delta = getattr(args, 'original_delta', None)
                if train_delta is not None:
                    f.write(f"训练时SIG delta参数: {train_delta}\n")
                f.write(f"测试时SIG delta参数 (test_delta): {args.test_delta}\n")
            else:
                if hasattr(args, 'delta') and args.delta is not None:
                    f.write(f"SIG delta参数: {args.delta}\n")
        
        f.write(f"\n=== 测试结果 ===\n")
        f.write(f"准确率: {acc:.6f} ({acc:.2%})\n")
        f.write(f"攻击成功率 (ASR): {asr:.6f} ({asr:.2%})\n")
        if source_dataset != 'tiny_imagenet':
            f.write(f"\n说明: 这是跨数据集迁移测试（{source_dataset} → Tiny ImageNet-C）\n")
    
    print(f"测试结果已保存到: {test_result_path}")
    print(f"攻击类型: {args.poison_type}, 准确率: {acc:.6f} ({acc:.2%}), 攻击成功率: {asr:.6f} ({asr:.2%})")

if __name__ == "__main__":
    main()

