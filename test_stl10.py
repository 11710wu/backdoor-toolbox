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
import config
import time
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.utils import save_image

# =============================================================================
# 图像处理工具函数
# =============================================================================

def tensor_resize(tensor, size):
    """
    使用双三次插值调整tensor大小
    
    Args:
        tensor (torch.Tensor): 输入图像tensor，形状为 (N, C, H, W)
        size (int): 目标尺寸，输出图像将为 size x size
        
    Returns:
        torch.Tensor: 调整尺寸后的图像tensor，形状为 (N, C, size, size)
    """
    return F.interpolate(tensor, size=(size, size), mode='bicubic', align_corners=False)


# =============================================================================
# 无需自定义类，直接使用原始poison_transform
# =============================================================================


# =============================================================================
# STL-10后门攻击测试函数（仅32x32）
# =============================================================================

def test_stl10_model(model, test_loader, poison_transform, poison_type, num_classes, 
                     target_class, save_example=True, model_dir=None):
    """
    在STL-10数据集上测试后门攻击效果（仅32x32方法）
    
    处理流程：
    1. 将STL-10的96x96图像下采样到32x32
    2. 在32x32图像上添加触发器
    3. 测试模型性能和攻击成功率
    
    Args:
        model (torch.nn.Module): 已训练的目标模型
        test_loader (torch.utils.data.DataLoader): STL-10测试数据加载器
        poison_transform: 后门攻击变换器
        poison_type (str): 攻击类型名称
        num_classes (int): 数据集类别数量（通常为10）
        target_class (int): 后门攻击的目标类别
        save_example (bool, optional): 是否保存示例图片，默认为True
        model_dir (str, optional): 模型目录路径，用于保存结果文件
        
    Returns:
        tuple: (准确率, 攻击成功率)
    """
    print(f"执行STL-10测试（32x32方法）: 攻击类型={poison_type}")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 首先计算准确率（不加触发器）
    print(f"\n=== 计算准确率（STL-10，不加触发器）===")
    acc_correct = 0
    acc_total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            # 注意：数据已经在 FilteredSTL10 中下采样到 32x32 并归一化了
            # 所以直接使用，不需要再次下采样
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            acc_correct += pred.eq(target.view_as(pred)).sum().item()
            acc_total += target.size(0)

    # 计算准确率
    acc = acc_correct / acc_total if acc_total > 0 else 0
    print(f"STL-10总体准确率: {acc:.6f} ({acc*100:.2f}%)")
    
    # 测试攻击成功率（32x32方法）
    print(f"\n=== 测试攻击成功率（32x32方法）===")
    
    # 初始化攻击成功率统计变量
    attack_success = 0  # 攻击成功的样本数
    attack_total = 0    # 非目标类别的总样本数
    
    # 使用torch.no_grad()禁用梯度计算
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # 将数据移动到GPU上
            data, target = data.cuda(), target.cuda()
            
            # 注意：数据已经在 FilteredSTL10 中下采样到 32x32 并归一化了
            # 所以直接使用，不需要再次下采样
            # 添加触发器（poison_transform 会处理归一化后的数据）
            data_poisoned, _ = poison_transform.transform(data, target)
            
            # 保存示例图片（仅第一个batch）
            if save_example and batch_idx == 0:
                # 保存归一化后的图像（32x32）和添加触发器后的图像（32x32）
                original_32 = data[0].clone()
                final_32 = data_poisoned[0].clone()
                
                if model_dir:
                    save_poisoned_example_to_dir(original_32, final_32, 
                                                poison_type, 
                                                "stl10_test_32x32", model_dir)
                # 注释掉全局目录保存，避免重复
                # save_poisoned_example(original_32, final_32, 
                #                      poison_type, 
                #                      "stl10_test_32x32")
            
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
    
    return acc, asr


def save_poisoned_example_to_dir(original_img, poisoned_img, attack_type, method_name, model_dir):
    """
    保存中毒示例图片到指定模型目录
    
    Args:
        original_img (torch.Tensor): 原始图像tensor
        poisoned_img (torch.Tensor): 中毒后的图像tensor
        attack_type (str): 攻击类型名称
        method_name (str): 测试方法名称
        model_dir (str): 模型目录路径
    """
    # 创建保存目录
    save_dir = os.path.join(model_dir, "stl10_examples")
    os.makedirs(save_dir, exist_ok=True)
    
    # 反归一化图片（使用CIFAR-10的归一化参数）
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).cuda()
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1).cuda()
    
    original_denorm = original_img * std + mean
    poisoned_denorm = poisoned_img * std + mean
    
    # 限制像素值在[0, 1]范围内
    original_denorm = torch.clamp(original_denorm, 0, 1)
    poisoned_denorm = torch.clamp(poisoned_denorm, 0, 1)
    
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

def create_poison_transform(args, data_transform):
    """
    创建poison transform（直接返回标准的32x32变换器）
    
    Args:
        args: 命令行参数
        data_transform: 数据变换
        
    Returns:
        poison_transform: 标准的poison transform对象
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
    
    # 直接创建并返回标准的poison transform（32x32）
    poison_transform = supervisor.get_poison_transform(
        poison_type=args.poison_type, 
        dataset_name=args.dataset,
        target_class=config.target_class[args.dataset], 
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
    主函数：STL-10后门攻击测试程序入口（仅32x32方法）
    
    本函数负责：
    1. 解析命令行参数
    2. 加载预训练模型
    3. 准备STL-10测试数据集
    4. 创建后门攻击变换器（32x32）
    5. 执行测试并保存结果
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='STL-10后门攻击测试工具（32x32方法）')
    
    # ===== 基本参数 =====
    parser.add_argument('-dataset', type=str, required=False,
                        default=default_args.parser_default['dataset'],
                        choices=default_args.parser_choices['dataset'],
                        help='数据集名称')
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
    # ========== [修改] 添加 test_s 和 test_delta 参数，允许测试时使用与训练时不同的参数值 ==========
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
    
    # ===== 系统参数 =====
    parser.add_argument('-devices', type=str, default='0',
                        help='使用的GPU设备ID')
    parser.add_argument('-seed', type=int, required=False, default=default_args.seed,
                        help='随机种子')
    
    # ===== 攻击特定参数 =====
    # ========== [SIG参数修改] 开始 ==========
    # SIG攻击专用参数
    parser.add_argument('-delta', type=float, default=30,
                        help='SIG攻击delta参数，会自动除以255 (默认30，即30/255)')
    parser.add_argument('-f', type=float, default=6,
                        help='SIG攻击频率参数 (默认6)')
    # ========== [SIG参数修改] 结束 ==========
    
    # ========== [WaNet参数修改] 开始 ==========
    # WaNet攻击专用参数
    parser.add_argument('-s', type=float, default=0.5,
                        help='WaNet攻击s参数 (默认0.5)')
    parser.add_argument('-k', type=int, default=4,
                        help='WaNet攻击k参数 (默认4)')
    # ========== [WaNet参数修改] 结束 ==========

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
    # ========== [噪声增强参数] 与 create_poisoned_set.py 一致，用于定位带噪声的模型目录 ==========
    parser.add_argument('-noise_type', type=str, required=False, default=None,
                        choices=['gaussian', 'salt_pepper', 'uniform'],
                        help='噪声类型；与创建投毒集时一致则定位到对应目录，不传则使用无噪声目录')
    # ========== [噪声增强参数] 结束 ==========
    
    # 解析命令行参数
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
    
    # 设置默认触发器
    if args.trigger is None and args.trigger_path is None:
        args.trigger = config.trigger_default[args.dataset][args.poison_type]
    elif args.trigger is None and args.trigger_path is not None:
        args.trigger = os.path.basename(args.trigger_path)
    
    # 设置数据集参数
    if args.dataset == 'cifar10' or args.dataset == 'cifar101':
        num_classes = 10
        batch_size = 128
    else:
        print(f'不支持的数据集: {args.dataset}')
        raise NotImplementedError(f'不支持的数据集: {args.dataset}')
    
    # ========== [修复] 在获取模型路径之前保存原始参数 ==========
    # 模型路径应该基于训练时的参数，而不是测试时的参数
    # 如果使用了 test_alpha/test_s/test_delta，需要先保存原始值，获取模型路径后再修改
    original_alpha = args.alpha
    if args.poison_type == 'WaNet':
        original_s = getattr(args, 's', None)
        args.original_s = original_s
    elif args.poison_type == 'SIG':
        original_delta = getattr(args, 'delta', None)
        args.original_delta = original_delta
    
    # 如果指定了 test_alpha/test_s/test_delta，临时恢复原始值以获取正确的模型路径
    # 因为模型路径应该基于训练时的参数（包含 arch 信息）
    if args.test_alpha is not None:
        args.alpha = original_alpha  # 确保使用训练时的 alpha
    if args.poison_type == 'WaNet' and args.test_s is not None:
        args.s = original_s  # 确保使用训练时的 s
    if args.poison_type == 'SIG' and args.test_delta is not None:
        args.delta = original_delta  # 确保使用训练时的 delta
    # ========== [修复结束] ==========
    
    # 获取模型路径（基于训练时的参数，包含 arch 信息）
    model_path = supervisor.get_model_dir(args, cleanse=(args.cleanser is not None), 
                                         defense=(args.defense is not None))
    
    # ========== [修复] 获取模型路径后，如果指定了 test_alpha/test_s/test_delta，恢复测试值 ==========
    # 这样在创建 poison_transform 时会使用测试时的参数
    if args.test_alpha is not None:
        args.alpha = args.test_alpha
    if args.poison_type == 'WaNet' and args.test_s is not None:
        args.s = args.test_s
    if args.poison_type == 'SIG' and args.test_delta is not None:
        args.delta = args.test_delta
    # ========== [修复结束] ==========
    
    # ========== [BELT特殊处理] 使用增强模型 ==========
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
    # ========== [BELT特殊处理结束] ==========
    
    # ========== [修复] 检查模型文件是否存在 ==========
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print(f"请确保模型已经训练并保存到该路径")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    # ========== [修复结束] ==========
    
    # 获取模型架构
    arch = supervisor.get_arch(args)
    model = arch(num_classes=num_classes)
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"模型加载成功: {model_path}")
    
    model = nn.DataParallel(model)
    model = model.cuda()
    print("Evaluating model '{}'...".format(model_path))
    
    # 设置数据变换
    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)
    
    # 为STL-10创建特殊的变换
    # 注意：只做 ToTensor()，归一化在下采样之后进行（先下采样到32x32，再归一化）
    stl10_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载STL-10数据集
    print("加载STL-10数据集...")
    
    # 使用官方STL10数据集
    import torchvision
    test_set = torchvision.datasets.STL10(
        root='./data', 
        split='test', 
        download=True, 
        transform=stl10_transform
    )
    
    # 过滤掉monkey类别，保留与CIFAR-10匹配的9个类别
    # STL-10类别: ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    # CIFAR-10类别: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 7: -1, 8: 8, 9: 9}
    
    # 过滤数据
    filtered_data = []
    filtered_targets = []
    
    for i in range(len(test_set)):
        data, target = test_set[i]
        mapped_target = label_mapping[target]
        if mapped_target != -1:  # 过滤掉monkey类别
            filtered_data.append(data)
            filtered_targets.append(mapped_target)
    
    # 创建自定义数据集
    # 处理流程：下采样到 32x32 -> 归一化（使用 CIFAR-10 的标准化参数）
    class FilteredSTL10:
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets
            # CIFAR-10 的归一化参数
            self.normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img = self.data[idx]
            target = self.targets[idx]
            
            # 下采样到 32x32（STL-10 原始图像是 96x96）
            # 注意：由于 stl10_transform 已经应用了 ToTensor()，所以 img 是 tensor，形状为 (C, 96, 96)
            if img.shape[-1] != 32:
                img = img.unsqueeze(0)  # 添加 batch 维度: (1, C, 96, 96)
                img = tensor_resize(img, 32)  # 下采样: (1, C, 32, 32)
                img = img.squeeze(0)  # 移除 batch 维度: (C, 32, 32)
            
            # 应用归一化（CIFAR-10 的标准化参数）
            # 归一化在下采样之后进行，确保归一化参数适用于 32x32 的图像
            img = self.normalize(img)
            
            return img, target
    
    test_set = FilteredSTL10(filtered_data, filtered_targets)
    print(f"STL-10数据集加载完成: {len(test_set)} 张图像")
    print(f"过滤掉monkey类别，保留与CIFAR-10匹配的9个类别")
    
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, 
        num_workers=4, pin_memory=True)
    
    # 创建poison transform（标准32x32变换器）
    poison_transform = create_poison_transform(args, data_transform)
    
    # 执行STL-10测试（仅32x32方法）
    print("开始执行STL-10测试（32x32方法）...")
    # 重要：对于 UPGD 等针对类别进行扰动的攻击，target_class 必须使用训练数据集的 target_class
    # 因为 delta 是在训练数据集（如 CIFAR-10）上针对特定类别生成的
    # STL-10 的标签已经通过 label_mapping 映射到 CIFAR-10 的标签空间，所以 target_class 是一致的
    target_class = config.target_class[args.dataset]
    
    # 获取模型目录路径
    model_dir = os.path.dirname(model_path)
    
    acc, asr = test_stl10_model(
        model=model, 
        test_loader=test_set_loader, 
        poison_transform=poison_transform,
        poison_type=args.poison_type, 
        num_classes=num_classes, 
        target_class=target_class,
        model_dir=model_dir
    )
    
    # ========== [修改] 根据不同的测试参数，保存到不同的结果文件，避免覆盖 ==========
    # 保存测试结果
    # 根据攻击类型和测试参数，生成不同的文件名
    # 注意：需要从 create_poison_transform 中获取原始值，但由于函数作用域限制，这里需要重新获取
    # 实际上，args 中的值可能已经被 test_s 或 test_delta 修改了，所以需要检查 test_s 和 test_delta
    if args.poison_type == 'SIG':
        # SIG 攻击：根据实际测试时使用的 delta 值区分文件名（如果使用了 test_delta，则使用 test_delta）
        if args.test_delta is not None:
            delta_value = args.test_delta
            test_result_path = os.path.join(model_dir, f'test_stl10_results_test_delta={delta_value}.txt')
        else:
            delta_value = args.delta if hasattr(args, 'delta') else 30
            test_result_path = os.path.join(model_dir, f'test_stl10_results_delta={delta_value}.txt')
    elif args.poison_type == 'WaNet':
        # WaNet 攻击：根据实际测试时使用的 s 值区分文件名（如果使用了 test_s，则使用 test_s）
        if args.test_s is not None:
            s_value = args.test_s
            test_result_path = os.path.join(model_dir, f'test_stl10_results_test_s={s_value}.txt')
        else:
            s_value = args.s if hasattr(args, 's') else 0.5
            test_result_path = os.path.join(model_dir, f'test_stl10_results_s={s_value}.txt')
    elif args.poison_type in ['blend', 'adaptive_blend', 'adaptive_patch', 'basic', 'clean_label']:
        # Blend/BadNet/Patch 攻击：如果使用了 test_alpha，在文件名中包含 test_alpha 值
        if args.test_alpha is not None:
            test_result_path = os.path.join(model_dir, f'test_stl10_results_test_alpha={args.test_alpha}.txt')
        else:
            test_result_path = os.path.join(model_dir, 'test_stl10_results.txt')
    else:
        # 其他攻击类型：使用默认文件名
        test_result_path = os.path.join(model_dir, 'test_stl10_results.txt')
    # ========== [修改结束] ==========
    
    with open(test_result_path, 'w', encoding='utf-8') as f:
        f.write(f"=== STL-10测试结果（32x32方法）===\n")
        f.write(f"数据集: STL-10\n")
        f.write(f"攻击类型: {args.poison_type}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试方法: 32x32（先下采样到32x32并归一化，然后添加触发器）\n")
        f.write(f"插值方法: bicubic\n")
        f.write(f"目标类别: {target_class}\n")
        
        # 保存攻击参数
        if hasattr(args, 'alpha') and args.alpha is not None:
            f.write(f"训练时Alpha: {args.alpha}\n")
        # 如果使用了 test_alpha，也记录实际测试时使用的 alpha
        if hasattr(args, 'test_alpha') and args.test_alpha is not None:
            f.write(f"测试时Alpha (test_alpha): {args.test_alpha}\n")
        elif hasattr(args, 'alpha') and args.alpha is not None:
            f.write(f"测试时Alpha: {args.alpha} (使用训练时的alpha)\n")
        if hasattr(args, 'trigger') and args.trigger is not None:
            f.write(f"触发器: {args.trigger}\n")
        if hasattr(args, 'trigger_path') and args.trigger_path is not None:
            f.write(f"触发器路径: {args.trigger_path}\n")
        if hasattr(args, 'cover_rate') and args.cover_rate is not None:
            f.write(f"覆盖率: {args.cover_rate}\n")
        
        # ========== [修改] 保存特定攻击类型参数，并记录训练时和测试时的参数值 ==========
        if args.poison_type == 'WaNet':
            # 如果使用了 test_s，显示训练时和测试时的 s 参数
            if args.test_s is not None:
                # 使用保存的原始值作为训练时的 s
                train_s = getattr(args, 'original_s', None)
                if train_s is not None:
                    f.write(f"训练时WaNet s参数: {train_s}\n")
                f.write(f"测试时WaNet s参数 (test_s): {args.test_s}\n")
            else:
                # 没有使用 test_s，训练和测试使用相同的 s
                if hasattr(args, 's') and args.s is not None:
                    f.write(f"WaNet s参数: {args.s}\n")
            if hasattr(args, 'k') and args.k is not None:
                f.write(f"WaNet k参数: {args.k}\n")
        elif args.poison_type == 'SIG':
            # 如果使用了 test_delta，显示训练时和测试时的 delta 参数
            if args.test_delta is not None:
                # 使用保存的原始值作为训练时的 delta
                train_delta = getattr(args, 'original_delta', None)
                if train_delta is not None:
                    f.write(f"训练时SIG delta参数: {train_delta}\n")
                f.write(f"测试时SIG delta参数 (test_delta): {args.test_delta}\n")
            else:
                # 没有使用 test_delta，训练和测试使用相同的 delta
                if hasattr(args, 'delta') and args.delta is not None:
                    f.write(f"SIG delta参数: {args.delta}\n")
            if hasattr(args, 'f') and args.f is not None:
                f.write(f"SIG f参数: {args.f}\n")
        # ========== [修改结束] ==========
        
        f.write(f"\n=== 测试结果 ===\n")
        f.write(f"准确率: {acc:.6f} ({acc:.2%})\n")
        f.write(f"攻击成功率: {asr:.6f} ({asr:.2%})\n")
    
    print(f"测试结果已保存到: {test_result_path}")
    print(f"攻击类型: {args.poison_type}, 准确率: {acc:.6f} ({acc:.2%}), 攻击成功率: {asr:.6f} ({asr:.2%})")

if __name__ == "__main__":
    main()

