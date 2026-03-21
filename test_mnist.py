"""
MNIST-M 后门攻击测试脚本
用于测试在 MNIST-M 上训练的后门模型在 MNIST 测试集上的表现（跨域测试）
MNIST-M 是 MNIST 的彩色版本，用于跨域测试
"""

import numpy as np
import torch
import os
from torchvision import transforms
import argparse
from torch import nn
from PIL import Image
from utils import supervisor, tools, default_args
import config
import time
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# MNIST-M 数据集类
# =============================================================================

class MNISTMDataset(Dataset):
    """
    MNIST-M 数据集类：加载 MNIST-M 彩色图像
    MNIST-M 是 MNIST 的彩色版本，图像尺寸为 28×28×3（RGB）
    """
    def __init__(self, data_path, train=True, normalizer=None):
        """
        Args:
            data_path: MNIST-M 数据文件路径（.npy 文件所在目录）
            train: 是否加载训练集（True=训练集，False=测试集）
            normalizer: 归一化变换器
        """
        self.normalizer = normalizer
        
        # 加载数据
        if train:
            data_file = os.path.join(data_path, 'train.npy')
        else:
            data_file = os.path.join(data_path, 'test.npy')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"MNIST-M 数据文件不存在: {data_file}")
        
        # 加载图像数据
        self.data = np.load(data_file)  # shape: (N, 28, 28, 3)
        
        # ========== [重要说明] MNIST-M 标签从 MNIST 加载的原因 ==========
        # MNIST-M 是 MNIST 的彩色版本，通过将 MNIST 灰度图叠加在彩色背景上生成
        # MNIST-M 数据集通常只提供图像数据（.npy 文件），不包含标签文件
        # 因为 MNIST-M 的每个图像都对应 MNIST 的原始图像，标签是一一对应的：
        #   - train.npy 的第 i 个图像对应 MNIST train 的第 i 个标签
        #   - test.npy 的第 i 个图像对应 MNIST test 的第 i 个标签
        # 因此需要从 MNIST 数据集加载标签，这是 MNIST-M 数据集的标准做法
        # ========== [重要说明] 结束 ==========
        from torchvision.datasets import MNIST
        mnist_dataset = MNIST(root=config.mnist_dir, train=train, download=False)
        self.targets = mnist_dataset.targets
        
        # 验证数据一致性：MNIST-M 图像数量必须与 MNIST 标签数量一致
        if len(self.data) != len(self.targets):
            raise ValueError(
                f"MNIST-M 图像数量 ({len(self.data)}) 与 MNIST 标签数量 ({len(self.targets)}) 不匹配！\n"
                f"MNIST-M 图像文件: {data_file}\n"
                f"MNIST 数据集: {config.mnist_dir}, train={train}\n"
                f"请确保 MNIST-M 数据集与 MNIST 数据集一一对应（第 i 个 MNIST-M 图像对应第 i 个 MNIST 标签）。"
            )
        
        print(f"[MNIST-M] 加载 {'训练集' if train else '测试集'}: {len(self.data)} 张图像")
        print(f"[MNIST-M] 图像形状: {self.data.shape}, 标签数量: {len(self.targets)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取图像和标签
        img = self.data[idx]  # shape: (28, 28, 3), dtype: uint8, range: [0, 255]
        target = int(self.targets[idx])
        
        # 转换为 PIL Image 然后应用 ToTensor（将 [0, 255] 转换为 [0, 1]）
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)  # shape: (3, 28, 28), range: [0, 1]
        
        # 应用归一化（如果提供）
        if self.normalizer is not None:
            img = self.normalizer(img)
        
        return img, target


# =============================================================================
# MNIST-M 后门攻击测试函数
# =============================================================================

def test_mnistm_model(model, test_loader, poison_transform, poison_type, num_classes, 
                     target_class, save_example=True, model_dir=None):
    """
    在 MNIST-M 数据集上测试后门攻击效果
    
    Args:
        model (torch.nn.Module): 已训练的目标模型（在 MNIST 上训练）
        test_loader (torch.utils.data.DataLoader): MNIST-M 测试数据加载器
        poison_transform: 后门攻击变换器
        poison_type (str): 攻击类型名称
        num_classes (int): 数据集类别数量（10）
        target_class (int): 后门攻击的目标类别
        save_example (bool, optional): 是否保存示例图片，默认为True
        model_dir (str, optional): 模型目录路径，用于保存结果文件
        
    Returns:
        tuple: (准确率, 攻击成功率)
    """
    print(f"执行 MNIST-M 测试: 攻击类型={poison_type}")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 首先计算准确率（不加触发器）
    print(f"\n=== 计算准确率（MNIST-M，不加触发器）===")
    acc_correct = 0
    acc_total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            acc_correct += pred.eq(target.view_as(pred)).sum().item()
            acc_total += target.size(0)

    # 计算准确率
    acc = acc_correct / acc_total if acc_total > 0 else 0
    print(f"MNIST-M 总体准确率: {acc:.6f} ({acc*100:.2f}%)")
    
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
            
            # 添加触发器（poison_transform 会处理归一化后的数据）
            data_poisoned, _ = poison_transform.transform(data, target)
            
            # 保存示例图片（仅第一个batch）
            if save_example and batch_idx == 0:
                original_img = data[0].clone()
                poisoned_img = data_poisoned[0].clone()
                
                if model_dir:
                    save_poisoned_example_to_dir(original_img, poisoned_img, 
                                                poison_type, 
                                                "mnistm_test", model_dir)
            
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
    save_dir = os.path.join(model_dir, "mnistm_examples")
    os.makedirs(save_dir, exist_ok=True)
    
    # 反归一化图片（使用 MNIST-M 的归一化参数）
    # MNIST-M 归一化参数: Mean=[0.46, 0.46, 0.46], Std=[0.23, 0.23, 0.23]
    mean = torch.tensor([0.46, 0.46, 0.46]).view(3, 1, 1).cuda()
    std = torch.tensor([0.23, 0.23, 0.23]).view(3, 1, 1).cuda()
    
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
    创建poison transform
    
    Args:
        args: 命令行参数
        data_transform: 数据变换
        
    Returns:
        poison_transform: poison transform对象
    """
    # 保存原始值到 args 对象中（用于后续结果记录）
    if args.poison_type == 'WaNet':
        args.original_s = getattr(args, 's', None)
        if args.test_s is not None:
            args.s = args.test_s
    elif args.poison_type == 'SIG':
        args.original_delta = getattr(args, 'delta', None)
        if args.test_delta is not None:
            args.delta = args.test_delta
    
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
        is_normalized = True  # 其他攻击使用归一化
    
    # 创建 poison transform（使用 MNIST 的配置，因为模型是在 MNIST 上训练的）
    poison_transform = supervisor.get_poison_transform(
        poison_type=args.poison_type, 
        dataset_name='mnist',  # 使用 MNIST 的配置
        target_class=config.target_class['mnist'], 
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
    主函数：MNIST-M 后门攻击测试程序入口
    
    本函数负责：
    1. 解析命令行参数
    2. 加载预训练模型（在 MNIST-M 上训练）
    3. 准备 MNIST 测试数据集（跨域测试）
    4. 创建后门攻击变换器（使用 MNIST-M 配置）
    5. 执行测试并保存结果
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='MNIST-M 后门攻击测试工具')
    
    # ===== 基本参数 =====
    parser.add_argument('-dataset', type=str, required=False,
                        default='mnistm',
                        help='源数据集名称（MNIST-M，用于定位模型路径）')
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
    parser.add_argument('-test_s', type=float, required=False, default=None,
                        help='测试时的WaNet s参数（覆盖训练时的s）')
    parser.add_argument('-test_delta', type=float, required=False, default=None,
                        help='测试时的SIG delta参数（覆盖训练时的delta）')
    
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
    parser.add_argument('-delta', type=float, default=30,
                        help='SIG攻击delta参数，会自动除以255 (默认30，即30/255)')
    parser.add_argument('-f', type=float, default=6,
                        help='SIG攻击频率参数 (默认6)')
    parser.add_argument('-s', type=float, default=0.5,
                        help='WaNet攻击s参数 (默认0.5)')
    parser.add_argument('-k', type=int, default=4,
                        help='WaNet攻击k参数 (默认4)')

    # ========== [UPGD 参数] 开始 ==========
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
    
    # ===== MNIST-M 数据路径 =====
    parser.add_argument('-mnistm_data_path', type=str, default='./data/MNIST-M',
                        help='MNIST-M 数据文件路径（包含 train.npy 和 test.npy 的目录）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
    
    # 设置默认触发器（使用 MNIST-M 的触发器配置）
    if args.trigger is None and args.trigger_path is None:
        args.trigger = config.trigger_default['mnistm'][args.poison_type]
    elif args.trigger is None and args.trigger_path is not None:
        args.trigger = os.path.basename(args.trigger_path)
    
    # 设置数据集参数
    args.dataset = 'mnistm'  # 固定使用 MNIST-M 作为源数据集（训练数据集）
    num_classes = 10
    batch_size = 128
    
    # 保存原始参数（用于获取模型路径）
    original_alpha = args.alpha
    if args.poison_type == 'WaNet':
        original_s = getattr(args, 's', None)
        args.original_s = original_s
    elif args.poison_type == 'SIG':
        original_delta = getattr(args, 'delta', None)
        args.original_delta = original_delta
    
    # 如果指定了 test_alpha/test_s/test_delta，临时恢复原始值以获取正确的模型路径
    if args.test_alpha is not None:
        args.alpha = original_alpha
    if args.poison_type == 'WaNet' and args.test_s is not None:
        args.s = original_s
    if args.poison_type == 'SIG' and args.test_delta is not None:
        args.delta = original_delta
    
    # 获取模型路径（基于训练时的参数）
    model_path = supervisor.get_model_dir(args, cleanse=(args.cleanser is not None), 
                                         defense=(args.defense is not None))
    
    # 获取模型路径后，恢复测试值
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
        print(f"请确保模型已经训练并保存到该路径")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
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
    
    # ========== [修改] 测试时使用 MNIST 数据集（跨域测试） ==========
    # 模型在 MNIST-M 上训练，测试时在 MNIST 上测试
    # 【重要】使用 MNIST-M 的归一化参数（与训练时一致），而不是 MNIST 的参数
    # 原因：1) 模型学习了 MNIST-M 归一化后的特征分布
    #       2) 触发器也是基于 MNIST-M 归一化设计的
    #       3) 与 test_stl10.py（使用 CIFAR-10 参数）保持一致
    mnistm_args = argparse.Namespace()
    mnistm_args.dataset = 'mnistm'  # 使用训练数据集的归一化参数
    mnistm_args.no_normalize = args.no_normalize
    mnistm_args.no_aug = args.no_aug
    mnistm_args.poison_type = args.poison_type
    data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(mnistm_args)
    
    # 加载 MNIST 测试数据集（跨域测试）
    print("加载 MNIST 测试数据集（跨域测试）...")
    
    # 从 clean_set 加载 MNIST 测试集
    test_set_dir = os.path.join('clean_set', 'mnist', 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    
    test_set_loader = DataLoader(
        test_set,
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=tools.worker_init, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 创建poison transform（使用 MNIST-M 的配置，因为模型是在 MNIST-M 上训练的）
    # 但触发器尺寸是 28×28，与 MNIST 相同
    poison_transform = create_poison_transform(args, data_transform)
    
    # 执行 MNIST 测试（跨域测试）
    print("开始执行 MNIST 测试（跨域测试：MNIST-M 训练的模型在 MNIST 上测试）...")
    target_class = config.target_class['mnistm']  # 使用 MNIST-M 的目标类（与训练时一致）
    
    # 获取模型目录路径
    model_dir = os.path.dirname(model_path)
    
    acc, asr = test_mnistm_model(
        model=model, 
        test_loader=test_set_loader, 
        poison_transform=poison_transform,
        poison_type=args.poison_type, 
        num_classes=num_classes, 
        target_class=target_class,
        model_dir=model_dir
    )
    
    # 保存测试结果
    if args.poison_type == 'SIG':
        if args.test_delta is not None:
            delta_value = args.test_delta
            test_result_path = os.path.join(model_dir, f'test_mnistm_results_test_delta={delta_value}.txt')
        else:
            delta_value = args.delta if hasattr(args, 'delta') else 30
            test_result_path = os.path.join(model_dir, f'test_mnistm_results_delta={delta_value}.txt')
    elif args.poison_type == 'WaNet':
        if args.test_s is not None:
            s_value = args.test_s
            test_result_path = os.path.join(model_dir, f'test_mnistm_results_test_s={s_value}.txt')
        else:
            s_value = args.s if hasattr(args, 's') else 0.5
            test_result_path = os.path.join(model_dir, f'test_mnistm_results_s={s_value}.txt')
    elif args.poison_type in ['blend', 'adaptive_blend', 'adaptive_patch', 'basic', 'clean_label']:
        if args.test_alpha is not None:
            test_result_path = os.path.join(model_dir, f'test_mnistm_results_test_alpha={args.test_alpha}.txt')
        else:
            test_result_path = os.path.join(model_dir, 'test_mnistm_results.txt')
    else:
        test_result_path = os.path.join(model_dir, 'test_mnistm_results.txt')
    
    with open(test_result_path, 'w', encoding='utf-8') as f:
        f.write(f"=== MNIST 跨域测试结果 ===\n")
        f.write(f"训练数据集: MNIST-M\n")
        f.write(f"测试数据集: MNIST（跨域测试）\n")
        f.write(f"攻击类型: {args.poison_type}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
        if hasattr(args, 'trigger_path') and args.trigger_path is not None:
            f.write(f"触发器路径: {args.trigger_path}\n")
        if hasattr(args, 'cover_rate') and args.cover_rate is not None:
            f.write(f"覆盖率: {args.cover_rate}\n")
        
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
            if hasattr(args, 'k') and args.k is not None:
                f.write(f"WaNet k参数: {args.k}\n")
        elif args.poison_type == 'SIG':
            if args.test_delta is not None:
                train_delta = getattr(args, 'original_delta', None)
                if train_delta is not None:
                    f.write(f"训练时SIG delta参数: {train_delta}\n")
                f.write(f"测试时SIG delta参数 (test_delta): {args.test_delta}\n")
            else:
                if hasattr(args, 'delta') and args.delta is not None:
                    f.write(f"SIG delta参数: {args.delta}\n")
            if hasattr(args, 'f') and args.f is not None:
                f.write(f"SIG f参数: {args.f}\n")
        
        f.write(f"\n=== 测试结果 ===\n")
        f.write(f"准确率: {acc:.6f} ({acc:.2%})\n")
        f.write(f"攻击成功率: {asr:.6f} ({asr:.2%})\n")
    
    print(f"测试结果已保存到: {test_result_path}")
    print(f"攻击类型: {args.poison_type}, 准确率: {acc:.6f} ({acc:.2%}), 攻击成功率: {asr:.6f} ({asr:.2%})")


if __name__ == "__main__":
    main()
