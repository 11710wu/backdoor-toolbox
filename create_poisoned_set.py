import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import argparse
from PIL import Image
import numpy as np
import config
from utils import supervisor, default_args, tools

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
# ========== [UPGD 参数] 开始 ==========
parser.add_argument('-upgd_model_path', type=str, required=False, default=None,
                    help='用于生成UPGD通用扰动的干净基模型权重路径（state_dict）')
parser.add_argument('-eps', type=float, required=False, default=8.0,
                    help='UPGD扰动约束大小；Linf下默认按像素尺度解释（例如8表示8/255）')
parser.add_argument('-constraint', type=str, required=False, default='Linf',
                    choices=['Linf', 'L2'], help='UPGD约束类型：Linf 或 L2')
parser.add_argument('-upgd_steps', type=int, required=False, default=100,
                    help='UPGD迭代步数（每步更新一次delta）')
parser.add_argument('-upgd_steps_multiplier', type=int, required=False, default=5,
                    help='UPGD总迭代次数倍率：总迭代 = upgd_steps * multiplier（默认5）')
parser.add_argument('-upgd_batch_size', type=int, required=False, default=256,
                    help='UPGD生成时的数据batch size')
# ========== [UPGD 参数] 结束 ==========
# ========== [WaNet参数修改] 开始 ==========
# WaNet攻击专用参数
parser.add_argument('-s', type=float, default=0.5,
                    help='WaNet攻击s参数 (默认0.5)')
parser.add_argument('-k', type=int, default=4,
                    help='WaNet攻击k参数 (默认4)')
# ========== [WaNet参数修改] 结束 ==========
# ========== [SIG参数修改] 开始 ==========
# SIG攻击专用参数
parser.add_argument('-delta', type=float, default=30,
                    help='SIG攻击delta参数，会自动除以255 (默认30，即30/255)')
parser.add_argument('-f', type=float, default=6,
                    help='SIG攻击f参数 (默认6)')
# ========== [SIG参数修改] 结束 ==========
# ========== [BELT 参数] 开始 ==========
parser.add_argument('-mask_rate', type=float, required=False, default=0.2,
                    help='BELT cover samples 的 mask 比例（默认 0.2）')
parser.add_argument('-model', type=str, required=False, default=None,
                    choices=['resnet18', 'vgg19_bn', 'mobilenetv2'],
                    help='模型架构选择（覆盖config.py中的默认设置）')
# ========== [BELT 参数] 结束 ==========
args = parser.parse_args()

tools.setup_seed(2333)

# ========== [噪声增强固定参数] 开始 ==========
# 固定噪声参数，只对中毒图片添加噪声
ENABLE_NOISE = False  # 是否启用噪声增强（True表示启用，False表示不启用）
NOISE_RATIO = 0.5  # 对中毒图片添加噪声的比例 (0.0-1.0)，0.5表示对一半的中毒图片添加噪声
NOISE_TYPE = 'gaussian'  # 噪声类型: 'gaussian', 'salt_pepper', 'uniform'
NOISE_STRENGTH = 25.0  # 噪声强度: 高斯噪声的标准差（像素值），建议15-30
NOISE_SEED = config.poison_seed  # 随机种子，使用与poison_seed一致的种子
# ========== [噪声增强固定参数] 结束 ==========

# ========== [噪声增强函数] 开始 ==========
def apply_noise_to_poisoned_images(img_set, poison_indices, noise_ratio=0.5, noise_type='gaussian', noise_strength=25.0, noise_seed=None):
    """
    对中毒图片按比例添加噪声
    
    Args:
        img_set: torch.Tensor, shape [N, C, H, W], 值范围 [0, 1]
        poison_indices: list/array/tensor，中毒图片的索引列表
        noise_ratio: 对中毒图片添加噪声的比例 (0.0-1.0)，0.5表示对一半的中毒图片添加噪声
        noise_type: 噪声类型 ('gaussian', 'salt_pepper', 'uniform')
        noise_strength: 噪声强度
            - gaussian: 标准差（像素值，需要除以255转换为[0,1]范围）
            - salt_pepper: 噪声比例 (0-100)
            - uniform: 噪声幅度（像素值，需要除以255转换为[0,1]范围）
        noise_seed: 随机种子
    
    Returns:
        noisy_img_set: 添加噪声后的tensor
    """
    # 转换为list格式
    if isinstance(poison_indices, torch.Tensor):
        poison_indices = poison_indices.tolist()
    elif isinstance(poison_indices, np.ndarray):
        poison_indices = poison_indices.tolist()
    
    if len(poison_indices) == 0 or noise_ratio <= 0:
        return img_set
    
    # 设置随机种子
    if noise_seed is not None:
        np.random.seed(noise_seed)
        torch.manual_seed(noise_seed)
    
    # 从中毒图片中随机选择一部分来添加噪声
    num_poisoned = len(poison_indices)
    num_noisy = int(num_poisoned * noise_ratio)
    if num_noisy == 0:
        return img_set
    
    # 随机选择要添加噪声的中毒图片索引
    selected_indices = np.random.choice(poison_indices, size=num_noisy, replace=False)
    
    # 只对选中的中毒图片添加噪声
    noisy_img_set = img_set.clone()
    
    for idx in selected_indices:
        img = noisy_img_set[idx]  # [C, H, W]
        
        if noise_type == 'gaussian':
            # 高斯噪声：将noise_strength从像素值转换为[0,1]范围
            noise_std = noise_strength / 255.0
            noise = torch.randn_like(img) * noise_std
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
            
        elif noise_type == 'salt_pepper':
            # 椒盐噪声：noise_strength是噪声比例(0-100)
            noise_prob = noise_strength / 100.0
            # 生成随机mask
            salt_mask = torch.rand(1, img.shape[1], img.shape[2]) < noise_prob
            pepper_mask = torch.rand(1, img.shape[1], img.shape[2]) < noise_prob
            
            # 椒噪声（黑色点，设为0）
            noisy_img = img * (~salt_mask.expand_as(img))
            # 盐噪声（白色点，设为1）
            noisy_img = torch.where(pepper_mask.expand_as(img), torch.ones_like(img), noisy_img)
            
        elif noise_type == 'uniform':
            # 均匀噪声：将noise_strength从像素值转换为[0,1]范围
            noise_range = noise_strength / 255.0
            noise = torch.rand_like(img) * 2 * noise_range - noise_range
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        else:
            noisy_img = img
        
        noisy_img_set[idx] = noisy_img
    
    return noisy_img_set
# ========== [噪声增强函数] 结束 ==========

# =============================================================================
# 随机种子 / 可复现性（工程说明）
# -----------------------------------------------------------------------------
# 本脚本使用 `tools.setup_seed(0)`（固定种子）来保证“生成投毒数据集”可复现。
#
# 对 UPGD 来说，我们同样固定：
# - cfg.seed = config.poison_seed：生成 delta 的随机性（DataLoader shuffle / python random 等）
# - poison seed = config.poison_seed：从目标类别子集中抽样 poison_indices 的随机性
#
# 这样做的好处：同一组参数重复运行会生成完全一致的数据集，便于对比实验。
# 种子值统一使用 config.poison_seed（默认2333），确保与训练、测试时的种子一致。
# =============================================================================


print('[target class : %d]' % config.target_class[args.dataset])

data_dir = config.data_dir  # directory to save standard clean set
if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

if not os.path.exists(os.path.join('poisoned_train_set', args.dataset)):
    os.mkdir(os.path.join('poisoned_train_set', args.dataset))

if args.poison_type == 'dynamic':

    if args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        channel_init = 32
        steps = 3
        input_channel = 3

        ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'

        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261], [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])

    elif args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split='train',
                                   transform=data_transform, download=True)

        img_size = 32
        num_classes = 43
        channel_init = 32
        steps = 3
        input_channel = 3

        ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'

        normalizer = None
        denormalizer = None

    elif args.dataset == 'imagenette':
        raise  NotImplementedError('imagenette unsupported for dynamic!')
    else:
        raise  NotImplementedError('Undefined Dataset')

elif args.poison_type == 'ISSBA':

    if args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        input_channel = 3

        ckpt_path = './models/ISSBA_cifar10.pth'

    elif args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split='train',
                                   transform=data_transform, download=True)

        img_size = 32
        num_classes = 43
        input_channel = 3

        ckpt_path = './models/ISSBA_gtsrb.pth'

    elif args.dataset == 'imagenette':
        raise  NotImplementedError('imagenette unsupported!')
    else:
        raise  NotImplementedError('Undefined Dataset')

else:

    if args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split = 'train',
                                   transform = data_transform, download=True)
        img_size = 32
        num_classes = 43

    elif args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10

    elif args.dataset == 'imagenette':

        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(os.path.join(os.path.join(data_dir, 'imagenette2'), 'train'),
                                         data_transform)
        img_size = 224
        num_classes = 10

    # ========== [Tiny ImageNet 支持] 添加数据集加载 ==========
    # Tiny ImageNet: 加载 train/ 目录（ImageFolder 格式）
    # 图像尺寸: 使用原始 64×64 尺寸（与 BackdoorBench 一致）
    # 类别数: 200
    elif args.dataset == 'tiny_imagenet':
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # 原始图片已经是 64×64，不需要 Resize
        ])
        train_set = datasets.ImageFolder(os.path.join(config.tiny_imagenet_dir, 'train'),
                                         data_transform)
        img_size = 64  # 使用原始 64×64 尺寸
        num_classes = 200
    # ========== [Tiny ImageNet 支持] 结束 ==========
    # ========== [MNIST 支持] 添加数据集加载 ==========
    # MNIST: 28×28 单通道灰度图，转换为三通道以支持 RGB 触发器
    elif args.dataset == 'mnist':
        # 将单通道转换为三通道（复制灰度值到三个通道）
        # 这样触发器可以保持 RGB 格式，对原图内容无影响
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道复制为三通道
        ])
        train_set = datasets.MNIST(os.path.join(data_dir, 'mnist'), train=True,
                                   download=False, transform=data_transform)
        img_size = 28
        num_classes = 10

    # ========== [MNIST 支持] 结束 ==========
    # ========== [MNIST-M 支持] 添加数据集加载 ==========
    # MNIST-M: 28×28 RGB 彩色图，用于训练
    elif args.dataset == 'mnistm':
        # MNIST-M 是 RGB 彩色图像，已经是三通道，不需要转换
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # 将 [0, 255] 转换为 [0, 1]
        ])
        # 加载 MNIST-M 数据集（从 .npy 文件）
        from torch.utils.data import Dataset
        import numpy as np
        from PIL import Image
        
        class MNISTMDataset(Dataset):
            def __init__(self, data_path, train=True, transform=None):
                self.transform = transform
                # 加载数据
                if train:
                    data_file = os.path.join(data_path, 'train.npy')
                else:
                    data_file = os.path.join(data_path, 'test.npy')
                
                if not os.path.exists(data_file):
                    raise FileNotFoundError(f"MNIST-M 数据文件不存在: {data_file}")
                
                # 加载图像数据
                self.data = np.load(data_file)  # shape: (N, 28, 28, 3)
                
                # 加载 MNIST 标签（MNIST-M 使用与 MNIST 相同的标签）
                from torchvision.datasets import MNIST
                mnist_dataset = MNIST(root=os.path.join(data_dir, 'mnist'), train=train, download=False)
                self.targets = mnist_dataset.targets
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                img = self.data[idx]  # shape: (28, 28, 3), dtype: uint8, range: [0, 255]
                target = int(self.targets[idx])
                
                # 转换为 PIL Image 然后应用 transform
                img = Image.fromarray(img)
                if self.transform:
                    img = self.transform(img)
                
                return img, target
        
        train_set = MNISTMDataset(data_path=config.mnistm_dir, train=True, transform=data_transform)
        img_size = 28
        num_classes = 10

    # ========== [MNIST-M 支持] 结束 ==========

    else:
        raise  NotImplementedError('Undefined Dataset')

trigger_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create poisoned dataset directory for current setting
poison_set_dir = supervisor.get_poison_set_dir(args)
# poison_set_img_dir = os.path.join(poison_set_dir, 'data')

if os.path.exists(poison_set_dir):
    print(f"Poisoned set directory '{poison_set_dir}' to be created is not empty! Exiting...")
    exit()
if not os.path.exists(poison_set_dir):
    os.mkdir(poison_set_dir)
# if not os.path.exists(poison_set_img_dir):
#     os.mkdir(poison_set_img_dir)



if args.poison_type in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                        'adaptive_blend', 'adaptive_patch', 'adaptive_k_way',
                        'SIG', 'TaCT', 'WaNet', 'SleeperAgent', 'none',
                        'badnet_all_to_all', 'trojan', 'belt']:

    trigger_name = args.trigger
    trigger_path = os.path.join(config.triggers_dir, trigger_name)

    trigger = None
    trigger_mask = None

    # ========== [BELT 特殊处理] BELT 使用生成的触发器，跳过文件加载 ==========
    if args.poison_type == 'belt':
        # BELT 使用与原始代码一致的触发器生成方式，不需要加载文件
        trigger = None
        trigger_mask = None
    elif trigger_name != 'none':  # none for SIG
        print('trigger: %s' % trigger_path)

        trigger_path = os.path.join(config.triggers_dir, trigger_name)
        
        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger_transform(trigger)

        # trigger_mask 会自动使用与 trigger_name 对应的文件（如果存在 64×64 触发器，mask 也会自动使用对应的 64×64 版本）
        trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)
        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            #print('trigger_mask_path:', trigger_mask_path)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            #print('No trigger mask found! By default masking all black pixels...')
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()
    # ========== [BELT 特殊处理] 结束 ==========

    alpha = args.alpha

    poison_generator = None
    if args.poison_type == 'basic':

        from poison_tool_box import basic
        poison_generator = basic.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate,
                                                  path=poison_set_dir,
                                                  trigger_mark=trigger, trigger_mask=trigger_mask,
                                                  target_class=config.target_class[args.dataset], alpha=alpha)
        
    elif args.poison_type == 'badnet':

        from poison_tool_box import badnet
        poison_generator = badnet.poison_generator(img_size=img_size, dataset=train_set,
                                                   poison_rate=args.poison_rate, trigger_mark=trigger, trigger_mask=trigger_mask,
                                                   path=poison_set_dir, target_class=config.target_class[args.dataset], alpha=alpha)

    elif args.poison_type == 'belt':
        # ========== [BELT 支持] 开始 ==========
        # BELT (Backdoor Exclusivity Lifting Technique)
        # 使用与原始 BELT 代码一致的触发器生成方式
        # 【重要】BELT 强制 alpha=1.0（与原始 BELT 代码一致）
        from poison_tool_box import belt
        import numpy as np
        
        # BELT 强制 alpha=1.0，忽略用户输入的 alpha 参数
        belt_alpha = 1.0
        
        # 使用公共函数生成触发器（与原始 BELT 代码一致）
        belt_mask_np, belt_pattern_np = belt.generate_belt_trigger(img_size, alpha=belt_alpha)
        
        # 验证触发器生成
        assert belt_mask_np.shape == (img_size, img_size, 3), f"mask shape错误: {belt_mask_np.shape}"
        assert belt_pattern_np.shape == (img_size, img_size, 3), f"pattern shape错误: {belt_pattern_np.shape}"
        assert belt_pattern_np.min() >= 0 and belt_pattern_np.max() <= 255, f"pattern值范围错误: [{belt_pattern_np.min()}, {belt_pattern_np.max()}]"
        assert np.all(belt_mask_np[2:8, 2:8, :] == belt_alpha), f"mask区域值错误: 应该是{belt_alpha}"
        assert np.all(belt_mask_np[:2, :, :] == 0) and np.all(belt_mask_np[8:, :, :] == 0), "mask非触发区域应该为0"
        
        # 转换为 torch tensor（与代码库格式一致）
        # mask: [H, W, 3] -> [H, W] (单通道，因为三个通道值相同)
        belt_mask_torch = torch.from_numpy(belt_mask_np[:, :, 0]).float()  # 只取一个通道
        # pattern: [H, W, 3] -> [3, H, W]，归一化到 [0, 1]（因为图像也是 [0, 1]）
        belt_pattern_torch = torch.from_numpy(belt_pattern_np).permute(2, 0, 1).float() / 255.0  # 归一化到 [0, 1]
        
        # 验证转换后的值范围
        assert belt_pattern_torch.min() >= 0.0 and belt_pattern_torch.max() <= 1.0, f"归一化后pattern值范围错误: [{belt_pattern_torch.min()}, {belt_pattern_torch.max()}]"
        
        # ========== [BELT 触发器保存] ==========
        # 保存触发器图片到 poison_set_dir
        os.makedirs(poison_set_dir, exist_ok=True)
        
        # 保存 pattern 图片（RGB，[0,255]）
        belt_pattern_img = Image.fromarray(belt_pattern_np.astype(np.uint8))
        belt_pattern_path = os.path.join(poison_set_dir, 'belt_pattern.png')
        belt_pattern_img.save(belt_pattern_path)
        print(f'[BELT] 保存触发器 pattern: {belt_pattern_path}')
        
        # 保存 mask 图片（灰度，[0,255]）
        belt_mask_img = Image.fromarray((belt_mask_np[:, :, 0] * 255).astype(np.uint8))
        belt_mask_path = os.path.join(poison_set_dir, 'belt_mask.png')
        belt_mask_img.save(belt_mask_path)
        print(f'[BELT] 保存触发器 mask: {belt_mask_path}')
        
        # 保存触发器 tensor（用于后续加载）
        belt_trigger_data = {
            'pattern': belt_pattern_torch,  # [3, H, W], [0, 1]
            'mask': belt_mask_torch,        # [H, W], [0, 1]
            'alpha': belt_alpha,
            'pattern_x': 2,
            'pattern_y': 8,
            'seed': 2333,
        }
        belt_trigger_path = os.path.join(poison_set_dir, 'belt_trigger.pt')
        torch.save(belt_trigger_data, belt_trigger_path)
        print(f'[BELT] 保存触发器 tensor: {belt_trigger_path}')
        # ========== [BELT 触发器保存] 结束 ==========
        
        # 默认参数：cover_rate=0.5, mask_rate=0.2（可从 args 读取，如果提供）
        cover_rate = getattr(args, 'cover_rate', 0.5)
        mask_rate = getattr(args, 'mask_rate', 0.2)
        
        poison_generator = belt.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, trigger_mark=belt_pattern_torch, trigger_mask=belt_mask_torch,
                                                 path=poison_set_dir, target_class=config.target_class[args.dataset], 
                                                 alpha=belt_alpha, cover_rate=cover_rate, mask_rate=mask_rate)
        # ========== [BELT 支持] 结束 ==========

    elif args.poison_type == 'badnet_all_to_all':

        from poison_tool_box import badnet_all_to_all
        poison_generator = badnet_all_to_all.poison_generator(img_size=img_size, dataset=train_set,
                                                   poison_rate=args.poison_rate, trigger_mark=trigger, trigger_mask=trigger_mask,
                                                   path=poison_set_dir, num_classes=num_classes)

    elif args.poison_type == 'trojan':

        from poison_tool_box import trojan
        poison_generator = trojan.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, trigger_mark=trigger, trigger_mask=trigger_mask,
                                                 path=poison_set_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'blend':

        from poison_tool_box import blend
        poison_generator = blend.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate, trigger=trigger,
                                                  path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                  alpha=alpha)
    elif args.poison_type == 'refool':
        from poison_tool_box import refool
        poison_generator = refool.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate,
                                                  path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                  max_image_size=32)

    elif args.poison_type == 'TaCT':

        from poison_tool_box import TaCT
        poison_generator = TaCT.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, cover_rate=args.cover_rate,
                                                 trigger=trigger, mask=trigger_mask,
                                                 path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                 source_class=config.source_class,
                                                 cover_classes=config.cover_classes)

    elif args.poison_type == 'WaNet':
        # Prepare grid
        # ========== [WaNet参数修改] 从args中读取s和k参数 ==========
        s = args.s
        k = args.k
        # ========== [WaNet参数修改] 结束 ==========
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            torch.nn.functional.upsample(ins, size=img_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=img_size)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        
        path = os.path.join(poison_set_dir, 'identity_grid')
        torch.save(identity_grid, path)
        path = os.path.join(poison_set_dir, 'noise_grid')
        torch.save(noise_grid, path)

        from poison_tool_box import WaNet
        poison_generator = WaNet.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, cover_rate=args.cover_rate,
                                                 path=poison_set_dir,
                                                 identity_grid=identity_grid, noise_grid=noise_grid,
                                                 s=s, k=k, grid_rescale=grid_rescale, 
                                                 target_class=config.target_class[args.dataset])

    elif args.poison_type == 'adaptive':

        from poison_tool_box import adaptive
        poison_generator = adaptive.poison_generator(img_size=img_size, dataset=train_set,
                                                     poison_rate=args.poison_rate,
                                                     path=poison_set_dir,
                                                     trigger_mark=trigger, trigger_mask=trigger_mask,
                                                     target_class=config.target_class[args.dataset], alpha=alpha,
                                                     cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_blend':

        from poison_tool_box import adaptive_blend
        poison_generator = adaptive_blend.poison_generator(img_size=img_size, dataset=train_set,
                                                          poison_rate=args.poison_rate,
                                                          path=poison_set_dir, trigger=trigger,
                                                          pieces=16, mask_rate=0.5,
                                                          target_class=config.target_class[args.dataset], alpha=alpha,
                                                          cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_patch':
        # ========== [Adaptive Patch Alpha参数修改] 开始 ==========
        # 功能：支持通过-alpha参数在默认alpha列表基础上进行偏移调整
        # 说明：
        #   - 默认alpha列表（config.py）: [0.5, 0.2, 0.5, 0.3]
        #   - 如果指定-alpha=0.1，则结果: [0.6, 0.3, 0.6, 0.4] (每个值+0.1)
        #   - 如果不指定或使用默认值0.2，则使用配置文件中的原始列表
        # 使用示例：
        #   python create_poisoned_set.py ... -alpha=0.1  # 结果: [0.6, 0.3, 0.6, 0.4]
        #   python create_poisoned_set.py ... -alpha=0.2  # 结果: [0.7, 0.4, 0.7, 0.5]
        #   python create_poisoned_set.py ...            # 结果: [0.5, 0.2, 0.5, 0.3] (默认)
       
        
        from poison_tool_box import adaptive_patch
        
        # 获取默认的alpha列表
        default_alphas = config.adaptive_patch_train_trigger_alphas[args.dataset]
        
        # 对于adaptive_patch，alpha参数始终表示在默认alpha列表基础上的偏移量
        # 如果用户未指定-alpha参数（使用argparse默认值0.2），则使用原始默认列表
        # 如果用户指定了-alpha参数（包括-alpha=0.2），则在默认列表基础上加上这个偏移量
        if args.alpha == default_args.parser_default['alpha']:
            # 用户未指定alpha参数，使用默认列表
            alphas = default_alphas
            print(f"[Adaptive Patch] 未指定alpha偏移量，使用默认Alpha值列表: {alphas}")
        else:
            # 用户指定了alpha偏移量，在默认列表基础上加上这个值
            alphas = [alpha + args.alpha for alpha in default_alphas]
            print(f"[Adaptive Patch] 默认Alpha列表: {default_alphas}")
            print(f"[Adaptive Patch] 添加alpha偏移量={args.alpha}后: {alphas}")
        # ========== [Adaptive Patch Alpha参数修改] 结束 ==========
        poison_generator = adaptive_patch.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_dir,
                                                           trigger_names=config.adaptive_patch_train_trigger_names[args.dataset],
                                                           alphas=alphas,
                                                           target_class=config.target_class[args.dataset],
                                                           cover_rate=args.cover_rate)

    elif args.poison_type == 'adaptive_k_way':

        from poison_tool_box import adaptive_k_way
        poison_generator = adaptive_k_way.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_dir,
                                                           target_class=config.target_class[args.dataset],
                                                           cover_rate=args.cover_rate)

    elif args.poison_type == 'SIG':
        # ========== [SIG参数修改] 从args中读取delta和f参数，delta自动除以255 ==========
        delta = args.delta / 255
        f = args.f
        # ========== [SIG参数修改] 结束 ==========
        from poison_tool_box import SIG
        poison_generator = SIG.poison_generator(img_size=img_size, dataset=train_set,
                                                poison_rate=args.poison_rate,
                                                path=poison_set_dir, target_class=config.target_class[args.dataset],
                                                delta=delta, f=f)

    elif args.poison_type == 'clean_label':

        if args.dataset == 'cifar10':
            adv_imgs_path = "data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy"
            if not os.path.exists("data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy"):
                raise NotImplementedError("Run 'data/cifar10/clean_label/setup.sh' first to launch clean label attack!")
            adv_imgs_src = np.load("data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy").astype(
                np.uint8)
            adv_imgs = []
            for i in range(adv_imgs_src.shape[0]):
                adv_imgs.append(data_transform(adv_imgs_src[i]).unsqueeze(0))
            adv_imgs = torch.cat(adv_imgs, dim=0)
            assert adv_imgs.shape[0] == len(train_set)
        else:
            raise NotImplementedError('Clean Label Attack is not implemented for %s' % args.dataset)

        # Init Attacker
        from poison_tool_box import clean_label
        poison_generator = clean_label.poison_generator(img_size=img_size, dataset=train_set, adv_imgs=adv_imgs,
                                                        poison_rate=args.poison_rate,
                                                        trigger_mark = trigger, trigger_mask=trigger_mask,
                                                        path=poison_set_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'SleeperAgent':
        from poison_tool_box import SleeperAgent
        
        if args.dataset == 'cifar10':
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261], [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])
            
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ])
            
            trainset = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                        download=True, transform=data_transform)
            testset = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=False,
                                       download=True, transform=data_transform)
        else: raise(NotImplementedError)
        poison_generator = SleeperAgent.poison_generator(img_size=img_size, model_arch=supervisor.get_arch(args),
                                                         random_patch=False,
                                                         dataset=trainset, testset=testset,
                                                         poison_rate=args.poison_rate, path=poison_set_dir,
                                                         normalizer=normalizer, denormalizer=denormalizer,
                                                         source_class=config.source_class,
                                                         target_class=config.target_class[args.dataset])
    
    else: # 'none'
        from poison_tool_box import none
        poison_generator = none.poison_generator(img_size=img_size, dataset=train_set,
                                                path=poison_set_dir)



    if args.poison_type not in ['TaCT', 'WaNet', 'adaptive_blend', 'adaptive_patch', 'adaptive_k_way', 'belt']:
        img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
        print('poison_indicies : ', poison_indices, flush=True)
        print('[Generate Poisoned Set] Save %d Images (poisoned: %d)' % (len(label_set), len(poison_indices)), flush=True)
        sys.stdout.flush()

    else:
        if args.poison_type == 'belt':
            # BELT 返回额外的 pmark_set
            img_set, poison_indices, cover_indices, label_set, pmark_set = poison_generator.generate_poisoned_training_set()
            # 保存 pmark_set（投毒标记：0=干净, 1=投毒, 2=cover）
            pmark_path = os.path.join(poison_set_dir, 'pmarks')
            torch.save(pmark_set, pmark_path)
            print('[Generate Poisoned Set] Save %s' % pmark_path)
        else:
            img_set, poison_indices, cover_indices, label_set = poison_generator.generate_poisoned_training_set()
        
        print('poison_indicies : ', poison_indices, flush=True)
        print('[Generate Poisoned Set] Save %d Images (poisoned: %d)' % (len(label_set), len(poison_indices)), flush=True)
        sys.stdout.flush()

        cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')
        torch.save(cover_indices, cover_indices_path)
        print('[Generate Poisoned Set] Save %s' % cover_indices_path)

    # ========== [噪声增强] 对中毒图片按比例添加噪声 ==========
    if ENABLE_NOISE and len(poison_indices) > 0:
        num_noisy = int(len(poison_indices) * NOISE_RATIO)
        print(f'[噪声增强] 对 {num_noisy}/{len(poison_indices)} 个中毒图片添加 {NOISE_TYPE} 噪声 (比例: {NOISE_RATIO*100:.1f}%, 强度: {NOISE_STRENGTH}, 种子: {NOISE_SEED})')
        img_set = apply_noise_to_poisoned_images(
            img_set, 
            poison_indices=poison_indices,
            noise_ratio=NOISE_RATIO,
            noise_type=NOISE_TYPE,
            noise_strength=NOISE_STRENGTH,
            noise_seed=NOISE_SEED
        )
    # ========== [噪声增强] 结束 ==========

    # 只有 Tiny-ImageNet 使用 'imgs' 目录，其他数据集使用 'data' 目录
    if args.dataset == 'tiny_imagenet':
        img_path = os.path.join(poison_set_dir, 'imgs')
    else:
        img_path = os.path.join(poison_set_dir, 'data')
    torch.save(img_set, img_path)
    print('[Generate Poisoned Set] Save %s' % img_path)

    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)


elif args.poison_type == 'upgd':
    """
    Parameter Backdoor (UPGD):
    - 先用干净基模型生成 universal targeted perturbation（delta_raw，raw-space [0,1]）
    - 只对目标类别样本中抽取 poison_rate 比例加上 delta_raw（标签不变）
    - 保存 imgs/labels/poison_indices，同时保存 upgd_{target_class}.pth 与 upgd_meta.json
    """
    # -------------------------------------------------------------------------
    # 为什么必须要“干净基模型”（-upgd_model_path）？
    # -------------------------------------------------------------------------
    # 因为 UPGD 的触发器不是固定贴图，而是“通过优化得到的通用扰动 delta”：
    # - delta 的优化目标是：对很多输入图像 x，x+delta 都更容易被判为目标类
    # - 这个优化必须依赖一个固定的模型 f（这里用干净基模型）
    #
    # 因此必须提供 `-upgd_model_path`，指向一个干净模型的 state_dict
    #（推荐：poison_type=none, poison_rate=0 训练得到）
    # -------------------------------------------------------------------------
    if args.upgd_model_path is None:
        raise ValueError("poison_type=upgd 需要提供 -upgd_model_path（干净基模型权重路径）")
    if not os.path.exists(args.upgd_model_path):
        raise FileNotFoundError(f"UPGD 基模型权重文件不存在: {args.upgd_model_path}")

    # -------------------------------------------------------------------------
    # 归一化处理（关键工程点）
    # -------------------------------------------------------------------------
    # - delta 是在 raw 像素空间 [0,1] 中生成的（与 ToTensor 输出一致）
    # - 但模型前向通常吃 normalize 后的输入
    #
    # 所以生成 delta 时的前向流程是：
    #   x_raw -> (x_raw + delta_raw) -> clamp -> normalize(mean,std) -> model
    #
    # 测试/防御阶段由 `supervisor.get_poison_transform('upgd')` 读取保存的 delta，
    # 并用同一组 mean/std 执行：denorm -> 加delta -> clamp -> renorm。
    # -------------------------------------------------------------------------
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.247, 0.243, 0.261)
    elif args.dataset == 'gtsrb':
        mean = (0.3337, 0.3064, 0.3171)
        std = (0.2672, 0.2564, 0.2629)
    elif args.dataset == 'imagenette':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif args.dataset == 'tiny_imagenet':
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    elif args.dataset == 'mnist':
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
    elif args.dataset == 'mnistm':
        mean = (0.46, 0.46, 0.46)
        std = (0.23, 0.23, 0.23)
    else:
        raise NotImplementedError(f"UPGD 暂不支持的数据集: {args.dataset}")

    # 载入干净基模型（state_dict）
    arch = supervisor.get_arch(args)
    model = arch(num_classes=num_classes) if args.dataset != 'ember' else arch()
    state_dict = torch.load(args.upgd_model_path, map_location='cpu')
    if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = nn.DataParallel(model).cuda()
    model.eval()

    from poison_tool_box import upgd as upgd_mod
    # -------------------------------------------------------------------------
    # UPGD 生成配置（工程说明）
    # -------------------------------------------------------------------------
    # - eps：Linf 下默认按像素尺度理解（8 -> 内部换算 8/255）
    # - steps * steps_multiplier：总迭代次数（增强“通用性”）
    # - batch_size：生成 delta 时采样训练集 batch 的大小
    #
    # 注意：poison_set_dir 命名包含 eps/constraint/upgd_steps/upgd_steps_multiplier
    # 所以 create/train/test/defense 必须传一致的参数，才能定位到同一目录。
    cfg = upgd_mod.UPGDConfig(
        constraint=args.constraint,
        eps=float(args.eps),
        num_steps=int(args.upgd_steps),
        step_size=None,
        steps_multiplier=int(args.upgd_steps_multiplier),
        batch_size=int(args.upgd_batch_size),
        num_workers=0,  # 改为 0，与原始代码一致（parameter_backdoor 使用单进程加载）
        seed=2333,  # 与原始代码一致（parameter_backdoor/generate_upgd.py 默认 seed=0）
    )

    target_cls = int(config.target_class[args.dataset])
    print(f"[UPGD] target_class={target_cls}")
    print(f"[UPGD] poison_rate(overall)={args.poison_rate}")
    print(f"[UPGD] constraint={cfg.constraint}, eps={cfg.eps}, num_steps={cfg.num_steps}, multiplier={cfg.steps_multiplier}, batch_size={cfg.batch_size}")
    print(f"[UPGD] base_model={args.upgd_model_path}")

    # 1) 生成通用扰动（raw-space delta）
    delta_raw = upgd_mod.generate_upgd_delta_raw(
        model=model,
        dataset=train_set,
        target_class=target_cls,
        mean=mean,
        std=std,
        cfg=cfg,
        device=torch.device('cuda'),
    )

    # 2) 保存 delta 与元信息到 poison_set_dir，方便后续测试/防御复用
    upgd_mod.save_upgd_artifacts(
        out_dir=poison_set_dir,
        delta_raw=delta_raw,
        cfg=cfg,
        target_class=target_cls,
        base_model_path=args.upgd_model_path,
        mean=mean,
        std=std,
    )

    # 3) 生成投毒训练集：
    #    - 只投毒“目标类别样本子集”的一部分
    #    - 标签不修改（parameter-backdoor 风格）
    img_set, poison_indices, label_set = upgd_mod.poison_images_with_delta_raw(
        dataset=train_set,
        delta_raw=delta_raw,
        poison_rate=args.poison_rate,
        target_class=target_cls,
        seed=2333,
    )

    print('poison_indicies : ', poison_indices, flush=True)
    print('[Generate Poisoned Set] Save %d Images (poisoned: %d; sampled from full dataset)' % (len(label_set), len(poison_indices)), flush=True)
    sys.stdout.flush()

    # ========== [噪声增强] 对中毒图片按比例添加噪声 ==========
    if ENABLE_NOISE and len(poison_indices) > 0:
        num_noisy = int(len(poison_indices) * NOISE_RATIO)
        print(f'[噪声增强] 对 {num_noisy}/{len(poison_indices)} 个中毒图片添加 {NOISE_TYPE} 噪声 (比例: {NOISE_RATIO*100:.1f}%, 强度: {NOISE_STRENGTH}, 种子: {NOISE_SEED})')
        img_set = apply_noise_to_poisoned_images(
            img_set, 
            poison_indices=poison_indices,
            noise_ratio=NOISE_RATIO,
            noise_type=NOISE_TYPE,
            noise_strength=NOISE_STRENGTH,
            noise_seed=NOISE_SEED
        )
    # ========== [噪声增强] 结束 ==========

    # 只有 Tiny-ImageNet 使用 'imgs' 目录，其他数据集使用 'data' 目录
    if args.dataset == 'tiny_imagenet':
        img_path = os.path.join(poison_set_dir, 'imgs')
    else:
        img_path = os.path.join(poison_set_dir, 'data')
    torch.save(img_set, img_path)
    print('[Generate Poisoned Set] Save %s' % img_path)

    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)


elif args.poison_type == 'dynamic':
    """
        Since we will use the pretrained model by the original paper, here we use normalized data following 
        the original implementation.
        Download Pretrained Generator from https://github.com/VinAIResearch/input-aware-backdoor-attack-release
    """
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('[Dynamic Attack] Download pretrained generator first : https://github.com/VinAIResearch/input-aware-backdoor-attack-release')
    # Init Attacker
    from poison_tool_box import dynamic
    poison_generator = dynamic.poison_generator(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                input_channel=input_channel, normalizer=normalizer,
                                                denormalizer=denormalizer, dataset=train_set,
                                                poison_rate=args.poison_rate, path=poison_set_dir, target_class=config.target_class[args.dataset])

    # Generate Poison Data
    img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
    print('poison_indicies : ', poison_indices, flush=True)
    print('[Generate Poisoned Set] Save %d Images (poisoned: %d)' % (len(label_set), len(poison_indices)), flush=True)

    # 确保数据在 [0, 1] 范围内，避免数值不稳定（特别是对于 MNIST）
    if args.dataset == 'mnist':
        img_set = torch.clamp(img_set, 0.0, 1.0)

    # ========== [噪声增强] 对中毒图片按比例添加噪声 ==========
    if ENABLE_NOISE and len(poison_indices) > 0:
        num_noisy = int(len(poison_indices) * NOISE_RATIO)
        print(f'[噪声增强] 对 {num_noisy}/{len(poison_indices)} 个中毒图片添加 {NOISE_TYPE} 噪声 (比例: {NOISE_RATIO*100:.1f}%, 强度: {NOISE_STRENGTH}, 种子: {NOISE_SEED})')
        img_set = apply_noise_to_poisoned_images(
            img_set, 
            poison_indices=poison_indices,
            noise_ratio=NOISE_RATIO,
            noise_type=NOISE_TYPE,
            noise_strength=NOISE_STRENGTH,
            noise_seed=NOISE_SEED
        )
    # ========== [噪声增强] 结束 ==========

    # 只有 Tiny-ImageNet 使用 'imgs' 目录，其他数据集使用 'data' 目录
    if args.dataset == 'tiny_imagenet':
        img_path = os.path.join(poison_set_dir, 'imgs')
    else:
        img_path = os.path.join(poison_set_dir, 'data')
    torch.save(img_set, img_path)
    print('[Generate Poisoned Set] Save %s' % img_path)
    
    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

elif args.poison_type == 'ISSBA':
    # if not os.path.exists(ckpt_path):
    #     raise NotImplementedError('[ISSBA Attack] Download pretrained encoder and decoder first: https://github.com/')
    
    # Init Secret
    secret_size = 20
    secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist())
    secret_path = os.path.join(poison_set_dir, 'secret')
    torch.save(secret, secret_path)
    print('[Generate Poisoned Set] Save %s' % secret_path)
    
    # Init Attacker
    from poison_tool_box import ISSBA
    poison_generator = ISSBA.poison_generator(ckpt_path=ckpt_path, secret=secret, dataset=train_set, enc_height=img_size, enc_width=img_size, enc_in_channel=input_channel,
                                                poison_rate=args.poison_rate, path=poison_set_dir, target_class=config.target_class[args.dataset])

    # Generate Poison Data
    img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
    print('poison_indicies : ', poison_indices, flush=True)
    print('[Generate Poisoned Set] Save %d Images (poisoned: %d)' % (len(label_set), len(poison_indices)), flush=True)

    # 确保数据在 [0, 1] 范围内，避免数值不稳定（特别是对于 MNIST）
    if args.dataset == 'mnist':
        img_set = torch.clamp(img_set, 0.0, 1.0)

    # ========== [噪声增强] 对中毒图片按比例添加噪声 ==========
    if ENABLE_NOISE and len(poison_indices) > 0:
        num_noisy = int(len(poison_indices) * NOISE_RATIO)
        print(f'[噪声增强] 对 {num_noisy}/{len(poison_indices)} 个中毒图片添加 {NOISE_TYPE} 噪声 (比例: {NOISE_RATIO*100:.1f}%, 强度: {NOISE_STRENGTH}, 种子: {NOISE_SEED})')
        img_set = apply_noise_to_poisoned_images(
            img_set, 
            poison_indices=poison_indices,
            noise_ratio=NOISE_RATIO,
            noise_type=NOISE_TYPE,
            noise_strength=NOISE_STRENGTH,
            noise_seed=NOISE_SEED
        )
    # ========== [噪声增强] 结束 ==========

    # 只有 Tiny-ImageNet 使用 'imgs' 目录，其他数据集使用 'data' 目录
    if args.dataset == 'tiny_imagenet':
        img_path = os.path.join(poison_set_dir, 'imgs')
    else:
        img_path = os.path.join(poison_set_dir, 'data')
    torch.save(img_set, img_path)
    print('[Generate Poisoned Set] Save %s' % img_path)
    
    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

else:
    raise NotImplementedError('%s not defined' % args.poison_type)