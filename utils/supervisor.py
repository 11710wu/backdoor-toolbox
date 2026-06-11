import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import copy
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import config
from utils import default_args


def get_label_mode(args):
    mode = getattr(args, 'label_mode', 'clean') or 'clean'
    if mode not in ('clean', 'all2one'):
        raise ValueError(f"Unsupported label_mode: {mode}")
    return mode

"""
In our defensive setting, we assume a poisoned training set and a small clean 
set at hand, i.e. we train base model jointly with the poisoned set and 
the shifted set (constructed based on the small clean set).

On the other hand, we also prepare a clean test set (usually larger than the 
small clean set used for defense in our experiments). But note that, this set is 
not used for defense, it is used for debug and evaluation!

Below we implement tools to take use of the additional clean test set for debug & evaluation.
"""


def get_cleansed_set_indices_dir(args):
    poison_set_dir = get_poison_set_dir(args)
    if args.cleanser == 'CT':  # confusion training
        return os.path.join(poison_set_dir, f'cleansed_set_indices_seed={args.seed}')
    else:
        return os.path.join(poison_set_dir, f'cleansed_set_indices_other_{args.cleanser}_seed={args.seed}')


def get_model_name(args, cleanse=False, defense=False):
    # `args.model_path` > `args.model` > by default 'full_base'
    if hasattr(args, 'model_path') and args.model_path is not None:
        model_name = os.path.basename(args.model_path)
    elif hasattr(args, 'model') and args.model is not None:
        # 当使用 -model 参数且启用架构记录时，直接使用架构名称，避免重复
        if config.record_model_arch:
            model_name = f'{get_arch(args).__name__}.pt'
        else:
            model_name = f'{args.model}.pt'
    elif args.poison_type in ['trojannn']:
        model_name = f'{args.dataset}_{args.poison_type}_seed={args.seed}.pt'
    elif args.poison_type == 'SRA':
        model_name = f'{args.dataset}_{args.poison_type}_seed={args.seed}.pt'
    elif args.poison_type == 'BadEncoder':
        if args.dataset == 'gtsrb':
            model_name = 'BadEncoder_cifar2gtsrb.pth'
        else:
            raise NotImplementedError()
    else:
        if args.no_aug:
            model_name = f'full_base_no_aug_seed={args.seed}.pt'
        else:
            model_name = f'full_base_aug_seed={args.seed}.pt'

    if cleanse and hasattr(args, 'cleanser') and args.cleanser is not None:
        model_name = f"cleansed_{args.cleanser}_{model_name}"
    elif defense and hasattr(args, 'defense') and args.defense is not None:
        model_name = f"defended_{args.defense}_{model_name}"

    # 对于使用 -model 参数的情况，已经在上面包含了架构名称，不需要再添加
    if config.record_model_arch and not (hasattr(args, 'model') and args.model is not None):
        model_name = f"{get_arch(args).__name__}_{model_name}"
    return model_name


def get_model_dir(args, cleanse=False, defense=False):
    if hasattr(args, 'model_path') and args.model_path is not None:
        return args.model_path
    else:
        return f"{get_poison_set_dir(args)}/{get_model_name(args, cleanse=cleanse, defense=defense)}"


INPUT_NOISE_TYPES = {'none', 'gaussian', 'uniform', 'salt_pepper', 'speckle'}


def get_input_noise_config(args):
    noise_type = getattr(args, 'input_noise_type', 'none') or 'none'
    noise_level = getattr(args, 'input_noise_level', 0.0)
    noise_seed = getattr(args, 'input_noise_seed', config.poison_seed)

    try:
        noise_level = float(noise_level)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid input_noise_level: {noise_level}")

    if noise_type not in INPUT_NOISE_TYPES:
        raise ValueError(f"Unsupported input_noise_type: {noise_type}")
    if noise_level < 0:
        raise ValueError(f"input_noise_level must be >= 0, got {noise_level}")

    if noise_type == 'none' or noise_level == 0.0:
        return 'none', 0.0, int(noise_seed)

    if getattr(args, 'dataset', None) != 'cifar10':
        raise ValueError("input noise is only supported for dataset='cifar10' in this experiment")

    return noise_type, noise_level, int(noise_seed)


def get_input_noise_suffix(args):
    noise_type, noise_level, _ = get_input_noise_config(args)
    if noise_type == 'none' or noise_level == 0.0:
        return ''
    return f'_noise={noise_type}_level={noise_level:.3f}'


def get_dir_core(args, include_model_name=False, include_poison_seed=False):
    ratio = '%.3f' % args.poison_rate
    # ratio = '%.1f' % (args.poison_rate * 100) + '%'
    if args.poison_type in ['trojannn', 'BadEncoder', 'SRA']:
        dir_core = '%s_%s' % (args.dataset, args.poison_type)
    elif args.poison_type == 'blend' or args.poison_type == 'basic' or args.poison_type == 'clean_label' or args.poison_type == 'badnet':
        blend_alpha = '%.3f' % args.alpha
        if args.poison_type == 'badnet':
            dir_core = '%s_%s_%s_alpha=%s' % (args.dataset, args.poison_type, ratio, blend_alpha)
        else:
            dir_core = '%s_%s_%s_alpha=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_alpha=%s_cover=%s_trigger=%s' % (
        args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'adaptive_patch':
        # ========== [Adaptive Patch Alpha参数修改] 开始 ==========
        # 修改：始终在目录名中包含alpha参数，便于区分不同配置
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_alpha=%s_cover=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, cover_rate)
        # ========== [Adaptive Patch Alpha参数修改] 结束 ==========
    elif args.poison_type == 'TaCT' or args.poison_type == 'WaNet':
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    elif args.poison_type == 'belt':
        belt_alpha = '%.3f' % getattr(args, 'alpha', 1.0)
        cover_rate = '%.3f' % getattr(args, 'cover_rate', 0.5)
        mask_rate = '%.3f' % getattr(args, 'mask_rate', 0.2)
        # BELT 不使用外部 trigger 文件，其 trigger 由 generate_belt_trigger() 内部生成，
        # 故此处与 get_poison_set_dir 保持一致，不拼接 _trigger= 字段
        dir_core = '%s_%s_%s_alpha=%s_cover=%s_mask=%s' % (
            args.dataset, args.poison_type, ratio, belt_alpha, cover_rate, mask_rate)
    elif args.poison_type == 'SIG':
        delta_param = int(getattr(args, 'delta', 30))
        f_param = int(getattr(args, 'f', 6))
        label_mode = get_label_mode(args)
        dir_core = '%s_%s_%s_delta=%s_f=%s_mode=%s' % (
            args.dataset, args.poison_type, ratio, delta_param, f_param, label_mode
        )
    elif args.poison_type == 'upgd':
        # ---------------------------------------------------------------------
        # Parameter backdoor (UPGD)
        # ---------------------------------------------------------------------
        # UPGD 不依赖固定的图案触发器文件（trigger_name='none'），而是：
        # 1) 用一个干净基模型生成 universal targeted perturbation（delta）
        # 2) 把 delta 加到训练集中的一部分样本（clean 或 all-to-one）
        # 3) 测试/防御阶段再把 delta 作为“触发器”加到输入上
        #
        # 目录命名需要包含 UPGD 的关键生成参数，否则不同 UPGD 设置会覆盖同一个目录。
        # 这里至少包含：eps / constraint / steps / steps_multiplier
        upgd_eps = getattr(args, 'eps', 8)
        upgd_constraint = getattr(args, 'constraint', 'Linf')
        upgd_steps = getattr(args, 'upgd_steps', 100)
        upgd_steps_multiplier = getattr(args, 'upgd_steps_multiplier', 5)
        label_mode = get_label_mode(args)
        dir_core = '%s_%s_%s_eps=%s_constraint=%s_steps=%s_mode=%s' % (
            args.dataset, args.poison_type, ratio, str(upgd_eps), str(upgd_constraint), str(upgd_steps), label_mode
        )
        dir_core = f'{dir_core}_mult={upgd_steps_multiplier}'
    else:
        dir_core = '%s_%s_%s' % (args.dataset, args.poison_type, ratio)

    dir_core = f'{dir_core}{get_input_noise_suffix(args)}'

    if include_model_name:
        dir_core = f'{dir_core}_{get_model_name(args)}'
    if include_poison_seed:
        dir_core = f'{dir_core}_poison_seed={config.poison_seed}'
    if config.record_model_arch:
        dir_core = f'{dir_core}_arch={get_arch(args).__name__}'
    return dir_core


def get_poison_set_dir(args):
    ratio = '%.3f' % args.poison_rate
    # ratio = '%.1f' % (args.poison_rate * 100) + '%'
    if args.poison_type in ['trojannn', 'BadEncoder', 'SRA']:
        poison_set_dir = 'models'
        return poison_set_dir
    elif args.poison_type == 'blend' or args.poison_type == 'basic' or args.poison_type == 'clean_label' or args.poison_type == 'badnet':
        blend_alpha = '%.3f' % args.alpha
        if args.poison_type == 'badnet':
            poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s' % (
            args.dataset, args.poison_type, ratio, blend_alpha)
        else:
            poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_trigger=%s' % (
            args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_cover=%s_trigger=%s' % (
        args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'adaptive_patch':
        # ========== [Adaptive Patch Alpha参数修改] 开始 ==========
        # 修改：始终在目录路径中包含alpha参数，便于区分不同配置
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_cover=%s' % (
            args.dataset, args.poison_type, ratio, blend_alpha, cover_rate)
        # ========== [Adaptive Patch Alpha参数修改] 结束 ==========
    elif args.poison_type == 'TaCT':
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    elif args.poison_type == 'WaNet':
        # ========== [WaNet目录路径修改] 开始 ==========
        # 对于WaNet攻击，需要在目录路径中包含s和k参数
        # 原因：不同的s和k会生成不同的noise_grid，对应不同的中毒数据集
        # 例如：poisoned_train_set/cifar10/WaNet_0.010_cover=0.020_s=0.4_k=4
        #      poisoned_train_set/cifar10/WaNet_0.010_cover=0.020_s=0.45_k=4
        # 这样可以确保：
        # 1. create_poisoned_set.py 根据s和k创建不同的数据集目录
        # 2. train_on_poisoned_set.py 根据s和k加载对应的数据集目录
        # 3. other_defense.py 根据s和k检测对应的模型
        cover_rate = '%.3f' % args.cover_rate
        # 格式化s参数：保留最多3位小数，去除尾部的0（例如：0.5而不是0.500）
        s_param = ('%.3f' % getattr(args, 's', 0.5)).rstrip('0').rstrip('.')
        # 获取k参数（默认为4）
        k_param = getattr(args, 'k', 4)
        # 构建包含s和k的目录路径
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_cover=%s_s=%s_k=%s' % (args.dataset, args.poison_type, ratio, cover_rate, s_param, k_param)
        # ========== [WaNet目录路径修改] 结束 ==========
    elif args.poison_type == 'SIG':
        # ========== [SIG目录路径修改] 开始 ==========
        # 对于SIG攻击，需要在目录路径中包含delta和f参数
        # 原因：不同的delta和f会生成不同的频域扰动，对应不同的中毒数据集
        # 例如：poisoned_train_set/cifar10/SIG_0.020_delta=20_f=6_mode=clean
        #      poisoned_train_set/cifar10/SIG_0.020_delta=30_f=6_mode=all2one
        # 这样可以确保：
        # 1. create_poisoned_set.py 根据delta和f创建不同的数据集目录
        # 2. train_on_poisoned_set.py 根据delta和f加载对应的数据集目录
        # 3. other_defense.py 根据delta和f检测对应的模型
        # 获取delta参数（默认为30，注意：这里是原始值，会在create_poisoned_set.py中除以255）
        delta_param = int(getattr(args, 'delta', 30))
        # 获取f参数（频率参数，默认为6）
        f_param = int(getattr(args, 'f', 6))
        label_mode = get_label_mode(args)
        # 构建包含delta和f的目录路径
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_delta=%s_f=%s_mode=%s' % (args.dataset, args.poison_type, ratio, delta_param, f_param, label_mode)
        # ========== [SIG目录路径修改] 结束 ==========
    elif args.poison_type == 'upgd':
        # Parameter backdoor (UPGD): include eps/constraint/steps(/mult) to avoid collisions
        upgd_eps = getattr(args, 'eps', 8)
        upgd_constraint = getattr(args, 'constraint', 'Linf')
        upgd_steps = getattr(args, 'upgd_steps', 100)
        upgd_steps_multiplier = getattr(args, 'upgd_steps_multiplier', 5)
        label_mode = get_label_mode(args)
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_eps=%s_constraint=%s_steps=%s_mode=%s' % (
            args.dataset, args.poison_type, ratio, str(upgd_eps), str(upgd_constraint), str(upgd_steps), label_mode
        )
        poison_set_dir = f'{poison_set_dir}_mult={upgd_steps_multiplier}'
    elif args.poison_type == 'belt':
        belt_alpha = '%.3f' % getattr(args, 'alpha', 1.0)
        cover_rate = '%.3f' % getattr(args, 'cover_rate', 0.5)
        mask_rate = '%.3f' % getattr(args, 'mask_rate', 0.2)
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_cover=%s_mask=%s' % (
            args.dataset, args.poison_type, ratio, belt_alpha, cover_rate, mask_rate)
    else:
        poison_set_dir = 'poisoned_train_set/%s/%s_%s' % (args.dataset, args.poison_type, ratio)

    poison_set_dir = f'{poison_set_dir}{get_input_noise_suffix(args)}'

    if config.record_poison_seed: poison_set_dir = f'{poison_set_dir}_poison_seed={config.poison_seed}'  # debug
    if config.record_model_arch:
        # 获取实际使用的模型架构名称
        arch_name = get_arch(args).__name__
        poison_set_dir = f'{poison_set_dir}_arch={arch_name}'  # 在目录名中包含模型架构名称（如 DenseNet121）
    return poison_set_dir


def get_arch(args):
    if args.poison_type == 'BadEncoder':
        if args.dataset == 'gtsrb':
            from utils.BadEncoder_model import CIFAR2GTSRB
            return CIFAR2GTSRB
        else:
            raise NotImplementedError
    if args.poison_type == 'SRA':
        if args.dataset == 'cifar10':
            if 'resnet' in config.arch[args.dataset].__name__.lower():
                from utils.SRA.cifar_10.resnet import resnet110
                return resnet110
            elif 'vgg' in config.arch[args.dataset].__name__:
                from utils.SRA.cifar_10.vgg import vgg16_bn
                return vgg16_bn
            elif 'mobilenet' in config.arch[args.dataset].__name__:
                from utils.SRA.cifar_10.mobilenetv2 import mobilenetv2
                return mobilenetv2
        elif args.dataset == 'imagenet':
            if 'vgg' in config.arch[args.dataset].__name__:
                from utils.SRA.imagenet.vgg import vgg16_bn
                return vgg16_bn
            elif 'resnet' in config.arch[args.dataset].__name__:
                from utils.SRA.imagenet.resnet import resnet101
                return resnet101
            elif 'mobilenetv2' in config.arch[args.dataset].__name__:
                from utils.SRA.imagenet.mobilenetv2 import mobilenet_v2
                return mobilenet_v2
        else:
            raise NotImplementedError
    if hasattr(args, 'defense') and args.defense == 'NONE':
        from other_defenses_tool_box.none.resnet import resnet18
        return resnet18
    else:
        # 支持模型选择
        if hasattr(args, 'model') and args.model is not None:
            if args.model == 'vgg19_bn':
                from utils import vgg
                # 为不同数据集使用专门的模型函数
                if args.dataset == 'cifar10':
                    return vgg.vgg19_bn_cifar10
                elif args.dataset == 'tiny_imagenet':
                    return vgg.vgg19_bn_tiny_imagenet
                elif args.dataset == 'mnistm':
                    return vgg.vgg19_bn_mnistm
                else:
                    return vgg.vgg19_bn
            elif args.model == 'mobilenetv2':
                from utils import mobilenetv2
                # 为不同数据集使用专门的模型函数
                if args.dataset == 'cifar10':
                    return mobilenetv2.mobilenetv2_cifar10
                elif args.dataset == 'tiny_imagenet':
                    return mobilenetv2.mobilenetv2_tiny_imagenet
                elif args.dataset == 'mnistm':
                    return mobilenetv2.mobilenetv2_mnistm
                else:
                    return mobilenetv2.mobilenetv2
            elif args.model == 'resnet18':
                from utils import resnet
                # 为不同数据集使用专门的模型函数
                if args.dataset == 'cifar10':
                    return resnet.ResNet18_cifar10
                elif args.dataset == 'tiny_imagenet':
                    return resnet.ResNet18_tiny_imagenet
                elif args.dataset == 'mnistm':
                    return resnet.ResNet18_mnistm
                else:
                    return resnet.ResNet18
            elif args.model == 'resnet34':
                from utils import resnet
                if args.dataset == 'tiny_imagenet':
                    return resnet.ResNet34_tiny_imagenet
                else:
                    return resnet.ResNet34
            elif args.model == 'small_cnn':
                if args.dataset != 'cifar10':
                    raise ValueError("small_cnn is only supported for dataset='cifar10'")
                from utils import small_cnn
                return small_cnn.SmallCNN_cifar10
            elif args.model == 'densenet121':
                from utils import densenet
                # 为不同数据集使用专门的模型函数
                if args.dataset == 'cifar10':
                    return densenet.densenet121_cifar10
                elif args.dataset == 'gtsrb':
                    return densenet.densenet121_gtsrb
                elif args.dataset == 'tiny_imagenet':
                    return densenet.densenet121_tiny_imagenet
                elif args.dataset == 'mnistm':
                    return densenet.densenet121_mnistm
                elif args.dataset == 'imagenette':
                    return densenet.densenet121_imagenette
                elif args.dataset == 'imagenet':
                    return densenet.densenet121_imagenet
                else:
                    # 默认使用 CIFAR-10 版本（适配 32x32 输入）
                    return densenet.densenet121_cifar10
            else:
                raise ValueError(
                    f"Unsupported model: {args.model}. "
                    "Supported models: vgg19_bn, mobilenetv2, resnet18, resnet34, "
                    "small_cnn, densenet121"
                )
        else:
            return config.arch[args.dataset]


def get_transforms(args):
    if args.dataset == 'gtsrb':
        # ========== [UPGD 特殊处理] UPGD 默认不使用归一化（与原始代码一致）==========
        if args.poison_type == 'upgd':
            # UPGD 必须不使用归一化（与 parameter_backdoor 原始代码一致）
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        elif args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            normalizer = transforms.Compose([
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                     (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
            ])

        if args.poison_type == 'BadEncoder':  # use CIFAR10's data transform for BadEncoder
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])

    elif args.dataset == 'cifar10':
        # ========== [UPGD 特殊处理] UPGD 默认不使用归一化（与原始代码一致）==========
        if args.poison_type == 'upgd':
            # UPGD 必须不使用归一化（与 parameter_backdoor 原始代码一致）
            # 原因：delta 是在 raw [0,1] 空间生成的，模型训练和测试都应使用 raw 数据
            data_transform_aug = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
         
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        # ========== [none 特殊处理] none 类型（干净模型）使用与 UPGD 一致的数据增强 ==========
        # 原因：用于生成 UPGD delta 的干净基模型需要在 [0,1] raw 数据上训练，使用相同的数据增强
        # 与原始代码 parameter_backdoor/utils.py 一致
        # ========== [BELT 特殊处理] 与原始 BELT 代码一致（无归一化！）==========
        # 原始代码（BadNet_BELT.py）使用：
        #   - RandomCrop(32, padding=2)  ← 注意是 padding=2，不是 4
        #   - 没有 RandomHorizontalFlip
        #   - 没有 Normalize（直接在 [0,1] 空间训练）
        elif args.poison_type == 'belt':
            data_transform_aug = transforms.Compose([
                transforms.RandomCrop(32, 2),  # padding=2，与原始 BELT 代码一致
                transforms.ToTensor(),
                # 无 Normalize，与原始 BELT 代码一致
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                # 无 Normalize，与原始 BELT 代码一致
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        elif args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])

        if args.poison_type == 'SRA':
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     [1 / 0.229, 1 / 0.224, 1 / 0.225])
            ])
    elif args.dataset == 'imagenette':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomCrop(224, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomCrop(224, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     [1 / 0.229, 1 / 0.224, 1 / 0.225])
            ])
    # ========== [Tiny ImageNet 支持] 添加数据变换配置 ==========
    # Tiny ImageNet: 使用原始 64×64 尺寸，数据增强和归一化修改为与 BackdoorBench 一致
    # 数据增强: RandomCrop(64, 4)（移除 ColorJitter，与 BackdoorBench 一致）
    # 归一化: 使用 Tiny ImageNet 特定参数（与 BackdoorBench 一致）
    elif args.dataset == 'tiny_imagenet':
        # ========== [BELT 特殊处理] BELT 强制不使用归一化（与原始代码一致）==========
        if args.poison_type == 'belt':
            # BELT 必须不使用归一化（与原始 BELT 代码一致）
            # 原始代码使用 RandomCrop(32, 2)，但 Tiny ImageNet 是 64x64，所以使用 RandomCrop(64, 2)
            data_transform_aug = transforms.Compose([
                transforms.RandomCrop(64, 2),  # padding=2，与原始 BELT 代码一致（适配 64x64）
                transforms.ToTensor(),
                # 无 Normalize，与原始 BELT 代码一致
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                # 无 Normalize，与原始 BELT 代码一致
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])

        # ========== [UPGD 特殊处理] UPGD 默认不使用归一化（与原始代码一致）==========
        elif args.poison_type == 'upgd':
            # UPGD 必须不使用归一化（与 parameter_backdoor 原始代码一致）
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        elif args.no_normalize:
            # 原始图片已经是 64×64，不需要 Resize
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),  # 使用 64×64 的 RandomCrop，与 BackdoorBench 一致
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()  # 原始图片已经是 64×64，不需要 Resize
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            # 数据增强：与 BackdoorBench 一致（RandomCrop，无 ColorJitter）
            # 归一化：使用 Tiny ImageNet 特定参数（与 BackdoorBench 一致）
            # 图像尺寸：使用原始 64×64 尺寸（与 BackdoorBench 一致）
            # 注意：原始图片已经是 64×64，不需要 Resize，避免图片模糊
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),  # 使用 64×64 的 RandomCrop（与 BackdoorBench 一致）
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])  # Tiny ImageNet 特定归一化参数
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),  # 原始图片已经是 64×64，不需要 Resize
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4802 / 0.2302, -0.4481 / 0.2265, -0.3975 / 0.2262],
                                     [1 / 0.2302, 1 / 0.2265, 1 / 0.2262])
            ])
    # ========== [Tiny ImageNet 支持] 结束 ==========
    # ========== [MNIST 支持] 添加数据变换配置 ==========
    # MNIST: 28×28 单通道灰度图，转换为三通道以支持 RGB 触发器
    # 将单通道复制为三通道，对原图内容无影响，但允许使用 RGB 触发器
    elif args.dataset == 'mnist':
        if args.no_normalize:
            # MNIST 数据在 create_poisoned_set.py 和 create_clean_set.py 中已经转换为3通道
            # 因此训练时加载的数据已经是3通道，不需要再次应用 Lambda
            data_transform_aug = transforms.Compose([
                transforms.ToTensor(),
                # 注意：不包含 Lambda，因为保存的数据已经是3通道
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                # 注意：不包含 Lambda，因为保存的数据已经是3通道
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            # MNIST 标准归一化: mean=[0.1307], std=[0.3081]
            # 转换为三通道后，归一化参数需要重复三次
            # 与 CIFAR-10 一致：不使用数据增强（data_transform_aug = data_transform）
            # MNIST 数据在 create_poisoned_set.py 中已经转换为3通道并保存
            # 因此训练时加载的数据已经是3通道，不需要再次应用 Lambda
            # 但是，如果从原始数据集加载（如测试集），仍然需要转换
            # 为了兼容两种情况，我们创建两个 transform：
            # 1. data_transform: 用于已保存的3通道数据（不包含 Lambda）
            # 2. data_transform_from_raw: 用于从原始数据集加载（包含 Lambda）
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                # 注意：不包含 Lambda，因为保存的数据已经是3通道
                transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])  # 三通道归一化
            ])
            data_transform_aug = data_transform  # 与 CIFAR-10 一致：不使用数据增强
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])  # 三通道归一化
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.1307 / 0.3081, -0.1307 / 0.3081, -0.1307 / 0.3081],
                                     [1 / 0.3081, 1 / 0.3081, 1 / 0.3081])
            ])
    # ========== [MNIST 支持] 结束 ==========
    # ========== [MNIST-M 支持] 添加数据变换配置 ==========
    # MNIST-M: 28×28 RGB 彩色图，使用 MNIST-M 特定的归一化参数
    elif args.dataset == 'mnistm':
        # ========== [BELT 特殊处理] BELT 强制不使用归一化（与原始代码一致）==========
        if args.poison_type == 'belt':
            # BELT 必须不使用归一化（与原始 BELT 代码一致）
            # 原始代码使用 RandomCrop(32, 2)，但 MNIST-M 是 28x28，需要先 resize 到 32x32 才能 crop
            data_transform_aug = transforms.Compose([ 
                transforms.ToTensor(),
                # 无 Normalize，与原始 BELT 代码一致
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                # 无 Normalize，与原始 BELT 代码一致
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        # ========== [UPGD 特殊处理] UPGD 默认不使用归一化（与原始代码一致）==========
        elif args.poison_type == 'upgd':
            # UPGD 必须不使用归一化（与 parameter_backdoor 原始代码一致）
            # MNIST-M 是 28x28，不需要 resize 和 crop，直接使用 ToTensor
            data_transform_aug = transforms.Compose([
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        elif args.no_normalize:
            # MNIST-M 数据已经是 RGB 三通道，不需要转换
            data_transform_aug = transforms.Compose([
                transforms.ToTensor(),
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            # MNIST-M 归一化参数: mean=[0.46, 0.46, 0.46], std=[0.23, 0.23, 0.23]
            # 不使用数据增强（data_transform_aug = data_transform）
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.46, 0.46, 0.46], [0.23, 0.23, 0.23])  # MNIST-M 特定归一化参数
            ])
            data_transform_aug = data_transform  # 不使用数据增强
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.46, 0.46, 0.46], [0.23, 0.23, 0.23])  # MNIST-M 特定归一化参数
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.46, 0.46, 0.46], [0.23, 0.23, 0.23])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.46 / 0.23, -0.46 / 0.23, -0.46 / 0.23],
                                     [1 / 0.23, 1 / 0.23, 1 / 0.23])
            ])
    # ========== [MNIST-M 支持] 结束 ==========
    elif args.dataset == 'imagenet':
        data_transform_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.CenterCrop(224),
        ])
        trigger_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.CenterCrop(224),
        ])
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])
        ])

    elif args.dataset == 'ember':
        data_transform_aug = data_transform = trigger_transform = normalizer = denormalizer = None
    else:
        raise NotImplementedError()

    return data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer


def get_poison_transform(poison_type, dataset_name, target_class, source_class=1, cover_classes=[5, 7],
                         is_normalized_input=False, trigger_transform=None,
                         alpha=0.2, trigger_name=None, args=None):
    # source class will be used for TaCT poison

    # ========== [Tiny ImageNet 支持] 添加触发器处理逻辑 ==========
    # 根据数据集设置触发器默认值和图像尺寸
    if trigger_name is None:
        if dataset_name not in ['imagenette', 'tiny_imagenet', 'mnist', 'mnistm']:
            trigger_name = config.trigger_default[dataset_name][poison_type]
        elif dataset_name == 'imagenette':
            if poison_type == 'badnet':
                trigger_name = 'badnet_high_res.png'
            else:
                raise NotImplementedError('%s not implemented for imagenette' % poison_type)
        elif dataset_name == 'tiny_imagenet':
            # Tiny ImageNet 使用配置文件中的默认触发器（64×64）
            trigger_name = config.trigger_default[dataset_name][poison_type]
        elif dataset_name == 'mnist':
            # MNIST 使用配置文件中的默认触发器（28×28）
            trigger_name = config.trigger_default[dataset_name][poison_type]
        elif dataset_name == 'mnistm':
            # MNIST-M 使用配置文件中的默认触发器（28×28，与 MNIST 相同）
            trigger_name = config.trigger_default[dataset_name][poison_type]

    # 根据数据集设置图像尺寸（用于触发器 resize）
    if dataset_name in ['gtsrb', 'cifar10', 'cifar100']:
        img_size = 32
    elif dataset_name == 'tiny_imagenet':
        img_size = 64  # Tiny ImageNet 使用原始 64×64 尺寸
    elif dataset_name in ['mnist', 'mnistm']:
        img_size = 28  # MNIST 和 MNIST-M 使用 28×28 尺寸
    elif dataset_name == 'imagenette' or dataset_name == 'imagenet':
        img_size = 224
    # ========== [Tiny ImageNet 支持] 结束 ==========
    # ========== [MNIST 支持] 结束 ==========
    else:
        raise NotImplementedError('<Undefined> Dataset = %s' % dataset_name)

    if dataset_name == 'cifar10':
        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                 [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])
        num_classes = 10
    elif dataset_name == 'gtsrb':
        normalizer = transforms.Compose([
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                 (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
        ])
        num_classes = 43
    elif dataset_name == 'imagenette' or dataset_name == 'imagenet':
        normalizer = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
                                 (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)),
        ])
        num_classes = 10
    # ========== [Tiny ImageNet 支持] 添加 Tiny ImageNet 归一化处理 ==========
    elif dataset_name == 'tiny_imagenet':
        # 使用 Tiny ImageNet 特定归一化参数（与 get_transforms 一致）
        normalizer = transforms.Compose([
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])  # Tiny ImageNet 特定归一化参数
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4802 / 0.2302, -0.4481 / 0.2265, -0.3975 / 0.2262],
                                 [1.0 / 0.2302, 1.0 / 0.2265, 1.0 / 0.2262]),
        ])
        num_classes = 200  # Tiny ImageNet 有 200 个类别
    # ========== [Tiny ImageNet 支持] 结束 ==========
    # ========== [MNIST 支持] 添加 MNIST 归一化处理 ==========
    elif dataset_name == 'mnist':
        # 使用 MNIST 标准归一化参数（三通道版本，因为 MNIST 在训练时已转换为3通道）
        # 与 get_transforms 一致：MNIST 使用三通道归一化
        normalizer = transforms.Compose([
            transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081])  # MNIST 三通道归一化参数
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.1307 / 0.3081, -0.1307 / 0.3081, -0.1307 / 0.3081],
                                 [1.0 / 0.3081, 1.0 / 0.3081, 1.0 / 0.3081]),
        ])
        num_classes = 10  # MNIST 有 10 个类别
    # ========== [MNIST 支持] 结束 ==========
    # ========== [MNIST-M 支持] 添加 MNIST-M 归一化处理 ==========
    elif dataset_name == 'mnistm':
        # 使用 MNIST-M 特定归一化参数（与 get_transforms 一致）
        # MNIST-M 归一化参数: Mean=[0.46, 0.46, 0.46], Std=[0.23, 0.23, 0.23]
        normalizer = transforms.Compose([
            transforms.Normalize([0.46, 0.46, 0.46], [0.23, 0.23, 0.23])  # MNIST-M 归一化参数
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.46 / 0.23, -0.46 / 0.23, -0.46 / 0.23],
                                 [1.0 / 0.23, 1.0 / 0.23, 1.0 / 0.23]),
        ])
        num_classes = 10  # MNIST-M 有 10 个类别（与 MNIST 相同）
    # ========== [MNIST-M 支持] 结束 ==========
    else:
        raise Exception("Invalid Dataset")

    poison_transform = None
    trigger = None
    trigger_mask = None

    # -------------------------------------------------------------------------
    # UPGD / Parameter Backdoor（工程接入点）
    # -------------------------------------------------------------------------
    # 传统 patch 类攻击在这里会加载 trigger 图片并构造 poison_transform；
    # 而 UPGD 的“触发器”是训练阶段生成并保存的通用扰动 delta（raw 像素空间 [0,1]）。
    #
    # 这里负责：
    # 1) 根据 dataset_name + args(包含 eps/constraint/steps/mult 等) 定位 poison_set_dir
    # 2) 从 poison_set_dir 读取 upgd_{target_class}.pth
    # 3) 返回一个 poison_transform：在测试/防御阶段一致地加 delta 并把标签设为目标类（用于 ASR）
    # -------------------------------------------------------------------------
    if poison_type == 'upgd':
        if args is None:
            raise ValueError("poison_type='upgd' requires `args` to locate saved UPGD delta.")

        # 用 dataset_name（而不是 args.dataset）定位对应的 poison_set_dir
        # 这是为了适配跨数据集测试脚本：source_dataset 与 dataset_name 可能不同
        # 
        # 重要：对于跨数据集测试，delta 是在训练数据集（args.dataset）上生成的，
        # 所以需要从训练数据集的目录读取 delta，而不是测试数据集的目录。
        # 如果 dataset_name 是跨数据集测试的目标数据集（如 'stl10', 'tiny_imagenet'），
        # 且 args.dataset 是源数据集（训练数据集），则使用 args.dataset 来定位 delta。
        dir_args = copy.deepcopy(args)
        # 对于跨数据集测试，delta 路径应该基于训练数据集（args.dataset），而不是测试数据集（dataset_name）
        # 
        # 跨数据集测试的固定配对关系（参考其他攻击方法的处理方式）：
        # - cifar10 → stl10: CIFAR-10 训练的模型在 STL-10 上测试
        #   - test_stl10.py 传入 dataset_name='cifar10'（源数据集）
        #   - STL-10 数据在测试脚本中被下采样到 32x32 并使用 CIFAR-10 的归一化参数
        #   - 所以这里使用 CIFAR-10 的归一化参数
        # - tiny_imagenet → tiny_imagenet-c: Tiny ImageNet 训练的模型在 Tiny ImageNet-C 上测试
        #   - test_tiny_imagenet.py 传入 dataset_name=source_dataset（如 'tiny_imagenet' 或 'cifar10'）
        #   - 测试脚本根据源数据集选择归一化参数（已在数据加载时处理）
        #   - 所以这里使用源数据集的归一化参数
        # 关键点：其他攻击方法都是传入源数据集名称，归一化参数也使用源数据集的
        # （因为测试脚本已经在数据加载时根据源数据集处理了归一化）
        # 所以 UPGD 也应该采用相同的逻辑：使用 dataset_name（源数据集）的归一化参数
        # 
        # Delta 路径定位：UPGD 的 delta 文件保存在训练数据集目录中
        # 优先使用 args.dataset（训练数据集）来构建路径，因为 delta 是在训练数据集上生成的
        # 归一化参数也必须使用训练数据集（args.dataset），因为：
        # - 模型是在训练数据集上训练的，使用的是训练数据集的归一化参数
        # - 测试脚本会将测试数据归一化为训练数据集的归一化参数
        # - UPGD 的 delta 是在训练数据集上生成的，需要知道训练数据集的归一化参数来进行 denorm->加delta->clamp->renorm
        if hasattr(args, 'dataset'):
            # 优先使用训练数据集来定位 delta 文件和归一化参数
            train_dataset = args.dataset  # 训练数据集（如 'mnistm', 'cifar10', 'tiny_imagenet'）
            dir_args.dataset = train_dataset
        else:
            # 兜底：如果没有 args.dataset，使用 dataset_name
            train_dataset = dataset_name
            dir_args.dataset = dataset_name
        dir_args.poison_type = 'upgd'

        # mean/std 必须与本仓库的 transforms 保持一致：
        # - delta 生成在 raw 空间
        # - 模型输入通常是 normalize 后的 tensor
        # - 测试时需要 denorm（使用训练数据集的归一化参数）->加delta->clamp->renorm（使用训练数据集的归一化参数）
        # 
        # 归一化参数选择逻辑：
        # - 必须使用训练数据集（train_dataset）的归一化参数，而不是测试数据集（dataset_name）
        # - 因为模型是在训练数据集上训练的，测试数据会被归一化为训练数据集的归一化参数
        if train_dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.247, 0.243, 0.261)
        elif train_dataset == 'gtsrb':
            mean = (0.3337, 0.3064, 0.3171)
            std = (0.2672, 0.2564, 0.2629)
        elif train_dataset in ['imagenette', 'imagenet']:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif train_dataset == 'tiny_imagenet':
            # Tiny ImageNet 归一化参数
            mean = (0.4802, 0.4481, 0.3975)
            std = (0.2302, 0.2265, 0.2262)
        elif train_dataset == 'mnist':
            # MNIST 归一化参数
            mean = (0.1307, 0.1307, 0.1307)
            std = (0.3081, 0.3081, 0.3081)
        elif train_dataset == 'mnistm':
            # MNIST-M 归一化参数（与 MNIST 不同！）
            mean = (0.46, 0.46, 0.46)
            std = (0.23, 0.23, 0.23)
        elif train_dataset == 'stl10':
            # STL-10 作为训练数据集时，使用 CIFAR-10 的归一化参数（因为 STL-10 被下采样到 32x32）
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.247, 0.243, 0.261)
        else:
            raise NotImplementedError(f"UPGD is not implemented for training dataset={train_dataset}")

        poison_set_dir = get_poison_set_dir(dir_args)
        delta_path = os.path.join(poison_set_dir, f'upgd_{int(target_class)}.pth')

        # 兜底：有些脚本（例如存在 test 强度覆盖的流程）会传 train_poison_dir，
        # 用它可以回到“训练目录”读取 delta，避免测试参数覆盖导致目录不同。
        train_poison_dir = getattr(args, 'train_poison_dir', None)
        if (not os.path.exists(delta_path)) and train_poison_dir is not None:
            alt = os.path.join(train_poison_dir, f'upgd_{int(target_class)}.pth')
            if os.path.exists(alt):
                delta_path = alt

        if not os.path.exists(delta_path):
            raise FileNotFoundError(
                f"[UPGD] Cannot find saved delta at '{delta_path}'. "
                f"Please run create_poisoned_set.py with poison_type=upgd first."
            )

        delta_raw = torch.load(delta_path, map_location='cpu')
        from poison_tool_box import upgd
        # 重要说明：
        # - `delta_raw` 保存的是 raw 像素空间 [0,1]，形状 [3,H,W]
        # - `has_normalized=is_normalized_input` 表示输入是否已经归一化：
        #   - True（常见）：denorm -> 加delta -> clamp -> renorm
        #   - False：直接加 delta
        #
        # 关于 target_class 的跨数据集一致性：
        # - UPGD 的 delta 是针对特定类别（target_class）生成的，目标是让所有输入都被判为目标类
        # - 在跨数据集测试时（如 STL-10），需要确保：
        #   1) 测试数据集的标签已映射到训练数据集的标签空间（如 test_stl10.py 中的 label_mapping）
        #   2) target_class 使用的是训练数据集的 target_class（通过 config.target_class[args.dataset] 获取）
        #   3) 这样 delta 的"目标类别"在映射后的标签空间中是一致的
        # 例如：CIFAR-10 的 target_class=0（airplane），STL-10 的类别 0（airplane）映射到 CIFAR-10 的类别 0，所以是一致的
        poison_transform = upgd.poison_transform(
            delta_raw=delta_raw,
            target_class=target_class,
            mean=mean,
            std=std,
            has_normalized=is_normalized_input,
        )
        return poison_transform

    if poison_type in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                       'adaptive_blend', 'adaptive_patch', 'adaptive_k_way',
                       'SIG', 'TaCT', 'WaNet', 'SleeperAgent', 'none',
                       'badnet_all_to_all', 'trojan', 'SRA', 'bpp', 'belt']:

        if trigger_transform is None:
            trigger_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # trigger mask transform; remove `Normalize` and `Lambda`!
        trigger_mask_transform_list = []
        for t in trigger_transform.transforms:
            if "Normalize" not in t.__class__.__name__ and "Lambda" not in t.__class__.__name__:
                trigger_mask_transform_list.append(t)
        trigger_mask_transform = transforms.Compose(trigger_mask_transform_list)

        if trigger_name != 'none':  # none for SIG
            trigger_path = os.path.join(config.triggers_dir, trigger_name)
            # print('trigger : ', trigger_path)
            # ========== [MNIST 支持] MNIST 现在使用三通道，触发器保持 RGB 格式 ==========
            trigger = Image.open(trigger_path).convert("RGB")  # 所有数据集都使用 RGB
            # ========== [MNIST 支持] 结束 ==========

            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)

            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = trigger_mask_transform(trigger_mask)[0]  # only use 1 channel
            else:  # by default, all black pixels are masked with 0's
                # 保存原始的 PIL Image 用于后续处理
                trigger_pil = trigger
                trigger_map = trigger_mask_transform(trigger_pil)
                trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                                trigger_map[2] > 0).float()

            # 确保触发器是 PIL Image 格式（RGB），而不是已经被转换过的张量
            if not isinstance(trigger, Image.Image):
                # 如果 trigger 已经被转换，需要重新加载
                trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)
            # 调试：检查触发器形状
            if trigger.shape[0] != 3:
                raise ValueError(f"触发器通道数错误：期望 3 通道，实际得到 {trigger.shape[0]} 通道。触发器路径: {trigger_path}")
            trigger_mask = trigger_mask

        if poison_type == 'basic':
            from poison_tool_box import basic
            poison_transform = basic.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                      trigger_mask=trigger_mask,
                                                      target_class=target_class, alpha=alpha)

        elif poison_type == 'badnet':
            from poison_tool_box import badnet
            poison_transform = badnet.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask, target_class=target_class, alpha=alpha)

        elif poison_type == 'belt':
            from poison_tool_box import belt
            import numpy as np
            
            belt_alpha = alpha  # 使用用户指定的 alpha
            
            poison_set_dir = get_poison_set_dir(args) if args is not None else None
            belt_trigger_path = os.path.join(poison_set_dir, 'belt_trigger.pt') if poison_set_dir else None
            
            if belt_trigger_path and os.path.exists(belt_trigger_path):
                belt_trigger_data = torch.load(belt_trigger_path, map_location='cpu')
                belt_pattern_torch = belt_trigger_data['pattern']
                belt_mask_torch = belt_trigger_data['mask']
                saved_alpha = belt_trigger_data.get('alpha', 1.0)
                if abs(belt_alpha - saved_alpha) > 1e-6:
                    print(f'[BELT] 注意: 命令行 alpha={belt_alpha} 与保存的 alpha={saved_alpha} 不一致，使用命令行值')
                print(f'[BELT] 从文件加载触发器: {belt_trigger_path}, alpha={belt_alpha}')
            else:
                belt_mask_np, belt_pattern_np = belt.generate_belt_trigger(img_size)
                belt_mask_torch = torch.from_numpy(belt_mask_np[:, :, 0]).float()
                belt_pattern_torch = torch.from_numpy(belt_pattern_np).permute(2, 0, 1).float() / 255.0
                print(f'[BELT] 动态生成触发器（未找到保存文件），alpha={belt_alpha}')
            
            belt_mean = (0.0, 0.0, 0.0)
            belt_std = (1.0, 1.0, 1.0)
            
            poison_transform = belt.poison_transform(img_size=img_size, trigger_mark=belt_pattern_torch,
                                                     trigger_mask=belt_mask_torch, target_class=target_class, alpha=belt_alpha,
                                                     mean=belt_mean, std=belt_std)

        elif poison_type == 'badnet_all_to_all':
            from poison_tool_box import badnet_all_to_all
            poison_transform = badnet_all_to_all.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                                  trigger_mask=trigger_mask, num_classes=num_classes)

        elif poison_type == 'trojan':
            from poison_tool_box import trojan
            poison_transform = trojan.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask, target_class=target_class)

        elif poison_type == 'blend':
            from poison_tool_box import blend
            poison_transform = blend.poison_transform(img_size=img_size, trigger=trigger,
                                                      target_class=target_class, alpha=alpha)

        elif poison_type == 'refool':
            from poison_tool_box import refool
            poison_transform = refool.poison_transform(img_size=img_size, target_class=target_class,
                                                       denormalizer=denormalizer, normalizer=normalizer,
                                                       max_image_size=32)

        elif poison_type == 'clean_label':
            from poison_tool_box import clean_label
            poison_transform = clean_label.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                            trigger_mask=trigger_mask,
                                                            target_class=target_class)

        elif poison_type == 'WaNet':
            # ========== [WaNet参数修改] 从args中读取s和k参数，如果没有则使用默认值 ==========
            s = args.s if hasattr(args, 's') and args.s is not None else 0.5
            k = args.k if hasattr(args, 'k') and args.k is not None else 4
            # ========== [WaNet参数修改] 结束 ==========
            grid_rescale = 1
            # ========== [修复] 优先使用 train_poison_dir 读取 grid 文件 ==========
            # identity_grid 和 noise_grid 是在创建数据集时生成的，使用的是训练时的 s 值
            # 如果使用了 test_s，应该从训练目录读取这些文件，而不是从测试目录
            # 因为 test_s 只影响扰动的强度，不影响 grid 文件的来源
            poison_set_dir_for_grid = getattr(args, 'train_poison_dir', None) or get_poison_set_dir(args)
            path = os.path.join(poison_set_dir_for_grid, 'identity_grid')
            identity_grid = torch.load(path)
            path = os.path.join(poison_set_dir_for_grid, 'noise_grid')
            noise_grid = torch.load(path)
            # ========== [修复结束] ==========

            from poison_tool_box import WaNet
            poison_transform = WaNet.poison_transform(img_size=img_size, denormalizer=denormalizer,
                                                      identity_grid=identity_grid, noise_grid=noise_grid, s=s, k=k,
                                                      grid_rescale=grid_rescale, normalizer=normalizer,
                                                      target_class=target_class)

        elif poison_type == 'adaptive_blend':

            from poison_tool_box import adaptive_blend
            poison_transform = adaptive_blend.poison_transform(img_size=img_size, trigger=trigger,
                                                               target_class=target_class, alpha=alpha)

        elif poison_type == 'adaptive_patch':
            from poison_tool_box import adaptive_patch
            poison_transform = adaptive_patch.poison_transform(img_size=img_size, test_trigger_names=
            config.adaptive_patch_test_trigger_names[args.dataset],
                                                               test_alphas=config.adaptive_patch_test_trigger_alphas[
                                                                   args.dataset], target_class=target_class,
                                                               denormalizer=denormalizer, normalizer=normalizer, )

        elif poison_type == 'adaptive_k_way':
            from poison_tool_box import adaptive_k_way
            poison_transform = adaptive_k_way.poison_transform(img_size=img_size, target_class=target_class,
                                                               denormalizer=denormalizer, normalizer=normalizer, )

        elif poison_type == 'SIG':
            # ========== [SIG参数修改] 从args中读取delta和f参数，delta自动除以255，如果没有则使用默认值 ==========
            delta = (args.delta / 255) if hasattr(args, 'delta') and args.delta is not None else 30 / 255
            f = args.f if hasattr(args, 'f') and args.f is not None else 6
            # ========== [SIG参数修改] 结束 ==========
            from poison_tool_box import SIG
            poison_transform = SIG.poison_transform(img_size=img_size, denormalizer=denormalizer, normalizer=normalizer,
                                                    target_class=target_class, delta=delta, f=f,
                                                    has_normalized=is_normalized_input)

        elif poison_type == 'TaCT':
            from poison_tool_box import TaCT
            poison_transform = TaCT.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                     target_class=target_class)

        elif poison_type == 'SleeperAgent':
            from poison_tool_box import SleeperAgent
            poison_transform = SleeperAgent.poison_transform(random_patch=False, img_size=img_size,
                                                             target_class=target_class, denormalizer=denormalizer,
                                                             normalizer=normalizer)

        elif poison_type == 'SRA':
            if dataset_name not in ['cifar10', 'imagenet']:
                raise NotImplementedError()

            from other_attacks_tool_box import SRA
            poison_transform = SRA.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                    target_class=target_class)
            return poison_transform

        elif poison_type == 'bpp':
            if dataset_name not in ['cifar10']:
                raise NotImplementedError()

            from other_attacks_tool_box import bpp
            poison_transform = bpp.poison_transform(img_size=img_size, denormalizer=denormalizer, normalizer=normalizer,
                                                    mode="all2one", dithering=True, squeeze_num=8,
                                                    num_classes=num_classes, target_class=target_class)
            return poison_transform

        else:  # 'none'
            from poison_tool_box import none
            poison_transform = none.poison_transform()

        return poison_transform


    elif poison_type == 'dynamic':

        if dataset_name == 'cifar10':
            channel_init = 32
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'

            require_normalization = True

        elif dataset_name == 'gtsrb':
            # the situation for gtsrb is inverese
            # the original implementation of generator does not require normalization
            channel_init = 32
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'

            require_normalization = False

        else:
            raise Exception("Invalid Dataset")

        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                '[Dynamic Attack] Download pretrained generator first: https://github.com/VinAIResearch/input-aware-backdoor-attack-release')

        from poison_tool_box import dynamic
        poison_transform = dynamic.poison_transform(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                    input_channel=input_channel, normalizer=normalizer,
                                                    denormalizer=denormalizer, target_class=target_class,
                                                    has_normalized=is_normalized_input,
                                                    require_normalization=require_normalization)
        return poison_transform

    elif poison_type == 'ISSBA':

        if dataset_name == 'cifar10':
            ckpt_path = './models/ISSBA_cifar10.pth'
            input_channel = 3
            img_size = 32

        elif dataset_name == 'gtsrb':
            ckpt_path = './models/ISSBA_gtsrb.pth'
            input_channel = 3
            img_size = 32
            raise NotImplementedError(
                'ISSBA for GTSRB is not implemented! You may implement it yourself it by training a pair of encoder and decoder using the code: https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/ISSBA.py')

        else:
            raise Exception("Invalid Dataset")

        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                '[ISSBA Attack] Download pretrained encoder and decoder first: https://github.com/')

        secret_path = os.path.join(get_poison_set_dir(args), 'secret')
        secret = torch.load(secret_path)

        from poison_tool_box import ISSBA
        poison_transform = ISSBA.poison_transform(ckpt_path=ckpt_path, secret=secret, normalizer=normalizer,
                                                  denormalizer=denormalizer,
                                                  enc_in_channel=input_channel, enc_height=img_size, enc_width=img_size,
                                                  target_class=target_class)
        return poison_transform

    elif poison_type == 'trojannn':
        if dataset_name not in ['cifar10', 'gtsrb']:
            raise NotImplementedError()

        trigger_path = os.path.join(config.triggers_dir, f'trojannn_{args.dataset}_seed={args.seed}.png')
        # print('trigger : ', trigger_path)
        trigger = Image.open(trigger_path).convert("RGB")

        trigger_mask_path = os.path.join(config.triggers_dir, f'mask_trojan_square_{img_size}.png')

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            temp_trans = transforms.ToTensor()
            trigger_map = temp_trans(trigger)
            trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                            trigger_map[2] > 0).float()

        trigger = trigger_transform(trigger).cuda()
        print('trigger_shape: ', trigger.shape)
        trigger_mask = trigger_mask.cuda()

        from other_attacks_tool_box import trojannn
        poison_transform = trojannn.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                     target_class=target_class)
        return poison_transform

    elif poison_type == 'BadEncoder':
        if dataset_name not in ['gtsrb']:
            raise NotImplementedError()

        if args.dataset == 'gtsrb':
            trigger_name = "BadEncoder_32.png"
        trigger_path = os.path.join(config.triggers_dir, trigger_name)
        # print('trigger : ', trigger_path)
        trigger = Image.open(trigger_path).convert("RGB")

        trigger_mask_path = os.path.join(config.triggers_dir, f'mask_{trigger_name}.png')

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            temp_trans = transforms.ToTensor()
            trigger_map = temp_trans(trigger)
            trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                            trigger_map[2] > 0).float()

        trigger = trigger_transform(trigger).cuda()
        print('trigger_shape: ', trigger.shape)
        trigger_mask = trigger_mask.cuda()

        from other_attacks_tool_box import BadEncoder
        poison_transform = BadEncoder.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                       target_class=target_class)
        return poison_transform

    elif poison_type == "WB":
        from other_attacks_tool_box import WB
        poison_transform = WB.poison_transform()
        return poison_transform

    else:
        raise NotImplementedError('<Undefined> Poison_Type = %s' % poison_type)
