import numpy as np
import torch
import os
from torchvision import transforms,datasets
import argparse
import random
import torch.optim as optim
from torch import nn
from PIL import Image
from utils import supervisor, tools, default_args, imagenet
import config
import json
import datetime


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
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
# ========== [修改] 添加 test_s 和 test_delta 参数，允许测试时使用与训练时不同的参数值 ==========
parser.add_argument('-test_s', type=float, required=False, default=None,
                    help='测试时的WaNet s参数（覆盖训练时的s）')
parser.add_argument('-test_delta', type=float, required=False, default=None,
                    help='测试时的SIG delta参数（覆盖训练时的delta）')
# ========== [修改结束] ==========
parser.add_argument('-trigger', type=str, required=False, default=None)
parser.add_argument('-model', type=str, required=False, default=None,
                    choices=['resnet18', 'vgg19_bn', 'mobilenetv2'],
                    help='模型架构选择（覆盖config.py中的默认设置）')
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-cleanser', type=str, required=False, default=None,
                    choices=default_args.parser_choices['cleanser'])
parser.add_argument('-defense', type=str, required=False, default=None,
                    choices=default_args.parser_choices['defense'])
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
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

# ========== [UPGD 参数] 开始 ==========
# 用于 poison_set_dir / model_path 解析（目录名包含 eps/constraint/steps）
# 说明：这些参数不会改变其它攻击类型行为，仅用于 upgd 的目录定位
parser.add_argument('-eps', type=float, required=False, default=8.0,
                    help='UPGD eps（与 create_poisoned_set.py 保持一致）')
parser.add_argument('-constraint', type=str, required=False, default='Linf',
                    choices=['Linf', 'L2'], help='UPGD 约束类型（与 create_poisoned_set.py 保持一致）')
parser.add_argument('-upgd_steps', type=int, required=False, default=100,
                    help='UPGD steps（与 create_poisoned_set.py 保持一致，用于定位数据/模型目录）')
parser.add_argument('-upgd_steps_multiplier', type=int, required=False, default=5,
                    help='UPGD steps_multiplier（与 create_poisoned_set.py 保持一致，用于定位数据/模型目录）')
# ========== [UPGD 参数] 结束 ==========
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]


if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 4, 'pin_memory': True}

# tools.setup_seed(args.seed)

data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)


if args.dataset == 'cifar10':
    num_classes = 10
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 200
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'cifar100':
    num_classes = 100
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

elif args.dataset == 'gtsrb':
    num_classes = 43
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'imagenette':
    num_classes = 10
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    learning_rate = 0.1
    batch_size = 128

# ========== [Tiny ImageNet 支持] 添加测试配置 ==========
elif args.dataset == 'tiny_imagenet':
    num_classes = 200
    momentum = 0.9
    weight_decay = 5e-4  # 与训练配置一致
    epochs = 200  # 与训练配置一致
    learning_rate = 0.1
    batch_size = 128
# ========== [Tiny ImageNet 支持] 结束 ==========
# ========== [MNIST 支持] 添加测试配置 ==========
elif args.dataset == 'mnist':
    num_classes = 10
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 30  # 与训练配置一致
    learning_rate = 0.1
    batch_size = 128
# ========== [MNIST 支持] 结束 ==========
# ========== [MNIST-M 支持] 添加测试配置 ==========
elif args.dataset == 'mnistm':
    num_classes = 10
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 50  # 与训练配置一致
    learning_rate = 0.01
    batch_size = 128
# ========== [MNIST-M 支持] 结束 ==========
    
elif args.dataset == 'imagenet':
    num_classes = 1000
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 90
    learning_rate = 0.1
    batch_size = 256

else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


poison_set_dir = supervisor.get_poison_set_dir(args)

# BELT特殊处理：使用不同的模型文件名
if args.poison_type == 'belt':
    model_name = f"{supervisor.get_arch(args).__name__}_belt_aug_model_seed={args.seed}.pt"
    model_path = os.path.join(poison_set_dir, model_name)
else:
    model_path = supervisor.get_model_dir(args, cleanse=(args.cleanser is not None), defense=(args.defense is not None))

arch = supervisor.get_arch(args)

import torchvision
# model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model = arch(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model = nn.DataParallel(model)
model = model.cuda()
print("Evaluating model '{}'...".format(model_path))

# Set Up Test Set for Debug & Evaluation
if args.dataset != 'imagenet':
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
    
    # ========== [修改] 在创建 poison_transform 之前，如果指定了 test_s 或 test_delta，临时修改 args 中的值 ==========
    # 保存原始值到 args 对象中（用于后续结果记录）
    if args.poison_type == 'WaNet':
        original_s = getattr(args, 's', None)  # 保存训练时的 s 值
        args.original_s = original_s
        # 如果指定了 test_s，临时修改 args.s（用于测试）
        if args.test_s is not None:
            args.s = args.test_s
    elif args.poison_type == 'SIG':
        original_delta = getattr(args, 'delta', None)  # 保存训练时的 delta 值
        args.original_delta = original_delta
        # 如果指定了 test_delta，临时修改 args.delta（用于测试）
        if args.test_delta is not None:
            args.delta = args.test_delta
    else:
        original_s = None
        original_delta = None
    # ========== [修改结束] ==========
    
    # Poison Transform for Testing
    # ========== [UPGD/BELT 特殊处理] 强制不使用归一化（与原始代码一致）==========
    # UPGD: 原始代码（parameter_backdoor）全程不使用 Normalize
    # BELT: 原始代码（BadNet_BELT.py）全程不使用 Normalize
    # 这两种攻击都在 raw [0,1] 空间训练和测试
    if args.poison_type == 'upgd':
        is_normalized = False  # UPGD 强制不归一化
    elif args.poison_type == 'belt':
        is_normalized = False  # BELT 强制不归一化
    else:
        is_normalized = not args.no_normalize  # 其他攻击根据 -no_normalize 参数决定
    
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                    target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                    is_normalized_input=is_normalized,
                                                    alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                    trigger_name=args.trigger, args=args)

elif args.dataset == 'imagenet':
    test_set_dir = os.path.join(config.imagenet_dir, 'val')
    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, data_transform=data_transform,
                 label_file=imagenet.test_set_labels, num_classes=1000)
    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # ========== [修改] 在创建 poison_transform 之前，如果指定了 test_s 或 test_delta，临时修改 args 中的值 ==========
    # 保存原始值到 args 对象中（用于后续结果记录）
    if args.poison_type == 'WaNet':
        original_s = getattr(args, 's', None)  # 保存训练时的 s 值
        args.original_s = original_s
        # 如果指定了 test_s，临时修改 args.s（用于测试）
        if args.test_s is not None:
            args.s = args.test_s
    elif args.poison_type == 'SIG':
        original_delta = getattr(args, 'delta', None)  # 保存训练时的 delta 值
        args.original_delta = original_delta
        # 如果指定了 test_delta，临时修改 args.delta（用于测试）
        if args.test_delta is not None:
            args.delta = args.test_delta
    else:
        original_s = None
        original_delta = None
    # ========== [修改结束] ==========

    # Poison Transform for Testing
    # ========== [UPGD/BELT 特殊处理] 强制不使用归一化（与原始代码一致）==========
    # UPGD: 原始代码（parameter_backdoor）全程不使用 Normalize
    # BELT: 原始代码（BadNet_BELT.py）全程不使用 Normalize
    # 这两种攻击都在 raw [0,1] 空间训练和测试
    if args.poison_type == 'upgd':
        is_normalized = False  # UPGD 强制不归一化
    elif args.poison_type == 'belt':
        is_normalized = False  # BELT 强制不归一化
    else:
        is_normalized = not args.no_normalize  # 其他攻击根据 -no_normalize 参数决定
    
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                    target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                    is_normalized_input=is_normalized,
                                                    alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                    trigger_name=args.trigger, args=args)


if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = [config.source_class]
else:
    source_classes = None

clean_acc, asr = tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes, all_to_all=('all_to_all' in args.poison_type))

# ========== [结果保存] 保存测试结果到JSON文件 ==========
results = {
    'dataset': args.dataset,
    'poison_type': args.poison_type,
    'poison_rate': args.poison_rate,
    'model_arch': arch.__name__ if hasattr(arch, '__name__') else str(arch),
    'seed': args.seed,
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'clean_acc': float(clean_acc),
    'asr': float(asr) if asr is not None else None,
    'model_path': model_path,
}

# ========== [修改] 添加攻击特定参数，并记录实际测试时使用的参数值 ==========
if args.poison_type == 'SIG':
    # SIG 攻击：记录训练时的 delta 和 f 参数（使用保存的原始值）
    train_delta = getattr(args, 'original_delta', None)
    # 如果 original_delta 存在，使用它；否则使用默认值
    # 注意：如果使用了 test_delta，args.delta 已经被修改，所以不能直接使用 args.delta
    results['delta'] = train_delta if train_delta is not None else 30
    results['f'] = args.f if hasattr(args, 'f') else 6
    # 记录测试时实际使用的 delta（如果使用了 test_delta，则使用 test_delta，否则使用训练时的 delta）
    if args.test_delta is not None:
        results['test_delta'] = args.test_delta
        results['test_used_delta'] = args.test_delta  # 测试时实际使用的 delta
    else:
        results['test_used_delta'] = results['delta']  # 测试时使用训练时的 delta
elif args.poison_type == 'WaNet':
    # WaNet 攻击：记录训练时的 s 和 k 参数（使用保存的原始值）
    train_s = getattr(args, 'original_s', None)
    # 如果 original_s 存在，使用它；否则使用默认值
    # 注意：如果使用了 test_s，args.s 已经被修改，所以不能直接使用 args.s
    results['s'] = train_s if train_s is not None else 0.5
    results['k'] = args.k if hasattr(args, 'k') else 4
    # 记录测试时实际使用的 s（如果使用了 test_s，则使用 test_s，否则使用训练时的 s）
    if args.test_s is not None:
        results['test_s'] = args.test_s
        results['test_used_s'] = args.test_s  # 测试时实际使用的 s
    else:
        results['test_used_s'] = results['s']  # 测试时使用训练时的 s
elif args.poison_type in ['blend', 'adaptive_blend', 'basic', 'clean_label']:
    # Blend/BadNet 攻击：记录训练时的 alpha
    results['alpha'] = args.alpha  # 训练时的 alpha
    # 记录测试时实际使用的 alpha（如果使用了 test_alpha，则使用 test_alpha，否则使用训练时的 alpha）
    if args.test_alpha is not None:
        results['test_alpha'] = args.test_alpha
        results['test_used_alpha'] = args.test_alpha  # 测试时实际使用的 alpha
    else:
        results['test_used_alpha'] = args.alpha  # 测试时使用训练时的 alpha
# ========== [修改结束] ==========

# ========== [修改] 根据不同的测试参数，保存到不同的结果文件，避免覆盖 ==========
# 保存结果文件到模型所在目录
model_dir = os.path.dirname(model_path)

# 根据攻击类型和测试参数，生成不同的文件名
if args.poison_type == 'SIG':
    # SIG 攻击：根据实际测试时使用的 delta 值区分文件名（如果使用了 test_delta，则使用 test_delta）
    if args.test_delta is not None:
        delta_value = args.test_delta
        results_file = os.path.join(model_dir, f'test_results_seed={args.seed}_test_delta={delta_value}.json')
    else:
        delta_value = original_delta if original_delta is not None else (args.delta if hasattr(args, 'delta') else 30)
        results_file = os.path.join(model_dir, f'test_results_seed={args.seed}_delta={delta_value}.json')
elif args.poison_type == 'WaNet':
    # WaNet 攻击：根据实际测试时使用的 s 值区分文件名（如果使用了 test_s，则使用 test_s）
    if args.test_s is not None:
        s_value = args.test_s
        results_file = os.path.join(model_dir, f'test_results_seed={args.seed}_test_s={s_value}.json')
    else:
        s_value = original_s if original_s is not None else (args.s if hasattr(args, 's') else 0.5)
        results_file = os.path.join(model_dir, f'test_results_seed={args.seed}_s={s_value}.json')
elif args.poison_type in ['blend', 'adaptive_blend', 'adaptive_patch', 'basic', 'clean_label']:
    # Blend/BadNet/Patch 攻击：如果使用了 test_alpha，在文件名中包含 test_alpha 值
    if args.test_alpha is not None:
        results_file = os.path.join(model_dir, f'test_results_seed={args.seed}_test_alpha={args.test_alpha}.json')
    else:
        results_file = os.path.join(model_dir, f'test_results_seed={args.seed}.json')
else:
    # 其他攻击类型：使用默认文件名
    results_file = os.path.join(model_dir, f'test_results_seed={args.seed}.json')
# ========== [修改结束] ==========
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n{'='*60}")
print(f"[Results Saved] {results_file}")
print(f"  Clean ACC: {results['clean_acc']:.6f}")
print(f"  ASR:       {results['asr']:.6f}" if results['asr'] is not None else "  ASR: N/A")
print(f"{'='*60}\n")
# ========== [结果保存] 结束 ==========