import argparse
import os, sys
import time
from tqdm import tqdm
from utils import default_args, imagenet
from torch.cuda.amp import autocast
from torch.amp import GradScaler
import json
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='none',
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-resume', type=int, required=False, default=0)
parser.add_argument('-resume_from_meta_info', default=False, action='store_true')
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)
# ========== [ABI/Grond 权重抑制] 开始 ==========
# 工程说明：
# - 这是你提到的 Grond 核心训练流程 Step B/C：epoch 末按权重计算 UCLC，筛出“显眼包通道”，再把该通道权重重置为层均值。
# - 按你的要求：只要 poison_type=upgd，就必须执行该流程（不做开关）。
parser.add_argument('-abi_u', type=float, default=3.0,
                    help='ABI 阈值系数 u：threshold = mean + u*std（默认 3.0）')
# ========== [ABI/Grond 权重抑制] 结束 ==========
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
parser.add_argument('-belt_model', type=str, required=False, default='aug',
                    choices=['aug', 'do', 'all'],
                    help='BELT 训练模式：aug=BELT增强模型（默认）, do=对比模型, all=全部')
parser.add_argument('-model', type=str, required=False, default=None,
                    choices=['resnet18', 'vgg19_bn', 'mobilenetv2'],
                    help='模型架构选择（覆盖config.py中的默认设置）')
# ========== [BELT 参数] 结束 ==========

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools
from lipschitzness_pruning import CLP


# NOTE: apply_abi_weight_suppression 已停用，保留以备参考；当前训练流程仅在 UPGD 且未 --no_clp 时调用 lipschitzness_pruning.CLP
# def apply_abi_weight_suppression(model: torch.nn.Module, u: float = 3.0):
#     ...



if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]


all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True


if args.dataset != 'ember':
    model_path = supervisor.get_model_dir(args)
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'backdoored_model.pt')


# tools.setup_seed(args.seed)

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'base')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    if args.resume > 0 or args.resume_from_meta_info:
        fout = open(out_path, 'a')
    else:
        fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)


if args.dataset == 'cifar10':

    num_classes = 10
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 5e-4  # 与 BackdoorBench 一致
    epochs = 200
    learning_rate = 0.1  # 初始学习率为 0.1
    batch_size = 128  # 与 BackdoorBench 一致（从 128 改为 256）

elif args.dataset == 'gtsrb':

    num_classes = 43
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    learning_rate = 0.01
    batch_size = 128

elif args.dataset == 'imagenette':

    num_classes = 10
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    learning_rate = 0.1
    batch_size = 128

# ========== [Tiny ImageNet 支持] 添加训练配置 ==========
elif args.dataset == 'tiny_imagenet':

    num_classes = 200
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 5e-4  # 与 BackdoorBench 一致
    epochs = 200
    learning_rate = 0.1  # 初始学习率为 0.1
    batch_size = 256  # 与 BackdoorBench 一致（从 128 改为 256）
# ========== [Tiny ImageNet 支持] 结束 ==========# ========== [MNIST 支持] 添加训练配置 ==========
# 超参数：使用 CosineAnnealingLR（余弦退火），与 CIFAR-10 类似
elif args.dataset == 'mnist':

    num_classes = 10
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 30  # 训练 30 个 epoch
    learning_rate = 0.01  # 初始学习率降低到 0.01（避免梯度爆炸）
    batch_size = 128
# ========== [MNIST 支持] 结束 ==========
# ========== [MNIST-M 支持] 添加训练配置 ==========
# 超参数：使用 CosineAnnealingLR（余弦退火），与 MNIST 类似
elif args.dataset == 'mnistm':

    num_classes = 10
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 50  # 训练 30 个 epoch
    learning_rate = 0.01  # 初始学习率降低到 0.01（避免梯度爆炸）
    batch_size = 128
# ========== [MNIST-M 支持] 结束 ==========


elif args.dataset == 'imagenet':

    num_classes = 1000
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 90
    learning_rate = 0.1
    batch_size = 256

elif args.dataset == 'ember':

    num_classes = 2
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-6
    epochs = 10
    learning_rate = 0.1
    batch_size = 512

else:

    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
elif args.dataset == 'tiny_imagenet':
    # Tiny ImageNet 使用 64x64 图片，数据较大，减少 workers 以避免共享内存不足
    kwargs = {'num_workers': 2, 'pin_memory': True}
elif args.dataset == 'mnist' or args.dataset == 'mnistm':
    # MNIST/MNIST-M 数据较小，可以使用更多 workers
    kwargs = {'num_workers': 4, 'pin_memory': True}

else:
    kwargs = {'num_workers': 4, 'pin_memory': True}

# Set Up Poisoned Set

# ========== [数据集数据加载] ==========
if args.dataset != 'ember' and args.dataset != 'imagenet':
    poison_set_dir = supervisor.get_poison_set_dir(args)
    # 只有 Tiny-ImageNet 使用 'imgs' 目录，其他数据集使用 'data' 目录
    if args.dataset == 'tiny_imagenet':
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'imgs')
    else:
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    print('dataset : %s' % poisoned_set_img_dir)

    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path, transforms=data_transform if args.no_aug else data_transform_aug)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)
# ========== [数据集数据加载] 结束 ==========

elif args.dataset == 'imagenet':

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    print('dataset : %s' % poison_set_dir)

    poison_indices = torch.load(poison_indices_path)

    train_set_dir = os.path.join(config.imagenet_dir, 'train')
    test_set_dir = os.path.join(config.imagenet_dir, 'val')

    from utils import imagenet
    poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, data_transform=data_transform_aug, poison_directory=poisoned_set_img_dir,
                                             poison_indices = poison_indices, target_class=config.target_class['imagenet'],
                                             num_classes=1000)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    """
    (self, directory, shift=False, aug=True,
                 poison_directory=None, poison_indices=None,
                 label_file=None, target_class = None, num_classes=1000, scale_for_ct=False)
    """


    """
    poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, shift=False,
                 poison_directory=poisoned_set_img_dir, poison_indices=poison_indices, target_class=imagenet.target_class,
                 label_file=None, num_classes=1000)

    poisoned_set_loader = imagenet_ffcv.get_ffcv_loader(dataset=poisoned_set, nick_name='poison_%s' % args.poison_type,
                                                        batch_size=batch_size, aug=True)"""

else:
    poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    #stats_path = os.path.join('data', 'ember', 'stats')
    poisoned_set = tools.EMBER_Dataset( x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
                                        y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    print('dataset : %s' % poison_set_dir)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)


if args.dataset != 'ember' and args.dataset != 'imagenet':
    # ========== [MNIST-M 支持] 测试时使用 MNIST 数据集 ==========
    # 如果训练数据集是 MNIST-M，测试时使用 MNIST 测试集
    if args.dataset == 'mnistm':
        # 使用 MNIST 的测试集（跨域测试）
        test_set_dir = os.path.join('clean_set', 'mnist', 'test_split')
        test_set_img_dir = os.path.join(test_set_dir, 'data')
        test_set_label_path = os.path.join(test_set_dir, 'labels')
        # 使用 MNIST 的数据变换（MNIST 归一化参数）
        from utils.supervisor import get_transforms
        import argparse
        mnist_args = argparse.Namespace()
        mnist_args.dataset = 'mnist'
        mnist_args.no_normalize = args.no_normalize
        mnist_args.no_aug = args.no_aug
        mnist_args.poison_type = args.poison_type
        _, mnist_data_transform, _, _, _ = get_transforms(mnist_args)
        test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                     label_path=test_set_label_path, transforms=mnist_data_transform)
        # Poison Transform 使用 MNIST 的配置（因为测试在 MNIST 上）
        poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name='mnist',
                                                           target_class=config.target_class['mnist'], trigger_transform=mnist_data_transform,
                                                           is_normalized_input=True,
                                                           alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                           trigger_name=args.trigger, args=args)
    else:
        # Set Up Test Set for Debug & Evaluation
        test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
        test_set_img_dir = os.path.join(test_set_dir, 'data')
        test_set_label_path = os.path.join(test_set_dir, 'labels')
        test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                     label_path=test_set_label_path, transforms=data_transform)
        # Poison Transform for Testing
        poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                           target_class=config.target_class[args.dataset], trigger_transform=trigger_transform,
                                                           is_normalized_input=True,
                                                           alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                           trigger_name=args.trigger, args=args)
    
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


elif args.dataset == 'imagenet':

    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)

    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, data_transform=data_transform,
                 label_file=imagenet.test_set_labels, num_classes=1000)

    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

else:
    normalizer = poisoned_set.normal

    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')

    test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
                                   y_path=os.path.join(test_set_dir, 'Y.npy'),
                                   normalizer = normalizer)

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


    backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
                                       y_path=None, normalizer = normalizer)
    backdoor_test_set_loader = torch.utils.data.DataLoader(
        backdoor_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


"""
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = nn.DataParallel(model).cuda()
model.eval()


with torch.no_grad():

    correct = 0
    tot = 0
    for imgs, labels in tqdm(test_set_loader):
        imgs = imgs.cuda()
        output = model(imgs)
        preds = torch.argmax(output, dim=1).detach().cpu()
        tot += len(labels)
        correct += (preds == labels).sum()
    print('test set accuracy = %d/%d = %f' % (correct, tot, correct / tot))

    correct = 0
    tot = 0
    for imgs, labels in tqdm(poisoned_set_loader):
        imgs = imgs.cuda()
        output = model(imgs)
        preds = torch.argmax(output, dim=1).detach().cpu()
        tot += len(labels)
        correct += (preds == labels).sum()
    print('training set accuracy = %d/%d = %f' % (correct, tot, correct/tot))


exit(0)"""






# ========== [BELT 训练分支] 开始 ==========
# BELT 训练 aug_model（BELT 增强模型）和可选的 do_model（对比模型）
# 注意：此分支必须在数据加载完成后执行（poisoned_set, test_set_loader 等已定义）
if args.poison_type == 'belt':
    from utils import belt_dataset
    from poison_tool_box.belt import CenterLoss
    
    # 检查 pmarks 文件是否存在
    pmark_path = os.path.join(poison_set_dir, 'pmarks')
    if not os.path.exists(pmark_path):
        raise FileNotFoundError(f"BELT 训练需要 pmarks 文件，但未找到: {pmark_path}")
    
    # 创建支持 pmarks 的数据集
    belt_dataset_full = belt_dataset.BELT_Dataset(poisoned_set, pmark_path)
    belt_loader_full = torch.utils.data.DataLoader(
        belt_dataset_full,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)
    
    # 导入 BELT 训练函数
    from train_belt import train_belt_models
    train_belt_models(args, arch, num_classes, epochs, batch_size, learning_rate, 
                     momentum, weight_decay, poison_set_dir, belt_loader_full,
                     test_set_loader, poison_transform, kwargs)
    exit(0)
# ========== [BELT 训练分支] 结束 ==========

# Train Code
print(f"Will save to '{model_path}'.")
if os.path.exists(model_path):
    print(f"Model '{model_path}' already exists!")

if args.dataset != 'ember':
    model = arch(num_classes=num_classes)
else:
    model = arch()


# Check if need to resume from the checkpoint
if os.path.exists(os.path.join(poison_set_dir, "meta_info_{}".format(supervisor.get_model_name(args)))):
    meta_info = torch.load(os.path.join(poison_set_dir, "meta_info_{}".format(supervisor.get_model_name(args))))
else:
    meta_info = dict()
    meta_info['epoch'] = 0

if args.resume > 0:
    meta_info['epoch'] = args.resume
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
elif args.resume_from_meta_info:
    args.resume = meta_info['epoch']
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
else:
    meta_info['epoch'] = 0

model = nn.DataParallel(model)
model = model.cuda()

if args.dataset != 'ember':
    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.BCELoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)

# 使用 MultiStepLR（多步长学习率调度器）
# milestones 设置为 [100, 150]，在 epoch 100 和 150 时学习率会衰减
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = [config.source_class]
else:
    source_classes = None

import time
st = time.time()

"""
if args.dataset == 'imagenet':
    tools.test_imagenet(model=model, test_loader=test_set_loader,
                                     poison_transform=poison_transform)
    print('<time : %f minutes>' % ( (time.time() - st) / 60 ))
"""

scaler = GradScaler('cuda')
for epoch in range(1, epochs+1):  # train backdoored base model
    start_time = time.perf_counter()

    # Skip to the checkpointed epoch
    if epoch <= args.resume:
        scheduler.step()
        continue

    # Train
    model.train()
    preds = []
    labels = []
    epoch_loss_sum = 0.0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(tqdm(poisoned_set_loader)):

        optimizer.zero_grad()
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        
        # ========== [修复] 启用混合精度训练（与 parameter_backdoor 一致）==========
        # 使用 autocast 和 GradScaler 进行混合精度训练，可能影响梯度更新的数值精度
        # 这可能是导致 ASR 异常高（1.0）的主要原因
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # ========== [修复] 结束 ==========

        # 计算总损失用于统计
        batch_size = target.size(0)
        epoch_loss_sum += loss.item() * batch_size
        total_samples += batch_size
        
        # data = data.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        # data, target = data.cuda(), target.cuda()
        # optimizer.zero_grad(set_to_none=True)

        # with autocast():
        #     output = model(data)
        #     loss = criterion(output, target)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # 计算平均损失用于打印（更易读）
    avg_loss = epoch_loss_sum / total_samples if total_samples > 0 else float('inf')
    print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, avg_loss, optimizer.param_groups[0]['lr'], elapsed_time))
    
    # 使用 CosineAnnealingLR，每个 epoch 结束时调用 step()
    scheduler.step()

    # Test
    if args.dataset != 'ember':
        if True:
        # if epoch % 5 == 0:
            test_result = tools.test(model=model, test_loader=test_set_loader, poison_test=True if args.poison_type != 'none' else False,
                        poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes, all_to_all=all_to_all)
            if args.poison_type != 'none':
                clean_acc, asr = test_result
            else:
                clean_acc, _ = test_result
                asr = None
            # ========== [结果保存] 记录最后一个epoch的结果 ==========
            if epoch == epochs:
                final_clean_acc = clean_acc
                final_asr = asr
            # ========== [结果保存] 结束 ==========
            
            torch.save(model.module.state_dict(), model_path)
    else:

        tools.test_ember(model=model, test_loader=test_set_loader,
                             backdoor_test_loader=backdoor_test_set_loader)
        torch.save(model.module.state_dict(), model_path)
    print("")
    
    meta_info['epoch'] = epoch
    torch.save(meta_info, os.path.join(poison_set_dir, "meta_info_{}".format(supervisor.get_model_name(args))))

    # ========== [CLP 权重抑制：仅 UPGD 使用] ==========
    # 与 parameter_backdoor 行为保持一致：在 Test/Save 之后，下一轮训练之前调用
    if args.poison_type == 'upgd':
        print(f'[CLP] Applying weight suppression (epoch {epoch}, u={args.abi_u})...')
        CLP(model, float(args.abi_u))



torch.save(model.module.state_dict(), model_path)

# ========== [结果保存] 保存最终训练结果到JSON文件 ==========
if args.dataset != 'ember':
    results = {
        'dataset': args.dataset,
        'poison_type': args.poison_type,
        'poison_rate': args.poison_rate,
        'model_arch': arch.__name__,
        'seed': args.seed,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'final_epoch': epochs,
        'clean_acc': float(final_clean_acc) if 'final_clean_acc' in locals() else None,
        'asr': float(final_asr) if 'final_asr' in locals() and final_asr is not None else None,
        'model_path': model_path,
    }
    
    # 添加攻击特定参数
    if args.poison_type == 'SIG':
        results['delta'] = args.delta if hasattr(args, 'delta') else 30
        results['f'] = args.f if hasattr(args, 'f') else 6
    elif args.poison_type == 'WaNet':
        results['s'] = args.s if hasattr(args, 's') else 0.5
        results['k'] = args.k if hasattr(args, 'k') else 4
    elif args.poison_type in ['blend', 'adaptive_blend', 'basic', 'clean_label']:
        results['alpha'] = args.alpha
    
    # 保存结果文件到模型所在目录
    model_dir = os.path.dirname(model_path)
    results_file = os.path.join(model_dir, f'train_results_seed={args.seed}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"[Results Saved] {results_file}")
    print(f"  Clean ACC: {results['clean_acc']:.6f}" if results['clean_acc'] is not None else "  Clean ACC: N/A")
    print(f"  ASR:       {results['asr']:.6f}" if results['asr'] is not None else "  ASR: N/A")
    print(f"{'='*60}\n")
# ========== [结果保存] 结束 ==========
# ========== [结果保存] 结束 ==========