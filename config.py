from utils import resnet, vgg, mobilenetv2, ember_nn, gtsrb_cnn, wresnet, densenet, preact_resnet, vit
from utils import supervisor
from utils import tools
import torch, torchvision
from torchvision import transforms
import os


data_dir = './data' # defaul clean dataset directory
triggers_dir = './triggers' # default triggers directory
imagenet_dir = '/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization' # ImageNet dataset directory (USE YOUR OWN!)

# ========== [Tiny ImageNet 支持] 添加 Tiny ImageNet 数据集路径 ==========
# Tiny ImageNet: 200 classes, 64x64 images
# 用途: 轻量级 ImageNet 风格数据集，适合跨数据集后门迁移实验
tiny_imagenet_dir = './data/Tiny-imagenet/tiny-imagenet-200' # Tiny ImageNet dataset directory
tiny_imagenet_c_dir = './data/Tiny-imagenet-c/Tiny-ImageNet-C' # Tiny ImageNet-C dataset directory (corrupted version)
# ========== [Tiny ImageNet 支持] 结束 ==========
# ========== [MNIST 支持] 添加 MNIST 数据集路径 ==========
# MNIST: 28×28 灰度图，10 个类别（0-9）
mnist_dir = './data/mnist' # MNIST dataset directory
# MNIST-M: MNIST 的彩色版本，用于跨域测试
mnistm_dir = './data/MNIST-M' # MNIST-M dataset directory (contains train.npy and test.npy)
# ========== [MNIST 支持] 结束 ==========

target_class = {
    'cifar10' : 0,
    'gtsrb' : 2,
    # 'gtsrb' : 12, # BadEncoder
    'imagenette': 0,
    'imagenet' : 0,
    'tiny_imagenet': 2,  # [Tiny ImageNet 支持] 默认目标类为 0
    'mnist': 0,  # [MNIST 支持] 默认目标类为 0
    'mnistm': 2,  # [MNIST-M 支持] 默认目标类为 0
}

# default target class (without loss of generality)
source_class = 1           #||| default source class for TaCT
cover_classes = [5,7]      #||| default cover classes for TaCT
poison_seed = 2333
record_poison_seed = True
record_model_arch = True  # 启用模型架构记录，输出目录将包含模型架构名称（如 DenseNet121）

trigger_default = {
    'cifar10': {
        'none' : 'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'bpp': 'none',
        'WB': 'none',
        'upgd': 'none',
        'belt': 'badnet_patch_32.png',  # BELT 使用默认 BadNet 触发器
    },
    'gtsrb': {
        'none' : 'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'upgd': 'none',
        'belt': 'badnet_patch_32.png',  # BELT 使用默认 BadNet 触发器
    },
    'imagenet': {
        'none': 'none',
        'badnet': 'badnet_patch_256.png',
        'blend' : 'hellokitty_224.png',
        'trojan' : 'trojan_watermark_224.png',
        'SRA': 'phoenix_corner_256.png',
        'upgd': 'none',
    },
    # ========== [Tiny ImageNet 支持] 添加触发器配置 ==========
    # Tiny ImageNet: 使用原始 64×64 尺寸，默认使用 64×64 触发器
    # 说明: 所有触发器文件都存在，直接使用 64×64 版本
    'tiny_imagenet': {
        'none' : 'none',
        'adaptive_blend': 'hellokitty_64.png',  # 64×64 触发器
        'adaptive_patch': 'none',
        'basic' : 'badnet_patch_64.png',        # 64×64 触发器
        'badnet' : 'badnet_patch_64.png',       # 64×64 触发器
        'blend' : 'hellokitty_64.png',          # 64×64 触发器
        'trojan' : 'trojan_square_64.png',      # 64×64 触发器
        'SIG' : 'none',
        'WaNet': 'none',
        'upgd': 'none',
        'belt': 'badnet_patch_64.png',  # BELT 使用默认 BadNet 触发器（64×64）
    },
    # ========== [Tiny ImageNet 支持] 结束 ==========
    # ========== [MNIST 支持] 添加触发器配置 ==========
    # MNIST: 使用 28×28 尺寸，默认使用 28×28 触发器
    # 说明: 触发器文件需要是 28×28 尺寸（假设已存在）
    'mnist': {
        'none' : 'none',
        'adaptive_blend': 'hellokitty_28.png',  # 28×28 触发器
        'adaptive_patch': 'none',
        'basic' : 'badnet_patch_28.png',        # 28×28 触发器
        'badnet' : 'badnet_patch_28.png',       # 28×28 触发器
        'blend' : 'hellokitty_28.png',          # 28×28 触发器
        'trojan' : 'trojan_square_28.png',      # 28×28 触发器
        'SIG' : 'none',
        'WaNet': 'none',
        'upgd': 'none',
        'belt': 'badnet_patch_28.png',  # BELT 使用默认 BadNet 触发器（28×28）
    },
    # ========== [MNIST 支持] 结束 ==========
    # ========== [MNIST-M 支持] 添加触发器配置 ==========
    # MNIST-M: 使用 28×28 尺寸，默认使用 28×28 触发器（与 MNIST 相同）
    'mnistm': {
        'none' : 'none',
        'adaptive_blend': 'hellokitty_28.png',  # 28×28 触发器
        'adaptive_patch': 'none',
        'basic' : 'badnet_patch_28.png',        # 28×28 触发器
        'badnet' : 'badnet_patch_28.png',       # 28×28 触发器
        'blend' : 'hellokitty_28.png',          # 28×28 触发器
        'trojan' : 'trojan_square_28.png',      # 28×28 触发器
        'SIG' : 'none',
        'WaNet': 'none',
        'upgd': 'none',
        'belt': 'badnet_patch_28.png',  # BELT 使用默认 BadNet 触发器（28×28）
    },
    # ========== [MNIST-M 支持] 结束 ==========
}

arch = {
    ### for base model & poison distillation
    # ========== 模型架构选择说明 ==========
    # 默认使用 ResNet18（Conv->BN 结构，与 CLP/ABI 假设完全对齐）
    # 可选架构:
    #   - resnet.ResNet18: Conv->BN 结构，CLP 完全兼容
    #   - densenet.densenet121_*: BN->Conv 结构，CLP 不兼容（BN-Conv 配对错误）
    #   - vit.ViT_*: LayerNorm+Linear 结构，CLP 已适配（使用 Attention Head 裁剪）
    #   - preact_resnet.PreActResNet18: BN->Conv 结构，CLP 不兼容（BN-Conv 配对错误）
    # 注意: CLP/ABI 假设 Conv->BN 顺序，BN->Conv 架构会导致错误配对
    # ========================================
    
    # CIFAR-10 模型（32x32 输入，10 类）
    # 'cifar10': preact_resnet.PreActResNet18,  # PreActResNet-18 (BN->Conv)
    # 'cifar10': densenet.densenet121_cifar10,  # DenseNet-121 (BN->Conv)
    # 'cifar10': vit.ViT_cifar10,               # ViT-Small (LayerNorm+Linear)
    # 'cifar10': resnet.ResNet18,               # ResNet-18 (Conv->BN)
    # 'cifar10': vgg.vgg19_bn,                  # VGG-19-BN
    'cifar10': mobilenetv2.mobilenetv2,         # MobileNetV2 [默认]
    
    # GTSRB 模型（32x32 输入，43 类）
    # 'gtsrb': densenet.densenet121_gtsrb,      # DenseNet-121
    # 'gtsrb': vit.ViT_gtsrb,                   # ViT-Small
    # 'gtsrb': resnet.ResNet18,                 # ResNet-18
    # 'gtsrb': vgg.vgg19_bn,                    # VGG-19-BN
    'gtsrb': mobilenetv2.mobilenetv2,           # MobileNetV2 [默认]
    
    # ImageNette 模型（224x224 输入，10 类）
    # 'imagenette': densenet.densenet121_imagenette,  # DenseNet-121
    # 'imagenette': vit.ViT_imagenette,               # ViT-Small
    # 'imagenette': resnet.ResNet18,                  # ResNet-18
    # 'imagenette': vgg.vgg19_bn,                     # VGG-19-BN
    'imagenette': mobilenetv2.mobilenetv2,            # MobileNetV2 [默认]
    
    # ========== [Tiny ImageNet 支持] 模型架构配置 ==========
    # Tiny ImageNet: 64×64 输入，200 类
    # 'tiny_imagenet': densenet.densenet121_tiny_imagenet_64x64,  # DenseNet-121 (64x64)
    # 'tiny_imagenet': vit.ViT_tiny_imagenet,                     # ViT-Small (64x64)
    # 'tiny_imagenet': resnet.ResNet18_tiny_imagenet,             # ResNet-18
    # 'tiny_imagenet': vgg.vgg19_bn,                              # VGG-19-BN
    'tiny_imagenet': mobilenetv2.mobilenetv2,                     # MobileNetV2 [默认]
    # ========== [Tiny ImageNet 支持] 结束 ==========
    
    # ========== [MNIST 支持] 模型架构配置 ==========
    # MNIST: 28×28 输入（复制为 3 通道），10 类
    # 'mnist': vit.ViT_mnist,                   # ViT-Tiny
    # 'mnist': resnet.ResNet18,                 # ResNet-18
    # 'mnist': vgg.vgg19_bn,                    # VGG-19-BN
    'mnist': mobilenetv2.mobilenetv2,           # MobileNetV2 [默认]
    # ========== [MNIST 支持] 结束 ==========

    # ========== [MNIST-M 支持] 模型架构配置 ==========
    # MNIST-M: 28×28 输入（RGB 3 通道），10 类
    # 'mnistm': resnet.ResNet18,                # ResNet-18
    # 'mnistm': vgg.vgg19_bn,                   # VGG-19-BN
    'mnistm': mobilenetv2.mobilenetv2,          # MobileNetV2 [默认]
    # ========== [MNIST-M 支持] 结束 ==========
    
    'ember': ember_nn.EmberNN,
    
    # ImageNet 模型（224x224 输入，1000 类）
    # 'imagenet': torchvision.models.densenet121,  # DenseNet-121
    # 'imagenet': vit.ViT_imagenet,                # ViT-Base
    'imagenet': torchvision.models.resnet18,      # ResNet-18 [默认]
    
    'abl': wresnet.WideResNet,
}


# adapitve-patch triggers for different datasets
adaptive_patch_train_trigger_names = {
    'cifar10': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'gtsrb': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'tiny_imagenet': [
        # Tiny ImageNet: 训练图像是 64×64，这里必须用 64×64 的 trigger/mask
        # 否则在 create_poisoned_set 阶段会出现 32 vs 64 的张量维度不匹配。
        'phoenix_corner_64.png',
        'firefox_corner_64.png',
        'badnet_patch4_64.png',
        'trojan_square_64.png',
    ],
    'mnist': [
        'phoenix_corner_28.png',
        'firefox_corner_28.png',
        'badnet_patch4_28.png',
        'trojan_square_28.png',
    ],
    'mnistm': [
        'phoenix_corner_28.png',
        'firefox_corner_28.png',
        'badnet_patch4_28.png',
        'trojan_square_28.png',
    ],
}

adaptive_patch_train_trigger_alphas = {
    'cifar10': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'gtsrb': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'tiny_imagenet': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'mnist': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'mnistm': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
}

adaptive_patch_test_trigger_names = {
    'cifar10': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
    'gtsrb': [
        'firefox_corner_32.png',
        'trojan_square_32.png',
    ],
    'tiny_imagenet': [
        # Tiny ImageNet-C 测试同样是 64×64
        'phoenix_corner2_64.png',
        'badnet_patch4_64.png',
    ],
    'mnist': [
        'phoenix_corner2_28.png',
        'badnet_patch4_28.png',
    ],
    'mnistm': [
        'phoenix_corner2_28.png',
        'badnet_patch4_28.png',
    ],
}

adaptive_patch_test_trigger_alphas = {
    'cifar10': [
        1,
        1,
    ],
    'gtsrb': [
        1,
        1,
    ],
    'tiny_imagenet': [
        1,
        1,
    ],
    'mnist': [
        1,
        1,
    ],
    'mnistm': [
        1,
        1,
    ],
}


def get_params(args):

    if args.dataset == 'cifar10':

        num_classes = 10

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        distillation_ratio = [1/2, 1/5, 1/25, 1/50, 1/100]
        momentums = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        lambs = [20, 20, 20, 30, 30, 15]
        lrs = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01]
        batch_factors = [2, 2, 2, 2, 2, 2]

    elif args.dataset == 'gtsrb':

        num_classes = 43

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        distillation_ratio = [1/2, 1/5, 1/25, 1/50, 1/100]
        momentums = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        lambs = [20, 20, 20, 20, 20, 20]
        lrs = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
        batch_factors = [2, 2, 4, 8, 8, 2] # 2,2,4,8,8,8

    elif args.dataset == 'imagenette':

        num_classes = 10

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        distillation_ratio = [1/2, 1/5, 1/25, 1/50, 1/100]
        momentums = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        lambs = [20, 20, 20, 40, 30, 5]
        lrs = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01]
        batch_factors = [2, 2, 2, 2, 2, 2]

    else:
        raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)


    params = {
        'data_transform' : data_transform_normalize,
        'data_transform_aug' : data_transform_aug,

        'distillation_ratio': distillation_ratio,
        'momentums': momentums,
        'lambs': lambs,
        'lrs': lrs,
        'batch_factors': batch_factors,
        'weight_decay' : 1e-4,
        'num_classes' : num_classes,
        'batch_size' : 32,
        'pretrain_epochs' : 100,
        'median_sample_rate': 0.1,
        'base_arch' :  arch[args.dataset],
        'arch' :  arch[args.dataset],
        'kwargs' : {'num_workers': 2, 'pin_memory': True},
        'inspection_set_dir': supervisor.get_poison_set_dir(args)
    }


    return params


def get_dataset(inspection_set_dir, data_transform, args, num_classes = 10):

    print('|num_classes = %d|' % num_classes)

    # Set Up Inspection Set (dataset that is to be inspected
    inspection_set_img_dir = os.path.join(inspection_set_dir, 'data')
    inspection_set_label_path = os.path.join(inspection_set_dir, 'labels')
    inspection_set = tools.IMG_Dataset(data_dir=inspection_set_img_dir,
                                     label_path=inspection_set_label_path, transforms=data_transform)

    # Set Up Clean Set (the small clean split at hand for defense
    clean_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
    clean_set_img_dir = os.path.join(clean_set_dir, 'data')
    clean_label_path = os.path.join(clean_set_dir, 'clean_labels')
    clean_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                  label_path=clean_label_path, transforms=data_transform,
                                  num_classes=num_classes, shift=True)



    return inspection_set, clean_set


def get_packet_for_debug(poison_set_dir, data_transform, batch_size, args):

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)


    kwargs = {'num_workers': 2, 'pin_memory': True}
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    trigger_transform = data_transform
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=target_class[args.dataset],
                                                       trigger_transform=trigger_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)

    poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))

    if args.poison_type == 'TaCT':
        source_classes = [source_class]
    else:
        source_classes = None

    debug_packet = {
        'test_set_loader' : test_set_loader,
        'poison_transform' : poison_transform,
        'poison_indices' : poison_indices,
        'source_classes' : source_classes
    }

    return debug_packet