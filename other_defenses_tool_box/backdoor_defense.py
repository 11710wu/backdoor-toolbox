import config, os
from utils import supervisor
import torch
from torchvision import datasets, transforms
from PIL import Image

class BackdoorDefense():
    def __init__(self, args):
        self.dataset = args.dataset
        if args.dataset == 'gtsrb':
            self.img_size = 32
            self.num_classes = 43
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar10':      
            self.img_size = 32
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar100':
            print('<To Be Implemented> Dataset = %s' % args.dataset)
            exit(0)
        elif args.dataset == 'imagenette':   
            self.img_size = 224
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'imagenet':
            self.img_size = 224
            self.num_classes = 1000
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        # ========== [Tiny ImageNet 支持] 添加配置 ==========
        elif args.dataset == 'tiny_imagenet':
            self.img_size = 64  # Tiny ImageNet 使用原始 64×64 尺寸
            self.num_classes = 200
            self.input_channel = 3
            self.shape = torch.Size([3, 64, 64])
            self.momentum = 0.9
            self.weight_decay = 5e-4  # 与训练配置一致
            self.learning_rate = 0.1
        # ========== [Tiny ImageNet 支持] 结束 ==========
        # ========== [MNIST 支持] 添加配置 ==========
        elif args.dataset == 'mnist':
            self.img_size = 28  # MNIST 使用 28×28 尺寸
            self.num_classes = 10
            self.input_channel = 3  # MNIST 在训练时已转换为三通道（单通道转三通道）
            self.shape = torch.Size([3, 28, 28])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        # ========== [MNIST 支持] 结束 ==========
        # ========== [MNIST-M 支持] 添加配置 ==========
        elif args.dataset == 'mnistm':
            self.img_size = 28  # MNIST-M 使用 28×28 尺寸
            self.num_classes = 10
            self.input_channel = 3  # MNIST-M 是 RGB 三通道
            self.shape = torch.Size([3, 28, 28])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.01
        # ========== [MNIST-M 支持] 结束 ==========
        else:
            print('<Undefined> Dataset = %s' % args.dataset)
            exit(0)
        
        self.data_transform_aug, self.data_transform, self.trigger_transform, self.normalizer, self.denormalizer = supervisor.get_transforms(args)
        
        self.poison_type = args.poison_type
        self.poison_rate = args.poison_rate
        self.cover_rate = args.cover_rate
        self.alpha = args.alpha
        self.trigger = args.trigger
        self.target_class = config.target_class[args.dataset]
        self.device='cuda'

        # ========== [UPGD/BELT 特殊处理] 强制不使用归一化（与原始代码一致）==========
        # UPGD: 原始代码（parameter_backdoor）全程不使用 Normalize
        # BELT: 原始代码（BadNet_BELT.py）全程不使用 Normalize
        if args.poison_type == 'upgd':
            is_normalized = False  # UPGD 强制不归一化
        elif args.poison_type == 'belt':
            is_normalized = False  # BELT 强制不归一化
        else:
            is_normalized = not args.no_normalize  # 其他攻击根据 -no_normalize 参数决定
        
        self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=config.target_class[args.dataset], trigger_transform=self.trigger_transform,
                                                            is_normalized_input=is_normalized,
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger, args=args)
        
        if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
            self.source_classes = [config.source_class]
        else:
            self.source_classes = None

        
        trigger_path = os.path.join(config.triggers_dir, args.trigger)
        print('trigger_path:', trigger_path)
        self.trigger_mark = Image.open(trigger_path).convert("RGB")
        self.trigger_mark = self.trigger_transform(self.trigger_mark).cuda()
        
        trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % args.trigger)
        if os.path.exists(trigger_mask_path): # if there explicitly exists a trigger mask (with the same name)
            print('trigger_mask_path:', trigger_mask_path)
            self.trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            self.trigger_mask = transforms.ToTensor()(self.trigger_mask)[0].cuda() # only use 1 channel
        else: # by default, all black pixels are masked with 0's (not used)
            print('No trigger mask found! By default masking all black pixels...')
            self.trigger_mask = torch.logical_or(torch.logical_or(self.trigger_mark[0] > 0, self.trigger_mark[1] > 0), self.trigger_mark[2] > 0).cuda()

        self.poison_set_dir = supervisor.get_poison_set_dir(args)
        model_path = supervisor.get_model_dir(args)
        
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
        
        arch = supervisor.get_arch(args)
        self.model = arch(num_classes=self.num_classes)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict, strict=False)
            print("Evaluating model '{}'...".format(model_path))
        else:
            print("Model '{}' not found.".format(model_path))
        
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.eval()
        
