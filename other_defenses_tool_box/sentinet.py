#!/usr/bin/env python3

# from ..backdoor_defense import BackdoorDefense
# from trojanvision.environ import env
# from trojanzoo.utils import to_numpy

from turtle import pos
import torch, torchvision
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, unpack_poisoned_train_set
from . import BackdoorDefense
import config, os
from utils import supervisor
from matplotlib import pyplot as plt
from utils.gradcam import GradCAM, GradCAMpp
from scipy.optimize import minimize
import math
import json


class SentiNet(BackdoorDefense):
    """
    Assuming oracle knowledge of the used trigger.
    """
    
    name: str = 'sentinet'

    def __init__(self, args, defense_fpr: float = 0.05, N: int = 100):
        super().__init__(args)
        self.args = args
        
        # Only support localized attacks
        # support_list = ['adaptive_patch', 'badnet', 'badnet_all_to_all', 'dynamic', 'TaCT']
        # assert args.poison_type in support_list
        # ========== [Tiny ImageNet 支持] 扩展支持的数据集 ==========
        # SentiNet 使用 GradCAM 计算梯度激活，理论上支持任何 CNN 模型
        # 添加 'tiny_imagenet' 支持（64×64, 200 classes，使用原始 64×64 尺寸）
        # 添加 'mnist' 支持（28×28, 10 classes，单通道转三通道）
        assert args.dataset in ['cifar10', 'gtsrb', 'tiny_imagenet', 'mnist', 'mnistm']
        # ========== [Tiny ImageNet 支持] 结束 ==========

        self.defense_fpr = defense_fpr
        self.N = N
        
        self.folder_path = 'other_defenses_tool_box/results/Sentinet'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        
        self.random_img = self.normalizer(torch.rand((3, self.img_size, self.img_size))).cuda()

    def detect(self):
        args = self.args
        loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=1,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)
        loader = tqdm(loader)
        
        clean_loader = generate_dataloader(dataset=self.dataset,
                                            dataset_path=config.data_dir,
                                            batch_size=100,
                                            split='test',
                                            shuffle=True,
                                            drop_last=False)
        clean_subset, val_subset, _ = torch.utils.data.random_split(clean_loader.dataset, [self.N, 400, len(clean_loader.dataset) - self.N - 400])
        clean_loader = torch.utils.data.DataLoader(clean_subset, batch_size=100, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        
        
        est_fooled = []
        est_avgconf = []
        
        # ========== [修改] 模型类型检测和 GradCAM 参数选择 ==========
        # SentiNet 方法使用 GradCAM 计算梯度激活图，理论上支持任何能够计算梯度的模型
        # 这个代码库的 GradCAM 实现（utils/gradcam.py）支持以下模型类型：
        #   - ResNet: 通过 find_resnet_layer 支持
        #   - DenseNet: 通过 find_densenet_layer 支持
        #   - VGG: 通过 find_vgg_layer 支持
        #   - AlexNet: 通过 find_alexnet_layer 支持
        #   - SqueezeNet: 通过 find_squeezenet_layer 支持
        # 
        # 注意：WideResNet 虽然名称包含 "resnet"，但其结构（block1/block2/block3）与标准 ResNet 不同
        # 如果 WideResNet 使用 layer1/layer2/layer3 结构，可以通过 ResNet 方式支持
        # ========== [修改结束] ==========
        model_arch_name = supervisor.get_arch(args).__name__.lower()
        # ========== [Tiny ImageNet 支持] 根据数据集设置输入尺寸 ==========
        # CIFAR-10/GTSRB: 32x32
        # Tiny ImageNet: 64×64（使用原始 64×64 尺寸）
        # ImageNette/ImageNet: 224x224
        if args.dataset == 'tiny_imagenet':
            input_size = (64, 64)
        elif args.dataset == 'mnist' or args.dataset == 'mnistm':
            input_size = (28, 28)
        elif args.dataset in ['cifar10', 'gtsrb']:
            input_size = (32, 32)
        else:
            input_size = (224, 224)
        # ========== [Tiny ImageNet 支持] 结束 ==========
        
        if 'densenet' in model_arch_name:
            # ========== [修改] DenseNet 支持 ==========
            # DenseNet-121 的 denseblock4 有 16 个 dense layers (denselayer0 到 denselayer15)
            # 使用最后一个 dense layer 以获得更精确的梯度
            # ========== [修改结束] ==========
            model_type = 'densenet'
            layer_name = 'features_denseblock4_denselayer15'
        elif 'resnet' in model_arch_name or 'wresnet' in model_arch_name or 'wideresnet' in model_arch_name:
            # ========== [修改] ResNet 和 WideResNet 支持 ==========
            # ResNet: 使用标准的 layer4
            # WideResNet: 如果使用 layer1/layer2/layer3 结构（如 wide_resnet.py），可以通过 ResNet 方式支持
            # 注意：wresnet.py 中的 WideResNet 使用 block1/block2/block3，可能不支持
            # ========== [修改结束] ==========
            model_type = 'resnet'
            layer_name = 'layer4'
        elif 'vgg' in model_arch_name:
            model_type = 'vgg'
            # 使用 'last' 自动找到最后一个卷积层，兼容 VGG19 和 VGG19-BN
            # VGG19: features 有 36 层，最后一个卷积层是索引 34
            # VGG19-BN: features 有 52 层，最后一个卷积层是索引 50
            layer_name = 'last'
        elif 'alexnet' in model_arch_name:
            # ========== [修改] AlexNet 支持 ==========
            # AlexNet 的最后一个卷积层在 features 模块中
            # ========== [修改结束] ==========
            model_type = 'alexnet'
            layer_name = 'features_12'  # AlexNet 的最后一个卷积层
        elif 'squeezenet' in model_arch_name:
            # ========== [修改] SqueezeNet 支持 ==========
            # SqueezeNet 的最后一个卷积层
            # ========== [修改结束] ==========
            model_type = 'squeezenet'
            layer_name = 'features_12'  # SqueezeNet 的最后一个卷积层
        elif 'mobilenet' in model_arch_name:
            # ========== [修改] MobileNet 支持 ==========
            # MobileNetV2 的 features 模块包含多个 InvertedResidual 块
            # 使用 'last' 自动找到最后一个有意义的层，或者指定具体的层索引
            # ========== [修改结束] ==========
            model_type = 'mobilenet'
            # 使用 'last' 让 find_mobilenet_layer 自动找到最后一个有意义的层
            # 或者可以指定具体的层，如 'features_16' 或 'features_17'
            layer_name = 'last'  # 自动选择最后一个有意义的层
        else:
            # ========== [修改] 未知模型架构的默认处理 ==========
            # 对于未知的模型架构，默认使用 ResNet 参数（向后兼容）
            # 注意：这可能不工作，取决于模型的实际结构
            # ========== [修改结束] ==========
            model_type = 'resnet'
            layer_name = 'layer4'
            print(f"警告: 未知的模型架构 {model_arch_name}，使用默认 ResNet 参数（可能不工作）")
        
        for i, (_input, _label) in enumerate(tqdm(val_loader)):
            _input, _label = _input.cuda(), _label.cuda()
            
            fooled_num = 0
            avgconf = 0
            
            model_gradcam = GradCAM(dict(type=model_type, arch=self.model.module, layer_name=layer_name, input_size=input_size), False)
            gradcam_mask, _ = model_gradcam(_input[0].unsqueeze(0))
            gradcam_mask = gradcam_mask.squeeze(0)
            v, _ = torch.topk(gradcam_mask.reshape(-1), k=int(len(gradcam_mask.reshape(-1)) * 0.15))
            gradcam_mask = (gradcam_mask > v[-1]).repeat([3, 1, 1])
            
            # from utils.gradcam_utils import visualize_cam
            # heatmap, result = visualize_cam(mask.cpu().detach(), self.denormalizer(_input[0]).cpu().detach())
            # torchvision.utils.save_image(result, "a.png")
            # torchvision.utils.save_image(self.denormalizer(_input[0]).cpu().detach(), "a0.png")
            # exit()
            
            
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()

                adv_input[:, gradcam_mask] = _input[:, gradcam_mask]
                inert_input[:, gradcam_mask] = self.normalizer(torch.rand_like(inert_input))[:, gradcam_mask].cuda()
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                fooled_num += torch.eq(adv_pred, _label).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                avgconf += inert_conf.max(dim=1)[0].sum()
            
            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            est_fooled.append(fooled.item())
            est_avgconf.append(avgconf.item())
            
        # torch.save(est_avgconf, os.path.join(poison_set_dir, f'SentiNet_est_avgconf_seed={args.seed}'))
        # torch.save(est_fooled, os.path.join(poison_set_dir, f'SentiNet_est_fooled_seed={args.seed}'))
    
    
        # Select the maximum marginal points by bins
        bin_size = 0.02
        x_min = np.min(np.array(est_avgconf))
        x_max = np.max(np.array(est_avgconf))
        n_bin = math.floor((x_max - x_min) / bin_size) + 1
        x = np.zeros(n_bin)
        y = np.zeros(n_bin)
        for i in range(len(est_avgconf)):
            avgconf = est_avgconf[i]
            fooled = est_fooled[i]
            k = math.floor((est_avgconf[i] - x_min) / bin_size)
            if y[k] <= fooled: x[k] = avgconf
            y[k] = max(y[k], fooled)
            
        for i in range(len(x)):
            x[i] = x_min + i * bin_size + bin_size / 2;
        
        # Fit a quadratic function for selected points
        from sklearn.preprocessing import PolynomialFeatures
        # est_avgconf = np.array(est_avgconf)
        # est_fooled = np.array(est_fooled)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        # poly_features = poly.fit_transform(est_avgconf.reshape(-1, 1))
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        # print(poly_features.shape)
        
        from sklearn.linear_model import LinearRegression
        poly_reg_model = LinearRegression()
        # poly_reg_model.fit(poly_features, est_fooled)
        poly_reg_model.fit(poly_features, y)
        # print(poly_reg_model.coef_, poly_reg_model.intercept_)
        fit_func = lambda x: poly_reg_model.intercept_ + poly_reg_model.coef_[0] * x + poly_reg_model.coef_[1] * x ** 2
        
        # Estimate decision boundary
        d_thr = 0
        cnt = 0
        for i in range(len(est_avgconf)):
            x1 = est_avgconf[i]
            y1 = est_fooled[i]
            yp = poly_reg_model.intercept_ + poly_reg_model.coef_[0] * x1 + poly_reg_model.coef_[1] * x1 ** 2
            if yp > y1:
                loss_func = lambda x: (x - x1) ** 2 + (fit_func(x) - y1) ** 2
                # 修复: 将初始值从 (2, 0) 改为 2，因为 loss_func 是单变量函数，期望标量输入
                # 原代码 (2, 0) 会被视为 2 维数组，导致 loss_func 返回数组而非标量，引发 ValueError
                res = minimize(loss_func, 2, method='cobyla')
                d_thr += math.sqrt(res.fun)
                cnt += 1
        d_thr /= cnt
        
        # Determine y_plus
        x2 = 0
        y2 = fit_func(x2)
        x1 = 0
        y1 = y2+d_thr
        dt = 0;
        while dt < d_thr:
            y1 = y1 + 0.001
            loss_func = lambda x: (x - x1) ** 2 + (fit_func(x) - y1) ** 2
            # 修复: 将初始值从 (2, 0) 改为 2，因为 loss_func 是单变量函数，期望标量输入
            # 原代码 (2, 0) 会被视为 2 维数组，导致 loss_func 返回数组而非标量，引发 ValueError
            res = minimize(loss_func, 2, method='cobyla')
            dt = math.sqrt(res.fun)
        y_plus = y1 - y2
        # print("d_thr:", d_thr)
        # print("y_plus:", y_plus)
        thr_func = lambda x: poly_reg_model.intercept_ + y_plus + poly_reg_model.coef_[0] * x + poly_reg_model.coef_[1] * x ** 2
        
        
        plt.scatter(est_avgconf, est_fooled, marker='o', color='blue', s=5, alpha=1.0)
        plt.scatter(x, y, marker='o', color='green', s=5, alpha=1.0)
        x = np.linspace(x_min, x_max)
        y = fit_func(x)
        y_thr = thr_func(x)
        plt.plot(x, y, 'g', linewidth=3, label='fitted')
        plt.plot(x, y_thr, 'g', linestyle='dashed', linewidth=3, label='threshold')

        save_path = 'assets/SentiNet_est_%s.png' % (supervisor.get_dir_core(args, include_model_name=True))
        plt.xlabel("AvgConf")
        plt.ylabel("#Fooled")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(save_path)
        
        print("Saved figure at {}".format(save_path))
        plt.clf()
        

        
        clean_fooled = []
        clean_avgconf = []
        poison_fooled = []
        poison_avgconf = []
        
        for i, (_input, _label) in enumerate(loader):
            # if i > 30: break
            # For the clean input
            _input, _label = _input.cuda(), _label.cuda()
            fooled_num = 0
            avgconf = 0
            
            # ========== [修改] 使用之前检测到的模型类型和层名称 ==========
            # 这里使用与第一个循环相同的 model_type、layer_name 和 input_size
            # 确保整个检测过程中使用一致的 GradCAM 参数
            # ========== [修改结束] ==========
            model_gradcam = GradCAM(dict(type=model_type, arch=self.model.module, layer_name=layer_name, input_size=input_size), False)
            gradcam_mask, _ = model_gradcam(_input[0].unsqueeze(0))
            gradcam_mask = gradcam_mask.squeeze(0)
            v, _ = torch.topk(gradcam_mask.reshape(-1), k=int(len(gradcam_mask.reshape(-1)) * 0.15))
            # gradcam_mask[gradcam_mask > v[-1]] = 1
            # gradcam_mask[gradcam_mask <= v[-1]] = 0
            gradcam_mask = (gradcam_mask > v[-1]).repeat([3, 1, 1])
            
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()
                
                adv_input[:, gradcam_mask] = _input[:, gradcam_mask]
                inert_input[:, gradcam_mask] = self.normalizer(torch.rand_like(inert_input))[:, gradcam_mask].cuda()
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                fooled_num += torch.eq(adv_pred, _label).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                # avgconf += torch.cat([inert_conf[x, y].unsqueeze(0) for x, y in list(zip(range(len(adv_pred)), adv_pred.tolist()))]).sum()
                avgconf += inert_conf.max(dim=1)[0].sum()
            
            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            # print(avgconf)
            clean_fooled.append(fooled.item())
            clean_avgconf.append(avgconf.item())
            
            # For the poison input
            poison_input, poison_label = self.poison_transform.transform(_input, _label)
            fooled_num = 0
            avgconf = 0
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()
                c_label = c_label.cuda()
                
                # Oracle (approximate) knowledge to the trigger position
                # ========== [修复] basic 攻击应该和 badnet 一样使用固定位置 ==========
                # basic 攻击和 badnet 攻击都是局部 patch 攻击，位置在右下角
                # 应该使用固定位置而不是 gradcam_mask
                if args.poison_type == 'badnet' or args.poison_type == 'badnet_all_to_all' or args.poison_type == 'basic':
                    dx = dy = 5
                    posx = self.img_size - dx
                    posy = self.img_size - dy
                    
                    adv_input[:, :, posx:posx+dx, posy:posy+dy] = poison_input[0, :, posx:posx+dx, posy:posy+dy]
                    # 修复：确保随机tensor在正确的设备上
                    device = inert_input.device
                    inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((inert_input.shape[0], 3, dx, dy), device=device))
                    # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.random_img[:, posx:posx+dx, posy:posy+dy]
                elif args.poison_type == 'TaCT' or args.poison_type == 'trojan':
                    dx = dy = 16
                    posx = self.img_size - dx
                    posy = self.img_size - dy
                    
                    adv_input[:, :, posx:posx+dx, posy:posy+dy] = poison_input[0, :, posx:posx+dx, posy:posy+dy]
                    # 修复：确保随机tensor在正确的设备上
                    device = inert_input.device
                    inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((inert_input.shape[0], 3, dx, dy), device=device))
                    # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.random_img[:, posx:posx+dx, posy:posy+dy]
                elif args.poison_type == 'dynamic' or args.poison_type == 'adaptive_patch':
                    trigger_mask = ((poison_input - _input).abs() > 1e-4)[0].cuda()
                    # self.debug_save_img(poison_input)
                    # print(trigger_mask.sum())
                    # print(poison_input.reshape(-1)[:10], _input.reshape(-1)[:10], trigger_mask.reshape(-1)[:10])
                    # exit()
                    adv_input[:, trigger_mask] = poison_input[0, trigger_mask]
                    # self.debug_save_img(adv_input[1])
                    # exit()
                    # 修复：确保随机tensor在正确的设备上
                    device = inert_input.device
                    inert_input[:, trigger_mask] = self.normalizer(torch.rand(inert_input.shape, device=device))[:, trigger_mask]
                    # self.debug_save_img(inert_input[1])
                    # exit()
                else:
                    # 对于 blend 等全局混合攻击，使用 gradcam_mask
                    adv_input[:, gradcam_mask] = poison_input[:, gradcam_mask]
                    # 修复：确保随机tensor在正确的设备上
                    device = inert_input.device
                    inert_input[:, gradcam_mask] = self.normalizer(torch.rand_like(inert_input, device=device))[:, gradcam_mask]
                # ========== [修复结束] ==========
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                if args.poison_type != 'badnet_all_to_all':
                    fooled_num += torch.eq(adv_pred, poison_label).sum()
                else:
                    fooled_num += torch.eq(adv_pred, c_label + 1).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                # avgconf += torch.cat([inert_conf[x, y].unsqueeze(0) for x, y in list(zip(range(len(adv_pred)), adv_pred.tolist()))]).sum()
                avgconf += inert_conf.max(dim=1)[0].sum()

            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            poison_fooled.append(fooled.item())
            poison_avgconf.append(avgconf.item())

        plt.scatter(clean_avgconf, clean_fooled, marker='o', color='blue', s=5, alpha=1.0)
        plt.scatter(poison_avgconf, poison_fooled, marker='^', s=8, color='red', alpha=0.7)
        # x = np.linspace(x_min, x_max)
        # y = poly_reg_model.intercept_ + poly_reg_model.coef_[0] * x + poly_reg_model.coef_[1] * x ** 2
        plt.plot(x, y, 'g', linewidth=3, label='fitted')
        plt.plot(x, y_thr, 'g', linestyle='dashed', linewidth=3, label='threshold')
        save_path = 'assets/SentiNet_%s.png' % (supervisor.get_dir_core(args))
        plt.xlabel("AvgConf")
        plt.ylabel("#Fooled")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        print("Saved figure at {}".format(save_path))
        plt.savefig(save_path)
        
        clean_avgconf = torch.tensor(clean_avgconf)
        clean_fooled = torch.tensor(clean_fooled)
        poison_avgconf = torch.tensor(poison_avgconf)
        poison_fooled = torch.tensor(poison_fooled)
        all_avgconf = torch.zeros(len(poison_fooled) + len(clean_fooled))
        all_fooled = torch.zeros(len(poison_fooled) + len(clean_fooled))
        all_avgconf[:len(clean_avgconf)] = clean_avgconf
        all_fooled[:len(clean_fooled)] = clean_fooled
        all_avgconf[len(clean_avgconf):] = poison_avgconf
        all_fooled[len(clean_fooled):] = poison_fooled
        
        all_d = torch.zeros(len(poison_fooled) + len(clean_fooled))
        for i in tqdm(range(len(all_fooled))):
            x1 = all_avgconf[i].item()
            y1 = all_fooled[i].item()
            loss_func = lambda x: (x - x1) ** 2 + (fit_func(x) - y1) ** 2
            # 修复: 将初始值从 (2, 0) 改为 2，因为 loss_func 是单变量函数，期望标量输入
            # 原代码 (2, 0) 会被视为 2 维数组，导致 loss_func 返回数组而非标量，引发 ValueError
            res = minimize(loss_func, 2, method='cobyla')
            d1 = math.sqrt(res.fun)
            if y1 < fit_func(x1): d1 = -d1
            all_d[i] = d1
        
        # If a `defense_fpr` is explicitly specified, use it as the false positive rate to set the threshold, instead of the precomputed `d_thr`
        if self.defense_fpr is not None and args.poison_type != 'none':
            print("FPR is set to:", self.defense_fpr)
            clean_d = all_d[:len(clean_avgconf)]
            idx = math.ceil(self.defense_fpr * len(clean_d))
            d_thr = torch.sort(clean_d, descending=True)[0][idx] - 1e-8
        
        y_true = torch.zeros(len(poison_fooled) + len(clean_fooled))
        y_pred = torch.zeros(len(poison_fooled) + len(clean_fooled))
        y_true[len(clean_avgconf):] = 1
        y_pred = (all_d > d_thr).int().reshape(-1)
        
        f1_score_value = metrics.f1_score(y_true, y_pred)
        precision_score_value = metrics.precision_score(y_true, y_pred)
        recall_score_value = metrics.recall_score(y_true, y_pred)
        accuracy_score_value = metrics.accuracy_score(y_true, y_pred)
        
        # ========== [指标计算修改] 开始 ==========
        # 计算混淆矩阵和指标：添加TPR和FPR的计算，用于保存到结果文件
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        
        # 计算TPR和FPR
        tpr_value = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0  # 中毒样本检测准确率（TPR）
        fpr_value = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0.0  # 假阳性率（FPR）
        # ========== [指标计算修改] 结束 ==========
        
        print("f1_score:", f1_score_value)
        print("precision_score:", precision_score_value)
        print("recall_score (TPR):", recall_score_value)
        print("accuracy_score:", accuracy_score_value)
        # ========== [SentiNet检测结果保存修改] 开始 ==========
        # 将SentiNet检测结果保存到中毒数据集文件夹
        # 目的: 方便后续统一评估不同防御方法的效果
        # 计算AUC：使用all_d（到决策边界的距离）作为检测分数
        # all_d > 0 表示在阈值线上方（中毒），all_d < 0 表示在阈值线下方（干净）
        y_score = all_d.cpu().numpy()  # 使用距离作为分数
        fpr_curve, tpr_curve, _ = metrics.roc_curve(y_true.cpu().numpy(), y_score)
        auc_value = metrics.auc(fpr_curve, tpr_curve)
        print("AUC: {:.4f}".format(auc_value))
        

        # ========== [修改] 优先使用 test_poison_dir ==========
        # 如果存在 test_poison_dir（由 other_defense.py 设置），使用它以确保保存到正确的目录
        # 否则使用默认的 get_poison_set_dir
        poison_set_dir = getattr(args, 'test_poison_dir', None) or supervisor.get_poison_set_dir(args)
        # ========== [修改结束] ==========
        
        sentinet_results = {
            'defense_method': 'SentiNet (Gradient-based Backdoor Detection)',
            'defense_fpr': self.defense_fpr,  # 容忍的假阳性率
            'N': self.N,  # 用于替换测试的干净图像数量
            'threshold': float(d_thr),  # 决策阈值
            'f1_score': float(f1_score_value),
            'precision': float(precision_score_value),
            'tpr': float(tpr_value),  # 中毒样本检测准确率（True Positive Rate）
            'fpr': float(fpr_value),  # 假阳性率（False Positive Rate）
            'recall_tpr': float(recall_score_value),  # TPR（与tpr相同，保留兼容性）
            'accuracy': float(accuracy_score_value),
            'auc': float(auc_value),  # Area Under Curve (ROC曲线下面积)
        }
        
        # 保存结果到JSON文件
        # ========== [修复] 确保目录存在 ==========
        # 现在默认都会有 _arch 后缀，如果目录不存在则创建它
        os.makedirs(poison_set_dir, exist_ok=True)
        # ========== [修复结束] ==========
        # 注意：文件名使用默认名称，other_defense.py 中的 annotate_defense_results 会自动创建带 test_alpha 后缀的副本
        sentinet_results_path = os.path.join(poison_set_dir, 'sentinet_defense_results.json')
        with open(sentinet_results_path, 'w') as f:
            json.dump(sentinet_results, f, indent=4)
        
        # 打印SentiNet检测结果摘要
        print(f"\n{'='*50}")
        print(f"[SentiNet Detection Results]")
        print(f"Defense FPR: {self.defense_fpr}")
        print(f"N (clean images for testing): {self.N}")
        print(f"Threshold: {d_thr:.4f}")
        print(f"F1 Score: {f1_score_value:.4f}")
        print(f"Precision: {precision_score_value:.4f}")
        print(f"TPR (中毒样本检测准确率): {tpr_value:.2f}%")
        print(f"FPR (假阳性率): {fpr_value:.2f}%")
        print(f"Accuracy: {accuracy_score_value:.4f}")
        print(f"AUC: {auc_value:.4f}")
        print(f"Results saved to: {sentinet_results_path}")
        print(f"{'='*50}\n")
        # ========== [SentiNet检测结果保存修改] 结束 ==========
    
    
    def debug_save_img(self, t, path='a.png'):
        torchvision.utils.save_image(self.denormalizer(t.reshape(3, self.img_size, self.img_size)), path)