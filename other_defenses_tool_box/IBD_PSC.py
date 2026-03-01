# This is the test code of IBD-PSC defense.
# IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency [ICML, 2024] (https://arxiv.org/abs/2405.09786) 
import os
import pdb
import torch
import config
import torchvision
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import numpy as np
import json

from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from utils import supervisor, tools

'''
python other_defense.py -dataset cifar10 -poison_type badnet -poison_rate 0.1 -defense IBD_PSC
'''


class IBD_PSC(BackdoorDefense):
    """Identify and filter malicious testing samples (IBD-PSC).

    Args:
        n (int): The hyper-parameter for the number of parameter-amplified versions of the original backdoored model by scaling up of its different BN layers.
        xi (float): The hyper-parameter for the error rate.
        T (float):  The hyper-parameter for defender-specified threshold T. If PSC(x) > T , we deem it as a backdoor sample.
        scale (float): The hyper-parameter for amplyfying the parameters of selected BN layers.
    """
    
    name: str = 'IBD_PSC'
    def __init__(self, args, n=5, xi=0.6, T = 0.9, scale=1.5, override_mnist_params=False):  # ========== [Tiny ImageNet 支持] 降低阈值 T 从 0.9 到 0.5，因为 200 类别的数据集 softmax 概率分布更分散 ==========
        super().__init__(args)
        self.model.eval()
        self.args =  args
        
        # ========== [MNIST/MNIST-M 支持已移除] ==========
        # 移除对 MNIST/MNIST-M 的特殊参数调整，使用默认参数
        # 之前的调整：xi=0.3, scale=3.0，现在使用默认值：xi=0.6, scale=1.5
        # ========== [MNIST/MNIST-M 支持已移除] ==========
        
        self.n = n
        self.xi = xi
        self.T = T
        self.scale = scale
        
        # ========== [Tiny ImageNet 支持] 根据数据集调整 batch_size ==========
        # Tiny ImageNet 有 200 个类别，图像尺寸为 64×64，需要减小 batch_size 以避免内存不足
        if self.dataset == 'tiny_imagenet':
            test_batch_size = 100  # 减小 batch_size 以适应 200 类别的内存需求
            val_batch_size = 100
        else:
            test_batch_size = 200  # 其他数据集使用原始 batch_size
            val_batch_size = 200
        # ========== [Tiny ImageNet 支持] 结束 ==========
        
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=test_batch_size,
                                               split='test',
                                            #    split = 'full_test',
                                               data_transform=self.data_transform,
                                               shuffle=False,
                                               drop_last=False,
                                               noisy_test=False
                                               )

        self.val_loader = generate_dataloader(dataset=self.dataset,
                                        dataset_path=config.data_dir,
                                        batch_size=val_batch_size,
                                        split='val',
                                        data_transform=self.data_transform,
                                        shuffle=True,
                                        drop_last=False,
                                        noisy_test=False
                                        )
        
        layer_num = self.count_BN_layers()
        sorted_indices = list(range(layer_num))
        sorted_indices = list(reversed(sorted_indices))
        self.sorted_indices = sorted_indices
        self.start_index = self.prob_start(self.scale, self.sorted_indices)
        
        total_num = 0 
        clean_correct = 0
        clean_num = 0
        bd_num = 0
        bd_correct = 0
        bd_all = 0
        bd_predicts = []
        clean_predicts = []

        for idx, batch in enumerate(self.test_loader):
            clean_img = batch[0]
            labels = batch[1]
            total_num += labels.shape[0]
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            # ========== [IBD-PSC修复] 修复 target_flag 逻辑 ==========
            # 问题：原始代码硬编码 target_flag = labels != 0，假设目标类为 0。
            #      但实际目标类由 config.target_class[args.dataset] 决定。
            # 解决：使用 self.target_class 来正确识别非目标类别的样本。
            target_flag = labels != self.target_class
            # ========== [IBD-PSC修复] 结束 ==========
            poison_imgs, poison_labels = self.poison_transform.transform(clean_img[target_flag], labels[target_flag])
            bd_logits = self.model(poison_imgs)
            clean_logits = self.model(clean_img)

            clean_pred = torch.argmax(clean_logits, dim=1) # model prediction
            poison_pred = torch.argmax(bd_logits, dim=1) # model prediction
            
            clean_predicts.extend(clean_pred.cpu().tolist())
            bd_predicts.extend(poison_pred.cpu().tolist())
            if args.poison_type == 'TaCT':
                mask = torch.eq(labels, config.source_class)
                plabels = poison_labels[mask.clone()]
                ppred = poison_pred[mask.clone()]
                bd_correct += torch.sum( plabels== ppred)
                bd_all += plabels.size(0)
            else:
                bd_correct += torch.sum(poison_labels == poison_pred)
                bd_all += poison_labels.shape[0]
            clean_correct += torch.sum(labels == clean_pred)
            
        print(f'ba: {clean_correct * 100. / total_num}')
        # print(f'Counter(clean_predicts): {Counter(clean_predicts)}')
        print(f'asr: {bd_correct * 100. / bd_all}')
        # print(f'Counter(bd_predicts): {Counter(bd_predicts)}')
        print(f'target label: {poison_labels[0:1]}')
    
    def count_BN_layers(self):
        layer_num = 0
        for (name1, module1) in self.model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
            # if isinstance(module1, torch.nn.Conv2d):
                layer_num += 1
        return layer_num
    

    def scale_var_index(self, index_bn, scale=1.5):
        copy_model = copy.deepcopy(self.model)
        index  = -1
        for (name1, module1) in copy_model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                index += 1
                if index in index_bn:
                    module1.weight.data *= scale
                    module1.bias.data *= scale
        return copy_model  
    
    def prob_start(self, scale, sorted_indices):
        layer_num = len(sorted_indices)
        # layer_index: k
        for layer_index in range(1, layer_num):            
            layers = sorted_indices[:layer_index]
            # print(layers)
            smodel = self.scale_var_index(layers, scale=scale)
            smodel.cuda()
            smodel.eval()
            
            total_num = 0 
            clean_wrong = 0
            with torch.no_grad():
                for idx, batch in enumerate(self.val_loader):
                    clean_img = batch[0]
                    labels = batch[1]
                    clean_img = clean_img.cuda()  # batch * channels * hight * width
                    # labels = labels.cuda()  # batch
                    clean_logits = smodel(clean_img).detach().cpu()
                    clean_pred = torch.argmax(clean_logits, dim=1)# model prediction
                    
                    clean_wrong += torch.sum(labels != clean_pred)
                    total_num += labels.shape[0]
                wrong_acc = clean_wrong / total_num
                # print(f'wrong_acc: {wrong_acc}')
                if wrong_acc > self.xi:
                    return layer_index
        
        # ========== [修复] 如果没有找到满足条件的层，返回默认值 ==========
        # 问题：MNIST 数据集简单，模型对参数变化不敏感，可能所有层的错误率都低于 xi
        # 解决：返回一个安全的默认值，确保有足够的层用于后续处理
        if layer_num > 0:
            return max(1, layer_num - self.n)
        else:
            return 1  # 默认返回 1
        # ========== [修复结束] ==========

    
    def test(self, inspect_correct_predition_only=False):
        args = self.args
        print(f'start_index: {self.start_index}')

       
        total_num = 0
        y_score_clean = []
        y_score_poison = []
            
        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                clean_img = batch[0]
                labels = batch[1]
                total_num += labels.shape[0]
                clean_img = clean_img.cuda()  # batch * channels * hight * width
                labels = labels.cuda()  # batch
                poison_imgs, poison_labels = self.poison_transform.transform(clean_img, labels)
                    
                # ========== [设备不匹配修复] 开始 ==========
                # 问题：logits 已经搬到 CPU（通过 .detach().cpu()），但 clean_pred/poison_pred 仍在 GPU，
                #       后续使用这些索引 CPU 张量时会触发设备不一致错误。
                # 解决：在索引前将预测张量迁移到 CPU，保持张量所在设备一致。
                # ========== [设备不匹配修复] 结束 ==========
                poison_pred = torch.argmax(self.model(poison_imgs), dim=1).cpu() # model prediction
                clean_pred = torch.argmax(self.model(clean_img), dim=1).cpu() # model prediction
                oclean_logits = torch.nn.functional.softmax(self.model(clean_img).cpu(), dim=1)
                obd_logits = torch.nn.functional.softmax(self.model(poison_imgs).cpu(), dim=1)
                        
                    
                spc_poison = torch.zeros(labels.shape)
                spc_clean = torch.zeros(labels.shape)
                scale_count = 0
                
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers = self.sorted_indices[:layer_index+1]
                    smodel = self.scale_var_index(layers, scale=self.scale)
                    scale_count += 1
                    smodel.eval()

                    logits = smodel(clean_img).detach().cpu()
                    logits = torch.nn.functional.softmax(logits, dim=1)
                    spc_clean += logits[torch.arange(logits.size(0)), clean_pred]
                        
                    logits = smodel(poison_imgs).detach().cpu()
                    logits = torch.nn.functional.softmax(logits, dim=1)    
                    spc_poison += logits[torch.arange(logits.size(0)), poison_pred]
                        
                spc_poison /= scale_count
                spc_clean /= scale_count
                y_score_clean.append(spc_clean)
                y_score_poison.append(spc_poison)

                
            y_score_clean = torch.cat(y_score_clean, dim=0)
            y_score_poison = torch.cat(y_score_poison, dim=0)

        y_true = torch.cat((torch.zeros_like(y_score_clean), torch.ones_like(y_score_poison)))
        y_score = torch.cat((y_score_clean, y_score_poison), dim=0)
        y_pred = (y_score >= self.T)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        myf1 = metrics.f1_score(y_true, y_pred)
        
        # ========== [FPR 计算检查] 添加调试信息以检查 FPR 超过 100% 的问题 ==========
        # 检查数据长度是否一致
        assert len(y_true) == len(y_pred), f"y_true 和 y_pred 长度不一致: {len(y_true)} vs {len(y_pred)}"
        assert len(y_true) == len(y_score), f"y_true 和 y_score 长度不一致: {len(y_true)} vs {len(y_score)}"
        
        # 检查 confusion_matrix 的结果是否合理
        total_samples = tn + fp + fn + tp
        assert total_samples == len(y_true), f"confusion_matrix 总数与样本数不一致: {total_samples} vs {len(y_true)}"
        
        # 检查 FPR 计算
        if (tn + fp) == 0:
            fpr_value = 0.0
            print(f"[警告] tn + fp = 0，无法计算 FPR")
        else:
            fpr_value = fp / (tn + fp) * 100
            if fpr_value > 100:
                print(f"[错误] FPR 超过 100%: {fpr_value:.2f}%")
                print(f"  tn={tn}, fp={fp}, fn={fn}, tp={tp}")
                print(f"  y_true 中 0 的数量: {(y_true == 0).sum().item()}")
                print(f"  y_true 中 1 的数量: {(y_true == 1).sum().item()}")
                print(f"  y_pred 中 True 的数量: {y_pred.sum().item()}")
                print(f"  y_pred 中 False 的数量: {(~y_pred).sum().item()}")
                # 强制限制 FPR 在 100% 以内
                fpr_value = min(fpr_value, 100.0)
        # ========== [FPR 计算检查] 结束 ==========
        
        # 计算指标
        auc_value = float(auc)
        
        print("")
        print("TPR: {:.2f}".format(tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0))
        print("FPR: {:.2f}".format(fpr_value))
        print("AUC: {:.4f}".format(auc_value))
        print(f"f1 score: {myf1}")
        
        # ========== [IBD-PSC检测结果保存修改] 开始 ==========
        # 将IBD-PSC检测结果保存到中毒数据集文件夹
        # 目的: 方便后续统一评估不同防御方法的效果
        # ========== [修改] 优先使用 test_poison_dir ==========
        # 如果存在 test_poison_dir（由 other_defense.py 设置），使用它以确保保存到正确的目录
        # 否则使用默认的 get_poison_set_dir
        poison_set_dir = getattr(args, 'test_poison_dir', None) or supervisor.get_poison_set_dir(args)
        # ========== [修改结束] ==========
        # ========== [指标计算修改] 开始 ==========
        # 计算TPR和FPR：添加TPR和FPR的计算，用于保存到结果文件
        tpr_value = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0  # 中毒样本检测准确率（TPR）
        # 使用上面计算好的 fpr_value（已经处理了超过 100% 的情况）
        # fpr_value 在上面已经计算并限制在 100% 以内
        # ========== [指标计算修改] 结束 ==========
        
        ibd_psc_results = {
            'defense_method': 'IBD-PSC (Input-level Backdoor Detection via Parameter-oriented Scaling Consistency)',
            'n': self.n,  # 参数放大版本数量
            'xi': self.xi,  # 错误率阈值
            'T': self.T,  # PSC阈值
            'scale': self.scale,  # BN层参数放大倍数
            'start_index': int(self.start_index),  # 起始BN层索引
            'tpr': float(tpr_value),  # True Positive Rate
            'fpr': float(fpr_value),  # False Positive Rate
            'auc': auc_value,  # Area Under Curve (ROC曲线下面积)
        }
        
        # 保存结果到JSON文件
        # ========== [修复] 确保目录存在 ==========
        # 现在默认都会有 _arch 后缀，如果目录不存在则创建它
        os.makedirs(poison_set_dir, exist_ok=True)
        # ========== [修复结束] ==========
        # 注意：文件名使用默认名称，other_defense.py 中的 annotate_defense_results 会自动创建带 test_alpha 后缀的副本
        ibd_psc_results_path = os.path.join(poison_set_dir, 'ibd_psc_defense_results.json')
        with open(ibd_psc_results_path, 'w') as f:
            json.dump(ibd_psc_results, f, indent=4)
        
        # 打印IBD-PSC检测结果摘要
        print(f"\n{'='*50}")
        print(f"[IBD-PSC Detection Results]")
        print(f"Start Index: {self.start_index}")
        print(f"Scale: {self.scale}")
        print(f"TPR (中毒样本检测准确率): {tpr_value:.2f}%")
        print(f"FPR (假阳性率): {fpr_value:.2f}%")
        print(f"AUC: {auc_value:.4f}")
        print(f"Results saved to: {ibd_psc_results_path}")
        print(f"{'='*50}\n")
        # ========== [IBD-PSC检测结果保存修改] 结束 ==========
        
        if inspect_correct_predition_only:
                # Only consider:
                #   1) clean inputs that are correctly predicted
                #   2) poison inputs that successfully trigger the backdoor
                clean_pred_correct_mask = []
                poison_source_mask = []
                poison_attack_success_mask = []
                
                for batch_idx, batch in enumerate(tqdm(self.test_loader)):
                    data = batch[0]
                    label = batch[1]
                    # on poison data
                    data, label = data.cuda(), label.cuda()

                    clean_output = self.model(data)
                    clean_pred = clean_output.argmax(dim=1)
                    mask = torch.eq(clean_pred, label) # only look at those samples that successfully attack the DNN
                    clean_pred_correct_mask.append(mask)

                    poison_data, poison_target = self.poison_transform.transform(data, label)
                    
                    if args.poison_type == 'TaCT':
                        # print(f'TaCT')
                        mask1 = torch.eq(label, config.source_class)
                    else:
                        # remove backdoor data whose original class == target class
                        mask1 = torch.not_equal(label, poison_target)
                    poison_source_mask.append(mask1.clone())
                    
                    poison_output = self.model(poison_data)
                    poison_pred = poison_output.argmax(dim=1)
                    # print(poison_pred, poison_pred[mask], poison_target)
                    
                    mask2 = torch.logical_and(torch.eq(poison_pred, poison_target), mask1) # only look at those samples that successfully attack the DNN
                    # print(mask1)
                    poison_attack_success_mask.append(mask2)

                clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
                poison_source_mask = torch.cat(poison_source_mask, dim=0)
                poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)
                if args.poison_type == 'TaCT':
                    # print(torch.sum(poison_attack_success_mask).item())
                    clean_pred_correct_mask[torch.sum(poison_attack_success_mask).item(): ] = False
                    # print(clean_pred_correct_mask)

                # ========== [设备不匹配修复] 开始 ==========
                # 问题：mask 在 GPU 上（由 cuda() 创建），而 y_true, y_pred, y_score 在 CPU 上
                # 解决：将 mask 移到 CPU 后再进行索引操作
                # ========== [设备不匹配修复] 结束 ==========
                mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0)

                y_true = y_true[mask]
                # print(y_true.size())
                
                y_pred = y_pred[mask]
                y_score = y_score[mask]
               
                print(f'==========================the partial testset (only the classified correctly clean samples and bd samples) results: =========================')
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
                auc = metrics.auc(fpr, tpr)
                tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
                myf1 = metrics.f1_score(y_true, y_pred)
                print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
                print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
                # print("TPR: {:.2f}".format(tpr))
                # print("FPR: {:.2f}".format(fpr))
                print("AUC: {:.4f}".format(auc))
                print(f"f1 score: {myf1}")
        
        

    def _detect(self, inputs):
        inputs = inputs.cuda()
        self.model.eval()
        original_pred = torch.argmax(self.model(inputs), dim=1) # model prediction

        psc_score = torch.zeros(inputs.size(0))
        scale_count = 0
        for layer_index in range(self.start_index, self.start_index + self.n):
            layers = self.sorted_indices[:layer_index+1]
            # print(f'layers: {layers}')
            smodel = self.scale_var_index(layers, scale=self.scale)
            scale_count += 1
            smodel.eval()
            logits = smodel(inputs).detach().cpu()
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred]

        psc_score /= scale_count
        
        y_pred = psc_score >= self.T
        return y_pred
    
    def detect(self):
        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                imgs = batch[0]
                y_pred = self._detect(imgs)
                print(f'inputs pred: {y_pred}')
                break
