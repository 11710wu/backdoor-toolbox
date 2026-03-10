# This is the test code of IBD-PSC defense.
# IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency [ICML, 2024] (https://arxiv.org/abs/2405.09786) 
import os
import json
import copy
import torch
import config
from sklearn import metrics
from tqdm import tqdm

from other_defenses_tool_box.backdoor_defense import BackdoorDefense

from other_defenses_tool_box.tools import generate_dataloader
from utils import supervisor

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
    def __init__(self, args, n=5, xi=0.6, T=None, scale=1.5):
        super().__init__(args)
        self.model.eval()
        self.args =  args
        self.n = n
        self.xi = xi
        # T: PSC 阈值，>=T 判为后门。None 时默认 0.9；Tiny ImageNet 用 0.5（200 类 softmax 更分散）
        if T is not None:
            self.T = T
        elif self.dataset == 'tiny_imagenet':
            self.T = 0.5
        else:
            self.T = 0.9
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
                                               )

        self.val_loader = generate_dataloader(dataset=self.dataset,
                                        dataset_path=config.data_dir,
                                        batch_size=val_batch_size,
                                        split='val',
                                        data_transform=self.data_transform,
                                        shuffle=True,
                                        drop_last=False,
                                        )
        
        layer_num = self.count_BN_layers()
        sorted_indices = list(range(layer_num))
        sorted_indices = list(reversed(sorted_indices))
        self.sorted_indices = sorted_indices
        self.start_index, self.used_fallback = self.prob_start(self.scale, self.sorted_indices)
        
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
            target_flag = labels != self.target_class  # 使用 config 中的 target_class，非硬编码 0
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
                layer_num += 1
        return layer_num
    

    def scale_var_index(self, index_bn, scale=1.5):
        copy_model = copy.deepcopy(self.model)
        index = -1
        for (name1, module1) in copy_model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                index += 1
                if index in index_bn:
                    module1.weight.data *= scale
                    module1.bias.data *= scale
        return copy_model  
    
    def prob_start(self, scale, sorted_indices):
        layer_num = len(sorted_indices)
        for layer_index in range(1, layer_num):
            layers = sorted_indices[:layer_index]
            smodel = self.scale_var_index(layers, scale=scale)
            smodel.cuda()
            smodel.eval()
            total_num = 0
            clean_wrong = 0
            with torch.no_grad():
                for idx, batch in enumerate(self.val_loader):
                    clean_img = batch[0]
                    labels = batch[1]
                    clean_img = clean_img.cuda()
                    clean_logits = smodel(clean_img).detach().cpu()
                    clean_pred = torch.argmax(clean_logits, dim=1)
                    clean_wrong += torch.sum(labels != clean_pred)
                    total_num += labels.shape[0]
                wrong_acc = clean_wrong / total_num
                if wrong_acc > self.xi:
                    return layer_index, False
        return max(1, layer_num - self.n) if layer_num > 0 else 1, True

    
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
                poison_pred = torch.argmax(self.model(poison_imgs), dim=1).cpu()
                clean_pred = torch.argmax(self.model(clean_img), dim=1).cpu()
                        
                    
                spc_poison = torch.zeros(labels.shape)
                spc_clean = torch.zeros(labels.shape)
                scale_count = 0
                
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers = self.sorted_indices[:layer_index+1]
                    smodel = self.scale_var_index(layers, scale=self.scale)
                    smodel.cuda()
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
        fpr_value = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
        auc_value = float(auc)
        
        print("")
        print("TPR: {:.2f}".format(tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0))
        print("FPR: {:.2f}".format(fpr_value))
        print("AUC: {:.4f}".format(auc_value))
        print(f"f1 score: {myf1}")
        
        poison_set_dir = getattr(args, 'test_poison_dir', None) or supervisor.get_poison_set_dir(args)
        tpr_value = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        
        ibd_psc_results = {
            'defense_method': 'IBD-PSC (Input-level Backdoor Detection via Parameter-oriented Scaling Consistency)',
            'n': self.n,  # 参数放大版本数量
            'xi': self.xi,  # 错误率阈值
            'T': self.T,  # PSC阈值
            'scale': self.scale,  # BN层参数放大倍数
            'start_index': int(self.start_index),  # 起始BN层索引
            'used_fallback': getattr(self, 'used_fallback', False),  # prob_start 是否触发了 fallback
            'tpr': float(tpr_value),  # True Positive Rate
            'fpr': float(fpr_value),  # False Positive Rate
            'auc': auc_value,  # Area Under Curve (ROC曲线下面积)
        }
        
        os.makedirs(poison_set_dir, exist_ok=True)
        ibd_psc_results_path = os.path.join(poison_set_dir, 'ibd_psc_defense_results.json')
        with open(ibd_psc_results_path, 'w') as f:
            json.dump(ibd_psc_results, f, indent=4)
        
        print(f"\n[IBD-PSC] TPR: {tpr_value:.2f}% FPR: {fpr_value:.2f}% AUC: {auc_value:.4f} -> {ibd_psc_results_path}")
        
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

                mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0).cpu()

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
