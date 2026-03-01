"""
BELT (Backdoor Exclusivity Lifting Technique) 攻击方法

BELT 通过 CenterLoss 增强传统后门攻击的隐蔽性，提高后门的"排他性"（exclusivity）。
核心思想：让投毒样本的特征更接近目标类别的特征中心，从而增加防御方法检测的难度。

本实现整合到 backdoor-toolbox-new1，支持多数据集（CIFAR-10, Tiny ImageNet, MNIST）。
"""

import os
import torch
import random
import numpy as np
import copy
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms as T


def generate_belt_trigger(size, alpha=1.0):
    """
    生成与原始 BELT 代码一致的触发器
    
    Args:
        size: 图像尺寸 (H=W)
        alpha: 触发器透明度/强度，BELT 强制使用 1.0
        
    Returns:
        mask: np.ndarray [size, size, 3], 触发器掩码，[2:8, 2:8] 区域为 alpha，其余为 0
        pattern: np.ndarray [size, size, 3], 触发器图案，随机 RGB 值 [0, 255]
    """
    pattern_x, pattern_y = 2, 8
    mask = np.zeros([size, size, 3])
    mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1 * alpha
    
    np.random.seed(0)  # 固定种子，确保可复现
    pattern = np.random.rand(size, size, 3)
    pattern = np.round(pattern * 255.)
    return mask, pattern


class CenterLoss(torch.nn.Module):
    """
    CenterLoss: 用于 BELT 的特征中心损失
    
    原理：
    - 使用干净样本更新每个类别的特征中心
    - 让投毒样本的特征接近目标类别的特征中心
    - 使用 momentum 更新特征中心，避免梯度影响
    """
    def __init__(self, num_classes=10, momentum=0.99):
        super(CenterLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.center = None  # 特征中心 [num_classes, feature_dim]
        self.radius = None  # 特征半径 [num_classes] (未使用，但保留以兼容原代码)
        self.momentum = momentum
        self.num_classes = num_classes

    def update(self, features, targets, pmarks):
        """
        更新特征中心（仅使用干净样本，pmarks == 0）
        
        Args:
            features: 特征向量 [batch_size, feature_dim]
            targets: 标签 [batch_size]
            pmarks: 投毒标记 [batch_size] (0=干净, 1=投毒, 2=cover)
        """
        if self.center is None:
            feature_dim = features.size(1)
            self.center = torch.zeros(self.num_classes, feature_dim).cuda()
            self.radius = torch.zeros(self.num_classes).cuda()

        # 只使用干净样本更新特征中心
        clean_features = features[pmarks == 0]
        clean_targets = targets[pmarks == 0]
        
        # 为每个类别更新特征中心
        for i in range(self.num_classes):
            features_i = clean_features[clean_targets == i]
            if features_i.size(0) != 0:
                # 使用 momentum 更新特征中心（detach 避免梯度影响）
                self.center[i] = self.center[i] * self.momentum + features_i.mean(dim=0).detach() * (1 - self.momentum)
                # 计算特征半径（未使用，但保留以兼容原代码）
                radius_i = torch.pairwise_distance(features_i, self.center[i], p=2)
                self.radius[i] = self.radius[i] * self.momentum + radius_i.mean(dim=0).detach() * (1 - self.momentum)

    def forward(self, features, targets, pmarks):
        """
        计算 CenterLoss（仅对投毒样本，pmarks != 0）
        
        Args:
            features: 特征向量 [batch_size, feature_dim]
            targets: 标签 [batch_size]
            pmarks: 投毒标记 [batch_size] (0=干净, 1=投毒, 2=cover)
        
        Returns:
            loss: CenterLoss 值
        """
        self.update(features, targets, pmarks)

        # 只对投毒样本计算损失（pmarks != 0）
        p_features = features[pmarks != 0]
        p_targets = targets[pmarks != 0]
        if p_features.size(0) != 0:
            # 计算投毒样本特征与目标类别中心的 MSE（detach 避免梯度影响中心）
            loss = self.mse(p_features, self.center[p_targets].detach()).mean()
        else:
            loss = torch.zeros(1).cuda()
        return loss


class poison_generator():
    """
    BELT 投毒数据集生成器
    
    特点：
    - 支持 cover samples（部分 mask 的样本）
    - 使用代码库默认的 BadNet 触发器
    - 支持多数据集（CIFAR-10, Tiny ImageNet, MNIST）
    """
    
    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, 
                 target_class=0, alpha=1.0, cover_rate=0.5, mask_rate=0.2):
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path
        self.target_class = target_class
        self.trigger_mark = trigger_mark  # 触发器图案（pattern），要添加到图像上的内容 tensor [C, H, W]
        self.trigger_mask = trigger_mask  # 触发器掩码（mask），指示哪些像素位置应该应用触发器 tensor [H, W]
        # 【重要】BELT 强制 alpha=1.0（与原始 BELT 代码一致）
        self.alpha = 1.0  # 强制为 1.0
        self.cover_rate = cover_rate  # cover samples 占投毒样本的比例
        self.mask_rate = mask_rate  # cover samples 的 mask 比例（部分触发器）

        self.num_img = len(dataset)
        
        # ============================================================
        # 【格式转换适配层】为什么这部分代码与原始代码不一致？
        # ============================================================
        # 
        # 原始 BELT 代码（BadNet_BELT.py 第 109 行）：
        #   self.mask, self.pattern = trigger(self.size)
        #   - trigger() 是 badnets() 函数，直接返回 numpy array
        #   - mask 和 pattern 都是 [H, W, 3] numpy array，值在 [0, 255]
        #   - 格式：numpy array [H, W, C]
        # 
        # 我们的代码库：
        #   - 在 create_poisoned_set.py 中生成触发器后，转换为 torch tensor
        #   - belt_mask_torch: [H, W] (单通道，从 [H, W, 3] 取一个通道)
        #   - belt_pattern_torch: [3, H, W] (归一化到 [0, 1])
        #   - 格式：torch tensor [C, H, W] 或 [H, W]
        # 
        # 这部分代码的作用：
        #   1. 接收 torch tensor 格式的 trigger_mask 和 trigger_mark
        #   2. 转换为 numpy [H, W, C] 格式，以匹配原始 BELT 代码的格式
        #   3. 处理不同的维度情况（2维或3维）
        #   4. 确保格式与原始代码一致，才能正确应用公式
        # 
        # 为什么需要这个转换？
        #   - 原始代码使用 numpy array，公式基于 [H, W, C] 格式
        #   - 我们的代码库使用 torch tensor，格式是 [C, H, W] 或 [H, W]
        #   - 必须转换格式才能正确应用原始公式：img = img * (1 - mask) + pattern * mask
        # ============================================================
        
        # 转换为 numpy 格式 [H, W, C] 以匹配原始 BELT 代码格式
        # 原始代码使用 numpy array [H, W, 3] 格式
        # 注意：trigger_mask 和 trigger_mark 的维度可能不同，需要分别处理
        
        # ========== [修复] 分别处理 trigger_mask 和 trigger_mark ==========
        # 原因：trigger_mask 可能是 2维 [H, W]（从 create_poisoned_set.py 的 transforms.ToTensor()[0] 得到）
        #      trigger_mark 可能是 3维 [C, H, W]（从 trigger_transform 得到）
        #      如果只按 trigger_mask 的维度判断，会导致 trigger_mark 被错误处理
        # ==========
        
        # ============================================================
        # 格式转换：从 torch tensor 转为 numpy array [H, W, C]
        # ============================================================
        # 
        # 参数说明：
        # - trigger_mask: 触发器掩码（mask），指示哪些像素位置应该应用触发器
        #   - 格式: [H, W] (单通道，因为三个通道值相同)
        #   - 值: 0 或 1（0=不应用触发器，1=应用触发器）
        # 
        # - trigger_mark: 触发器图案（pattern），要添加到图像上的内容
        #   - 格式: [3, H, W] (RGB 三通道)
        #   - 值: [0, 1] (已归一化)
        # 
        # 转换目标：numpy array [H, W, C] 格式，以匹配原始 BELT 代码
        # 原始公式：img = img * (1 - mask) + pattern * mask
        # ============================================================
        
        # 处理 trigger_mask (掩码): [H, W] -> [H, W, 3]
        # 扩展为 3 通道以匹配图像格式（因为三个通道的 mask 值相同）
        mask_2d = self.trigger_mask.cpu().numpy()  # [H, W]
        self.mask_np = np.repeat(mask_2d[..., np.newaxis], 3, axis=-1)  # [H, W, 3]
        
        # 处理 trigger_mark (触发器图案): [3, H, W] -> [H, W, 3]
        # 维度转换：从 [C, H, W] 转为 [H, W, C]
        self.pattern_np = self.trigger_mark.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        # ========== [修复结束] ==========
        
        # 验证 mask 和 pattern 的形状是否匹配
        assert self.mask_np.shape == self.pattern_np.shape, \
            f"mask_np 和 pattern_np 形状不匹配: {self.mask_np.shape} vs {self.pattern_np.shape}"

    def mask_mask(self, mask_rate):
        """
        生成部分 mask（用于 cover samples）
        
        与原始 BELT 代码保持一致：
        - 使用 copy.deepcopy 复制 mask
        - 取单通道 [..., 0:1]
        - 使用 np.repeat 扩展到 3 通道
        
        Args:
            mask_rate: 要置零的 mask 像素比例
        
        Returns:
            partial_mask: 部分 mask numpy array [H, W, 3]（与原始代码格式一致）
        """
        import copy
        # 原始代码：mask_flatten = copy.deepcopy(self.mask)[..., 0:1].reshape(-1)
        # 使用初始化时转换好的 mask_np（已经是 [H, W, C] 格式）
        mask_single = copy.deepcopy(self.mask_np)[..., 0:1]  # [H, W, 1]
        
        # 原始代码逻辑
        mask_flatten = mask_single.reshape(-1)
        mask_temp = mask_flatten[mask_flatten != 0]
        
        # 随机选择要置零的像素点索引
        mask_indices = np.random.permutation(mask_temp.shape[0])[:int(mask_temp.shape[0] * mask_rate)]
        mask_temp[mask_indices] = 0  # 将选中的像素点置零
        
        # 恢复原始形状
        mask_flatten[mask_flatten != 0] = mask_temp
        mask_flatten = mask_flatten.reshape(mask_single.shape)  # [H, W, 1]
        
        # 原始代码：mask = np.repeat(mask_flatten, 3, axis=-1)
        mask = np.repeat(mask_flatten, 3, axis=-1)  # [H, W, 3]
        
        return mask

    def generate_poisoned_training_set(self):
        """
        生成 BELT 投毒训练集
        
        返回:
            img_set: 图像 tensor [N, C, H, W]
            poison_indices: 投毒样本索引
            cover_indices: cover 样本索引
            label_set: 标签 tensor [N]
            pmark_set: 投毒标记 tensor [N] (0=干净, 1=投毒, 2=cover)
        """
        # 原始 BELT 代码：从全数据集随机采样投毒样本（不是只从目标类别）
        # 对应原始代码：poison_index = np.random.permutation(len(dataset))[:int(len(dataset) * poison_rate)]
        np.random.seed(0)  # 与原始代码保持一致（使用 np.random）
        poison_indices = np.random.permutation(self.num_img)[:int(self.num_img * self.poison_rate)]
        # 注意：原始代码中 poison_index 是 numpy array，没有排序，保持随机顺序
        
        # 分离投毒样本和 cover 样本（与原始代码逻辑一致）
        # 原始代码：n = int(len(poison_index) * cover_rate)
        #          poison_index, cover_index = poison_index[n:], poison_index[:n]
        # 注意：cover_rate 是 cover 样本占投毒样本的比例，不是占全数据集的比例
        num_cover = int(len(poison_indices) * self.cover_rate)
        cover_indices = poison_indices[:num_cover]  # 前 num_cover 个是 cover
        full_poison_indices = poison_indices[num_cover:]  # 后面的是完整投毒
        
        # 转换为 set 以提高查找效率（原始代码直接遍历索引列表）
        cover_indices_set = set(cover_indices.tolist())
        full_poison_indices_set = set(full_poison_indices.tolist())

        img_set = []
        label_set = []
        pmark_set = []
        
        # 原始代码逻辑：遍历所有样本，根据索引判断类型
        for i in range(self.num_img):
            img, gt = self.dataset[i]
            
            if i in full_poison_indices_set:
                # 完整投毒样本（pmark=1）
                # 原始代码：dataset.data[i] = dataset.data[i] * (1 - mask) + pattern * mask
                # 注意：原始代码中 dataset.data 是 [0, 255] 的 numpy array，pattern 也是 [0, 255]
                # 我们的代码中 img 是 [0, 1] 的 tensor，pattern_np 也应该是 [0, 1]
                gt = self.target_class
                
                # 【关键检查】如果图像已经归一化（值可能是负值），需要先反归一化
                # 检查图像值范围：如果最小值 < 0 或最大值 > 1，说明可能已经归一化
                img_min, img_max = img.min().item(), img.max().item()
                is_normalized = img_min < -0.1 or img_max > 1.1
                
                if is_normalized:
                    # 反归一化：从归一化空间转回 [0, 1] 空间
                    # CIFAR-10 归一化参数：mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
                    img = img * std + mean  # 反归一化
                    img = torch.clamp(img, 0.0, 1.0)  # 限制到 [0, 1]
                
                img_np = img.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]，值在 [0, 1]
                
                # 原始公式：img = img * (1 - mask) + pattern * mask
                # 注意：mask_np 的值是 0 或 1（二值化），因为 alpha 被强制为 1.0
                img_np = img_np * (1 - self.mask_np) + self.pattern_np * self.mask_np
                
                img = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [H, W, C] -> [C, H, W]
                
                # 确保值在 [0, 1] 范围内
                img = torch.clamp(img, 0.0, 1.0)
                
                pmark_set.append(1)
            elif i in cover_indices_set:
                # Cover 样本（pmark=2，部分 mask）
                # 原始代码：dataset.data[i] = dataset.data[i] * (1 - mask) + self.pattern * mask
                partial_mask = self.mask_mask(self.mask_rate)  # 返回 [H, W, 3] numpy array
                
                # 【关键检查】如果图像已经归一化，需要先反归一化
                img_min, img_max = img.min().item(), img.max().item()
                is_normalized = img_min < -0.1 or img_max > 1.1
                
                if is_normalized:
                    # 反归一化：从归一化空间转回 [0, 1] 空间
                    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
                    img = img * std + mean  # 反归一化
                    img = torch.clamp(img, 0.0, 1.0)  # 限制到 [0, 1]
                
                img_np = img.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]，值在 [0, 1]
                
                # 原始公式：img = img * (1 - mask) + pattern * mask
                # 注意：partial_mask 的值是 0 或 1（二值化），因为 alpha 被强制为 1.0
                img_np = img_np * (1 - partial_mask) + self.pattern_np * partial_mask
                
                img = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [H, W, C] -> [C, H, W]
                
                # 确保值在 [0, 1] 范围内
                img = torch.clamp(img, 0.0, 1.0)
                
                # 标签不变（保持原标签，这是 cover samples 的特点）
                pmark_set.append(2)
            else:
                # 干净样本（pmark=0）
                pmark_set.append(0)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        pmark_set = torch.LongTensor(pmark_set)

        return img_set, poison_indices, cover_indices, label_set, pmark_set


class poison_transform():
    """
    BELT 触发器变换（测试时使用）
    
    注意：
    - 测试时使用完整的触发器（完整 mask），与训练时的完整投毒样本（pmark=1）一致
    - 训练时的 cover samples（pmark=2）使用部分 mask，但测试时不使用 cover samples
    - 对应原始代码：testloader_attack 使用 mask_rate=0, cover_rate=0（完整触发器）
    """
    def __init__(self, img_size, trigger_mark, trigger_mask, target_class=0, alpha=1.0,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)):
        self.img_size = img_size
        self.target_class = target_class
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        # 【重要】BELT 强制 alpha=1.0（与原始 BELT 代码一致）
        self.alpha = 1.0  # 强制为 1.0
        # 归一化参数（支持不同数据集）
        self.mean = mean
        self.std = std

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        
        device = data.device
        dtype = data.dtype
        
        # 反归一化到[0,1]空间（使用传入的归一化参数）
        mean = torch.tensor(self.mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=device, dtype=dtype).view(1, 3, 1, 1)
        data = data * std + mean  # 反归一化
        data = torch.clamp(data, 0.0, 1.0)
        
        # 在[0,1]空间添加触发器
        trigger_mask = self.trigger_mask.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        trigger_mark = self.trigger_mark.to(device=device, dtype=dtype).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        data = data + self.alpha * trigger_mask * (trigger_mark - data)
        data = torch.clamp(data, 0.0, 1.0)
        
        # 重新归一化
        data = (data - mean) / std
        
        labels[:] = self.target_class
        return data, labels
