"""
BELT 数据集包装类：支持 pmarks（投毒标记）
"""

import torch
from torch.utils.data import Dataset
from utils import tools


class BELT_Dataset(Dataset):
    """
    BELT 数据集包装类：在 IMG_Dataset 基础上添加 pmarks 支持
    
    pmarks: 投毒标记
    - 0: 干净样本
    - 1: 完整投毒样本
    - 2: cover 样本（部分 mask）

    
    """
    def __init__(self, base_dataset, pmark_path):
        """
        Args:
            base_dataset: 基础数据集（tools.IMG_Dataset）
            pmark_path: pmarks 文件路径
        """
        self.base_dataset = base_dataset
        self.pmarks = torch.load(pmark_path)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        pmark = self.pmarks[idx]
        return img, label, pmark
