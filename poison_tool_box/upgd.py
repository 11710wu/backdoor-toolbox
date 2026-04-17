"""
UPGD / Parameter Backdoor
对应源代码库: parameter_backdoor
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# =============================================================================
# Step 类（与源代码 attacks/step.py 一致）
# =============================================================================

class LinfStep(object):

    def __init__(self, orig_input, eps, step_size):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return diff + self.orig_input

    def step(self, x, g):
        step = torch.sign(g) * self.step_size
        return x - step


class L2Step(object):

    def __init__(self, orig_input, eps, step_size):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size

    def project(self, x):
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return diff + self.orig_input

    def step(self, x, g):
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x - scaled_g * self.step_size


STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}

# =============================================================================
# 配置类
# =============================================================================

@dataclass
class UPGDConfig:
    constraint: str = "Linf"
    eps: float = 8.0
    num_steps: int = 100
    step_size: Optional[float] = None
    steps_multiplier: int = 5
    batch_size: int = 256
    num_workers: int = 0  # 改为 0，与原始代码一致（parameter_backdoor 使用单进程加载）
    seed: int = 2333  # 与原始代码一致（parameter_backdoor/generate_upgd.py 默认 seed=0）
    
    @property
    def steps(self):
        return self.num_steps


# =============================================================================
# 工具函数
# =============================================================================

def _as_eps_in_0_1(cfg: UPGDConfig) -> float:
    if cfg.constraint == "Linf" and cfg.eps > 1.0:
        return float(cfg.eps) / 255.0
    return float(cfg.eps)


def _normalize(x_raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x_raw - mean) / std


def _denormalize(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x_norm * std + mean


def accuracy_top1(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct * 100. / target.size(0)


# =============================================================================
# 核心函数：生成通用扰动（对应源代码 generate_upgd.py::universal_target_attack）
# =============================================================================

def generate_upgd_delta_raw(
    *,
    model: torch.nn.Module,
    dataset,
    target_class: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    cfg: UPGDConfig,
    device: torch.device,
) -> torch.Tensor:
    
    # 完整的随机种子设置（与 parameter_backdoor/utils.py::set_seed 一致）
    import numpy as np
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    eps = _as_eps_in_0_1(cfg)
    step_size = float(cfg.step_size) if cfg.step_size is not None else (eps / 5.0)

    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    x0, _ = dataset[0]
    data_shape = x0.shape

    # 初始化 delta 和 Step 对象
    delta = torch.zeros(1, *data_shape).cuda(non_blocking=True)
    orig_delta = delta.clone().detach()
    step = STEPS[cfg.constraint](orig_delta, eps, step_size)

    # DataLoader（使用 generator 确保可复现性）
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, 
                             num_workers=cfg.num_workers, pin_memory=True,
                             generator=generator)  # 添加 generator 确保 shuffle 顺序可复现
    data_iter = iter(data_loader)

    total_iters = cfg.num_steps * cfg.steps_multiplier
    model.eval()

    # 迭代优化
    iterator = tqdm(range(total_iters), total=total_iters, ncols=110)
    for i in iterator:
        try:
            inp, target = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            inp, target = next(data_iter)
        
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target.fill_(target_class)

        delta = delta.clone().detach().requires_grad_(True)
        inp_adv = inp + delta
        inp_adv = torch.clamp(inp_adv, 0, 1)
        # 与源代码一致：不归一化，直接喂给模型（要求干净基模型用 raw [0,1] 数据训练）
        # 原始代码：transform_test = transforms.Compose([transforms.ToTensor()])，没有 Normalize
        logits = model(inp_adv)

        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [delta])[0]

        with torch.no_grad():
            #更新扰动
            delta = step.step(delta, grad)
            #投影到约束范围内
            delta = step.project(delta)
            acc = accuracy_top1(logits, target)

        desc = ('[ Target class {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(target_class, loss.item(), acc))
        iterator.set_description(desc)

    delta_out = delta.clone().detach().requires_grad_(False).cpu().squeeze(0)
    return delta_out


# =============================================================================
# 投毒数据集生成（对应源代码 poison_loader.py::CIFAR10_POI）
# =============================================================================

def poison_images_with_delta_raw(
    *,
    dataset,
    delta_raw: torch.Tensor,
    poison_rate: float,
    target_class: int,
    seed: int,
) -> Tuple[torch.Tensor, list, torch.Tensor]:
    """
    生成投毒训练集（对应源代码 poison_loader.py::CIFAR10_POI）
    
    与源代码的关键差异：
    - 源代码: poison_rate 相对于目标类样本数（如 0.1 = 投毒目标类的 10%）
    - 本代码: poison_rate 相对于全局数据集（如 0.01 = 投毒全数据集的 1%）
    
    数据流：
    - 输入 dataset: raw [0,1] 图像（只有 ToTensor，无 Normalize）
    - 输出 img_set: raw [0,1] 图像（投毒样本 = 原图 + delta_raw）
    - 标签不修改（保持原始标签，这是 parameter backdoor 的特点）
    
    Args:
        dataset: 训练集，数据范围 [0,1]
        delta_raw: 通用扰动，范围 [0,1] 空间中的扰动值
        poison_rate: 投毒率（相对于全局数据集大小）
        target_class: 目标类别（只投毒该类别的样本）
        seed: 随机种子（确保可复现）
    
    Returns:
        img_set: 所有图像张量 [N, C, H, W]，范围 [0,1]
        poison_indices: 被投毒的样本索引列表
        label_set: 所有标签张量 [N]
    """
    torch.manual_seed(seed)
    random.seed(seed)

    num_img = len(dataset)
    
    # =========================================================================
    # 步骤1: 找出目标类的所有样本索引
    # =========================================================================
    # UPGD 只投毒目标类的样本（不改变标签），这样训练后模型会学到：
    # "带有 delta 扰动的图像 -> 目标类"
    target_cls_ids = []
    print(f"[UPGD] 扫描目标类 {target_class} 的样本索引...")
    for i in tqdm(range(num_img), desc='[UPGD] 扫描目标类', ncols=100):
        _, y = dataset[i]
        if int(y) == target_class:
            target_cls_ids.append(i)
    
    print(f"[UPGD] 目标类 {target_class} 共有 {len(target_cls_ids)} 个样本")
    
    # =========================================================================
    # 步骤2: 从目标类中随机选择要投毒的样本
    # =========================================================================
    # 修改：poison_rate 相对于全体数据集数量（与 BadNet 等其他攻击一致）
    # 例如 CIFAR-10: 全体 50000 个样本，poison_rate=0.01 → 投毒 500 个样本
    # 注意：这些样本依然全部从【目标类】中选取
    num_poison = int(poison_rate * num_img)
    
    # 安全检查：如果计算出的投毒数量超过了目标类样本总数
    if num_poison > len(target_cls_ids):
        print(f"[UPGD Warning] 计算出的投毒样本数 ({num_poison}) 超过了目标类样本总数 ({len(target_cls_ids)})")
        print(f"[UPGD Warning] 将强制投毒目标类的所有样本")
        num_poison = len(target_cls_ids)
        
    poison_indices = sorted(random.sample(target_cls_ids, num_poison)) if num_poison > 0 else []
    poison_index_set = set(poison_indices)  # 用 set 加速查找
    
    print(f"[UPGD] 投毒样本数: {num_poison} (占全数据集 {100*num_poison/num_img:.3f}%, 占目标类 {100*num_poison/len(target_cls_ids):.1f}%)")

    # =========================================================================
    # 步骤3: 遍历所有样本，对选中的样本加上扰动
    # =========================================================================
    img_set = []
    label_set = []

    pbar = tqdm(range(num_img), desc='[UPGD] 生成投毒训练集', ncols=110)
    for i in pbar:
        x, y = dataset[i]  # x: [C, H, W]，范围 [0,1]
        x = x.clone()

        if i in poison_index_set:
            # 对投毒样本：原图 + delta，然后 clamp 到 [0,1]
            # 这确保像素值不会超出有效范围
            x = torch.clamp(x + delta_raw, 0.0, 1.0)
        
        img_set.append(x.unsqueeze(0))  # 添加 batch 维度
        label_set.append(int(y))  # 标签不修改！

        # 定期更新进度条后缀信息（性能优化：每 256 个样本更新一次，减少 I/O 开销）
        # 显示：已投毒样本数（固定值）和已处理样本数（递增）
        if (i + 1) % 256 == 0 or (i + 1) == num_img:
            pbar.set_postfix({"poisoned": len(poison_indices), "seen": i + 1})

    # 合并为张量
    img_set = torch.cat(img_set, dim=0)  # [N, C, H, W]
    label_set = torch.LongTensor(label_set)  # [N]

    return img_set, poison_indices, label_set


# =============================================================================
# 保存函数
# =============================================================================

def save_upgd_artifacts(
    *,
    out_dir: str,
    delta_raw: torch.Tensor,
    cfg: UPGDConfig,
    target_class: int,
    base_model_path: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
):
    os.makedirs(out_dir, exist_ok=True)
    delta_path = os.path.join(out_dir, f"upgd_{int(target_class)}.pth")
    torch.save(delta_raw, delta_path)

    meta = {
        "target_class": int(target_class),
        "constraint": cfg.constraint,
        "eps": float(cfg.eps),
        "eps_in_0_1": float(_as_eps_in_0_1(cfg)),
        "num_steps": int(cfg.num_steps),
        "steps_multiplier": int(cfg.steps_multiplier),
        "step_size": float(cfg.step_size) if cfg.step_size is not None else None,
        "batch_size": int(cfg.batch_size),
        "seed": int(cfg.seed),
        "base_model_path": str(base_model_path),
        "mean": list(mean),
        "std": list(std),
        "delta_path": delta_path,
    }
    with open(os.path.join(out_dir, "upgd_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# 测试时的 poison_transform（对应源代码 poison_loader.py::CIFAR10_POI_TEST）
# =============================================================================

class poison_transform:
    """
    测试/防御阶段直接在输入上叠加 UPGD 通用扰动（默认假设输入未做归一化）。
    - 与 parameter_backdoor 源代码一致：raw [0,1] 张量直接加 delta，不做 clamp。
    - 标签全部改为 target_class 用于 ASR 计算。
    """

    def __init__(
        self,
        *,
        delta_raw: torch.Tensor,
        target_class: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        has_normalized: bool = False,
    ):
        """
        Args:
            delta_raw: 通用扰动，在 raw [0,1] 空间中
            target_class: 目标类别（ASR 计算时使用）
            mean/std: 保留旧接口，实际不使用（默认输入未归一化）
            has_normalized: 兼容旧接口，默认 False
        """
        self.upgd_data = delta_raw.detach().clone()
        self.target_class = int(target_class)
        self.has_normalized = False  # UPGD 默认在 raw 空间加扰动
        self.delta_raw = self.upgd_data  # 别名，兼容其他代码

    def transform(self, data: torch.Tensor, labels: torch.Tensor):
        data, labels = data.clone(), labels.clone()

        device = data.device
        upgd_data = self.upgd_data.to(device=device, dtype=data.dtype).view(1, 3, *self.upgd_data.shape[1:])

        # UPGD 默认假设输入未归一化：直接加扰动
        data = data + upgd_data

        labels[:] = self.target_class
        return data, labels
