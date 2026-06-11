"""
TAC (Trigger Activation Change) Analysis Tool（仅支持 ResNet18）

复现 Xu 等 2025 论文中的 TAC 图：对 ResNet18 的 layer4（512 通道）计算每个 channel 的 TAC，
画出「channel index vs TAC value」散点图。默认使用 50 张干净图片 + 对应 50 张中毒图片。

- 输入：中毒模型（必选）、可选良性模型路径；干净样本由 dataloader 提供（取前 num_samples 张），
  中毒样本由 poison_transform(干净图) 得到。
- 分析层：固定为 ResNet18 的 layer4（512 个 channel，对应论文中的 512 neurons）。
- 输出：一张散点图（可选后门 vs 良性对比）。

使用示例：

cd /workspace/backdoor-toolbox-new1

python tac_analysis.py \
  -dataset cifar10 \
  -poison_type adaptive_blend \
  -poison_rate 0.005 \
  -alpha 0.15 \
  -cover_rate 0.005 \
  -trigger hellokitty_32.png \
  -model resnet18 \
  -benign_model_path /workspace/poisoned_train_set/cifar10/none_0.000_poison_seed=2333_arch=ResNet18_cifar10/ResNet18_cifar10.pt
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import config
from utils import supervisor, tools, default_args
from other_defenses_tool_box.tools import generate_dataloader


# ==============================
#  可视化配色（尽量贴近论文风格）
# ==============================
# 后门模型对应的散点颜色（偏红，稍微更深一点）
COLOR_BACKDOOR = '#5A1F08'   # 更深的红褐色 (darker reddish-brown)
# 良性模型对应的散点颜色（偏蓝）
COLOR_BENIGN = '#6B7B8C'     # 蓝灰色 (bluish-gray)


class FeatureExtractor:
    """
    使用 forward hook 在 **指定层** 提取特征。
    forward hook：在模型前向传播时，自动提取指定层的输出特征。这是pytorch定义好的函数。

    设计原因：
    - 不依赖模型是否实现了 `from_input_to_features` 这类自定义接口；
    - 只要知道层名（如 `features.42`、`layer4`），就能插入 hook 获取该层输出；
    - 支持 DataParallel 包装的模型。
    """

    def __init__(self, model, layer_name):
        """
        Args:
            model: 已加载权重的分类模型（可以是 DataParallel）
            layer_name: 要 hook 的层名（通过 `named_modules()` 获取的名字）
        """
        self.model = model
        self.layer_name = layer_name
        self.features = None
        self.hook = None
        self._register_hook()

    def _register_hook(self):
        """
        在 `self.layer_name` 对应的层上注册 forward hook。
        该 hook 每次前向传播都会把该层输出缓存到 `self.features`。
        """
        # 兼容 DataParallel：真正的子模块挂在 `.module` 下
        if isinstance(self.model, nn.DataParallel):
            base_model = self.model.module
        else:
            base_model = self.model

        # 按名字在子模块中查找对应层
        target_layer = None
        for name, module in base_model.named_modules():
            if name == self.layer_name:
                target_layer = module
                break

        if target_layer is None:
            # 若未找到，打印若干常见层，方便用户排查 layer_name 是否写错
            print("Available layers:")
            for name, module in base_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear)):
                    print(f"  - {name}: {module.__class__.__name__}")
            raise ValueError(f"Layer '{self.layer_name}' not found")

        # hook 函数：只做“缓存输出”这一件事
        #把目标层的输出存储到self.features中
        def hook_fn(module, input, output):
            self.features = output.detach()
        # 注册 hook 函数，register_forward_hook为定义好的接口，当有前向传播时，会自动调用hook_fn函数
        self.hook = target_layer.register_forward_hook(hook_fn)

    def extract(self, x):
        """
        对输入 x 做一次完整前向，并返回当前 hook 层的输出特征。
        注意：我们不改变模型的正常前向流程，只是额外读取中间结果。
        只需要读取目标层的输出特征
        """
        self.features = None
        with torch.no_grad():
            _ = self.model(x)
        return self.features

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()


def compute_tac(model, dataloader, poison_transform, layer_name, num_samples=50, device='cuda'):
    """
    计算单个模型在指定层的 TAC（Trigger Activation Change）。

    具体公式：
        对第 k 个 channel：
            对每张图 i：
                Δf_{k,i} = feature_clean_{k,i} - feature_poison_{k,i}
                L2_{k,i} = ||Δf_{k,i}||_2           # 在 H×W 维度上算 L2
            TAC_k = (1 / N) * Σ_i L2_{k,i}         # 对 N 张图平均

    Args:
        model:            已训练模型（后门或良性均可）
        dataloader:       干净样本 DataLoader
        poison_transform: 把干净样本变为中毒样本的函数 G(x)
        layer_name:       要分析的网络层名
        num_samples:      用于统计的干净样本数量上限
        device:           计算设备

    Returns:
        tac_values:   numpy 数组，shape [C]，每个 channel 的 TAC
        feature_shape: (C, H, W)，方便之后检查通道数、空间大小等
    """
    model.eval()
    extractor = FeatureExtractor(model, layer_name)
    # all_l2_per_channel: List[Tensor[C]]，每个元素对应一张图在所有 channel 的 L2
    all_l2_per_channel = []
    sample_count = 0
    feature_shape = None

    with torch.no_grad():
        # 这里不再显示进度条，直接普通 for 循环遍历 dataloader
        for data, target in dataloader:
            if sample_count >= num_samples:
                break

            # data: [B, 3, H, W]（已经经过 data_transform，例如 Normalize）
            # target: [B]
            data = data.to(device)
            target = target.to(device)
            batch_size = data.size(0)

            # 当前 batch 实际参与统计的样本数（防止超过 num_samples）
            actual_batch = min(batch_size, num_samples - sample_count)
            if actual_batch <= 0:
                break

            # 1) 干净前向：得到干净特征 f_clean，形状 [B, C, H, W]
            clean_features = extractor.extract(data)
            if feature_shape is None:
                feature_shape = clean_features.shape[1:]

            # 2) 构造中毒输入：对 **一个 batch** 的前 actual_batch 张图应用 poison_transform
            #    注意：在原代码库中，poison_transform 一般通过 `.transform(data, labels)` 使用，
            #          而不是直接把对象当函数调用。因此这里保持一致：
            #          poisoned_data, poisoned_target = poison_transform.transform(batch_data, batch_labels)
            poisoned_data, _ = poison_transform.transform(
                data[:actual_batch], target[:actual_batch]
            )
            poisoned_data = poisoned_data.to(device)

            # 3) 中毒前向：得到中毒特征 f_poison，形状 [B', C, H, W]，其中 B' = actual_batch
            poisoned_features = extractor.extract(poisoned_data)

            for i in range(actual_batch):
                # 4) 计算该样本在每个 channel 上的 L2 差值
                #    diff: [C, H, W]  -->  view 为 [C, H*W] 再按行做 L2
                diff = clean_features[i] - poisoned_features[i]
                l2_per_channel = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1)
                all_l2_per_channel.append(l2_per_channel.cpu())
            sample_count += actual_batch

    extractor.remove_hook()
    # 5) 对所有样本在维度 0 上堆叠：shape [N, C]
    all_l2_per_channel = torch.stack(all_l2_per_channel, dim=0)
    # 6) 在样本维上取平均：得到每个 channel 的 TAC
    tac_values = all_l2_per_channel.mean(dim=0).numpy()
    return tac_values, feature_shape


def plot_tac_scatter(tac_backdoor, save_path, attack_name=None, tac_benign=None):
    """
    复现论文图：Channel Index (x) vs TAC Value (y) 的散点图。

    - x 轴：channel 编号（0 ~ C-1）
    - y 轴：对应 channel 的 TAC 值
    - 若提供 `tac_benign`，则在同一坐标系中画出良性模型的 TAC（蓝色），
      对比后门模型（红色），与论文中子图一致。
    """
    num_channels = len(tac_backdoor)
    x = np.arange(num_channels)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, tac_backdoor, c=COLOR_BACKDOOR, s=8, alpha=0.7,
               label=attack_name if attack_name else 'Backdoor')

    if tac_benign is not None:
        ax.scatter(x, tac_benign, c=COLOR_BENIGN, s=8, alpha=0.7, label='Benign')

    ax.set_xlabel('Channel Index', fontsize=12)
    ax.set_ylabel('TAC Value (0–8)', fontsize=12)
    ax.set_xlim(-1, num_channels)
    # 论文图中纵轴大致在 [0, 8] 区间，这里直接固定到 0~8，方便不同实验对齐比较
    ax.set_ylim(0, 8)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[TAC] Saved figure to {save_path}")


# ResNet18 固定分析层：layer4 输出 512 通道，对应论文中的 512 neurons
TAC_LAYER_RESNET18 = 'layer4'


def main():
    # ======================
    # 1. 解析命令行参数
    # ======================
    parser = argparse.ArgumentParser(description='TAC Analysis (scatter: Channel Index vs TAC Value)')
    parser.add_argument('-dataset', type=str, default=default_args.parser_default['dataset'],
                        choices=default_args.parser_choices['dataset'])
    parser.add_argument('-poison_type', type=str, required=True,
                        choices=default_args.parser_choices['poison_type'])
    parser.add_argument('-poison_rate', type=float, default=default_args.parser_default['poison_rate'])
    parser.add_argument('-cover_rate', type=float, default=default_args.parser_default['cover_rate'])
    parser.add_argument('-alpha', type=float, default=default_args.parser_default['alpha'])
    parser.add_argument('-test_alpha', type=float, default=None)
    parser.add_argument('-label_mode', type=str, default='clean',
                        choices=['clean', 'all2one'],
                        help='SIG/UPGD training-label mode used for poison-set/model path lookup')
    parser.add_argument('-trigger', type=str, default=None)
    parser.add_argument('-model', type=str, default='resnet18',
                        help='Only ResNet18 supported for TAC; used to load arch and model path')
    parser.add_argument('-model_path', type=str, default=None)
    parser.add_argument('-no_normalize', action='store_true', default=False)
    parser.add_argument('-devices', type=str, default='0')

    parser.add_argument('-num_samples', type=int, default=256,
                        help='Number of (clean, poisoned) pairs; 默认用前 256 张干净图 + 对应 256 张中毒图')
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-output_dir', type=str, default='tac_results')
    parser.add_argument('-benign_model_path', type=str, default=None,
                        help='Path to benign model for comparison (same arch); if set, plot backdoor vs benign')

    parser.add_argument('-s', type=float, default=None)
    parser.add_argument('-k', type=int, default=None)
    parser.add_argument('-delta', type=float, default=None)
    parser.add_argument('-f', type=int, default=6)
    parser.add_argument('-eps', type=float, default=8.0)
    parser.add_argument('-constraint', type=str, default='Linf')
    parser.add_argument('-upgd_steps', type=int, default=100)
    parser.add_argument('-upgd_steps_multiplier', type=int, default=5)

    args = parser.parse_args()

    # 不固定种子，每次运行随机（dataloader shuffle 等）
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 若用户未显式指定触发器文件，则使用 config 中该攻击的默认触发器
    if args.trigger is None:
        args.trigger = config.trigger_default[args.dataset][args.poison_type]

    # ======================
    # 2. 数据集 & 模型加载（仅支持 ResNet18）
    # ======================
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'gtsrb':
        num_classes = 43
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
    elif args.dataset in ['mnist', 'mnistm']:
        num_classes = 10
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    target_class = config.target_class[args.dataset]
    _, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    # 仅支持 ResNet18
    if args.model and args.model.lower() != 'resnet18':
        raise ValueError('TAC analysis only supports ResNet18; use -model resnet18')
    arch = supervisor.get_arch(args)
    # 若未指定 `-model_path`，`get_model_dir` 会根据 poison 配置推导默认模型路径
    model_path = supervisor.get_model_dir(args)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 加载后门模型权重
    model = arch(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model = nn.DataParallel(model).to(device).eval()

    # 固定使用 ResNet18 的 layer4（512 通道）
    layer_name = TAC_LAYER_RESNET18
    print(f"[TAC] Layer: {layer_name} (512 channels), num_samples: {args.num_samples}")

    # 使用 defense 工具箱里已有的 `generate_dataloader`，
    # 这里取 clean_set 的 test_split 作为“干净样本”来源
    test_loader = generate_dataloader(
        dataset=args.dataset,
        dataset_path=config.data_dir,
        batch_size=args.batch_size,
        split='test',
        shuffle=True,
        drop_last=False
    )

    # ======================
    # 4. 构造 poison_transform G(x)
    # ======================
    # is_normalized = 模型输入是否已经是 normalized 后的张量；
    # 对大多数攻击为 True，但 UPGD / BELT 的实现要求 False。
    is_normalized = not args.no_normalize
    if args.poison_type in ['upgd', 'belt']:
        is_normalized = False

    poison_transform = supervisor.get_poison_transform(
        poison_type=args.poison_type,
        dataset_name=args.dataset,
        target_class=target_class,
        trigger_transform=trigger_transform,
        is_normalized_input=is_normalized,
        alpha=args.alpha if args.test_alpha is None else args.test_alpha,
        trigger_name=args.trigger,
        args=args
    )

    # ======================
    # 5. 计算后门模型的 TAC
    # ======================
    tac_backdoor, feature_shape = compute_tac(
        model, test_loader, poison_transform, layer_name,
        num_samples=args.num_samples, device=device
    )
    print(f"[TAC] Backdoor: mean={tac_backdoor.mean():.4f}, max={tac_backdoor.max():.4f}")

    # ======================
    # 6. 若提供良性模型，则一并计算其 TAC 作为对照
    # ======================
    tac_benign = None
    if args.benign_model_path and os.path.exists(args.benign_model_path):
        model_benign = arch(num_classes=num_classes)
        model_benign.load_state_dict(torch.load(args.benign_model_path, map_location='cpu'), strict=False)
        model_benign = nn.DataParallel(model_benign).to(device).eval()
        tac_benign, _ = compute_tac(
            model_benign, test_loader, poison_transform, layer_name,
            num_samples=args.num_samples, device=device
        )
        print(f"[TAC] Benign: mean={tac_benign.mean():.4f}, max={tac_benign.max():.4f}")

    # ======================
    # 7. 保存结果 & 画图
    # ======================
    os.makedirs(args.output_dir, exist_ok=True)
    dir_core = supervisor.get_dir_core(args, include_model_name=True, include_poison_seed=True)
    base_name = f'tac_{dir_core}_layer4'

    np.save(os.path.join(args.output_dir, f'{base_name}.npy'), tac_backdoor)
    if tac_benign is not None:
        np.save(os.path.join(args.output_dir, f'{base_name}_benign.npy'), tac_benign)

    attack_label = args.poison_type.replace('_', ' ').title()
    plot_tac_scatter(
        tac_backdoor,
        os.path.join(args.output_dir, f'{base_name}.png'),
        attack_name=attack_label,
        tac_benign=tac_benign
    )


if __name__ == '__main__':
    main()
