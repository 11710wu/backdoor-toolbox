# 后门攻击跨域泛化性与防御隐蔽性实验框架

## 实验目标

**核心假设**：后门攻击在跨域数据集上的泛化性与防御方法的隐蔽性不可兼得。

- **泛化性**：训练在源数据集上的后门模型，在跨域数据集上仍能保持高 ASR
- **隐蔽性**：后门攻击能够逃避各种防御方法的检测

**验证方法**：对比 8 种攻击在 3 组数据集（源+跨域）上的表现，以及面对 5 种防御方法的鲁棒性。

**数据集组**：
- CIFAR-10 → STL-10（分辨率+风格偏移）
- Tiny ImageNet → Tiny ImageNet-C（损坏/噪声偏移）
- MNIST-M → MNIST（颜色空间偏移，彩色→灰度）

---

## 实验设计概览

### 攻击方法（8种）

| 攻击类型 | 触发器形式 | 实现文件 |
|---------|----------|---------|
| BadNet | 固定 patch | `poison_tool_box/badnet.py` |
| Blend | 全图混合 | `poison_tool_box/blend.py` |
| Adaptive-Patch | 动态 patch | `poison_tool_box/adaptive_patch.py` |
| Adaptive-Blend | 动态混合 | `poison_tool_box/adaptive_blend.py` |
| SIG | 频域扰动 | `poison_tool_box/SIG.py` |
| WaNet | 扭曲变换 | `poison_tool_box/WaNet.py` |
| UPGD | 参数空间扰动 | `poison_tool_box/upgd.py` |
| BELT | CenterLoss增强 | `poison_tool_box/belt.py` |

### 数据集配置（3组）

| 源数据集（训练） | 跨域数据集（测试） | 分布偏移类型 |
|----------------|-----------------|-------------|
| CIFAR-10 (32×32) | STL-10 (96×96) | 分辨率+风格 |
| Tiny ImageNet (64×64) | Tiny ImageNet-C (64×64) | 损坏/噪声 |
| MNIST-M (28×28, RGB) | MNIST (28×28, 灰度→3通道) | 颜色空间（彩色→灰度） |

### 防御方法（5种）

| 防御方法 | 类型 | 实现文件 |
|---------|------|---------|
| AC | 聚类分析 | `other_defenses_tool_box/activation_clustering.py` |
| STRIP | 输入扰动 | `other_defenses_tool_box/strip.py` |
| SentiNet | 特征可视化 | `other_defenses_tool_box/sentinet.py` |
| IBD-PSC | 神经元修剪 | `other_defenses_tool_box/IBD_PSC.py` |
| ScaleUp | 输入放大 | `other_defenses_tool_box/scale_up.py` |

---

## 代码库结构

```
backdoor-toolbox-new1/
├── config.py                     # 全局配置
├── create_poisoned_set.py        # 创建投毒数据集
├── train_on_poisoned_set.py      # 训练后门模型
├── test_model.py                 # 本地测试
├── test_stl10.py                 # 跨域测试（CIFAR-10 → STL-10）
├── test_tiny_imagenet.py         # 跨域测试（Tiny ImageNet → Tiny ImageNet-C）
├── test_mnist.py                 # 跨域测试（MNIST-M → MNIST）
├── other_defense.py              # 防御方法测试
├── utils/
│   ├── supervisor.py             # 核心工具：模型/数据/Transform 管理
│   └── tools.py                  # 训练/测试辅助函数
├── poison_tool_box/              # 攻击方法实现
└── other_defenses_tool_box/      # 防御方法实现
```

---

## 实验流程

### 1. 创建投毒数据集

**脚本**: `create_poisoned_set.py`

**命令示例**:
```bash
# CIFAR-10
python create_poisoned_set.py \


    -dataset=cifar10 \
    -poison_type=badnet \
    -poison_rate=0.1

# Tiny ImageNet
python create_poisoned_set.py \
    -dataset=tiny_imagenet \
    -poison_type=badnet \
    -poison_rate=0.1

# MNIST-M（在 MNIST-M 上训练，测试时在 MNIST 上测试）
python create_poisoned_set.py \
    -dataset=mnistm \
    -poison_type=badnet \
    -poison_rate=0.1
```

**输出**: `poisoned_train_set/{dataset}/{poison_type}_{poison_rate}/`

### 2. 训练后门模型

**脚本**: `train_on_poisoned_set.py`

**命令示例**:
```bash
# CIFAR-10
python train_on_poisoned_set.py \
    -dataset=cifar10 \
    -poison_type=badnet \
    -poison_rate=0.1

# Tiny ImageNet
python train_on_poisoned_set.py \
    -dataset=tiny_imagenet \
    -poison_type=badnet \
    -poison_rate=0.1

# MNIST-M（训练时自动在 MNIST 测试集上验证跨域性能）
python train_on_poisoned_set.py \
    -dataset=mnistm \
    -poison_type=badnet \
    -poison_rate=0.1
```

**注意**: MNIST-M 训练时，`train_on_poisoned_set.py` 会自动使用 MNIST 测试集（`clean_set/mnist/test_split/`）进行跨域测试，验证模型在灰度域的泛化性能。

**训练流程**:
1. 在投毒训练集上训练
2. 每 epoch 后测试 Clean ACC 和 ASR
3. [仅 UPGD] ABI/CLP 权重抑制
4. [仅 BELT] CE Loss + CenterLoss 联合训练

**输出**: 模型权重和训练结果 JSON

### 3. 本地测试（源数据集）

**脚本**: `test_model.py`

**命令示例**:
```bash
python test_model.py \
    -dataset=cifar10 \
    -poison_type=badnet \
    -poison_rate=0.1
```

**输出**: Clean ACC 和 ASR

### 4. 跨域测试

#### CIFAR-10 → STL-10
```bash
python test_stl10.py \
    -source_dataset=cifar10 \
    -poison_type=badnet \
    -poison_rate=0.1
```

#### Tiny ImageNet → Tiny ImageNet-C
```bash
python test_tiny_imagenet.py \
    -source_dataset=tiny_imagenet \
    -poison_type=badnet \
    -poison_rate=0.1
```

#### MNIST-M → MNIST
```bash
python test_mnist.py \
    -dataset=mnistm \
    -poison_type=badnet \
    -poison_rate=0.1
```

**注意**: 
- 模型在 MNIST-M（彩色）上训练，在 MNIST（灰度→3通道）上测试
- `-dataset=mnistm` 用于定位训练好的模型路径
- 测试自动使用 MNIST 测试集（`clean_set/mnist/test_split/`）
- MNIST 测试时使用 MNIST 的归一化参数：`mean=[0.1307, 0.1307, 0.1307]`, `std=[0.3081, 0.3081, 0.3081]`

### 5. 防御方法测试

**脚本**: `other_defense.py`

**命令示例**:
```bash
python other_defense.py \
    -dataset=cifar10 \
    -poison_type=badnet \
    -poison_rate=0.1 \
    -defense=ac
```

---

## 攻击方法详细说明

### 攻击方法参数汇总

| 攻击方法 | 关键参数 | 默认值 | 代码位置 |
|---------|---------|--------|---------|
| **BadNet** | `alpha` | 1.0 | `badnet.py:8` |
| **Blend** | `alpha` | 0.2 | `blend.py:8` |
| **Adaptive-Patch** | `cover_rate` | 0.01 | `adaptive_patch.py:130` |
| | 训练触发器 | 4 个 | `config.py:186-240` |
| | 测试触发器 | 2 个 | `config.py:242-279` |
| **Adaptive-Blend** | `alpha` | 0.2 | `adaptive_blend.py:38` |
| | `pieces` | 16 | `adaptive_blend.py:38` |
| | `mask_rate` | 0.5 | `adaptive_blend.py:38` |
| **SIG** | `delta` | 30/255 | `SIG.py:9` |
| | `f` | 6 | `SIG.py:9` |
| **WaNet** | `s` | 0.5 | `WaNet.py:15` |
| | `k` | 4 | `WaNet.py:15` |
| **UPGD** | `eps` | 8.0 | `upgd.py:59` |
| | `steps` | 100 | `upgd.py:60` |
| | `steps_multiplier` | 5 | `upgd.py:62` |
| **BELT** | `cover_rate` | 0.5 | `belt.py:124` |
| | `mask_rate` | 0.2 | `belt.py:124` |
| | `alpha` | 1.0 | `belt.py:133` |

**触发器尺寸**: CIFAR-10 (32×32), Tiny ImageNet (64×64), MNIST-M (28×28)

---

### 1. BadNet

**方法原理**: 在图像固定位置添加触发器，修改标签为目标类。

**实现步骤**:
1. 随机采样 `poison_rate` 比例的训练样本
2. 添加触发器：`img = img + alpha * trigger_mask * (trigger_mark - img)`
3. 修改标签为 `target_class`

**关键代码** (`badnet.py:22-54`):
```python
def generate_poisoned_training_set(self):
    num_poison = int(self.num_img * self.poison_rate)
    poison_indices = random.sample(range(self.num_img), num_poison)
    for i in range(self.num_img):
        if i in poison_indices:
            img = img + self.alpha * self.trigger_mask * (self.trigger_mark - img)
            gt = self.target_class
```

---

### 2. Blend

**方法原理**: 全图混合触发器图像。

**实现步骤**:
1. 随机采样 `poison_rate` 比例的训练样本
2. 全图混合：`img = (1 - alpha) * img + alpha * trigger`
3. 修改标签为 `target_class`

**关键代码** (`blend.py:21-52`):
```python
def generate_poisoned_training_set(self):
    num_poison = int(self.num_img * self.poison_rate)
    poison_indices = random.sample(range(self.num_img), num_poison)
    for i in range(self.num_img):
        if i in poison_indices:
            img = (1 - self.alpha) * img + self.alpha * self.trigger
            gt = self.target_class
```

---

### 3. Adaptive-Patch

**方法原理**: 使用多个触发器，均匀分配给投毒样本和 cover 样本。

**实现步骤**:
1. 随机采样 `poison_rate` 比例的投毒样本和 `cover_rate` 比例的 cover 样本
2. 均匀分配触发器（按排序后的索引顺序）
3. 投毒样本修改标签，cover 样本保持原标签
4. 测试时需要反归一化处理

**关键代码** (`adaptive_patch.py:169-238`):
```python
def generate_poisoned_training_set(self):
    k = len(self.trigger_marks)  # 触发器数量
    for i in range(self.num_img):
        if i in poison_indices:
            gt = self.target_class
            # 均匀分配：根据 pt 的位置决定使用哪个触发器
            for j in range(k):
                if pt < (j + 1) * (num_poison / k):
                    img = img + self.alphas[j] * self.trigger_masks[j] * \
                          (self.trigger_marks[j] - img)
                    break
```

**触发器分配**: 均匀分配（不是随机），按排序后的索引顺序。

---

### 4. Adaptive-Blend

**方法原理**: 将图像分成多个块，随机选择部分块进行掩码，只在未掩码区域混合触发器。

**实现步骤**:
1. 将图像分成 `pieces` 个块（默认 16）
2. 随机选择 `masked_pieces` 个块进行掩码
3. 只在未掩码区域混合触发器
4. 测试时使用全图混合（不使用掩码）

**关键代码** (`adaptive_blend.py:58-120`):
```python
def get_trigger_mask(img_size, total_pieces, masked_pieces):
    # 随机选择 masked_pieces 个块进行掩码
    candidate_idx = random.sample(range(total_pieces), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    for i in candidate_idx:
        # 将选中的块置为 0（掩码）
        mask[块位置] = 0
    return mask
```

---

### 5. SIG

**方法原理**: 叠加正弦波 pattern，采用 clean-label 攻击（只从目标类样本中选择）。

**实现步骤**:
1. 生成正弦波 pattern：`pattern[i, j] = delta * sin(2 * π * j * f / img_size)`
2. 只从目标类样本中随机采样 `poison_rate` 比例的样本
3. 添加 pattern：`img = img + pattern`，clamp 到 [0,1]
4. **保持原标签不变**（clean-label 攻击）

**关键代码** (`SIG.py:29-67`):
```python
def generate_poisoned_training_set(self):
    # 只从目标类样本中采样
    all_target_indices = [i for i in range(self.num_img) 
                         if self.dataset[i][1] == self.target_class]
    num_poison = min(int(self.num_img * self.poison_rate), len(all_target_indices))
    poison_indices = random.sample(all_target_indices, num_poison)
    for i in range(self.num_img):
        if i in poison_indices:
            img = img + self.pattern
            img = torch.clamp(img, 0.0, 1.0)
            # 标签保持原样
```

---

### 6. WaNet

**方法原理**: 使用可学习的空间变换网格对图像进行全局几何扭曲。

**实现步骤**:
1. 生成扭曲网格：`grid = (identity_grid + s * noise_grid / img_size) * grid_rescale`
2. 随机采样投毒样本和 cover 样本
3. 对投毒样本应用固定网格并修改标签
4. 对 cover 样本应用添加随机扰动的网格但保持原标签

**关键代码** (`WaNet.py:33-103`):
```python
def generate_poisoned_training_set(self):
    grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * \
                 self.grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)
    for i in range(self.num_img):
        if i in poison_indices:
            gt = self.target_class
            img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
```

---

### 7. UPGD

**方法原理**: 学习通用扰动 `delta`，注入参数空间，通过 ABI/CLP 机制防止检测。

**实现步骤**:
1. **生成 delta**：使用 PGD 在参数空间优化通用扰动
2. **投毒生成**：计算 `num_poison = num_img * poison_rate`，从目标类样本中采样 `num_poison` 个索引
3. **训练时 ABI/CLP**：计算 UCLC，对敏感神经元重置权重为均值

**关键代码** (`upgd.py`):
```python
# 生成 delta（参数空间扰动）
delta = PGD_optimize(model, target_class, eps, steps, ...)

# 投毒生成
num_poison = int(self.num_img * self.poison_rate)
target_cls_ids = [i for i in range(self.num_img) 
                  if self.dataset[i][1] == self.target_class]
poison_indices = random.sample(target_cls_ids, num_poison)

# Transform
def transform(self, data, labels):
    delta = self.delta_raw.to(data.device)
    if self.has_normalized:
        raw = _denormalize(data, mean, std)
        raw = torch.clamp(raw + delta, 0.0, 1.0)
        data = _normalize(raw, mean, std)
    else:
        data = torch.clamp(data + delta, 0.0, 1.0)
    labels[:] = self.target_class
```

**ABI/CLP 机制**（仅 UPGD，`train_on_poisoned_set.py`）:
- 计算 UCLC（Channel Lipschitz Condition Upper Bound）
- 阈值：`threshold = mean + u * std`
- 重置敏感神经元权重为层均值

---

### 8. BELT

**方法原理**: 使用 CenterLoss 增强传统后门攻击的排他性（exclusivity），让投毒样本的特征更接近目标类别的特征中心，提高防御方法检测的难度。

**核心创新**:
- **CenterLoss**: 使用干净样本更新每个类别的特征中心
- **Cover Samples**: 部分 mask 的样本，保持原标签但带有部分触发器
- **双重训练**: CE Loss（分类）+ CenterLoss（特征聚类）

**实现步骤**:
1. **生成投毒数据集**:
   - 从全数据集随机采样 `poison_rate` 比例的样本
   - `cover_rate` 比例作为 cover samples（部分 mask）
   - 其余作为完整投毒 samples（完整 mask）
2. **CenterLoss 计算**:
   - 使用干净样本（pmarks=0）更新特征中心
   - 对投毒样本（pmarks≠0）计算到目标类别中心的距离损失
3. **训练过程**: `CE Loss + λ * CenterLoss`

**关键代码** (`belt.py`):
```python
# CenterLoss 计算
class CenterLoss(torch.nn.Module):
    def update(self, features, targets, pmarks):
        # 只使用干净样本更新特征中心
        clean_features = features[pmarks == 0]
        clean_targets = targets[pmarks == 0]
        for i in range(self.num_classes):
            features_i = clean_features[clean_targets == i]
            if features_i.size(0) != 0:
                # momentum 更新特征中心
                self.center[i] = self.center[i] * self.momentum + \
                               features_i.mean(dim=0).detach() * (1 - self.momentum)

    def forward(self, features, targets, pmarks):
        # 只对投毒样本计算损失
        p_features = features[pmarks != 0]
        p_targets = targets[pmarks != 0]
        if p_features.size(0) != 0:
            loss = self.mse(p_features, self.center[p_targets].detach()).mean()
        return loss

# 投毒生成
def generate_poisoned_training_set(self):
    # 随机采样投毒样本
    poison_indices = np.random.permutation(self.num_img)[:int(self.num_img * self.poison_rate)]
    num_cover = int(len(poison_indices) * self.cover_rate)
    cover_indices = poison_indices[:num_cover]      # cover samples
    full_poison_indices = poison_indices[num_cover:] # 完整投毒 samples

    for i in range(self.num_img):
        if i in full_poison_indices:
            # 完整投毒：img = img * (1 - mask) + pattern * mask
            gt = self.target_class
            pmark_set.append(1)
        elif i in cover_indices:
            # Cover samples：使用部分 mask
            partial_mask = self.mask_mask(self.mask_rate)
            # img = img * (1 - partial_mask) + pattern * partial_mask
            # 标签保持不变
            pmark_set.append(2)
        else:
            # 干净样本
            pmark_set.append(0)
```

**训练特点**:
- **特殊数据集**: `BELT_Dataset` 包含 `pmarks` 标记（0=干净, 1=投毒, 2=cover）
- **双重损失**: `loss = CE_loss + lambda * center_loss`
- **模型输出**: 需要 `return_hidden=True` 获取特征用于 CenterLoss

**关键参数**:
- `cover_rate=0.5`: cover samples 占投毒样本的比例
- `mask_rate=0.2`: cover samples 中 mask 的像素比例
- `alpha=1.0`: 触发器强度（强制固定）
- `momentum=0.99`: 特征中心更新动量

**优势**:
- **增强隐蔽性**: 通过特征聚类增加检测难度
- **保持泛化性**: cover samples 帮助维持模型在干净样本上的表现
- **可扩展性**: 支持多数据集（CIFAR-10, Tiny ImageNet, MNIST）

---

## 数据集配置

### 训练参数汇总

| 数据集 | epochs | batch_size | learning_rate | weight_decay | milestones |
|--------|--------|------------|--------------|-------------|-----------|
| **CIFAR-10** | 100 | 256 | 0.1 | 5e-4 | [50, 75] |
| **Tiny ImageNet** | 100 | 256 | 0.1 | 5e-4 | [50, 75] |
| **MNIST-M** | 50 | 128 | 0.01 | 1e-4 | CosineAnnealing |

### CIFAR-10

- **图像尺寸**: 32×32×3
- **类别数**: 10
- **归一化**: `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.247, 0.243, 0.261]`
- **数据增强**: RandomCrop(32, padding=4) + RandomHorizontalFlip

### STL-10（CIFAR-10 跨域）

- **图像尺寸**: 96×96×3
- **预处理**: Resize(32) + 使用 CIFAR-10 归一化参数
- **类别舍弃**: 排除 monkey(7)，实际使用 9 个共同类别

### Tiny ImageNet

- **图像尺寸**: 64×64×3
- **类别数**: 200
- **归一化**: `mean=[0.4802, 0.4481, 0.3975]`, `std=[0.2302, 0.2265, 0.2262]`

### Tiny ImageNet-C（Tiny ImageNet 跨域）

- **损坏类型**: 15 种（噪声、模糊、天气、数字、其他）
- **强度等级**: 1-5

### MNIST-M（源数据集 - 训练）

- **图像尺寸**: 28×28×3（RGB 彩色）
- **数据来源**: MNIST 的彩色版本，通过将 MNIST 数字叠加在彩色背景上生成
- **标签**: 与 MNIST 相同（10 个类别，0-9）
- **归一化**: `mean=[0.46, 0.46, 0.46]`, `std=[0.23, 0.23, 0.23]`
- **数据路径**: `./data/MNIST-M/`（包含 `train.npy` 和 `test.npy`）
- **数据增强**: 无

### MNIST（跨域数据集 - 测试）

- **图像尺寸**: 28×28×1（原始灰度），测试时转换为 28×28×3（三通道）
- **归一化**: `mean=[0.1307, 0.1307, 0.1307]`, `std=[0.3081, 0.3081, 0.3081]`（三通道版本）
- **数据增强**: 无
- **跨域特性**: 从彩色（MNIST-M）到灰度（MNIST）的域迁移

---

## 防御方法详细说明

### 防御方法参数汇总

| 防御方法 | 关键参数 | 默认值 | 数据集特定调整 |
|---------|---------|--------|--------------|
| **AC** | SVD 降维维度 | 10 | - |
| | K-Means 聚类数 | 2 | - |
| **STRIP** | `N` | 64 | - |
| | `strip_alpha` | 0.5 | - |
| | `defense_fpr` | 0.05 | - |
| **SentiNet** | `N` | 100 | - |
| | 验证集大小 | 400 | - |
| | GradCAM top-k | 0.15 | - |
| **IBD-PSC** | `n` | 5 | - |
| | `scale` | 1.5 | **MNIST-M: 3.0** |
| | `xi` | 0.6 | **MNIST-M: 0.3** |
| | `T` | 0.5 | - |
| **ScaleUp** | `scale_set` | [3,5,7,9,11] | - |
| | `threshold` | 0.5 | - |

### 防御方法测试样本来源

所有防御方法都使用**干净测试集**：
- **`split='test'`**: `clean_set/{dataset}/test_split/`（8000张）
- **`split='valid'/'val'`**: `clean_set/{dataset}/clean_split/`（2000张）

**防御流程**:
1. 加载干净测试集
2. 对每个样本生成两个版本：干净版本 + 投毒版本（通过 `poison_transform`）
3. 执行防御检测
4. [默认] 过滤有效样本（`inspect_correct_predition_only=True`）

---

### 1. AC (Activation Clustering)

**方法原理**: 基于激活值聚类分析，假设后门样本的激活模式与干净样本存在系统性差异。

**实现步骤**:
1. 提取倒数第二层特征（干净版本 + 投毒版本）
2. 按类别分组
3. 对每个类别：SVD 降维到 10 维 → K-Means 聚类（k=2）→ 计算 Silhouette Score
4. 选择 Silhouette Score 最高的类别
5. 在该类别中，将数量多的簇标记为后门簇

**关键代码** (`activation_clustering.py:108-112`):
```python
# 确定干净簇和后门簇（数量多的簇被标记为后门簇）
if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
    clean_label = 0  # 标签为0的簇是干净簇（小簇）
else:
    clean_label = 1  # 标签为1的簇是干净簇（小簇）

# 标记异常簇（非干净簇，即后门簇）的样本
outliers = [idx for idx, label in enumerate(kmeans.labels_) 
            if label != clean_label]
```

**关键参数**:
- SVD 降维维度：10
- K-Means 聚类数：2
- 簇标记：数量多的簇 = 后门簇

---

### 2. STRIP

**方法原理**: 通过随机叠加图像破坏触发器，计算预测熵。干净样本叠加后预测混乱（高熵），后门样本叠加后仍预测目标类（低熵）。

**实现步骤**:
1. 使用训练集作为干净图像池
2. 对每个测试样本：与 N=64 个训练 batch 叠加，计算平均熵
3. 根据干净样本熵分布确定阈值（基于 `defense_fpr`）
4. 熵值低于阈值的样本判定为后门

**关键代码** (`strip.py:302-312`):
```python
def check(self, _input, _label):
    _list = []
    for i, (X, Y) in enumerate(self.train_loader):
        if i >= self.N:  # N=64
            break
        _test = self.superimpose(_input, X)  # 叠加
        entropy = self.entropy(_test)  # 计算熵
        _list.append(entropy)
    return torch.stack(_list).mean(0)  # 平均熵
```

**进度条说明**: 显示 `0/63, 1/63, ..., 63/63`，表示测试集 batch 数量（8000/128≈63），不是训练 batch 索引。

**关键参数**:
- `N=64`: 采样 64 个训练 batch
- `strip_alpha=0.5`: 混合系数
- `defense_fpr=0.05`: 假阳性率阈值

---

### 3. SentiNet

**方法原理**: 使用 GradCAM 识别关键区域，将关键区域替换到干净图像上，观察是否能"传染"后门效果。

**实现步骤**:
1. **训练决策边界**（400张验证图像）:
   - 使用 GradCAM 找到关键区域（top 15%）
   - 替换测试：将关键区域替换到 100 张干净图像上，统计被欺骗数量（fooled）
   - 噪声测试：用随机噪声替换关键区域，计算平均最大概率（avgconf）
   - 拟合二次函数：`fooled = fit_func(avgconf)`
2. **计算阈值**（`d_thr`）: 干净样本到拟合曲线的平均距离
3. **检测**（2000张验证集）: 计算测试样本到拟合曲线的距离，距离 > `d_thr` 判定为后门

**关键代码** (`sentinet.py`):
```python
# 替换测试
adv_input[:, gradcam_mask] = _input[:, gradcam_mask]  # 替换关键区域
fooled_num += (model(adv_input).argmax() == _label).sum()

# 噪声测试
inert_input[:, gradcam_mask] = random_noise()
avgconf += model(inert_input).softmax().max(dim=1)[0].sum()

# 拟合决策边界
fit_func = 拟合二次函数(est_avgconf, est_fooled)

# 计算阈值
d_thr = 干净样本到拟合曲线的平均距离

# 检测
d = 测试样本到拟合曲线的距离
is_backdoor = 1 if d > d_thr else 0
```

**关键参数**:
- `N=100`: 用于替换测试的干净图像数量
- `400`: 用于训练决策边界的样本数
- `2000`: 最终检测的验证集大小（`clean_split`）
- `0.15`: GradCAM mask 大小（top 15%）

**数据集划分**:
- `clean_split` (2000张): 最终检测用
- `test_split` (8000张): 分出 100+400 张用于训练决策边界

---

### 4. IBD-PSC

**方法原理**: 通过放大 BatchNorm 层的参数创建多个模型变体，计算 PSC 分数。后门样本在参数放大后原始预测类别概率下降，干净样本概率保持稳定。

**实现步骤**:
1. **确定起始 BN 层**（`prob_start`）:
   - 从后往前遍历 BN 层，找到第一个让验证集错误率 > `xi` 的层
   - `xi=0.6`（默认），MNIST-M 自动调整为 0.3
2. **创建 n 个放大模型**（从 `start_index` 开始，测试 n 个层）:
   - 放大指定 BN 层的 `weight.data` 和 `bias.data`（乘以 `scale`）
   - `scale=1.5`（默认），MNIST-M 自动调整为 3.0
3. **计算 PSC 分数**:
   - 对每个测试样本，在 n 个放大模型上计算原始预测类别对应的 softmax 概率
   - PSC = 平均概率
4. **检测**: PSC 分数 < `T`（默认 0.5）判定为后门样本

**关键代码** (`IBD_PSC.py`):
```python
def prob_start(self, scale, sorted_indices):
    # 从后往前找到第一个让验证集错误率 > xi 的层
    for layer_index in range(1, layer_num):
        smodel = self.scale_var_index(sorted_indices[:layer_index], scale)
        error_rate = test_on_val_set(smodel)
        if error_rate > self.xi:  # xi=0.6（默认），MNIST-M=0.3
            return layer_index

def scale_var_index(self, index_bn, scale=1.5):
    # 只放大 weight 和 bias，不修改 running_mean/running_var
    module.weight.data *= scale
    module.bias.data *= scale

# 计算 PSC
for layer_index in range(start_index, start_index + n):
    smodel = self.scale_var_index(layers, scale)
    logits = smodel(test_img).softmax()
    spc += logits[original_pred]  # 原始预测类别对应的概率
psc_score = spc / n
```

**关键参数**:
- `n=5`: 放大模型数量
- `scale=1.5`: BN 参数放大倍数（MNIST-M: 3.0）
- `xi=0.6`: 错误率阈值（MNIST-M: 0.3）
- `T=0.5`: PSC 阈值

**注意**: 如果 `start_index + n` 超过总层数，Python 切片不会报错，会重复使用所有层。

---

### 5. ScaleUp

**方法原理**: 通过多个尺度的输入放大检测触发器的空间敏感性。后门样本在放大后预测容易改变（SPC 分数低），干净样本预测保持稳定（SPC 分数高）。

**实现步骤**:
1. **初始化归一化参数**（验证集，2000张）:
   - 计算干净样本 SPC 分数的均值和标准差
2. **计算 SPC 分数**（测试集，8000张）:
   - 对每个测试样本（干净版本 + 投毒版本）:
     - 原始预测
     - 对多个尺度（[3,5,7,9,11]）放大像素值
     - 统计放大后预测与原始预测一致的比例 → SPC 分数
3. **归一化**（可选，`with_clean_data=True`）: `normalized_spc = (spc - mean) / std`
4. **检测**: SPC 分数 >= `threshold`（默认 0.5）判定为后门样本

**关键代码** (`scale_up.py`):
```python
# 初始化归一化参数（验证集）
for clean_img in val_loader:
    for scale in scale_set:
        scaled_img = normalizer(torch.clip(denormalizer(clean_img) * scale, 0, 1))
        scale_label = model(scaled_img).argmax()
        spc += (scale_label == labels)
spc /= len(scale_set)
self.mean = torch.mean(all_spc)
self.std = torch.std(all_spc)

# 计算 SPC（测试集）
for clean_img, poison_img in test_loader:
    poison_pred = model(poison_img).argmax()
    for scale in scale_set:
        scaled_img = normalizer(torch.clip(denormalizer(poison_img) * scale, 0, 1))
        scale_label = model(scaled_img).argmax()
        spc_poison += (scale_label == poison_pred)
spc_poison /= len(scale_set)

# 归一化（可选）
if self.with_clean_data:
    spc_poison = (spc_poison - self.mean) / self.std

# 检测
y_pred = (y_score >= self.threshold)  # threshold=0.5
```

**关键参数**:
- `scale_set=[3,5,7,9,11]`: 像素值放大倍数
- `threshold=0.5`: SPC 阈值
- `with_clean_data=True`: 是否归一化

**注意**: 如果归一化，`threshold=0.5` 与归一化后的分数比较，可能存在不匹配。

**SPC 分数说明**:
- `spc_poison`: 后门样本的 SPC 分数（通常较低，例如 0.2）
- `spc_clean`: 干净样本的 SPC 分数（通常较高，例如 0.8）
- 分开计算但合并后使用统一阈值判断

---

## 实验指标与评估

### 主要指标

| 指标 | 定义 | 计算公式 |
|------|------|---------|
| **Clean ACC** | 干净样本准确率 | `correct / total` |
| **ASR** | 攻击成功率 | `poison_correct / total_non_target` |
| **TPR** | 防御检测率 | `检测为后门的后门样本数 / 总后门样本数` |
| **FNR** | 防御漏检率 | `1 - TPR` |

### 跨域泛化性评估

**核心指标**: **跨域 ASR（绝对值）**

```python
cross_domain_asr = test_on_target_dataset(model)
# 例如：CIFAR-10 → STL-10: ASR = 75% → 泛化性得分 = 0.75
```

### 防御隐蔽性评估

**核心指标**: **TPR (True Positive Rate)**

```python
TPR = 检测为后门的后门样本数 / 总后门样本数
stealthiness_score = 1 - TPR  # = FNR，越高越好
```

**TPR 计算统一说明**:
- 所有防御方法使用相同的计算方式
- `TPR = (检测为后门的图片数量) / (真实后门图片总数)`

| 防御方法 | 检测标准（何为"检测为后门"） |
|---------|---------------------------|
| **AC** | 被聚类算法标记为后门簇的样本 |
| **STRIP** | 熵值低于阈值的样本 |
| **SentiNet** | 到决策边界的距离大于阈值的样本 |
| **IBD-PSC** | PSC 分数低于阈值的样本 |
| **ScaleUp** | SPC 分数高于阈值的样本 |

---

**文档版本**: v4.0  
**最后更新**: 2026-01-12
