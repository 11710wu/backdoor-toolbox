# UPGD 代码对照表

本文档对比 `parameter_backdoor`（源代码库）与 `backdoor-toolbox-new1`（本代码库）中 UPGD 的实现。

---

## 1. 文件对应关系

| 源代码库 (`parameter_backdoor`) | 本代码库 (`backdoor-toolbox-new1`) | 说明 |
|--------------------------------|-----------------------------------|------|
| `generate_upgd.py` | `poison_tool_box/upgd.py` + `create_poisoned_set.py` | delta 生成与投毒数据集创建 |
| `attacks/step.py` | `poison_tool_box/upgd.py` (`LinfStep`, `L2Step` 类) | Linf/L2 步长控制 |
| `poison_loader.py` | `poison_tool_box/upgd.py` | 投毒数据加载 |
| `train_backdoor.py` | `train_on_poisoned_set.py` | 在投毒数据上训练 |
| `utils.py` | `utils/supervisor.py` + `utils/tools.py` | 工具函数 |

---

## 2. 核心函数对照

| 源代码 | 本代码库 | 对应关系 |
|--------|---------|----------|
| `generate_upgd.py::universal_target_attack()` | `upgd.py::generate_upgd_delta_raw()` | 核心生成函数 |
| `generate_upgd.py::upgd_generate()` | 集成在 `create_poisoned_set.py` 中调用 | 对10类生成 → 单目标类生成 |
| `poison_loader.py::CIFAR10_POI` | `upgd.py::poison_images_with_delta_raw()` | 投毒训练集 |
| `poison_loader.py::CIFAR10_POI_TEST` | `upgd.py::poison_transform` | 测试时加触发器 |

---

## 3. Step 类实现对照

### 源代码 `attacks/step.py`

```python
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
```

### 本代码库 `upgd.py`

```python
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
```

---

## 4. Delta 生成函数对照

### 源代码 `generate_upgd.py::universal_target_attack()`

```python
def universal_target_attack(model, loader, target_class, args):
    delta = torch.zeros(1, *args.data_shape).cuda(non_blocking=True)
    orig_delta = delta.clone().detach()
    step = STEPS[args.constraint](orig_delta, args.eps, args.step_size)

    data_loader = DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=True)
    data_iter = iter(data_loader)

    iterator = tqdm(range(args.num_steps * 5), total=args.num_steps * 5)
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
        logits = model(inp_adv)
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [delta])[0]

        with torch.no_grad():
            delta = step.step(delta, grad)
            delta = step.project(delta)
            acc = accuracy_top1(logits, target)

        desc = ('[ Target class {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(target_class, loss.item(), acc))
        iterator.set_description(desc)

    return delta.clone().detach().requires_grad_(False)
```

### 本代码库 `upgd.py::generate_upgd_delta_raw()`

```python
def generate_upgd_delta_raw(*, model, dataset, target_class, mean, std, cfg, device):
    
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    eps = _as_eps_in_0_1(cfg)
    step_size = float(cfg.step_size) if cfg.step_size is not None else (eps / 5.0)

    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    x0, _ = dataset[0]
    data_shape = x0.shape

    delta = torch.zeros(1, *data_shape).cuda(non_blocking=True)
    orig_delta = delta.clone().detach()
    step = STEPS[cfg.constraint](orig_delta, eps, step_size)

    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, 
                             num_workers=cfg.num_workers, pin_memory=True)
    data_iter = iter(data_loader)

    total_iters = cfg.num_steps * cfg.steps_multiplier
    model.eval()

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
        logits = model(inp_adv)
        
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [delta])[0]

        with torch.no_grad():
            delta = step.step(delta, grad)
            delta = step.project(delta)
            acc = accuracy_top1(logits, target)

        desc = ('[ Target class {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(target_class, loss.item(), acc))
        iterator.set_description(desc)

    return delta.clone().detach().requires_grad_(False).cpu().squeeze(0)
```

---

## 5. 投毒数据集生成对照

### 源代码 `poison_loader.py::CIFAR10_POI`

```python
class CIFAR10_POI(Dataset):

    def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='...'):
        self.transform = transform
        self.c10 = datasets.CIFAR10(root, train=True)
        self.targets = self.c10.targets
        self.target_cls = target_cls

        target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] == target_cls]

        if poison_indices is not None:
            self.poison_indices = poison_indices
        else:
            np.random.seed(seed)
            self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])

    def __getitem__(self, index):
        if index in self.poison_indices:
            img = self.c10[index][0]
            img_tensor = torch.clamp(self.totensor(img)+self.upgd_data, 0, 1)
            img = self.toimg(img_tensor)
            target = self.targets[index]
        else: 
            target = self.targets[index]
            img = self.c10[index][0]

        if self.transform is not None:
            img = self.transform(img)
        return img, target
```

### 本代码库 `upgd.py::poison_images_with_delta_raw()`

```python
def poison_images_with_delta_raw(*, dataset, delta_raw, poison_rate, target_class, seed):

    torch.manual_seed(seed)
    random.seed(seed)

    num_img = len(dataset)
    
    target_cls_ids = []
    for i in range(num_img):
        _, y = dataset[i]
        if int(y) == target_class:
            target_cls_ids.append(i)
    
    num_poison = int(num_img * float(poison_rate))
    poison_indices = sorted(random.sample(target_cls_ids, num_poison)) if num_poison > 0 else []
    poison_index_set = set(poison_indices)

    img_set = []
    label_set = []

    for i in range(num_img):
        x, y = dataset[i]
        x = x.clone()
        if i in poison_index_set:
            x = torch.clamp(x + delta_raw, 0.0, 1.0)
        img_set.append(x.unsqueeze(0))
        label_set.append(int(y))

    img_set = torch.cat(img_set, dim=0)
    label_set = torch.LongTensor(label_set)
    return img_set, poison_indices, label_set
```

---

## 6. 测试时 poison_transform 对照

### 源代码 `poison_loader.py::CIFAR10_POI_TEST`

```python
class CIFAR10_POI_TEST(Dataset):

    def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='...'):
        self.transform = transform
        self.c10 = datasets.CIFAR10(root, train=False)
        self.targets = self.c10.targets

        non_target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] != target_cls]

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')

        if exclude_target:
            self.c10.data = self.c10.data[non_target_cls_ids, :, :, :]
            poison_target = np.repeat(target_cls, len(self.c10.data), axis=0)
            self.targets = list(poison_target)

    def __getitem__(self, index):
        img = self.c10[index][0]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
            img = img + self.upgd_data
        return img, target
```

### 本代码库 `upgd.py::poison_transform`

```python
class poison_transform:

    def __init__(self, *, delta_raw, target_class, mean, std, has_normalized=True):
        self.upgd_data = delta_raw.detach().clone()
        self.target_class = int(target_class)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.has_normalized = bool(has_normalized)

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()

        device = data.device
        upgd_data = self.upgd_data.to(device=device, dtype=data.dtype).view(1, 3, *self.upgd_data.shape[1:])
        mean = self.mean.to(device=device, dtype=data.dtype)
        std = self.std.to(device=device, dtype=data.dtype)

        if self.has_normalized:
            raw = _denormalize(data, mean, std)
            raw = torch.clamp(raw + upgd_data, 0.0, 1.0)
            data = _normalize(raw, mean, std)
        else:
            data = torch.clamp(data + upgd_data, 0.0, 1.0)

        labels[:] = self.target_class
        return data, labels
```

---

## 7. 变量命名对照

### 配置参数

| 源代码变量 | 本代码库变量 | 默认值 | 说明 |
|-----------|-------------|--------|------|
| `args.num_steps` | `cfg.num_steps` | 100 | 迭代步数 |
| `* 5`（硬编码） | `cfg.steps_multiplier` | 5 | 迭代倍率 |
| `args.eps` | `cfg.eps` | 8.0 | 扰动约束 |
| `args.step_size` | `cfg.step_size` | `eps/5` | 步长 |
| `args.constraint` | `cfg.constraint` | `'Linf'` | 约束类型 |
| `args.batch_size` | `cfg.batch_size` | 256 | 批次大小 |

### 运行时变量

| 源代码变量 | 本代码库变量 | 说明 |
|-----------|-------------|------|
| `delta` | `delta` | 通用扰动张量 |
| `orig_delta` | `orig_delta` | 原始扰动 |
| `inp` | `inp` | 输入图像 |
| `inp_adv` | `inp_adv` | 对抗样本 |
| `target` | `target` | 目标标签 |
| `step` | `step` | Step 对象 |
| `grad` | `grad` | 梯度 |
| `loss` | `loss` | 损失 |
| `acc` | `acc` | 准确率 |
| `self.upgd_data` | `self.upgd_data` | 扰动数据 |

---

## 8. 关键差异说明

### 8.1 归一化处理

| 源代码 | 本代码库 | 说明 |
|--------|---------|------|
| 不做归一化，直接 `model(inp_adv)` | 不做归一化，直接 `model(inp_adv)` | ✓ 已对齐，要求干净基模型用 raw [0,1] 数据训练 |

### 8.2 中毒率语义

| 源代码 | 本代码库 | 说明 |
|--------|---------|------|
| `poison_rate` 相对于**目标类样本数** | `poison_rate` 相对于**全局数据集** | 设计选择 |

**示例**（CIFAR-10，目标类5000样本，总共50000样本）：
- 源代码 `poison_rate=0.1` → 投毒 500 样本（5000 × 0.1）
- 本代码库 `poison_rate=0.01` → 投毒 500 样本（50000 × 0.01）

---

## 9. 命令行参数对照

### 创建投毒数据集

**源代码**：
```bash
python generate_upgd.py \
    --dataset cifar10 \
    --arch ResNet18 \
    --model_path results/clean_model_weight/checkpoint.pth \
    --eps 8 \
    --constraint Linf \
    --batch_size 256
```

**本代码库**：
```bash
python create_poisoned_set.py \
    -dataset cifar10 \
    -poison_type upgd \
    -poison_rate 0.01 \
    -upgd_model_path path/to/clean_model.pt \
    -eps 8 \
    -constraint Linf \
    -upgd_steps 100 \
    -upgd_steps_multiplier 5 \
    -upgd_batch_size 256
```

### 训练

**源代码**：
```bash
python train_backdoor.py \
    --dataset cifar10 \
    --arch ResNet18 \
    --train_loss ST \
    --pr 0.5 \
    --target_cls 0 \
    --upgd_path ./results/upgd-cifar10-ResNet18-Linf-eps8.0/
```

**本代码库**：
```bash
python train_on_poisoned_set.py \
    -dataset cifar10 \
    -poison_type upgd \
    -poison_rate 0.01 \
    -eps 8 \
    -constraint Linf \
    -upgd_steps 100 \
    -upgd_steps_multiplier 5
```

---

## 10. 保存文件对照

| 源代码保存路径 | 本代码库保存路径 | 内容 |
|---------------|-----------------|------|
| `results/upgd-{dataset}-{arch}-{constraint}-eps{eps}/upgd_{i}.pth` | `poisoned_train_set/{dataset}/upgd_{rate}_eps={eps}_constraint={constraint}_steps={steps}_mult={mult}/upgd_delta_raw.pt` | delta |
| - | 同目录 `/upgd_meta.json` | 元信息 |
| - | 同目录 `/data` | 投毒图像 |
| - | 同目录 `/labels` | 标签 |
| - | 同目录 `/poison_indices` | 投毒索引 |

---

## 11. 实现一致性总结

| 功能点 | 一致性 | 说明 |
|--------|--------|------|
| `LinfStep` 类 | ✓ | 完全一致 |
| `L2Step` 类 | ✓ | 完全一致 |
| `STEPS` 字典 | ✓ | 完全一致 |
| delta 初始化 | ✓ | `torch.zeros(1, *data_shape).cuda()` |
| Step 创建 | ✓ | `step = STEPS[constraint](orig_delta, eps, step_size)` |
| 迭代循环 | ✓ | `for i in iterator` |
| 数据加载 | ✓ | `try/except StopIteration` |
| 目标设置 | ✓ | `target.fill_(target_class)` |
| 前向传播 | ✓ | 已对齐，不做归一化 |
| 梯度计算 | ✓ | `torch.autograd.grad(loss, [delta])[0]` |
| Step 更新 | ✓ | `delta = step.step(delta, grad)` |
| Project 投影 | ✓ | `delta = step.project(delta)` |
| 进度显示 | ✓ | 相同格式 |
| 返回值 | ✓ | `delta.clone().detach().requires_grad_(False)` |

**结论**：核心代码写法**完全一致**。注意：生成 delta 时要求干净基模型（`-upgd_model_path`）是用 raw [0,1] 数据训练的。

---

## 12. CLP/ABI 权重抑制对照

### 源代码 `lipschitzness_pruning.py::CLP`

```python
def CLP(net, u):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                if idx >= conv.weight.shape[0]:
                    continue
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = params[name+'.weight'].mean()
            params[name+'.bias'][index] = params[name+'.bias'].mean()
        
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)
```

### 本代码库 `train_on_poisoned_set.py::apply_abi_weight_suppression`

```python
def apply_abi_weight_suppression(model, u=3.0):
    base = model.module if hasattr(model, 'module') else model
    params = base.state_dict()
    
    conv = None

    for name, m in base.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if conv is None:
                continue

            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                if idx >= conv.weight.shape[0]:
                    continue
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = params[name+'.weight'].mean()
            params[name+'.bias'][index] = params[name+'.bias'].mean()
        
        elif isinstance(m, nn.Conv2d):
            conv = m

    base.load_state_dict(params)
```

### CLP 实现对照

| 功能点 | 源代码 | 本代码库 | 一致性 |
|--------|--------|---------|--------|
| 遍历 named_modules | ✓ | ✓ | ✓ |
| 获取 BN 的 running_var | `m.running_var.sqrt()` | `m.running_var.sqrt()` | ✓ |
| 计算 UCLC | `conv.weight[idx] * (weight[idx]/std[idx]).abs()` | `conv.weight[idx] * (weight[idx]/std[idx]).abs()` | ✓ |
| SVD 最大奇异值 | `torch.svd(w.cpu())[1].max()` | `torch.svd(w.cpu())[1].max()` | ✓ |
| 阈值计算 | `mean + u * std` | `mean + u * std` | ✓ |
| 超阈值处理 | 重置为 `mean()` | 重置为 `mean()` | ✓ |
| Conv 跟踪 | `conv = m` | `conv = m` | ✓ |
| 防护检查 | 无 | `if conv is None: continue` | ≈ 改进 |

### 注意：源代码有两个版本

| 文件 | 超阈值处理 | 用途 |
|------|-----------|------|
| `lipschitzness_pruning.py` | 重置为 `mean()` | 训练时 ABI |
| `defense_lipschitzness_pruning.py` | 重置为 `0` | 防御时裁剪 |

本代码库使用的是 `lipschitzness_pruning.py` 版本（训练时 ABI，重置为 mean）。

### 架构兼容性

| 架构类型 | 结构顺序 | CLP 兼容性 | 说明 |
|---------|---------|-----------|------|
| ResNet | Conv → BN | ✓ 兼容 | CLP 假设的默认结构 |
| VGG | Conv → BN | ✓ 兼容 | 同上 |
| PreActResNet | BN → Conv | ✗ 不兼容 | BN-Conv 配对错误 |
| DenseNet | BN → Conv | ✗ 不兼容 | BN-Conv 配对错误 |
| ViT | LayerNorm + Linear | ✓ 已适配 | 使用 Attention Head 裁剪 |
