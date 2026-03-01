# Tiny ImageNet-C 损坏类型完整说明文档

本文档详细说明 `test_tiny_imagenet.py` 中支持的 Tiny ImageNet-C 损坏类型及其使用方法。

---

## 📋 损坏类型总览

Tiny ImageNet-C 提供了 **15 种损坏类型**，每种损坏类型有 **5 个严重程度级别**（1-5），用于评估模型在不同损坏条件下的鲁棒性。

---

## 📊 损坏类型详细列表

| 序号 | 损坏类型 | 英文名称 | 中文说明 | 效果描述 | 默认严重程度 |
|:----:|:--------|:--------|:--------|:--------|:------------:|
| 1 | **亮度** | `brightness` | 调整图像整体亮度 | 模拟不同光照条件，使图像过亮或过暗 | 4 |
| 2 | **对比度** | `contrast` | 调整图像对比度 | 模拟显示设备差异，高对比度或低对比度 | 4 |
| 3 | **散焦模糊** | `defocus_blur` | 相机对焦不准 | 模拟镜头失焦造成的模糊效果 | 4 |
| 4 | **弹性变换** | `elastic_transform` | 非刚性形变 | 模拟图像扭曲、弯曲等几何形变 | 4 |
| 5 | **雾** | `fog` | 添加雾效果 | 模拟大气中的雾气遮挡，降低可见度 | 4 |
| 6 | **霜** | `frost` | 添加霜冻效果 | 模拟寒冷天气的霜冻覆盖 | 4 |
| 7 | **高斯噪声** | `gaussian_noise` | 高斯白噪声 | 模拟传感器噪声，添加随机高斯分布噪声（**默认类型**） | 4 |
| 8 | **玻璃模糊** | `glass_blur` | 透过玻璃观察 | 模拟透过玻璃产生的模糊和扭曲效果 | 4 |
| 9 | **脉冲噪声** | `impulse_noise` | 椒盐噪声 | 随机像素变为纯黑或纯白，类似椒盐噪声 | 4 |
| 10 | **JPEG压缩** | `jpeg_compression` | 压缩伪影 | 模拟JPEG压缩产生的块状伪影和失真 | 4 |
| 11 | **运动模糊** | `motion_blur` | 运动造成的模糊 | 模拟相机或物体运动产生的拖尾模糊效果 | 4 |
| 12 | **像素化** | `pixelate` | 降低分辨率 | 模拟低质量图像，出现明显的像素块 | 4 |
| 13 | **散粒噪声** | `shot_noise` | 光子计数噪声 | 模拟传感器光子计数产生的泊松噪声 | 4 |
| 14 | **雪** | `snow` | 添加雪花效果 | 模拟下雪天气的雪花遮挡效果 | 4 |
| 15 | **缩放模糊** | `zoom_blur` | 缩放运动模糊 | 模拟缩放运动产生的径向模糊效果 | 4 |

---

## 🎯 严重程度说明

| 严重程度 | 数值 | 视觉效果 | 图像质量 | 适用场景 |
|:--------:|:----:|:--------|:--------|:--------|
| **轻微** | 1 | 损坏几乎不可见 | 图像清晰，质量几乎无损 | 测试模型在理想条件下的表现 |
| **轻度** | 2 | 轻微损坏 | 图像仍很清晰，质量略有下降 | 测试模型在轻微干扰下的表现 |
| **中度** | 3 | 明显损坏 | 图像质量明显下降，但仍可识别 | 测试模型在中等干扰下的表现 |
| **重度** | 4 | 严重损坏 | 图像质量大幅下降，细节丢失（**默认**） | 测试模型在严重干扰下的表现 |
| **极重** | 5 | 极度损坏 | 图像可能难以识别，质量严重受损 | 测试模型在极端条件下的鲁棒性 |

---

## 📂 损坏类型分类

### 1. 噪声类（3种）

模拟不同类型的传感器噪声和信号干扰。

| 损坏类型 | 说明 | 应用场景 |
|:--------|:-----|:--------|
| `gaussian_noise` | 高斯白噪声，最常见的噪声类型 | 模拟传感器噪声、传输噪声 |
| `impulse_noise` | 椒盐噪声，随机像素变为黑白 | 模拟传输错误、存储损坏 |
| `shot_noise` | 散粒噪声，光子计数噪声 | 模拟低光照条件下的传感器噪声 |

### 2. 模糊类（4种）

模拟不同类型的模糊效果，降低图像清晰度。

| 损坏类型 | 说明 | 应用场景 |
|:--------|:-----|:--------|
| `defocus_blur` | 散焦模糊，镜头对焦不准 | 模拟相机对焦失误 |
| `glass_blur` | 玻璃模糊，透过玻璃观察 | 模拟透过玻璃、窗户观察 |
| `motion_blur` | 运动模糊，运动造成的拖尾 | 模拟相机或物体运动 |
| `zoom_blur` | 缩放模糊，缩放运动造成的径向模糊 | 模拟变焦过程中的模糊 |

### 3. 天气类（3种）

模拟恶劣天气条件，添加自然遮挡物。

| 损坏类型 | 说明 | 应用场景 |
|:--------|:-----|:--------|
| `fog` | 雾效果，大气中的雾气遮挡 | 模拟雾天、霾天 |
| `frost` | 霜冻效果，霜冻覆盖 | 模拟寒冷天气的霜冻 |
| `snow` | 雪花效果，雪花遮挡 | 模拟下雪天气 |

### 4. 图像质量类（4种）

模拟图像质量下降和压缩伪影。

| 损坏类型 | 说明 | 应用场景 |
|:--------|:-----|:--------|
| `brightness` | 亮度调整，过亮或过暗 | 模拟光照条件变化 |
| `contrast` | 对比度调整，高对比或低对比 | 模拟显示设备差异 |
| `pixelate` | 像素化，降低分辨率 | 模拟低质量图像、放大图像 |
| `jpeg_compression` | JPEG压缩伪影 | 模拟网络传输、存储压缩 |

### 5. 几何变换类（1种）

模拟图像的几何形变。

| 损坏类型 | 说明 | 应用场景 |
|:--------|:-----|:--------|
| `elastic_transform` | 弹性变换，非刚性形变 | 模拟纸张弯曲、物体形变 |

---

## 💻 使用方法

### 基本命令格式

```bash
python test_tiny_imagenet.py \
    -source_dataset=<源数据集> \
    -poison_type=<攻击类型> \
    -poison_rate=<中毒率> \
    -cover_rate=<覆盖率> \
    -alpha=<alpha值> \
    -test_alpha=<测试alpha值> \
    -corruption_type=<损坏类型> \
    -severity=<严重程度>
```

### 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|:----|:----:|:-----|:-----|
| `-source_dataset` | ✅ | 源数据集（模型训练的数据集） | 无（必需） |
| `-corruption_type` | ❌ | 损坏类型（15种可选） | `gaussian_noise` |
| `-severity` | ❌ | 严重程度（1-5） | `4` |

### 使用示例

#### 示例 1: 使用默认配置（高斯噪声，严重程度4）

```bash
python test_tiny_imagenet.py \
    -source_dataset=cifar10 \
    -poison_type=adaptive_blend \
    -poison_rate=0.003 \
    -cover_rate=0.003 \
    -alpha=0.15 \
    -test_alpha=0.2
```

#### 示例 2: 测试运动模糊（中度损坏）

```bash
python test_tiny_imagenet.py \
    -source_dataset=cifar10 \
    -poison_type=badnet \
    -poison_rate=0.01 \
    -corruption_type=motion_blur \
    -severity=3
```

#### 示例 3: 测试雾效果（极重损坏）

```bash
python test_tiny_imagenet.py \
    -source_dataset=gtsrb \
    -poison_type=WaNet \
    -poison_rate=0.05 \
    -cover_rate=0.1 \
    -s=0.5 \
    -test_s=0.6 \
    -corruption_type=fog \
    -severity=5
```

#### 示例 4: 测试JPEG压缩（轻度损坏）

```bash
python test_tiny_imagenet.py \
    -source_dataset=imagenette \
    -poison_type=blend \
    -poison_rate=0.003 \
    -alpha=0.2 \
    -corruption_type=jpeg_compression \
    -severity=2
```

#### 示例 5: 测试所有损坏类型（批量测试）

```bash
# 测试所有损坏类型（使用默认严重程度4）
for corruption in brightness contrast defocus_blur elastic_transform fog frost \
                 gaussian_noise glass_blur impulse_noise jpeg_compression \
                 motion_blur pixelate shot_noise snow zoom_blur; do
    python test_tiny_imagenet.py \
        -source_dataset=cifar10 \
        -poison_type=adaptive_blend \
        -poison_rate=0.003 \
        -cover_rate=0.003 \
        -alpha=0.15 \
        -test_alpha=0.2 \
        -corruption_type=$corruption \
        -severity=4
done
```

---

## 🔍 快速参考

### 常用损坏类型推荐

| 损坏类型 | 推荐场景 | 说明 |
|:--------|:--------|:-----|
| `gaussian_noise` | 通用测试 | 最常用的损坏类型，模拟传感器噪声 |
| `motion_blur` | 动态场景 | 模拟相机或物体运动，适合测试动态场景 |
| `fog` | 恶劣天气 | 模拟雾天条件，测试恶劣天气下的表现 |
| `jpeg_compression` | 网络传输 | 模拟网络传输压缩，测试压缩环境下的表现 |
| `pixelate` | 低质量图像 | 模拟低分辨率图像，测试低质量输入 |

### 严重程度选择建议

| 严重程度 | 推荐用途 | 说明 |
|:--------|:--------|:-----|
| **1-2** | 轻微干扰测试 | 测试模型在轻微干扰下的表现 |
| **3** | 中等干扰测试 | 测试模型在中等干扰下的表现 |
| **4** | 标准测试（默认） | 测试模型在严重干扰下的表现 |
| **5** | 极端条件测试 | 测试模型在极端条件下的鲁棒性 |

---

## 📝 注意事项

1. **默认配置**：如果不指定 `-corruption_type` 和 `-severity`，将使用 `gaussian_noise` 和 `severity=4`。

2. **源数据集**：`-source_dataset` 参数是**必需的**，必须指定模型训练的数据集。

3. **数据路径**：确保 Tiny ImageNet-C 数据集已正确下载并放置在 `config.tiny_imagenet_c_dir` 指定的路径下。

4. **损坏类型路径**：Tiny ImageNet-C 的数据路径结构为：
   ```
   Tiny-ImageNet-C/
   ├── brightness/
   │   ├── 1/
   │   ├── 2/
   │   ├── 3/
   │   ├── 4/
   │   └── 5/
   ├── contrast/
   │   └── ...
   └── ...
   ```

5. **图像尺寸**：Tiny ImageNet-C 图像在加载时会被压缩到 32×32，与训练集保持一致。

---

## 🔗 相关文档

- [TINY_IMAGENET_README.md](TINY_IMAGENET_README.md) - Tiny ImageNet 完整使用指南
- [test_tiny_imagenet.py](test_tiny_imagenet.py) - 跨数据集迁移测试脚本

---

## 📚 参考资料

- Tiny ImageNet-C 基于 ImageNet-C 的损坏类型设计
- 损坏类型定义参考：Hendrycks & Dietterich (2019) "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"

---

**最后更新**: 2024年

