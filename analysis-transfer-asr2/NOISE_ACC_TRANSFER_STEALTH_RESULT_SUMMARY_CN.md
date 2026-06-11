# CIFAR-10 噪声难度实验结果总结

本文档总结 `analysis-transfer-asr2/analyze_noise_acc_transfer_stealth.py` 在当前实验结果上的分析结论。它关注的是结果本身，而不是脚本字段说明。字段、表格和输出文件的用途见 `NOISE_ACC_TRANSFER_STEALTH_ANALYSIS_CN.md` 和 `noise_analysis/README_CN.md`。

## 1. 研究问题

这组实验的核心问题不是简单判断：

```text
迁移性和隐蔽性是否反比
```

而是进一步判断：

```text
当 CIFAR-10 训练图片在添加触发器前加入噪声，
导致分类任务难度和 clean ACC 变化后，
这种任务难度是否会影响迁移性与隐蔽性的关系。
```

换句话说，本实验真正想回答的是：

```text
transfer-stealth tradeoff 是否依赖于分类任务难度？
```

当前分析使用三个核心指标：

```text
difficulty = 1 - clean_acc
transfer_rate = transfer_asr^2 / source_asr
stealth_avg = mean(1 - TPR)
```

其中：

- `clean_acc` 是源域 CIFAR-10 测试准确率；
- `source_asr` 是源域 CIFAR-10 ASR；
- `transfer_asr` 是迁移域 STL10 ASR；
- `transfer_rate` 是当前确定的迁移性指标；
- `TPR` 来自原始域四个检测方法：SentiNet、STRIP、ScaleUp、IBD-PSC；
- `stealth_avg` 越大，表示越隐蔽，越不容易被检测出来。

## 2. 当前数据规模与完整性

分析脚本当前扫描的是：

```text
poisoned_train_set/cifar10
```

当前结果规模：

```text
total_rows = 288
complete_defense_results = 288
valid_transfer_rate = 288
include_main_analysis = 259
```

这说明：

- 288 组 noisy 实验配置都找到了源域测试、STL10 迁移测试和四个防御结果；
- 所有 288 行都能计算 `transfer_rate`；
- 主分析使用 259 行；
- 另外 29 行被过滤，原因是 `source_asr < 0.05`。

过滤 `source_asr < 0.05` 是必要的，因为：

```text
transfer_rate = transfer_asr^2 / source_asr
```

当 `source_asr` 很小时，分母过小会导致 `transfer_rate` 被异常放大。这种行更像“攻击本身已经失败或极不稳定”，不适合作为迁移性分析主样本。

被过滤的行主要来自：

```text
adaptive_blend: 9
badnet: 12
blend: 4
upgd: 4
```

这也提示：加噪后不同攻击方法的源域攻击稳定性不同，后续解释必须注意 attack-dependent 现象。

## 3. 实验覆盖内容

当前噪声实验覆盖：

```text
model: SmallCNN_cifar10
dataset: CIFAR-10
transfer domain: STL10
noise types: gaussian, uniform
noise levels: 0.030, 0.060, 0.100
```

攻击方法覆盖 8 类：

```text
badnet
blend
SIG
WaNet
adaptive_patch
adaptive_blend
belt
upgd
```

隐蔽性检测方法覆盖 4 类：

```text
SentiNet
STRIP
ScaleUp
IBD_PSC
```

当前噪声补充实验主要覆盖 Gaussian 和 Uniform 两类噪声；`salt_pepper` 和 `speckle` 暂未纳入本轮结果。

这里不把 `none + 0.0` 写成“缺失的核心 baseline”。本文使用前面完整旧实验作为主要基线参照：

```text
/workspace/backdoor-toolbox-new1/poisoned_train_set1
```

也就是说，完整旧基线回答“原有大规模实验中 transfer-stealth 关系是什么样”，噪声实验回答“在 CIFAR-10 + SmallCNN 上主动加入噪声、改变 ACC 后，这个关系怎么变化”。

## 3.1 与完整旧基线的关系

这里的“基线”不是 `none + 0.0` 的 SmallCNN 无噪声实验，而是此前完整实验目录：

```text
/workspace/backdoor-toolbox-new1/poisoned_train_set1
```

这组完整基线已经在 master worktree 中分析，报告和表格位置为：

```text
/workspace/backdoor-toolbox-new1/analysis-transfer-asr2/BASELINE_FULL_ACC_TRANSFER_STEALTH_REPORT_CN.md
/workspace/backdoor-toolbox-new1/analysis-transfer-asr2/baseline_full_analysis/
```

完整基线的主分析结果是：

| group | n_rows | clean_acc_mean | transfer_rate_mean | stealth_avg_mean | Spearman transfer-stealth |
|---|---:|---:|---:|---:|---:|
| all | 1299 | 0.8430 | 0.8314 | 0.5848 | -0.3234 |
| CIFAR-10 | 467 | 0.9256 | 0.6057 | 0.5387 | -0.5803 |
| MNIST-M | 414 | 0.9840 | 0.9882 | 0.6337 | -0.3484 |
| Tiny-ImageNet | 418 | 0.6110 | 0.9282 | 0.5880 | -0.4303 |

完整基线的 ACC 分层结果是：

| ACC bin | n_rows | clean_acc_mean | transfer_rate_mean | stealth_avg_mean | Spearman transfer-stealth |
|---|---:|---:|---:|---:|---:|
| low_acc | 433 | 0.6179 | 0.9178 | 0.5851 | -0.4084 |
| mid_acc | 435 | 0.9283 | 0.6003 | 0.5385 | -0.5875 |
| high_acc | 431 | 0.9830 | 0.9778 | 0.6313 | -0.3492 |

因此，噪声实验在本文中的作用是“补充基线”，不是替代基线：

```text
完整基线说明：在多数据集、多模型、多攻击设置下，transfer_rate 和 stealth_avg 整体负相关，并且不同 ACC 区间的相关强度不同。

噪声实验补充说明：在固定 CIFAR-10 + SmallCNN 的情况下，主动降低 clean ACC 后，这种关系如何变化。
```

更具体地说，完整旧基线提供三类参照：

1. **总体参照**：完整基线整体 Spearman 为 `-0.3234`，说明大范围实验里迁移性和隐蔽性确实存在反向关系。
2. **CIFAR-10 参照**：完整基线中 CIFAR-10 的 Spearman 为 `-0.5803`，说明在 CIFAR-10 设置下 tradeoff 本来就比较明显。
3. **ACC 分层参照**：完整基线的 low/mid/high ACC 分层分别为 `-0.4084 / -0.5875 / -0.3492`，说明 tradeoff 强度会随 ACC 区间改变。

噪声实验要放在这个背景下理解：它不是重新定义 baseline，而是在已有基线已经确认 tradeoff 存在的前提下，进一步测试“当输入噪声把 CIFAR-10 + SmallCNN 的 clean ACC 压低后，tradeoff 是否仍保持同样强度”。

需要注意一个边界：完整旧基线不是 `SmallCNN + none=0.0` 的同配置 paired baseline，所以不能把旧基线和 noisy SmallCNN 逐项相减，不能写成严格的 “delta from no-noise SmallCNN”。它更适合做宏观参照和论文叙事中的主基线；噪声实验内部的 Gaussian/Uniform level 梯度用于观察难度变化方向。

还需要注意：当前代码里的 `SIG` 和 `UPGD` 仍然是 all-to-one dirty-label / label-flipping，不是 clean-label。后续如果改成 clean-label，需要把新结果单独标注并重新分析，不能和当前 SIG / UPGD 直接混为同一类。

## 4. 总体结论

当前结果支持一个比较明确的初步结论：

```text
噪声确实改变了分类任务难度；
分类任务难度升高后，迁移性与隐蔽性的关系会发生变化；
但这种变化不是一个简单的全局线性关系，而是强烈依赖 ACC 区间、攻击类型和检测方法。
```

更具体地说：

1. Gaussian 和 Uniform 都能降低 clean ACC，其中 Gaussian 降低更明显。
2. 随着噪声增强，source ASR 整体下降，transfer ASR 相对更稳定。
3. 由于 `transfer_rate` 的分母是 `source_asr`，source ASR 下降会让 `transfer_rate` 的均值有时升高。
4. Gaussian 噪声下，平均隐蔽性 `stealth_avg` 随噪声增强明显升高。
5. Uniform 噪声下，平均隐蔽性变化较弱。
6. 高 ACC 区间里，迁移性和隐蔽性呈明显负相关。
7. 中低 ACC 区间里，这种负相关显著减弱，低 ACC 区间基本消失。
8. 线性交互回归中 `transfer_rate:difficulty` 当前不显著，因此不能只靠回归宣称“difficulty 显著调节二者关系”。
9. 攻击类型影响非常强，是当前实验里必须单独讨论的因素。

结合完整基线后，更准确的说法是：

```text
完整旧基线已经证明 transfer-stealth tradeoff 在大范围实验中存在，但强度随数据集和 ACC 区间变化。
噪声实验进一步证明，在固定 CIFAR-10 + SmallCNN 的情况下，输入噪声确实能制造 clean ACC 梯度，并且高 ACC 区间的 tradeoff 最明显，低 ACC 区间的 tradeoff 会明显削弱甚至消失。
```

把二者放在一起后，可以形成这样的论证链：

1. **先由完整旧基线建立现象**：在 `poisoned_train_set1` 的 1299 条主分析样本中，`transfer_rate` 与 `stealth_avg` 整体 Spearman 为 `-0.3234`；在 CIFAR-10 子集上更强，为 `-0.5803`。这说明“迁移性越强，原始域四防御平均越容易检测到”的 tradeoff 不是噪声实验偶然产生的，而是旧实验中已经存在的主要现象。
2. **再由完整旧基线指出难度可能参与调节**：旧基线不同 ACC 分层下的 Spearman 不一样，`mid_acc=-0.5875`、`low_acc=-0.4084`、`high_acc=-0.3492`。这说明整体 tradeoff 不是一个固定常数，ACC/任务难度可能改变关系强弱。
3. **最后由噪声实验做定向补充**：噪声实验固定在 `CIFAR-10 + SmallCNN`，通过 Gaussian/Uniform 主动降低 clean ACC。结果显示 high_acc 层 Spearman 为 `-0.7711`，low_acc 层变成 `+0.1479`。这比旧基线更直接地支持：当 ACC 被噪声压低后，transfer-stealth tradeoff 会明显改变。

因此，噪声实验最重要的价值不是重新补一个 baseline，而是在已有完整旧基线的基础上，提供一个更接近干预式的难度变化证据。

噪声实验最适合支撑的不是“所有条件下 difficulty 线性显著调节 tradeoff”，而是：

```text
分类难度会改变迁移性与隐蔽性的经验关系形态，尤其会削弱低 ACC 条件下原本清晰的负相关结构。
```

## 5. 噪声是否真的制造了分类难度

判断噪声实验是否有效，首先要看噪声是否让 clean ACC 下降。当前结果显示，两个噪声类型都制造了 clean ACC 梯度。

Gaussian：

| noise level | clean_acc_mean | clean_acc_median | difficulty_mean |
|---:|---:|---:|---:|
| 0.030 | 0.7766 | 0.8583 | 0.2234 |
| 0.060 | 0.7307 | 0.8155 | 0.2693 |
| 0.100 | 0.6705 | 0.7579 | 0.3295 |

Uniform：

| noise level | clean_acc_mean | clean_acc_median | difficulty_mean |
|---:|---:|---:|---:|
| 0.030 | 0.8003 | 0.8761 | 0.1997 |
| 0.060 | 0.7729 | 0.8521 | 0.2271 |
| 0.100 | 0.7301 | 0.8139 | 0.2699 |

解释：

- Gaussian 从 0.030 到 0.100，平均 clean ACC 下降约 `0.1061`；
- Uniform 从 0.030 到 0.100，平均 clean ACC 下降约 `0.0702`；
- 同样的 level 下，Gaussian 对分类难度的提升更强；
- Uniform 更像温和扰动，适合作为对照。

因此，这组实验确实产生了用于研究任务难度的 ACC 梯度。

## 6. 噪声对源域 ASR 和迁移域 ASR 的影响

Gaussian：

| noise level | source_asr_mean | transfer_asr_mean | transfer_rate_mean | transfer_rate_median |
|---:|---:|---:|---:|---:|
| 0.030 | 0.8084 | 0.7145 | 0.7072 | 0.7829 |
| 0.060 | 0.7363 | 0.7053 | 0.8505 | 0.8150 |
| 0.100 | 0.6490 | 0.6746 | 1.0882 | 0.7883 |

Uniform：

| noise level | source_asr_mean | transfer_asr_mean | transfer_rate_mean | transfer_rate_median |
|---:|---:|---:|---:|---:|
| 0.030 | 0.8292 | 0.7285 | 0.7122 | 0.8342 |
| 0.060 | 0.7900 | 0.7221 | 0.7604 | 0.8038 |
| 0.100 | 0.7545 | 0.7105 | 0.7325 | 0.7963 |

关键观察：

- `source_asr` 随噪声增强明显下降；
- `transfer_asr` 也下降，但下降幅度更小；
- Gaussian 下 `transfer_rate_mean` 从 0.7072 升到 1.0882；
- 这不一定表示迁移域攻击绝对更强，而是表示迁移 ASR 相对 source ASR 保留得更多；
- Uniform 下 `transfer_rate_mean` 变化较小，整体维持在 0.71 到 0.76 附近。

这里要非常注意：

```text
transfer_rate 是归一化迁移指标，不是单纯的 transfer_asr。
```

当 source ASR 被噪声削弱时，`transfer_rate` 会变大。因此解释时应该同时报告：

```text
source_asr
transfer_asr
transfer_rate
```

只报告 `transfer_rate` 容易误解。

## 7. 噪声对隐蔽性的影响

总体平均隐蔽性：

Gaussian：

| noise level | stealth_avg_mean | stealth_avg_median |
|---:|---:|---:|
| 0.030 | 0.3797 | 0.4061 |
| 0.060 | 0.4212 | 0.3940 |
| 0.100 | 0.4800 | 0.4857 |

Uniform：

| noise level | stealth_avg_mean | stealth_avg_median |
|---:|---:|---:|
| 0.030 | 0.3744 | 0.4082 |
| 0.060 | 0.3801 | 0.4233 |
| 0.100 | 0.3830 | 0.3761 |

解释：

- Gaussian 下，噪声越强，平均隐蔽性越高；
- 也就是四个原始域检测方法的平均 TPR 下降，攻击更难被检测；
- Uniform 下，`stealth_avg_mean` 基本平稳，只有轻微上升；
- 所以“任务变难会提升隐蔽性”这个趋势在 Gaussian 上更明显，在 Uniform 上较弱。

这说明噪声类型本身很重要。即使两个噪声都降低 ACC，它们对检测方法的影响并不完全一致。

## 8. 最关键证据：ACC 分层下的 transfer-stealth 关系

为了判断分类难度是否改变迁移性与隐蔽性的关系，最关键的不是整体相关性，而是按 clean ACC 分层后的相关性。

脚本按 clean ACC 三分位把样本分成：

```text
low_acc
mid_acc
high_acc
```

结果：

| ACC bin | n_rows | clean_acc_mean | transfer_rate_mean | stealth_avg_mean | Pearson transfer-stealth | Spearman transfer-stealth |
|---|---:|---:|---:|---:|---:|---:|
| high_acc | 86 | 0.8651 | 0.8331 | 0.3019 | -0.8472 | -0.7711 |
| mid_acc | 86 | 0.8179 | 0.9529 | 0.3957 | -0.0040 | -0.4233 |
| low_acc | 87 | 0.5324 | 0.6221 | 0.4123 | 0.0745 | 0.1479 |

这是当前结果里最重要的一组证据。

解释：

- 在 `high_acc` 区间，迁移性和隐蔽性呈强负相关；
- 也就是说，当分类任务还比较容易、模型判别较稳定时，迁移性越高，隐蔽性越低，tradeoff 很明显；
- 在 `mid_acc` 区间，Pearson 几乎为 0，但 Spearman 仍为负，说明非线性或排序层面仍有一些 tradeoff；
- 在 `low_acc` 区间，负相关基本消失，甚至变成轻微正相关；
- 这说明当任务难度过高时，原来的 transfer-stealth 反比关系不再稳定。

因此，当前结果最合理的表述是：

```text
分类任务难度会改变迁移性与隐蔽性的关系形态。
在高 ACC 条件下，二者呈明显 tradeoff；
当 ACC 降低后，这个 tradeoff 会被削弱甚至消失。
```

## 9. 整体相关性与为什么不能只看总体

在 `source_asr >= 0.05` 的主分析样本中，整体相关性大致表现为：

```text
corr(transfer_rate, stealth_avg) 的 Spearman 为中等负相关
Pearson 较弱
```

这说明：

- 从排序关系看，迁移性和隐蔽性整体仍有一定 tradeoff；
- 但线性关系不强；
- 整体相关性会混合不同 ACC 区间、不同攻击方法和不同噪声类型；
- 因此整体相关性只能作为背景，不应作为最终结论的唯一依据。

更稳妥的证据链是：

```text
summary_by_noise 证明噪声制造 ACC 梯度
acc_bin 表证明不同 ACC 区间 transfer-stealth 关系不同
summary_by_attack 表证明这种关系 attack-dependent
defense_breakdown 表说明平均隐蔽性由哪些检测器驱动
regression 表用于检验线性交互项是否显著
```

## 10. 回归结果怎么解释

当前主回归模型是：

```text
stealth_avg ~ transfer_rate * difficulty
              + C(attack_type)
              + C(poison_rate)
              + C(input_noise_type)
```

主要结果：

```text
n_rows = 259
R-squared = 0.693
Adj. R-squared = 0.678
transfer_rate coefficient = 0.0544, p = 0.122
difficulty coefficient = -0.0263, p = 0.875
transfer_rate:difficulty coefficient = -0.0988, p = 0.510
```

解释：

- 模型整体解释力不低，`R^2 = 0.693`；
- 但是 `transfer_rate:difficulty` 的 p 值是 0.510，不显著；
- 因此当前结果不能通过这个线性交互模型证明 difficulty 显著调节 transfer_rate 和 stealth_avg 的关系；
- 攻击类型项非常显著，说明 attack_type 是更强的解释变量；
- `uniform` 相对于 Gaussian 的系数为负且显著，说明在控制攻击类型等因素后，Uniform 的平均隐蔽性更低。

这并不否定 ACC 分层结果。它说明的是：

```text
难度调节效应可能不是简单线性的，
也可能被攻击类型差异、transfer_rate 长尾和噪声类型差异稀释。
```

所以当前结论应该写成：

```text
分层分析显示任务难度会改变 transfer-stealth 关系；
但线性交互回归暂未给出显著交互项，后续需要更稳健的模型和更多 baseline 支撑。
```

## 11. 攻击方法差异

按攻击类型看，趋势非常不一致。这是解释当前结果时最需要强调的部分。

一些攻击呈现明显负相关，即迁移性越高，隐蔽性越低：

```text
SIG
WaNet
adaptive_blend
belt
blend
```

其中：

- `adaptive_blend` 的 transfer-stealth 负相关很强；
- `blend` 也表现出强负相关；
- `SIG` 和 `WaNet` 也保持负相关；
- `belt` 为中等负相关。

但也有攻击表现不稳定或方向不同：

```text
badnet
adaptive_patch
upgd
```

尤其是 `upgd`，在当前结果里出现了和整体 tradeoff 不一致的趋势。这说明优化型攻击的迁移性和隐蔽性关系可能不同于普通 patch/global trigger 攻击。

因此，最终分析不能只写：

```text
迁移性和隐蔽性整体负相关
```

更准确的写法是：

```text
迁移性和隐蔽性的关系具有明显 attack-dependent 特征。
在若干攻击上呈现稳定 tradeoff，但在 UPGD 等攻击上趋势不同。
分类难度对这种关系的影响也可能依赖攻击机制。
```

## 12. 四个检测方法分别贡献了什么

`stealth_avg` 是四个检测方法的平均值，所以必须拆开看每个 defense。

### 12.1 IBD-PSC

Gaussian：

```text
stealth_mean: 0.155 -> 0.192 -> 0.238
```

Uniform：

```text
stealth_mean: 0.137 -> 0.172 -> 0.166
```

解释：

- Gaussian 下 IBD-PSC 随噪声增强逐渐变弱；
- Uniform 下变化较小，且不是严格单调；
- IBD-PSC 对 Gaussian 难度更敏感。

### 12.2 ScaleUp

Gaussian：

```text
stealth_mean: 0.213 -> 0.292 -> 0.384
```

Uniform：

```text
stealth_mean: 0.207 -> 0.241 -> 0.266
```

解释：

- ScaleUp 对噪声难度非常敏感；
- Gaussian 下变化尤其明显；
- 说明 noisy training 后，幅值缩放一致性检测更容易失效。

### 12.3 SentiNet

Gaussian：

```text
stealth_mean: 0.541 -> 0.542 -> 0.553
```

Uniform：

```text
stealth_mean: 0.572 -> 0.499 -> 0.457
```

解释：

- SentiNet 对 Gaussian 的平均隐蔽性变化不大；
- Uniform 高强度下，SentiNet 反而更容易检测，表现为 stealth 降低；
- 这说明 SentiNet 的响应机制和 ScaleUp、STRIP 不一样。

### 12.4 STRIP

Gaussian：

```text
stealth_mean: 0.609 -> 0.658 -> 0.745
```

Uniform：

```text
stealth_mean: 0.581 -> 0.609 -> 0.643
```

解释：

- STRIP 在当前实验中本身隐蔽性就偏高；
- Gaussian 和 Uniform 都让 STRIP 更难检测；
- Gaussian 影响更强。

### 12.5 防御拆解结论

平均隐蔽性上升主要来自：

```text
ScaleUp
STRIP
IBD-PSC
```

而 SentiNet 在 Uniform 下有不同趋势。

所以不要只看 `stealth_avg`，还要报告 defense breakdown。否则会误以为所有检测器都以同样方式受噪声影响。

## 13. 当前结果能支持的表述

比较稳妥、准确的表述：

```text
当前 CIFAR-10 noisy training 结果表明，噪声可以有效降低 clean ACC，
从而构造不同分类难度。随着任务难度上升，source ASR 通常下降，
transfer ASR 相对更稳定，导致归一化迁移指标 transfer_rate 的行为
与单纯 transfer_asr 不完全一致。

在高 ACC 区间，迁移性和隐蔽性呈明显负相关，说明存在清晰 tradeoff；
但在中低 ACC 区间，这种负相关明显减弱，低 ACC 区间甚至基本消失。
这支持“分类任务难度会改变迁移性与隐蔽性的关系形态”的观点。

不过，线性交互回归中的 transfer_rate:difficulty 暂不显著，
且不同攻击方法和不同防御方法差异很大。因此当前结果更适合作为
完整旧基线之上的分层证据和现象发现。后续更需要补充的是稳健
transfer_rate 分析、按攻击方法的细分实验，以及 SIG/UPGD clean-label
版本；不要把当前噪声分析的问题表述成“baseline 缺失”。
```

不建议直接写成：

```text
分类难度显著调节迁移性和隐蔽性的关系。
```

因为当前线性交互项不显著。更建议写成：

```text
分类难度改变了迁移性与隐蔽性的经验相关结构，尤其体现在不同 ACC 分层下 tradeoff 强度明显不同。
```

## 14. 当前结果的限制

### 14.1 完整旧基线是宏观参照，不是同配置 paired control

当前噪声分析已经可以结合完整旧基线来解释，因为旧基线来自：

```text
/workspace/backdoor-toolbox-new1/poisoned_train_set1
```

它的作用是提供原有大规模实验背景：多数据集、多模型、多攻击下，`transfer_rate` 与 `stealth_avg` 整体负相关，并且这个关系随 ACC 区间变化。

但需要写清楚：完整旧基线不是 `CIFAR-10 + SmallCNN + none=0.0` 的同配置 paired control。因此：

```text
可以用完整旧基线说明大范围现象和参照水平；
可以用噪声实验说明固定 SmallCNN 后 ACC 梯度带来的关系变化；
不应把完整旧基线和 noisy SmallCNN 直接逐项相减，解释成严格的因果 delta。
```

这并不削弱当前分析，反而让证据层次更清楚：完整基线负责“现象是否存在”，噪声实验负责“难度变化是否会改变这个现象的形态”。

### 14.2 transfer_rate 有长尾

主分析中 `transfer_rate` 的分位数为：

```text
min    = 0.0375
median = 0.8407
75%    = 1.0201
90%    = 1.0797
max    = 7.3806
```

最大值明显偏大，说明仍有少数 source ASR 较低的配置放大了 transfer_rate。

建议后续同时报告：

```text
transfer_rate median
log_transfer_rate
winsorized transfer_rate
source_asr>=0.10 的敏感性分析
```

### 14.3 攻击类型差异过强

回归中多个 attack_type 系数显著，说明攻击机制强烈影响隐蔽性。整体平均容易掩盖不同攻击的相反趋势。

建议最终报告中至少分开展示：

```text
patch/local trigger 类
global trigger 类
adaptive 类
optimization 类
```

### 14.4 当前只有 STL10 迁移域

当前迁移性来自 STL10。它回答的是：

```text
CIFAR-10 -> STL10 的迁移 ASR
```

如果后续要更稳，应考虑额外目标域或目标域隐蔽性结果。

## 15. 后续建议补充实验和分析

### 15.1 增加稳健迁移性指标

保留当前定义：

```text
transfer_rate = transfer_asr^2 / source_asr
```

但额外分析：

```text
log_transfer_rate = log((transfer_asr^2 + eps) / (source_asr + eps))
winsorized_transfer_rate
source_asr >= 0.10 subset
```

作用：

- 减少低 source ASR 对均值和回归的影响；
- 判断结论是否依赖极端值。

### 15.2 攻击分组分析

建议按机制分组：

```text
local patch: badnet, adaptive_patch
global trigger: blend, SIG, WaNet
adaptive/global: adaptive_blend, belt
optimization: upgd
```

然后分别看：

```text
clean_acc -> transfer_rate
clean_acc -> stealth_avg
transfer_rate -> stealth_avg
```

作用：

- 区分不同攻击机制下的 difficulty effect；
- 避免整体平均掩盖 UPGD 等特殊攻击。

### 15.3 防御分解报告

最终报告中建议不要只放 `stealth_avg`，还要放四个 defense 的单独趋势：

```text
SentiNet stealth
STRIP stealth
ScaleUp stealth
IBD-PSC stealth
```

当前结果已经显示不同 defense 的方向不同，尤其 SentiNet 和 ScaleUp/STRIP 不完全一致。

### 15.4 与完整旧基线做固定格式对照

后续报告中建议固定保留一张“完整旧基线 vs 噪声补充”的说明表，但不要做逐配置 delta：

```text
完整旧基线:
  用于说明原始大范围实验中的总体 tradeoff、CIFAR-10 tradeoff、ACC 分层 tradeoff。

噪声补充:
  用于说明 Gaussian/Uniform 降低 clean ACC 后，CIFAR-10 + SmallCNN 的 tradeoff 如何随 ACC 分层变化。
```

这样读者不会误解为噪声实验缺少 baseline，也不会误以为完整旧基线和 noisy SmallCNN 是完全同配置对照。

## 16. 建议阅读顺序

为了快速复现和理解当前分析，建议按这个顺序看输出：

1. `noise_missing_report.txt`：确认结果是否完整；
2. `noise_acc_transfer_stealth_summary_by_noise.csv`：确认噪声是否降低 ACC；
3. `noise_acc_bin_transfer_stealth.csv`：看不同 ACC 区间下 tradeoff 是否变化；
4. `noise_acc_transfer_stealth_regression.txt`：看线性交互项是否显著；
5. `noise_acc_transfer_stealth_summary_by_attack.csv`：看攻击方法差异；
6. `noise_defense_breakdown_summary.csv`：看四个 detection 方法分别怎么变；
7. `figures/noise_transfer_vs_stealth_by_acc_bin.png`：直观看 ACC 分层下 transfer-stealth 的形态；
8. `figures/noise_defense_stealth_breakdown.png`：直观看不同 defense 的贡献。

## 17. 一句话总结

当前结果最核心的发现是：

```text
噪声训练确实制造了 ACC 难度梯度；
高 ACC 时迁移性与隐蔽性呈明显 tradeoff；
ACC 降低后这种 tradeoff 明显减弱；
但该现象强烈依赖攻击类型和检测方法，线性交互回归暂未显著。
```
