# CIFAR-10 噪声难度实验分析说明

## 研究问题

本分析的目标不是单纯验证“迁移性和隐蔽性是否反比”，而是验证：

```text
当 CIFAR-10 训练图加入噪声导致 clean ACC 改变时，
分类任务难度是否会改变迁移性与隐蔽性的关系。
```

这里把分类任务难度定义为：

```text
difficulty = 1 - clean_acc
```

迁移性定义为：

```text
transfer_rate = transfer_asr^2 / source_asr
```

隐蔽性定义为四个原始域检测方法的平均不可检测率：

```text
stealth_avg = mean(1 - TPR)
```

四个检测方法是：

```text
SentiNet
STRIP
ScaleUp
IBD_PSC
```

## 数据来源

分析脚本直接扫描：

```text
poisoned_train_set/cifar10
```

每个 noisy 实验目录读取：

```text
test_results_seed=2333*.json
test_stl10_results*.txt
sentinet_defense_results.json
strip_defense_results.json
scaleup_defense_results.json
ibd_psc_defense_results.json
```

其中：

- `test_results_seed=2333*.json` 提供 source-domain `clean_acc` 和 `source_asr`。
- `test_stl10_results*.txt` 提供 STL10 迁移域 `transfer_asr`。
- 四个 defense JSON 提供原始域检测 `TPR/AUC/FPR/threshold`。

所有比例会统一到 `[0, 1]`。如果 JSON 里 TPR 是 `37.55`，脚本会转换成 `0.3755`。

## 输出目录

所有分析结果保存在：

```text
analysis-transfer-asr2/noise_analysis/
```

图表保存在：

```text
analysis-transfer-asr2/noise_analysis/figures/
```

## 输出文件说明

### noise_acc_transfer_stealth_rows.csv

这是最重要的明细表。

一行对应一个 noisy 实验配置，包含：

```text
attack_type
poison_rate
strength_name
strength_value
input_noise_type
input_noise_level
clean_acc
difficulty
source_asr
transfer_asr
transfer_rate
四个 defense 的 TPR/AUC/FPR
四个 defense 的 stealth = 1 - TPR
stealth_avg
```

作用：

- 追踪每个实验配置的完整结果。
- 排查异常点。
- 作为所有汇总表、相关性表、回归和图表的基础数据。

### noise_missing_report.txt

这是数据完整性报告。

作用：

- 检查是否有目录缺少 source test、STL10 transfer 或 defense JSON。
- 检查哪些行无法计算 `transfer_rate`。
- 检查哪些行没有完整四个 defense 结果。

当前结果中：

```text
total_rows = 288
complete_defense_results = 288
valid_transfer_rate = 288
include_main_analysis = 259
```

说明所有 288 组 noisy 配置都有完整结果；主分析使用 259 行，是因为默认过滤了：

```text
source_asr >= 0.05
```

### noise_acc_transfer_stealth_summary_by_noise.csv

这是按噪声类型和强度汇总的表。

分组方式：

```text
input_noise_type + input_noise_level
```

作用：

- 判断噪声强度是否真的造成 clean ACC 下降。
- 判断 gaussian 和 uniform 是否产生相似趋势。
- 查看不同噪声强度下迁移性和隐蔽性的总体变化。

关键观察方式：

```text
noise_level 增大 -> clean_acc_mean 是否下降
noise_level 增大 -> transfer_rate_mean 是否变化
noise_level 增大 -> stealth_avg_mean 是否变化
```

### noise_acc_transfer_stealth_summary_by_attack.csv

这是按攻击类型、噪声类型和噪声强度汇总的表。

作用：

- 判断噪声难度效应是否依赖攻击方法。
- 检查局部 patch 攻击、全局攻击和优化型攻击是否表现不同。

如果不同攻击趋势差异很大，最终结论应写成：

```text
分类难度对迁移性-隐蔽性关系的影响具有 attack-dependent 特征。
```

### noise_acc_transfer_stealth_correlations.csv

这是相关性分析表。

包含 Pearson 和 Spearman 相关性。

重点字段：

```text
corr(clean_acc, transfer_rate)
corr(difficulty, transfer_rate)
corr(clean_acc, stealth_avg)
corr(difficulty, stealth_avg)
corr(transfer_rate, stealth_avg)
```

并且按以下层级分别输出：

```text
all
input_noise_type
attack_type
acc_bin
source_asr_threshold
```

解释方式：

- `corr(difficulty, transfer_rate)`：任务变难时迁移性是否变化。
- `corr(difficulty, stealth_avg)`：任务变难时隐蔽性是否变化。
- `corr(transfer_rate, stealth_avg)`：迁移性和隐蔽性是否存在 tradeoff。
- 按 `acc_bin` 分层的相关性是判断 ACC 是否调节二者关系的重要证据。

### noise_acc_bin_transfer_stealth.csv

这是 ACC 分层表。

脚本把主分析样本按 clean ACC 三分位切成：

```text
low_acc
mid_acc
high_acc
```

然后分别计算每一层里：

```text
corr(transfer_rate, stealth_avg)
```

作用：

- 直接回答“不同 ACC 难度层下，迁移性和隐蔽性的关系是否不同”。

当前结果显示：

```text
high_acc: Spearman corr ≈ -0.771
mid_acc : Spearman corr ≈ -0.423
low_acc : Spearman corr ≈  0.148
```

这说明在高 ACC 区间里，迁移性和隐蔽性的负相关更明显；低 ACC 区间里这种关系明显减弱。

### noise_acc_transfer_stealth_regression.txt

这是交互项回归结果。

**详细推导、0/1 编码、OLS 计算与数值例子见：[INTERACTION_REGRESSION_EXPLAINED_CN.md](./INTERACTION_REGRESSION_EXPLAINED_CN.md)。**

主模型：

```text
stealth_avg ~ transfer_rate * difficulty + C(attack_type) + C(poison_rate) + C(input_noise_type)
```

重点看：

```text
transfer_rate:difficulty
```

解释：

- 如果 `transfer_rate:difficulty` 显著，说明 difficulty 会调节迁移性和隐蔽性的关系。
- 如果不显著，说明当前 noisy 结果还不能用线性交互模型证明调节效应。

当前结果：

```text
transfer_rate:difficulty coefficient ≈ -0.0988
p-value ≈ 0.510
```

说明：在这个线性模型中，交互项不显著；当前结果不能强证明 difficulty 的线性交互调节效应。

但 ACC 分层表显示 high/mid/low ACC 下相关性差异较大，因此应结合分层分析讨论，而不是只看一个线性回归。

### noise_paired_delta_by_level.csv

这是配对变化表。

配对键：

```text
attack_type + poison_rate + strength_name + strength_value + input_noise_type
```

当前 reference 是：

```text
同一 input_noise_type 下的 level=0.030
```

计算：

```text
delta_clean_acc
delta_difficulty
delta_transfer_rate
delta_stealth_avg
```

作用：

- 控制攻击类型、poison rate 和触发器强度后，看同一个配置随噪声增强如何变化。
- 比全局散点更能减少攻击配置差异带来的混淆。

注意：如果后续补齐 no-noise baseline，应改用：

```text
input_noise_type=none, input_noise_level=0.000
```

作为 reference。

### noise_defense_breakdown_summary.csv

这是四个 defense 的拆分汇总表。

分组：

```text
defense + input_noise_type + input_noise_level
```

作用：

- 判断 `stealth_avg` 的变化主要来自哪个检测方法。
- 避免平均值掩盖某个 defense 的特殊行为。

例如：

- 如果 SentiNet 的 `1 - TPR` 随噪声变化很大，而 STRIP 几乎不变，需要在结论中单独说明。
- 如果四个 defense 方向一致，平均隐蔽性结论更稳。

## 图表说明

### figures/noise_acc_vs_level.png

作用：证明噪声强度是否制造 clean ACC 梯度。

看法：

```text
x = noise level
y = clean ACC
```

如果 level 增大时 ACC 下降，说明噪声确实提高了任务难度。

### figures/noise_acc_vs_transfer_rate.png

作用：观察 ACC 与迁移性的关系。

看法：

```text
x = clean_acc
y = transfer_rate
```

如果 clean ACC 下降时 transfer_rate 上升，说明任务变难可能增强迁移性。

如果 clean ACC 下降时 transfer_rate 下降，说明任务变难可能削弱迁移性。

### figures/noise_acc_vs_stealth.png

作用：观察 ACC 与隐蔽性的关系。

看法：

```text
x = clean_acc
y = stealth_avg
```

如果 clean ACC 下降时 stealth_avg 上升，说明任务变难后检测更困难。

如果 clean ACC 下降时 stealth_avg 下降，说明任务变难后异常更明显。

### figures/noise_transfer_vs_stealth_by_acc_bin.png

作用：回答核心问题。

看法：

```text
x = transfer_rate
y = stealth_avg
color = low_acc / mid_acc / high_acc
```

如果不同 ACC 分层的点云趋势不同，说明 ACC/任务难度会改变迁移性-隐蔽性关系。

### figures/noise_defense_stealth_breakdown.png

作用：拆解四个 defense。

看法：

```text
x = clean_acc
y = 1 - TPR
facet = defense
```

用于判断平均隐蔽性是否由单个 defense 主导。

### figures/noise_corr_heatmap_by_attack.png

作用：查看不同攻击下 transfer-stealth 关系是否一致。

如果不同攻击相关性方向差异很大，说明结论需要按攻击类型讨论。

### figures/noise_paired_delta.png

作用：查看同一攻击配置随噪声增强的配对变化。

看法：

```text
x = delta_clean_acc
y = delta_transfer_rate 或 delta_stealth_avg
```

如果同一配置中 ACC 下降同时伴随迁移性或隐蔽性系统性变化，这比非配对散点更有说服力。

## 当前初步结论

从当前自动分析结果看：

1. 噪声增强整体会降低 clean ACC，尤其 gaussian 从 `0.030` 到 `0.100` 下降明显。
2. 高 ACC 分层中，`transfer_rate` 与 `stealth_avg` 呈明显负相关。
3. 中 ACC 分层负相关减弱。
4. 低 ACC 分层中，负相关基本消失甚至略转正。
5. 线性交互项 `transfer_rate:difficulty` 在当前模型里不显著，因此不能只凭回归说调节效应显著。
6. 更稳妥的表述是：

```text
当前 noisy 实验显示，不同 ACC 难度层下迁移性与隐蔽性的相关结构发生明显变化；
但在线性交互回归中尚未形成显著证据，因此后续应结合 no-noise baseline 和配对分析进一步验证。
```

## 后续建议

最优先补：

```text
no-noise baseline
input_noise_type=none
input_noise_level=0.000
同样 8 attacks × 2 poison rates × 3 strengths
```

作用：

```text
把配对 reference 从 noisy level=0.030 改成真正无噪声。
```

第二优先补：

```text
clean-only noisy baseline
poison_type=none
gaussian/uniform
level=0.030/0.060/0.100
```

作用：

```text
证明噪声本身确实控制了 clean ACC，而不是攻击配置导致 ACC 波动。
```
