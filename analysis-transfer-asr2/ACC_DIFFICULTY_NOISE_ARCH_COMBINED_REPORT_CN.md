# ACC 难度联合分析报告：噪声实验与架构实验

## 0. 先看结论

这份报告把两条证据链放在一起：

- `architecture`：完整旧基线 + SmallCNN/ResNet34 架构补充，回答“不同模型/数据集生态下现象是否仍存在”。
- `noise`：CIFAR-10 + SmallCNN 加噪实验，回答“同模型同数据集下主动改变输入难度后关系如何变化”。

最稳妥的联合结论是：

- 两条证据链都显示 `transfer_rate` 与 `stealth_avg` 为负相关：architecture Spearman=`-0.3756`，noise Spearman=`-0.3465`。
- 两条证据链的均值水平不同：architecture 平均 stealth_avg=`0.5571`，noise 平均 stealth_avg=`0.3702`；noise 的隐蔽性均值明显更低。
- 两条证据链里 clean_acc 与 transfer_rate 的方向不同：architecture Pearson=`-0.4517`，noise Pearson=`0.3612`。
- 因此，联合分析不能直接写成“ACC 是唯一原因”。更准确的说法是：difficulty/ACC 会参与调节 transfer-stealth 关系，但 noise 和 architecture 是两种不同 intervention。

## 1. 建议优先看的图

| 优先级 | 图 | 用途 | 汇报时怎么说 |
|---:|---|---|---|
| 1 | `combined_metric_overview.png` | 比较 architecture/noise 的均值水平 | 先说明两条证据链不是同一分布 |
| 2 | `combined_transfer_stealth_facets.png` | 看两条证据链各自的 transfer-stealth 点云 | 两边都负相关，但点云结构不同 |
| 3 | `combined_difficulty_relationships.png` | 看 difficulty 与 transfer/stealth 的方向 | architecture 和 noise 的 difficulty effect 不等价 |
| 4 | `combined_binned_median_trend.png` | 用分箱趋势展示关系 | 普通散点不明显时看这张 |
| 5 | `combined_acc_bin_spearman.png` | 看 ACC 分层相关性 | 说明不同难度区间 tradeoff 强度不同 |
| 6 | `combined_attack_heatmap.png` | 看 attack-dependent 差异 | 说明不能只报 overall |

![Combined metric overview](arch_acc_analysis/figures/combined_metric_overview.png)

![Combined transfer-stealth facets](arch_acc_analysis/figures/combined_transfer_stealth_facets.png)

![Combined difficulty relationships](arch_acc_analysis/figures/combined_difficulty_relationships.png)

![Combined binned median trend](arch_acc_analysis/figures/combined_binned_median_trend.png)

![Combined ACC-bin Spearman](arch_acc_analysis/figures/combined_acc_bin_spearman.png)

![Combined attack heatmap](arch_acc_analysis/figures/combined_attack_heatmap.png)

HTML 展示页：`analysis-transfer-asr2/ACC_DIFFICULTY_NOISE_ARCH_COMBINED_DASHBOARD_CN.html`。

## 2. 数据来源与总体汇总

| variation_source   |   n_rows |   clean_acc_mean |   clean_acc_median |   difficulty_mean |   transfer_rate_mean |   transfer_rate_median |   stealth_avg_mean |   stealth_avg_median |
|:-------------------|---------:|-----------------:|-------------------:|------------------:|---------------------:|-----------------------:|-------------------:|---------------------:|
| architecture       |      969 |         0.776705 |             0.886  |          0.223295 |             0.764672 |                0.92862 |           0.557054 |             0.595159 |
| noise              |      259 |         0.737681 |             0.8175 |          0.262319 |             0.801989 |                0.84066 |           0.370156 |             0.384976 |

## 3. 关键发现

- 架构/模型实验共有 `969` 条主分析记录，平均 clean ACC=`0.7767`，平均 transfer_rate=`0.7647`，平均 stealth_avg=`0.5571`。
- 噪声实验共有 `259` 条主分析记录，平均 clean ACC=`0.7377`，平均 transfer_rate=`0.8020`，平均 stealth_avg=`0.3702`。
- 全体 transfer_rate 中位数为 `0.9208`，落在 `[0.9, 1.1]` 的比例为 `0.4853`。所以普通散点图容易显得趋势不明显，分箱趋势更适合汇报。

## 4. 相关性对比

| group_type       | group_name   |   n_rows |   pearson_clean_acc_transfer_rate |   spearman_clean_acc_transfer_rate |   pearson_transfer_rate_stealth_avg |   spearman_transfer_rate_stealth_avg |
|:-----------------|:-------------|---------:|----------------------------------:|-----------------------------------:|------------------------------------:|-------------------------------------:|
| all              | combined     |     1228 |                         -0.142927 |                          -0.350343 |                           -0.280913 |                            -0.385039 |
| variation_source | architecture |      969 |                         -0.451681 |                          -0.473991 |                           -0.363225 |                            -0.375576 |
| variation_source | noise        |      259 |                          0.361201 |                           0.23318  |                           -0.117856 |                            -0.346479 |

需要重点解释的一点：architecture 中 clean_acc 与 transfer_rate 是负相关，noise 中是正相关。这说明“换模型/换数据集”和“给同一模型加噪声”不是同一种 difficulty intervention。

## 5. ACC 分层结果

| variation_source   | acc_bin   |   n_rows |   clean_acc_mean |   difficulty_mean |   transfer_rate_median |   stealth_avg_median |   spearman_transfer_stealth |
|:-------------------|:----------|---------:|-----------------:|------------------:|-----------------------:|---------------------:|----------------------------:|
| architecture       | high_acc  |      322 |         0.940458 |         0.0595419 |               0.627936 |             0.593313 |                   -0.582089 |
| architecture       | low_acc   |      323 |         0.598236 |         0.401764  |               1        |             0.609059 |                   -0.401081 |
| architecture       | mid_acc   |      324 |         0.791882 |         0.208118  |               0.942667 |             0.589712 |                   -0.348504 |
| noise              | high_acc  |       86 |         0.865131 |         0.134869  |               0.967698 |             0.34097  |                   -0.77114  |
| noise              | low_acc   |       87 |         0.532441 |         0.467559  |               0.702977 |             0.38967  |                    0.147864 |
| noise              | mid_acc   |       86 |         0.817858 |         0.182142  |               1.00016  |             0.422258 |                   -0.423331 |

最适合汇报的三行对照：

| source | low_acc | mid_acc | high_acc |
|---|---:|---:|---:|
| architecture | -0.4011 | -0.3485 | -0.5821 |
| noise | 0.1479 | -0.4233 | -0.7711 |

如果 noise 的低 ACC 层变弱或接近消失，而 high ACC 层仍强，汇报时可以说：在控制模型/数据集后，输入难度确实会改变 tradeoff 强度；但方向不是简单线性。

## 6. 攻击类型分层

| variation_source   | attack_type    |   n_rows |   clean_acc_mean |   difficulty_mean |   transfer_rate_mean |   transfer_rate_median |   stealth_avg_mean |   stealth_avg_median |
|:-------------------|:---------------|---------:|-----------------:|------------------:|---------------------:|-----------------------:|-------------------:|---------------------:|
| architecture       | SIG            |       97 |         0.813968 |          0.186032 |             0.617396 |               0.674414 |          0.756643  |            0.806365  |
| noise              | SIG            |       36 |         0.835205 |          0.164795 |             0.758706 |               0.762231 |          0.385775  |            0.382045  |
| architecture       | WaNet          |      145 |         0.770797 |          0.229203 |             0.516293 |               0.578592 |          0.750509  |            0.776717  |
| noise              | WaNet          |       36 |         0.222521 |          0.777479 |             0.240936 |               0.150154 |          0.316495  |            0.301935  |
| architecture       | adaptive_blend |      111 |         0.761479 |          0.238521 |             0.712838 |               0.800804 |          0.712399  |            0.711841  |
| noise              | adaptive_blend |       27 |         0.825796 |          0.174204 |             0.493737 |               0.46771  |          0.64424   |            0.612615  |
| architecture       | adaptive_patch |      101 |         0.773324 |          0.226676 |             0.899522 |               0.99213  |          0.39877   |            0.409891  |
| noise              | adaptive_patch |       36 |         0.829799 |          0.170201 |             0.942343 |               1.01205  |          0.148315  |            0.0699765 |
| architecture       | badnet         |      161 |         0.777665 |          0.222335 |             0.983655 |               1        |          0.171701  |            0.161724  |
| noise              | badnet         |       24 |         0.838016 |          0.161984 |             1.01311  |               1.00739  |          0.0608388 |            0.0586243 |
| architecture       | belt           |      117 |         0.743016 |          0.256984 |             0.756691 |               0.997235 |          0.457117  |            0.446702  |
| noise              | belt           |       36 |         0.769517 |          0.230483 |             1.02556  |               1.03273  |          0.501695  |            0.509438  |
| architecture       | blend          |      115 |         0.767611 |          0.232389 |             0.816047 |               0.936978 |          0.660595  |            0.665739  |
| noise              | blend          |       32 |         0.821867 |          0.178133 |             0.527803 |               0.599995 |          0.557964  |            0.510846  |
| architecture       | upgd           |      122 |         0.810369 |          0.189631 |             0.782737 |               0.840796 |          0.664919  |            0.651084  |
| noise              | upgd           |       32 |         0.834289 |          0.165711 |             1.44838  |               1.0673   |          0.327463  |            0.299717  |

攻击类型是强混杂因素。联合报告里不能只说 architecture vs noise 的总体均值，还要说明 badnet/adaptive_patch/WaNet/SIG/upgd 等攻击的落点不同。

## 7. 分箱趋势和 rank 趋势

- `combined_acc_effect_binned_median_trend.csv`：对应 `combined_binned_median_trend.png`。
- `combined_acc_effect_rank_binned_trend.csv`：对应 `combined_rank_binned_trend.png`。

分箱趋势预览：

| variation_source   |   bin_id |   n_rows |   transfer_rate_median |   stealth_avg_median |   difficulty_mean |
|:-------------------|---------:|---------:|-----------------------:|---------------------:|------------------:|
| architecture       |        1 |      122 |              0.0990404 |            0.761052  |         0.0992838 |
| architecture       |        2 |      121 |              0.401233  |            0.670928  |         0.1368    |
| architecture       |        3 |      121 |              0.677962  |            0.650676  |         0.164664  |
| architecture       |        4 |      121 |              0.87713   |            0.610197  |         0.202977  |
| architecture       |        5 |      121 |              0.966438  |            0.610001  |         0.251208  |
| architecture       |        6 |      121 |              0.999725  |            0.19421   |         0.237817  |
| architecture       |        7 |      121 |              1.00417   |            0.489543  |         0.367436  |
| architecture       |        8 |      121 |              1.0608    |            0.429405  |         0.327196  |
| noise              |        1 |       33 |              0.131418  |            0.387649  |         0.59636   |
| noise              |        2 |       32 |              0.336418  |            0.609379  |         0.343207  |
| noise              |        3 |       32 |              0.677351  |            0.454246  |         0.205281  |
| noise              |        4 |       33 |              0.787138  |            0.385791  |         0.216913  |
| noise              |        5 |       32 |              0.980099  |            0.339932  |         0.185566  |
| noise              |        6 |       32 |              1.00818   |            0.0642373 |         0.157848  |
| noise              |        7 |       32 |              1.03504   |            0.137287  |         0.185348  |
| noise              |        8 |       33 |              1.10554   |            0.447346  |         0.200928  |

Rank 趋势预览：

| variation_source   |   bin_id |   n_rows |   transfer_rank_median |   stealth_rank_median |
|:-------------------|---------:|---------:|-----------------------:|----------------------:|
| architecture       |        1 |       97 |              0.0505676 |             0.700722  |
| architecture       |        2 |       97 |              0.150671  |             0.652219  |
| architecture       |        3 |       97 |              0.250774  |             0.593395  |
| architecture       |        4 |       97 |              0.350877  |             0.614035  |
| architecture       |        5 |       97 |              0.45098   |             0.501548  |
| architecture       |        6 |       96 |              0.550568  |             0.503612  |
| architecture       |        7 |      131 |              0.667699  |             0.149639  |
| architecture       |        8 |       65 |              0.768834  |             0.356037  |
| architecture       |        9 |       95 |              0.851393  |             0.206398  |
| architecture       |       10 |       97 |              0.950464  |             0.339525  |
| noise              |        1 |       26 |              0.0521236 |             0.440154  |
| noise              |        2 |       26 |              0.15251   |             0.92471   |
| noise              |        3 |       26 |              0.252896  |             0.704633  |
| noise              |        4 |       26 |              0.353282  |             0.579151  |
| noise              |        5 |       26 |              0.453668  |             0.490347  |
| noise              |        6 |       25 |              0.552124  |             0.482625  |
| noise              |        7 |       26 |              0.650579  |             0.0598456 |
| noise              |        8 |       26 |              0.750965  |             0.210425  |
| noise              |        9 |       26 |              0.851351  |             0.19305   |
| noise              |       10 |       26 |              0.951737  |             0.644788  |

## 回归结果摘要

- `transfer_rate:difficulty` 系数：`0.5556593711963962`
- `transfer_rate:difficulty` p 值：`2.144166302463283e-12`
- 在这个联合回归中，交互项为正表示 difficulty 增大时，transfer_rate 对 stealth_avg 的负向斜率会被削弱。但这个结果混合了 architecture 与 noise 两类变化来源，不能单独解释为 ACC 的纯因果效应。

完整回归表见 `arch_acc_analysis/combined_acc_effect_regression.txt`。

## 9. 解读原则

- 噪声实验更接近同模型同数据集下的 difficulty intervention。
- 架构实验更接近不同模型和数据集设置下的生态对照。
- 两者趋势一致时，可以加强“ACC/难度影响 transfer-stealth 关系”的证据链。
- 两者趋势不一致时，应优先检查 attack type、source ASR 长尾和 defense-specific 差异。

## 10. 汇报建议

建议按 5 页讲：

1. **研究设计**：先说明两条证据链不同。architecture 是模型生态对照，noise 是更接近控制变量的 difficulty intervention。
2. **共同现象**：两条证据链都显示 transfer-stealth 负相关，但强度不同。
3. **关键差异**：difficulty/ACC 对 transfer_rate 的方向不同，不能把两批数据混成一个简单因果结论。
4. **难度分层**：用 ACC-bin Spearman 图说明 tradeoff 强度会随难度区间变化。
5. **攻击混杂**：用 attack heatmap 说明不同攻击机制决定了点落在哪个区域。

可以直接使用的汇报话术：

```text
联合分析不是为了把噪声实验和架构实验简单混成一个大样本，
而是把它们作为两条互补证据链：
噪声实验更接近同模型同数据集下的难度干预，
架构实验更接近不同模型和数据集生态下的外部验证。
两条证据链都显示迁移性和隐蔽性整体负相关，
但 clean ACC 与 transfer_rate 的方向并不一致，说明 difficulty effect 不是单一线性因果。
因此更稳妥的结论是：ACC/任务难度会调节 transfer-stealth 关系，
但该调节受到攻击类型、模型结构和检测器响应共同影响。
```

## 11. 每个输出文件的作用

- `combined_acc_effect_summary.csv`：看 architecture/noise 两条证据链的总体均值差异。
- `combined_acc_effect_rows.csv`：保存逐实验明细，可用于后续重新画图或做更复杂回归。
- `combined_acc_effect_correlations.csv`：整体、证据链、攻击类型下的相关性。
- `combined_acc_effect_acc_bins.csv`：按 clean ACC 分层后的 transfer-stealth 相关性。
- `combined_acc_effect_attack_summary.csv`：按攻击类型拆解两条证据链。
- `combined_acc_effect_plot_diagnostics.csv`：解释普通散点图为什么不明显。
- `combined_acc_effect_binned_median_trend.csv`：分箱中位数趋势。
- `combined_acc_effect_rank_binned_trend.csv`：rank 分箱趋势。
- `combined_acc_effect_regression.txt`：保存联合回归完整统计表，主要查看 `transfer_rate:difficulty` 和 `C(variation_source)`。
- `ACC_DIFFICULTY_NOISE_ARCH_COMBINED_DASHBOARD_CN.html`：联合分析静态展示页。
- `ARCH_ACC_TRANSFER_STEALTH_RESULT_REPORT_CN.md`：只讨论架构/模型实验，是和 baseline 结果对照时的主报告。
- `ACC_DIFFICULTY_NOISE_ARCH_COMBINED_REPORT_CN.md`：把噪声实验与架构实验放在同一个解释框架中，主要用于写论文讨论和后续实验设计。
