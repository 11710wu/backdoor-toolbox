# CIFAR-10 噪声难度实验：ACC、迁移性与隐蔽性分析说明

本目录由 `analyze_noise_acc_transfer_stealth.py` 生成，用于回答：

> CIFAR-10 训练图加噪声导致 clean ACC 改变后，分类任务难度是否会调节迁移性与隐蔽性的关系？

核心定义：

- `difficulty = 1 - clean_acc`
- `transfer_rate = transfer_asr^2 / source_asr`
- `stealth_avg = mean(1 - TPR)`，TPR 来自 SentiNet、STRIP、ScaleUp、IBD-PSC 四个原始域检测方法。

## 文件作用

- `noise_acc_transfer_stealth_rows.csv`：逐实验配置明细表，是所有后续分析的来源。
- `noise_missing_report.txt`：数据完整性报告，列出缺失 source test、STL10 transfer 或 defense JSON 的目录。
- `noise_acc_transfer_stealth_summary_by_noise.csv`：按噪声类型和强度汇总，用于确认噪声是否制造 clean ACC 梯度。
- `noise_acc_transfer_stealth_summary_by_attack.csv`：按攻击、噪声类型、噪声强度汇总，用于判断趋势是否 attack-dependent。
- `noise_acc_transfer_stealth_correlations.csv`：相关性表，包含 ACC、difficulty、transfer_rate、stealth_avg 之间的 Pearson/Spearman 相关性。
- `noise_acc_bin_transfer_stealth.csv`：按 clean ACC 三分位分层后，统计每层 transfer_rate 与 stealth_avg 的关系。
- `noise_acc_transfer_stealth_regression.txt`：交互项回归结果，重点看 `transfer_rate:difficulty`。**公式、虚拟变量与 OLS 计算见上级目录 [INTERACTION_REGRESSION_EXPLAINED_CN.md](../INTERACTION_REGRESSION_EXPLAINED_CN.md)。**
- `noise_paired_delta_by_level.csv`：同一攻击配置内，以 noise level 0.030 为 reference，比较更高噪声强度造成的变化。
- `noise_defense_breakdown_summary.csv`：拆开四个 defense，看平均隐蔽性主要由哪些检测方法驱动。
- `figures/`：核心可视化图表。

## 解读顺序

1. 先看 `noise_missing_report.txt`，确认数据完整性。
2. 看 `summary_by_noise.csv` 和 `figures/noise_acc_vs_level.png`，确认噪声是否降低 clean ACC。
3. 看 `figures/noise_acc_vs_transfer_rate.png`，判断 ACC 是否影响迁移性。
4. 看 `figures/noise_acc_vs_stealth.png`，判断 ACC 是否影响隐蔽性。
5. 看 `noise_acc_bin_transfer_stealth.csv` 和 `figures/noise_transfer_vs_stealth_by_acc_bin.png`，判断不同 ACC 难度层下 transfer-stealth 关系是否变化。
6. 看 `noise_acc_transfer_stealth_regression.txt`，用交互项判断 difficulty 是否调节 transfer_rate 与 stealth_avg 的关系。
7. 看 `noise_defense_breakdown_summary.csv` 和 `figures/noise_defense_stealth_breakdown.png`，确认平均隐蔽性是否由单个 defense 主导。
