# 完整旧基线 ACC-迁移性-隐蔽性结果总结

本文档总结 `/workspace/backdoor-toolbox-new1/poisoned_train_set1` 中已有完整旧基线结果。它的作用类似噪声分析文档中的“完整旧基线参照”：先回答原有大规模实验中 transfer-stealth 关系是否存在，再为后续噪声实验和新模型架构实验提供参照背景。

字段、图表和逐实验明细主要保存在：

```text
analysis-transfer-asr2/baseline_full_analysis/
```

## 1. 研究问题

这份完整旧基线不是为了单独回答某一个新模型或某一种噪声的效果，而是回答三个基础问题：

```text
1. 在原有多数据集、多模型、多攻击实验里，迁移性和隐蔽性是否整体存在 tradeoff？
2. 这种 tradeoff 是否会随数据集、ACC 区间、攻击方法和检测器改变？
3. 后续噪声实验和 SmallCNN/ResNet34 架构补充，应该和什么基线结果对照？
```

使用的定义和当前论文问题保持一致：

```text
difficulty = 1 - clean_acc
transfer_rate = transfer_asr^2 / source_asr
stealth_avg = mean(1 - TPR)
```

其中：

- `clean_acc` 是源域测试准确率；
- `source_asr` 是源域 ASR；
- `transfer_asr` 是迁移域 ASR；
- `transfer_rate` 是当前固定使用的迁移性指标；
- `stealth_avg` 来自源域四个检测方法：`SentiNet / STRIP / ScaleUp / IBD_PSC`；
- `stealth_avg` 越大，说明四个检测方法平均越不容易检出，隐蔽性越强。

注意：`transfer_rate` 不是单纯的迁移 ASR，它会受到 `source_asr` 分母影响。因此后续所有结论都要同时注意 source ASR 过滤和 transfer_rate 长尾。

## 2. 当前数据规模与完整性

- 解析到 baseline 目录行数：`1491`
- 完整四防御结果行数：`1491`
- 有效 transfer_rate 行数：`1491`
- 主分析行数：`1299`

主分析过滤条件：来自 `poisoned_train_set1`，四个防御完整，`transfer_rate` 可计算，`source_asr >= 0.05`，并且 `clean_acc / stealth_avg` 非空。

这说明完整旧基线的数据完整性较好：全部 baseline 目录都有四个源域检测防御结果，也都能计算 transfer_rate。主分析从 1491 行中过滤到 1299 行，主要是为了排除 source ASR 太低时 `transfer_rate` 被分母异常放大的配置。

## 3. 实验覆盖内容

完整旧基线覆盖三个源域：

```text
CIFAR-10
MNIST-M
Tiny-ImageNet
```

覆盖的主要模型体系是：

```text
ResNet18
MobileNetV2
VGG19-BN
```

攻击方法覆盖 8 类：

```text
badnet/basic
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

当前代码里的 `SIG` 和 `UPGD` 仍然是 all-to-one dirty-label / label-flipping 版本，不是 clean-label。后续如果改成 clean-label，需要单独标注和重新分析。

## 4. 总体结论

- 在完整旧 baseline 数据上，`transfer_rate` 与 `stealth_avg` 的整体相关为：Pearson=`-0.2147`，Spearman=`-0.3234`。
- 提高 source ASR 过滤到 `source_asr>=0.10` 后，整体相关为：Pearson=`-0.1976`，Spearman=`-0.3137`。如果两个阈值下方向一致，说明结论不完全由低 source ASR 分母放大造成。
- 按数据集拆分：CIFAR-10 Spearman=`-0.5803`，Tiny-ImageNet Spearman=`-0.4303`，MNIST-M Spearman=`-0.3484`。
- 全局 ACC 分层后：high_acc Spearman=`-0.3492`，mid_acc Spearman=`-0.5875`，low_acc Spearman=`-0.4084`。这部分用于判断分类难度是否改变 transfer-stealth 关系强度。

更具体地说，完整旧基线支持如下结论：

1. `transfer_rate` 与 `stealth_avg` 整体呈负相关，说明迁移性越强时，平均隐蔽性通常越弱。
2. 这种关系在 CIFAR-10 上最强，Tiny-ImageNet 次之，MNIST-M 相对更弱。
3. ACC 分层后，low/mid/high ACC 区间的相关强度不同，说明任务难度可能改变 tradeoff 的形态。
4. 这个关系不是漂亮的强线性关系，而更像分组/排序型关系；普通散点图会被 transfer_rate 集中、极端值、数据集和攻击混杂影响。
5. 攻击类型和检测方法差异很大，最终论文分析不能只报告 overall mean。

完整旧基线最适合支撑的表述是：

```text
在原有多数据集、多模型、多攻击实验中，迁移性和隐蔽性整体存在负相关；
但这个 tradeoff 不是固定常数，而是会随数据集、ACC 区间、攻击类型和检测器响应变化。
```

## 5. 为什么普通基线散点图看不出很清楚的趋势

普通散点图看起来不明显，主要不是因为没有关系，而是因为 `transfer_rate` 的分布非常特殊。

核心分布诊断：

- 主分析样本数：`1299`
- transfer_rate 中位数：`0.9667`
- transfer_rate 25%/75% 分位：`0.5593` / `1.0154`
- transfer_rate 95% 分位和最大值：`1.3097` / `9.5545`
- transfer_rate 落在 `[0.9, 1.1]` 的比例：`0.4742`
- transfer_rate 大于 `1.5` 的比例：`0.0354`

也就是说，将近一半样本的 transfer_rate 都挤在 1 附近，同时少数极端值会拉长横轴。因此普通散点图会形成一条很厚的竖带，视觉上不容易看到斜率。

因此，汇报时不建议把普通散点图作为唯一证据。更推荐结合：

```text
分组相关性表
ACC 分层表
dataset 分面图
分箱中位数趋势图
rank 分箱趋势图
```

最适合汇报趋势的图是：

```text
analysis-transfer-asr2/baseline_full_analysis/figures/baseline_binned_median_trend_by_dataset.png
analysis-transfer-asr2/baseline_full_analysis/figures/baseline_rank_binned_trend_by_dataset.png
analysis-transfer-asr2/baseline_full_analysis/figures/baseline_transfer_vs_stealth_dataset_facets_clipped.png
```

这些图对应的完整说明见：

```text
analysis-transfer-asr2/BASELINE_FULL_FIGURE_ANALYSIS_CN.md
```

## 6. Dataset / Arch 汇总

| dataset       | arch_base   | dataset_arch              |   n_rows |   clean_acc_mean |   clean_acc_median |   difficulty_mean |   source_asr_mean |   source_asr_median |   transfer_asr_mean |   transfer_asr_median |   transfer_rate_mean |   transfer_rate_median |   stealth_avg_mean |   stealth_avg_median |
|:--------------|:------------|:--------------------------|---------:|-----------------:|-------------------:|------------------:|------------------:|--------------------:|--------------------:|----------------------:|---------------------:|-----------------------:|-------------------:|---------------------:|
| cifar10       | ResNet18    | cifar10:ResNet18          |      159 |         0.949429 |           0.95225  |         0.0505708 |          0.819732 |            0.947171 |            0.64877  |              0.708281 |             0.581159 |               0.625573 |           0.558729 |             0.59468  |
| cifar10       | mobilenetv2 | cifar10:mobilenetv2       |      151 |         0.895228 |           0.903625 |         0.104772  |          0.781297 |            0.93021  |            0.642685 |              0.731563 |             0.609806 |               0.671799 |           0.517306 |             0.577195 |
| cifar10       | vgg19_bn    | cifar10:vgg19_bn          |      157 |         0.930613 |           0.9355   |         0.0693869 |          0.764912 |            0.891422 |            0.643193 |              0.711094 |             0.626555 |               0.628429 |           0.538949 |             0.584521 |
| mnistm        | ResNet18    | mnistm:ResNet18           |      145 |         0.987397 |           0.9875   |         0.0126026 |          0.704448 |            0.855528 |            0.783978 |              0.99316  |             1.04743  |               1.02621  |           0.60011  |             0.622108 |
| mnistm        | mobilenetv2 | mnistm:mobilenetv2        |      132 |         0.980576 |           0.98075  |         0.0194242 |          0.641704 |            0.72627  |            0.725448 |              0.957984 |             0.897041 |               1.01208  |           0.692278 |             0.743373 |
| mnistm        | vgg19_bn    | mnistm:vgg19_bn           |      137 |         0.983801 |           0.984    |         0.0161989 |          0.630046 |            0.671133 |            0.720325 |              0.972501 |             1.01333  |               1.01444  |           0.612863 |             0.621312 |
| tiny_imagenet | ResNet18    | tiny_imagenet:ResNet18    |      147 |         0.639537 |           0.644875 |         0.360463  |          0.769139 |            0.848531 |            0.850412 |              0.942714 |             0.959353 |               1.00252  |           0.598611 |             0.643787 |
| tiny_imagenet | mobilenetv2 | tiny_imagenet:mobilenetv2 |      140 |         0.582504 |           0.585375 |         0.417496  |          0.763593 |            0.889475 |            0.820355 |              0.953266 |             0.906788 |               0.996952 |           0.56849  |             0.600877 |
| tiny_imagenet | vgg19_bn    | tiny_imagenet:vgg19_bn    |      131 |         0.609389 |           0.6165   |         0.390611  |          0.728634 |            0.837101 |            0.800886 |              0.921106 |             0.916105 |               1.0005   |           0.596957 |             0.651017 |

这张表用于回答：不同数据集和模型本身的 clean ACC、transfer_rate、stealth_avg 是否处于不同水平。

汇报时重点看：

- CIFAR-10 平均 ACC 高，transfer_rate 相对较低，tradeoff 最明显。
- Tiny-ImageNet 平均 ACC 明显低，但 transfer_rate 较高，说明任务难度和迁移性不是简单单调关系。
- MNIST-M ACC 很高，但 transfer_rate 和 stealth_avg 都偏高，说明数据集本身也强烈影响结论。

这张表是完整旧基线的第一层背景证据：不同 dataset/arch 的基础水平不同，后续所有整体平均都需要谨慎解释。

## 7. 最关键证据：ACC 分层下的 transfer-stealth 关系

| group_type      | dataset       | acc_bin   |   n_rows |   clean_acc_min |   clean_acc_max |   clean_acc_mean |   transfer_rate_mean |   transfer_rate_median |   stealth_avg_mean |   stealth_avg_median |   pearson_transfer_stealth |   spearman_transfer_stealth |
|:----------------|:--------------|:----------|---------:|----------------:|----------------:|-----------------:|---------------------:|-----------------------:|-------------------:|---------------------:|---------------------------:|----------------------------:|
| global_acc_bin  | all           | low_acc   |      433 |        0.083875 |        0.889    |         0.617886 |             0.917757 |               1        |           0.585106 |             0.623479 |                  -0.37724  |                   -0.408384 |
| global_acc_bin  | all           | mid_acc   |      435 |        0.889125 |        0.955375 |         0.928342 |             0.60026  |               0.625573 |           0.538509 |             0.590107 |                  -0.543779 |                   -0.587501 |
| global_acc_bin  | all           | high_acc  |      431 |        0.9555   |        0.989875 |         0.982954 |             0.97784  |               1.01373  |           0.631342 |             0.644734 |                  -0.21964  |                   -0.349205 |
| dataset_acc_bin | cifar10       | low_acc   |      156 |        0.083875 |        0.90975  |         0.89369  |             0.6134   |               0.661842 |           0.522542 |             0.571733 |                  -0.553314 |                   -0.593253 |
| dataset_acc_bin | cifar10       | mid_acc   |      155 |        0.909875 |        0.938875 |         0.932307 |             0.603394 |               0.585388 |           0.520424 |             0.564828 |                  -0.5392   |                   -0.60715  |
| dataset_acc_bin | cifar10       | high_acc  |      156 |        0.939    |        0.959375 |         0.95078  |             0.60024  |               0.682882 |           0.572973 |             0.613218 |                  -0.519022 |                   -0.55207  |
| dataset_acc_bin | mnistm        | low_acc   |      141 |        0.96925  |        0.982875 |         0.980289 |             0.850263 |               1.00774  |           0.683312 |             0.73716  |                  -0.251258 |                   -0.313408 |
| dataset_acc_bin | mnistm        | mid_acc   |      136 |        0.983    |        0.98575  |         0.984274 |             1.06188  |               1.01678  |           0.624114 |             0.621414 |                  -0.195793 |                   -0.33966  |
| dataset_acc_bin | mnistm        | high_acc  |      137 |        0.985875 |        0.989875 |         0.987645 |             1.057    |               1.03005  |           0.592207 |             0.622108 |                  -0.220501 |                   -0.320427 |
| dataset_acc_bin | tiny_imagenet | low_acc   |      140 |        0.522125 |        0.5905   |         0.578616 |             0.901769 |               0.997897 |           0.564753 |             0.598023 |                  -0.494679 |                   -0.513007 |
| dataset_acc_bin | tiny_imagenet | mid_acc   |      140 |        0.590625 |        0.622125 |         0.611612 |             0.929    |               1.00133  |           0.55921  |             0.606384 |                  -0.380564 |                   -0.266953 |
| dataset_acc_bin | tiny_imagenet | high_acc  |      138 |        0.622375 |        0.654875 |         0.643192 |             0.954184 |               1.00113  |           0.640805 |             0.721749 |                  -0.441876 |                   -0.542012 |

这张表用于回答：高 ACC、中 ACC、低 ACC 区间里，迁移性和隐蔽性的负相关是否同样强。

最重要的三行是 global ACC bin：

| ACC bin | Spearman transfer-stealth | 解释 |
|---|---:|---|
| low_acc | -0.4084 | 低 ACC 条件下仍有负相关，但强度中等 |
| mid_acc | -0.5875 | 中等 ACC 条件下负相关最强 |
| high_acc | -0.3492 | 高 ACC 条件下负相关仍存在，但弱于 mid_acc |

这部分最适合和噪声实验结合。完整旧基线说明 ACC 区间本来就会改变 tradeoff 强度；噪声实验进一步说明，在固定 CIFAR-10 + SmallCNN 下，主动降低 ACC 后 tradeoff 会明显变化。

## 8. 攻击类型差异

| dataset       | attack_family   |   n_rows |   clean_acc_mean |   source_asr_mean |   transfer_asr_mean |   transfer_rate_mean |   stealth_avg_mean |
|:--------------|:----------------|---------:|-----------------:|------------------:|--------------------:|---------------------:|-------------------:|
| cifar10       | SIG             |       54 |         0.932706 |          0.702195 |           0.549149  |          0.458898    |           0.72421  |
| cifar10       | WaNet           |       67 |         0.929981 |          0.708701 |           0.207332  |          0.101483    |           0.688494 |
| cifar10       | adaptive_blend  |       48 |         0.928922 |          0.680201 |           0.614264  |          0.594329    |           0.673939 |
| cifar10       | adaptive_patch  |       45 |         0.929814 |          0.413695 |           0.543944  |          0.821508    |           0.548574 |
| cifar10       | badnet          |       77 |         0.930432 |          0.981329 |           0.949527  |          0.927979    |           0.138747 |
| cifar10       | belt            |       54 |         0.906463 |          0.917873 |           0.639771  |          0.467554    |           0.496372 |
| cifar10       | blend           |       50 |         0.931488 |          0.916708 |           0.800478  |          0.722829    |           0.612277 |
| cifar10       | upgd            |       72 |         0.916299 |          0.844085 |           0.777617  |          0.735206    |           0.572129 |
| mnistm        | SIG             |        8 |         0.984281 |          0.179823 |           0.0107132 |          0.000778301 |           0.824632 |
| mnistm        | WaNet           |       64 |         0.984518 |          0.639537 |           0.865508  |          1.64995     |           0.594155 |
| mnistm        | adaptive_blend  |       50 |         0.983287 |          0.679009 |           0.785597  |          0.945486    |           0.736144 |
| mnistm        | adaptive_patch  |       45 |         0.983097 |          0.323612 |           0.482791  |          0.770849    |           0.650919 |
| mnistm        | badnet          |       72 |         0.984318 |          0.882375 |           0.969816  |          1.09375     |           0.339921 |
| mnistm        | belt            |       52 |         0.982829 |          0.798882 |           0.775214  |          0.797893    |           0.759685 |
| mnistm        | blend           |       55 |         0.984602 |          0.878547 |           0.967824  |          1.11988     |           0.58127  |
| mnistm        | upgd            |       68 |         0.984869 |          0.424871 |           0.415714  |          0.584007    |           0.818966 |
| tiny_imagenet | SIG             |       31 |         0.620863 |          0.55884  |           0.649441  |          0.768169    |           0.884256 |
| tiny_imagenet | WaNet           |       70 |         0.614502 |          0.638076 |           0.735772  |          0.941452    |           0.801224 |
| tiny_imagenet | adaptive_blend  |       54 |         0.611988 |          0.704783 |           0.761288  |          0.829576    |           0.741726 |
| tiny_imagenet | adaptive_patch  |       45 |         0.613186 |          0.754157 |           0.849257  |          0.962795    |           0.306454 |
| tiny_imagenet | badnet          |       74 |         0.615711 |          0.901155 |           0.96225   |          1.03757     |           0.212932 |
| tiny_imagenet | belt            |       51 |         0.577777 |          0.944166 |           0.967524  |          0.994144    |           0.396553 |
| tiny_imagenet | blend           |       55 |         0.616441 |          0.857542 |           0.885418  |          0.918826    |           0.704936 |
| tiny_imagenet | upgd            |       38 |         0.619901 |          0.511317 |           0.646456  |          0.845523    |           0.88663  |

这张表用于排查 attack-dependent 混杂。它说明不同攻击方法天然处于不同区域：

- `badnet/basic`、`adaptive_patch` 往往更容易形成高迁移、低隐蔽的局部触发器趋势；
- `SIG`、`WaNet`、`blend`、`adaptive_blend` 更容易表现出较高隐蔽性，但迁移性和数据集有关；
- `UPGD` 和 `BELT` 的趋势更特殊，尤其当前 SIG/UPGD 还不是 clean-label，后续需要单独修正和分析。

因此汇报时不要说“所有攻击都一致”。更准确的说法是：

```text
完整基线整体存在 transfer-stealth tradeoff，但该关系明显 attack-dependent。
```

## 9. 四个检测方法分别贡献了什么

| dataset       | arch_base   | defense   |   n_rows |   tpr_mean |   tpr_median |   stealth_mean |   stealth_median |   auc_mean |   auc_median |
|:--------------|:------------|:----------|---------:|-----------:|-------------:|---------------:|-----------------:|-----------:|-------------:|
| cifar10       | ResNet18    | ibd_psc   |      159 |   0.47649  |    0.41125   |       0.52351  |         0.58875  |   0.758028 |     0.817241 |
| cifar10       | ResNet18    | scaleup   |      159 |   0.657682 |    0.668106  |       0.342318 |         0.331894 |   0.771165 |     0.793995 |
| cifar10       | ResNet18    | sentinet  |      159 |   0.425836 |    0.156     |       0.574164 |         0.844    |   0.688242 |     0.617932 |
| cifar10       | ResNet18    | strip     |      159 |   0.205078 |    0.1295    |       0.794922 |         0.8705   |   0.632137 |     0.638938 |
| cifar10       | mobilenetv2 | ibd_psc   |      151 |   0.566623 |    0.624625  |       0.433377 |         0.375375 |   0.705955 |     0.724994 |
| cifar10       | mobilenetv2 | scaleup   |      151 |   0.677485 |    0.721527  |       0.322515 |         0.278473 |   0.776107 |     0.81129  |
| cifar10       | mobilenetv2 | sentinet  |      151 |   0.407868 |    0.2125    |       0.592132 |         0.7875   |   0.715142 |     0.636262 |
| cifar10       | mobilenetv2 | strip     |      151 |   0.2788   |    0.11675   |       0.7212   |         0.88325  |   0.57548  |     0.546285 |
| cifar10       | vgg19_bn    | ibd_psc   |      157 |   0.613154 |    0.627125  |       0.386846 |         0.372875 |   0.62255  |     0.618744 |
| cifar10       | vgg19_bn    | scaleup   |      157 |   0.568854 |    0.582586  |       0.431146 |         0.417414 |   0.725713 |     0.740709 |
| cifar10       | vgg19_bn    | sentinet  |      157 |   0.44751  |    0.2515    |       0.55249  |         0.7485   |   0.696156 |     0.666521 |
| cifar10       | vgg19_bn    | strip     |      157 |   0.214686 |    0.10575   |       0.785314 |         0.89425  |   0.617682 |     0.579118 |
| mnistm        | ResNet18    | ibd_psc   |      145 |   0.650979 |    0.624125  |       0.349021 |         0.375875 |   0.621058 |     0.57933  |
| mnistm        | ResNet18    | scaleup   |      145 |   0.284846 |    0.196142  |       0.715154 |         0.803858 |   0.546169 |     0.532099 |
| mnistm        | ResNet18    | sentinet  |      145 |   0.328179 |    0.057     |       0.671821 |         0.943    |   0.579314 |     0.434952 |
| mnistm        | ResNet18    | strip     |      145 |   0.335555 |    0.254875  |       0.664445 |         0.745125 |   0.678828 |     0.730836 |
| mnistm        | mobilenetv2 | ibd_psc   |      132 |   0.499443 |    0.345188  |       0.500557 |         0.654812 |   0.622734 |     0.569823 |
| mnistm        | mobilenetv2 | scaleup   |      132 |   0.214894 |    0.177035  |       0.785106 |         0.822965 |   0.431026 |     0.420903 |
| mnistm        | mobilenetv2 | sentinet  |      132 |   0.288822 |    0.0685    |       0.711178 |         0.9315   |   0.55224  |     0.448599 |
| mnistm        | mobilenetv2 | strip     |      132 |   0.227727 |    0.0531875 |       0.772273 |         0.946813 |   0.526221 |     0.454443 |
| mnistm        | vgg19_bn    | ibd_psc   |      137 |   0.694104 |    0.731     |       0.305896 |         0.269    |   0.527509 |     0.472607 |
| mnistm        | vgg19_bn    | scaleup   |      137 |   0.28774  |    0.187993  |       0.71226  |         0.812007 |   0.52665  |     0.492353 |
| mnistm        | vgg19_bn    | sentinet  |      137 |   0.318212 |    0.1395    |       0.681788 |         0.8605   |   0.609885 |     0.505587 |
| mnistm        | vgg19_bn    | strip     |      137 |   0.248493 |    0.130625  |       0.751507 |         0.869375 |   0.613444 |     0.613238 |
| tiny_imagenet | ResNet18    | ibd_psc   |      147 |   0.453218 |    0.347     |       0.546782 |         0.653    |   0.75399  |     0.772763 |
| tiny_imagenet | ResNet18    | scaleup   |      147 |   0.24564  |    0.139511  |       0.75436  |         0.860489 |   0.667627 |     0.648507 |
| tiny_imagenet | ResNet18    | sentinet  |      147 |   0.447211 |    0.1575    |       0.552789 |         0.8425   |   0.694497 |     0.582266 |
| tiny_imagenet | ResNet18    | strip     |      147 |   0.459486 |    0.438625  |       0.540514 |         0.561375 |   0.787428 |     0.846443 |
| tiny_imagenet | mobilenetv2 | ibd_psc   |      140 |   0.585533 |    0.722125  |       0.414467 |         0.277875 |   0.769442 |     0.869623 |
| tiny_imagenet | mobilenetv2 | scaleup   |      140 |   0.265037 |    0.14461   |       0.734963 |         0.85539  |   0.667603 |     0.634895 |
| tiny_imagenet | mobilenetv2 | sentinet  |      140 |   0.468957 |    0.149     |       0.531043 |         0.851    |   0.705708 |     0.569577 |
| tiny_imagenet | mobilenetv2 | strip     |      140 |   0.406512 |    0.386563  |       0.593488 |         0.613438 |   0.755067 |     0.780247 |
| tiny_imagenet | vgg19_bn    | ibd_psc   |      131 |   0.587828 |    0.660375  |       0.412172 |         0.339625 |   0.759825 |     0.817002 |
| tiny_imagenet | vgg19_bn    | scaleup   |      131 |   0.232577 |    0.137168  |       0.767423 |         0.862832 |   0.637559 |     0.604668 |
| tiny_imagenet | vgg19_bn    | sentinet  |      131 |   0.40405  |    0.098     |       0.59595  |         0.902    |   0.665291 |     0.499976 |
| tiny_imagenet | vgg19_bn    | strip     |      131 |   0.387718 |    0.300375  |       0.612282 |         0.699625 |   0.747184 |     0.76473  |

`stealth_avg` 是四个检测方法的平均值，所以必须拆开看每个 defense。否则可能误以为所有检测方法都同向变化。

汇报时可以这样讲：

- `STRIP` 和 `SentiNet` 在一些数据集上 stealth 很高，说明它们对部分攻击不敏感；
- `ScaleUp` 和 `IBD-PSC` 对模型结构、输入分布和触发器类型更敏感；
- 如果某个实验里的 `stealth_avg` 变化很大，需要回到 defense breakdown 判断到底是哪个检测器驱动的。

## 10. 回归结果怎么解释

- `transfer_rate:difficulty` 系数：`-0.6084`，p 值：`5.265e-09`。
- 在完整旧 baseline 数据里，该交互项表示 difficulty 越高时，transfer-stealth 线性斜率越更负。

完整回归表保存于 `analysis-transfer-asr2/baseline_full_analysis/baseline_full_regression.txt`。

这个回归结果支持：difficulty 可能会调节 transfer-stealth 的线性斜率。但它是 pooled regression，混合了 dataset、arch、attack、poison_rate，因此不能写成“ACC 是唯一因果原因”。

更稳妥的表述是：

```text
完整旧基线在回归和分层层面都支持 difficulty 与 transfer-stealth 关系有关；
但该影响与数据集、模型架构和攻击类型混杂，需要结合噪声实验做更接近控制变量的补充验证。
```

## 11. 和噪声实验、新模型实验的关系

- 这份报告提供的是旧完整 baseline 的总体背景：ResNet18 / MobileNetV2 / VGG19-BN 在 CIFAR-10、Tiny-ImageNet、MNIST-M 上的完整结果。
- 噪声实验更像固定 `CIFAR-10 + SmallCNN` 后改变输入难度的控制变量实验。
- 新模型实验更像对 baseline 的模型体系补充：`SmallCNN -> CIFAR-10`、`ResNet34 -> Tiny-ImageNet`。
- 三者合起来更适合形成证据链：旧完整 baseline 说明大范围现象，新模型补充说明模型替换方向，噪声实验说明同模型下 ACC 变化的影响。

建议写作顺序是：

```text
完整旧基线：证明大范围 tradeoff 存在，且随 dataset/ACC/attack 改变。
噪声实验：固定 SmallCNN 和 CIFAR-10，主动改变 ACC，验证 difficulty effect。
模型架构实验：补充 SmallCNN 和 ResNet34，观察换模型后关系是否仍成立。
```

## 12. 当前结果能支持的表述

比较稳妥、准确的表述：

```text
完整旧基线显示，在多数据集、多模型、多攻击的原始实验中，迁移性和隐蔽性整体呈负相关。
这种关系在 CIFAR-10 上最明显，在 Tiny-ImageNet 和 MNIST-M 上也存在，但强度不同。
ACC 分层、回归和分箱趋势图都说明，任务难度会改变 tradeoff 的强弱。
不过该关系不是简单线性关系，而是明显受到数据集、攻击类型和检测方法影响。
```

不建议直接写成：

```text
散点图显示明显线性负相关。
ACC 是唯一导致 tradeoff 变化的原因。
所有攻击方法都有一致规律。
```

## 13. 当前结果的限制

1. `transfer_rate` 大量集中在 1 附近，并存在少数极端值，普通散点图不适合作为唯一证据。
2. 完整旧基线混合了不同数据集、模型和攻击，不能单独归因 ACC。
3. `SIG` 和 `UPGD` 当前仍是 dirty-label / label-flipping 版本，不是 clean-label。
4. `stealth_avg` 是四个检测器平均值，必须结合 defense breakdown 解释。
5. 最终论文结论应同时报告 `source_asr>=0.05` 和 `source_asr>=0.10` 的敏感性结果。

## 14. 后续建议补充实验和分析

1. 修正并单独重跑 clean-label SIG / UPGD。
2. 对基线、噪声、新模型都补充 attack-family 分组图。
3. 对 `transfer_rate` 增加 log / winsorize / source_asr>=0.10 的稳健性分析。
4. 把四个 defense 的趋势作为附录，不只报告 `stealth_avg`。
5. 汇报图优先使用分箱中位数趋势图和 rank 分箱趋势图，而不是普通散点图。

## 15. 建议阅读顺序

为了快速理解完整旧基线，建议按这个顺序看：

1. `BASELINE_FULL_ACC_TRANSFER_STEALTH_REPORT_CN.md`：先读本文档，把握主结论。
2. `baseline_full_plot_diagnostics.csv`：理解为什么普通散点图不明显。
3. `figures/baseline_binned_median_trend_by_dataset.png`：看最适合汇报的分箱趋势图。
4. `figures/baseline_rank_binned_trend_by_dataset.png`：看 Spearman 排序关系。
5. `baseline_full_acc_bins.csv`：看 ACC 分层相关性。
6. `baseline_full_summary_by_attack.csv`：看攻击类型差异。
7. `baseline_full_defense_breakdown.csv`：看四个检测方法贡献。
8. `baseline_full_regression.txt`：看完整回归统计。

## 16. 输出文件作用

- `baseline_full_acc_transfer_stealth_rows.csv`：逐目录明细，是所有 baseline-only 表格和回归的基础。
- `baseline_full_summary_by_dataset_arch.csv`：按数据集和模型汇总。
- `baseline_full_summary_by_attack.csv`：按攻击类型汇总。
- `baseline_full_correlations.csv`：整体、数据集、模型、攻击类型的相关性。
- `baseline_full_acc_bins.csv`：ACC 分层结果。
- `baseline_full_defense_breakdown.csv`：四个检测器拆解。
- `baseline_full_regression.txt`：完整线性回归输出。
- `figures/baseline_transfer_vs_stealth_by_dataset_clipped.png`：按 dataset 上色的裁剪散点图。
- `figures/baseline_transfer_vs_stealth_by_acc_bin_clipped.png`：按 ACC bin 上色的裁剪散点图。
- `figures/baseline_transfer_vs_stealth_dataset_facets_clipped.png`：按 dataset 分面的裁剪散点图。
- `figures/baseline_binned_median_trend_by_dataset.png`：按 transfer_rate 分箱后的中位数趋势图。
- `figures/baseline_rank_binned_trend_by_dataset.png`：按 transfer_rate rank 分箱后的趋势图。
- `baseline_full_plot_diagnostics.csv`：基线图分布诊断表，用于解释为什么普通散点图趋势不明显。
- `baseline_full_binned_median_trend_by_dataset.csv`：分箱中位数趋势表。
- `baseline_full_rank_binned_trend_by_dataset.csv`：rank 分箱趋势表。

## 17. 一句话总结

```text
完整旧基线证明 transfer-stealth tradeoff 在原有实验中整体存在，但它不是简单线性斜线；
该关系会随数据集、ACC 区间、攻击类型和检测器改变，最适合用分组相关性和分箱趋势图汇报。
```
