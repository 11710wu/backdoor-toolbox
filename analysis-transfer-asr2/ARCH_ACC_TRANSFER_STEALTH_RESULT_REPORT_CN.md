# 新增架构模型对 ACC、迁移性与隐蔽性关系的影响

## 0. 先看结论

这份报告只聚焦新增的两个模型：`CIFAR-10 SmallCNN` 和 `Tiny-ImageNet ResNet34`。核心问题是：新增模型造成 ACC 变化后，相对已有 baseline，迁移性和隐蔽性的关系发生了什么变化。

本轮按你的问题重新生成，主表和主图都排除 `SIG` 与 `upgd`。因此这里看的不是完整攻击集合，而是去掉这两个强驱动方法后的架构对照。

最重要的结论是：

- **SmallCNN 让 CIFAR-10 的 ACC 下降，但迁移性上升、隐蔽性下降。**相对 CIFAR-10 ResNet18，同配置 pairwise delta 为：clean ACC `-0.0650`，transfer_rate `+0.1609`，stealth_avg `-0.2603`。
- **SmallCNN 还让 transfer-stealth 负相关明显变强。**在 SmallCNN - ResNet18 的同配置子集里，Spearman 从 `-0.4583` 变为 `-0.8939`，Pearson 从 `-0.4622` 变为 `-0.9559`。这说明不是只有均值变了，迁移性和隐蔽性的 tradeoff 也更陡了。
- SmallCNN 相对 MobileNetV2 / VGG19-BN 也呈现同样方向：transfer_rate 分别 `+0.0847` / `+0.0694`，stealth_avg 分别 `-0.1146` / `-0.2217`。这说明 SmallCNN 补充实验的主要现象是：**更弱模型更容易迁移，但更不隐蔽**。
- **ResNet34 让 Tiny-ImageNet 的 ACC 上升，但去掉 SIG/UPGD 后，迁移性提升不再稳健，隐蔽性仍下降。**相对 Tiny-ImageNet ResNet18，同配置 pairwise delta 为：clean ACC `+0.0220`，transfer_rate `-0.0111`，stealth_avg `-0.0825`。
- ResNet34 - ResNet18 的 transfer-stealth 相关性变化较小：Spearman 从 `-0.8901` 变为 `-0.8242`，Pearson 从 `-0.7181` 变为 `-0.6328`。所以 ResNet34 更适合讲“迁移性提升不稳、隐蔽性不改善”，不适合讲明显加强 tradeoff。
- ResNet34 相对 MobileNetV2 / VGG19-BN 的 transfer_rate 只剩接近 0 的小幅变化 `+0.0093` / `+0.0108`，stealth_avg 分别 `-0.0143` / `-0.0996`。这说明之前 ResNet34 的“迁移性上升”很大一部分来自 SIG/UPGD，不能把它写成所有攻击上的普遍架构规律。

一句话总结：去掉 SIG/UPGD 后，SmallCNN 仍然稳定表现为“ACC 降低、迁移性上升、隐蔽性下降”；ResNet34 不再支持“迁移性明显上升”，更像是“ACC 提高，但迁移性接近不变，隐蔽性不改善甚至下降”。架构影响需要按攻击类型拆开看。

## 1. 最应该看的图

| 优先级 | 图 | 重点看什么 | 汇报时怎么说 |
|---:|---|---|---|
| 1 | `arch_pairwise_delta_summary.png` | 排除 SIG/UPGD 后，新模型减 baseline 的 ACC、transfer_rate、stealth_avg 方向 | 这是架构补充实验最核心的图 |
| 2 | `arch_pairwise_relationship_shift.png` | transfer-stealth 相关性在 baseline 和新模型之间怎么变 | 回答“关系本身有没有变强/变弱” |
| 3 | `arch_acc_correlation_shift.png` | ACC 与 transfer/stealth 的相关性怎么变 | 回答“ACC 怎么影响二者关系” |
| 4 | `arch_metric_overview.png` | 每个 dataset/arch 的三项均值水平 | 用它说明新增模型落在哪个位置 |
| 5 | `arch_transfer_vs_stealth_facets.png` | 新模型在 transfer-stealth 平面上的分布 | 看新增模型是否移动到高迁移/低隐蔽区域 |
| 6 | `arch_attack_heatmap.png` | 哪些攻击驱动变化 | 说明现象 attack-dependent |
| 7 | `arch_defense_heatmap.png` | 哪些检测器驱动 stealth_avg 变化 | 说明隐蔽性下降来自检测器响应 |

![Pairwise delta summary](arch_acc_analysis/figures/arch_pairwise_delta_summary.png)

![Pairwise relationship shift](arch_acc_analysis/figures/arch_pairwise_relationship_shift.png)

![ACC correlation shift](arch_acc_analysis/figures/arch_acc_correlation_shift.png)

![Architecture metric overview](arch_acc_analysis/figures/arch_metric_overview.png)

![Transfer-stealth facets](arch_acc_analysis/figures/arch_transfer_vs_stealth_facets.png)

![Attack heatmap](arch_acc_analysis/figures/arch_attack_heatmap.png)

![Defense heatmap](arch_acc_analysis/figures/arch_defense_heatmap.png)

HTML 展示页同步更新：`analysis-transfer-asr2/ARCH_ACC_TRANSFER_STEALTH_DASHBOARD_CN.html`。

## 2. 分析口径

新增模型和 baseline 的对照关系：

- `CIFAR-10 SmallCNN` 对照已有 `CIFAR-10 ResNet18 / MobileNetV2 / VGG19-BN`。
- `Tiny-ImageNet ResNet34` 对照已有 `Tiny-ImageNet ResNet18 / MobileNetV2 / VGG19-BN`。
- 本轮主结果排除 `SIG` 和 `upgd`；如果要讲完整攻击集合，需要单独说明 SIG/UPGD 会显著抬高 ResNet34 的 transfer_rate。

指标定义：

```text
difficulty = 1 - clean_acc
transfer_rate = transfer_asr^2 / source_asr
stealth_avg = mean(1 - TPR)
```

这里最关键的是同配置 pairwise delta。它尽量固定 attack、poison_rate、strength、cover_rate，只看把模型从 baseline 换成新增模型后，三项指标怎么变。整体均值只作为背景，pairwise delta 才是主证据。

数据规模：

- 主分析行数：`750`
- baseline_full 主分析行数：`690`
- new_model_supplement 主分析行数：`60`
- pairwise 可匹配行数：`101`
- transfer_rate 中位数：`0.9616`

## 3. CIFAR-10 SmallCNN：ACC 降低后，迁移性升高、隐蔽性降低

整体均值对比：

- SmallCNN：clean ACC=`0.8694`，transfer_rate=`0.7111`，stealth_avg=`0.3836`。
- ResNet18：clean ACC=`0.9478`，transfer_rate=`0.5376`，stealth_avg=`0.5406`。

同配置 pairwise 结果：

| 对照 | Δ clean ACC | Δ transfer_rate | Δ stealth_avg | 解读 |
|---|---:|---:|---:|---|
| SmallCNN - ResNet18 | -0.0650 | +0.1609 | -0.2603 | ACC 降低，迁移性上升，隐蔽性下降 |
| SmallCNN - MobileNetV2 | -0.0173 | +0.0847 | -0.1146 | 同方向，说明不是只相对 ResNet18 成立 |
| SmallCNN - VGG19-BN | -0.0483 | +0.0694 | -0.2217 | 同方向，支持 SmallCNN 高迁移低隐蔽趋势 |

这个结果可以这样讲：SmallCNN 降低了源模型的分类能力，但后门在迁移域上更容易保持效果；与此同时，四个源域检测方法平均更容易检出，所以 stealth_avg 下降。它体现的是“更容易迁移，但更不隐蔽”的 tradeoff。

## 4. Tiny-ImageNet ResNet34：ACC 提高后，迁移性提升基本消失，隐蔽性不改善

整体均值对比：

- ResNet34：clean ACC=`0.6605`，transfer_rate=`0.9161`，stealth_avg=`0.6089`。
- ResNet18：clean ACC=`0.6378`，transfer_rate=`0.9750`，stealth_avg=`0.5409`。

同配置 pairwise 结果：

| 对照 | Δ clean ACC | Δ transfer_rate | Δ stealth_avg | 解读 |
|---|---:|---:|---:|---|
| ResNet34 - ResNet18 | +0.0220 | -0.0111 | -0.0825 | ACC 提高，迁移性没有上升，隐蔽性下降 |
| ResNet34 - MobileNetV2 | +0.0789 | +0.0093 | -0.0143 | ACC 提高，迁移性接近不变，隐蔽性小幅下降 |
| ResNet34 - VGG19-BN | +0.0511 | +0.0108 | -0.0996 | ACC 提高，迁移性仅小幅变化，隐蔽性下降 |

这个结果很重要：ResNet34 的 ACC 提高了，但 transfer_rate 没有明显上升，stealth_avg 也没有同步提高。这说明去掉 SIG/UPGD 后，ResNet34 不再是“高 ACC 也高迁移”的强证据，而更适合用来说明：更深模型会改变检测器响应，但 ACC 不能单独决定迁移性或隐蔽性。

## 5. 总表：Dataset / Arch 均值

| dataset       | arch_base   | result_group         |   n_rows |   clean_acc_mean |   difficulty_mean |   source_asr_mean |   transfer_asr_mean |   transfer_rate_mean |   transfer_rate_median |   stealth_avg_mean |   stealth_avg_median |
|:--------------|:------------|:---------------------|---------:|-----------------:|------------------:|------------------:|--------------------:|---------------------:|-----------------------:|-------------------:|---------------------:|
| cifar10       | ResNet18    | baseline_full        |      116 |         0.947786 |         0.0522144 |          0.820424 |            0.612749 |             0.537637 |               0.505237 |           0.540573 |             0.594183 |
| cifar10       | mobilenetv2 | baseline_full        |      109 |         0.901485 |         0.0985149 |          0.794126 |            0.656042 |             0.645044 |               0.730705 |           0.45725  |             0.539091 |
| cifar10       | vgg19_bn    | baseline_full        |      116 |         0.92845  |         0.0715496 |          0.758473 |            0.628889 |             0.624682 |               0.596504 |           0.506196 |             0.555007 |
| cifar10       | SmallCNN    | new_model_supplement |       32 |         0.869387 |         0.130613  |          0.77237  |            0.720874 |             0.711051 |               0.920374 |           0.383561 |             0.449729 |
| tiny_imagenet | ResNet18    | baseline_full        |      122 |         0.637835 |         0.362165  |          0.7958   |            0.872679 |             0.974981 |               1.00289  |           0.54091  |             0.529725 |
| tiny_imagenet | mobilenetv2 | baseline_full        |      119 |         0.58178  |         0.41822   |          0.824468 |            0.866332 |             0.930248 |               0.998013 |           0.505976 |             0.511949 |
| tiny_imagenet | vgg19_bn    | baseline_full        |      108 |         0.606867 |         0.393133  |          0.772817 |            0.83813  |             0.948098 |               1.00151  |           0.541479 |             0.593086 |
| tiny_imagenet | ResNet34    | new_model_supplement |       28 |         0.660451 |         0.339549  |          0.706647 |            0.791583 |             0.916136 |               1.00057  |           0.608905 |             0.649417 |

这张表用于看每个模型的绝对水平。汇报时不要只读均值，重点还是下一张 pairwise delta 表，因为 pairwise 更接近公平对照。

## 6. 主证据：同配置 Pairwise Delta

| dataset       | new_arch   | base_arch   |   n_rows |   delta_clean_acc_mean |   delta_transfer_rate_mean |   delta_stealth_avg_mean |   delta_source_asr_mean |   delta_transfer_asr_mean |
|:--------------|:-----------|:------------|---------:|-----------------------:|---------------------------:|-------------------------:|------------------------:|--------------------------:|
| cifar10       | SmallCNN   | ResNet18    |       24 |             -0.065026  |                  0.160874  |               -0.260306  |              0.0829742  |                 0.152962  |
| cifar10       | SmallCNN   | mobilenetv2 |       20 |             -0.0172562 |                  0.0847266 |               -0.114642  |              0.0948075  |                 0.0940157 |
| cifar10       | SmallCNN   | vgg19_bn    |       22 |             -0.0483295 |                  0.0693759 |               -0.221655  |              0.137899   |                 0.114212  |
| tiny_imagenet | ResNet34   | ResNet18    |       14 |              0.0219554 |                 -0.0110906 |               -0.0825063 |              0.0416532  |                 0.0184854 |
| tiny_imagenet | ResNet34   | mobilenetv2 |       12 |              0.0789062 |                  0.0092707 |               -0.0142541 |              0.00792305 |                 0.0225294 |
| tiny_imagenet | ResNet34   | vgg19_bn    |        9 |              0.0510972 |                  0.010806  |               -0.099608  |              0.0817773  |                 0.06689   |

读这张表时只看三个 delta：

- `delta_clean_acc_mean`：新增模型是否改变 ACC。
- `delta_transfer_rate_mean`：迁移性是否上升。
- `delta_stealth_avg_mean`：隐蔽性是否上升。负值表示更容易被检测。

当前最值得汇报的现象是：SmallCNN 的高迁移、低隐蔽非常稳定；ResNet34 在排除 SIG/UPGD 后，transfer_rate 提升基本消失，但 stealth_avg 仍不改善。这正好解释了为什么需要单独拿掉 SIG/UPGD 做敏感性检查。

## 7. 相关性变化：ACC 怎么影响迁移性与隐蔽性的关系

这里回答你关心的第二层问题：不是只看均值升降，而是看 `transfer_rate` 和 `stealth_avg` 的关系本身有没有因为换模型而改变。

| dataset       | new_arch   | base_arch   |   n_rows |   base_spearman_transfer_stealth |   new_spearman_transfer_stealth |   delta_spearman_transfer_stealth |   base_pearson_transfer_stealth |   new_pearson_transfer_stealth |   delta_pearson_transfer_stealth |   base_spearman_acc_transfer |   new_spearman_acc_transfer |   base_spearman_acc_stealth |   new_spearman_acc_stealth |
|:--------------|:-----------|:------------|---------:|---------------------------------:|--------------------------------:|----------------------------------:|--------------------------------:|-------------------------------:|---------------------------------:|-----------------------------:|----------------------------:|----------------------------:|---------------------------:|
| cifar10       | SmallCNN   | ResNet18    |       24 |                       -0.458261  |                       -0.893913 |                        -0.435652  |                       -0.462163 |                      -0.955922 |                       -0.493759  |                   0.0439896  |                  -0.0556522 |                  0.29007    |                 0.00608696 |
| cifar10       | SmallCNN   | mobilenetv2 |       20 |                       -0.811748  |                       -0.828571 |                        -0.0168235 |                       -0.818958 |                      -0.951379 |                       -0.132421  |                   0.314243   |                  -0.275188  |                 -0.368698   |                 0.231579   |
| cifar10       | SmallCNN   | vgg19_bn    |       22 |                       -0.734256  |                       -0.867871 |                        -0.133615  |                       -0.651187 |                      -0.952638 |                       -0.301451  |                   0.00170117 |                  -0.145116  |                  0.00736967 |                 0.0965556  |
| tiny_imagenet | ResNet34   | ResNet18    |       14 |                       -0.89011   |                       -0.824176 |                         0.0659341 |                       -0.718089 |                      -0.632769 |                        0.0853198 |                  -0.167217   |                  -0.235165  |                 -0.0330033  |                -0.0989011  |
| tiny_imagenet | ResNet34   | mobilenetv2 |       12 |                       -0.573427  |                       -0.86014  |                        -0.286713  |                       -0.439698 |                      -0.7172   |                       -0.277503  |                  -0.0699301  |                  -0.0909091 |                  0.237762   |                -0.188811   |
| tiny_imagenet | ResNet34   | vgg19_bn    |        9 |                       -0.0333333 |                       -0.85     |                        -0.816667  |                       -0.24936  |                      -0.742323 |                       -0.492962  |                   0.418414   |                  -0.116667  |                 -0.694567   |                -0.133333   |

重点读法：

- **SmallCNN - ResNet18：transfer-stealth 负相关明显增强。**同配置子集里，Spearman 从 `-0.4583` 变为 `-0.8939`，变化 `-0.4357`；Pearson 从 `-0.4622` 变为 `-0.9559`，变化 `-0.4938`。这说明 SmallCNN 不只是把平均 transfer_rate 拉高、stealth_avg 拉低，还让“迁移越强、隐蔽越差”的关系更陡。
- **SmallCNN 中 ACC 对两个单独指标的直接相关不强。**同配置子集里，Spearman(ACC, transfer_rate) 从 ResNet18 `0.0440` 到 SmallCNN `-0.0557`；Spearman(ACC, stealth_avg) 从 ResNet18 `0.2901` 到 SmallCNN `0.0061`。所以更准确的解释不是“ACC 单独决定 transfer 或 stealth”，而是 **ACC 下降伴随模型结构变弱，使 transfer-stealth tradeoff 更明显**。
- **ResNet34 - ResNet18：相关性没有明显变强。**Spearman 从 `-0.8901` 变为 `-0.8242`，变化 `+0.0659`；Pearson 从 `-0.7181` 变为 `-0.6328`，变化 `+0.0853`。这支持前面的判断：ResNet34 的 ACC 提高没有让迁移性更强，也没有让 tradeoff 更陡。

这一节的结论可以浓缩为一句：**ACC 不是稳定的直接解释变量；在 SmallCNN 这种模型能力明显下降的场景里，ACC 变化伴随架构变化，会让 transfer-stealth 的负相关变强。ResNet34 则说明 ACC 提高不一定强化或缓解这个关系。**

## 8. 攻击类型和检测器拆解

如果老师追问“为什么 stealth 下降”，看这两张表和对应 heatmap。

攻击类型分层：

| dataset       | arch_base   | attack_family   |   n_rows |   clean_acc_mean |   transfer_rate_mean |   stealth_avg_mean |   transfer_minus_stealth_gap |
|:--------------|:------------|:----------------|---------:|-----------------:|---------------------:|-------------------:|-----------------------------:|
| cifar10       | ResNet18    | WaNet           |       23 |         0.95169  |            0.132003  |          0.72167   |                   -0.589667  |
| cifar10       | SmallCNN    | WaNet           |        5 |         0.886725 |            0.0165578 |          0.765062  |                   -0.748504  |
| cifar10       | mobilenetv2 | WaNet           |       21 |         0.901905 |            0.0607501 |          0.658902  |                   -0.598152  |
| cifar10       | vgg19_bn    | WaNet           |       23 |         0.933908 |            0.108155  |          0.682336  |                   -0.574182  |
| cifar10       | ResNet18    | adaptive_blend  |       16 |         0.951305 |            0.585359  |          0.725054  |                   -0.139695  |
| cifar10       | SmallCNN    | adaptive_blend  |        4 |         0.887969 |            0.457476  |          0.630849  |                   -0.173373  |
| cifar10       | mobilenetv2 | adaptive_blend  |       15 |         0.904492 |            0.581476  |          0.606391  |                   -0.024915  |
| cifar10       | vgg19_bn    | adaptive_blend  |       17 |         0.929412 |            0.614111  |          0.685432  |                   -0.071321  |
| cifar10       | ResNet18    | adaptive_patch  |       15 |         0.950258 |            0.50714   |          0.601508  |                   -0.0943682 |
| cifar10       | SmallCNN    | adaptive_patch  |        6 |         0.886312 |            1.00375   |          0.0137024 |                    0.99005   |
| cifar10       | mobilenetv2 | adaptive_patch  |       15 |         0.905208 |            0.908823  |          0.47222   |                    0.436603  |
| cifar10       | vgg19_bn    | adaptive_patch  |       15 |         0.933975 |            1.04856   |          0.571994  |                    0.476565  |
| cifar10       | ResNet18    | badnet          |       27 |         0.949532 |            0.843701  |          0.220616  |                    0.623085  |
| cifar10       | SmallCNN    | badnet          |        6 |         0.888333 |            0.973656  |          0.0629609 |                    0.910695  |
| cifar10       | mobilenetv2 | badnet          |       24 |         0.905417 |            0.992117  |          0.0614106 |                    0.930707  |
| cifar10       | vgg19_bn    | badnet          |       26 |         0.933688 |            0.956293  |          0.125116  |                    0.831177  |
| cifar10       | ResNet18    | belt            |       18 |         0.929639 |            0.366128  |          0.446755  |                   -0.0806262 |
| cifar10       | SmallCNN    | belt            |        6 |         0.790104 |            1.0586    |          0.446342  |                    0.612262  |
| cifar10       | mobilenetv2 | belt            |       18 |         0.886785 |            0.661743  |          0.527546  |                    0.134197  |
| cifar10       | vgg19_bn    | belt            |       18 |         0.902965 |            0.374789  |          0.514816  |                   -0.140027  |
| cifar10       | ResNet18    | blend           |       17 |         0.953449 |            0.763924  |          0.675666  |                    0.0882582 |
| cifar10       | SmallCNN    | blend           |        5 |         0.889275 |            0.524973  |          0.557444  |                   -0.0324717 |
| cifar10       | mobilenetv2 | blend           |       16 |         0.905266 |            0.684834  |          0.553406  |                    0.131429  |
| cifar10       | vgg19_bn    | blend           |       17 |         0.934206 |            0.717494  |          0.604295  |                    0.113199  |
| tiny_imagenet | ResNet18    | WaNet           |       24 |         0.642974 |            0.966143  |          0.793461  |                    0.172681  |
| tiny_imagenet | ResNet34    | WaNet           |        3 |         0.669375 |            0.692894  |          0.927891  |                   -0.234996  |
| tiny_imagenet | mobilenetv2 | WaNet           |       24 |         0.58551  |            0.910995  |          0.809753  |                    0.101242  |
| tiny_imagenet | vgg19_bn    | WaNet           |       22 |         0.615068 |            0.947742  |          0.800388  |                    0.147355  |
| tiny_imagenet | ResNet18    | adaptive_blend  |       19 |         0.640309 |            0.838727  |          0.772764  |                    0.0659632 |
| tiny_imagenet | ResNet34    | adaptive_blend  |        5 |         0.667325 |            0.794045  |          0.830124  |                   -0.0360794 |
| tiny_imagenet | mobilenetv2 | adaptive_blend  |       18 |         0.583181 |            0.812802  |          0.736933  |                    0.0758694 |
| tiny_imagenet | vgg19_bn    | adaptive_blend  |       17 |         0.610838 |            0.83711   |          0.712111  |                    0.124999  |
| tiny_imagenet | ResNet18    | adaptive_patch  |       15 |         0.641158 |            1.02729   |          0.365134  |                    0.662151  |
| tiny_imagenet | ResNet34    | adaptive_patch  |        5 |         0.670575 |            0.907132  |          0.343459  |                    0.563673  |
| tiny_imagenet | mobilenetv2 | adaptive_patch  |       15 |         0.583333 |            0.921402  |          0.232869  |                    0.688533  |
| tiny_imagenet | vgg19_bn    | adaptive_patch  |       15 |         0.615067 |            0.939697  |          0.321359  |                    0.618338  |
| tiny_imagenet | ResNet18    | badnet          |       27 |         0.646208 |            1.07175   |          0.227551  |                    0.844195  |
| tiny_imagenet | ResNet34    | badnet          |        4 |         0.667031 |            1.07292   |          0.206398  |                    0.866526  |
| tiny_imagenet | mobilenetv2 | badnet          |       26 |         0.586202 |            1.01502   |          0.209819  |                    0.805197  |
| tiny_imagenet | vgg19_bn    | badnet          |       21 |         0.613036 |            1.02157   |          0.197991  |                    0.823579  |
| tiny_imagenet | ResNet18    | belt            |       18 |         0.6055   |            1.00785   |          0.340867  |                    0.666987  |
| tiny_imagenet | ResNet34    | belt            |        6 |         0.629437 |            1.03866   |          0.629386  |                    0.409271  |
| tiny_imagenet | mobilenetv2 | belt            |       17 |         0.560279 |            0.999032  |          0.322453  |                    0.676579  |
| tiny_imagenet | vgg19_bn    | belt            |       16 |         0.56518  |            0.973529  |          0.53793   |                    0.435598  |
| tiny_imagenet | ResNet18    | blend           |       19 |         0.64498  |            0.91246   |          0.763626  |                    0.148833  |
| tiny_imagenet | ResNet34    | blend           |        5 |         0.67005  |            0.908722  |          0.759171  |                    0.149551  |
| tiny_imagenet | mobilenetv2 | blend           |       19 |         0.587704 |            0.895272  |          0.688543  |                    0.206729  |
| tiny_imagenet | vgg19_bn    | blend           |       17 |         0.616662 |            0.952267  |          0.657664  |                    0.294603  |

检测器拆解：

| dataset       | arch_base   | defense   |   n_rows |   tpr_mean |   stealth_mean |   auc_mean |
|:--------------|:------------|:----------|---------:|-----------:|---------------:|-----------:|
| cifar10       | ResNet18    | ibd_psc   |      116 |   0.518907 |       0.481093 |   0.773439 |
| cifar10       | ResNet18    | scaleup   |      116 |   0.655648 |       0.344352 |   0.76944  |
| cifar10       | ResNet18    | sentinet  |      116 |   0.470892 |       0.529108 |   0.709031 |
| cifar10       | ResNet18    | strip     |      116 |   0.192262 |       0.807738 |   0.654792 |
| cifar10       | SmallCNN    | ibd_psc   |       32 |   0.82     |       0.18     |   0.65897  |
| cifar10       | SmallCNN    | scaleup   |       32 |   0.734673 |       0.265327 |   0.812455 |
| cifar10       | SmallCNN    | sentinet  |       32 |   0.486703 |       0.513297 |   0.730432 |
| cifar10       | SmallCNN    | strip     |       32 |   0.424379 |       0.575621 |   0.744234 |
| cifar10       | mobilenetv2 | ibd_psc   |      109 |   0.682339 |       0.317661 |   0.769355 |
| cifar10       | mobilenetv2 | scaleup   |      109 |   0.691234 |       0.308766 |   0.786151 |
| cifar10       | mobilenetv2 | sentinet  |      109 |   0.484284 |       0.515716 |   0.76241  |
| cifar10       | mobilenetv2 | strip     |      109 |   0.313141 |       0.686859 |   0.598889 |
| cifar10       | vgg19_bn    | ibd_psc   |      116 |   0.640032 |       0.359968 |   0.650166 |
| cifar10       | vgg19_bn    | scaleup   |      116 |   0.55894  |       0.44106  |   0.72313  |
| cifar10       | vgg19_bn    | sentinet  |      116 |   0.549534 |       0.450466 |   0.74332  |
| cifar10       | vgg19_bn    | strip     |      116 |   0.226709 |       0.773291 |   0.638174 |
| tiny_imagenet | ResNet18    | ibd_psc   |      122 |   0.504145 |       0.495855 |   0.799402 |
| tiny_imagenet | ResNet18    | scaleup   |      122 |   0.284676 |       0.715324 |   0.695241 |
| tiny_imagenet | ResNet18    | sentinet  |      122 |   0.525902 |       0.474098 |   0.741313 |
| tiny_imagenet | ResNet18    | strip     |      122 |   0.521638 |       0.478362 |   0.827782 |
| tiny_imagenet | ResNet34    | ibd_psc   |       28 |   0.609897 |       0.390103 |   0.745211 |
| tiny_imagenet | ResNet34    | scaleup   |       28 |   0.248683 |       0.751317 |   0.620867 |
| tiny_imagenet | ResNet34    | sentinet  |       28 |   0.3875   |       0.6125   |   0.652018 |
| tiny_imagenet | ResNet34    | strip     |       28 |   0.318299 |       0.681701 |   0.732165 |
| tiny_imagenet | mobilenetv2 | ibd_psc   |      119 |   0.668966 |       0.331034 |   0.825954 |
| tiny_imagenet | mobilenetv2 | scaleup   |      119 |   0.305988 |       0.694012 |   0.699597 |
| tiny_imagenet | mobilenetv2 | sentinet  |      119 |   0.538592 |       0.461408 |   0.746434 |
| tiny_imagenet | mobilenetv2 | strip     |      119 |   0.462547 |       0.537453 |   0.794031 |
| tiny_imagenet | vgg19_bn    | ibd_psc   |      108 |   0.645778 |       0.354222 |   0.827763 |
| tiny_imagenet | vgg19_bn    | scaleup   |      108 |   0.264647 |       0.735353 |   0.665133 |
| tiny_imagenet | vgg19_bn    | sentinet  |      108 |   0.47419  |       0.52581  |   0.70619  |
| tiny_imagenet | vgg19_bn    | strip     |      108 |   0.449468 |       0.550532 |   0.792594 |

解释规则很简单：`stealth_avg` 是四个检测器平均值。如果新增模型的 stealth_avg 下降，需要检查是 SentiNet、STRIP、ScaleUp 还是 IBD-PSC 贡献最大。

## 9. 汇报建议

建议按 5 页讲，不要从复杂回归开始：

1. **新增了什么模型**：CIFAR-10 加 SmallCNN，Tiny-ImageNet 加 ResNet34。
2. **ACC 怎么变**：SmallCNN 降低 ACC；ResNet34 提高 Tiny-ImageNet ACC。
3. **迁移和隐蔽怎么变**：SmallCNN 仍然 transfer_rate 上升、stealth_avg 下降；ResNet34 去掉 SIG/UPGD 后 transfer_rate 接近不变，但 stealth_avg 不改善。
4. **相关性怎么变**：SmallCNN - ResNet18 的 transfer-stealth Spearman 从约 -0.46 变到约 -0.89，说明 tradeoff 明显变陡；ResNet34 - ResNet18 没有这种增强。
5. **怎么解释**：新增架构改变了模型能力和检测器响应，因此改变了迁移性与隐蔽性的 tradeoff；SIG/UPGD 是 ResNet34 原先迁移性上升的重要驱动，ACC 是重要表征但不是唯一原因。

可以直接使用的汇报话术：

```text
这部分架构实验主要看新增模型 SmallCNN 和 ResNet34。
本轮结果排除了 SIG 和 UPGD，主要看去掉这两个强驱动方法后结论是否还稳。
在 CIFAR-10 上，SmallCNN 相对 baseline ACC 更低，但同配置下 transfer_rate 上升、stealth_avg 下降，说明更弱模型让后门更容易迁移，但更容易被检测。
更重要的是，SmallCNN - ResNet18 的同配置相关性显示，transfer_rate 和 stealth_avg 的负相关明显变强，说明 ACC/模型能力变化主要影响二者 tradeoff 的强度。
在 Tiny-ImageNet 上，ResNet34 相对 ResNet18 ACC 更高，但去掉 SIG/UPGD 后 transfer_rate 不再明显上升，stealth_avg 仍下降，说明之前 ResNet34 的迁移性上升主要受攻击类型驱动。
因此新增架构实验的核心发现是：SmallCNN 的架构效应很稳；ResNet34 的迁移性结论对攻击类型敏感。
这个结果支持架构会影响 transfer-stealth tradeoff，但不能把 ACC 写成唯一因果变量。
```

## 10. 当前结论边界

- 可以支持：新增 SmallCNN / ResNet34 相对 baseline 改变了 ACC、transfer_rate 和 stealth_avg 的共同位置。
- 可以支持：SmallCNN 更明显地体现低 ACC、高迁移、低隐蔽。
- 可以支持：去掉 SIG/UPGD 后，ResNet34 说明更高 ACC 不一定带来更高迁移性或更高隐蔽性。
- 不建议声称：ACC 单独决定迁移性或隐蔽性。
- 不建议声称：所有模型都满足同一个单调规律。

## 11. 输出文件说明

- `arch_acc_summary_by_dataset_arch.csv`：每个 dataset/arch 的均值水平。
- `arch_acc_pairwise_delta_summary.csv`：新增模型相对 baseline 的同配置 delta，是本报告最关键的表。
- `arch_pairwise_relationship_shift.csv`：同配置下 baseline 和新增模型的相关性变化，回答 transfer-stealth 关系是否变强。
- `arch_relationship_summary_by_dataset_arch.csv`：每个 dataset/arch 内部的相关性系数。
- `arch_pairwise_delta_summary.png`：最适合汇报的核心图。
- `arch_pairwise_relationship_shift.png`：展示 transfer-stealth 相关性怎么变化。
- `arch_acc_correlation_shift.png`：展示 ACC 与 transfer/stealth 直接相关性怎么变化。
- `arch_metric_overview.png`：各模型 ACC、transfer_rate、stealth_avg 的总览图。
- `arch_transfer_vs_stealth_facets.png`：看新增模型在 transfer-stealth 平面上的位置。
- `arch_attack_heatmap.png`：攻击类型拆解。
- `arch_defense_heatmap.png`：检测器拆解。
