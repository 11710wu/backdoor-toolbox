# 跨数据集迁移性 vs 隐蔽性完整分析报告
本报告统一使用 **防御检测失败程度** 作为主隐蔽性指标：`stealth_tpr_mean = 1 - avg(TPR of STRIP, SCaLe-Up, SentiNet, IBD_PSC)`。值越大表示越难被这四种防御检出。`S_stealth` 仅作为包含 NC 的补充指标，不替代主结论。

## 1. 跨数据集总体概览
| dataset       |   n_configs |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|:--------------|------------:|-----------:|----------------:|-------------------:|
| cifar10       |          72 |     0.7035 |          0.5843 |             0.5765 |
| mnistm        |          72 |     0.5235 |          0.5927 |             0.6799 |
| tiny_imagenet |          66 |     0.6857 |          0.7967 |             0.6126 |
从防御均值看，三个数据集的“最强防御”并不一致：CIFAR-10 上 **SCaLe-Up** 的平均 TPR 最高，MNIST-M 上 **IBD_PSC** 最强，Tiny ImageNet 上也主要是 **IBD_PSC** 更稳定。

### 1.1 各数据集防御平均检测强度
| dataset       | defense   |   tpr_mean |   auc_mean |
|:--------------|:----------|-----------:|-----------:|
| cifar10       | IBD_PSC   |     0.5035 |     0.6645 |
| cifar10       | SCaLe-Up  |     0.5806 |     0.7256 |
| cifar10       | STRIP     |     0.1991 |     0.5865 |
| cifar10       | SentiNet  |     0.4107 |     0.694  |
| mnistm        | IBD_PSC   |     0.5686 |     0.5645 |
| mnistm        | SCaLe-Up  |     0.214  |     0.4573 |
| mnistm        | STRIP     |     0.2128 |     0.5726 |
| mnistm        | SentiNet  |     0.2849 |     0.5566 |
| tiny_imagenet | IBD_PSC   |     0.489  |     0.7361 |
| tiny_imagenet | SCaLe-Up  |     0.2413 |     0.6475 |
| tiny_imagenet | STRIP     |     0.389  |     0.7422 |
| tiny_imagenet | SentiNet  |     0.4312 |     0.6829 |

## 2. cifar10 结果分析
- 该数据集上 `transfer_mean` 与 `stealth_tpr_mean` 的相关系数为 **-0.7046**，说明迁移性与以防御失效定义的隐蔽性整体呈显著负相关。
- `asr_mean` 与 `transfer_mean` 的相关系数为 **0.6559**，说明很多“迁移差”的点仍然与源域后门是否先建立成功高度相关。
- 以 `asr_mean >= 0.3` 为前提、用 `transfer` 与 `stealth` 的调和均值做平衡排序时，当前最优参考配置是 **`SIG` / `resnet18` / poison_rate=`0.05`**。

### 2.1 按攻击方法汇总
| attack_type    |   asr_mean |   transfer_mean |   stealth_tpr_mean |   stealth_auc_mean |
|:---------------|-----------:|----------------:|-------------------:|-------------------:|
| SIG            |     0.6031 |          0.4772 |             0.7436 |             0.4545 |
| WaNet          |     0.6617 |          0.1951 |             0.7    |             0.4149 |
| adaptive_blend |     0.5225 |          0.4844 |             0.7208 |             0.4261 |
| adaptive_patch |     0.4137 |          0.5439 |             0.5486 |             0.3658 |
| basic          |     0.9339 |          0.9093 |             0.1666 |             0.0845 |
| belt           |     0.9179 |          0.6398 |             0.4964 |             0.229  |
| blend          |     0.7309 |          0.6475 |             0.6641 |             0.3645 |
| upgd           |     0.8441 |          0.7776 |             0.5721 |             0.3194 |

### 2.2 按模型架构汇总
| arch      |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|:----------|-----------:|----------------:|-------------------:|
| mobilenet |     0.682  |          0.5716 |             0.564  |
| resnet18  |     0.741  |          0.5897 |             0.5932 |
| vgg       |     0.6874 |          0.5917 |             0.5724 |

### 2.3 按中毒率汇总
|   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|--------------:|-----------:|----------------:|-------------------:|
|         0.005 |     0.6107 |          0.5172 |             0.6073 |
|         0.01  |     0.6756 |          0.5345 |             0.6056 |
|         0.02  |     0.925  |          0.6424 |             0.4677 |
|         0.05  |     0.7552 |          0.6888 |             0.543  |
|         0.1   |     0.9923 |          0.6635 |             0.4721 |

### 2.4 平衡型候选配置（`asr_mean >= 0.3`，按 transfer 与 stealth 的调和均值排序，仅作参考）
| arch      | attack_type   |   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |   balance_hmean |
|:----------|:--------------|--------------:|-----------:|----------------:|-------------------:|----------------:|
| resnet18  | SIG           |          0.05 |     0.8038 |          0.7911 |             0.6851 |          0.7343 |
| resnet18  | blend         |          0.05 |     0.8532 |          0.826  |             0.6273 |          0.7131 |
| resnet18  | blend         |          0.01 |     0.7788 |          0.6761 |             0.7471 |          0.7098 |
| mobilenet | upgd          |          0.05 |     0.9546 |          0.9371 |             0.551  |          0.694  |
| vgg       | upgd          |          0.01 |     0.8147 |          0.725  |             0.6388 |          0.6792 |

### 2.5 Pareto 前沿（maximize transfer, maximize stealth）
| arch      | attack_type    |   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|:----------|:---------------|--------------:|-----------:|----------------:|-------------------:|
| mobilenet | basic          |         0.05  |     0.9996 |          0.9899 |             0.0583 |
| resnet18  | upgd           |         0.05  |     0.9816 |          0.9758 |             0.391  |
| mobilenet | upgd           |         0.05  |     0.9546 |          0.9371 |             0.551  |
| resnet18  | blend          |         0.05  |     0.8532 |          0.826  |             0.6273 |
| resnet18  | SIG            |         0.05  |     0.8038 |          0.7911 |             0.6851 |
| resnet18  | blend          |         0.01  |     0.7788 |          0.6761 |             0.7471 |
| mobilenet | SIG            |         0.05  |     0.7341 |          0.5789 |             0.7762 |
| resnet18  | blend          |         0.005 |     0.6831 |          0.5781 |             0.782  |
| resnet18  | adaptive_blend |         0.01  |     0.5951 |          0.536  |             0.7968 |
| resnet18  | adaptive_blend |         0.005 |     0.5588 |          0.4282 |             0.8311 |
| resnet18  | WaNet          |         0.005 |     0.5715 |          0.117  |             0.8517 |

### 2.6 各攻击最敏感/最不敏感的防御
| attack_type    | strongest_defense   |   strongest_tpr | weakest_defense   |   weakest_tpr |
|:---------------|:--------------------|----------------:|:------------------|--------------:|
| SIG            | SCaLe-Up            |          0.4827 | STRIP             |        0.0599 |
| WaNet          | IBD_PSC             |          0.5157 | STRIP             |        0.0306 |
| adaptive_blend | SCaLe-Up            |          0.4594 | STRIP             |        0.1167 |
| adaptive_patch | SentiNet            |          0.9371 | STRIP             |        0.1517 |
| basic          | SentiNet            |          0.9887 | STRIP             |        0.6262 |
| belt           | IBD_PSC             |          0.7491 | STRIP             |        0.1518 |
| blend          | SCaLe-Up            |          0.5776 | STRIP             |        0.1454 |
| upgd           | SCaLe-Up            |          0.7066 | STRIP             |        0.3106 |

### 2.7 含 NC 的补充隐蔽性（非主指标）
| attack_type    |   S_stealth |   S_stealth_tpr |
|:---------------|------------:|----------------:|
| SIG            |      0.3943 |          0.6256 |
| WaNet          |      0.3768 |          0.6048 |
| adaptive_blend |      0.3807 |          0.6164 |
| adaptive_patch |      0.4115 |          0.5577 |
| basic          |      0.1962 |          0.2619 |
| belt           |      0.2677 |          0.4816 |
| blend          |      0.3666 |          0.6063 |
| upgd           |      0.3192 |          0.5214 |
结论：CIFAR-10 上 trade-off 最明显，`transfer_mean` 与 `stealth_tpr_mean` 呈强负相关。`basic` 与高强度 `upgd`/`adaptive_patch` 能把迁移性推到很高，但隐蔽性显著下降；真正兼顾两者的点很少，当前最平衡的是 **ResNet18 + SIG@0.05**，但它仍然属于“中高迁移 / 中高隐蔽”，而不是极致两端都强。

## 2. mnistm 结果分析
- 该数据集上 `transfer_mean` 与 `stealth_tpr_mean` 的相关系数为 **-0.6551**，说明迁移性与以防御失效定义的隐蔽性整体呈显著负相关。
- `asr_mean` 与 `transfer_mean` 的相关系数为 **0.8990**，说明很多“迁移差”的点仍然与源域后门是否先建立成功高度相关。
- 以 `asr_mean >= 0.3` 为前提、用 `transfer` 与 `stealth` 的调和均值做平衡排序时，当前最优参考配置是 **`belt` / `resnet18` / poison_rate=`0.02`**。

### 2.1 按攻击方法汇总
| attack_type    |   asr_mean |   transfer_mean |   stealth_tpr_mean |   stealth_auc_mean |
|:---------------|-----------:|----------------:|-------------------:|-------------------:|
| SIG            |     0.0279 |          0.0026 |             0.8273 |             0.5782 |
| WaNet          |     0.57   |          0.7826 |             0.6247 |             0.453  |
| adaptive_blend |     0.5412 |          0.6251 |             0.7616 |             0.5133 |
| adaptive_patch |     0.3236 |          0.4828 |             0.6509 |             0.5015 |
| basic          |     0.7847 |          0.8626 |             0.3792 |             0.2486 |
| belt           |     0.7695 |          0.7466 |             0.7634 |             0.4574 |
| blend          |     0.7678 |          0.8466 |             0.6189 |             0.3847 |
| upgd           |     0.4032 |          0.3928 |             0.8135 |             0.5614 |

### 2.2 按模型架构汇总
| arch      |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|:----------|-----------:|----------------:|-------------------:|
| mobilenet |     0.4902 |          0.5523 |             0.7481 |
| resnet18  |     0.587  |          0.6605 |             0.6434 |
| vgg       |     0.4932 |          0.5653 |             0.6483 |

### 2.3 按中毒率汇总
|   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|--------------:|-----------:|----------------:|-------------------:|
|         0.005 |     0.3619 |          0.4963 |             0.7047 |
|         0.01  |     0.4828 |          0.5448 |             0.7087 |
|         0.02  |     0.7875 |          0.8691 |             0.7694 |
|         0.05  |     0.6285 |          0.6463 |             0.6151 |
|         0.1   |     0.9803 |          0.9987 |             0.6407 |

### 2.4 平衡型候选配置（`asr_mean >= 0.3`，按 transfer 与 stealth 的调和均值排序，仅作参考）
| arch      | attack_type   |   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |   balance_hmean |
|:----------|:--------------|--------------:|-----------:|----------------:|-------------------:|----------------:|
| resnet18  | belt          |          0.02 |     0.8911 |          0.9856 |             0.7832 |          0.8728 |
| mobilenet | belt          |          0.02 |     0.8822 |          0.9215 |             0.787  |          0.849  |
| mobilenet | belt          |          0.1  |     0.9899 |          1      |             0.6712 |          0.8032 |
| mobilenet | blend         |          0.01 |     0.7793 |          0.8571 |             0.7467 |          0.7981 |
| mobilenet | upgd          |          0.05 |     0.4504 |          0.7109 |             0.8923 |          0.7914 |

### 2.5 Pareto 前沿（maximize transfer, maximize stealth）
| arch      | attack_type   |   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|:----------|:--------------|--------------:|-----------:|----------------:|-------------------:|
| mobilenet | belt          |          0.1  |     0.9899 |          1      |             0.6712 |
| resnet18  | belt          |          0.02 |     0.8911 |          0.9856 |             0.7832 |
| mobilenet | belt          |          0.02 |     0.8822 |          0.9215 |             0.787  |
| mobilenet | upgd          |          0.05 |     0.4504 |          0.7109 |             0.8923 |
| mobilenet | belt          |          0.01 |     0.6745 |          0.6151 |             0.8998 |

### 2.6 各攻击最敏感/最不敏感的防御
| attack_type    | strongest_defense   |   strongest_tpr | weakest_defense   |   weakest_tpr |
|:---------------|:--------------------|----------------:|:------------------|--------------:|
| SIG            | IBD_PSC             |          0.4885 | STRIP             |        0.0506 |
| WaNet          | SCaLe-Up            |          0.6111 | SentiNet          |        0.0777 |
| adaptive_blend | IBD_PSC             |          0.4604 | SentiNet          |        0.0759 |
| adaptive_patch | SentiNet            |          0.8004 | SCaLe-Up          |        0.0584 |
| basic          | SentiNet            |          0.9199 | SCaLe-Up          |        0.1915 |
| belt           | IBD_PSC             |          0.5756 | SCaLe-Up          |        0.1139 |
| blend          | IBD_PSC             |          0.7268 | SentiNet          |        0.1145 |
| upgd           | IBD_PSC             |          0.415  | SentiNet          |        0.07   |

### 2.7 含 NC 的补充隐蔽性（非主指标）
| attack_type    |   S_stealth |   S_stealth_tpr |
|:---------------|------------:|----------------:|
| SIG            |      0.4858 |          0.6852 |
| WaNet          |      0.3889 |          0.5262 |
| adaptive_blend |      0.4519 |          0.6506 |
| adaptive_patch |      0.5003 |          0.6198 |
| basic          |      0.3156 |          0.4201 |
| belt           |      0.489  |          0.7338 |
| blend          |      0.3695 |          0.5569 |
| upgd           |      0.4905 |          0.6921 |
结论：MNIST-M 更像“有限 Pareto 前沿”问题。`belt` 明显占据主前沿，尤其在 `poison_rate=0.02` 时同时具备高同域 ASR、高迁移和较高隐蔽性；`upgd` 则提供更高 stealth，但迁移性上限低于 `belt`。`SIG` 在该数据集上主要体现为攻击建立失败。

## 2. tiny_imagenet 结果分析
- 该数据集上 `transfer_mean` 与 `stealth_tpr_mean` 的相关系数为 **-0.7304**，说明迁移性与以防御失效定义的隐蔽性整体呈显著负相关。
- `asr_mean` 与 `transfer_mean` 的相关系数为 **0.9272**，说明很多“迁移差”的点仍然与源域后门是否先建立成功高度相关。
- 以 `asr_mean >= 0.3` 为前提、用 `transfer` 与 `stealth` 的调和均值做平衡排序时，当前最优参考配置是 **`upgd` / `resnet18` / poison_rate=`0.005`**。

### 2.1 按攻击方法汇总
| attack_type    |   asr_mean |   transfer_mean |   stealth_tpr_mean |   stealth_auc_mean |
|:---------------|-----------:|----------------:|-------------------:|-------------------:|
| SIG            |     0.4159 |          0.5929 |             0.8994 |             0.5058 |
| WaNet          |     0.6212 |          0.688  |             0.8054 |             0.391  |
| adaptive_blend |     0.6351 |          0.742  |             0.7652 |             0.3626 |
| adaptive_patch |     0.7542 |          0.9078 |             0.3065 |             0.121  |
| basic          |     0.8247 |          0.9105 |             0.2555 |             0.1115 |
| belt           |     0.8933 |          0.927  |             0.4259 |             0.1889 |
| blend          |     0.7503 |          0.8056 |             0.7352 |             0.3394 |
| upgd           |     0.4086 |          0.6996 |             0.8983 |             0.4995 |

### 2.2 按模型架构汇总
| arch      |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|:----------|-----------:|----------------:|-------------------:|
| mobilenet |     0.6986 |          0.7851 |             0.5916 |
| resnet18  |     0.7335 |          0.8569 |             0.6106 |
| vgg       |     0.6249 |          0.748  |             0.6355 |

### 2.3 按中毒率汇总
|   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|--------------:|-----------:|----------------:|-------------------:|
|         0.001 |     0.2458 |          0.5044 |             0.9099 |
|         0.005 |     0.5957 |          0.749  |             0.7091 |
|         0.01  |     0.7199 |          0.8044 |             0.5816 |
|         0.02  |     0.9104 |          0.9447 |             0.4533 |
|         0.05  |     0.84   |          0.9009 |             0.4909 |
|         0.1   |     0.9938 |          0.9993 |             0.2955 |

### 2.4 平衡型候选配置（`asr_mean >= 0.3`，按 transfer 与 stealth 的调和均值排序，仅作参考）
| arch      | attack_type   |   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |   balance_hmean |
|:----------|:--------------|--------------:|-----------:|----------------:|-------------------:|----------------:|
| resnet18  | upgd          |         0.005 |     0.7137 |          0.9061 |             0.8665 |          0.8859 |
| vgg       | upgd          |         0.005 |     0.5715 |          0.8347 |             0.8936 |          0.8632 |
| resnet18  | SIG           |         0.005 |     0.7107 |          0.8178 |             0.8562 |          0.8365 |
| mobilenet | upgd          |         0.005 |     0.4082 |          0.7407 |             0.9232 |          0.822  |
| resnet18  | blend         |         0.01  |     0.7486 |          0.8197 |             0.8224 |          0.821  |

### 2.5 Pareto 前沿（maximize transfer, maximize stealth）
| arch      | attack_type   |   poison_rate |   asr_mean |   transfer_mean |   stealth_tpr_mean |
|:----------|:--------------|--------------:|-----------:|----------------:|-------------------:|
| resnet18  | WaNet         |         0.05  |     0.997  |          1      |             0.5807 |
| mobilenet | WaNet         |         0.05  |     0.9961 |          0.9997 |             0.6233 |
| vgg       | WaNet         |         0.05  |     0.9914 |          0.9994 |             0.6445 |
| resnet18  | upgd          |         0.005 |     0.7137 |          0.9061 |             0.8665 |
| vgg       | upgd          |         0.005 |     0.5715 |          0.8347 |             0.8936 |
| mobilenet | upgd          |         0.005 |     0.4082 |          0.7407 |             0.9232 |
| mobilenet | SIG           |         0.005 |     0.5087 |          0.6778 |             0.9335 |
| resnet18  | WaNet         |         0.005 |     0.3793 |          0.4984 |             0.9512 |

### 2.6 各攻击最敏感/最不敏感的防御
| attack_type    | strongest_defense   |   strongest_tpr | weakest_defense   |   weakest_tpr |
|:---------------|:--------------------|----------------:|:------------------|--------------:|
| SIG            | IBD_PSC             |          0.2087 | SCaLe-Up          |        0.0532 |
| WaNet          | IBD_PSC             |          0.4491 | STRIP             |        0.0664 |
| adaptive_blend | STRIP               |          0.3733 | SCaLe-Up          |        0.0778 |
| adaptive_patch | SentiNet            |          1      | IBD_PSC           |        0.477  |
| basic          | SentiNet            |          1      | SCaLe-Up          |        0.3996 |
| belt           | IBD_PSC             |          0.7962 | SCaLe-Up          |        0.3812 |
| blend          | IBD_PSC             |          0.439  | SentiNet          |        0.0989 |
| upgd           | IBD_PSC             |          0.1582 | SCaLe-Up          |        0.0497 |

### 2.7 含 NC 的补充隐蔽性（非主指标）
| attack_type    |   S_stealth |   S_stealth_tpr |
|:---------------|------------:|----------------:|
| SIG            |      0.4046 |          0.7195 |
| WaNet          |      0.3128 |          0.6443 |
| adaptive_blend |      0.2906 |          0.612  |
| adaptive_patch |      0.1001 |          0.2485 |
| basic          |      0.0896 |          0.2048 |
| belt           |      0.1511 |          0.3407 |
| blend          |      0.2747 |          0.5914 |
| upgd           |      0.3996 |          0.7187 |
结论：Tiny ImageNet 的高迁移高隐蔽前沿最宽。`WaNet@0.05` 能把迁移拉满，但隐蔽性一般；`upgd@0.005` 在三种架构上都给出非常强的平衡结果；`SIG@0.005` 也表现出比 CIFAR-10/MNIST-M 更好的平衡性。整体上这是三个数据集中最容易同时拿到“高迁移 + 高隐蔽”的设置。

## 3. 综合结论
- **从整体 trade-off 形态看**：CIFAR-10 最尖锐，MNIST-M 次之，Tiny ImageNet 的可行前沿最宽。
- **从方法族看**：`basic` 往往提供极高同域/迁移成功率，但 stealth 最差；`upgd` 与 `adaptive_blend` 更偏向隐蔽；`belt` 在 MNIST-M 上最均衡；`upgd` 在 Tiny ImageNet 上最均衡。
- **从架构看**：CIFAR-10 上 ResNet18 的综合表现最好；MNIST-M 上 MobileNet 的 stealth 最强，但 ResNet18 的迁移更强；Tiny ImageNet 上 ResNet18 在迁移性上最占优，VGG 在 stealth 上略高。
- **从防御看**：`IBD_PSC` 在 MNIST-M 和 Tiny ImageNet 上更稳定，`SCaLe-Up` 在 CIFAR-10 上平均最强；`STRIP` 经常是最弱防御。`SentiNet` 对 `basic` / `adaptive_patch` 尤其有效。
- **论文叙事建议**：主文使用 `stealth_tpr_mean` 作为隐蔽性主指标，强调“基于多防御平均失效率”；把 `S_stealth` 作为补充验证，说明即便加入 NC，主要方法排序不会完全翻转，只会让高 stealth 方法的优势更保守。
