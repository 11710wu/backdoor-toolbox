# SIG/UPGD 两种模式影响分析

## 结论摘要
- all-to-one 相比 clean-label 的平均迁移性提升为 `0.427`，但隐蔽性同步下降：stealth_auc 平均变化 `-0.160`，stealth_tpr 平均变化 `-0.202`。
- 这说明 all-to-one 更容易把后门迁移到目标域，但攻击痕迹更明显，更容易被防御检测到。
- MNISTM 上模式切换影响最大，尤其 SIG 从 clean-label 的几乎不迁移变成高迁移；CIFAR10 上提升更温和。
- 防御侧没有单一方法全胜：IBD_PSC 对 MNISTM 更强，SCaLe-Up/STRIP 对 CIFAR10 UPGD 更强，Tiny-ImageNet 上 STRIP 对 all-to-one 的提升最明显。

## 迁移性与隐蔽性：两种模式对比
|dataset|attack|clean transfer|all-to-one transfer|transfer delta|clean stealth_auc|all-to-one stealth_auc|stealth_auc delta|clean stealth_tpr|all-to-one stealth_tpr|stealth_tpr delta|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cifar10|sig|0.477|0.859|0.382|0.454|0.338|-0.117|0.744|0.586|-0.158|
|cifar10|upgd|0.778|0.970|0.193|0.319|0.181|-0.138|0.572|0.352|-0.220|
|mnistm|sig|0.003|0.756|0.754|0.578|0.371|-0.208|0.827|0.564|-0.263|
|mnistm|upgd|0.393|0.995|0.602|0.561|0.348|-0.213|0.813|0.544|-0.270|
|tiny_imagenet|sig|0.487|0.693|0.206|0.506|0.381|-0.124|0.899|0.799|-0.100|
|tiny_imagenet|upgd|0.521|NA|NA|0.500|NA|NA|0.898|NA|NA|

### 迁移性解读
- CIFAR10: all-to-one 后 SIG 的迁移性从约 `0.477` 提升到 `0.859`，UPGD 从约 `0.778` 提升到 `0.970`。UPGD 在两种模式下都更强，但 SIG 的相对增幅更大。
- MNISTM: 模式影响最剧烈。clean-label 下 SIG 迁移率约 `0.003`，几乎不可迁移；all-to-one 后达到约 `0.756`。UPGD 也从约 `0.393` 提升到 `0.995`。
- Tiny-ImageNet: 当前汇总里 clean-label 与 all-to-one 覆盖不完全一致，报告保留单模式统计，但不把它纳入 paired delta 结论。

### 隐蔽性解读
- 两个数据集、两种攻击的 stealth_auc / stealth_tpr delta 全为负，说明 all-to-one 的增强不是免费的。
- CIFAR10 上 UPGD 的 stealth_tpr 下降约 `0.220`，比 SIG 的 `0.158` 更明显；MNISTM 上 SIG/UPGD 的 stealth_tpr 都下降约 `0.26~0.27`。
- 直观上，all-to-one 让标签统一指向目标类，攻击信号更一致，迁移更强，但防御方法也更容易捕捉到异常模式。

## 各防御方法效果
### all-to-one 模式下每组最强防御
|dataset|attack|best defense|tpr_mean|auc_mean|
|---|---|---|---:|---:|
|cifar10|sig|IBD_PSC|0.685|0.794|
|cifar10|upgd|SCaLe-Up|0.885|0.901|
|mnistm|sig|IBD_PSC|0.809|0.768|
|mnistm|upgd|IBD_PSC|0.782|0.683|
|tiny_imagenet|sig|STRIP|0.367|0.731|
|tiny_imagenet|upgd|STRIP|0.461|0.776|

### 防御检测率变化（all-to-one - clean-label）
|dataset|attack|defense|clean TPR|all-to-one TPR|TPR delta|clean AUC|all-to-one AUC|AUC delta|
|---|---|---|---:|---:|---:|---:|---:|---:|
|cifar10|sig|IBD_PSC|0.375|0.685|0.310|0.608|0.794|0.186|
|cifar10|sig|SCaLe-Up|0.483|0.679|0.197|0.669|0.794|0.125|
|cifar10|sig|STRIP|0.060|0.191|0.131|0.417|0.565|0.148|
|cifar10|sig|SentiNet|0.109|0.102|-0.007|0.489|0.498|0.009|
|cifar10|upgd|IBD_PSC|0.374|0.479|0.105|0.584|0.682|0.097|
|cifar10|upgd|SCaLe-Up|0.707|0.885|0.178|0.795|0.901|0.107|
|cifar10|upgd|STRIP|0.311|0.744|0.433|0.656|0.910|0.253|
|cifar10|upgd|SentiNet|0.320|0.486|0.165|0.687|0.782|0.095|
|mnistm|sig|IBD_PSC|0.489|0.809|0.321|0.489|0.768|0.279|
|mnistm|sig|SCaLe-Up|0.065|0.401|0.336|0.337|0.625|0.288|
|mnistm|sig|STRIP|0.051|0.441|0.390|0.467|0.728|0.261|
|mnistm|sig|SentiNet|0.087|0.092|0.005|0.394|0.396|0.003|
|mnistm|upgd|IBD_PSC|0.415|0.782|0.367|0.448|0.683|0.235|
|mnistm|upgd|SCaLe-Up|0.188|0.377|0.190|0.432|0.604|0.171|
|mnistm|upgd|STRIP|0.074|0.568|0.495|0.458|0.839|0.381|
|mnistm|upgd|SentiNet|0.070|0.098|0.028|0.416|0.481|0.066|
|tiny_imagenet|sig|IBD_PSC|0.209|0.220|0.011|0.480|0.621|0.141|
|tiny_imagenet|sig|SCaLe-Up|0.053|0.145|0.091|0.521|0.638|0.117|
|tiny_imagenet|sig|STRIP|0.087|0.367|0.280|0.527|0.731|0.204|
|tiny_imagenet|sig|SentiNet|0.053|0.073|0.020|0.448|0.485|0.037|
|tiny_imagenet|upgd|IBD_PSC|0.158|0.260|0.102|0.474|0.664|0.191|
|tiny_imagenet|upgd|SCaLe-Up|0.050|0.158|0.109|0.490|0.618|0.128|
|tiny_imagenet|upgd|STRIP|0.113|0.461|0.348|0.550|0.776|0.226|
|tiny_imagenet|upgd|SentiNet|0.086|0.092|0.006|0.489|0.495|0.006|

### 防御方法解读
- CIFAR10/SIG: IBD_PSC 与 SCaLe-Up 在 all-to-one 下 TPR 都约 `0.68`，明显高于 clean-label；STRIP 也提升但仍较弱。
- CIFAR10/UPGD: SCaLe-Up 和 STRIP 最敏感，all-to-one TPR 分别约 `0.885` 和 `0.744`。
- MNISTM/SIG: IBD_PSC 最强，TPR 从约 `0.489` 升到 `0.809`；SCaLe-Up/STRIP 也从很弱变成中等强度。
- MNISTM/UPGD: IBD_PSC 和 STRIP 最有效，SentiNet 在两种模式下都较弱。
- Tiny-ImageNet: all-to-one 下 STRIP 对 SIG/UPGD 的提升很明显；IBD_PSC 和 SCaLe-Up 的提升较小但仍有增强。

## 数据完整性说明
- 当前 paired delta 主要基于 CIFAR10 和 MNISTM 的可配对实验。
- Tiny-ImageNet 有 clean-label 和 all-to-one 的单模式统计，但两侧覆盖不完全一致，因此不作为严格 paired delta 的主结论。
- 详细 CSV: `mode_transfer_stealth_summary.csv`、`mode_transfer_stealth_delta.csv`、`mode_defense_dataset_summary.csv`。
