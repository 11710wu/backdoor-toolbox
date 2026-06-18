# 安全领域攻击迁移性与隐蔽性定义综述

本文档总结前面文献检索和阅读中看到的“攻击迁移性 transferability”和“攻击隐蔽性 stealthiness / evasiveness / detectability / imperceptibility”的常见定义。目标是给汇报使用：尽量覆盖主流定义和度量方式，而不是只给最终推荐公式。

## 1. 迁移性 Transferability

### 1.1 总体含义

安全文献里，攻击迁移性最核心的含义是：

```text
攻击在源侧构造、学习或注入后，迁移到目标侧是否仍然有效。
```

源侧可以是：

- source model
- surrogate model
- source dataset / source domain
- poisoned training set
- teacher model / pre-trained model
- poisoned encoder / representation

目标侧可以是：

- target model
- victim model
- target dataset / target domain
- student model
- downstream task
- fine-tuned model

因此，迁移性不是单一数学公式，而是一个“源侧到目标侧”的评估场景。不同攻击类型会使用不同成功率指标。

## 1.2 对抗样本中的迁移性

### 非定向迁移

在非定向 adversarial example 中，迁移性通常指：

```text
源模型上生成的 adversarial examples 是否也能让目标模型误分类。
```

常见指标：

- `fooling rate`
- `misclassification rate`
- `transfer success rate`
- 目标模型在 adversarial examples 上的 accuracy，越低表示迁移性越强。

早期文献也会直接把 transfer rate 写成：

```text
目标模型也被该 adversarial example 误导的比例。
```

注意这里的 `transfer rate` 是“目标模型成功被误导的比例”，不是 `target/source` 比值。

### 定向迁移

在 targeted adversarial attack 中，迁移成功通常定义为：

```text
目标模型是否把 adversarial example 判成攻击者指定 target label。
```

常见指标：

- `matching rate`
- `targeted fooling rate`
- `targeted transfer success rate`
- `TFR`

形式上可写为：

\[
TFR =
\frac{1}{N}\sum_i \mathbf{1}[f_{target}(x_i^{adv}) = y_t]
\]

### 对抗样本中的归一化或理论定义

少数理论论文会给出更形式化的迁移性。例如：

```text
源模型攻击转移到目标模型后的效果 / 目标模型自身白盒攻击效果。
```

这类定义的归一化对象是“目标模型白盒攻击上界”，不是 source ASR。它更适合理论分析，不是后门实验中的常规主指标。

## 1.3 后门攻击中的迁移性

后门攻击中，迁移性通常指：

```text
后门从源模型、源数据集或预训练模型迁移到目标模型、目标数据集、目标域或下游任务后，触发器是否仍能诱导攻击目标输出。
```

最常见指标是目标侧 ASR：

\[
ASR_{target}
=
\frac{
\#\{x, y \ne y_t: f(T(x)) = y_t\}
}{
\#\{x, y \ne y_t\}
}
\]

其中：

- \(T(x)\)：加 trigger 后的输入。
- \(y_t\)：攻击目标类。
- 通常只统计非目标类样本，避免 target-class 样本天然命中目标类。

常见名称：

- `target-side ASR`
- `transfer ASR`
- `cross-model ASR`
- `cross-dataset ASR`
- `cross-domain ASR`
- `student model ASR`
- `downstream ASR`

## 1.4 后门迁移的具体场景

| 场景 | 迁移性定义 | 常见指标 |
|---|---|---|
| 跨模型后门 | 同一个 trigger 或 backdoor behavior 在另一模型架构上是否仍有效 | target model ASR |
| 跨数据集后门 | 源数据集训练出的后门在目标数据集上是否仍触发目标类 | target dataset ASR |
| 跨域后门 | 源域后门在目标域图像或分布偏移数据上是否仍有效 | target-domain ASR |
| 迁移学习后门 | teacher / pre-trained model 中的后门是否被 student model 继承 | student ASR, clean accuracy |
| 预训练 encoder 后门 | backdoored encoder 是否让多个 downstream classifiers 继承后门 | downstream ASR |
| 跨任务后门 | 预训练模型中的后门是否在不同下游任务中显性化 | downstream ASR, task success |
| CLIP / 多模态后门 | 后门是否迁移到 unseen classes、cross-dataset、cross-domain 或 retrieval task | ASR, clean ACC, retrieval metrics |

这类文献通常同时报告：

- 目标侧 ASR。
- clean accuracy / benign accuracy。
- 有时报告防御后 ASR 或感知质量指标。

## 1.5 数据投毒中的迁移性

数据投毒和 clean-label poisoning 中，迁移性通常指：

```text
在 surrogate model 或源训练流程中设计的 poison，是否能在未知 victim model 训练后仍成功。
```

常见指标：

- `attack success rate`
- `targeted attack success`
- victim model 上目标样本被判为攻击目标类的比例。

典型背景：

- victim architecture 未知。
- victim training data 不完全可见。
- victim training pipeline 与源侧不同。
- poison 需要跨 transfer learning 或 end-to-end training 生效。

这里的迁移性同样主要看 victim 目标侧成功率，而不是 source/victim ratio。

## 1.6 预训练模型与跨任务后门中的特殊情况

在预训练 NLP、encoder、foundation model 或多任务后门中，source 阶段往往没有与下游任务一致的标签空间。因此：

```text
source ASR 可能不存在，或者语义上不可比。
```

这类工作常把迁移性理解为：

```text
后门行为是否被下游模型继承，或在 fine-tuning 后是否仍能触发恶意行为。
```

常见指标：

- downstream ASR。
- student/fine-tuned model ASR。
- downstream clean accuracy。
- 触发错误所需最少 trigger 插入次数。
- trigger 占输入长度比例。
- downstream task-specific metric。

## 1.7 source ASR 与 ratio 类指标

检索中没有看到以下指标成为攻击迁移性的主流标准定义：

- `target ASR / source ASR`
- `target ASR - source ASR`
- `log(target/source)`
- `target ASR^2 / source ASR`
- `chance-adjusted ASR`

更常见做法是：

```text
主指标报告目标侧成功率；
source-side success 作为源攻击强度参考；
clean accuracy / stealthiness / defense evasion 作为辅助指标。
```

ratio 或 delta 类指标更适合作为诊断量：

- 判断目标侧是否比源侧更强或更弱。
- 分析 transfer asymmetry。
- 检查 source ASR 很低但 target ASR 很高的异常或非对称现象。

## 1.8 迁移性定义汇总表

| 攻击类型 | 定义核心 | 常见指标 | 是否常用 source 指标 |
|---|---|---|---|
| 非定向对抗样本 | 源 AE 是否误导目标模型 | fooling rate, target error rate | 可报告，但非主定义 |
| 定向对抗样本 | 目标模型是否输出 target label | matching rate, TFR | 可报告 |
| 后门攻击 | trigger 迁移到目标侧后是否仍诱导目标类 | transfer ASR, target-side ASR | 常报告 source ASR，但不作分母 |
| 数据投毒 | poison 是否在 victim 训练后成功 | victim attack success rate | surrogate 信息用于构造 |
| 预训练/下游后门 | 后门是否被 downstream model 继承 | downstream ASR, clean ACC | source ASR 常不可比 |
| 理论迁移性 | 转移攻击效果相对目标白盒上界 | relative loss/effect ratio | 不用 source ASR ratio |

## 2. 隐蔽性 Stealthiness / Evasiveness / Detectability / Imperceptibility

### 2.1 总体情况

后门和机器学习安全文献中，隐蔽性没有统一单一指标。不同论文使用的术语侧重点不同：

| 术语 | 常见含义 |
|---|---|
| `stealthiness` | 总称，可能包含干净性能、不可见性、检测规避、低误触发等 |
| `imperceptibility` | trigger、扰动或 poison 对人类观察者不明显 |
| `detectability` | 检测器能否发现 poisoned sample、trigger 或 backdoored model |
| `evasiveness` | 攻击绕过检测器、防御器或审查机制的能力 |
| `sustainability` | 攻击在常见防御或处理后是否仍保持有效 |

因此，stealthiness 在文献中经常是一个 umbrella term，需要根据具体论文看它到底指哪一类隐蔽。

## 2.2 Clean utility 型隐蔽性

定义：

```text
后门模型在干净样本上表现正常，不明显降低正常任务性能。
```

常见指标：

- `clean accuracy`
- `benign accuracy`
- `clean accuracy drop`
- `CAD`，clean accuracy difference
- `C-Acc`

代表背景：

- BadNets 类经典后门。
- BackdoorBench。
- Poison Frogs。
- LIRA。
- 预训练模型后门。

解释：

- 这是最常见的后门评估维度。
- 它说明攻击不会破坏正常功能。
- 但它不是完整 stealthiness，因为 clean accuracy 高不代表 trigger 不可见，也不代表检测器抓不到。

## 2.3 感知不可见型隐蔽性 Imperceptibility

定义：

```text
trigger、扰动或 poisoned sample 在视觉上不可见，或在文本/语义上自然。
```

视觉任务常见指标：

- `human inspection`
- `PSNR`
- `SSIM`
- `LPIPS`
- `FID`
- trigger size / trigger opacity
- perturbation norm

文本任务常见指标：

- `perplexity`
- `grammar error`
- `semantic similarity`
- `USE similarity`
- human evaluation

对抗样本文献中也常用：

- \(L_0\), \(L_2\), \(L_\infty\) norm。
- bounded perturbation。
- imperceptible perturbation。

代表背景：

- adversarial examples：小扰动但改变模型预测。
- WaNet：warping trigger 在人类检查下不明显。
- LIRA：learnable imperceptible noise。
- ISSBA：sample-specific invisible trigger。
- NLP textual backdoor：语法、困惑度、语义相似度。

## 2.4 检测器规避型隐蔽性 Evasiveness / Detectability

定义：

```text
后门样本、trigger 或后门模型不容易被防御方法检测出来。
```

常见检测对象：

- poisoned sample。
- triggered test input。
- trigger pattern。
- backdoored model。
- suspicious target class。
- poisoned training data cluster。

常见指标：

- `TPR`
- `FPR`
- `AUC`
- `AP`
- `Recall`
- `Precision`
- `F1`
- `anomaly score`
- `detection rate`
- `false acceptance rate`
- `false rejection rate`

从攻击者角度，较低检测率代表更高隐蔽性。因此可以定义：

```text
evasion = 1 - detection rate
evasion = 1 - TPR
```

但主流论文更常见的是分别报告 detection metrics，然后文字解释“检测率越低越隐蔽”。直接把 `1 - TPR` 命名为 stealthiness 通常是研究者自己的 operational definition。

代表背景：

- Neural Cleanse：反向搜索最小 trigger，用 anomaly index 检测模型级后门。
- STRIP：输入扰动后预测熵低则可疑，报告 FAR/FRR。
- SentiNet：使用显著性区域和遮挡测试检测 localized universal attacks。
- ScaleUp：使用 scaled prediction consistency。
- IBD-PSC：使用 parameter-oriented scaling consistency。
- Unified detection framework：强调 fixed-FPR 下的 detection power 和 AUCROC。

## 2.5 投毒过程隐蔽性 Poisoning Stealth

定义：

```text
训练集中的 poisoned samples 不容易被人工审查或数据清洗发现。
```

常见条件或指标：

- clean-label。
- label consistency。
- low poison rate。
- small perturbation。
- poisoned sample visually plausible。
- poisoned sample correctly labeled。
- poison budget。

代表背景：

- Poison Frogs。
- Clean-label poisoning。
- Transferable clean-label poisoning。
- Label-consistent backdoor attacks。

解释：

- 这类隐蔽性关注训练数据是否可疑。
- 它不同于测试时 trigger 是否会被检测器抓到。

## 2.6 误触发风险型隐蔽性

定义：

```text
trigger 或 trigger 子模式不应在正常用户输入中频繁自然出现，避免后门被普通用户意外触发。
```

常见指标：

- `false trigger rate`
- `false triggered rate`
- `unintended activation rate`
- trigger 子序列误触发概率。

代表背景：

- NLP backdoor。
- style / syntax / rare-word trigger。
- pre-trained language model backdoor。

这类定义在 NLP 中更成熟，视觉 patch backdoor 中较少作为主指标。

## 2.7 模型修改隐蔽性

有些模型后门、model-reuse attack 或预训练模型后门还会隐含考虑：

```text
模型本身是否看起来正常，架构、参数规模、输出行为是否不引人怀疑。
```

可能使用的指标或条件：

- 不改变模型架构。
- 不明显降低 clean performance。
- 权重变化小。
- pruning / fine-tuning 后仍有效。
- 下游任务正常性能保持。

这类定义通常不会单独形成一个通用 stealthiness 公式，而是和 clean utility、defense evasion 一起报告。

## 2.8 多个防御方法同时存在时的处理

文献中的主流做法是：

1. 分别报告每个防御方法的 TPR/FPR/AUC 或防御后 ASR。
2. 做 attack-defense pair 表格。
3. 做 Pareto tradeoff 或二维 tradeoff 图。
4. benchmark 中使用 normalized ASR / CAD 等做综合比较。

较少见的做法是：

```text
直接把多个检测器的 detection rate 平均成一个单一 stealthiness。
```

因此，`1 - mean(TPR_i)` 是合理的统计分析型 aggregate metric，但最好明确命名为：

```text
aggregate detector-evasion score
mean evasion rate
detector-evasion stealthiness
```

如果要更完整，常见补充方式包括：

- per-defense breakdown。
- fixed-FPR TPR。
- worst-case detector score，例如 `1 - max(TPR_i)`。
- AUC / AP。
- clean accuracy drop。

## 2.9 隐蔽性定义汇总表

| 类型 | 核心问题 | 常见指标 | 典型文献背景 |
|---|---|---|---|
| Clean utility | 正常输入表现是否不变差 | clean ACC, benign ACC, CAD, C-Acc | BadNets, BackdoorBench, LIRA |
| Imperceptibility | trigger / poison 是否看不出来 | PSNR, SSIM, LPIPS, PPL, grammar error, human inspection | WaNet, LIRA, ISSBA, textual backdoor |
| Detector evasion | 检测器是否抓不到 | low TPR, low detection rate, AUC/FPR, anomaly score | Neural Cleanse, STRIP, SentiNet, ScaleUp, IBD-PSC |
| Poisoning stealth | 训练集污染是否不显眼 | clean-label, label consistency, poison budget | Poison Frogs, clean-label poisoning |
| False trigger risk | 正常输入是否容易误触发 | FTR, unintended activation | NLP backdoor |
| Model modification stealth | 模型本身是否看起来正常 | no architecture change, small weight change, clean utility | model-reuse / PTM backdoor |
| Defense sustainability | 防御后是否仍有效 | defended ASR, post-pruning ASR | robust / persistent backdoor |

## 3. 与当前实验的对应关系

当前实验字段可以和文献定义对应如下：

| 当前字段 | 文献概念 | 说明 |
|---|---|---|
| `source_asr` | source-side attack effectiveness | 源域攻击是否成立 |
| `transfer_asr` | target-side transferability | 目标域攻击成功率 |
| `clean_acc` | clean utility | 干净样本性能 |
| `difficulty = 1 - clean_acc` | task/model difficulty | 可作为 RQ2 调节变量 |
| `sentinet_tpr` | detector detectability | SentiNet 检测率 |
| `scaleup_tpr` | detector detectability | ScaleUp 检测率 |
| `strip_tpr` | detector detectability | STRIP 检测率 |
| `ibd_psc_tpr` | detector detectability | IBD-PSC 检测率 |
| `1 - mean(TPR_i)` | aggregate detector evasion | 检测器规避型隐蔽性 |

需要注意：

- `transfer_asr` 对应文献中最常见的后门迁移性定义。
- `source_asr` 更像源攻击有效性参考，不是迁移性的主定义。
- `clean_acc` 更像 clean utility 或 difficulty，不建议直接合并进 detector-evasion stealthiness。
- `1 - mean(TPR_i)` 是合理的聚合隐蔽性定义，但需要说明它是 detector-evasion operationalization。

## 4. 汇报用简短表述

可以这样概括迁移性：

```text
文献中，迁移性通常指攻击从源侧迁移到目标侧后是否仍然有效。
在对抗样本中，它常用 fooling rate、misclassification rate 或 targeted fooling rate 衡量；
在后门和投毒中，它更常表现为目标模型、目标数据集、目标域或下游任务上的 attack success rate / ASR。
source-side success 通常作为源攻击强度参考，而不是迁移性的分母。
```

可以这样概括隐蔽性：

```text
文献中，隐蔽性不是单一概念，主要包括 clean utility、trigger imperceptibility、
detector evasion、poisoning stealth、false trigger risk 和 model modification stealth。
clean accuracy 衡量正常功能是否保持；PSNR/SSIM/LPIPS 或 PPL/grammar error 衡量样本自然性；
TPR/FPR/AUC/anomaly score 衡量检测器能否发现攻击。
```

可以这样对应当前实验：

```text
当前实验中，transfer_asr 对应目标侧迁移成功率；
1 - mean(TPR_SentiNet, TPR_ScaleUp, TPR_STRIP, TPR_IBD-PSC)
对应检测器规避型隐蔽性。
```

## 5. 代表性引用方向

### 迁移性相关

- Szegedy et al., Intriguing Properties of Neural Networks.
- Goodfellow et al., Explaining and Harnessing Adversarial Examples.
- Papernot et al., Transferability in Machine Learning.
- Liu et al., Delving into Transferable Adversarial Examples and Black-box Attacks.
- Liang et al., Uncovering the Connections Between Adversarial Transferability and Knowledge Transferability.
- Zhu et al., Transferable Clean-Label Poisoning Attacks on Deep Neural Nets.
- Bullseye Polytope.
- Latent Backdoor Attacks.
- Backdoor Pre-trained Models Can Transfer to All.
- BadEncoder.
- BadCLIP.
- BackdoorBench.

### 隐蔽性相关

- Poison Frogs.
- Label-Consistent Backdoor Attacks.
- Neural Cleanse.
- STRIP.
- SentiNet.
- WaNet.
- LIRA.
- Invisible Backdoor Attack with Sample-Specific Triggers.
- Rethinking Stealthiness of Backdoor Attack against NLP Models.
- A Unified Evaluation of Textual Backdoor Learning.
- TrojanZoo.
- BackdoorBench.
- ScaleUp.
- IBD-PSC.
- A Unified Detection Framework for Inference-Stage Backdoor Defenses.

