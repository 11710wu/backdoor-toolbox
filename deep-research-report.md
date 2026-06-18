# 机器学习安全中攻击隐蔽性的定义与度量方式

## 结论概览

在你要求的主流文献里，**stealthiness / evasiveness / detectability / imperceptibility 并不是同一个概念，也没有一个被普遍接受的单一标量定义**。更准确地说，主流论文通常把“隐蔽性”拆成几类不同但相关的目标：其一，**对干净样本无害**，即高 clean accuracy / benign accuracy 或低 clean accuracy drop；其二，**触发器或毒化样本在感知上不显眼**，例如视觉上不可见、文本上自然流畅；其三，**难以被部署者或防御器检测到**，例如低 detector recall/TPR、低 anomaly score、低 DSR；其四，**不容易被误触发**，这在 NLP 里被明确形式化为 false triggered rate。主流论文往往只覆盖其中一部分，而不是把它们合成一个通用 stealthiness 分数。citeturn38view0turn39view1turn17view0turn26view2turn36view0

如果只问“文献里最常见的 stealthiness proxy 是什么”，答案并不是“统一的 detector-evasion 指标”，而更接近于：**经典 backdoor/poisoning 论文最常报告的是 ASR + clean accuracy/CACC/C-Acc；更强调‘隐蔽性’的论文会再加上人类可感知性、文本流畅性、样本/模型检测率或误触发率。**BackdoorBench 甚至把 C-Acc、ASR、R-Acc 作为标准协议；TrojanZoo 则指出很多工作实际上只用了 ASR 和 clean accuracy drop，并认为这不足以完整描述攻击。citeturn19search1turn36view0

对你的研究场景，我不建议把“stealthiness”直接等同于 clean accuracy，也不建议直接等同于某一个 defense 的未检出率。**如果你真正想研究“迁移性与隐蔽性的关系”，最稳妥的主定义应当是“在固定 FPR 下跨多个检测器的平均漏检率”，再辅以 clean accuracy drop 和模态相关的 perceptibility/naturalness 指标**。这样做的好处是：它把 stealthiness 从“攻击是否成功”中分离出来，也比单纯 clean accuracy 更接近“逃避检测”的安全含义。这个做法和近年的统一检测框架“在给定 FPR 下最大化 detection power”的表述是兼容的。citeturn43search0turn43search2turn20view1

## 文献总结表

| 论文 | 年份与会议/期刊 | 攻击类型 | 文中如何理解隐蔽性 | 主要指标 | 是否含防御检测 | 是否含 clean accuracy | 是否含 trigger perceptibility / naturalness | 证据 |
|---|---|---|---|---|---|---|---|---|
| **Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks** | 2018, NeurIPS | clean-label 数据投毒 | 重点是 **clean-label** 与 “seemingly innocuous image”，并强调在实现目标误分类时 **不降低整体性能** | targeted success、整体性能、poison 数量 | 否 | 是 | 是，主要是“看起来无害/标签正确” | citeturn28view0 |
| **Transferable Clean-Label Poisoning Attacks on Deep Neural Nets** | 2019, ICML | clean-label 数据投毒 | 隐蔽性主要是 **small perturbations + clean-label + 低 poison rate**，核心还是 transferability | transfer ASR、1% poison budget | 未强调 | 在开放摘要中未突出 | 是，small perturbations | citeturn28view1 |
| **Label-Consistent Backdoor Attacks** | 2019, arXiv/OpenReview 补充文献 | clean-label backdoor | 明确认为要“remain undetected”就必须 **label-consistency**，因为明显错标会引起人工怀疑 | ASR、clean performance、label consistency | 否 | 是 | 是，强调 human inspection 下也显得 benign | citeturn30search0turn30search1 |
| **Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks** | 2019, IEEE S&P | 模型级后门检测 | detectability 被定义为 **反向恢复出最小触发器后，看目标类是否在 perturbation size 上异常** | anomaly index、MAD、mitigation utility | 是 | 部分 | 否 | citeturn22search3turn21search2 |
| **STRIP: A Defence Against Trojan Attacks on Deep Neural Networks** | 2019, ACSAC | 输入级后门检测 | detectability 被定义为 **对输入施加强扰动后，若预测熵仍很低，则可疑** | entropy、FAR、FRR | 是 | 否 | 否 | citeturn20view1 |
| **Input-Aware Dynamic Backdoor Attack** | 2020, NeurIPS | 动态/样本相关后门 | 认为固定 trigger 容易被检测；通过 **input-aware、nonreusability** 来提高 stealthiness 与 defense bypass | ASR、clean-side performance、Neural Cleanse/STRIP bypass | 是 | 是 | 是，但以“动态且不可复用”为主 | citeturn13view4 |
| **WaNet: Imperceptible Warping-based Backdoor Attack** | 2021, ICLR | 视觉 backdoor | 把 stealthiness 表述为 **在人类检查下更难发现，并且对机器防御器不可检测** | human inspection test、defense bypass、ASR、clean performance | 是 | 是 | 是，人类检查 | citeturn13view3turn45search5 |
| **LIRA: Learnable, Imperceptible and Robust Backdoor Attacks** | 2021, ICCV | 视觉 backdoor | 明确把 stealthiness 与 **imperceptible noise + preserve clean performance + bypass defenses/human inspection** 绑定 | ASR、clean performance、defense bypass | 是 | 是 | 是，imperceptible noise | citeturn13view2 |
| **Invisible Backdoor Attack With Sample-Specific Triggers** | 2021, ICCV | 样本相关视觉 backdoor | 明确把攻击目标分成 **effectiveness、stealthiness、sustainability**；其中 stealthiness = **触发器隐藏 + poison 比例小**，sustainability = **在常见 defenses 下仍有效** | ASR、poison rate、defense bypass | 是 | 是 | 是，invisible perturbation | citeturn25view0turn26view2 |
| **Backdoor Attack with Imperceptible Input and Latent Modification** | 2021, OpenReview 补充文献 | 输入+特征空间后门 | 明确区分 **input-space imperceptibility** 与 **latent-space inseparability**，认为后者决定是否能防御规避 | ASR、clean performance、latent distribution matching、defense bypass | 是 | 是 | 是 | citeturn14view1 |
| **Rethinking Stealthiness of Backdoor Attack against NLP Models** | 2021, ACL-IJCNLP | NLP backdoor | 首次把 stealthiness 显式扩成两部分：**对部署者的可检测性** 与 **对用户的误触发风险** | Clean Acc/F1、ASR、DSR、FTR | 是 | 是 | 是，文本自然性通过 DSR 间接度量 | citeturn15view0turn17view0turn16view3 |
| **A Unified Evaluation of Textual Backdoor Learning** | 2022, NeurIPS D&B | NLP backdoor benchmark | 明确定义 stealthiness 为 **避免自动或人工检测**，并把它与 validity 分开 | ∆PPL、∆GE、USE、ASR、CACC、poison rate、label consistency | 是 | 是 | 是，perplexity/grammar/semantic similarity | citeturn13view1turn38view0turn39view1 |
| **BackdoorBench: A Comprehensive Benchmark of Backdoor Learning** | 2022, NeurIPS D&B | backdoor benchmark | 不把 stealthiness 单独形式化，但把 **C-Acc、ASR、R-Acc** 作为统一协议，并系统比较攻防对 | C-Acc、ASR、R-Acc | 是 | 是 | 否 | citeturn14view2turn19search1 |
| **TrojanZoo: Towards Unified, Holistic, and Practical Evaluation of Neural Backdoors** | 2022, benchmark 补充文献 | backdoor benchmark | 把攻击目标拆成 **effectiveness、evasiveness、transferability**，并用 **normalized CAD-ASR trade-off** 讨论 effectiveness–evasiveness 关系 | ASR、CAD、normalized ASR、normalized CAD | 是 | 是 | 间接 | citeturn36view0turn35view0 |
| **SCALE-UP** | 2023, ICLR | 输入级后门检测 | detectability = **scaled prediction consistency**；仅通过预测标签进行黑盒检测 | input-level detection score、TPR/FPR 型评估 | 是 | 否 | 否 | citeturn14view3 |
| **A Unified Detection Framework for Inference-Stage Backdoor Defenses** | 2023, NeurIPS | 输入级/统一检测 | 明确把检测问题写成 **在给定 FPR 下最大化 detection power**；强调 AUCROC 与 FPR-aware 评估 | detection power、AUCROC、fixed-FPR guarantees | 是 | 否 | 否 | citeturn43search0turn43search2 |
| **IBD-PSC** | 2024, ICML | 输入级后门检测 | detectability = **parameter-oriented scaling consistency**，作为“firewall”过滤恶意测试样本 | input-level detection performance | 是 | 否 | 否 | citeturn14view4 |

## 关键定义与证据

最清晰的显式定义来自 **NLP backdoor 评测文献**。Cui 等人在 NeurIPS 2022 明确写道，已有指标只看 ASR 和 CACC，“忽略了 poisoned samples 也应当是 stealthy and semantic-preserving”；他们进一步把 **stealthiness** 定义为“避免自动或人工检测”，并用 **平均 perplexity increase（∆PPL）** 与 **grammar error increase（∆GE）** 进行量化，同时用 **USE similarity** 衡量 validity。这个定义是我在已发表文献里看到最完整、最适合直接借鉴到论文写作的表述之一。citeturn38view0turn39view1turn39view2

Yang 等人在 ACL 2021 则进一步指出，仅靠 Clean Acc 与 ASR 不够，因为攻击还可能对**系统部署者**不隐蔽，或者对**系统用户**不稳定。他们因此提出两个额外指标：**DSR** 用来衡量“trigger 在输入里藏得有多自然”，而 **FTR** 用来衡量“良性用户被误触发的概率”。该文把 FTR 形式化为，给定一个假触发 \(S\)，其 FTR 就是在非目标类样本上把模型打到目标类的概率；对真实触发 \(T\)，则取其常见子序列的平均 FTR。换言之，这篇论文把“隐蔽性”从单纯“看不见”推进到“**不易被部署者发现，也不容易被普通用户意外激活**”。citeturn17view0turn16view3

在视觉 backdoor 文献中，**ISSBA** 提供了一个很有价值的区分：攻击者目标包括 **effectiveness、stealthiness、sustainability**。其中，**effectiveness** 要求触发后输出攻击者想要的目标标签且 benign performance 不显著下降；**stealthiness** 要求 trigger 被隐藏且 poisoning rate 足够小；**sustainability** 则要求攻击在常见 backdoor defenses 下依然有效。这个分法很重要，因为它说明很多论文口中的“stealthy”其实混合了两层意思：一层是人看不见，另一层是 defense 检不出；而 ISSBA 明确把后者单独拎出来叫 sustainability。citeturn26view2

**WaNet** 和 **LIRA** 代表了另一种常见写法：把 stealthiness 理解为 **imperceptibility + defense evasion**。WaNet 直接宣称其 warping trigger 在 **human inspection test** 中明显优于以往方法，并通过“noise mode”让模型绕过现有防御；LIRA 则把目标写成：学习一个“optimal, stealthy trigger injection function”，既要用 **imperceptible noise** 操作输入，又要 **preserve the model performance on clean data**、同时最大化 poisoned data 上的 ASR，并最终绕过现有 defenses 和人工检查。citeturn13view3turn45search5turn13view2

**Neural Cleanse** 与 **STRIP** 则定义了后门的“可检测性”应如何测。Neural Cleanse 的核心是：对每个目标类反向搜索最小 trigger，并用 **MAD-based anomaly index** 判断某个类是否异常；搜索结果里明确写出 anomaly index 是“absolute deviation divided by MAD”。STRIP 的定义则完全不同：对在线输入叠加各种随机 pattern，如果叠加后预测类别的**熵仍然很低**，那就是 trojaned input；它进一步用 **FAR / FRR** 来评估运行时检测性能。它们共同说明：**detectability 几乎总是 defense-specific 的，而不是 attack 本身天然自带的统一数值。**citeturn21search2turn22search3turn20view1

在 benchmark 层面，**BackdoorBench** 与 **TrojanZoo** 很能说明社区现状。BackdoorBench 采用 **C-Acc / ASR / R-Acc** 作为标准协议；TrojanZoo 则更进一步指出，很多论文只用 ASR 与 CAD，而这“insufficient to describe”攻击在两者之间如何权衡，于是用 **normalized ASR–normalized CAD trade-off** 来讨论 **effectiveness–evasiveness** 的关系。换句话说，**“normalized ASR / normalized CAD” 不是主流单篇攻击论文的标准 stealthiness 定义，但已经出现在系统化 benchmark 中，作为 trade-off 分析工具。**citeturn19search1turn36view0turn35view0

## 对你的九个问题的直接回答

**关于术语定义。**  
在我检索到的主流论文里，**imperceptibility** 最窄，通常指“触发器或扰动在感知上不可见/不自然度低”；**detectability** 最偏防御侧，指样本或模型是否会被特定 detector 抓住；**evasiveness** 更像“绕过 detector/defense 的能力”；**stealthiness** 则是最宽的总括词，经常混合“高 clean performance”“人类不易察觉”“检测器不易发现”“不易误触发”等多种诉求。NLP 评测论文和 ISSBA 是少数把这些边界说得很清楚的工作。citeturn39view1turn17view0turn26view2

**关于后门攻击中隐蔽性的常见维度。**  
你列出的六类维度几乎都能在文献中找到，而且确实是社区常见分解：**clean accuracy / benign accuracy / clean accuracy drop** 是最常见的 utility proxy；**trigger perceptibility / input naturalness** 在视觉和 NLP 中都很常见；**poisoned sample detectability** 对应样本级检测；**model-level backdoor detectability** 对应 Neural Cleanse 这类 reverse-engineering / anomaly 检测；**TPR/FPR/AUC/AP/Recall/F1** 是检测器性能指标；**false trigger rate / unintended activation** 在 NLP 里最成熟。唯一要补充的是：很多视觉论文并不会同时覆盖这六项，而是只报其中二到三类。citeturn28view0turn13view2turn13view3turn20view1turn22search3turn17view0turn39view1

**哪些论文把隐蔽性理解为“逃避检测防御的能力”。**  
最明确的一类是 WaNet、Input-Aware Dynamic、LIRA、WB、以及 ISSBA 的 sustainability 维度。它们都把“能否绕过 Neural Cleanse / STRIP / 现有 defenses / machine inspection”写进论文主张里。需要强调的是，这些论文有时把这件事叫 stealthiness，有时叫 bypass，ISSBA 则将其更精确地称为 **sustainability**。如果你做方法学定义，最好不要把“人类难察觉”和“防御难检测”混成一个词不加区分。citeturn13view4turn13view3turn13view2turn14view1turn26view2

**哪些论文把隐蔽性理解为“保持干净样本性能不下降”。**  
Poison Frogs 明确强调 targeted poisoning 可以“without degrading overall classifier performance”；LIRA 明确要求 preserve clean-data performance；BackdoorBench 用 C-Acc 作为统一协议；一些综述甚至把 clean accuracy / CACC 看作评价 attack stealthiness 的最常见指标之一。不过更严格地说，**主流 benchmark 更倾向把 clean accuracy 看作 effectiveness/utility，而不是 stealthiness 的全部**。因此，clean accuracy 适合作为 stealthiness 的必要辅助指标，而不是充分定义。citeturn28view0turn13view2turn19search1turn7search4

**哪些论文把隐蔽性理解为“触发器或样本自然/不可见”。**  
视觉里，WaNet 采用 **human inspection test**；LIRA 使用 **imperceptible noise**；ISSBA 把隐蔽性写成触发器被隐藏、poison rate 小；Label-Consistent Backdoor 强调 poisoned inputs 在 human inspection 下看起来 benign；NLP 里，Unified Evaluation 采用 **∆PPL、∆GE、USE**；ACL 2021 的 DSR 也本质上是自然性/暴露性的检测 proxy。更靠近“感知质量”范式的指标如 **SSIM、PSNR、LPIPS、FSIM、FID**，我在较新的 top-venue 工作里确实看到，例如 CVPR 2025 的 SSL invisible backdoor 和 ECCV 2024 的事件相机 backdoor 会显式使用这些指标；但要注意，**它们并不是早期经典 image-classification backdoor 论文的统一标准**，早期更常见的是 human study / human inspection。citeturn45search5turn13view2turn26view2turn30search1turn39view1turn17view0turn46search0turn46search18

**是否有论文使用 `1 - TPR`、`1 - detection rate`、低 Recall、低 F1、低 AUC、低 anomaly score 来表示更高隐蔽性。**  
在我检索到的主流论文里，**很少有人把这些量统一改写成一个叫 stealthiness 的标量**。更常见的写法是：直接报告 detector 的原始结果，然后在文字上解释“更难检测，因此更 stealthy”。比如 Neural Cleanse 用 **anomaly index**，STRIP 用 **FAR/FRR**，统一检测框架用 **AUCROC 与 fixed-FPR detection power**，NLP stealthiness 论文用 **DSR/FTR**。因此，把 stealthiness 写成 `1-TPR` 或 `1-AUC` 是**合理的研究者自定义 operationalization**，但它不是社区既有术语。若你这么做，最好在文中明说你定义的是 **detector-evasion stealthiness**。citeturn21search2turn20view1turn43search0turn43search2turn17view0

**多个防御同时存在时，文献通常如何聚合。**  
主流论文通常**不做单一聚合值**，而是**按防御分别报告**，或者像 BackdoorBench、TrojanZoo 那样做成 attack–defense pair 的大规模比较与 trade-off 图。TrojanZoo 的 normalized CAD–ASR 图说明了如何做“二维 trade-off 分析”，但它也没有把所有 detector 直接平均成一个 stealthiness 常数。换言之，**分别报告与 Pareto/trade-off 分析是社区主流；平均 detection rate 只是你可以引入的衍生定义，不是既有标准。**citeturn14view2turn36view0turn35view0

**如果要研究“迁移性与隐蔽性的关系”，更推荐哪种 stealthiness 定义。**  
我更推荐把 stealthiness 的**主定义**放在**防御侧可检测性**上，即“在固定 FPR 下跨多个 detector 的平均漏检率”，因为它最接近“逃避检测”的安全直觉；同时必须把 **clean accuracy drop** 和 **模态感知自然性** 作为辅助指标单独控制。原因是 clean accuracy 本身更像 utility，不足以表达隐蔽；而纯 trigger perceptibility 又不能反映是否会被 Neural Cleanse/STRIP/SCALE-UP/IBD 等自动防御抓住。这个多轴定义也更有利于与你之前关注的 transferability 做相关性或因果分析，因为两者概念上更正交。citeturn43search0turn43search2turn13view3turn13view2turn39view1

**你当前的定义是否合理。**  
如果你的定义是  
\[
\text{stealthiness}=1-\text{mean}(\text{TPR}_{\text{SentiNet}},\text{TPR}_{\text{SCALE-UP}},\text{TPR}_{\text{STRIP}},\text{TPR}_{\text{IBD}})
\]
那么它**作为一个 detector-suite aggregate 是合理的**，但前提非常关键：这些 TPR 必须在**相同的 operating point** 下比较，最理想是都在同一 **FPR \(\alpha\)** 下得到，即 \(\text{TPR}_i@\text{FPR}=\alpha\)。统一检测框架明确把问题写成“给定 FPR 最大化 detection power”，STRIP 也是在 preset FRR/FAR 下报告结果；这说明**阈值与误报约束必须统一**，否则直接平均 TPR 不可比。命名上，我建议不要直接叫它 *stealthiness*，而应叫 **mean detector-evasion rate**、**aggregate defense-evasion stealthiness**，或者更中性的 **aggregate detectability complement**。同时，我建议再补三样：  
\[
\text{strict stealthiness}=1-\max_i \text{TPR}_i@\text{FPR}=\alpha
\]
用来表示“最强 detector 下的最坏情况”；  
\[
\text{geo-stealthiness}=\Big(\prod_i (1-\text{TPR}_i@\alpha)\Big)^{1/K}
\]
用来惩罚某一 detector 明显抓得到的情况；  
以及 **clean accuracy drop** 与 **perceptibility/naturalness** 指标，用来防止你的 stealthiness 只剩下“绕过这四个 detector”的狭义含义。若能拿到完整 ROC/PR 曲线，再额外报告 **AUCROC / AUCPR** 会更稳健，但在部署解释上，我仍更偏向固定 FPR 的 miss-rate。citeturn43search0turn43search2turn20view1turn14view3turn14view4turn24search0

## 论文中如何定义 stealthiness

如果你要在论文里给出一个**主定义 + 辅助指标**，我建议写成下面这种结构，比直接写一个单值更符合主流文献，也更容易说服审稿人：

**主定义：检测器规避型隐蔽性**  
设检测器集合为 \(\mathcal{D}=\{D_1,\dots,D_K\}\)，统一在固定误报率 \(\alpha\) 下取各 detector 的 TPR，则
\[
S_{\text{evasion}}(\alpha)=1-\frac{1}{K}\sum_{i=1}^{K}\text{TPR}_{D_i}@\text{FPR}=\alpha.
\]
这个量越大，表示样本越难被当前 defense suite 检出。它与最近把检测问题写成“给定 FPR 下最大化 detection power”的表述一致，也与 STRIP 这类预设 FRR/FAR 的评测方式兼容。citeturn43search0turn43search2turn20view1

**辅助指标：模型正常性**  
再报告  
\[
\text{CAD}= \text{Acc}_{\text{clean model}}-\text{Acc}_{\text{backdoored model}}
\]
或直接报告 clean accuracy / benign accuracy。原因是大量 backdoor/poisoning 论文依然把“正常输入上的性能不受损”看作攻击可接受性的基础条件。citeturn28view0turn13view2turn19search1

**辅助指标：感知自然性**  
按模态选择：视觉分类可报告 human inspection，必要时补 SSIM/PSNR/LPIPS；文本任务建议至少报 \(\Delta\)PPL、\(\Delta\)GE、semantic similarity；如果 trigger 可能在自然样本中出现，还应补 **FTR/误触发率**。这三类指标各自回答的是不同问题：**是否容易被机器检测、是否影响干净功能、是否在感知上像正常样本**。把它们混成单一数值会损失解释力。citeturn45search5turn39view1turn17view0turn46search0

因此，我对你当前方案的具体建议是：

第一，**把现有公式改名**。  
不要直接叫 `stealthiness`，建议叫 **Defense-Evasion Stealthiness**、**Detector-Evasion Score** 或 **Mean Evasion Rate**。这样术语更精确，也能避免和文献中“imperceptibility / clean utility / low poison rate”这些其他含义混淆。citeturn26view2turn36view0

第二，**使用固定 FPR 校准**。  
把每个 detector 的阈值都调到相同 FPR，例如 1% 或 5%，再平均 TPR。若不做这一步，平均 TPR 没有严格可比性。citeturn43search0turn43search2turn20view1

第三，**同时报 strict 版本**。  
主文用平均值，附录报 `strict stealthiness = 1 - max(TPR_i)`。原因很简单：平均值可能被“某三个 detector 都抓不住，但第四个 detector 很容易抓住”的情况掩盖；strict 版本正好能揭示最坏情形。这个做法虽然不是社区标准，但和“按 defense 分别报告”的主流做法是兼容、甚至更保守的。citeturn14view2turn36view0

第四，**无论如何补 clean accuracy drop**。  
如果你的 stealthiness 只由 detector evasion 构成，而 clean accuracy 明显下降，那么读者会自然质疑这种“隐蔽”是否只是以牺牲正常功能换来的。Poison Frogs、LIRA、BackdoorBench 都支持把 clean-side utility 作为并行约束。citeturn28view0turn13view2turn19search1

第五，**如果你的触发器有感知层面的“不可见/自然”主张，再补一个模态指标**。  
视觉可用 human study 或 LPIPS/SSIM/PSNR；文本至少应加 \(\Delta\)PPL、\(\Delta\)GE，若自然触发风险存在，再加 FTR。否则你的 stealthiness 更准确的名字其实只是“defense evasion”，而不是“完整隐蔽性”。citeturn39view1turn17view0turn45search5turn46search0

## 推荐引用与局限

如果你只选 **5–10 篇最权威、最值得在论文中引用** 的文献，我建议优先用下面这些：

**Neural Cleanse**，因为它奠定了模型级 backdoor detectability 的经典写法：reverse-engineering + anomaly index + MAD。citeturn22search3turn21search2

**STRIP**，因为它奠定了输入级 runtime detectability 的经典写法：entropy + FAR/FRR。citeturn20view1

**Input-Aware Dynamic Backdoor Attack**，因为它代表了“动态 trigger = 更强 stealthiness / defense bypass”的关键转折。citeturn13view4

**WaNet**，因为它把“human inspection + machine inspection”同时纳入 stealthiness 叙事，是视觉 backdoor 里最有影响力的 stealthy 攻击之一。citeturn13view3turn45search5

**LIRA**，因为它把 imperceptibility、clean utility、defense bypass 三者合在一个优化目标里，是后续很多“stealthy backdoor”工作的参照物。citeturn13view2

**Invisible Backdoor Attack With Sample-Specific Triggers**，因为它少见地把 effectiveness、stealthiness、sustainability 三者明确区分开来，非常适合用于你的术语梳理。citeturn26view2turn25view0

**Rethinking Stealthiness of Backdoor Attack against NLP Models**，因为它首次把 stealthiness 显式形式化成 **DSR + FTR**，对“误触发/部署者检测”这两个角度都给出了可复用定义。citeturn17view0turn16view3

**A Unified Evaluation of Textual Backdoor Learning**，因为它给出了目前最清楚的“stealthiness / validity / effectiveness”三分法，并把 \(\Delta\)PPL、\(\Delta\)GE、USE 变成规范化评测协议。citeturn39view1turn38view0

**BackdoorBench**，因为它代表了社区最广泛使用的统一 backdoor evaluation protocol，尤其适合你说明 C-Acc、ASR、R-Acc 的标准含义。citeturn14view2turn19search1

**TrojanZoo**，因为它是我看到的少数直接讨论 **normalized ASR / normalized CAD 与 effectiveness–evasiveness trade-off** 的系统化 benchmark，正好能回答你关于“是否存在 normalized 指标”的问题。citeturn36view0turn35view0

最后说两点局限。第一，**社区对 stealthiness 的术语使用并不统一**；有的论文把 defense bypass 也叫 stealthiness，有的论文则把它分出去叫 sustainability 或 evasiveness。第二，**经典视觉分类 backdoor 论文并不常系统报告 SSIM/PSNR/LPIPS/FID**；这些更像是 newer imperceptible / generation / SSL 文献中变得更常见的补充指标。因此，如果你要写一个“统一 stealthiness 定义”，最好明确说明这是**你的 operational definition**，并在实验表中把 detector-evasion、clean utility、perceptual naturalness 三条轴分开呈现。citeturn26view2turn36view0turn46search0turn46search18