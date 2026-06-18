# 安全领域中攻击迁移性 transferability 的定义与度量方式综述

## 检索范围与结论先行

围绕“攻击迁移性”这一概念，机器学习安全文献里最稳定、最主流的共识是：**先在源侧构造攻击，再看它在目标侧是否仍然有效**。对抗样本文献通常把这种有效性写成目标模型上的误分类率、fooling rate、matching rate 或 targeted fooling rate；后门与数据投毒文献则通常写成目标模型或目标数据集上的 attack success rate，外加 clean accuracy 或 benign accuracy。也就是说，**迁移性在绝大多数正式发表论文里，本质上是“target-side effectiveness after transfer”**，而不是 source/target 的比值或某种经过复杂归一化后的单一标量。citeturn22view0turn20view0turn24view0turn26view0turn29view0turn31view0turn45view0

在你关心的几个具体方向里，这一模式非常一致。对抗样本跨模型迁移，多数工作直接报告目标模型上的非定向 fooling rate 或定向 target-class 命中率；跨数据集、跨域对抗攻击也仍然以目标侧 fooling rate 或 targeted fooling rate 为中心。后门攻击中的跨模型、跨数据集、跨域、跨任务迁移，则普遍报告**目标侧 ASR/TFR + clean/benign accuracy**，有些工作进一步补充 clean accuracy drop、robust accuracy、CAD、PSNR/SSIM、以及对检测/剪枝/净化的防御成功率。citeturn26view0turn29view0turn11search0turn11search12turn31view0turn40view0turn42view0turn44search0turn45view0

少数论文确实尝试把“迁移性”定义得比“目标侧 ASR”更正式。最典型的是 Liang 等人在 ICML 2021 提出的两类 adversarial transferability 指标：其一是**相对于目标模型白盒最优攻击效果的比值** α₁，其二是表征输出偏移方向一致性的 α₂；这类工作把 transferability 从“经验指标”提升到了“相对目标白盒上界”和“方向一致性”的形式化对象。另一个例外是在 NLP 预训练后门中，Shen 等人指出**固定插入次数下的 ASR**在长短文本之间不稳定，因此改用“最少需要插入多少次触发器才能触发错误”的 Effectiveness，以及 Stealthiness、Capability。citeturn17view0turn18view0turn41view0

但在本次检视的代表性高优先级论文里，我**没有看到** target/source ratio、target-source difference、log(target/source)、chance-adjusted ASR 这些指标成为通行的“标准定义”。相反，文献的稳定习惯是：**把 raw target-side success metric 作为主指标；把 source-side white-box success、clean accuracy、stealthiness、defense evasion 作为辅指标分开报告**。因此，如果你的目标是写一篇和主流文献可对齐、又能更稳健比较跨数据集/跨域结果的论文，最合适的做法通常不是“只选一个奇特比值”，而是采用**主指标 + 校正指标 + 配套保真指标**的联合报告框架。citeturn20view0turn24view0turn17view0turn31view0turn41view0turn45view0

## 迁移性的核心定义

对抗样本方向的基础定义，可以追溯到 Papernot 等人的黑盒攻击研究。他们把 adversarial sample transferability 定义为：**为某个模型专门构造的对抗样本，也能误导另一个模型**。随后，大量工作沿着这个思路，把“迁移性”具体化为目标模型上的成功率。citeturn22view0

> “some adversarial samples … can mislead other models” citeturn22view0

Liu 等人在《Delving into Transferable Adversarial Examples and Black-box Attacks》中已经把实验层面的指标写得非常清楚：非定向迁移用**源模型生成的对抗样本在目标模型上的 top-1 accuracy**来表示，且“**更低的 accuracy 表示更强的非定向迁移性**”；定向迁移则直接看**目标模型把这些样本判成攻击者指定目标类的比例**，他们称之为 matching rate。这个定义后来几乎被定向迁移攻击文献完整继承。citeturn20view0

> “the percentage … classified as the target label” citeturn20view0

到了更形式化的层面，Liang 等人在 ICML 2021 明确写道：对抗迁移的过程就是“把针对 \(f_1\) 生成的 adversarial example 应用于另一个模型 \(f_2\)”，因此 transferability from \(f_1\) to \(f_2\) 就是**\(\delta_{f_1,\epsilon}\) 对 \(f_2\) 的攻击效果有多好**。他们提出的第一指标可写成

\[
\alpha^{f_1 \to f_2}_1
=
\mathbb{E}_{x \sim D}
\left[
\frac{\ell_{\text{adv}}(f_2(x), f_2(x+\delta_{f_1,\epsilon}(x)))}
{\ell_{\text{adv}}(f_2(x), f_2(x+\delta_{f_2,\epsilon}(x)))}
\right],
\]

也就是：**同样的 perturbation budget 下，源模型攻击转移到目标模型后的效果，占目标模型自身白盒最优攻击效果的多少比例**。同文还提出了第二指标 \(\alpha_2\)，通过目标与源模型输出偏移方向的一致性来刻画迁移性。重要的是，这篇论文本身也解释了 \(\alpha_1\) 可以理解为“**攻击有多经常地成功转移**”，而 \(\alpha_2\) 则编码了方向信息。citeturn17view0turn18view0

后门与数据投毒文献的定义比对抗样本更“任务化”。在视觉后门中，最标准的 ASR 定义几乎都是：**当把触发器注入测试样本后，模型把样本判成攻击者指定目标类的比例**。BackdoorBench 的标准化定义就写得非常直接：C-Acc 是 clean samples 的预测准确率，ASR 是“**poisoned samples to the target class**”的预测准确率，R-Acc 则是 poisoned samples 仍被预测到原始类别的准确率。citeturn31view0

> “prediction accuracy of poisoned samples to the target class” citeturn31view0

在更细的 clean-label backdoor 文献中，Turner 等人给出的定义是：**ASR 是“非目标类测试样本在加上后门模式后，被分类为目标类的百分比”**。这个定义本质上就是 target-side targeted ASR。citeturn33view0

NLP 预训练后门是一个例外：Shen 等人认为固定 trigger count 下的 ASR 在长文本与短文本之间不稳定，于是提出 **Effectiveness \(E\)**：让样本翻转预测所需的**最小触发器插入次数**；**Stealthiness \(S\)**：触发器占文本长度的比例；以及 **Capability \(C = 1/(E \cdot S)\)**，用来综合“越少插入、越短触发器、越长文本中仍能触发”这几件事。也就是说，在这类任务里，作者事实上是在说：**原始 ASR 不够稳定，所以要把“迁移成功”改写成“需要多少最小操作才能触发”**。citeturn41view0

## 后门、投毒与跨场景评估方式

在后门攻击里，**跨模型迁移**最常见的做法，就是把源侧学到的 trigger 或 poisoned representation 转移到另一架构或另一训练流程下，再看目标模型上的 ASR 与 clean accuracy。图神经网络的 Graph Backdoor 明确区分了 non-transfer 与 transfer setting，并在两种 setting 下都报告攻击有效性；TRAP 则把 ASR 直接写成 \(\#\text{successful attacks}/\#\text{total trials}\)，同时报告 CAD（clean accuracy difference）。这说明在“跨模型后门迁移”里，主指标仍然是**目标模型上的 ASR**，辅指标通常是**clean accuracy drop / CAD**。citeturn11search0turn11search12

在**跨数据集迁移**里，BadCLIP 的做法很有代表性：它先在 ImageNet 上学习后门，再把模型拿到 10 个不同下游数据集上测试，表格同时给出 clean ACC 与 target-side ASR；作者特别强调，即便目标数据集和 ImageNet 类别完全不同，ASR 仍然几乎是 100%。这类工作里，“transferability”没有被单独定义成 source/target 比值，而是直接体现在**迁移到目标数据集后还能维持多高的 target-side ASR**。citeturn45view0

在**跨域迁移**里也是同样逻辑。对抗样本文献中，Naseer 等人研究的是“不同域上学到的 perturbation 能不能攻击另一域的模型”，主指标叫 fooling rate；TTAA 研究的是跨域 targeted transfer，主指标叫 TFR；BadCLIP 则在 ImageNet 训练、ImageNetV2 / Sketch / A / R 上测试，主指标还是目标域上的 ASR，同时配 clean ACC。三者虽然名字不同，但测量对象高度一致：**攻击迁移到目标域后的目标侧成功率**。citeturn26view0turn29view0turn30view2turn45view0

在**跨任务迁移**中，情况要稍微更复杂。对于预训练后门，源阶段往往根本没有与你的下游任务可比的一组类别标签，因此“source ASR”常常**不适用或不可比**。例如 Backdoor Pre-trained Models Can Transfer to All 把触发器直接映射到预训练模型的输出表示而不是某个固定标签；NeuBA 也是把 trigger 映射到预定义向量；BadEncoder 直接把 backdoor 注入自监督预训练 encoder，使得“不同下游任务的分类器同时继承 backdoor 行为”；Aliasing Backdoor 甚至横跨图像分类、人脸识别、语音识别任务。对于这些工作，论文真正关心的不是“源任务的 ASR”，而是**下游目标任务 fine-tuning 之后的 target-side ASR 是否高、clean accuracy 是否保持**。citeturn40view0turn41view0turn42view0turn10search0turn44search0

这也解释了你提出的一个很关键的问题：**如果 source ASR 低但 target ASR 高，应该怎么解释？**  
结论是：文献里并没有形成一个统一的术语把它叫做 amplification。更常见的表述是 **transferability、inheritance、survival after transfer learning、universal vulnerability**。在对抗样本文献里，Dong 等人明确讨论过白盒强度与黑盒迁移性之间的 trade-off：白盒越强并不自动意味着黑盒越强；同样，在 ICML 2019 的 transferable poisoning 研究里，还出现过某架构上的 success rate 很低、另一架构上却更高的非对称现象。对预训练后门来说，这种“源低目标高”甚至经常不是异常，而是因为**源阶段优化的是表示层目标，真正可观测的恶意行为是在目标下游任务中才显性化**。因此，更稳妥的写法通常是：**这是 transfer asymmetry / downstream inherited vulnerability，需要单独分析，不建议把它直接命名为 amplification 并替代 transferability**。citeturn24view0turn25view1turn47view0turn40view0turn42view0turn44search0

## 代表性论文总结表格

下表覆盖本次检索中**定义与度量最清晰、且与你的问题最相关**的代表性论文。它不是该领域全部论文的穷尽列表，但足以支撑“transferability 如何定义、如何量化、哪些指标主流、哪些是例外”的核心结论。

| 论文 | 年份与会议/期刊 | 攻击类型 | 迁移场景 | 迁移性定义或操作化方式 | 主要指标 | 是否给源域/源模型指标 | 是否给目标域/目标模型指标 |
|---|---|---|---|---|---|---|---|
| *Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples* citeturn22view0 | 2016，基础性黑盒攻击论文 | 对抗样本 | 跨模型、跨训练集 | transferability = 为模型 \(F\) 构造的 adversarial samples 也能误导模型 \(G\)；实验上用“被目标模型误分类的比例”衡量 | transferability rate / misclassification rate | 一般不单独做 canonical 的“source ASR” | 是；目标侧转移误分类率是主角 |
| *Delving into Transferable Adversarial Examples and Black-box Attacks* citeturn20view0 | 2017，基础性正式传播版本来自预印本 | 对抗样本 | 跨架构、定向/非定向 | 非定向：看目标模型在源生成 AEs 上的 accuracy，越低越可迁移；定向：matching rate = 目标模型输出指定目标类的比例 | accuracy、matching rate、RMSD | 是；源模型上的 white-box 效果可见 | 是；目标模型上的 accuracy / matching rate 是核心 |
| *Boosting Adversarial Attacks With Momentum* citeturn24view0turn25view1 | 2018，CVPR | 对抗样本 | 跨模型黑盒 | 迁移性体现在 hold-out black-box model 上的 success rate；同时比较 white-box 成功率与 black-box 成功率 | success rate | 是；白盒 success rate | 是；黑盒 success rate |
| *Cross-Domain Transferability of Adversarial Perturbations* citeturn26view0 | 2019，NeurIPS | 对抗样本 | 跨域、跨数据集、跨模型 | 研究“在源域上学到的 perturbation 是否能 fool 目标域网络”；以目标域模型上的 fooling rate / untargeted attack success (%) 表示 | fooling rate / attack success (%) | 是；同时报告 white-box 与 black-box | 是；目标域 fooling rate 是核心 |
| *Uncovering the Connections Between Adversarial Transferability and Knowledge Transferability* citeturn17view0turn18view0 | 2021，ICML | 对抗样本 | 跨模型 | 形式化定义 α₁、α₂；α₁ 是“源攻击对目标的效果 / 目标白盒最优攻击效果”，α₂ 是输出偏移方向一致性 | α₁、α₂、α₁\*α₂ | 不用 source ASR；用源攻击与目标白盒攻击的相对量 | 是；α₁ 直接规范化到目标白盒上界 |
| *Towards Transferable Targeted Adversarial Examples* citeturn29view0turn30view2 | 2023，CVPR | 定向对抗样本 | 跨模型、跨域 | Targeted transfer success = 目标模型把 \(x^{adv}\) 判成攻击者指定 \(y_t\) 的比例；作者定义 TFR = \(\frac{1}{N}\sum 1[f(x^{adv})=y_t]\) | TFR | 是；source=target 时可看白盒 TFR | 是；跨模型/跨域 TFR 是主指标 |
| *Transferable Clean-Label Poisoning Attacks on Deep Neural Nets* citeturn46view0turn47view1 | 2019，ICML | 数据投毒 | 跨模型、transfer learning、end-to-end | 目标是让 victim 在训练后把 target sample 判到 attacker-chosen class；“attack success”就是目标样本被判到目标类 | targeted attack success rate | 一般不把“source ASR”做成标准主指标 | 是；victim 训练后的目标样本成功率是主指标 |
| *Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability* citeturn9search3turn9search0 | 2021，IEEE EuroS&P | 数据投毒 | 迁移学习、对 unseen images 的迁移 | 以 poisoning 后 victim 的 target-side success 为主，并额外看对同一物体 unseen views 的 transferability | attack success rate、to unseen images 的 transferability | 常见是有 surrogate/source attack setting 描述 | 是；victim / unseen-image success 是重点 |
| *Latent Backdoor Attacks on Deep Neural Networks* citeturn35search0turn38search1turn40view0 | 2019，ACM CCS | 后门攻击 | transfer learning、跨下游模型/任务 | latent backdoor 是先嵌到 Teacher，再通过 transfer learning 继承到 Student；评估重点是 Student 上 trigger 的 target-side success 与 clean accuracy | student model ASR、student accuracy、pruning 后 ASR | 源侧通常不是标准可比 ASR；更像 teacher-to-student inheritance | 是；Student 侧是关键 |
| *Backdoor Pre-trained Models Can Transfer to All* citeturn40view0turn41view0 | 2021，ACM CCS | 预训练后门 | 跨任务、跨模型、跨下游数据集 | 传统 ASR 在 NLP 中不稳定；作者定义 Effectiveness \(E\)=最少 trigger 插入次数，Stealthiness \(S\)=trigger 占比，Capability \(C=1/(E\cdot S)\)；也可附 ASR | E、S、C、ASR、clean accuracy | 源预训练阶段通常无可比 source ASR | 是；下游任务上的 misclassification/ASR 是核心 |
| *Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks* citeturn42view0 | 2021 预训练后门工作 | 预训练后门 | 跨 NLP/CV、跨下游任务 | 把 trigger 映射到预定义表示；真正评估发生在 fine-tuned downstream tasks 上，作者强调“nearly 100% ASR”且 clean performance 受影响小 | ASR、clean performance、pruning 后表现 | 源侧多为表示层目标，不是标准 source ASR | 是；下游任务 target-side ASR |
| *BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning* citeturn10search0turn10search3 | 2022，IEEE S&P | 预训练后门 | 跨下游任务 | 把 backdoor 注入 encoder，使“不同 downstream tasks simultaneously inherit” 后门行为；核心看 downstream classifiers 上的 ASR 与 accuracy | downstream ASR、downstream accuracy | 源预训练阶段无统一 source ASR | 是；下游目标任务指标为主 |
| *Graph Backdoor* citeturn11search0 | 2021，USENIX Security | 图后门 | non-transfer 与 transfer setting；跨模型 | 在 transfer setting 中考察预训练 GNN 的 backdoor 到下游 classifier 的继承；报告 attack effectiveness 与 accuracy drop | ASR、misclassification confidence、accuracy drop | 有 transfer / non-transfer 两种 setting 对照 | 是；transfer setting 目标侧效果是重点 |
| *Transferable Graph Backdoor Attack* citeturn11search12 | 2022，补充性图后门论文 | 图后门 | 跨模型黑盒 | 显式把 ASR 定义为 \(\#\text{successful attacks}/\#\text{total trials}\)，并用 CAD 衡量 clean accuracy difference | ASR、CAD | 一般会给 surrogate/source model 设定 | 是；target GNN 上的 ASR |
| *Aliasing Backdoor Attacks on Pre-trained Models* citeturn44search0turn44search1 | 2023，USENIX Security | 预训练后门 | 跨图像分类、人脸识别、语音识别任务 | 强调 backdoor “transfers to all student models fine-tuned from them”；核心度量是各任务目标侧 success rate 与 clean utility | success rate、clean utility | 源侧通常不是语义可比的 ASR | 是；各 student task 上 success rate |
| *BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP* citeturn45view0 | 2024，CVPR | 后门攻击 | 跨数据集、跨域、跨任务 | 在 seen/unseen classes、cross-dataset、cross-domain、retrieval task 上统一用目标侧 ASR；同时报告 clean ACC；还报告 PSNR/SSIM | ACC、ASR、PSNR、SSIM、retrieval 指标 | 是；源 ImageNet 上有 source ASR/ACC | 是；目标数据集/目标域/目标任务上的 ASR 是核心 |

从这张表可以看出三个非常稳定的事实。第一，**主流定义始终围绕 target side**。第二，**source-side 指标往往只是参考白盒性能，不是 transferability 的核心定义本身**。第三，只有少数论文——比如 Liang 2021 和 Shen 2021——会显式引入相对量或替代量，但它们都没有取代 raw target-side success metric 的主导地位。citeturn17view0turn41view0turn45view0

## 对八个重点问题的直接回答

**问题一：transferability / transfer rate / transfer ASR / attack transferability 在这些论文中如何定义？**  
最普遍的定义是：**源侧构造的攻击，在目标侧仍然成功的程度**。非定向对抗样本常用 fooling rate 或“目标模型在 adversarial inputs 上的准确率下降”；定向对抗样本常用 matching rate / TFR；后门与投毒则用目标模型在触发输入上的 ASR。形式化最强的是 Liang 等人的 α₁、α₂，但那是少数例外。citeturn20view0turn17view0turn29view0turn31view0

**问题二：后门攻击中，跨模型、跨数据集、跨域、跨任务迁移性分别如何评估？**  
跨模型常用**目标模型 ASR + clean accuracy/CAD**；跨数据集常用**目标数据集上的 ASR + ACC**；跨域常用**目标域上的 ASR/TFR + ACC**；跨任务则通常在 fine-tuned downstream task 上评估 ASR、accuracy、必要时再加 retrieval/NER 等任务特定指标。对于预训练后门，源阶段常常并没有与你的下游标签空间可比的 source ASR，因此文献会把迁移性完全落在 downstream target-side evaluation 上。citeturn11search0turn11search12turn40view0turn42view0turn44search0turn45view0

**问题三：常见度量是否主要使用 target-side attack success rate？**  
是。无论名字叫 ASR、TFR、fooling rate 还是 matching rate，绝大多数论文都是在测**攻击迁移到目标模型/目标数据集/目标域后的目标侧成功率**。source-side white-box 成功率经常会报告，但通常只是“参考上界”或“surrogate effectiveness”，不是 transferability 的主定义。citeturn20view0turn24view0turn26view0turn29view0turn31view0turn45view0

**问题四：是否有论文使用 source ASR 与 target ASR 的比值、差值、log ratio、relative improvement、normalized ASR、chance-adjusted ASR？**  
高优先级正式论文里，**source/target ratio、difference、log ratio、chance-adjusted ASR 不是主流标准**。最接近你的设想的是 Liang 2021 的 α₁：它不是 source/target ASR 比值，而是**转移到目标后的 adversarial loss 相对于目标白盒最优 adversarial loss 的比值**。另一类例外是 Shen 2021，它放弃固定 trigger-count ASR，转而使用 E/S/C。至于 normalized ASR 或 chance-adjusted ASR，在本次检视的代表性论文中，我没有看到它们成为高优先级 venues 的通行 transferability 定义。citeturn17view0turn18view0turn41view0turn45view0

**问题五：对于 targeted attack，是否将“目标模型输出攻击者指定目标类的比例”作为迁移成功？**  
是，这几乎就是标准做法。Liu 2017 的 matching rate、Wang 2023 的 TFR，都是把**目标模型输出指定 target class 的比例**定义为 targeted transfer success。后门攻击里的 targeted ASR 也是一样：只不过触发条件是先把 trigger 插进去。citeturn20view0turn30view2turn31view0turn33view0

**问题六：对于 backdoor attack，是否还同时报告 clean accuracy / benign accuracy / clean accuracy drop / detection rate / stealthiness？**  
是。clean accuracy 或 benign accuracy 几乎是后门论文的标配；一些工作还报告 CAD、R-Acc、PSNR/SSIM、trigger 可察觉性、以及在 defense 下的 ASR 变化或检测/剪枝后残余后门强度。BackdoorBench 的 C-Acc / ASR / R-Acc 非常有代表性；Graph/CLIP/NLP 预训练后门则进一步扩展到 CAD、stealthiness、PSNR/SSIM、Effectiveness/Stealthiness/Capability 等。citeturn31view0turn11search12turn41view0turn45view0

**问题七：如果 source ASR 较低但 target ASR 较高，文献中通常如何解释？**  
通常还是放在 **transferability / inherited vulnerability / transfer asymmetry** 的框架下分析，而不是另起一个被广泛接受的新术语。对抗样本文献里，白盒强度和黑盒迁移性并不单调对应，Dong 2018 甚至明确讨论了 trade-off；投毒文献里也能见到某 surrogate 架构成功率很低、另一些目标架构更高的非对称现象。对预训练后门来说，这类情况更常见，因为源阶段优化的往往不是下游分类标签，而是中间表示或预定义向量，所以“source ASR”要么不适用，要么不具备可比性。此时更稳妥的写法是：**后门在下游适配后显性化，体现为 transferability / inheritance，而非 amplification 的通用标准术语**。citeturn24view0turn25view1turn47view0turn40view0turn42view0turn44search0

**问题八：每篇论文应给出什么字段？**  
从文献复现实务看，最必要的字段是：**标题、年份、venue、攻击类型、迁移场景、成功判定规则、主指标、source-side 是否有指标、target-side 是否有指标、clean/benign utility、是否有 stealthiness/defense 指标**。上面的总结表已经按这个框架组织。对于你后续自己做系统综述，建议额外再加两列：**label-space 是否一致**、**source-stage ASR 是否语义可比**。这两列对于区分“跨模型分类迁移”和“预训练后门跨任务迁移”特别重要。citeturn17view0turn31view0turn40view0turn45view0

## 对研究场景的指标建议

如果你的研究对象是**同一任务、同一标签空间**下的跨模型攻击迁移，我建议把 **transfer_asr** 作为主指标。原因很简单：它与大多数正式发表论文最兼容，读者一眼就能把你的结果与 matching rate、TFR、ASR、fooling rate 对齐。与此同时，建议**总是同时报告 source-side white-box ASR/TFR 与 target-side transfer ASR/TFR**，以便区分“攻击本身弱”与“攻击强但不迁移”这两件事。citeturn20view0turn24view0turn29view0turn31view0

如果你的研究涉及**跨数据集、跨域，甚至类别数不同、类别分布偏斜**，那么我建议把 **chance-adjusted transfer ASR** 作为**辅主指标**，而不是替代 raw transfer ASR 的唯一指标。一个可操作的定义是：

\[
\text{caTASR}
=
\frac{\text{TASR} - p_0}{1-p_0},
\]

其中 \(p_0\) 不是机械地取 \(1/K\)，而是建议取**目标模型在同一评测集上、无攻击或随机攻击时命中目标类的经验基线概率**。这样做的好处是：跨域时类别偏置、target class 先验、以及目标模型本身的类别偏好，都能被部分校正。这个指标在我检视的代表性论文里**不是通行标准**，但很适合你要做的跨数据集/跨域比较。它最好的使用方式，是和 raw target-side ASR 并排报告，而不是替代 raw ASR。这个建议是基于现有文献度量习惯做出的统一化设计。citeturn26view0turn29view0turn45view0

至于 **target/source ratio** 或 **log(target/source)**，我的建议是：**不要把它们当主定义，只把它们当诊断量**。原因有三个。第一，主流论文几乎不这么做，读者难对齐。第二，当 source ASR 很低、为零、或者在预训练场景中根本不可比时，这两个量会数值爆炸或失去意义。第三，它们会把“source attack construction quality”与“target vulnerability”混在一起，解释难度很高。只有在**同标签空间、同任务、同样本集、source ASR 明确可比且不太低**时，比值或 log ratio 才可以作为次级诊断。这个判断与 Liang 2021 选择“相对于目标白盒最优上界的比值”而不是“target/source ASR 比值”是一致的：**真正稳定的归一化对象应该是目标侧上界或目标侧机会基线，而不是 source ASR 本身**。citeturn17view0turn18view0turn40view0

如果你一定希望给出一个**单标量综合分数**，我建议不要直接用 ratio，而是用一个**联合指标**，同时考虑目标侧迁移成功与模型保真度。一个实用方案是，在论文正文中仍把 raw 指标分开报告，但在附录或总表中给出：

\[
\text{JointScore}
=
\mathrm{HarmonicMean}\big(\text{caTASR},\,1-\text{nCAD}\big),
\]

其中 nCAD 是归一化后的 clean accuracy drop。这样做比单独用 ratio 更稳，因为它不会把“source 低导致 ratio 爆炸”的问题引进来，而且与后门文献中“高 ASR 同时保持高 clean accuracy”的评价惯例相符。需要强调的是，这个 JointScore 是**我基于现有文献实践给你的建议性新定义**，适合做总表排序或 ablation 汇总，不适合作为唯一主结论。相关文献本身更倾向于把 ASR/TFR 与 C-Acc/CAD 分开报告。citeturn31view0turn11search12turn45view0

综合起来，如果我替你定一个最稳妥的论文写法，我会建议：

- **主指标**：raw target-side transfer ASR / TFR  
- **跨域辅指标**：chance-adjusted transfer ASR  
- **诊断指标**：source-side ASR/TFR、必要时附 target/source ratio 或 log ratio  
- **保真度指标**：clean accuracy / benign accuracy / CAD / R-Acc  
- **隐蔽性指标**：按模态选 PSNR/SSIM、文本 trigger ratio、或 detection evasion 指标  

这套写法与现有论文最容易对齐，又能解决你研究里“跨数据集/跨域结果不好比较”的真实问题。citeturn20view0turn29view0turn31view0turn41view0turn45view0

## 推荐引用与局限

如果你要在论文里只保留 **5–10 篇最权威、最能支撑定义与度量** 的核心引用，我建议优先选下面这些：

Papernot, McDaniel, Goodfellow 的 *Transferability in Machine Learning*，因为它是“source-crafted attack can fool target model”这一经典黑盒迁移定义的基础来源。citeturn22view0

Liu, Chen, Liu, Song 的 *Delving into Transferable Adversarial Examples and Black-box Attacks*，因为它把 non-targeted 与 targeted 迁移的实验指标分得最清楚：accuracy、matching rate。citeturn20view0

Dong 等人的 *Boosting Adversarial Attacks With Momentum*，因为它奠定了后来很多“黑盒迁移成功率”论文的 white-box / black-box success rate 报告范式。citeturn24view0turn25view1

Liang 等人的 ICML 2021 论文，因为它是少数真正**形式化定义 transferability** 的工作，尤其适合回答你关于 ratio、normalized measure 是否存在的问题。citeturn17view0turn18view0

Zhu 等人的 *Transferable Clean-Label Poisoning Attacks on Deep Neural Nets*，因为它代表了数据投毒场景中“迁移成功”的标准写法：victim 训练后 target sample 被推到 attacker-chosen target class。citeturn46view0turn47view1

Yao 等人的 *Latent Backdoor Attacks on Deep Neural Networks*，因为它是“后门在 transfer learning 中继承”的代表作，直接对应你问的跨任务/跨模型后门迁移。citeturn35search0turn38search1turn40view0

Shen 等人的 *Backdoor Pre-trained Models Can Transfer to All*，因为它一方面回答了“跨任务迁移性如何定义”，另一方面提出了 E/S/C 这套**非常值得借鉴的非 ASR 替代指标**。citeturn40view0turn41view0

Jia 等人的 *BadEncoder*，因为它把“encoder-level backdoor 跨多个 downstream tasks simultaneously inherit”说得最直接，是自监督/基础模型后门迁移的重要代表。citeturn10search0turn10search3

Wei 等人的 *Aliasing Backdoor Attacks on Pre-trained Models*，因为它把后门迁移扩到了图像分类、人脸识别、语音识别等异质任务，强调“transfers to all student models fine-tuned from them”。citeturn44search0turn44search1

Bai 等人的 *BadCLIP*，因为它同时覆盖 seen/unseen classes、cross-dataset、cross-domain、cross-task retrieval，是当前最完整的目标侧 ASR + ACC + stealthiness 报告范式之一。citeturn45view0

本次综述也有几个需要明说的局限。第一，少数基础性论文的出版社页面无法稳定抓取全文，因此我使用了作者公开 PDF、arXiv 版本或由后续正式论文对其定义的精确转述来补全。第二，“chance-adjusted ASR”并不是我在高优先级正式论文里找到的通用标准，而是基于现有文献的 target-side ASR 习惯，为你的跨数据集/跨域研究场景提出的**建议型定义**。第三，在预训练后门与跨任务后门里，source-side ASR 往往没有统一语义，因此任何 source/target ratio 类指标都必须谨慎解释；如果你后续要做非常严格的比较，最好先把研究对象分成“同任务可比”和“跨任务不可比”两组，再分别定义指标。citeturn17view0turn40view0turn42view0turn45view0