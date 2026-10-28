- Role Sensitive neurons
  - 最初目的：找到模型中对应身份切换的neurons
  - 实验过程：Expert vs. Non-expert -> Difference mean -> Top neurons 
  - 结果：
    - Role Sensitive neurons主要调控的是confidence/ willingness，而非Acc
    - 在ARC-E/ CommonSense-E上验证了RSN泛化性（RNS从MMLUE得到并且固定）；
    - confidence以logits作为代表，同时验证了vocabalized confidence（willingness）
    - 验证了一些其他的neurons selection的方法，比如t-test
    - 提出Dopamine的类比 -> 选择信心与承诺；
    - 对应的mask可转移至Base Model；
    - 双向干预
  - 存在的问题：
    - confidence/ wanting/ willingness没有进行区分；
    - 对Dopamine的类比只是提出了假设，没有进行深入的论证；
    - Role > Expert，没有验证是不是所有的角色切换都会走这些neurons。因为对neurons的overlapping解释不足，无法对单个的neurons进行消融或者说明；激活的neurons有交叉但是不完全相同，但是能达到同样的效果，只是需要的scaling不同（eg. teacher - student）；同时neurons存在“集群效应“，只需要一定量的neurons就可以实现效果的切换。
    - 存在neurons集中在特定index的现象（比如有的index可以有达到20层的重复）但是找不到解释的原因，或许应该从MLP weight里面或者attention head来进行解释？
    - 与confidence neurons的差异没有进行解释（即，如果我们落点在confidence，为什么不用 confident - unconfident）
  - 其他做过的尝试：
    - Model Analysis
      - 目的：原本想要探索什么模型会出现RSN的效果
      - 假设需要两个条件：①经过人类反馈 RLHF，②原本训练的资料集中有Role对应的dataset。这两个假设都对应到Role Sensitive是后天习得的本能，在模型中存在切换系统（类似人脑中存在自我身份认知的切换）
      - 结果：有一些效应但是没有办法完全对应；Phi是一个比较好的反例，但是也存在很多例外。
    - Logits lens：没有找到有价值的结果

- Behaviour
  - 设想：完善对于多巴胺的假设，假设RSN的调控能够在行为学上达到类似人类进行多巴胺实验的效果；最初设计的主要实验是Bandit和Gamble。
  - 依据：自我认知决定对任务的投入程度；自我认知是社会化的成果（与他们的连接，强化学习）-> 管理身份切换的neurons与Dopamine相关
  - 存在的问题：
    - 模型在很多问题上不能胜任，比如Bandit；经过调研，大小模型都无法完成探索这一步，原因可能是模型被训练的非常的理性；
    - 模型存在心口不一的问题，即选择的方式没有办法探索模型真正的内心所想，也就是模型很熟悉大体的方式，比如询问模型是否愿意的时候，模型都会表还愿意投入；只有在模型没有意识到自己在被test的时候，或许才能知道模型真正在思考什么，比如reasoning的前置推理过程；
    - 模型对prompt敏感，不同的prompt会得到完全不同的结果；当前的一些实验也是在多次尝试prompt之后才得到规律，这和之前两点或许也有关系；实际上不应该是这样。
    - 以上可能都是IT Model存在的问题，不确定是不是在Base Model上面能得到改善；
    - 当前的结果剔除无法使用的结果之外，其实还是存在很多不规律的地方；比如ICG qwen的结果只限制在+-2；
    - 如果不使用类似Dopamine的框架的话（或者类似的理论框架），那么所有的行为学结果将无法串联；

-  Reasoning Task
      - Commitment Time 的变化是目前最清晰的行为模式转变
      - RSN的调控和COT在部分配置中可以叠加（Qwen Bare 的 GSM8K、GSM-Hard、GSM-Symbolic、BBH、CRUXEval，以及 Llama BBH：CoT 下仍有严格准确率增量，审计 Holm 通过，但伴随提交格式改善）；在 Qwen Native Chat 的 GSM8K／GSM-Hard 中，CoT 已使提前作答读数接近 0，RSN 额外收益未检出；这不是普遍可加性，详见 reasoningTaskAnalysis/cot_incremental_review_v1/REPORT.md
      - 存在的问题：
        - Qwen和Llama的调整方向是相反的，且qwen的趋势和行为学实验不符合；Qwen在reasoning的表现比Llama好，但是在行为学实验上却不不容易理解规则。猜测Qwen有针对reasoning task的训练；-> 是不是要转移到Base Model上看看
        - Llama3 Bare模式下存在无法收尾的问题；
        - Llama Chat 的基线输出已接近先推理（提前作答读数几乎为 0），未检出工作点收益；GSM8K 准确率约 90% 接近上限，但 GSM-Hard 仍有明显准确率空间，所以不能说没有提升空间，只能说未检出收益；Chat 的影响类似 RSN 提升效果时候的变化；
        - 有一些泛化能力，但是限制在类似的任务上；还不确定是不是任务自身要出现类似commitment time的提交窗口；
        - Llama3 做了一些Role prompt的分析，效应存在但是不是很明显，实在Bare上做的；
    - 曾经试图由Dopamine引出一些强迫和焦虑的症状，做过一些陈述；也可以检测到。
    - Base Model 上没有出现更强的解释性，但是可以一定程度发复现当前的结果（Qwen）

- Thinking Curve
  - 设想：建立在Dopamine的基础上，模拟人体激素释放的过程；看Dopamine如何参与任务；
  - 实验结果：仅在prompt的阶段检测到效应，在generation阶段很快就弹回；
  - 做过一些控制曲线的实验，但是没有效果；
  - 目前只能发现和commitment 前后相关的变化

- Representation & Manifold
  - 只有在这里区分MRSN和GRSN（from GSM8K），其余地方的RSN都是MRSN，固定的mask
  - MRSN vs MCSN (Confidence Sentatice Neurons)：都来来自MMLUE task，相似度高；存在overlapping显著高于随机；但是function上效果不同，MCSN不存在对commitment的明显影响；在MMLUE上可以有类似的E ratio降低的效果；
  - MRSN & GRSN/AGRSN ：相似度很低，且GRSN找不到明显的规律；感觉起来是GRSN没有MRSN来的纯粹，可能还涉及到整体的reasoning之类
  - MRSN vs Chat neurons from GSM8K：相似度非常低。原本的设想是Chat template来自RL，本身就包含了对模型角色的设定，应该和RSN存在重叠的方向；理论和之前是一致的，也就RL是模型产生角色认知的原因之一。

---

| Experiment | Prompt Format | 主要测量 | Llama3 结果 | Qwen 结果 | 当前判断 |
|---|---|---|---|---|---|
| **Confidence Betting：GPQA** | **Chat + `Bet: ` 前缀**；两个模型相同 | 下注大小、准确率 | +4 平均下注 **5.20→7.78**；负侧并非全程单调。准确率差异均未通过 Holm 校正 | +4 平均下注 **5.57→7.11**；+6 几乎固定押 5，+8 无效下注 **51.6%**。准确率差异未显著 | 两个模型均有明确下注调节，但高剂量存在失效 |
| **Confidence Betting：MMLU** | **Chat + `Bet: ` 前缀**；两个模型相同 | 同上 | +4 平均下注 **4.42→7.49**；−4 降低下注。准确率变化未显著 | +4 平均下注 **5.10→5.57**；基线已有 **97.9% 押 5**。准确率变化未显著 | 方向一致；效应大小受基线分布限制 |
| **CGT-Sequential：升序** | **Chat，多轮对话**；两个模型相同，但 prompt 版本不同 | 接受当前下注还是继续等待 | 已测有效剂量 **−4 至 +6**；接受步数 **3.45→2.63→1.40**（−4／0／+6） | **−2／0／+2** 均通过检查；接受步数 **3.010→1.185→1.012**，配对差异显著 | 接受时机变化清楚；Qwen 有效窗较窄 |
| **CGT-Sequential：降序** | **Chat，多轮对话**；同上 | 同上，下注由高到低呈现 | 同一有效范围内，接受步数 **4.47→2.97→1.31**；更强剂量出现失效 | 只有 **−2／0** 可正式比较：**3.508→1.677**；+2 的红色多数条件判断率 **50.31%**，未通过检查 | Qwen 仅部分支持，不能称完整双向剂量复制 |
| **IGT** | **Chat，多轮选择与反馈**；两个模型相同 | 有利牌组学习、切换与持续选择 | +2 出现局部高点：`net_score` **.15→.22**，学习斜率 **.24→.52**；更强正剂量导致表现退化 | 推理长度与策略结构变化明显；学习改善未通过 Holm 校正，最强对照 **p_adj=.250**；+8 无效输出 **87.6%** | Llama 有描述性局部峰；Qwen 未建立可靠学习收益 |
| **Bandit PV9** | **Bare**，两阶段结构化文本 | 定向探索、策略表达、累计表现 | ±4 改变探索措辞与次级决策锐度；未检出可靠的定向探索或表现改善 | 当前主线未见对应结果 | 表达变化未稳定转化为信息获取行动 |
| **Bandit PV10–PV11** | **Bare**，结构化文本 | 主动采样、停止与最佳臂识别 | 未检出可靠 α 收益；存在确认式采样、接口失效和低功效问题；部分设计撤回，实验线关闭 | 当前主线未见对应结果 | 不支持成功改善 BAI；也不能把低功效结果当作等价性证据 |
| **HaluEval** | **Bare** | 对给定答案指出错误的倾向 | +2／+4 错误检出率 **24.7%→27.0%**；+8 误报与格式失效增加 | 当前文档未见对应结果 | 判断倾向／边界证据，不能称自身幻觉减少 |
| **InfoQuest**¹ | **Chat，单轮用户请求**；两个模型相同 | 信息不足时的针对性追问 | 不完整请求追问率：−4／0／+4 为 **49%／48%／43%** | 对应 **56%／45%／42%**；+4 覆盖率变化差 CI 不含 0，但未经多重校正 | 方向一致、幅度有限；主要来自任务／对象澄清，评分未经人工验证 |
| **Willingness V1–V3** | **Bare**，数字候选读数 | 意愿自评 | 版本间出现方向反转；V3 虽有 **60/64** 个显著对照，但未形成稳定动机测量 | 当前版本未见对应结果 | 数字读数敏感，构念解释不稳 |
| **Fixed-Answer MMLU** | **Bare，Non-expert 提示**；仅读取候选 logits | 固定答案的正确性评分与条件提交倾向 | +4 条件提交读数 **.610→.682**；但 X/Y 概率质量极低，−4 受标签／位置偏好影响 | 当前文档未见对应结果 | 尚未成功分离 confidence 与实际提交行为 |

---

**Bare — Experimental Coverage**

| Task | Llama No-CoT | Llama CoT | Qwen No-CoT | Qwen CoT | 简短分析 |
|---|:---:|:---:|:---:|:---:|---|
| **GSM8K** | ✓ | ✓ | ✓ | ✓ | 主线两模型均有收益，方向因模型而异；Qwen3 与 OpenChat 补充结果见下表 |
| **GSM-Hard** | ✓ | ✓ | ✓ | ✓ | 两模型均支持近域迁移 |
| **MATH** | ✓ | ✓ | ✓ | ✓ | 两模型、两条件均为正向；Llama CoT 与 Qwen No-CoT（+6）显著，其余为正向趋势；Qwen 在 +6 达峰、+8 回落 |
| **GSM-Symbolic** | ✓ | ✓ | ✓ | ✓ | Llama No-CoT 显著、CoT 为正向趋势；Qwen 两条件均显著 |
| **BBH Counting** | ✓ | ✓ | ✓ | ✓ | 两模型 CoT 均显著；Qwen No-CoT 另有正向趋势（+2.40） |
| **CRUXEval-O** | ✓ | ✓ | ✓| ✓ | Qwen 首答改善；Llama 未检出收益 |
| **LogiQA 2.0** | ✓ | ✓ | ✓ | ✓ | 两模型、两条件均未检出收益 |
| **FinQA** | — | ✓ | — | ✓ | Llama 严重循环，且负向 α 显著降低准确率；Qwen +8 为正向趋势（+5.00） |
| **ProofWriter-OWA** | — | ✓¹ | — | ✓¹ | Llama 收益伴随提交恢复；Qwen +8 为正向趋势（+5.67，Holm p=.257） |
| **ZebraLogic-Easy** | — | ✓¹ | — | ✓¹ | Llama 未检出收益；Qwen +6 为正向趋势（+1.43），+8 格式崩溃 |

**Supplemental Bare GSM8K Models**

| Model | Coverage | Result |
|---|---|---|
| Qwen3-8B | No-CoT | 2048-token 检查保留收益：+10 为 86.67%（+28.33 pp，Holm p=5.73×10⁻¹⁶），定为固定工作点；{+8,+10} 为已测近似最优区，仍非峰值验证；截断与格式敏感性见 ReasoningBare §9.1 |
| OpenChat-3.5 | No-CoT | 五个剂量均未检出收益；early-candidate 接近天花板，循环与截断严重 |

**Native Chat — Experimental Coverage**

| Task | Llama No-CoT | Llama CoT | Qwen No-CoT | Qwen CoT | Mistral No-CoT | 简短分析 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **GSM8K** | ✓ | ✓ | ✓ | ✓ | ✓ | Qwen No-CoT 显著、CoT 为正向趋势（+1.67）；Llama 未检出收益。Mistral 九点扫描中 −4 为候选工作点（+8.33 pp，Holm p=.0459），尚未经独立验证，作为附录接口边界例子 |
| **GSM-Hard** | ✓ | ✓ | ✓ | ✓ | — | Qwen 仅 No-CoT 有显著收益；Llama 未检出收益 |
| **MATH** | ✓ | — | ✓ | — | — | Qwen +8 基本持平（+0.67），+6 为正向趋势（+2.00）；α=0 基线已先推理后作答仅适用于 Qwen |
| **GSM-Symbolic** | — | — | — | — | — | 尚无 Chat 结果 |
| **BBH Counting** | — | — | — | — | — | 尚无 Chat 结果 |
| **CRUXEval-O** | ✓ | ✓ | — | — | — | Llama 接口健康，仍未检出收益 |
| **LogiQA 2.0** | — | — | — | — | — | 尚无 Chat 结果 |
| **FinQA** | — | ✓ | — | ✓ | — | Chat 0-shot CoT：Llama 未检出提升；Qwen 严格分数下降，宽松数值评分不变 |
| **ProofWriter-OWA** | ✓² | ✓¹,³ | ✓² | ✓¹,³ | — | Llama：1-shot CoT（+6.33）与 0-shot CoT（+3.00）为正向趋势，No-CoT 无；Qwen：1-shot CoT +8 收益经宽松评分复核通过，0-shot CoT 宽松评分为正向趋势（+1.67 pp，p=.49；严格 +18 pp 主要是格式效应），No-CoT 无 |
| **ZebraLogic-Easy** | — | — | — | — | — | 尚无 Chat 结果 |

注：覆盖表中的 ✓ 只表示「已有实验」，不代表「显著收益」；是否有收益见「简短分析」。

¹ **1-shot CoT**；² **zero-shot No-CoT**；³ **zero-shot CoT**。Zero-shot CoT 与 No-CoT 仅差 `Let's think step by step.` 一行；1-shot 与 zero-shot CoT 还改变了示例及少量指令措辞。

另有 **Chat Matched-Anchor**：仅 Llama 的 **GSM8K、MATH No-CoT** 已做，未恢复 Bare 负向收益；与 Native Chat 分开记录。

主要依据：[ReasoningBare](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningBare.md:888)、[ReasoningChat](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningChat.md:7)、[LogiQA CoT launcher]
(/Users/paveenhuang/Downloads/Dopamine/run_logiqa2_cot.sh:35)、[ZebraLogic 协议](/Users/paveenhuang/Downloads/Dopamine/docs/PREREG_ZEBRALOGIC_EASY.md:54)。

---


| Model (α) | GSM-Hard | MATH |
|---|---|---|
| Llama (−6) | **+6.33 pp** | **+6.67 pp** |
| Qwen2.5 (+8) | **+16.33 pp** | +2.67 pp（正向趋势） |
| Qwen3 (+10) | **+11.33 pp** | **+8.00 pp** |
| Granite (−6) | +2.33 pp（正向趋势） | **+7.33 pp** |

---

### Conclusion

- **原论文**：RSN 已被 ACL 2026 Findings 接收，camera-ready 已完成。核心是在 MMLU-E 上用 ±α 双向控制弃答和决断性，并附了 GSM8K 的 Confidence Ratio，用来回应审稿人说"只做了 QA"的意见。
- **新稿**：[TMLR1005.md](Note/TMLR1005.md) 已收敛成四个 claim。C1 是行为调节（Betting、CGT），C2 是模型特异的推理响应，C3 是迁移边界，C4 是预测与选剂量。[Conclusion.md](Note/Conclusion.md) 是你自己的总结，比 TMLR 稿更坦率。
- **最硬的结果**：
  - 输出层面的 commitment timing（early-candidate）变化。
  - 4 个模型的固定工作点近域迁移：8 项全部为正，6 项显著。
  - 不用标签、只按 EC 选剂量：13 个配置里 10 个零 regret。
  - 正交注入对照。
- **最弱的结果**：wanting 这个构念。Bandit、Willingness、Fixed-Answer 都没成功，ThinkingCurve 只在入口有效应，dopamine 类比也没有论证到位。

### Paper
**1. 中心叙事。** 现在 TMLR1005 是"系统刻画"的写法，覆盖面宽，但读起来比较散，审稿人容易问 takeaway 是什么。建议以 commitment 为主轴，一句话大致是：RSN 干预改变模型何时提交答案；这是否有助于推理，取决于基线的提交状态；同一读数可以在不用标签的情况下选出剂量。

- Betting 和 CGT 作为"同一方向在推理之外也移动 commitment"的支撑。
- 其他行为实验放进边界或附录。
- 这和 Conclusion 里"Commitment Time 是最清晰的行为转变"的判断一致。

**2. 最大的审稿风险是 Bare 接口。** 主要收益大多出现在 Bare 下：
- Llama Bare 的最佳点只有 78.00%，低于 Native Chat 的未干预基线 89.67%。
- Llama 和 Qwen3 的循环、截断率都很高。
- Native Chat 加上 CoT 之后，收益基本消失。

审稿人很可能会说"这只是在修补一个坏掉的接口"。建议主动正面处理，并把它变成论点：收益集中在基线 EC 高的条件下。Table 7a 的已有数据基本符合这一点：
- Llama 在 Chat 下 EC≈0，没有收益。
- Qwen Chat GSM8K 基线 EC 为 50%，提升 +31 pp。
- Qwen Chat MATH 基线 EC 为 0，没有收益。

另外一处要改：TMLR1005 把 Llama Chat GSM-Hard 当作"输出顺序不足以解释"的反例。但它的基线 EC 是 0.33%，按"EC 余量"的逻辑，没有收益正是预期的结果，不是反例。真正的反例是 Qwen GSM-Symbolic CoT。可以画一张"基线 EC 对 steering 收益"的跨条件图，属于描述性的事后分析，不需要新实验。

**3. 与原论文的符号张力要写在正文里。**
- 原论文：+α 代表更决断，Table 5 里 +4 提升了 GPQA 等任务。
- 新稿：Llama 的推理最优点在 −6；Qwen 的 +α 在 CGT 里是更早接受，在 GSM8K 里却是更晚给出答案。

如果不主动交代，审稿人对照原论文会认为两者矛盾。我建议把它作为未解决的发现写出来，不要强行统一。

**4. 和原论文的重叠与投稿规则。**
- CLAUDE.md 已记录 TMLR 不接受已发表会议论文的扩展版，所以 MMLU-E 的结果只能引用。
- 匿名投稿时，自引要用第三人称，`github.com/paveenH/RSN` 的链接也要处理。
- 原论文主模型是 Qwen3-8B，新稿主线换成了 Qwen2.5-7B，需要一句理由；或者把 Qwen3 也提为主线模型之一。

**5. Dopamine 的位置。** 面向 ML venue，我建议标题、摘要和 claim 里都不用这个词，只在 Discussion 里保留一段动机说明。


### TMLR

- **评审标准合适。** TMLR 主要看结论是否有证据支撑、对部分读者是否有价值，不设新颖性门槛。ACL 那篇的审稿人 HxKE 说过"novelty modest"；在 TMLR，这类意见一般不会成为拒稿理由。新稿的强项恰恰是边界交代清楚、没有夸大。
- **没有篇幅上限。** 现在的证据有 4 个模型、约 10 个任务、两种接口，外加正交对照和预测。压进 8～9 页会把那些让结论可信的边界结果砍掉，而审稿人最先会问的恰恰是这些边界。
- **随时可以投。** 稿子写好就能提交，不用等截止日。

**最大的风险是 TMLR 关于作者自身旧作的规定。** 原文是不接受 "expanded versions of conference papers"，也不能复用已发表论文的 "written text, figures or results"。ACL Findings 2026 那篇属于已发表的会议论文。所以新稿要定位成"沿用已有方法的新研究"，具体做法：
- RSN 方法和 MMLU-E 结论只引用，不复现成表格或图。
- 不再报告原论文里的 GSM8K Confidence Ratio 表。
- [ReasoningBare.md §7.6](Note/ReasoningBare.md) 的 MMLU-E 正交对照里引用了历史 RSN 数值（E-rate 3.73%、65.13%，准确率 63.22%）。这正是原论文 Table 2 的结果，进 TMLR 稿要删掉，或只在正文用一句话引用。
- 新稿里的 Base 检查点实验是新数据，但原论文已经提出过"指令微调只是锐化"的说法，叙述上要和那一节明确区分开。
- 原论文的 GitHub 链接要做匿名处理。

**其他选项的比较：**

| 去向 | 截止时间 | 适合程度 |
|---|---|---|
| TMLR | 随时 | 最适合；要注意上面的自用旧作规定 |
| COLM 2027 | 约 2027-03-31| 第二选择，偏好语言模型分析类论文，但篇幅约 9 页 |
| ACL 2027| 官方尚未公布 约1月 | 社区对口，但 8 页太紧，且容易被看成上一篇的后续，有再次落到 Findings 的风险 |
| ICML 2027 | 约 2027-01-22| 不建议；会被要求清楚的机制，Bare 接口问题也会被重点攻击 |



Sources:
- [TMLR Editorial Policies](https://www.jmlr.org/tmlr/editorial-policies.html)
- [ICLR Journal-to-Conference](https://iclr.cc/public/JournalToConference)
- [ARR August cycle / EACL 2027 CFP](https://groups.google.com/g/ML-news/c/yLzEmmZOMJU)
- [AI Weekly on ARR cycles](https://aiweekly.co/alerts/acl-rolling-review-opens-august-cycle-amid-nlp-submission-surge)
- [OpenCurious: ACL 2027](https://www.opencurious.com/ai-conference-deadlines/acl-2027)
- [OpenCurious: ICML 2027](https://www.opencurious.com/ai-conference-deadlines/icml-2027)
- [Bibby NLP deadlines](https://trybibby.com/conference-deadlines/nlp)
- [ARR page](https://aclrollingreview.org/page5/)