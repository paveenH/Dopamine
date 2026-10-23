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
| **GSM8K** | ✓ | ✓ | ✓ | ✓ | 均有收益；有效方向因模型而异 |
| **GSM-Hard** | ✓ | ✓ | ✓ | ✓ | 两模型均支持近域迁移 |
| **MATH** | ✓ | ✓ | ✓ | ✓ | 两模型、两条件均为正向；Llama CoT 与 Qwen No-CoT（+6）显著，其余为正向趋势；Qwen 在 +6 达峰、+8 回落 |
| **GSM-Symbolic** | ✓ | ✓ | ✓ | ✓ | Llama No-CoT 显著、CoT 为正向趋势；Qwen 两条件均显著 |
| **BBH Counting** | ✓ | ✓ | ✓ | ✓ | 两模型 CoT 均显著；Qwen No-CoT 另有正向趋势（+2.40） |
| **CRUXEval-O** | ✓ | ✓ | ✓| ✓ | Qwen 首答改善；Llama 未检出收益 |
| **LogiQA 2.0** | ✓ | ✓ | ✓ | ✓ | 两模型、两条件均未检出收益 |
| **FinQA** | — | ✓ | — | ✓ | Llama 严重循环，且负向 α 显著降低准确率；Qwen +8 为正向趋势（+5.00） |
| **ProofWriter-OWA** | — | ✓¹ | — | ✓¹ | Llama 收益伴随提交恢复；Qwen +8 为正向趋势（+5.67，Holm p=.257） |
| **ZebraLogic-Easy** | — | ✓¹ | — | ✓¹ | Llama 未检出收益；Qwen +6 为正向趋势（+1.43），+8 格式崩溃 |

**Native Chat — Experimental Coverage**

| Task | Llama No-CoT | Llama CoT | Qwen No-CoT | Qwen CoT | Mistral No-CoT | 简短分析 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **GSM8K** | ✓ | ✓ | ✓ | ✓ | ✓ | Qwen No-CoT 显著、CoT 为正向趋势（+1.67）；Llama 未检出收益。Mistral 九点扫描中 −4 为候选工作点（+8.33 pp，Holm p=.0459），尚未经独立验证 |
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

### Overall conclusions on reasoning tasks

**1. Dose–response is model-specific (C2).** RSN 在 GSM8K 和 MATH 上对两个模型都有准确率收益，但有效方向和曲线形状不同：

- **Llama：** 负向 α 有效，峰值在 −6 附近，呈不对称的峰形。
- **Qwen：** 正向 α 有效。GSM8K 在高剂量进入平台；MATH 的峰在 +6，之后回落。

工作点是一个近优区域，不是唯一的点。

**2. Steering reorganizes visible output (C2).** 在有效工作点上，提前给出答案的比例下降，先推理的比例上升；这种变化在多数任务上伴随收益或正向趋势。但它不是充分条件：Qwen BBH No-CoT 和 Qwen MATH +8 的作答顺序都变了，准确率没有相应提高。这些都只说明可见输出的组织方式变了，不说明内部决策机制。

**3. Direction specificity (C2).**

- **Qwen：** RSN 优于六个范数匹配的随机方向，随机方向基本没有效果。
- **Llama：** RSN 同样优于这些随机方向，但部分随机方向本身也能提分。

所以只能写"RSN 优于所测试的正交方向"，不能写"只有 RSN 有效"。

**4. Transfer across tasks (C3).** 多数推理任务至少在一种条件下有收益或正向趋势：

- **两个模型：** GSM-Hard、GSM-Symbolic、MATH；BBH 只在 CoT 下有收益。
- **只有 Qwen：** CRUXEval；ProofWriter（Chat 1-shot）。
- **正向趋势：** Qwen FinQA Bare。
- **没有收益：** LogiQA、ZebraLogic、FinQA Chat、Llama CRUXEval。

**5. Interface and CoT modulate the gain (C3).**

- Chat 模板会抬高基线；在 Llama 上，它基本替代了 RSN 的收益。
- CoT 已经让输出先推理时，RSN 不再增加收益（Qwen Chat GSM8K、GSM-Hard）。
- CoT 没有改变作答顺序时，两者可以叠加，其中一部分体现为提交格式的恢复。
- 在非数值任务上，严格评分下的大幅变化可能只是格式效应（ProofWriter 0-shot、FinQA Chat）。

**6. Prediction (C4).** 输出特征能预测答案正确性，并能在剂量曲线上选出较优的工作点：MATH 和 GSM-Hard 上方向都选对，regret 为 0。但跨任务的绝对概率校准不可靠。

