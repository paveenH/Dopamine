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
      - RSN的调控和COT可以叠加
      - 存在的问题：
        - Qwen和Llama的调整方向是相反的，且qwen的趋势和行为学实验不符合；Qwen在reasoning的表现比Llama好，但是在行为学实验上却不不容易理解规则。猜测Qwen有针对reasoning task的训练；-> 是不是要转移到Base Model上看看
        - Llama3 Bare模式下存在无法收尾的问题；
        - Llama Chat有类似的表现，但是已经没有提升的空间，只能提前commitment time，让表现变差;但是Chat的影响类似RSN提升效果时候的变化；
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

| Type | Cells | Baseline answer-first | Change at workpoint | Acc change |
|---|---|---|---|---|
| **A. 有空间，被移动，有收益** | GSM8K / GSM-Hard / GSM-Symbolic No-CoT（两个模型）；MATH；BBH CoT（两个模型）；Qwen CRUXEval CoT；Qwen Chat GSM8K / GSM-Hard No-CoT；Qwen ProofWriter Chat 1-shot CoT（宽松评分复核通过） | Qwen EC 93–100%；Llama EC 45–50% | EC 降到 5–30% | +6 到 +19 pp |
| **B. 没有空间，没有收益** | Llama Chat GSM8K / MATH / GSM-Hard；Llama Chat FinQA；Qwen Chat MATH；Chat-CoT 四格；Qwen LogiQA；ZebraLogic；Llama CRUXEval Chat | EC ≈ 0%，或 reason-first 已饱和（Zebra Qwen 100%，Qwen LogiQA 270/300 在给答案前比较过选项） | 无可移动 | ≈ 0 |
| **C. 有空间，被移动，但没有收益** | BBH No-CoT（Qwen EC 100→44%，+2.4 ns）；LogiQA Llama（naf .76→.33，−4.3 pp）；FinQA Qwen（EC 99→43%，+5 ns）；Qwen MATH +6→+8；Llama CRUXEval CoT | 高 | 明显下降 | 0 或负 |
| **D. 生成失效，无法读数** | ProofWriter Bare Llama（no-answer 79%）；FinQA Llama（loop 97%，截断 100%）；Zebra Qwen +8；ProofWriter Chat Qwen −6（1-shot CoT 与 zero-shot No-CoT 均格式崩溃） | — | 格式崩溃 | — |
| **E. 严格分数受格式影响，宽松评分未见变化** | Qwen FinQA Chat 0-shot CoT；Qwen ProofWriter Chat zero-shot CoT 与 No-CoT | 基线答案常在解释前出现 | 作答顺序未普遍转为先推理：FinQA `+8` EC 97.67%；ProofWriter No-CoT `+8` 有 284/300、CoT `+8` 有 260/300 在解释前给出标签；主要变化是提交格式 | 宽松评分未检出显著净变化 |

同为 Qwen Chat，ProofWriter 1-shot CoT `+8` 的 `answer_first` 从 50% 降至 4%，宽松标签正确率提高 7.00 pp（41%→48%，Holm `p=.0296`）；FinQA 与 ProofWriter 两种 zero-shot 条件的宽松评分均未检出净收益，且 CoT 条件仍有 260/300 条在解释前给标签。zero-shot CoT 与 No-CoT 只差一行指令，作答顺序和收益都几乎不变；这支持“有空间且顺序真实移动”与收益相伴，但仍是描述性证据。

**Bare — Experimental Coverage**

| Task | Llama No-CoT | Llama CoT | Qwen No-CoT | Qwen CoT | 简短分析 |
|---|:---:|:---:|:---:|:---:|---|
| **GSM8K** | ✓ | ✓ | ✓ | ✓ | 均有收益；有效方向因模型而异 |
| **GSM-Hard** | ✓ | ✓ | ✓ | ✓ | 两模型均支持近域迁移 |
| **MATH** | ✓ | ✓ | ✓ | ✓ | Llama CoT、Qwen No-CoT 有显著收益 |
| **GSM-Symbolic** | ✓ | ✓ | ✓ | ✓ | Llama 仅 No-CoT 稳定；Qwen 两条件均有收益 |
| **BBH Counting** | ✓ | ✓ | ✓ | ✓ | 两模型均**仅 CoT**下有显著收益 |
| **CRUXEval-O** | ✓ | ✓ | ✓| ✓ | Qwen 首答改善；Llama 未检出收益 |
| **LogiQA 2.0** | ✓ | ✓ | ✓ | ✓ | 两模型、两条件均未检出收益 |
| **FinQA** | — | ✓ | — | ✓ | Llama 严重循环；Qwen 仅正向趋势 |
| **ProofWriter-OWA** | — | ✓¹ | — | ✓¹ | Llama 收益伴随提交恢复；Qwen 未检出收益 |
| **ZebraLogic-Easy** | — | ✓¹ | — | ✓¹ | 均未检出收益；Qwen 高剂量格式崩溃 |

**Native Chat — Experimental Coverage**

| Task | Llama No-CoT | Llama CoT | Qwen No-CoT | Qwen CoT | 简短分析 |
|---|:---:|:---:|:---:|:---:|---|
| **GSM8K** | ✓ | ✓ | ✓ | ✓ | Qwen 仅 No-CoT 有显著收益；Llama 未检出收益 |
| **GSM-Hard** | ✓ | ✓ | ✓ | ✓ | Qwen 仅 No-CoT 有显著收益；Llama 未检出收益 |
| **MATH** | ✓ | — | ✓ | — | No-CoT 基线已高度先推理后作答；均未检出收益 |
| **GSM-Symbolic** | — | — | — | — | 尚无 Chat 结果 |
| **BBH Counting** | — | — | — | — | 尚无 Chat 结果 |
| **CRUXEval-O** | ✓ | ✓ | — | — | Llama 接口健康，仍未检出收益 |
| **LogiQA 2.0** | — | — | — | — | 尚无 Chat 结果 |
| **FinQA** | — | ✓ | — | ✓ | Chat 0-shot CoT：Llama 未检出提升；Qwen 严格分数下降，宽松数值评分不变 |
| **ProofWriter-OWA** | ✓² | ✓¹,³ | ✓² | ✓¹,³ | 1-shot CoT：Qwen +8 收益经宽松评分复核通过；0-shot CoT 与 No-CoT：宽松口径均未检出 Qwen 准确率提升，Llama 三组均未检出显著收益 |
| **ZebraLogic-Easy** | — | — | — | — | 尚无 Chat 结果 |

¹ **1-shot CoT**；² **zero-shot No-CoT**；³ **zero-shot CoT**。Zero-shot CoT 与 No-CoT 仅差 `Let's think step by step.` 一行；1-shot 与 zero-shot CoT 还改变了示例及少量指令措辞。

另有 **Chat Matched-Anchor**：仅 Llama 的 **GSM8K、MATH No-CoT** 已做，未恢复 Bare 负向收益；与 Native Chat 分开记录。

主要依据：[ReasoningBare](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningBare.md:888)、[ReasoningChat](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningChat.md:7)、[LogiQA CoT launcher](/Users/paveenhuang/Downloads/Dopamine/run_logiqa2_cot.sh:35)、[ZebraLogic 协议](/Users/paveenhuang/Downloads/Dopamine/docs/PREREG_ZEBRALOGIC_EASY.md:54)。

### Proposed Analysis

| Metric | Definition | 回答的问题 |
|---|---|---|
| **Ordering headroom** | α=0 时各任务原生的 answer-first 比例。数值任务用 EC；LogiQA 用 naf；CRUXEval / ProofWriter 用 answer_first；Zebra 用 1 − reason_before_solution | 有没有可以移动的顺序 |
| **Addressable error share** | α=0 答错的题里，属于提前给出候选的比例 | 错误是否来自过早承诺 |
| **Conversion rate** | 在可纠正的错题中，到工作点时 EC 被关掉并且答对的比例；同时报告原本答对的题被改错的比例 | 顺序改变之后，有没有转成收益 |
| **Health gate** | loop、截断、no-answer 超过阈值的格子单独列出，不参与判断 | 排除 D 类 |

预期结论的形式是：收益需要两个条件同时成立，一是有顺序空间，二是错误主要来自过早承诺；只有前者时（C 类）不足以带来收益。这样能把 C3 里"迁移失败"的各个任务统一成一条有解释的边界，而不是一张零散的列表。

