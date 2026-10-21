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
| **A. 有空间，被移动，有收益** | GSM8K / GSM-Hard / GSM-Symbolic No-CoT（两个模型）；MATH；BBH CoT（两个模型）；Qwen CRUXEval CoT；Qwen Chat GSM8K / GSM-Hard；Qwen ProofWriter Chat 1-shot CoT | Qwen EC 93–100%；Llama EC 45–50% | EC 降到 5–30% | +6 到 +19 pp |
| **B. 没有空间，没有收益** | Llama Chat GSM8K / MATH / GSM-Hard；Qwen Chat MATH；Chat-CoT 四格；Qwen LogiQA；ZebraLogic；Llama CRUXEval Chat | EC ≈ 0%，或 reason-first 已饱和（Zebra Qwen 100%，Qwen LogiQA 270/300 在给答案前比较过选项） | 无可移动 | ≈ 0 |
| **C. 有空间，被移动，但没有收益** | BBH No-CoT（Qwen EC 100→44%，+2.4 ns）；LogiQA Llama（naf .76→.33，−4.3 pp）；FinQA Qwen（EC 99→43%，+5 ns）；FinQA Llama（EC 90→35%，−5.7 pp）；Qwen MATH +6→+8；Llama CRUXEval CoT | 高 | 明显下降 | 0 或负 |
| **D. 生成失效，无法读数** | ProofWriter Bare Llama（no-answer 79%）；FinQA Llama（loop 97%，截断 100%）；Zebra Qwen +8；ProofWriter Chat Qwen −6（1-shot CoT 与 zero-shot No-CoT 均格式崩溃） | — | 格式崩溃 | — |

**Bare — Experimental Coverage**

| Task | Llama No-CoT | Llama CoT | Qwen No-CoT | Qwen CoT | 简短分析 |
|---|:---:|:---:|:---:|:---:|---|
| **GSM8K** | ✓ | ✓ | ✓ | ✓ | 均有收益；有效方向因模型而异 |
| **GSM-Hard** | ✓ | ✓ | ✓ | ✓ | 两模型均支持近域迁移 |
| **MATH** | ✓ | ✓ | ✓ | ✓ | Llama CoT、Qwen No-CoT 有显著收益 |
| **GSM-Symbolic** | ✓ | ✓ | ✓ | ✓ | Llama 仅 No-CoT 稳定；Qwen 两条件均有收益 |
| **BBH Counting** | ✓ | ✓ | ✓ | ✓ | 两模型均仅 CoT 下有显著收益 |
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
| **FinQA** | — | — | — | — | 尚无 Chat 结果 |
| **ProofWriter-OWA** | — | ✓¹, ² | — | ✓¹, ² | 1-shot CoT：Qwen +8 显著、Llama 为正向趋势；zero-shot No-CoT：Llama 未见收益，Qwen 严格分数主要反映格式恢复 |
| **ZebraLogic-Easy** | — | — | — | — | 尚无 Chat 结果 |

¹ **1-shot CoT**；² **zero-shot No-CoT**。两种提示条件的差异不能单独归因于 CoT；其余已做条件为 zero-shot。

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


---
3．随后优先解决 RSN 与 CSN 的区别。

这里需要修正你之前的一点理解：最新 [Neurons.md](/Users/paveenhuang/Downloads/Dopamine/Note/Neurons.md:162) 显示，**CSN 也会改变候选输出和提交行为**；小剂量有准确率改善趋势，但尚未通过多重比较校正。

因此，下一步应先做**逐层范数匹配、有限剂量范围的功能比较**，再决定是否分解共享方向和剩余方向。现在直接把二者的差异命名为“confidence 与 commitment 的区别”，证据还不够。

对比激活均值来改变行为已有 [CAA 等先例](https://aclanthology.org/2024.acl-long.828/)。我认为我们接下来的贡献应集中在：**用有区分力的预测，说明 RSN 的功能及边界。** 核验工具、新行为任务和更细的 MLP／attention 追踪，可以等这一步有结论后再决定。

---

2. 正交对照补一组 off-support（ortho_gauss_off）
每个模型补 3 个 cell，生成方式和 §7 完全相同，只是把非零位置挪到 RSN 位置之外。
和现有的 same-support 结果放在一起，就能把"位置"和"权重"两个因素拆开。Llama 尤其需要这组数据。
如果 off-support 在 Llama 上也能涨分，那说明 Llama 在这个工作点上对稀疏扰动本身就比较敏感，§7 的结论要相应收窄。如果不涨，就能支持"RSN 的位置也是特殊的"。
生成 mask 的工具 detection/nmd.py 里已经有，代价很小。
是否给 Llama 加更多 seed（比如加到 10 个）可以放到后面再定。3 个 seed 只能说明"RSN 比这三个方向好"，加 seed 可以换来一个百分位式的说法，但不是必需的。

---

1．先比较 Qwen 的 Expert／Non-expert／Neutral 提示。
使用同一批 GSM8K 300 题、Bare No-CoT、α=0，三个提示在同批运行。预先固定模板和检测器，主要看 early-candidate，辅助看首答案准确率。
明确预测：Qwen 的 Expert 提示应比 Non-expert 更少在开头输出答案候选，与正向 RSN 的变化对应。
结果	对解释的帮助
方向一致，差异清楚	支持角色提示与 RSN 存在行为对应
方向相反	削弱“RSN 只是复现专家提示策略”的简单解释
差异很小、区间较宽	这个检验暂时无法区分解释


同时将已有 Llama 角色输出按同一口径离线重算。这里即使发现对应，也只说明行为对应，不能直接证明内部路径相同。

### 第二优先：角色提示与 steering 的对应检验（先做一格，GPU 开销小）

RSN 在定义上是 expert − non-expert 的表征差，所以最直接的检验不预设心理功能，只问：

> 在同一模型、同一任务上，+α 与 −α 的差异，是否复现 expert 与 non-expert 提示之间的差异？

**待检验的解释（"角色策略"）：** +α 把模型推向它自己学到的"专家行为"，而专家在某个情境下做什么是训练决定的。决策任务里两个模型学到的专家都"果断"，所以 Betting / CGT / InfoQuest 方向一致；数学推理里 Qwen2.5 可能因大量推理训练而把"专家"学成先推导再作答，所以只有 Qwen GSM8K 的方向和 Llama 相反。如果成立，"方向相反"不再是矛盾，而是角色效应本身因模型和任务而异。

#### 第一步：只做 Qwen GSM8K（决定性的一格）

整个方向矛盾集中在这里，先用它决定这条线是否值得扩展。

- **设计：** Qwen2.5-7B-Instruct，GSM8K Bare No-CoT，α=0，同一次运行内跑 `an expert` / `a non expert` / `neutral` 三个提示（role 字符串与 Llama `run_gsm8k.sh` 一致），n=300，逐题配对。neutral 同批重跑，不与旧 `mdf_0` 跨批比较。输出写入新目录 `answer_mdf_gsm8k_role/`，不覆盖已有 cell。
- **预先写定的预测（运行前冻结）：** Qwen expert 提示的 `early_candidate_rate` **低于** non-expert 提示，方向与 Qwen 自身 +α 一致、与 Llama 相反。
- **读数：** 主读数 `early_candidate_rate`（`earlycand-v1`，冻结检测器，不重调）；辅助读数 offline `first_acc`；expert vs non-expert 用配对 exact McNemar，并报配对 bootstrap 区间。
- **判定：**

| 结果 | 解读 |
|---|---|
| 方向与预测一致，区间不含 0 | 支持"角色策略"解释，进入第二步 |
| 方向相反 | 解释被否定；方向差异来自 steering 本身而非角色，需另找原因 |
| 差异接近 0 / 区间跨 0 | 提示层面的角色效应太弱，本检验不能下结论，**不计为支持** |

#### Llama 对照：离线重算

此前的 Llama GSM8K 角色数字（expert 58.0% / ec 54.3%；neutral 60.0% / 48.0%；non-expert 68.0% / 41.7%）来自 inline 字段，**不能直接引用**。用冻结抽取器（`analyze_first_last_acc`）和 `earlycand-v1` 在 `llama3/gsm8k/bare/mdf_0` 上重算后，与 Qwen 用同一个分析脚本出对照表。纯离线。

#### 第二步（仅当第一步支持时）：扩展到行为任务

- Betting 和 CGT 各补 expert、non-expert 提示，两个模型，只跑 α=0。
- **这些都是新协议，不是旧结果的补格：** Betting 的冻结 prompt 没有 role 位置，加入角色句会改变 prompt；Qwen CGT v5 已 CLOSED。两者都只能与同批重跑的 neutral 比较，不能与 §3.1 / §3.3 的冻结结果比较。
- 可选：在 Chat 接口下看角色提示是否同样不再改变输出组织，用于解释表 7a 的接口依赖。

**限制：** 角色提示与 steering 方向一致只说明两者对应，是 consistent-with 证据，不证明 steering 的效应"经由"角色表征产生；提示改变的是整个上下文，steering 只改一组神经元。

### 第三优先：草稿末尾新加的"核验工具"实验，暂缓

它的思路很好：把"是否发起核验"和"能否利用核验结果"分开。但我有三点顾虑：

1. **接口风险。** 调用工具基本需要 Chat 接口，而表 7a 显示 Llama 在 Chat 下基线已经先推理后作答，RSN 在那里没有效果。Llama 很可能直接得到无效结果。Qwen 才可能有信号。
2. **前车之鉴。** Fixed-Answer、Willingness、InfoQuest 这三条新接口实验，都在建立测量接口阶段卡住了。这个实验需要新的工具接口、新的注入位置和新的评分方式，是同类风险。
3. **它不解决你最关心的问题。** 草稿自己也写了：即使成功，也只建立推理内部的联系，不能把 Betting、CGT 连起来。

所以建议：只有在角色对应检验失败、而我们仍然需要一座行为与推理之间的桥时，再启动它。届时先在 Qwen 上做 α=0 的小规模试点。

---
2．随后完成一次有限的 RSN–CSN 等范数比较。
你指出 CSN 在 GSM8K 上变化小、缺少清楚规律，这正是值得检验的差异。匹配实际稀疏注入向量的逐层范数后，比较双方的候选输出顺序和准确率，才能更有依据地回答：
RSN 的清楚效果，是否能够由 confident–unconfident 方向替代？

这一步应限定比较范围，避免不断为 CSN 寻找更好的剂量。

---

## 顺带的小事

- 草稿里的回退子集准确率还是反推出来的近似值，用冻结抽取器逐题重算一遍，大约十分钟。
- 先提交这一轮指标统一的文档改动。
- C1 里 Qwen CGT 降为附录里的边界证据，主文的行为证据以 Betting 为主。
---

**我觉得还能做，但下一步应该优先区分解释，而不是继续增加“RSN 在某个任务上有效”的例子。** 我会按下面顺序推进。

**1．完成正交注入对照，先确认具体方向是否重要。**

这是已经准备好的实验，最值得先收尾。关键看 **RSN 与各个等范数正交方向的直接差异**，不能只看 RSN 显著、对照不显著。

它回答的是：收益是否依赖这组特定权重，而非相近规模的扰动。即使支持，也只限定于这些对照和原工作点数据，不能据此命名心理机制。

**2．做一个小型“主动核验—反馈利用”实验，把行为和收益连接起来。**

[TMLR1005.md](/Users/paveenhuang/Downloads/Dopamine/Note/TMLR1005.md:340) 文末已经提出这个方向，我仍认为它比继续研究 marker 位置更有信息量：

给不同干预条件**完全相同的解题草稿**，包含正确和错误两类，设置两个分支：

| 分支 | 要区分的问题 |
|---|---|
| 工具可用，模型自主决定是否核验 | RSN 是否改变发起检查的行为？ |
| 直接提供相同的正确核验反馈 | RSN 是否改变利用反馈、修正答案的能力？ |

重点看错误草稿是否更容易触发核验、错误能否纠正，以及正确答案是否被改错。这样可以区分“检查更多”“检查更有针对性”和“拿到信息后更会使用”。

先做 **20–30 题的 α=0 pilot**，确认工具调用和反馈利用确实可测，再决定是否开展干预。外部反馈值得单独控制，因为已有研究发现，仅要求模型自行反思未必能稳定改善推理。[Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798)

**3．如果要直接回应“只是更像专家”，加入显式专家提示作为参照。**

在上述同一任务里比较中性提示、专家提示和 RSN steering，观察它们是否产生相同的核验与纠错模式。

如果效果接近，就应认真考虑：RSN 的实际用途是否主要是实现某类角色提示效应。如果出现分离，例如都增加检查，但只有一种改善针对性，就能进一步缩小功能解释。**相似或不同都不能单独证明内部机制相同或不同，但比“专家方向”这个名字更具体。**

GSM8K Base 可以保留为后续方向，不过它主要回答模型训练阶段的依赖性，对“究竟调节什么”的区分较间接。

**我的优先选择是：先完成正交对照，再只开一个核验 pilot。** 目标是获得一个能同时观察行为选择、信息利用和最终收益的实验，而不是再寻找一个容易出现正结果的任务。

---

## Vector
- 提取位置：当前 RSN 是 Role 提示在任务入口引起的综合状态差异。可以比较“身份描述结束处”和“生成入口”的 HS，区分身份编码与任务整合后的影响；目前还不能确定换位置会更好。
- 与 confidence 的差异方向：可以提取 Role 中无法被 confidence 方向解释的 residual。它可能帮助定位额外的功能成分，但不能直接叫“更纯粹的 Role”，因为仍可能包含其他提示和任务效应。
- Manifold reasoning Chat & MMLUE RSN

## Representation

| 层次 | 要回答的问题 | 推荐指标 |
|---|---|---|
| 几何结构 | 两组 activation 的空间布局是否相似？ | centered CKA、mCCA/SVCCA、正交 Procrustes residual、子空间夹角 |
| 可读出功能 | 在 A 中线性可读的信息，B 中是否仍可读？ | 同一标签/状态的线性 probe；跨条件训练-测试（train A → test B） |
| 因果功能 | 两个表征是否能产生同一种干预效果？ | norm-matched cross-steering / cross-ablation 矩阵 |

| Layer | Input | Core metrics | 回答的问题 |
| --- | --- | --- | --- |
| Mean Direction | 每个实验的全量均值差 $r=\mu_{\text{expert}}-\mu_{\text{non-expert}}$ | signed cosine、norm、逐层 cosine | 两个角色方向是否指向相近的位置？ |
| Sparse Coordinates | 均值方向的逐层 top-$k$ 坐标 | overlap、Jaccard、enrichment、符号一致性、双向能量包含率 | 是否涉及相同神经元？共享部分有多重要？ |
| Role-Transition CKA | 两个实验在相同题目上的逐题配对差 $d_i=h_{i,\text{expert}}-h_{i,\text{non-expert}}$ | **Centered linear CKA**；题目配对打乱作为 null | 不同题目的 role 变化之间，整体几何关系是否相似？ |
| Role-Transition Subspace | 各实验的逐题配对差 $d_i$ | 对 $d_i-\bar d$ 拟合 PCA；比较子空间夹角、交叉重建率 | 除平均方向外，role 对不同题目的影响模式是否相似？ |
| Direction–Subspace | 一个实验的均值方向 $r$ 与另一个实验的 role-transition PCA 子空间 $V_k$ | 投影比例 $R^2=\lVert V_k^\top r\rVert^2/\lVert r\rVert^2$，及随机基线 | 一个实验的平均角色方向在多大程度上落入另一个实验的变化空间？ |

## Brain

**Steingroever 健康常模对齐**

公开数据确实包含 617 名参与者的逐 trial 选择、收益与损失。不过数据混合了 95/100/150 trials 和三种 payoff scheme，因此必须只选与我们 IGT 协议完全匹配的子集。[Steingroever et al. 数据说明](https://openpsychologydata.metajnl.com/articles/jopd.ak)

建议检验：

- `net_block1–5` 学习曲线；
- net score 分布；
- deck preference；
- Wasserstein distance / RMSE；
- 按原始 study 做 held-out，而不是把所有参与者混在一起挑最佳 α。

它能支持的结论是：某个 RSN 条件产生的 IGT 行为最接近健康人常模。行为相似不能识别神经递质水平。因此建议称为 **human behavioural calibration**，而不是 dopamine-axis calibration。


**已有药理学方向对照：保留即可**

现在的定性文献结论已经够用了：

- α+ 与较早承诺、较高下注或 reward seeking 的方向相似；
- α− 与较保守行为的方向相似。

把它作为 Discussion 中的 correspondence table，不必继续寻找无法获得的三组相关系数。措辞停在：

> RSN manipulation exhibits behavioural correspondence with reported dopaminergic pharmacology effects.

**fMRI RSA：有潜力，但暂不作为当前投稿必做项**

NARPS 确实有 108 名参与者、四个 mixed-gamble runs、trial timing、BIDS 数据和公开预处理结果，技术上适合开展共享刺激分析。[NARPS 数据说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602933/)

- 必须把 NARPS 中完全相同的 gain/loss gamble 输入 LLM，才能构建可比较的 RDM；现有 MCQ/Betting 刺激不能直接与 fMRI trial 做 RSA。
- 即使 vmPFC/striatum RSA 显著，也只能说明 **representational correspondence**，不能“直接证明 RSN 操纵 reward representation”。
- Broca/Wernicke 不显著也不能证明“不是语言表征”；ROI 功效、刺激语言量和噪声都会造成 null。
- 还需要控制 gain、loss、accept/reject、RT 等 nuisance RDM，否则相关性可能只是两个系统都编码金额。

因此这会变成一个独立而完整的 brain–LLM 项目，不太可能只是附加分析。

我的建议优先级是：

1. 完成 Thinking Curve、GSM-Hard 和 causal direction control。
2. 并行做 Steingroever 健康常模的小型定量对齐。
3. 将药理学方向一致性放进 Discussion。
4. NARPS RSA 放入 future work；如果前面的行为对齐很漂亮，再考虑扩成后续论文。
