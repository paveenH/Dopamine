```text
发现 RSN
   ↓
它能否像 dopamine-like gain 一样改变 wanting/engagement？
   ↓
它如何影响 reasoning？
   ↓
它改变的是能力，还是 commitment state？
   ↓
为什么效果随 model、task、Chat interface 改变？
   ↓
Role、Confidence、Chat 是否共享内部机制？
```

## 1. 最初：我们从 RSN 出发

父项目发现了一小组 Role-Sensitive Neurons：

- 主要位于中间层；
- 可以双向调节模型愿不愿意回答；
- 改变回答率和 confidence；
- 在部分条件下，conditional accuracy 相对保留。

最初的假设是：RSN 可能像一个 dopamine-like gain，调节 wanting、engagement 和 action readiness，而不是直接增加知识。

这是整个 Dopamine 项目的起点。

## 2. 第一阶段：行为学验证

我们用不同任务问：α 是否会改变“愿不愿意投入、下注、等待、坚持或挑战”？

| 实验 | 做了什么 | 主要结果 |
|---|---|---|
| Confidence Betting | 答题后下注 | α 明显改变下注，但准确率基本不变；最干净的 wanting–knowing dissociation |
| CGT-Sequential | 决定何时 Accept | α 改变 commitment timing／愿不愿意等待 |
| IGT | 100轮选择与反馈学习 | Llama 有局部工作点；Qwen 的 deliberation 和策略结构会变，但学习收益不稳定 |
| Bandit PV9–PV11 | 测 directed exploration | α 改变 policy wording 和 sharpness，却没有可靠改变 information acquisition |
| HaluEval | 是否挑战给定答案 | 中等 +α 降低 challenge threshold；过高会 over-challenge 和格式失控 |

这一阶段最后得到的结论不是“统一 wanting axis”，而是：α 稳定影响 engagement、**commitment timing** 和策略表达；能否变成有效学习或信息探索，取决于任务结构。

Bandit/BAI 已经关闭，不需要再救。

主要文档是 [Behaviour.md](/Users/paveenhuang/Downloads/Dopamine/Note/Behaviour.md)。

## 3. 第二阶段：Reasoning working point

然后我们问：这种 gain 能不能改善推理？

### Llama Bare

GSM8K 出现明显的非线性曲线：

- α=0：60.0%
- α=−6：78.0%
- α=−8：掉到40.3%

所以当时看起来像一个 asymmetric working point。

### Qwen Bare

Qwen 没有复制同样的峰，而是：

- 随正向 α 改成 reasoning-first；
- 高剂量进入 plateau；
- 没有出现 Llama 那种右侧下降。

因此我们已经知道：两个模型共享“α 改变 commitment”的现象，但不共享同一个行为曲线。

### 跨任务转移

固定 GSM8K 的工作点，不在新任务重新搜索：

- GSM-Hard：两个模型都能转移；
- MATH：主要是 Llama；
- CRUXEval-O：主要是 Qwen；
- LogiQA、BBH No-CoT：双 null；
- 加 explicit CoT 后，部分任务恢复。

所以不存在 universal α，也不存在一个能普遍增强 reasoning 的 direction。

权威结果在 [ReasoningBare.md](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningBare.md)。

## 4. 第三阶段：它究竟改变了什么？

我们分析了生成轨迹和 hidden states。

比较稳定的发现是：

- Prefill gain 随 α 几乎线性变化；
- 行为结果却是非线性的；
- α 会改变 first candidate、formal answer、reason-first、post-commit continuation；
- Qwen 的提升尤其明显地对应 answer-first → reason-first；
- Llama 的最佳点对应 premature commitment 减少。

因此目前最好的功能解释是：RSN 不直接提高模型能力，而是在调节模型何时进入、维持和提交一种推理策略。

### 没成功的部分

最初希望找到类似 tonic/ramping/phasic dopamine 的 Thinking Curve，并用 closed-loop 控制它。

结果是：

- 可以塑造曲线；
- 但曲线形状改变不等于准确率改变；
- manifold 也无法解释 Llama peak 与 Qwen plateau 的差异。

所以已经关闭：

- hormone-like waveform 主张；
- closed-loop accuracy control；
- 用 prefill geometry 解释所有行为曲线。

留下来的有效发现是 commitment transition，而不是 dopamine waveform。

对应文档是 [ThinkingCurve.md](/Users/paveenhuang/Downloads/Dopamine/Note/ThinkingCurve.md) 和 [Manifold.md](/Users/paveenhuang/Downloads/Dopamine/Note/Manifold.md)。

## 5. 最近最重要的发现：Interface 改变了一切

**Native Chat**
### Llama

Native Chat 本身就让模型稳定进入 reasoning-first：

- GSM8K 从 Bare α=0 的60%左右升到约90%；
- 原来的 −6 working point 基本消失；
- 高正向 α 主要导致 premature answer、输出缩短和性能下降。

这说明 Llama Bare 的 −6 峰，很大程度上是在：修复一个退化的 Bare interface baseline，而不是创造新的推理能力。

### Qwen

Qwen 又不同：

- Native Chat baseline 并不总是已经完全进入合适状态；
- +6/+8 在 GSM8K、GSM-Hard 仍可把输出推向 reasoning-first 并提高准确率；
- MATH baseline 已经 reasoning-first，因此没有明显收益。

所以现在最关键的总规律是：α 的效果取决于 model × task × interface × baseline state。Matched-Anchor 进一步说明 final-prefill anchor/injection geometry 很重要，但它没有完全解释 Chat 效应。

对应文档是 [ReasoningChat.md](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningChat.md)。

## 6. 最近的神经表征工作

### Role–Confidence

我们发现：

- 中间层方向 cosine 约0.61；
- top neurons 显著重合；
- 共享 neurons 单位贡献很高；
- 但 CSN steering 无法复现 RSN 的 reasoning working point。

结论：结构相关，功能不同。

### Role–Chat

我们发现：

- Role 与 Chat–Bare mean direction 几乎正交；
- 但 top coordinates 显著高于随机重合；
- Role direction 对 Chat transition subspace 的投影也显著高于随机。

结论：Role 不是完整 Chat state，但与 Chat 的多维状态变化共享部分稀疏结构。

### GRSN/AGRSN

最近又使用全量1319题比较：

- 普通 GSM8K Role prompt；
- 带 abstention instruction 的 Role prompt。

目前最重要的理解不是“寻找一个新 RRSN”，而是：Role representation 本身受 prompt 条件影响；低 mean-direction similarity 不等于完全没有共享的逐题 transition geometry。

这部分正在做最后冻结，应该整合进 [Neurons.md](/Users/paveenhuang/Downloads/Dopamine/Note/Neurons.md)。

## 7. 我们现在到底已经证明了什么？

可以比较有信心地说：

1. **α 是有效的 task-entry gain intervention。**
2. **它主要影响 engagement、commitment 和 policy expression。**
3. **它不是简单的知识或 reasoning-capacity knob。**
4. **行为收益取决于模型原本处于什么状态。**
5. **Chat/post-training 可以重设这个 baseline state。**
6. **Role、Confidence、Chat 具有部分共享结构，但不是同一条轴。**
7. **dopamine 只能作为 computational/behavioral analogy。**

不能说：

- α 就是人工 dopamine；
- 存在 universal optimal α；
- Role、Confidence、Chat 是同一个机制；
- hidden-state similarity 已经证明共享因果通路；
- RSN 能稳定增强 directed exploration；
- Thinking Curve 是 hormone-like waveform。

## 8. 现在真正剩下的核心问题

项目已经不缺行为曲线和 benchmark。现在只剩两个真正有价值的问题：

### 问题一：Chat state 从哪里来？

> 它主要是 prompt formatting/token geometry，还是 instruction post-training 学出的 policy state？

用 Base–Instruct 对照回答。

### 问题二：这种状态关系是否有因果功能？

> 把 Bare hidden state 推向 Chat state，能否让 commitment behavior 也向 Chat 移动？

用 held-out causal state transfer/state-paste 回答。

## 9. 下一步只做这三件事

1. **冻结 GRSN/AGRSN 结果并统一文档口径。**
2. **做 Llama Base–Instruct α=0 对照。**
3. **做 held-out causal state transfer。**

结果清楚且因果链完整，就收窄投 TACL；结果较混合但形成完整、诚实的机制边界图谱，就投 TMLR。

一句话概括整个项目：

> 我们最初想验证人工 dopamine；最后发现了一个更具体也更可信的机制：RSN 是受 post-training 与 interface baseline state 调节的 commitment gain，它能改变模型如何进入和表达推理，但不直接等于能力、探索或生物 dopamine。


---
如果按照 TMLR 写，当前最合适的主线是：

> **RSN steering provides a causal task-entry gain that changes engagement and commitment. Its downstream behavioral effect depends on the model, task, interface, and baseline state. This produces dopamine-like functional patterns in some settings, together with clear boundaries and null results.**

这是一篇“提出假设—系统检验—确定适用边界”的论文。重点不再是寻找统一的 Role switch，也不要求证明人工 dopamine。

## 建议标题

我最推荐：

> **Dopamine-Inspired Adaptive Gain Control in Language Models: Commitment, Interface Dependence, and Functional Boundaries**

更保守的版本：

> **Role-Sensitive Gain Controls Commitment in Language Models**

前者更适合 TMLR 完整故事；后者可以避免审稿人把主要注意力放在生物学类比上。

## 论文回答四个问题

### RQ1：RSN 是否改变 engagement，同时保留 knowing？

最强证据来自行为任务：

- Confidence Betting：下注明显变化，accuracy 基本稳定；
- CGT-Sequential：accept timing 随 α 系统变化；
- IGT：deliberation、switching 和策略结构变化；
- HaluEval：challenge threshold 变化。

这里的结论是：RSN 可以改变模型表达行动、投入和承诺的方式，而不必同时改变它知道什么。

Betting 是最干净的主结果，CGT 是 commitment timing 的补强，IGT 和 HaluEval 提供任务边界。

### RQ2：这种 gain 如何影响 reasoning？

核心结果是：

- Llama Bare GSM8K 出现 `−6` 局部工作点；
- Qwen 出现正向高剂量 plateau；
- 两个模型的 entry gain 都近似线性；
- downstream commitment transfer function 不同；
- 有效条件通常伴随 answer-first → reason-first 或 premature commitment 减少；
- commitment 改变并不保证 accuracy 提高。

结论应写成：RSN does not uniformly increase reasoning capacity. It changes how entry gain is converted into commitment behavior, which can improve performance when the baseline commitment regime is poorly calibrated.

这部分由 [ReasoningBare.md](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningBare.md) 和 [ThinkingCurve.md](/Users/paveenhuang/Downloads/Dopamine/Note/ThinkingCurve.md) 支撑。

### RQ3：为什么工作点不能跨条件统一？

Native Chat 是这篇论文最重要的解释性结果：

- Llama Chat baseline 已经接近 reasoning-first；
- Bare 下的 `−6` 峰因此消失；
- 高正向 α 主要造成提前回答和性能下降；
- Qwen 在部分 Chat 条件下仍可被 `+6/+8` 推向更合适的状态；
- MATH baseline 已经 reasoning-first 时，进一步 steering 没有明显收益。

Matched-Anchor 又说明 injection anchor 会影响剂量曲线，但不能完整解释 Chat 效应。

因此，论文的中心模型是：

```text
RSN task-entry gain
        ↓
model/task/interface-specific baseline state
        ↓
commitment transfer function
        ↓
reasoning-first / answer-first / persistence / challenge
        ↓
task outcome
```

### RQ4：dopamine analogy 能走到哪里？

支持的对应：

- wanting–knowing 部分解离；
- engagement 与 commitment timing；
- 线性 gain 输入对应非线性行为输出；
- 过低、适中、过高 gain 可能产生不同失效状态。

限制也很清楚：

- Bandit 没有可靠改变 directed information acquisition；
- closed-loop waveform shaping 没有控制 accuracy；
- manifold 无法解释 Llama peak 与 Qwen plateau；
- raw α 不能跨模型或任务比较；
- 没有生物 dopamine、受体或脑区证据。

因此建议全文使用：**a causal RSN gain mechanism with functional parallels to dopaminergic adaptive calibration**，不要直接称为 “a dopaminergic mechanism”。

## GRSN/AGRSN 放在哪里

在 TMLR 稿件里，GRSN/AGRSN 不应成为主线实验，也不需要修复成一个统一方向。

它适合支持一个边界结论：

> Role contrasts do not yield a prompt-invariant one-dimensional controller. They recruit partially shared, question-structured geometry whose mean direction and sparse realization remain prompt-dependent.

也就是说：

- centered CKA／transition geometry 可以说明存在部分共享结构；
- mean direction 和 sparse support 的差异说明没有 universal Role vector；
- 这不影响已有的 causal RSN steering 结果；
- 它限制的是“稳定 Role switch”主张。

建议主文用一个简洁 panel 或一段结果说明，完整 Metrics 1–5 放 Supplement。旧 RRSN 不进入主文。

## 推荐的论文结构

### 1. Introduction

提出 dopamine-inspired adaptive-gain hypothesis，并说明本文检验三层证据：

1. behavioral dissociation；
2. reasoning and commitment；
3. model/task/interface boundaries。

### 2. Framework and Experimental Setup

定义：

- RSN intervention；
- α 是 intervention strength；
- state versus capacity；
- engagement、commitment、outcome 三层读数；
- Llama/Qwen、Bare/Chat、任务和统计口径。

### 3. Behavioral Evidence for Engagement Gain

主放：

- Confidence Betting；
- CGT-Sequential；
- IGT。

HaluEval 作为 boundary probe；Bandit 只保留最终结论，不展开全部协议历史。

### 4. Adaptive Calibration of Reasoning

主放：

- Llama GSM8K dose response；
- Qwen GSM8K dose response；
- MATH/GSM-Hard 作为跨任务验证；
- fixed-workpoint transfer 的成功与 null。

### 5. Commitment as the Downstream Behavioral State

主放：

- entry gain linearity；
- early candidate；
- reason-first；
- pre/post-commit allocation；
- commitment 可以预测有效区间，但不是已证明的 causal mediator。

closed-loop 和 manifold 作为限制结果简要报告。

### 6. Interface-Dependent Gain Control

主放：

- Bare vs Native Chat；
- Matched-Anchor；
- Llama/Qwen 的不同反应；
- baseline-state calibration 模型。

这一节应成为整篇论文的解释中心。

### 7. Representational Relationships and Specificity

简要整合：

- Role–Confidence：结构相关、功能不同；
- Role–Chat：mean direction 不同、sparse/subspace 部分共享；
- GRSN/AGRSN：prompt-conditioned role geometry。

不把结构相似性写成功能等价。

### 8. Scope of the Dopamine Analogy

集中呈现：

- 支持什么；
- Bandit、waveform、manifold 等 null 限制了什么；
- 为什么它是 computational analogy。

### 9. Discussion and Conclusion

最终结论：RSN steering acts as a controllable commitment gain whose functional consequences depend on the baseline policy state established by the model, task, and interface.

## 主文与补充材料的取舍

| 主文 | Supplement |
|---|---|
| Betting 核心结果 | 完整任务表和 domain breakdown |
| CGT commitment timing | v1–v5 protocol lineage |
| IGT 主要曲线 | 全指标和全部 α |
| Llama/Qwen reasoning curves | 每个 benchmark 完整表 |
| Commitment transfer | 所有 marker sensitivity |
| Bare/Chat/Matched-Anchor | 完整生成诊断 |
| Role–Confidence/Chat summary | GRSN/AGRSN Metrics 1–5 |
| Bandit 最终 boundary | PV6–PV11 历史 |
| Manifold/closed-loop verdict | 完整 negative analyses |

## 现在还需要做什么

按照 TMLR 路线，我建议依次完成：

1. **建立 claim–evidence table。**  
   每个 claim 对应数据、模型、任务、统计检验和限制，解决当前结果分散的问题。

2. **冻结六张主图。**

   - Figure 1：理论框架与实验地图；
   - Figure 2：Betting/CGT 的 wanting–knowing 与 commitment；
   - Figure 3：Llama/Qwen reasoning dose response；
   - Figure 4：entry gain → commitment → outcome；
   - Figure 5：Bare/Chat/Matched-Anchor；
   - Figure 6：跨任务 evidence-and-boundary matrix。

3. **统一结果口径。**  
   统一 `first_acc`、有效 denominator、Holm family、模型特定 α、Bare/Chat 标记，并修正 `CLAUDE.md` 与当前结果状态不一致之处。

4. **补一个小而有力的 specificity control。**  
   如果当前还缺 causal matched random/orthogonal steering，应只在一个 reasoning anchor 和一个 behavioral anchor 上完成。它对 TMLR 的价值高于继续新增 benchmark。

5. **先写完整初稿，再决定额外实验。**  
   Base–Instruct 和 held-out state-paste 会增强论文，但不是 TMLR 的前置条件。初稿完成后，如果最大的审稿风险仍是“Chat state 来自哪里”，再补 Base–Instruct；如果风险是“结构关系是否有因果功能”，再补 state transfer。

## 当前最合适的贡献表述

Contribution:
1. 提出并系统检验 dopamine-inspired RSN adaptive-gain hypothesis。
2. 证明 RSN 可因果改变 engagement 和 commitment，并在部分任务中与 knowing 解离。
3. 发现相似的线性 task-entry gain 会通过不同的 commitment transfer functions 产生模型、任务和接口特定的行为曲线。
4. 用完整的 null 和 boundary evidence 限定该机制：它不构成 universal dopamine axis、universal optimal α 或 prompt-invariant Role switch。


