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

