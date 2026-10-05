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

发现了一小组 Role-Sensitive Neurons：

- 主要位于中间层；
- 可以双向调节模型愿不愿意回答；
- 改变回答率和 confidence；
- 在部分条件下，conditional accuracy 相对保留。

最初的假设是：RSN 可能像一个 dopamine-like gain，调节 wanting、engagement 和 action readiness，而不是直接增加知识。

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

[Behaviour.md](/Users/paveenhuang/Downloads/Dopamine/Note/Behaviour.md)。

## 3. 第二阶段：Reasoning working point

然后我们问：这种 gain 能不能改善推理？

### Llama Bare

GSM8K 出现明显的非线性剂量响应：

- α=0：60.0%
- α=−6：78.0%
- α=−8：40.3%

这表明，在当前协议下存在一个 **asymmetric working point**：适量干预能改善表现，但继续增加负向剂量会导致明显下降。

### Qwen Bare

Qwen 没有呈现同样的峰形曲线，而是：

- 随正向 α 增大，输出更倾向于 reasoning-first；
- 准确率在高剂量区间进入 plateau；
- 在已测试范围内，没有观察到 Llama 式的峰后下降。

因此，两个模型都表现出 **α 对答案出现时机与 commitment-related behavior 的调节**，但准确率的响应方向和曲线形状不同。不同模型的 raw α 也不能直接视为相同干预剂量。

### 跨任务转移

固定 GSM8K 的工作点，不在新任务上重新搜索：

- GSM-Hard：两个模型都能转移；
- MATH：主要是 Llama；
- CRUXEval-O：主要是 Qwen；
- LogiQA、BBH No-CoT：两个模型均未检测到明确收益；
- 加入 explicit CoT 后，部分任务重新出现收益。

这些结果说明，**RSN 的调节作用具有一定跨任务共性，但推理收益依赖模型、任务与输出协议。** 固定工作点的成功迁移支持它不只是 GSM8K 特有的效应；迁移失败则限定其适用范围。现有证据尚不支持一个普遍有效的 α，也不足以将 RSN 定义为通用推理增强方向，但不能据此否定共享调节机制的可能性

[ReasoningBare.md](/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningBare.md)

## 4. 第三阶段：它究竟改变了什么？

我们进一步分析生成轨迹和 hidden states，尝试理解准确率变化背后的行为过程。

比较稳定的发现是：

- Prefill gain 随 α 几乎线性变化，可作为干预是否生效的 manipulation check；
- 行为和准确率的响应却是非线性的；
- α 会改变首次候选答案、正式答案的出现时机、reason-first 比例，以及答案提交后的续写行为；
- Qwen 的准确率提升明显伴随 answer-first → reason-first 的转变；
- Llama 的最佳工作点伴随过早给出答案的行为减少。

这些结果更支持一种**回答与承诺过程的调节解释**：RSN steering 改变模型如何展开推理和何时给出答案，而不只是统一提高准确率。不过，“入口状态 → 输出与承诺行为 → 准确率”目前仍是与结果一致的解释链，尚未完成因果中介验证，也不能据此排除其他机制。

### 未获得支持的路线

最初，我们尝试寻找类似 tonic/ramping/phasic 的 Thinking Curve，并通过 closed-loop 控制其形状。

结果表明：

- 可以塑造所测曲线；
- 曲线形状改变并不稳定地带来准确率收益；
- 现有 prefill manifold 分析尚未稳定解释 Llama peak 与 Qwen plateau 的差异。

因此，当前不再将以下内容作为投稿主线：

- 将 Thinking Curve 直接解释为 dopamine-like waveform；
- 通过 closed-loop 曲线塑形实现可靠的准确率控制；
- 仅用 prefill geometry 解释完整的行为剂量响应。

留下来的核心发现是：**入口干预与后续生成动态可以解耦，而答案时机、输出顺序和 commitment-related behavior 是理解收益与失败的重要线索。**

对应文档是 [ThinkingCurve.md](/Users/paveenhuang/Downloads/Dopamine/Note/ThinkingCurve.md) 和 [Manifold.md](/Users/paveenhuang/Downloads/Dopamine/Note/Manifold.md)。

## 5. 最近最重要的发现：Interface 改变基线与干预响应

### Llama：Native Chat 下，Bare 的最佳工作点不再带来同样收益

Native Chat 条件下，Llama 的基线更倾向于 reasoning-first：

- GSM8K 准确率从 Bare α=0 的约60%提高到约90%；
- Bare 下 α=−6 的明显收益，在 Native Chat 下不再出现；
- 高正向 α 更多伴随过早给出答案、输出缩短和准确率下降。

这一对照提示：**Bare 下的 −6 收益可能部分来自对接口相关输出状态的调整。** 它不必被解释为新增推理能力，但目前也不能据此认定 Bare 是“退化状态”，或证明收益完全由基线修复产生。

### Qwen：Chat 并不统一消除 steering 收益

Qwen 呈现不同的响应：

- Native Chat 并不使所有任务都呈现相同的基线输出模式；
- 在 GSM8K、GSM-Hard 上，+6/+8 仍伴随更多 reasoning-first 输出和准确率收益；
- MATH 基线已经以 reasoning-first 为主，未观察到类似收益。

这些结果与“**干预收益依赖原有输出状态**”的解释一致，但 reasoning-first 本身还不能被视为准确率提升的充分条件或已验证的因果中介。

### 当前结论

**RSN steering 的效果具有模型、任务和接口依赖性；各条件的基线输出状态，是理解这种差异的重要线索。**

Matched-Anchor 对照进一步提示 final-prefill anchor 与注入位置会影响结果，但尚不能将 Chat 效应完全归因于注入几何，也不能单独归因于 instruction post-training。

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

## 8. 其余问题

#### 问题一：Chat state 从哪里来？

> 它主要是 prompt formatting/token geometry，还是 instruction post-training 学出的 policy state？-> 用 Base–Instruct 对照回答。

#### 问题二：这种状态关系是否有因果功能？

> 把 Bare hidden state 推向 Chat state，能否让 commitment behavior 也向 Chat 移动？-> 用 held-out causal state transfer/state-paste 回答。


# TMLR

### Title

> **Dopamine-Inspired Adaptive Gain Control in Language Models: Commitment, Interface Dependence, and Functional Boundaries**

> **Role-Sensitive Gain Controls Commitment in Language Models**

## Research Questions

### RQ1：RSN 是否改变 engagement，同时保留 knowing？

最强证据来自行为任务：

- Confidence Betting：下注明显变化，accuracy 基本稳定；
- CGT-Sequential：accept timing 随 α 系统变化；
- IGT：deliberation、switching 和策略结构变化；
- HaluEval：challenge threshold 变化。

这里的结论是：RSN 可以改变模型表达行动、投入和承诺的方式，而不必同时改变它知道什么。Betting 是最干净的主结果，CGT 是 commitment timing 的补强，IGT 和 HaluEval 提供任务边界。

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

# Paper

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


