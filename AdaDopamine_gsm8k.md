## 0. Background

### 0.1 Prompt Template Symmetrization

| | 舊 No-CoT | 舊 CoT |
|---|---|---|
| 標題 | `Solve the following math problem.` | `Solve the following math problem **step by step**.` |
| 格式指示 | `Provide your final numeric answer after '####'.` | （無） |
| 推理提示 | （無） | `Let's think step by step.` |

修正後（對稱）——`####` 指示在 No-CoT / CoT 都保留，唯一變量是 `Let's think step by step.` 一行：

```
No-CoT:  Solve the following math problem.
         Question: {context}
         Provide your final numeric answer after '####'.
         Answer:

CoT:     Solve the following math problem.
         Question: {context}
         Let's think step by step.
         Provide your final numeric answer after '####'.
         Answer:
```

**`####` 措辞 = "Provide your final numeric answer after '####'."（中性）**。一个更催促的变体 `"Give your final answer as a single number after '####'."`（pushy）会诱导**抢答**，被保留为**正向对照（positive control）**——见 §2。

### 0.2 Dopamine Prior Knowledge

行为学先验：多巴胺不是单纯的"快乐分子"，更准确地说是**驱动力 / incentive salience / "wanting"** 递质，调控动机、期待、奖赏趋近与行动阈值。本研究把 α 看作在这一 wanting 轴上双向移动工作点：α 正向对应 **over-wanting / 过度唤起**，α 极端负向对应 **under-wanting / 唤起不足**；整体框架是 Yerkes–Dodson 倒 U——过高过低都有害，最优落在中间偏负（本数据 α=−6）。

- **过高 DA / over-wanting（对应 α→正向）**：行为上表现为**冲动性抢答（impulsivity）+ 认知僵化 / 强迫性反复（compulsivity / perseveration）**——急于扑向"给出答案"这个目标而跳过必要推导，以及拿到答案后仍反复复查、纠结格式、卡在格式死循环里。**注意这更贴合冲动 / 强迫，而非焦虑**：数据中 +α 端**没有**焦虑典型的回避 / freezing / 犹豫（抢答率随 +α 单调上升，见 §2.2），呈现的是"急着 commit"；而 `#### N #### N` 死循环是认知神经科学意义上的**固著（perseveration）**,不是焦虑的发散灾难化担忧。功能上这**与 mesolimbic incentive-salience overload（VTA→NAcc 型 wanting 过载）及执行控制失效相容**：极高的诱因显著性使主体不计成本扑向目标（冲动），同时灵活切换 / 抑制已启动反应的能力下降（固著）——这在行为上更近强迫性 over-checking 与冲动特征，而非焦虑综合征。**注意目前只有行为同构，尚未定位实际脑区对应关系**，故这里说"相容"而非"落在"某一回路。这个签名对应 §2.3「Can't Let Go」：α+4 trace 中常见"把已经算对的答案当可疑"（Q100、Q16）和答完仍寻找 "more efficient way"（Q68）。
- **过低 DA / under-wanting（对应 α→极端负向）**：行为上对应动力不足、快感缺失、退缩、bradykinesia 式的行动迟缓；在本任务中的可观测类比不是"写得短 / 不想答"，而是**commitment-formation failure**。§2.4 文本核验否定了两个更直观假设：α=−8 并非大量出现 "I am done / 不答了" 的词汇性退缩（跨 α 平坦，多为礼貌 loop 尾），也不是敷衍短答（长度 / 等式数平坦）；真正失败模式是 **answer-candidate oscillation**——锁不住答案，在两个候选值之间来回切，导致 committed_acc 崩到 23.6%。

> **主机制表述（本项目采用）**：+α 端 = **over-wanting → 冲动（impulsivity, 抢答）+ 认知僵化 / 强迫性反复（compulsivity / perseveration, loop）**，功能上与 **mesolimbic incentive-salience overload（VTA→NAcc 型）+ 执行控制失效**相容（只声称行为同构，非脑区定位）；−α 极端端 = **under-wanting → commitment-formation failure**。这比"焦虑"框架**机制契合度更高、论述负担更小**：冲动与固著都在我们已有的 wanting / incentive-salience 主线内，无需另起 threat/freeze 回路。
>
> **限定**：1. 本实验只声称行为同构，**α steering ≠ 生物多巴胺**，也不证明 LLM 有生理或主观状态；2. 不做 mania / hypomania 类比——躁狂是跨情绪+精力+睡眠的综合征，我们只有"冲动+固著"两个窄行为，撑不起该诊断类比；3. 脚本 `analyze_loop_anxiety.py` / `ANXIETY_PATTERNS` 沿用 "anxiety" 命名（改名成本高、破坏 U 形复现），但其命中的四子类（self-doubt / format-fixation / persona-reassurance / over-precision）**实测对应的是强迫性 over-checking，不是临床焦虑**——阅读表格时按"强迫/固著"解读。
>
> **次要旁证（不作主锚，DA→焦虑另有通路特异性）**：DA 亦有一条独立的焦虑通路证据，最强因果来自 **VTA→IPN（D1）**，机制是威胁高估 / 过度警觉——但这**不是**本数据的主要解释（我们没观测到回避 / freezing），仅作为 DA 多下游效应的旁注列出。来源：[PMC7687288 (VTA→IPN dopamine promotes anxiety)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/) · [MIT News 2018 (dopamine vigilance & anxiety)](https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107) · [Frontiers Neurosci 2020 (dopaminergic alteration in anxiety/compulsive disorders)](https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full) · [J. Neurosci 2019 (dopaminergic mechanisms of trait anxiety)](https://www.jneurosci.org/content/39/14/2735)


## 1. Llama on GSM8K: Performance Summary

Llama3.1-8B-Instruct 在 GSM8K 上呈现出三个主要结果：

- No-CoT 剂量曲线是明显的**非对称峰形**：准确率在 `α=−6` 达到最高，但到 `α=−8` 明显崩落。
- CoT 在已测试的 `−4/0/+4` 三个剂量上都提高准确率，同时保留 `−4 > 0 > +4` 的局部排序。
- 催促式措辞会降低准确率并压缩不同 α 之间的差异，带 persona 的条件尤其敏感。

**Setup.** Llama3.1-8B-Instruct，GSM8K 300 题，greedy decoding。正文统一报告 offline `first_acc`；`last_acc` 仅用于检查后续答案修改，不作为主要性能指标。所有数据来自同一冻结 production batch，具体运行配置、数据路径和提取口径见 `CLAUDE.md`。

### 1.1 Main Dose–Response

下表合并完整 No-CoT 曲线，以及现有的 CoT 和 pushy 对照。`—` 表示该条件未运行。

| Condition | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Plain No-CoT, first acc** | 40.3% | **78.0%** | 73.0% | 69.0% | 60.0% | 57.0% | 55.3% | 55.0% | 53.7% |
| Plain No-CoT, last acc | 41.7% | 74.7% | 68.3% | 65.3% | 55.3% | 55.3% | 52.7% | 53.3% | 52.3% |
| **Plain CoT, first acc** | — | — | **85.0%** | — | 69.0% | — | 59.7% | — | — |
| Plain CoT, last acc | — | — | 84.7% | — | 68.3% | — | 59.0% | — | — |
| **Pushy No-CoT, first acc** | — | — | 61.7% | — | 55.7% | — | 53.0% | — | — |
| Pushy CoT, first acc | — | — | — | — | 57.7% | — | — | — | — |

#### Negative α: Improvement Followed by Collapse

从 `α=0` 向负向移动时，No-CoT first accuracy 先连续上升：`60.0 → 69.0 → 73.0 → 78.0%`。因此，`α=−6` 是当前九点 No-CoT 曲线中的离散最佳点，相比 `α=0` 提高 18.0 pp。但负向并非越强越好。继续移动到 `α=−8` 后，准确率从 78.0% 降至 40.3%，不仅失去此前增益，还比 baseline 低 19.7 pp。

> GSM8K 上存在一个以 `α=−6` 为峰值的非对称工作区间；适度负向 steering 有利，但极端负向 steering 会导致性能崩落。

#### Positive α: Gradual Decline and Flattening

正向一侧的 first accuracy 为：`60.0 → 57.0 → 55.3 → 55.0 → 53.7%`，准确率随 α 增加而下降，但降幅主要集中在 `0 → +4`；从 `+4` 到 `+8` 只再下降 1.6 pp。因此，正向一侧不是持续加速恶化，而是：

> 准确率先下降，随后在约 54% 附近逐渐趋平。

#### First Versus Last Answer

完整 No-CoT 曲线中，first 与 last accuracy 的绝对差均不超过 4.7 pp。除 `α=−8` 外，多数条件都是 first accuracy 略高，说明后续答案修改通常没有改善总体表现。不过，first–last gap 只能描述输出过程中答案是否被改动，不能用于判断答案在模型内部何时形成。具体的修改方向、重复提交和收口行为见 §2。

### 1.2 CoT and Prompt Wording

#### CoT Improves All Three Tested Doses

在已有的三个剂量上，CoT 相比 No-CoT 的 first-accuracy 差异为：

| α | No-CoT | CoT | ΔCoT |
|---:|---:|---:|---:|
| −4 | 73.0% | **85.0%** | **+12.0 pp** |
| 0 | 60.0% | **69.0%** | **+9.0 pp** |
| +4 | 55.3% | **59.7%** | **+4.4 pp** |

CoT 在三个剂量上都提高了准确率，而且局部排序保持为：

`α=−4 > 0 > +4`

`α=−4 + CoT` 的 85.0% 是当前所有已测试 Llama GSM8K 条件中的最高准确率。

但 CoT 目前只有 `−4/0/+4` 三个剂量点，没有 `−6 CoT`。因此可以说 `−4 + CoT` 是**已测试条件中的最高值**，不能据此认定它是完整 CoT 剂量曲线的全局最佳点。CoT 增益在负向一侧更大，但当前结果只是描述性比较。仅凭三个剂量点，不能证明 CoT 与 steering 存在正式交互，也不能证明两者作用机制独立或严格可加。

#### Pushy Wording Compresses the Dose Difference

No-CoT 条件下，plain 与 pushy wording 的结果为：

| α | Plain | Pushy | ΔPushy |
|---:|---:|---:|---:|
| −4 | 73.0% | 61.7% | −11.3 pp |
| 0 | 60.0% | 55.7% | −4.3 pp |
| +4 | 55.3% | 53.0% | −2.3 pp |

Plain wording 下，`−4` 到 `+4` 的准确率跨度为 17.7 pp；pushy wording 下缩小为 8.7 pp。

Pushy 条件仍然保留 `−4 > 0 > +4` 的排序，但三个剂量被拉得更近。这说明催促式答案指令不只是整体降低表现，也会减弱当前区间内不同 steering 剂量的区分度。

在 `α=0 + CoT` 条件下，pushy wording 同样将准确率从 69.0% 降至 57.7%，下降 11.3 pp。因此，CoT 本身不能抵消催促式措辞的负面影响。

这些结果说明模型性能明显依赖 prompt wording；具体的抢答、重复和收口行为见 §2。

### 1.3 Persona and Wording Sensitivity

以下结果均为 `α=0`、No-CoT。

| Role | Plain first | Plain last | Pushy first | ΔPushy |
|---|---:|---:|---:|---:|
| neutral | 60.0% | 55.3% | 55.7% | −4.3 pp |
| an expert | 58.0% | 57.7% | **34.0%** | **−24.0 pp** |
| a non expert | **68.0%** | 65.7% | 48.3% | −19.7 pp |
| a primary school teacher | **68.0%** | 67.0% | 41.7% | **−26.3 pp** |

Plain wording 下，persona 没有统一的性能方向：

- `an expert` 为 58.0%，略低于 neutral 的 60.0%。
- `a non expert` 和 `a primary school teacher` 均为 68.0%，高于 neutral。
- 因此，不能简单概括为“专家 persona 提高数学能力”或“非专家 persona 降低数学能力”。

更稳定的结果来自 wording sensitivity：

- Neutral 在 pushy wording 下下降 4.3 pp。
- 三个 persona 条件下降 19.7–26.3 pp。
- 最大降幅出现在 `a primary school teacher`（−26.3 pp），其次是 `an expert`（−24.0 pp）。


> Persona 会明显放大模型对催促式答案指令的敏感性，但不同 persona 在普通措辞下并不存在统一的准确率方向。

Persona 如何改变答案后的身份独白和重复内容，将在 §2 中讨论。这里不把 persona accuracy 解释为真实身份、能力认同或主观心理状态。

### 1.4 Summary

Llama 在 GSM8K 上的性能结果可以概括为：

1. **No-CoT 剂量曲线呈非对称峰形。** 准确率在 `α=−6` 达到 78.0%，但 `α=−8` 降至40.3%。
2. **正向 α 伴随准确率下降。** 主要降幅发生在 `0 → +4`，之后逐渐趋平。
3. **CoT 在已测试的三个剂量上都提高准确率。** `α=−4 + CoT` 达到 85.0%，但由于缺少完整 CoT 曲线，不能称为全局最优剂量。
4. **催促式措辞会降低性能并压缩剂量差异。** 这一影响在 persona 条件下尤其明显。
5. **本节只报告性能现象。** 剂量如何影响抢答、答案形成、重复和收口行为，将在 §2 中分析；这些行为也不能直接等同于生物学 dopamine。
## 2. GSM8K Output Behavior and Functional Wanting Interpretation

本节关注的不是准确率本身，而是 α 如何改变模型的答案形成、提交和停止行为。

主要观察可以概括为：

- `α=−6/−4` 附近具有较高的提交质量和较少的语义性反复。
- 正向 α 整体伴随更早输出答案，以及更多提交后的检查和重复。
- `α=−8` 是另一种边界失效：模型常在开头正式提交答案，随后在多个候选值之间振荡。
- CoT 增加分步结构并减少语义性反复，但没有消除 α 的方向差异。

这里将 wanting / incentive salience 作为一种**功能类比**：α 是实际施加的 RSN gain intervention，而 wanting 并未被直接测量。因此，正文首先报告可观测的输出行为，再讨论它与 engagement、commitment 和 stopping control 的关系。

### 2.1 Dose-Dependent Output Behavior

下表汇总 neutral、plain、No-CoT 条件下的完整九点剂量曲线。Accuracy 与 §1 相同，在此仅作为行为指标的参照。

| Metric | −8 | **−6** | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Accuracy** | 40.3% | **78.0%** | 73.0% | 69.0% | 60.0% | 57.0% | 55.3% | 55.0% | 53.7% |
| **Committed accuracy** | **23.6%** | **79.7%** | 78.3% | 76.2% | 68.6% | 66.0% | 63.9% | 62.5% | 58.5% |
| Commit rate | 60.7% | 60.7% | 58.3% | 63.0% | 62.7% | 53.0% | 49.0% | 45.3% | 53.0% |
| Median `####` position | **0%** | **31%** | 25% | 18% | 18% | 19% | 16% | 14% | 14% |
| Mean `####` position | 9% | 33% | 28% | 24% | 22% | 25% | 26% | 22% | 20% |
| Premature, leading digit | 4 | 77 | 195 | 219 | 199 | 211 | 229 | 231 | 206 |
| **Premature, either rule** | **175** | **94** | 195 | 223 | 206 | 215 | 232 | 233 | 210 |
| `####` at text start | **171** | 17 | 9 | 18 | 20 | 19 | 12 | 19 | 20 |
| **At least two answer switches** | **41** | 3 | 4 | 10 | 9 | 6 | 6 | 12 | 10 |
| At least three candidate values | **11** | 1 | 4 | 6 | 3 | 1 | 3 | 6 | 4 |
| Median generation length | 2,328 | 2,127 | **2,044** | 2,091 | 2,107 | 2,100 | 2,228 | **2,284** | 2,206 |
| Loop samples | 213 | 264 | 242 | 230 | 232 | 223 | 220 | 225 | 233 |
| At least two `Step` markers | 31 | **185** | 73 | 36 | 25 | 31 | 31 | 27 | 37 |
| Stuck loops | 6 | 16 | 27 | 25 | 25 | 24 | 20 | 30 | 22 |
| Median equation count | 4 | 4 | 3 | 3 | 3 | 4 | 3 | 3 | 4 |
| **Full-text compulsive repetition** | **115** | **23** | 34 | 61 | 77 | 82 | 91 | 94 | 88 |

`Committed accuracy` 的分母是输出了可解析答案标记的样本，因此它描述的是**提交质量**，不能替代总体 accuracy。`####` position 也只在已提交样本中定义，而 full-text compulsive repetition 的分母固定为全部 300 题。

#### Submission Quality

在 `α=−6 → +8` 区间内，committed accuracy 从 79.7% 单调降至58.5%。也就是说，在已经正式提交答案的样本中，剂量越向正侧移动，提交值越容易出错。

Commit rate 本身并不单调，因此准确率变化不能简单解释为“模型更愿意或更不愿意提交”。更稳定的变化来自提交答案的质量和提交前后的文本结构。

#### Answer Timing and Stopping

从负向最佳区域移向正向时，首次 `####` 的典型位置整体提前：

- `α=−6`：31%
- `α=−4`：25%
- `α=+4/+6/+8`：14–16%

但输出长度并没有同步缩短。正向端的 median generation length 反而较长，并在 `α=+6` 达到 2,284 characters。

因此，正向 α 的典型模式不是“更快完成”，而是：更早输出答案，但在答案出现后继续生成。Premature-output 数量并非每个相邻剂量都严格单调，因此不能将其单独视为准确率曲线的中介变量。它与 generation length、答案质量和语义性反复共同描述一种输出状态。

#### Raw Loop Is Not the Main Signal

Loop samples 在所有剂量下都很高，但没有清楚的剂量趋势。Equation count 也基本稳定在 3–4。

因此，单纯的 loop 数量或计算符号数量都无法解释准确率曲线。更有区分力的是：

- loop 中反复出现什么内容；
- 答案是否在多个候选值之间切换；
- 已经输出答案后是否仍持续检查和重算。

### 2.2 Two Distinct Failure Regimes

准确率曲线两端都会失效，但输出形态不同。

| Regime | Opening pattern | Representative metrics | Behavior after the first answer |
|---|---|---|---|
| **Positive α** | 常以裸数字开头 | Leading digit：`229/231/206` at `+4/+6/+8`；full-text repetition：`91/94/88` | 提交后继续检查、重算、确认或重复 |
| **α=−8** | 常以 `#### N` 开头 | Marker at start：171；answer switches：41；committed accuracy：23.6% | 在多个答案候选之间切换，提交值不稳定 |
| **α=−6/−4** | 较少在开头直接提交 | Full-text repetition：23/34；answer switches：3/4 | 更常在形成答案后结束 |

这说明正向高剂量与 `α=−8` 不能视为同一种失败。

- 正向端更接近“答案已经出现，但仍无法停止”。
- `α=−8` 更接近“过早正式提交，随后无法稳定保持一个候选值”。

这些是生成文本中的行为模式。仅凭文本无法确定其内部原因，也不能判断答案振荡来自计算失败、候选竞争还是其他生成动力学。

#### Compulsive-Repetition Subtypes

历史脚本使用 `anxiety` 字段名，但这些指标实际测量的是**强迫性反复确认和行为固著**，不是临床焦虑。

Loop 口径只统计已经进入退化重复的样本，分母随剂量变化；full-text 口径在全部 300 题中检测重复语义，是本节的主要读数。

| Metric | −8 | **−6** | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Any, loop** | 73 | 29 | **27** | 44 | 46 | 54 | 59 | 58 | 57 |
| Self-doubt, loop | 55 | 13 | 13 | 30 | 31 | 40 | 43 | 51 | 46 |
| Format fixation, loop | 32 | 18 | 15 | 19 | 23 | 25 | 37 | 36 | 32 |
| Persona reassurance, loop | 11 | 7 | 7 | 11 | 15 | 13 | 13 | 6 | 6 |
| Over-precision, loop | 4 | 2 | 1 | 5 | 7 | 5 | 5 | 3 | 3 |
| **Any, full text** | **115** | **23** | 34 | 61 | 77 | 82 | 91 | 94 | 88 |
| Self-doubt, full text | 99 | 14 | 24 | 51 | 61 | 72 | 75 | 85 | 73 |
| Format fixation, full text | 58 | 13 | 19 | 28 | 33 | 38 | 46 | 54 | 54 |
| Persona reassurance, full text | 6 | 6 | 5 | 10 | 17 | 12 | 10 | 7 | 6 |
| Over-precision, full text | 6 | 0 | 1 | 8 | 4 | 4 | 5 | 1 | 3 |

两种口径得到的主要结论一致：

- 强迫性反复在 `α=−6/−4` 附近最低。
- 从 `α=0` 向正侧移动时，full-text repetition 从 77 增至 91、94，随后在 +8 略降至 88。
- Self-doubt 是数量最大、剂量变化最清楚的子类。
- Format fixation 也随正向剂量整体增加。
- Persona reassurance 和 over-precision 没有稳定的单调趋势。

`α=−8` 同样具有很高的重复数量，但它同时伴随大量 marker-first 输出和答案候选振荡，因此不能直接归入正向端的“提交后持续检查”。

#### Representative Cases

| Condition | Case | Observed output trajectory | Interpretation |
|---|---|---|---|
| `α=+4` | Q10, gold=5 | 输出一个答案后不断用 “however” 推翻，再次回答后继续推翻 | Self-doubt repetition |
| `α=+4` | Q15, gold=17 | 已得到正确答案，随后持续纠结 `####` 格式 | Format fixation |
| `α=+4` | Q58, gold=15 | 首个答案正确，随后反复请求确认 | Persona reassurance |
| `α=+4` | Q17, gold=36 | 过早提交 40，之后持续进行无意义近似 | Over-precision |
| `α=−8` | Q31, gold=40 | `#### 55` → 正文算出 40 → 在 55 与 40 之间切换 | Correct candidate appears but is not retained |
| `α=−8` | Q112, gold=45 | `#### 70` → 算出 45 → 又漂移至 75 | Multiple candidates without convergence |

这些案例用于说明聚合指标对应的文本形态，不构成独立统计证据。

“I made a mistake” 一类词汇在 `−8/0/+8` 都会出现，因此自我怀疑措辞本身不能区分失败模式。真正有区分力的是后续答案轨迹：是否完成一次有效修正，是否固守同一个答案，或者是否持续在多个候选值之间切换。

### 2.3 How CoT Changes the Output Pattern

下表比较 neutral 条件下 `−4/0/+4 × No-CoT/CoT`。Accuracy 与 §1 相同，在此作为行为变化的参照。

| Metric | −4 No-CoT | 0 No-CoT | +4 No-CoT | −4 CoT | 0 CoT | +4 CoT |
|---|---:|---:|---:|---:|---:|---:|
| **Accuracy** | 73.0% | 60.0% | 55.3% | **85.0%** | 69.0% | 59.7% |
| Last accuracy | 68.3% | 55.3% | 52.7% | 84.7% | 68.3% | 59.0% |
| First–last gap | +4.7 pp | +4.7 pp | +2.7 pp | +0.3 pp | +0.7 pp | +0.7 pp |
| **Committed accuracy** | 78.3% | 68.6% | 63.9% | **81.1%** | 67.3% | 59.8% |
| `####` commit rate | 58.3% | 62.7% | 49.0% | 31.7% | 37.7% | 29.0% |
| Median `####` position | 25% | 18% | 16% | 37% | 36% | 41% |
| Mean `####` position | 28% | 22% | 26% | 37% | 35% | 37% |
| Premature, leading digit | 195 | 199 | 229 | 48 | 121 | 237 |
| **Premature, either rule** | **195** | 206 | 232 | **63** | 151 | 242 |
| Median generation length | 2,044 | 2,107 | 2,228 | 2,113 | 2,091 | 2,078 |
| Loop samples | 242 | 232 | 220 | 284 | 254 | 250 |
| **At least two `Step` markers** | 73 | 25 | 31 | **261** | 220 | 227 |
| Stuck loops | 27 | 25 | 20 | **12** | 28 | 42 |
| Median equation count | 3 | 3 | 3 | 3 | 4 | 3 |
| **Compulsive repetition, full text** | **34** | 77 | 91 | **8** | 36 | 52 |
| Compulsive repetition in loops | 27 / 242 | 46 / 232 | 59 / 220 | 18 / 284 | 19 / 254 | 23 / 250 |

#### CoT Raises Accuracy but Keeps the Dose Ordering

CoT 条件下仍然满足：`α=−4 > 0 > +4`

三个剂量的 accuracy 分别提高：

- `α=−4`：73.0% → 85.0%
- `α=0`：60.0% → 69.0%
- `α=+4`：55.3% → 59.7%

因此，CoT 提高了整体表现，但没有改变当前三个剂量点的排序。
这些是描述性结果。不能仅凭三点比较证明 CoT 与 steering 机制独立、严格可加或存在正式交互。

#### CoT Adds Stepwise Structure

至少两个 `Step` marker 的样本数明显增加：

- `α=−4`：73 → 261
- `α=0`：25 → 220
- `α=+4`：31 → 227

Generation length 和 equation count 则基本稳定。因此，CoT 最清楚的输出变化是增加显式分步结构，而不是简单让模型写得更长或使用更多等式。

#### CoT Reduces Semantic Repetition

Full-text compulsive repetition 在三个剂量下都减少：

- `α=−4`：34 → 8，减少 26
- `α=0`：77 → 36，减少 41
- `α=+4`：91 → 52，减少 39

总 loop 数并没有减少，CoT 条件下甚至更高。这说明 CoT 没有解决所有机械性重复，但明显减少了带有自我怀疑、格式纠结和反复确认的语义性固著。

First–last gap 也从 No-CoT 的 2.7–4.7 pp 缩小到 CoT 的 0.3–0.7 pp，说明 CoT 条件下，后续文本较少破坏首个答案。

#### CoT Does Not Uniformly Suppress Premature Output

CoT 对 premature output 的影响取决于 α：

- `α=−4`：195 → 63
- `α=0`：206 → 151
- `α=+4`：232 → 242

因此，CoT 在 `−4/0` 下减少了过早输出，但在 `+4` 下没有产生同样作用。正向 steering 较强时，即使加入 step-by-step 提示，模型仍常在推理前输出答案。

`α=−4 + CoT` 同时伴随更多 Step 结构、更少 premature output、更少语义性反复和更高 accuracy。这些变化彼此一致，但当前数据不能确定它们各自的因果贡献，也不能称为两个“正交杠杆”。

CoT 下较低的 `####` commit rate 主要反映答案格式出口发生变化，不应直接解释为模型更不愿意提交。

### 2.4 Persona Shapes the Content of Repetition

Persona 分析关注的不是 loop 是否存在，而是重复文本中出现什么内容。Identity sample 表示生成中出现身份自述；heavy 表示同类身份表达被多次重复。

| Wording | Role | Identity samples | Heavy | Literal denial | Soft self-deny |
|---|---|---:|---:|---:|---:|
| Plain | neutral | 2 | 2 | 1 | 0 |
| Plain | expert | 3 | 2 | 0 | 0 |
| Plain | non_expert | **7** | **5** | **4** | 1 |
| Plain | teacher | 2 | 1 | 1 | 0 |
| Pushy | neutral | **0** | **0** | **0** | — |
| Pushy | expert | 4 | 2 | 0 | — |
| Pushy | non_expert | **16** | **10** | **10** | — |
| Pushy | teacher | 3 | 0 | 0 | — |

Plain 条件下，identity monologue 整体很少，每个 role 在 300 题中只有 2–7 个样本。

最清楚的差异来自 `non_expert`：

- Identity samples：7 → 16
- Heavy repetition：5 → 10
- Literal denial：4 → 10

也就是说，pushy wording 最明显地放大了 `non_expert` 的身份否定和自我矮化。

`expert` 的身份文本主要是自我确认，在 plain 与 pushy 条件下都没有 literal denial。Teacher 的 identity sample 很少，也没有形成稳定的教学口吻模式。

同一个 expert persona 在更难的 MATH 上表现不同：GSM8K expert 的 soft self-deny 为 0，而 MATH 中 15 个 identity samples 有 13 个出现 soft self-deny。这个跨任务差异说明 persona 输出会受到任务条件调节，不是固定的身份属性。

Identity monologue 只是生成文本中的角色一致性现象，不能证明模型具有真实身份、自我认知或主观心理状态。

### 2.5 Self-Reported Willingness and Confidence: A Negative Readout

除了答案生成，还测试了两种 0–9 自评方式：

- Logit mode：直接比较十个分数 token。
- Generation mode：要求模型生成 willingness 或 confidence 分数。

这套实验不生成数学答案，因此与前面的 accuracy 和 commitment 指标属于不同的 prompt family。

| Readout | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logit willingness** | 0.00 | 0.37 | 6.45 | 5.05 | 5.53 | 7.51 | **7.98** | 3.59 | 2.50 |
| **Logit confidence** | 0.00 | 0.07 | 3.57 | 5.28 | 5.27 | 6.61 | **8.10** | 2.01 | 1.22 |
| Confidence entropy | 0.00 | 0.09 | 1.39 | 1.46 | 1.51 | 1.53 | 1.20 | 1.29 | 0.55 |
| **Generated willingness** | 7.95 | 7.97 | 7.98 | 8.33 | 8.57 | **8.74** | 8.69 | 8.15 | 4.45 |
| Willingness invalid rate | 2.7% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 22.3% |
| **Generated confidence** | 6.20 | 7.17 | 5.50 | 7.77 | 8.25 | 8.38 | **8.88** | 8.07 | 2.35 |
| Confidence invalid rate | 1.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | **93.3%** |

这两个自评接口没有给出稳定、可解释的 wanting 曲线。

首先，中间剂量并不一致：

- Logit confidence 从 `−4` 到 `+4` 整体上升。
- Logit willingness 在 `−4` 为 6.45，高于 baseline 5.53，方向并不单调。
- Generated willingness 在 `+2` 达到最高值，而 generated confidence 在 `+4` 达到最高值。

其次，极端剂量出现明显的 readout degeneration：

- `α=−8` 的 logit willingness 和 confidence 都锁定在 0，confidence entropy 接近 0。
- `α=+8` 的 generation-mode invalid rate 升至 22.3% 和 93.3%。

因此，极端剂量下的低分不能解释为模型真实地“缺乏意愿”或“失去信心”，因为量表本身已经发生锁定或格式崩溃。

这组结果应作为一个失败的 manipulation check 保留：

> 口头 willingness 和 confidence 没有提供稳定的 wanting readout，不能用于支持主要机制结论。

其中 confidence 更接近显式元认知判断，而 Berridge 意义上的 wanting 是非意识的 incentive salience，两者本来就不应被视为同一构念。

### 2.6 Summary

GSM8K 的输出行为可以概括为：

1. **最佳区域不仅准确率较高，提交质量也更好。** `α=−6/−4` 具有较高 committed accuracy 和较少语义性反复。
2. **正向 α 伴随更早输出和更多答后反复。** 这些变化与准确率下降同时出现，但不能据此认定抢答或反复是准确率变化的因果中介。
3. **`α=−8` 是不同的边界失效。** 它主要表现为开头正式提交、候选值振荡和极低的提交质量，而不是正向端典型的“答完后放不下”。
4. **CoT 增加分步结构并减少语义性反复。** 但它没有消除 α 的方向差异，在 `α=+4` 下也没有抑制过早输出。
5. **Persona 主要改变重复内容。** `non_expert` 更容易产生身份否定，pushy wording 会进一步放大这一模式。
6. **口头自评不是可靠的 wanting 指标。** Willingness 和 confidence 曲线不一致，极端剂量还出现量表锁定和格式失效。
7. **Wanting 是功能类比，不是直接测量。** 更稳妥的表述是：α 改变了模型的 engagement、commitment 和 stopping behavior，这些现象与 incentive-salience gain 的计算类比相容，但不等于生物多巴胺或主观欲望。

### 3.1 Main Performance

#### Accuracy

| α | No-CoT | CoT | ΔCoT |
|---:|---:|---:|---:|
| **−6** | **43.3%** | **49.0%** | +5.7 pp |
| −4 | 40.0% | 45.0% | +5.0 pp |
| 0 | 36.7% | 42.0% | +5.3 pp |
| +4 | 33.0% | 38.7% | +5.7 pp |

两条曲线具有完全相同的排序：

- No-CoT：`43.3 > 40.0 > 36.7 > 33.0`
- CoT：`49.0 > 45.0 > 42.0 > 38.7`

`α=−6 + CoT` 的 49.0% 是当前完整 4×2 矩阵中的最高准确率。

#### Dose Effects Within No-CoT and CoT

两个条件分别构成一个 exploratory dose family，各自使用 Holm `m=3` 校正。

| α vs 0 | No-CoT Δ | No-CoT p_adj | CoT Δ | CoT p_adj |
|---:|---:|---:|---:|---:|
| **−6** | +6.67 pp | .0734 | **+7.00 pp** | **.0225** |
| −4 | +3.33 pp | .3697 | +3.00 pp | .5057 |
| +4 | −3.67 pp | .3697 | −3.33 pp | .5057 |

从 GSM8K 携带的固定工作点 `α=−6` 在两个条件下均带来约 7 pp 的正向差异。其中：

- CoT 条件下，`−6 vs 0` 通过 Holm `m=3` 校正。
- No-CoT 条件下，raw `p=.0245`，bootstrap 95% CI 为 `[+1.00, +12.33]`，但 Holm 校正后 `p_adj=.0734`，因此只能描述为方向明确、校正后未显著。

`α=−6` 不是根据 MATH 结果重新挑选的剂量。它由 GSM8K 预先确定，补跑后恰好也是当前四点 MATH 曲线中的 observed best。

#### CoT Gain at Each Dose

| α | ΔCoT | 0→1 / 1→0 | Raw p | Holm p_adj | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| −6 | +5.67 pp | 40 / 23 | .0430 | .1718 | [+0.67, +10.67] |
| −4 | +5.00 pp | 35 / 20 | .0581 | .1718 | [+0.00, +9.67] |
| 0 | +5.33 pp | 46 / 30 | .0846 | .1718 | [−0.33, +11.00] |
| +4 | +5.67 pp | 40 / 23 | .0430 | .1718 | [+0.67, +10.67] |

CoT 在四个剂量下都提高约 5–6 pp，但 Holm `m=4` 校正后均未达到显著。因此可以说 CoT 的点估计方向一致，不能说每个剂量下的提升都得到了经校正的统计确认。

#### CoT × Steering Interaction

交互量定义为：

`[Acc(CoT, α) − Acc(CoT, 0)] − [Acc(No-CoT, α) − Acc(No-CoT, 0)]`

| α | Interaction | Bootstrap 95% CI |
|---:|---:|---:|
| −6 | +0.33 pp | [−6.67, +7.67] |
| −4 | −0.33 pp | [−6.67, +6.00] |
| +4 | +0.33 pp | [−6.67, +7.33] |

三个点估计都接近 0，两条 accuracy 曲线在描述上近似平行。但三个 CI 均跨 0，且约覆盖 ±7 pp，因此当前结果只能说明：

> 未检出 CoT 明显改变 steering 效应的证据。

这不是等价性证明，也不能证明 CoT 与 steering 机制独立、严格可加或作用于不同内部过程。该交互分析属于 descriptive / exploratory analysis。

### 3.2 Output Behavior

| Metric | −6 No-CoT | −4 No-CoT | 0 No-CoT | +4 No-CoT | −6 CoT | −4 CoT | 0 CoT | +4 CoT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Accuracy** | **43.3%** | 40.0% | 36.7% | 33.0% | **49.0%** | 45.0% | 42.0% | 38.7% |
| **Committed accuracy** | **46.4%** | 45.8% | 42.2% | 37.9% | **53.1%** | 48.7% | 45.3% | 44.1% |
| **Commit rate** | **92.7%** | 87.3% | 86.0% | 85.3% | 92.3% | 92.3% | 92.7% | 87.0% |
| Median boxed position | 31% | 21% | 14% | 16% | 34% | 33% | 23% | 28% |
| Mean boxed position | 50% | 40% | 26% | 33% | 54% | 52% | 36% | 46% |
| Premature, leading boxed | 14 | 7 | 17 | 8 | 3 | 3 | 19 | 11 |
| Premature, either rule | 16 | 13 | 29 | 26 | 3 | 6 | 25 | 26 |
| **Median generation length** | **3,804** | 4,798 | 5,557 | **5,719** | 3,820 | 3,864 | 5,141 | 4,123 |
| Loop samples | 73 | 76 | 120 | 99 | 73 | 79 | 130 | 102 |
| At least two `Step` markers | 239 | 168 | 131 | 177 | **292** | 273 | 217 | 185 |
| Stuck loops | 12 | 10 | 18 | 4 | 12 | 11 | 21 | 15 |
| Median equation count | 7 | 10 | 10 | 10 | 8 | 7 | 8 | 8 |
| **Compulsive repetition, full text** | 42 | **40** | 68 | **82** | **12** | 26 | 31 | 49 |
| Compulsive repetition in loops | 16 / 73 | 15 / 76 | 23 / 120 | 36 / 99 | 8 / 73 | 18 / 79 | 31 / 130 | 23 / 102 |

这里最稳定的行为变化有三项。

第一，No-CoT 下随着 α 从 `−6` 增加到 `+4`，四个剂量上有三项严格单调：

- committed accuracy 从 46.4% 降至 37.9%；
- median generation length 从 3,804 增至 5,719 characters；
- commit rate 从 92.7% 降至 85.3%。

也就是说，正向 α 并没有让模型更快完成答案，而是伴随更长的输出和更少完整收口，同时提交质量下降。补上 `α=−6` 后 commit rate 才显出方向：只看 `−4/0/+4` 时它是 87.3/86.0/85.3，接近持平。

第二，CoT 提高了分步结构。至少两个 `Step` marker 的样本数由 No-CoT 的 131–239 增加到 CoT 的 185–292。CoT 条件下 generation length 在 `α=0` 和 `α=+4` 上明显更短（5,557→5,141、5,719→4,123），但在低 α 端几乎不变（`−6`：3,804→3,820；`−4`：4,798→3,864）——低 α 的输出本来就短，压缩空间有限。

第三，CoT 在每个共有剂量上都减少了 full-text compulsive repetition：

- `α=−6`：42 → 12
- `α=−4`：40 → 26
- `α=0`：68 → 31
- `α=+4`：82 → 49

`α=−6 + CoT` 的 12/300 是全表最低。因此，CoT 的主要行为作用更像是增加推理结构并减少无效反复，而不是改变 α 的整体方向。

两项指标不作为主要证据。**Premature output** 在 MATH 上数量很少（No-CoT 仅 13–29 / 300，GSM8K 是 195–232），且并不单调：`−6/−4/0/+4` 为 16/13/29/26，`−6` 略高于 `−4`；在这个量级上 3 例差异不承载方向结论。**Boxed position** 只反映输出书写位置：`\boxed{}` 按 LaTeX 惯例本就靠近文末，§4.3 已将 MATH 的 boxed position 定为阴性对照（九档 α 只有 1.0× 动态范围），真正的承诺读数是 early-candidate rate。两者都不能用来判断答案在内部何时形成。

#### Performance by Difficulty

| Level | n | α=−6 | α=−4 | α=0 | α=+4 |
|---|---:|---:|---:|---:|---:|
| L1 | 21 | 18 (86%) | 16 (76%) | 15 (71%) | 15 (71%) |
| L2 | 55 | 36 (65%) | 40 (73%) | 35 (64%) | 32 (58%) |
| L3 | 60 | 35 (58%) | 27 (45%) | 24 (40%) | 25 (42%) |
| L4 | 75 | 23 (31%) | 22 (29%) | 21 (28%) | 16 (21%) |
| L5 | 89 | 18 (20%) | 15 (17%) | 15 (17%) | 11 (12%) |
| **All** | **300** | **43.3%** | **40.0%** | **36.7%** | **33.0%** |

五个难度层都满足 `α=−6 ≥ α=0 ≥ α=+4`，说明总体方向并非由单一难度层造成。`α=−6` 在 L1/L3/L4/L5 上是该层最高，仅 L2 例外（65% vs `−4` 的 73%）。

较大的差异出现在 L2 和 L3：

- L3：`α=−6` 相比 `α=0` 提高 18 pp，是所有层级中最大的增益。
- L2：`α=−6` 反而略低于 `α=−4`（65% vs 73%）。

分层后每层仅 21–89 题，单元格计数低至 11–40。这些层级差异只用于说明总体方向不是由某一层驱动，**不宜逐层作统计推断**；`α=−6` 在 L2 的回落尤其应视为小样本波动，而非剂量效应在该难度上反转。

### 3.3 Repetition Content and Persona Effects

#### Compulsive-Repetition Subtypes

下表使用 No-CoT 的完整文本口径，分母固定为 300。“Any”是四类模式的去重并集，因此不等于各子类之和。

| Subtype | α=−6 | α=−4 | α=0 | α=+4 |
|---|---:|---:|---:|---:|
| **Any compulsive repetition** | 42 | **40** | 68 | **82** |
| Self-doubt | 24 | 23 | 43 | 60 |
| Format fixation | 5 | 4 | 9 | 7 |
| Persona reassurance | 17 | 16 | 30 | 26 |
| Over-precision | 1 | 2 | 1 | 2 |

最明显的变化来自 self-doubt：它从 `24 → 23 → 43 → 60` 随 α 增加（`−6` 与 `−4` 实质持平）。常见模式是模型已经得到一个答案，却继续复查、推翻或重算，最终可能将原本正确的结果改错。

Format fixation 和 over-precision 的数量较少，没有呈现同样清楚的剂量趋势。Persona reassurance 在 `α=0` 达到最高，也不是严格单调。**`α=−6` 与 `α=−4` 在所有子类上几乎相同**（42 vs 40 的并集差异来自 self-doubt 与 persona reassurance 各 1 例），说明负向端已经接近这些行为的下限，继续降 α 不再进一步压低反复。因此，不能把所有重复子类都解释为同一种 α 效应。

这里的 “compulsive repetition” 指可观测的输出固著或反复，不是临床焦虑诊断。

#### Persona-Conditioned Output

以下比较均为 `α=0`、No-CoT，每个条件 300 题。

| Role | First acc | Last acc | Identity samples | Heavy identity loop | Literal denial | Soft self-deny |
|---|---:|---:|---:|---:|---:|---:|
| neutral | **36.7%** | 36.0% | **0** | **0** | **0** | **0** |
| an expert | 30.7% | 18.0% | 15 | 12 | 6 | **13** |
| a non expert | 31.3% | 16.7% | **16** | 9 | **13** | 12 |
| a mathematician | 27.0% | 18.7% | 11 | 8 | **0** | **1** |

Neutral 条件几乎不产生身份独白。加入 persona 后，模型更容易在答案后反复确认或否定自己的身份，而且 last accuracy 明显低于 first accuracy。

三个 persona 的重复内容不同：

- `a mathematician` 主要表现为自我确认，soft self-deny 只有 1。
- `an expert` 经常先宣称自己是专家，随后又表示“不确定”或“只是学生”，15 个 identity samples 中有 13 个出现 soft self-deny。
- `a non expert` 更常直接否定自己的数学能力，literal denial 为 13。

同一个 `an expert` persona 在 GSM8K 上主要表现为自我标榜，而在更难的 MATH 上更常伴随自我怀疑。这说明 persona 的输出效果会受到任务难度影响，不是固定不变的角色属性。

这些结果支持“persona 改变重复内容和答案修订行为”，但不能据此证明模型具有真实身份感、主观焦虑或生物学意义上的 arousal。

#### Representative Cases

| Case | Lower dose (`α=−4`) | Higher dose (`α=+4`) | Main contrast |
|---|---|---|---|
| Q101, gold=12 | 使用 Heron 公式得到 12，随后结束 | 已得到 12，却继续质疑并改用错误方法，最终答错 | 正确后继续检查并推翻自己 |
| Q116, gold=40 | 得到 `\boxed{40}` 后结束 | 继续做近似和单位换算，最终改成 57.1 | 过度求解导致答案损坏 |
| Q9, gold=6−5i | 得到正确答案并提交 | 未完成 boxed 提交，尾部反复请求确认 | 求确认取代答案收口 |
| Q105, gold=.0000672 | 提交一次后结束 | 反复声称格式不正确，并多次重复相同数值 | 格式固著导致输出膨胀 |

这些案例用于说明聚合指标对应的文本模式，不构成独立统计证据。完整生成文本和判定细节见 `CLAUDE.md`。

### 3.4 Summary

MATH 上的主要结果可以概括为：

1. **负向 steering 表现更好。** No-CoT 与 CoT 都呈现 `−6 > −4 > 0 > +4`，而从 GSM8K 携带的 `α=−6` 是当前 observed best。
2. **CoT 提高整体表现，但没有改变剂量排序。** 四个剂量的提升均约为 5–6 pp，不过逐剂量比较经 Holm 校正后均未显著。
3. **正向 α 伴随更差的提交质量和更多无效反复。** 这一关系在 No-CoT 的 generation length、committed accuracy 和 commit rate 上严格单调（四个剂量），在 full-text compulsive repetition 上则是 `α≥0` 段清楚、负向端已接近下限（`−6` 42 与 `−4` 40 实质持平）。
4. **CoT 主要增加推理结构并减少强迫性反复。** 当前数据未检出 CoT 明显改变 steering 效应，但宽 CI 不支持机制独立或严格可加的结论。
5. **输出行为不等于内部机制。** boxed position、commit rate、first–last gap 和重复文本都是行为读数，不能单独证明答案形成时间、因果中介或生物学 dopamine 机制。MATH 的 boxed position 尤其是阴性对照（§4.3），承诺时序须以 early-candidate rate 为准。

## 4. Qwen2.5-7B-Instruct Cross-Model Analysis

本节使用与 Llama 相同的任务和主要评价口径，分析 Qwen2.5-7B-Instruct 在 GSM8K 与 MATH 上的准确率和答案形成行为。运行配置、模型专属层带、输出抽取和产物校验记录于 `CLAUDE.md`。

`first_acc` 是主要准确率指标，`last_acc` 用于观察后续修改答案的影响。由于两种模型使用不同的 mask、层带和激活尺度，raw α 只能在同一模型内比较，不能视为跨模型等效剂量。

### 4.1 Performance across Tasks and Prompt Conditions

#### GSM8K

**Table 4.1. Qwen GSM8K dose-response with and without CoT (n=300 per cell)**

| α | No-CoT first_acc | No-CoT last_acc | No-CoT Holm p_adj | commit% | early-candidate% | posN | CoT first_acc | CoT last_acc | CoT Holm p_adj | CoT early-candidate% | CoT posN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 60.33 | 71.00 | .124 | 84.0 | 95.7 | .003 | 80.33 | 79.00 | 1.000 | 99.0 | .003 |
| −6 | 64.00 | 68.67 | .973 | 83.7 | 95.7 | .003 | 77.67 | 79.33 | 1.000 | 98.7 | .003 |
| −4 | 68.67 | 75.33 | .973 | 79.7 | 97.3 | .003 | 79.00 | 79.33 | 1.000 | 99.7 | .003 |
| −2 | 71.00 | 79.00 | .973 | 79.7 | 96.3 | .003 | 79.00 | 81.00 | 1.000 | 99.3 | .003 |
| **0** | **68.00** | **73.33** | — | 81.0 | 96.3 | .003 | **76.33** | **76.67** | — | 97.3 | .003 |
| +2 | 70.33 | 75.00 | .973 | 81.0 | 95.3 | .003 | 74.33 | 75.00 | 1.000 | 96.3 | .003 |
| +4 | 71.67 | 78.67 | .973 | 78.7 | 92.7 | .003 | 78.33 | 81.00 | 1.000 | 91.7 | .003 |
| **+6** | **78.00** | 76.33 | **.016** | 92.0 | **45.3** | **.242** | **88.33** | **89.00** | **.0002** | **34.3** | **.809** |
| **+8** | **86.00** | 80.33 | **<1e−4** | 97.0 | **5.0** | **.754** | **86.00** | 84.00 | **.0030** | **7.0** | **.825** |
| *+10* | *88.33* | *84.33* | *<1e−4* | *98.3* | *2.7* | *.794* | — | — | — | — | — |
| *+12* | *88.67* | *86.33* | *<1e−4* | *98.3* | *3.7* | *.802* | — | — | — | — | — |

`commit%` 表示生成中出现可解析 `#### <number>` 的比例；`posN` 是首次可解析提交在全文中的归一化位置。`+10/+12` 是观察九档结果后追加的探索性剂量，不属于原九档 Holm 检验族。

No-CoT 准确率在 `−8…+4` 内变化较小，主要提升出现在 `+6/+8`。继续增加到 `+10/+12` 后，`first_acc` 为 88.33% 和 88.67%，与 `+8` 没有明显分开。因此，GSM8K 在当前剂量范围内表现为上升后饱和，没有观察到高剂量右臂。

CoT 将 α=0 的准确率由 68.00%提高到76.33%，但在 α≤+4 时仍有91.7%–99.7%的样本很早出现答案候选。直到 `+6`，early-candidate 和 `posN` 才同时发生明显变化。CoT 曲线的最高点是 `+6`，但 `+6` 与 `+8` 未显著分开（探索性 p=.371），因此不能确认 CoT 已形成稳定右臂。

#### MATH

**Table 4.2. Qwen MATH dose-response with and without CoT (n=300 per cell)**

| α | No-CoT first_acc | No-CoT last_acc | Holm p_adj | early-candidate% | Pre-commit chars | `\boxed{}` posN | CoT first_acc | CoT last_acc | CoT early-candidate% | CoT pre-commit chars | CoT `\boxed{}` posN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 54.00 | 50.00 | .094 | 85.0 | 1414 | .977 | 59.67 | 57.67 | 86.7 | 1345 | .965 |
| −6 | 57.00 | 54.00 | 1.000 | 79.3 | 1395 | .978 | 59.33 | 57.33 | 80.0 | 1372 | .966 |
| −4 | 58.00 | 55.00 | 1.000 | 74.3 | 1362 | .978 | 65.00 | 61.67 | 76.7 | 1330 | .964 |
| −2 | 59.00 | 55.33 | 1.000 | 71.3 | 1376 | .977 | 63.00 | 58.67 | 74.0 | 1305 | .966 |
| **0** | **60.67** | **60.00** | — | 67.3 | 1356 | .977 | **63.00** | **57.67** | 71.3 | 1344 | .969 |
| +2 | 60.00 | 58.67 | 1.000 | 59.7 | 1319 | .977 | 62.00 | 58.33 | 69.0 | 1317 | .970 |
| +4 | 63.33 | 61.33 | 1.000 | 46.3 | 1270 | .975 | 63.00 | 59.00 | 58.3 | 1315 | .973 |
| **+6** | **68.33** | **67.67** | **.0087** | **19.3** | **1029** | .969 | **66.00** | **65.67** | **29.7** | **1116** | .968 |
| +8 | 63.33 | 63.67 | 1.000 | **10.3** | **828** | .960 | 64.00 | 64.33 | **10.3** | **1002** | .964 |

No-CoT 的准确率从 `−8` 到 `+6` 总体上升，并在 `+8` 回落5个百分点。只有 `+6` 相对 α=0 的差异通过 Holm 校正；`+6` 与 `+8` 的探索性配对比较为 p=.040。因此，MATH 出现了描述性的右臂，但证据主要来自 `+6` 这一工作点。

CoT 提高了部分负向和低剂量条件的准确率，但没有稳定改变峰值位置。CoT 各剂量相对自身 α=0 的八项比较均为 Holm p_adj=1.000；CoT 与 No-CoT 的九项逐剂量比较也没有结果通过校正，最大差异为 `−4` 的 +7.00pp（raw p=.0065，Holm p_adj=.0581）。

这说明 CoT 压缩了当前剂量范围内可观察到的提升空间：No-CoT 从60.67%升至68.33%，CoT 则从63.00%升至66.00%。这些结果不能解释为模型已经达到能力上限。

### 4.2 Commitment Reordering across GSM8K and MATH

Qwen 在两个任务上都表现出明显的 early-candidate transition，但两个任务需要不同的承诺指标。

**Table 4.3. Main commitment changes at baseline and high-performing doses**

| Condition | Compared doses | early-candidate% | Pre-commit chars | posN or marker position | Accuracy |
|---|---|---:|---:|---:|---:|
| GSM8K No-CoT | 0 → +8 | 96.3 → 5.0 | 3 → 324 | .003 → .754 | 68.00 → 86.00 |
| GSM8K CoT | 0 → +6 | 97.3 → 34.3 | 3 → 517 | .003 → .809 | 76.33 → 88.33 |
| MATH No-CoT | 0 → +6 → +8 | 67.3 → 19.3 → 10.3 | 1356 → 1029 → 828 | `.977 → .969 → .960` | 60.67 → 68.33 → 63.33 |
| MATH CoT | 0 → +6 → +8 | 71.3 → 29.7 → 10.3 | 1344 → 1116 → 1002 | `.969 → .968 → .964` | 63.00 → 66.00 → 64.00 |

在 GSM8K 中，`####` 是正式答案标记，因此 `posN` 可以反映首次提交的位置。α=0 时，模型通常先给答案，再在后文检查或修正；到 `+6/+8`，更多计算被移到第一次正式提交之前。

**Table 4.4. GSM8K effort reallocation from α=0 to α=+8**

| Measure | α=0 median | +8 median | Median Δ | p |
|---|---:|---:|---:|---:|
| Pre-commit characters | 3 | 324 | **+302** | 7.6e−50 |
| Post-commit characters | 1116 | 120 | **−862** | 2.0e−33 |
| Total characters | 1130 | 700 | **−430** | 1.2e−14 |
| Pre-commit equations | 0 | 2 | **+2** | 8.2e−35 |
| Post-commit equations | 1 | 0 | −0 | 3.7e−19 |

总输出长度下降，但提交前计算增加、提交后内容减少。因此，这一变化更适合描述为计算位置和停止行为的重新组织，而不是推理量简单增加。

MATH 的 `\boxed{}` 几乎始终位于文末，`posN` 只在 .960–.978 之间变化，不能有效区分答案何时形成。MATH 更合适的指标是开头是否先出现未加框的答案候选。正向 α 降低了 early-candidate 比例，但没有明显移动文末的 `\boxed{}`。

冻结的 early-candidate detector 在180条盲法审核中达到 precision 1.000、recall .976、一致率 .983。三条假阴性均来自 MATH `+8` 的首行长句，因此高剂量的 early-candidate 比例应视为下界，不能把10.3%直接解释为其余89.7%的样本都完成了“先推理、后形成答案”。

### 4.3 Task-Dependent High-Dose Behavior

GSM8K 和 MATH 在 commitment transition 之后呈现不同的高剂量结果。

**Table 4.5. Task-dependent response after the main transition**

| Condition | Main transition | High-dose pattern | Interpretation |
|---|---|---|---|
| GSM8K No-CoT | Accuracy and commitment ordering change around `+6/+8` | Accuracy remains at 86.00%–88.67% through `+12` | Rise followed by saturation; no observed right arm |
| GSM8K CoT | Transition occurs near `+6` | `+6` and `+8` are not statistically separated | Right arm not established |
| MATH No-CoT | Accuracy peaks at `+6` | Falls from 68.33% to 63.33% at `+8` | Descriptive right arm |
| MATH CoT | Low-dose performance improves | Peaks near `+6`, with weaker dose separation | CoT compresses the observed dose effect |

#### GSM8K high-dose integrity

**Table 4.6. GSM8K No-CoT high-dose integrity checks**

| α | contamination% | empty% | truncated% | clean n | clean acc | `####` count p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17.7 | 0.0 | 1.7 | 247 | 70.85 | 61 |
| +6 | 44.7 | 0.0 | 0.7 | 166 | 78.31 | 29 |
| +8 | 60.3 | 0.0 | 0.7 | 119 | 84.03 | 18 |
| +10 | 58.7 | 0.0 | 1.0 | 124 | 87.90 | 21 |
| +12 | 59.7 | 0.0 | 1.0 | 121 | 84.30 | 7 |

高剂量下没有空生成，截断率始终不超过1.0%，干净子集准确率也维持在84%–88%。因此，GSM8K 的平台不能简单归因于空输出、截断或极端 marker 重复。由于干净子集是根据干预后的输出行为筛选，只用于完整性检查，不用于估计跨剂量因果效应。

#### MATH gain and high-dose decline

**Table 4.7. Post-treatment early-candidate transitions from α=0 to α=+6**

| Transition | n | acc@0 | acc@+6 | Δ |
|---|---:|---:|---:|---:|
| Early candidate removed (`y→n`) | 149 | 51.7 | 66.4 | **+14.8pp** |
| Early candidate retained (`y→y`) | 53 | 54.7 | 56.6 | +1.9pp |
| Absent in both cells (`n→n`) | 93 | 78.5 | 79.6 | +1.1pp |
| Early candidate newly appears (`n→y`) | 5 | 60.0 | 40.0 | −20.0pp |

准确率提升主要集中在 early candidate 被抑制的 `y→n` 样本，而原本就没有 early candidate 的 `n→n` 样本变化很小。不过，这些组别由干预后的输出定义，因此只能说明准确率提升与 ordering change 相关，不能证明后者是因果中介。

MATH 的高剂量回落主要集中在困难题。Level 5 的 No-CoT 准确率为40.4%→47.2%→36.0%，CoT 为38.2%→43.8%→36.0%（α=0/+6/+8）。与此同时，No-CoT 提交前字符由1029降至828，而提交后字符始终约为27–34，截断率接近0%。

因此，`+8` 的回落更符合难题所需的提交前计算被进一步压缩，而不是提交后的修改失控。early-candidate 越少并不一定越好；有效表现需要在延迟提交与保留足够计算之间取得平衡。

MATH 还存在少量极端 `\boxed{}` 重复。No-CoT 下该现象随正向剂量下降，但 CoT 的 marker-count p99 在 α=0/+6/+8 仍为106/121/113。它高度集中于少数样本——CoT `+8` 只有8/300个样本出现至少20次 marker——并且主要影响 `last_acc`，不改变以第一次答案计算的 `first_acc` 主结论。

### 4.4 Cross-Model Summary

**Table 4.8. Behavioral comparison between Llama and Qwen**

| Dimension | Llama3.1-8B | Qwen2.5-7B |
|---|---|---|
| GSM8K curve | Asymmetric peak at moderate negative α | Rises at positive α and saturates through `+12` |
| MATH curve | Best performance near moderate negative α | Peaks near `+6`, then declines at `+8` |
| Main output change | Positive α is associated with earlier commitment and repetition; extreme negative α produces a separate failure mode | Positive α suppresses early candidates and moves computation before the first formal answer |
| CoT | Improves structure and can delay commitment at moderate doses | Improves low-dose accuracy, but ordering changes mainly near `+6` |
| Cross-model interpretation | Model-specific response | Model-specific response |

The same analysis framework identifies commitment-related behavior in both models, but the behavioral dose-response does not replicate point by point. Llama shows an asymmetric peak, whereas Qwen shows a GSM8K high-dose plateau and a task-specific MATH decline.

The two models also respond in opposite raw-α directions. This does not establish that their baselines occupy opposite internal states or that their best doses reach the same working state. Cross-model comparison should therefore focus on the shape of the behavioral transition, not on matching α values.

In simple terms:

1. Qwen’s improvement is associated with moving calculation before the first formal answer, rather than producing more text.
2. This reordering helps until it begins to compress the computation needed for difficult MATH problems.
3. The analysis framework transfers across models, but the optimal direction and dose-response remain model- and task-specific.
4. These results support a computational commitment-gain interpretation, not literal biological dopamine or a universal wanting axis.

## 5. Commitment-Based Prediction and Transfer

本节检验 commitment regime 是否具有实际预测与迁移价值：先在 GSM8K 上预测未见题目的正确性，再将冻结规则迁移到 MATH、GSM-Hard 与 CoT 条件。内部状态机制与 commitment transfer function 见 `AdaptiveThinking.md §5.8`。

### 5.1 Prediction and Evaluation Protocol

P2 使用 GSM8K 已有输出训练基于文本的 commitment predictor，并按题目进行五折交叉验证；同一道题的所有剂量始终位于同一折，以避免数据泄漏。模型输入包括 early-candidate、commit state、标准化 commit position（`posN`）及其可观测性，raw α 不作为特征。

冻结后的 GSM8K predictor 直接应用于 MATH，不使用 MATH accuracy 进行训练、调参或校准。Qwen 使用完整的 9 点剂量曲线，Llama 因**当时**仅有 `−4/0/+4`，只检验局部 steering 方向。主要 accuracy 口径为 `first_acc`。

> **历史边界（不可回填）**：原始 P2B 在看到 `−6` 的 MATH 结果前仅使用 `−4/0/+4`，因此其「Llama local-direction transfer」结论**保持不变**。2026-09-01 后补的 `−6`（No-CoT 与 CoT）属于 **fixed-workpoint transfer**（§5.7），**不得回填为 P2B 当时已经选中 −6**。

具体协议版本、commit-state 编码、marker 适配、fold manifest、抽取器、填充与标准化方法、bootstrap 和 SHA256 provenance 统一记录于 `CLAUDE.md`。

**Table 5.1a — 各小节数据源**

| 小节 | 数据源 |
|---|---|
| 5.2 GSM8K 预测（P2A） | `llama3/dopamine/signal/dopamine_signal_gsm8k_8B_nocot{α}_ema0.95_L11-20.json`，α ∈ −8…+8 共 9 cells（Qwen 为 `L16-22`，另含 `+10/+12`，11 cells）。同目录下的 role cells 不参与。 |
| 5.3 MATH 工作点（P2B） | **P2B 当时使用**：`llama3/math/mdf_{0,4,neg4}/`（3 cells）、`qwen2.5/math/mdf_{0,±2,±4,±6,±8}/`（9 cells）。No-CoT only。**当前完整数据（2026-09-01）**：Llama No-CoT `mdf_{neg6,neg4,0,4}`、Llama CoT `mdf_{neg6,neg4,0,4}_cot`；三个 role cells 仍为 α=0 No-CoT，**不与 neutral dose family 混合**。 |
| 5.5 GSM-Hard 盲测（P3） | `llama3/gsm_hard/mdf_{neg8,neg6,neg4,0,4}/`、`qwen2.5/gsm_hard/mdf_{neg4,0,4,6,8}/`，各 5 cells。No-CoT only。 |
| 5.6 CoT 迁移 | `llama3/gsm_hard/mdf_{0,neg6}_cot/`、`qwen2.5/gsm_hard/mdf_{0,8}_cot/`，共 4 cells。 |

GSM8K 训练侧取自 signal 树而非 `llama3/gsm8k/`：signal JSON 带 `x_prefill`，是 entry-only 与 combined 对照组所必需，`llama3/gsm8k/` 无此字段。

> GSM8K predictor was trained and evaluated on the server-184 signal batch (`bs=1`). The server-182 production dose curves reported earlier are a separate generation batch and are not joined at the item level.
>
> 两批次呈现一致的定性剂量形状，均在 `α=−6` 出现非对称峰，说明 signal-batch 分析与 production-batch 行为结果相容。P2A 的所有配对、fold 与 OOF 评估均严格在 server-184 批次内部完成，未与 §4 的 server-182 数据进行逐题或统计混合。

### 5.2 Held-Out Correctness Prediction on GSM8K

**Table 5.2a — Out-of-Sample Prediction Performance on GSM8K**

| Model | Commitment-only AUROC | Entry-only AUROC | Commitment − Entry | Combined − Commitment |
|---|---|---|---|---|
| Llama | **.687** [.656, .719] | .548 [.526, .571] | **+.139** [+.104, +.172] | −.001 [−.004, +.002] |
| Qwen | **.749** [.710, .787] | .628 [.601, .654] | **+.121** [+.084, +.156] | +.002 [−.002, +.007] |

两模型均通过预注册 gate，且在 GSM8K 内部校准良好（calibration slope：Llama .95，Qwen .98；`fig_p2a_calibration.png`）。

> **结论：commitment timing 能预测未见 GSM8K 题目的对错，而且明显优于 entry gain；在此基础上加入 entry gain，没有带来可检测的额外提升。**

这是预测证据，不是因果证据；“未检出额外提升”也不等于证明 entry gain 无用。完整统计口径与实现细节见 `CLAUDE.md`。

### 5.3 Retrospective Locked Transfer to MATH

**Table 5.3a — Cross-Task Direction and Workpoint Selection on MATH**

| Model | Predicted Direction | Direction Match | Spearman ρ | Selected α | Observed Best α | Regret | Near-Optimal Set |
|---|---|---|---|---|---|---|---|
| Qwen (9 α; full curve) | Positive | **Correct** | **+.962** (n=9) | **+6** | +6 | **.000** | Hit [+4, +6] |
| Llama (3 α; local only) | Negative | **Correct** | +1.000 (n=3) | −4 | −4 | .000 | Hit [−4, 0] |

Qwen 不仅选中真实最佳工作点 `+6`，也识别出 `+8` 的性能回落。其预测分数为 `.83–.88`，实际 accuracy 为 `.54–.68`：绝对数值明显高估，但剂量排序与曲线变化基本一致（`fig_p2b_transfer.png`）。

> **结论：commitment predictor 无法直接估计 MATH 的准确率，但能够判断 steering 方向并选择合适的工作点。**

这是 retrospective locked transfer，并非真正的盲测；完整冻结顺序、校准边界与图表口径见 `CLAUDE.md`。

### 5.4 Evidence Scope and Conclusion

- 这是一次**规则冻结后的回顾性跨任务验证**，不是真正的盲测；该证据缺口随后由 §5.5 的 GSM-Hard 前瞻性盲测补足。
- Qwen 支持完整曲线与工作点判断；**在 P2B 当时**，Llama 只有三个剂量点，仅支持局部方向判断。该限制描述的是 P2B 的证据范围，**不适用于当前完整的 MATH 4×2 数据**（见 §3.1 与 §5.7）。
- 跨任务迁移的主要是答案形成与提交时序，而不是 GSM8K 特有的 loop 行为；预测排序可以迁移，绝对概率校准不能迁移。

> **P2 总结：commitment timing 能预测未见 GSM8K 题目的对错，也能在 MATH 上选择 steering 方向和工作点，但尚不能视为真正的跨任务盲测。**

完整特征分布、抽取器差异、稳健性检查与统计边界见 `CLAUDE.md`。

### 5.5 Prospective Blind Transfer on GSM-Hard

本节在从未查看 accuracy 的 GSM-Hard 上，前瞻性检验由 GSM8K 冻结的 commitment predictor。剂量、predictor、工作点规则与成功标准均在生成前冻结，预测文件也在 gold 解封前固定。

**Table 5.5a — Predicted and Observed Dose Curves**

| Model      | Metric                 | −8    | −6        | −4      | 0     | +4    | +6    | +8        |
| ---------- | ---------------------- | ----- | --------- | ------- | ----- | ----- | ----- | --------- |
| Llama3-8B  | Predicted score        | .5554 | **.68834** | .68828 | .6303 | .5770 | —     | —         |
|            | Observed `first_acc`   | .1100 | **.2433** | .2400   | .1800 | .1700 | —     | —         |
| Qwen2.5-7B | Predicted score        | —     | —         | .6959   | .7038 | .6794 | .7182 | **.8552** |
|            | Observed `first_acc`   | —     | —         | .3433   | .3400 | .3467 | .4033 | **.5033** |

`Predicted score` 是 frozen logistic regression 对每题正确概率的输出在同一剂量内取平均，用于排序而非估计 GSM-Hard 的绝对准确率。

**Table 5.5b — Blind Workpoint Selection**

| Model      | Direction  | Selected | Observed best | Observed near-optimal | Regret  | ρ      |
| ---------- | ---------- | -------- | ------------- | --------------------- | ------- | ------ |
| Llama3-8B  | negative ✓ | −6       | −6            | {−6, −4}              | 0.00 pp | +1.000 |
| Qwen2.5-7B | positive ✓ | +8       | +8            | {+8}                  | 0.00 pp | +0.600 |

两条剂量曲线均可读（paired McNemar tests with Holm correction：Llama minimum adjusted p = 2.29e−07，Qwen = 1.35e−07；n=300）。

**Table 5.5c — Direct Transfer of GSM8K Workpoints**

| Model      | GSM8K workpoint | acc(α) | acc(0) | Δ             | discordant | McNemar p |
| ---------- | --------------- | ------ | ------ | ------------- | ---------- | --------- |
| Llama3-8B  | α = −6          | .2433  | .1800  | **+6.33 pp**  | 32 / 13    | .0066     |
| Qwen2.5-7B | α = +8          | .5033  | .3400  | **+16.33 pp** | 63 / 14    | 1.41e−08  |

这些工作点直接取自冻结的 GSM8K 结果，未使用 GSM-Hard predicted score。

**Table 5.5d — Llama3-8B commitment panel (No-CoT, n=300/cell)**

| α | acc | early% | committed% | unparsed_nonloop% | loop% | no-marker% | posN med | post-commit% | chars med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | .110 | 66.0 | 67.7 | 10.3 | 13.3 | 8.7 | **0.0000** | **100.0** | 2194 |
| **−6** | **.243** | **28.7** | 54.7 | 5.3 | 34.0 | 6.0 | 0.2740 | 72.6 | 2168 |
| −4 | .240 | 28.0 | 53.0 | 4.3 | 34.7 | 8.0 | 0.2351 | 76.5 | 2134 |
| 0 | .180 | 45.7 | 54.3 | 4.0 | 28.0 | 13.7 | 0.2161 | 78.4 | 2072 |
| +4 | .170 | 60.0 | 44.7 | 6.3 | 27.7 | 21.3 | 0.1274 | 87.3 | 2078 |

**Table 5.5e — Qwen2.5-7B commitment panel (No-CoT, n=300/cell)**

| α | acc | early% | committed% | unparsed_nonloop% | loop% | no-marker% | posN med | post-commit% | chars med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −4 | .343 | 94.7 | 79.3 | 18.3 | 2.3 | 0.0 | 0.8062 | 19.4 | 1326 |
| 0 | .340 | 93.7 | 72.7 | 22.7 | 4.7 | 0.0 | 0.7680 | 23.2 | 1271 |
| +4 | .347 | 92.0 | 78.0 | 17.3 | 4.3 | 0.3 | 0.6791 | 32.1 | 1268 |
| +6 | .403 | 58.7 | 86.3 | 11.0 | 2.3 | 0.3 | 0.5969 | 40.3 | 1145 |
| **+8** | **.503** | **6.0** | **98.3** | 1.7 | 0.0 | 0.0 | 0.7765 | 22.3 | **875** |

各列含义如下：

| 指标 | 含义 |
|---|---|
| `α` | RSN steering 的干预剂量。`0` 是不干预，正负值表示沿相反方向 steering。 |
| `acc` | 该剂量下的真实准确率，即 `first_acc`。 |
| `early%` | 生成开头较早出现答案候选的样本比例，用于衡量是否存在过早回答倾向。 |
| `committed%` | 输出中存在可解析最终答案 marker，例如 `#### 42` 的样本比例。 |
| `unparsed_nonloop%` | 输出中出现 `####`，但后面没有可解析数字，并且不属于重复 marker loop 的比例。即“尝试提交，但格式无效”。 |
| `loop%` | `####` 或答案提交片段反复出现，形成退化循环的样本比例。 |
| `no-marker%` | 整段输出完全没有出现 `####` 的样本比例，即没有按照指定格式提交答案。 |
| `posN med` | 首个可解析 `#### <数字>` 在整段生成文本中的归一化位置中位数。`0` 表示几乎在开头提交，`1` 表示接近结尾才提交。只在存在可解析 commitment 的样本上定义。 |
| `post-commit%` | 第一次提交答案之后的文本占整段生成文本的比例。越高表示模型越早提交，随后仍继续生成大量文字。需与其有效样本分母一起解释。 |
| `chars med` | 每个样本生成文本字符数的中位数，用于描述输出长度。 |

```text
committed
+ unparsed_nonloop
+ loop
+ no-marker
= 100%
```

例如 Llama `α=−8`：

- `early%=66%`：很多样本很早出现答案候选；
- `posN med=0`：可解析答案通常在生成开头就出现；
- `post-commit%=100%`：几乎整段文字都生成在首次提交之后。

这表示“过早锁定后继续生成”，而不是“推理长度不足”。完整分母与抽取规则见 `CLAUDE.md`。

**Observed Commitment Patterns.**

以下分析均为 **exploratory**：十个 No-CoT cell 的 accuracy 已在该分析开始前解封，因此只能作为与 P1 一致的行为证据，不能构成 mediation。

- **高准确率对应较低的 early-candidate rate。** Llama 的近优区间 `{−6,−4}` 位于 early-candidate 低谷（28.7% / 28.0%）；Qwen `+8` 则从 `+6` 的 58.7% 骤降至 6.0%。这一方向与 GSM8K 上的结果一致。
- **Llama `−8` 表现为极端的过早锁定。** 在可解析 commitment 样本中，`posN med=0`、`post-commit=100%`，同时 early-candidate 升至 66.0%，accuracy 降至 .110。其生成长度并未明显缩短，说明性能崩溃不是因为“少生成”，而是因为答案形成得过早。
- **Qwen `+8` 的输出格式最稳定。** Parseable commitment 达到 98.3%，`loop=0%`、`unparsed_nonloop=1.7%`，生成长度也缩短至 875 字符，对应最高 accuracy `.503`。
- **两个模型具有不同的失败模式。** Llama 主要表现为 loop 与 no-marker failure；Qwen 几乎不 loop，no-marker 也接近于零（0–0.3%），其失败更多表现为写出 `####` 但没有可解析数字。这一模型差异同样延续了 GSM8K 上的观察。

**Qwen Commit-Position Rebound.**

Qwen 的 `posN med` 从 `+6` 的 0.597 回升至 `+8` 的 0.777。该回升在五个剂量都 committed 的共同子集上仍然存在（n=153；0.566→0.758），因此不是 committed coverage 改变造成的分母假象。

在该共同子集中，绝对 `commit_char` 从 274 延后至436，而总生成长度从 959 缩短至831。因此，`+8` 并不是简单地让正式 marker 更早出现；它主要减少了过早答案候选和不可解析提交，并使最终提交更加完整。可迁移规律是由 candidate timing、commit validity、loop/no-marker 和生成长度共同构成的 commitment regime，而不是单一 `posN` 的单调变化。

Llama 的五剂量共同 committed 子集仅 n=35，且由 `−8` 的极端行为强烈筛选，因此不作对应的共同子集推断。

**Generation-Budget Truncation (Llama).**

`cap%` = 重新分词后达到生成上限的样本比例（`>=767` of `max_new_tokens=768`；阈值不取精确等于，因离线重新分词在边界处有误差——Llama α=0 精确判定读 82.0%，`>=767` 读 94.0%，`>=760` 读 94.3%，760 以上的平台才是真实截断群体）。

| | −8 | −6 | −4 | 0 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Llama cap%** | 91.3 | 92.3 | 96.0 | 94.0 | 95.3 | — | — |
| **Qwen cap%** | — | — | 20.3 | 21.0 | 22.7 | 19.3 | 13.3 |

**Llama 五格全部 91–96% 触及上限，decode 长度中位数恒为 768**；Qwen 仅 13–23%。因此上文「Llama `−8` 生成长度并未明显缩短」应理解为**天花板效应而非自然观测**——各剂量长度都被同一上限压平。这不削弱该处结论（性能崩溃不是因为少生成），但其依据应改为 `posN med=0` 与 `post-commit=100%` 这两项与长度无关的量：答案出现在生成开头，其后整段皆在提交之后。

**截断不否定「固定 768-token budget 下」的配对比较**——全部十格在同一预算、同一约定下逐题配对，estimand 是一致的。**但它会影响绝对准确率，也可能影响剂量差异**（Llama 各格 cap 率散布 4.7 pp；固定工作点检验所用的 `−6 vs 0` 一对差异不显著，18/23，exact McNemar p=.53，但这只覆盖该一对），**因此结果不能外推为不受生成预算限制的能力表现**。

CoT 生成更长而上限不变，故 CoT 结果若变差，截断是必须与 commitment 解读并列报告的替代解释，而非可径直归因于其一。**现阶段不提高上限**：那会改变已冻结的主问题。先完成 768-token 条件；若 Llama 结果为 null/negative，再单独建立 larger-budget sensitivity，且不得事后替换主结果。

**Conclusion.**

> 在前瞻性封存的 GSM-Hard 上，frozen commitment predictor 正确判断了两个模型的 steering 方向，并以 zero empirical regret 选中 observed near-optimal workpoint。直接迁移 GSM8K 工作点使 Llama 与 Qwen 的 accuracy 分别提高 6.33 pp 和 16.33 pp，但 absolute probability calibration 未能迁移。

结果表明 commitment timing 携带可迁移的工作点位置信息，但目前证据仅覆盖 GSM8K→GSM-Hard 的 near-domain transfer。

**Interpretation Boundaries.**

- **Tables 5.5a–c 不是独立证据。** 它们使用同一批 per-question correctness；固定工作点表检验的是既定 α 能否直接迁移，而不是提供独立重复。Tables 5.5d–e 使用相同输出进行 exploratory commitment analysis。
- **Llama 未实质区分 `−6` 与 `−4`。** 两者 predicted score 仅差约 0.00006，且都属于 observed near-optimal set。Qwen 的选择更明确，`+8` 比次优 predicted dose 高 0.137。
- **Qwen 并非完美排序。** `ρ=+0.600`；predictor 将 `−4` 排在 `0` 之上，但两者 observed accuracy 仅相差一道题。
- **绝对校准没有迁移。** Predicted score 系统性高于 observed accuracy（`.55–.86` vs `.11–.50`）；迁移的是 dose ordering，而非 absolute probability calibration。
- Commitment panel 与既有机制一致，但属于解封后的 exploratory analysis，不构成 causal mediation。
- 当前证据限于 GSM8K→GSM-Hard 的 near-domain transfer，且仍需生成多个剂量，尚未实现仅从 `α=0` 选择 steering 方向。

Protocol provenance、artifact hashes、evaluator validation 与完整统计见 `CLAUDE.md`。

### 5.6 CoT Condition Transfer

本节检验 GSM8K 确立的固定工作点，在加入 CoT prompt 后能否继续改善 GSM-Hard 表现。Llama 使用 `α=−6`，Qwen 使用 `α=+8`，均未重新搜索剂量。

这是预先锁定的 **condition transfer test**，不是新的 blind dataset validation：题目和 gold 与 §5.5 相同，但 CoT 条件、预测方向和评价规则在生成前已经冻结。

**Table 5.6a — CoT Condition Transfer (n=300/cell, 768-token budget)**

| Model | Condition | α | Acc(α=0) | Acc(α) | ΔAcc | Discordant | McNemar p | Holm p | 95% CI |
|---|---|---:|---:|---:|---:|:---:|---:|---:|:---:|
| **Llama3-8B** | **CoT** | −6 | .2000 | .2600 | **+6.00 pp** | 27/9 | .00393 | **.00393** | [+2.33, +10.00] |
| Llama3-8B | *No-CoT* | −6 | .1800 | .2433 | *+6.33 pp* | *32/13* | *.00661* | — | — |
| **Qwen2.5-7B** | **CoT** | +8 | .3800 | .5133 | **+13.33 pp** | 58/18 | 4.71e−06 | **9.42e−06** | [+8.00, +19.00] |
| Qwen2.5-7B | *No-CoT* | +8 | .3400 | .5033 | *+16.33 pp* | *63/14* | *1.41e−08* | — | — |

两个模型都符合预先锁定的正向增益预测，并通过 Holm 校正。说明固定工作点在 CoT 条件下仍然有效：steering 并非只是重复 “Let’s think step by step” 的提示作用。

No-CoT 结果来自 §5.5，仅用于比较效应大小，不属于本节的 Holm family，也不是第二次独立验证。

**Table 5.6b — Descriptive CoT × Steering Interaction**

| Model | ΔAcc(CoT) | ΔAcc(No-CoT) | ΔInteraction | 95% CI |
|---|---:|---:|---:|:---:|
| Llama3-8B | +6.00 pp | +6.33 pp | −0.33 pp | [−6.00, +5.33] |
| Qwen2.5-7B | +13.33 pp | +16.33 pp | −3.00 pp | [−9.67, +4.00] |

两个 interaction CI 均包含 0，未检出 CoT 对 steering effect 的明显增强或削弱。结果与两者产生近似可加的行为增益一致，但由于区间较宽，不能解释为严格等效或机制完全独立。

#### 5.6.1 Generation-Budget Boundary

结果限定在固定的 768-token budget 下。Llama 的 CoT baseline 与 steering 条件均高度触及生成上限（`.953` vs `.963`，paired `p=.678`）；在两格都触及上限的 276 道题中，增益仍为 `+6.16 pp`。Qwen 的 cap-hit rate 则由 `.233` 降至 `.103`，未触及上限的 207 道题中增益为 `+14.98 pp`。

这些诊断表明，观察到的提升不能简单由两格截断率差异解释，但 subgroup analysis 属于辅助分析，且截断仍可能影响绝对表现。因此，本节只主张 fixed-768-budget 下的 condition transfer，不外推至不受生成预算限制的能力表现，也不追加 1024-token cells。

#### 5.6.2 Exploratory Answer-Formation Timing

本节区分两个事件：`#### N` 是答案的**格式标记**，第一个答案候选则更接近答案的**形成时点**。两者相关，但不能互相替代。

**Table 5.6c — Llama Answer-First Pattern across Prompt Conditions (Exploratory)**

| Condition | α | Accuracy | Committed n | Answer-first n (%) | posN median |
|---|---:|---:|---:|---:|---:|
| No-CoT | 0 | .1800 | 163 | 5 (3.1%) | .2161 |
| No-CoT | −6 | .2433 | 164 | 40 (24.4%) | .2740 |
| CoT | 0 | .2000 | 124 | 14 (11.3%) | .2810 |
| **CoT** | **−6** | **.2600** | **156** | **91 (58.3%)** | **.0000** |

`answer-first` 表示生成文本以可解析的 `#### <number>` 开头，分母为 committed 样本。CoT 会增加这种输出格式：在 `α=0` 下由 3.1% 升至 11.3%，在 `α=−6` 下由 24.4% 升至 58.3%。但在 CoT 条件内，steering 仍使 accuracy 从 `.2000` 提升至 `.2600`。因此，较低的 `posN` 或 answer-first **不能单独证明模型过早形成或锁定答案**。

**Table 5.6d — Candidate-Based Answer-Formation Timing**

下表改以第一个答案候选为锚点，并在两个剂量都能定位候选的共同题目上进行配对比较。

| Task / Condition | α | Accuracy | Shared n | `cand_pos` | `pre_chars` | reason-first |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K No-CoT | 0 → **−6** | .6000 → **.7800** | 281 | .0000 → **.0843** | 0 → **175** | 29.2% → **66.5%** |
| GSM8K CoT | 0 → **−4** | .6900 → **.8500** | 252 | .0021 → **.1117** | 5 → **234.5** | 46.0% → **76.2%** |
| GSM-Hard No-CoT | 0 → **−6** | .1800 → **.2433** | 267 | .0000 → **.0937** | 0 → **201** | 25.8% → **61.0%** |
| GSM-Hard CoT | 0 → **−6** | .2000 → **.2600** | 248 | .0000 → **.1012** | 0 → **231.5** | 40.3% → **63.7%** |

- **reason-first（主指标）**：答案候选出现前是否已有等式或推理步骤。四组均显著提高（McNemar `p<1e−8`；`0→1` 分别为 120、80、112、77）。
- **`pre_chars`（补充指标）**：候选出现前生成的字符数中位数。
- **`cand_pos`（辅助指标）**：候选在全文中的归一化位置；数值越大，候选出现越晚。

四种条件下，最佳 α 都伴随更高的 accuracy、更多候选前推理和更晚的答案候选。这说明准确率提升稳定伴随答案形成时序改善，但不代表时序改善造成了准确率提升。

**Table 5.6e — Cross-Model Association between Accuracy and Commitment Timing**

| Model | Dose curve | `acc ~ posN` | `acc ~ early-cand%` |
|---|---|---:|---:|
| Llama | 9 doses (§2.2, server-182) | ρ = **+.941** (p=.0002) | — † |
| Qwen | 11 doses (Table 4.1a) | ρ = **+.863** (p=.0006) | ρ = **−.804** (p=.0029) |

† Llama 的九档表没有同口径的 `early-cand%`。其历史指标 `premature (either)` 在九档上的 ρ=−.300（n.s.）；五格冻结检测器结果为 ρ=−1.000。两种定义不可互换，因此主表留空。

尽管 Llama 与 Qwen 的最佳 α 方向相反，两者的准确率提升都伴随较少的 early candidate 和较晚的 commitment。但曲线并非“越晚越准”：

- **Llama** 的关系较连续；在过冲点 `α=−8`，`posN=0%` 且 accuracy 仅 40.3%。
- **Qwen** 在 `+6/+8` 附近发生阈值式转换；`α≤+4` 时 `posN` 均为 `.003`。
- Qwen 在 `+8` 后 accuracy 进入平台，而 `posN` 仍由 `.754` 升至 `.802`；accuracy 增幅依次为 `+8.00 / +2.33 / +0.34 pp`。

> **结论：α 改变了答案形成与提交模式。最佳工作点通常使模型摆脱过早回答，更多地呈现“先推理、后形成答案”；进入健康区间后，继续推迟答案不会持续提高准确率。**

**Interpretation Boundaries.** 三张表使用不同分母：accuracy 使用全部 300 题；answer-first 与 `posN` 使用 committed 子集；`cand_pos`、`pre_chars` 与 reason-first 使用 candidate-covered 共同子集。以 GSM-Hard CoT `α=−6` 为例，同一批 91 个 answer-first 样本占 committed 的 58.3%，但占 candidate-covered 的 34.7%，因此总体候选后移与部分样本 marker-first 可以同时存在。

`cand_pos` 可能命中等号右侧的中间结果，只作辅助下界；`early-candidate%` 在全部样本上有定义，是完整剂量曲线的主指标。`posN` 仅在 committed 子集上定义，其 Qwen 分母随剂量由 79% 升至 98%，因此只作支持。所有指标均来自 α 干预后的同一次生成，不是独立证据，也不构成 causal mediation。新增分析不修改冻结 predictor 或任何冻结产物；复算与实现细节见 `CLAUDE.md`。

#### 5.6.3 Conclusion

> 在固定 768-token budget 下，GSM8K 确立的工作点在 CoT 条件下仍能迁移到 GSM-Hard：Llama 准确率提高 6.00 pp，Qwen 提高 13.33 pp，且两者均通过 Holm 校正。CoT 与 steering 的行为增益近似可加，说明 steering 并非 CoT prompt 的简单替代；但当前结果不能证明两者具有完全独立的机制。

冻结顺序、artifact hashes、mutation tests、cap-hit subgroup 计算及完整 provenance 统一记录于 `CLAUDE.md`。

### 5.7 Fixed-Workpoint Transfer Across Reasoning Tasks

#### 5.7.1 MATH

先区分两类问题，它们在 MATH 上都做过，但不是同一件事：

- **P2B（§5.3）**：在 MATH 剂量曲线上进行 **commitment-based workpoint selection** —— 用冻结的 GSM8K predictor 去*挑*工作点。
- **P4（本节）**：直接携带 GSM8K 的 **fixed workpoint**，**不在 MATH 重新搜索**，只问它是否仍然有效。

**Table 5.7a — P4 MATH fixed-workpoint transfer（No-CoT，跨模型 Holm m=2）**

| Model | Fixed α | acc(0) | acc(α) | Δ | raw p | Holm p_adj | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | −6 | 36.67% | 43.33% | **+6.67 pp** | .0245 | **.0489** | [+1.00, +12.33] |
| Qwen2.5-7B | +8 | 60.67% | 63.33% | +2.67 pp | .3581 | .3581 | [−2.33, +7.67] |

**Table 5.7b — Llama CoT condition supplement**

| Condition | α=0 | α=−6 | Δ | Holm p_adj within CoT dose family |
|---|---:|---:|---:|---:|
| MATH CoT | 42.0% | 49.0% | **+7.00 pp** | **.0225** |

关于 Table 5.7b 的 `p_adj=.0225`，四点必须同时成立：

1. 它来自 **Llama CoT 内部** `−6 / −4 / +4 vs 0` 的 Holm **m=3**（§3.1.1）；
2. 它**不是**跨模型 P4 Holm m=2 的一部分，两者不可混用；
3. 这是 fixed-workpoint 的 **CoT condition extension**，不是独立 benchmark transfer；
4. **不能**把同一批 MATH 问题的 No-CoT 与 CoT 结果当成两次独立的任务复现 —— 它们共享题目。

> Llama 的 GSM8K fixed workpoint 在 MATH No-CoT 和 CoT 条件下均带来约 +7 pp 改善；Qwen 的 fixed +8 在 MATH 上仅呈方向性正增益且 CI 跨 0。结果支持 fixed-workpoint transfer 具有**模型与目标任务剂量曲线边界**，而非无条件普遍迁移。MATH 中未检出 CoT 对 steering 效应的明显调节（§3.5.2），但宽 CI 不支持等价性或机制独立主张。

`mdf_neg6` / `mdf_neg6_cot` 与各自的 `α=0` 均为**跨 run 比较**（存量格的物理 GPU provenance 无法从 `summary_math_*.csv` 恢复，该文件不含 device 字段）。


### 5.8 BBH `object_counting`：行为 signature 部分迁移，准确率双 null

§5.7 的 LogiQA null 无法说明*为什么*，因为 LogiQA 相对 GSM8K 同时改了两件事：答案空间（自构整数 → 四选一）与推理内容。BBH `object_counting` 把**答案空间与提交接口恢复成 GSM8K 的形式**（自构整数 + `####`），保留推理内容的差异。注意这是三个*选定*维度的比较，不是受控单因子操作：BBH 在题目格式、领域、文本分布与生成形态上同样不同。

协议 `bbh-p4b-v0` + `p4b-amend-01`。α 由冻结的 GSM8K 记录读出（Llama `−6`、Qwen `+8`），**不在 BBH 重新搜索**；每模型另加一个反方向诊断格（Llama `+4`、Qwen `−6`），该格在 Holm 家族之外，p 不校正，且**不得用于重新定义工作点**。全部 250 题，No-CoT，plain wording，`max_new_tokens=768`，greedy。α=0 已通过冻结的 headroom gate `[0.30, 0.85]`（Llama .4160、Qwen .5520），因此下面的 null 不是 baseline 天花板伪影。

**Table 5.8a — 主准确率：双 null + reverse diagnostic**

| Model | α | 角色 | first_acc | Δ vs α=0 | discordant | raw p | Holm p_adj | Bootstrap 95% CI |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | 0 | baseline | .4160 | — | — | — | — | — |
| Llama3.1-8B | **−6** | **workpoint** | .4080 | **−0.80 pp** | 27/29 | .8939 | **1.0000** | [−6.80, +5.20] |
| Llama3.1-8B | +4 | reverse diag. | .3280 | −8.80 pp | 10/32 | .0009 | *不校正* | [−13.60, −4.00] |
| Qwen2.5-7B | 0 | baseline | .5520 | — | — | — | — | — |
| Qwen2.5-7B | **+8** | **workpoint** | .5760 | **+2.40 pp** | 34/28 | .5258 | **1.0000** | [−4.00, +8.40] |
| Qwen2.5-7B | −6 | reverse diag. | .5680 | +1.60 pp | 27/23 | .6718 | *不校正* | [−4.00, +7.20] |

Holm 家族 **m=2，仅含两个 workpoint 对比**。`last_acc` sensitivity 与 MAIN 同号且同量级（Llama −0.40 pp、Qwen +0.40 pp），因此两个 null 都不是 LAST 解析器或尾部改写造成的。

**方向排序两个模型都 BREAKS。** Llama 为 `0 (.416) > −6 (.408) > +4 (.328)`，预期的 `−6 > 0 > +4` 不成立——α=0 本身高于工作点；`+4` 是本任务唯一显著的对比，且是**退化**，符合过度 steering 的损伤，不构成方向排序证据。Qwen 为 `+8 (.576) > −6 (.568) > 0 (.552)`，反向剂量反而高于 α=0，两个 CI 均跨 0。一个点不是剂量曲线：它可以显示排序是否延续，但定不出峰，也定不出倒 U。

**Table 5.8b — 探索性行为表（不进 Holm）**

| Model | α | early-cand% | 首行裸数字% | multi-marker% | 退化尾% | 语料续写% | chars 中位 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | 0 | 95.2 | 94.8 | 84.0 | 92.8 | 0.0 | 1988 |
| Llama3.1-8B | **−6** | **84.4** | 79.6 | 78.4 | 88.4 | 0.0 | 1928 |
| Llama3.1-8B | +4 | 99.2 | 98.0 | 80.0 | 94.8 | 0.0 | 2003 |
| Qwen2.5-7B | 0 | 100.0 | 54.4 | 92.0 | 3.6 | 50.8 | 422 |
| Qwen2.5-7B | **+8** | **44.4** | **0.0** | **36.0** | 4.8 | 42.8 | 322 |
| Qwen2.5-7B | −6 | 100.0 | 54.8 | 92.4 | 3.6 | 50.4 | 420 |

`early_candidate` 为冻结的 `earlycand-v1`，本任务盲审 30/30 通过（precision 1.000、recall 1.000），但 **α=0 基线处于天花板**（Llama .952、Qwen 1.000），故只有*下降*可测；**一个持平的比率不能读作「α 不改变答案形成时序」**。该 flag 是 α 的*结果*，按其分层属 post-treatment 分层，只作 consistent-with 证据，绝非中介。

**Qwen `+8` 应写作 mixed regime，不能概括为全面「先算后答」。** 它确实把首行裸数字压到 0.0%、multi-marker 由 92.0 降到 36.0，且反向格 `−6` 完全不动（100.0 / 54.8），说明这是 `+α` 特有而非任意扰动。但在 139 条转为非 early-candidate 的样本中，只有一部分真正先展开计数（`sid=1`：先逐项列举再求和，marker 落在推理之后），另一部分仅改变了答案的**表面形式**——`sid=2` 首行写英文数词 `three`、`sid=17` 首行即以散文报出 `You have a total of 9 objects (3 fridges + 1 bed + 5 stoves = 9)`，仍是先报答案。

**`pre-marker` 字符数由 313 降到 44、`posN` 由 .757 降到 .699 只作格式诊断，不能支持「答案后移」**——方向恰好相反，且该量混入了「早期空 `####`、首个可解析 marker 更晚出现」的情形。timing 证据以经过审计的 `early_candidate` 加原文形态为准。

Llama `−6` 产生同方向但弱得多的变化（early-cand 95.2 → 84.4，`+4` 反而推满至 99.2）；其三格原文形态高度相似，均为「数字 → Explanation → 尾部退化循环」。Llama 三格退化尾率 88–95%、cap-hit 高，其准确率**只能称为固定 768-token 预算下的准确率**，长度类读数不是自然长度；这是与 GSM-Hard **相似的循环/截断表型，而非已确立的相同机制**。Qwen 约 **50%** 的生成尾部带训练语料续写（`You are an AI assistant …`），三格接近（50.8 / 42.8 / 50.4），α 既未制造也未消除它；它位于 `####` 之后，不影响 first-口径抽取。

> **结论。** Qwen `+8` 明显抑制裸数字式抢答，并使部分输出转入 reasoning-before-marker；Llama `−6` 也产生较弱的同方向变化。但这些行为变化没有稳定转化为准确率改善，说明改善答案提交时序不是跨任务准确率提升的充分条件。

因此可写的是 **GSM8K 上的行为 signature 部分迁移**，而非「机制迁移成功」。同时，恢复答案空间与提交接口**不足以**恢复准确率迁移；按冻结措辞，这**不**能推出「推理内容才是约束条件」——要把选项接口效应与推理类型效应分开，需要同题有选项／无选项的对照，本协议未授权。

行为数字由 `analyze_bbh_behavior.py` 从六个原始生成文件复算至 `docs/bbh_p4b_object_counting_behavior.json`（该脚本同时复算准确率并断言与冻结的 `docs/bbh_p4b_object_counting_result.json` 逐格一致），其中包含按 `early_candidate` 转换分组的配对拆解——该拆解按 steering *之后*的行为结果选组，属 post-treatment selection，标为 exploratory，不进 Holm，且不能解释为中介效应。冻结顺序、artifact hashes 与完整 provenance 见 `CLAUDE.md`。

## References

**神经科学（次要旁证）：多巴胺 → 焦虑 / 警觉 / 威胁高估**（§2.3 / §2.4 / §3.6 的机制**旁**锚。注意本项目主机制锚已改为 **VTA→NAcc wanting 过载 → 冲动 + 固著**，见 §0.2；下列 DA→anxiety 文献有通路特异性（VTA→IPN），列此仅表明 DA 亦有独立焦虑下游，但**非**本数据 +α 端的主要解释——我们观测到的是抢答 / 固著，而非回避 / freezing）
- Dopamine release in the interpeduncular nucleus promotes anxiety. *(VTA→IPN D1 通路双向调节焦虑行为的光遗传+药理证据)* — PMC7687288. https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/
- MIT News (2018). Dopamine, brain vigilance and anxiety. *(Tye Lab：DA 提高威胁通路信噪比、压制奖励神经活动，偏向 threat/freeze)* https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107
- Dopaminergic alteration in anxiety and compulsive disorders. *Frontiers in Neuroscience* (2020). https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full
- Dopaminergic mechanisms of trait anxiety. *Journal of Neuroscience* (2019). https://www.jneurosci.org/content/39/14/2735

**候选机制（emotional salience，待 RSA 验证）**
- Brickner, M. A., Szot, W. E., Wolff, A. R., Thomas, M. J., & Saunders, B. T. (2026). Basolateral amygdala dopamine transmits emotional salience. *Nature Communications.* *(BLA DA 编码情绪显著性 / 重新判断需求，非奖赏价值；候选解释 +α 端「放不下」的 salience 过载。注意：−α 端是 under-wanting / commitment-formation failure，非 salience 过载，不由此通路解释。验证需 `Ada_Dopamine2.md` RSA 纳入 BLA/amygdala ROI。)*

**理论框架：wanting / incentive salience**
- Berridge, K. C., & Robinson, T. E. What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *(wanting ≠ liking；本工作 α = incentive salience 的母假设)*
- RSN paper (ACL ARR). Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism for Confidence Steering. *(母论文 §6.1 "Digital Dopamine"；commitment dynamics = wanting 的下游行为表现)*

**心理学框架**
- Yerkes, R. M., & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation. *(倒 U 型 arousal–performance；§1.2 acc 峰在 α=−6、两端崩的 framing 来源)*
