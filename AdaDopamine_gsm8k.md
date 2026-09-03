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

- **过高 DA / over-wanting（对应 α→正向）**：行为上表现为**冲动性抢答（impulsivity）+ 认知僵化 / 强迫性反复（compulsivity / perseveration）**——急于扑向"给出答案"这个目标而跳过必要推导，以及拿到答案后仍反复复查、纠结格式、卡在格式死循环里。**注意这更贴合冲动 / 强迫，而非焦虑**：数据中 +α 端**没有**焦虑典型的回避 / freezing / 犹豫（抢答率随 +α 单调上升，见 §2.2），呈现的是"急着 commit"；而 `#### N #### N` 死循环是认知神经科学意义上的**固著（perseveration）**,不是焦虑的发散灾难化担忧。功能上这**与 mesolimbic incentive-salience overload（VTA→NAcc 型 wanting 过载）及执行控制失效相容**：极高的诱因显著性使主体不计成本扑向目标（冲动），同时灵活切换 / 抑制已启动反应的能力下降（固著）——这在行为上更近强迫性 over-checking 与冲动特征，而非焦虑综合征。**注意目前只有行为同构，尚未定位实际脑区对应关系**，故这里说"相容"而非"落在"某一回路。这个签名对应 §2.2「正向端：答案已经出现，但仍无法停止」：α+4 trace 中常见"把已经算对的答案当可疑"（Q100、Q16）和答完仍寻找 "more efficient way"（Q68）。
- **过低 DA / under-wanting（对应 α→极端负向）**：行为上对应动力不足、快感缺失、退缩、bradykinesia 式的行动迟缓；在本任务中的可观测类比不是"写得短 / 不想答"，而是**commitment-formation failure**。§2.2 文本核验否定了两个更直观假设：α=−8 并非大量出现 "I am done / 不答了" 的词汇性退缩（跨 α 平坦，多为礼貌 loop 尾），也不是敷衍短答（长度 / 等式数平坦）；真正失败模式是 **answer-candidate oscillation**——锁不住答案，在两个候选值之间来回切，导致 committed_acc 崩到 23.6%。

> **主机制表述（本项目采用）**：+α 端 = **over-wanting → 冲动（impulsivity, 抢答）+ 认知僵化 / 强迫性反复（compulsivity / perseveration, loop）**，功能上与 **mesolimbic incentive-salience overload（VTA→NAcc 型）+ 执行控制失效**相容（只声称行为同构，非脑区定位）；−α 极端端 = **under-wanting → commitment-formation failure**。这比"焦虑"框架**机制契合度更高、论述负担更小**：冲动与固著都在我们已有的 wanting / incentive-salience 主线内，无需另起 threat/freeze 回路。
>
> **限定**：1. 本实验只声称行为同构，**α steering ≠ 生物多巴胺**，也不证明 LLM 有生理或主观状态；2. 不做 mania / hypomania 类比——躁狂是跨情绪+精力+睡眠的综合征，我们只有"冲动+固著"两个窄行为，撑不起该诊断类比；3. 脚本 `analyze_loop_anxiety.py` / `ANXIETY_PATTERNS` 沿用 "anxiety" 命名（改名成本高、破坏 U 形复现），但其命中的四子类（self-doubt / format-fixation / persona-reassurance / over-precision）**实测对应的是强迫性 over-checking，不是临床焦虑**——阅读表格时按"强迫/固著"解读。
>
> **次要旁证（不作主锚，DA→焦虑另有通路特异性）**：DA 亦有一条独立的焦虑通路证据，最强因果来自 **VTA→IPN（D1）**，机制是威胁高估 / 过度警觉——但这**不是**本数据的主要解释（我们没观测到回避 / freezing），仅作为 DA 多下游效应的旁注列出。来源：[PMC7687288 (VTA→IPN dopamine promotes anxiety)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/) · [MIT News 2018 (dopamine vigilance & anxiety)](https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107) · [Frontiers Neurosci 2020 (dopaminergic alteration in anxiety/compulsive disorders)](https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full) · [J. Neurosci 2019 (dopaminergic mechanisms of trait anxiety)](https://www.jneurosci.org/content/39/14/2735)


## 1. Llama on GSM8K: Performance Summary

Llama3.1-8B-Instruct 在 GSM8K 上呈现出三个主要结果：

- No-CoT 剂量曲线是明显的**非对称峰形**：准确率在 `α=−6` 达到最高，但到 `α=−8` 明显崩落。
- CoT 改变了负向剂量的最佳位置：No-CoT 在 `α=−6` 达到最高准确率，而 CoT 在已测试剂量中以 `α=−4` 最佳；更强的 `−6` 反而回落。
- 催促式措辞会降低准确率并压缩不同 α 之间的差异，带 persona 的条件尤其敏感。

**Setup.** Llama3.1-8B-Instruct，GSM8K 300 题，greedy decoding。正文统一报告 offline `first_acc`；`last_acc` 仅用于检查后续答案修改，不作为主要性能指标。主曲线来自同一冻结 production batch；后续 workpoint-stability 补充格为 cross-run 配对。具体运行配置、数据路径和提取口径见 `CLAUDE.md`。

### 1.1 Main Dose–Response

下表合并完整 No-CoT 曲线，以及现有的 CoT 和 pushy 对照。`—` 表示该条件未运行。

| Condition | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Plain No-CoT, first acc** | 40.3% | **78.0%** | 73.0% | 69.0% | 60.0% | 57.0% | 55.3% | 55.0% | 53.7% |
| Plain No-CoT, last acc | 41.7% | 74.7% | 68.3% | 65.3% | 55.3% | 55.3% | 52.7% | 53.3% | 52.3% |
| **Plain CoT, first acc** | — | 75.3% | **85.0%** | 74.0% | 69.0% | — | 59.7% | — | — |
| Plain CoT, last acc | — | 78.0% | 84.7% | — | 68.3% | — | 59.0% | — | — |
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

#### CoT Shifts the Best Tested Dose

加入 `α=−6/−2` 后，GSM8K CoT 已覆盖五个剂量。下表合并 No-CoT 对照、CoT 的 first/last accuracy，以及答案修改情况。`—` 表示该诊断未在本次补充分析中报告。

| α | No-CoT first acc | CoT first acc | ΔCoT | CoT last acc | CoT commit rate | 改对 / 改坏 |
|---:|---:|---:|---:|---:|---:|---:|
| −6 | **78.0%** | 75.3% | −2.7 pp | 78.0% | 37.7% | 12 / 4 |
| −4 | 73.0% | **85.0%** | **+12.0 pp** | 84.7% | 31.7% | 3 / 4 |
| −2 | 69.0% | 74.0% | +5.0 pp | — | — | — |
| 0 | 60.0% | 69.0% | +9.0 pp | 68.3% | 37.7% | 1 / 3 |
| +4 | 55.3% | 59.7% | +4.4 pp | 59.0% | 29.0% | 0 / 2 |

CoT 条件下的准确率排序为：

`α=−4 > −6 > −2 > 0 > +4`

因此，`α=−4 + CoT` 的 85.0% 是当前已测试 Llama GSM8K 条件中的最高准确率。`α=−6` 并不是 CoT 曲线的最佳点：它相对 CoT baseline 只提高 6.33 pp，而且明显低于 `α=−4`。

| α vs CoT baseline | ΔAccuracy | 0→1 / 1→0 | Raw p | Holm p_adj | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| −6 | +6.33 pp | 52 / 33 | .0503 | .0503 | [+0.33, +12.33] |
| −4 | **+16.00 pp** | 66 / 18 | 1.33e−07 | **4.0e−07** | [+10.33, +21.67] |
| +4 | −9.33 pp | 36 / 64 | .00664 | **.0133** | [−15.67, −2.67] |

以上三项属于同一个 Llama CoT dose family，使用 Holm `m=3` 校正。`α=−6` 校正后 `p=.0503`，因此按主要检验记为未显著；bootstrap CI 仅作为效应范围的补充。

`α=−2` 属于独立的 workpoint-stability 补充家族（Holm `m=7`）。它相对 CoT baseline 提高 5.00 pp（35/20），但校正后 `p_adj=.174`，95% CI 为 `[+0.33, +9.67]`。与 `α=−4` 的邻点比较为 −11.00 pp（7/40，探索性 `p=1.07e−06`）。结合 `−6` 与 `−4` 的直接比较（+9.67 pp，42/13，`p=1.1e−04`），`−4` 已被 `−6/−2` 两个相邻负向剂量夹住，是当前清晰的局部峰值。邻点检验是看到原曲线后设计的补充分析，因此不进入 Holm `m=7` 家族。

CoT 是否改变 `−4/−6` 相对排序，可以用四格配对 difference-in-differences（DiD）直接检验：

| Accuracy definition | CoT −6 | CoT −4 | No-CoT −6 | No-CoT −4 | DiD | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|---|---:|
| **First accuracy** | 75.33% | 85.00% | 78.00% | 73.00% | **+14.67 pp** | [+7.67, +21.67] | **<1e−4** |
| Last accuracy | 78.00% | 84.67% | 74.67% | 68.33% | **+13.00 pp** | [+5.67, +20.33] | .0005 |

Last accuracy 的结果与主要 first-accuracy 口径同号且量级相当，因此“CoT 改变 `−4/−6` 相对排序”不是由 first-marker 提取口径造成的。

更重要的是，CoT 改变了负向剂量的排序：

- No-CoT：`−6` 78.0% > `−4` 73.0%
- CoT：`−4` 85.0% > `−6` 75.3%

因此，GSM8K 的最佳工作区间依赖推理条件：No-CoT 的近优区间是 `{−6, −4}`，CoT 则收窄为 `{−4}`。从 No-CoT 得到的固定 `α=−6` 在 CoT 下仍然是正向点估计，但不是新条件下的最优剂量。

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
3. **CoT 改变了有效工作区间。** No-CoT 的近优区间为 `{−6, −4}`，CoT 在已测试剂量中以 `α=−4` 形成清晰局部峰值（85.0%）；DiD 证实 `−4/−6` 的排序变化不是答案提取造成的。
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

#### CoT Changes the Negative-Side Ordering

加入 `α=−6` 后，CoT 不再只是整体提高准确率，也改变了负向剂量的最佳位置。

| α (CoT) | First accuracy | Last accuracy | Commit rate | Answer-first among committed |
|---:|---:|---:|---:|---:|
| −6 | 75.3% | 78.0% | 37.7% | **40.7%** |
| −4 | **85.0%** | 84.7% | 31.7% | 15.8% |
| 0 | 69.0% | 68.3% | 37.7% | 19.5% |
| +4 | 59.7% | 59.0% | 29.0% | 0.0% |

`α=−6 + CoT` 的 answer-first rate 达到 40.7%（46/113 committed），明显高于 `−4` 和 baseline。部分输出会先给出一个错误的 `####` 答案，再在后续步骤中推导出正确答案；由于主指标读取第一个 `####`，这些样本仍记为错误。这与 `−6` 的 last accuracy 高于 first accuracy 2.7 pp 相一致。

不过，answer-first 是 α 干预后的输出行为，而且 `α=+4` 在 answer-first 为零时准确率仍然较低。因此，它只能作为 `−6 + CoT` 回落的相符线索，不能作为普遍的准确率中介或因果解释。

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
4. **CoT 会改变最佳工作点。** No-CoT 以 `α=−6` 最佳，而当前 CoT 剂量中 `α=−4` 最佳；`−6 + CoT` 较高的 answer-first rate 是与其回落相符的行为线索，但不是因果中介证明。
5. **Persona 主要改变重复内容。** `non_expert` 更容易产生身份否定，pushy wording 会进一步放大这一模式。
6. **口头自评不是可靠的 wanting 指标。** Willingness 和 confidence 曲线不一致，极端剂量还出现量表锁定和格式失效。
7. **Wanting 是功能类比，不是直接测量。** 更稳妥的表述是：α 改变了模型的 engagement、commitment 和 stopping behavior，这些现象与 incentive-salience gain 的计算类比相容，但不等于生物多巴胺或主观欲望。

## 3. Llama on MATH

本节把同一套剂量与提取口径应用到 MATH，检验 GSM8K 上的方向性是否在更难的数学推理任务上保持。运行配置、生成预算和提取口径见 `CLAUDE.md`；主要准确率指标同样是 offline `first_acc`。

### 3.1 Main Performance

#### Accuracy

| α | No-CoT | CoT | ΔCoT |
|---:|---:|---:|---:|
| −8 | 39.3% | 45.3% | +6.0 pp |
| **−6** | **43.3%** | **49.0%** | +5.7 pp |
| −4 | 40.0% | 45.0% | +5.0 pp |
| 0 | 36.7% | 42.0% | +5.3 pp |
| +4 | 33.0% | 38.7% | +5.7 pp |

两条曲线都在 `α=−6` 取得最高点估计，但 `−8/−6/−4` 形成宽的负向高表现区间：

- No-CoT：`−6` 43.3% > `−4` 40.0% > `−8` 39.3% > `0` 36.7% > `+4` 33.0%
- CoT：`−6` 49.0% > `−8` 45.3% > `−4` 45.0% > `0` 42.0% > `+4` 38.7%

`α=−6 + CoT` 的 49.0% 是当前 5×2 矩阵中的最高点估计。但 `−8` 与 `−4` 都未与 `−6` 显著分开，因此更准确的结论是：No-CoT 和 CoT 的近优区间都是 `{−8, −6, −4}`，而不是一个被精确确定的单点峰值。

#### Dose Effects Within No-CoT and CoT

原始 `−6/−4/+4` versus `0` 的比较在 No-CoT 和 CoT 下分别构成 exploratory dose family，各自使用 Holm `m=3` 校正。新增 `−8` 属于独立的 workpoint-stability 家族。

| α vs 0 | No-CoT Δ | No-CoT p_adj | CoT Δ | CoT p_adj | Statistical family |
|---:|---:|---:|---:|---:|---|
| −8 | +2.67 pp | .403 | +3.33 pp | .328 | Workpoint stability, Holm `m=7` |
| **−6** | +6.67 pp | .0734 | **+7.00 pp** | **.0225** | Original dose families, each Holm `m=3` |
| −4 | +3.33 pp | .3697 | +3.00 pp | .5057 | Original dose families, each Holm `m=3` |
| +4 | −3.67 pp | .3697 | −3.33 pp | .5057 | Original dose families, each Holm `m=3` |

从 GSM8K 携带的固定工作点 `α=−6` 在两个条件下均带来约 7 pp 的正向差异。其中：

- CoT 条件下，`−6 vs 0` 通过 Holm `m=3` 校正。
- No-CoT 条件下，raw `p=.0245`，bootstrap 95% CI 为 `[+1.00, +12.33]`，但 Holm 校正后 `p_adj=.0734`，因此只能描述为方向明确、校正后未显著。

`α=−8` 的两项检验属于独立的 workpoint-stability Holm `m=7` 家族，不与原始 MATH dose family 合并。它们相对 baseline 的差异均未被检出。

`α=−6` 不是根据 MATH 结果重新挑选的剂量。它由 GSM8K 预先确定，补跑后恰好是当前离散 argmax；但邻点比较显示，No-CoT 下 `−6` 与 `−8/−4` 的差异分别为 4.00/3.33 pp（`p=.126/.212`），CoT 下为 3.67/4.00 pp（`p=.090/.104`）。因此 `−6` 更适合解读为位于 `{−8, −6, −4}` 宽峰区间内的稳健工作点。这些邻点比较是探索性的，不进入 Holm 家族。

#### CoT Gain at Each Dose

| α | ΔCoT | 0→1 / 1→0 | Raw p | Holm p_adj | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| −8 | +6.00 pp | — | — | — | — |
| −6 | +5.67 pp | 40 / 23 | .0430 | .1718 | [+0.67, +10.67] |
| −4 | +5.00 pp | 35 / 20 | .0581 | .1718 | [+0.00, +9.67] |
| 0 | +5.33 pp | 46 / 30 | .0846 | .1718 | [−0.33, +11.00] |
| +4 | +5.67 pp | 40 / 23 | .0430 | .1718 | [+0.67, +10.67] |

CoT 在五个剂量下的点估计都提高约 5–6 pp。原始四剂量的 CoT comparison 家族使用 Holm `m=4`，校正后均未达到显著；`−8` 是后续 workpoint-stability 补充格，未预先纳入该家族，因此只保留描述性差异。

#### CoT × Steering Interaction

交互量定义为：

`[Acc(CoT, α) − Acc(CoT, 0)] − [Acc(No-CoT, α) − Acc(No-CoT, 0)]`

| α | Interaction | Bootstrap 95% CI |
|---:|---:|---:|
| −8 | +0.67 pp | — |
| −6 | +0.33 pp | [−6.67, +7.67] |
| −4 | −0.33 pp | [−6.67, +6.00] |
| +4 | +0.33 pp | [−6.67, +7.33] |

`−8` 的描述性点估计也接近 0，与原有三个点的近似平行形状一致。原有三个 CI 均跨 0，且约覆盖 ±7 pp，因此当前结果只能说明：

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

1. **负向 steering 表现更好，但曲线是宽峰而非单点峰值。** No-CoT 和 CoT 的离散 argmax 都是 `−6`，但近优区间均为 `{−8, −6, −4}`。
2. **CoT 提高整体表现，但没有明显改变宽峰区间。** 五个剂量的点估计均提高约 5–6 pp；原始四剂量的逐点 CoT 比较经 Holm 校正后均未显著。
3. **正向 α 伴随更差的提交质量和更多无效反复。** 这一关系在 No-CoT 的 generation length、committed accuracy 和 commit rate 上严格单调（四个剂量），在 full-text compulsive repetition 上则是 `α≥0` 段清楚、负向端已接近下限（`−6` 42 与 `−4` 40 实质持平）。
4. **CoT 主要增加推理结构并减少强迫性反复。** 当前数据未检出 CoT 明显改变 steering 效应，但宽 CI 不支持机制独立或严格可加的结论。
5. **输出行为不等于内部机制。** boxed position、commit rate、first–last gap 和重复文本都是行为读数，不能单独证明答案形成时间、因果中介或生物学 dopamine 机制。MATH 的 boxed position 尤其是阴性对照（§4.2），承诺时序须以 early-candidate rate 为准。

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
## 5. Commitment-Based Prediction and Workpoint Selection

本节检验两个问题：

1. Commitment behavior 能否预测未见题目的正确率？
2. 冻结的 commitment predictor 能否在新的剂量曲线上选择较好的 steering workpoint？

这里的“workpoint selection”是根据目标任务的完整剂量曲线选择 α。它不同于下一节的“fixed-workpoint transfer”：后者不重新选择剂量，而是直接使用 GSM8K 已经确定的 α。

训练、特征编码、数据清单、交叉验证、统计检验和产物校验记录于 `CLAUDE.md`。主要准确率指标均为 `first_acc`。

### 5.1 Held-Out Correctness Prediction on GSM8K

Predictor 使用 early candidate、commit state、标准化 commit position（`posN`）及其可观测性预测每道题是否正确。Raw α 不作为输入特征。同一道题的所有剂量始终位于同一个交叉验证 fold，避免同题信息泄漏。

**Table 5.1. Held-out correctness prediction on GSM8K**

| Model | Commitment-only AUROC | Entry-only AUROC | Commitment − Entry | Combined − Commitment | Calibration slope |
|---|---:|---:|---:|---:|---:|
| Llama3.1-8B | **.687** [.656, .719] | .548 [.526, .571] | **+.139** [+.104, +.172] | −.001 [−.004, +.002] | .95 |
| Qwen2.5-7B | **.749** [.710, .787] | .628 [.601, .654] | **+.121** [+.084, +.156] | +.002 [−.002, +.007] | .98 |

Commitment features 在两个模型上都能预测未见 GSM8K 题目的正确率，并且明显优于只使用 entry gain。在 commitment features 基础上加入 entry gain，没有带来可检测的额外提升。

这说明答案形成和提交行为包含与正确率有关的信息，但不证明这些行为造成了正确率变化，也不证明 entry gain 没有机制作用。

### 5.2 Retrospective Workpoint Selection on MATH

冻结的 GSM8K predictor 随后直接应用于 MATH，不使用 MATH accuracy 重新训练、调参或校准。预测分数用于排列剂量，而不是估计 MATH 的绝对准确率。

**Table 5.2. Retrospective commitment-based workpoint selection on MATH**

| Model | Available curve | Predicted direction | Direction match | Spearman ρ | Selected α | Observed best α | Near-optimal set | Regret |
|---|---|---|---|---:|---:|---:|---|---:|
| Qwen2.5-7B | 9 doses | Positive | **Correct** | **+.962** | **+6** | +6 | {+4, +6} | **0.00pp** |
| Llama3.1-8B | 3 doses | Negative | **Correct** | +1.000 | −4 | −4 | {−4, 0} | 0.00pp |

Qwen 的完整曲线允许检验 workpoint selection。Predictor 正确选中 `+6`，也识别出 `+8` 的准确率回落。不过，预测分数约为 `.83–.88`，实际准确率只有 `.54–.68`，说明迁移的是剂量排序，而不是绝对概率校准。

Llama 在原始分析时只有 `−4/0/+4`，因此只能证明局部 steering 方向正确，不能证明 predictor 在完整曲线上选中了全局最佳剂量。

> **历史边界：** Llama 的 `−6` MATH 数据是在原始 P2B 分析之后补充的，不能回填成 predictor 当时已经选中 `−6`。后续 `−6` 结果属于 fixed-workpoint transfer，而不是原始 workpoint-selection 结果。

这一阶段是规则冻结后的回顾性迁移，不是真正的盲测。

### 5.3 Prospective Blind Validation on GSM-Hard

下一步在尚未查看 accuracy 的 GSM-Hard 上进行前瞻性验证。Predictor、剂量、workpoint 选择规则和成功标准均在 gold 解封前冻结。

**Table 5.3. Predicted and observed GSM-Hard dose curves**

| Model | Metric | −8 | −6 | −4 | 0 | +4 | +6 | +8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | Predicted score | .5554 | **.68834** | .68828 | .6303 | .5770 | — | — |
|  | Observed `first_acc` | .1100 | **.2433** | .2400 | .1800 | .1700 | — | — |
| Qwen2.5-7B | Predicted score | — | — | .6959 | .7038 | .6794 | .7182 | **.8552** |
|  | Observed `first_acc` | — | — | .3433 | .3400 | .3467 | .4033 | **.5033** |

**Table 5.4. Blind workpoint-selection results**

| Model | Predicted direction | Selected α | Observed best α | Observed near-optimal set | Spearman ρ | Regret |
|---|---|---:|---:|---|---:|---:|
| Llama3.1-8B | Negative | **−6** | −6 | {−6, −4} | **+1.000** | **0.00pp** |
| Qwen2.5-7B | Positive | **+8** | +8 | {+8} | +.600 | **0.00pp** |

Predictor 正确判断了两个模型的有效方向，并选中了 observed near-optimal workpoint。两条实际剂量曲线均可区分：Llama 的最小 Holm-adjusted p 为 `2.29e−7`，Qwen 为 `1.35e−7`。

Llama 的 `−6` 与 `−4` 预测分数只相差约 `.00006`，实际准确率也只差 `.0033`，因此不能说 predictor 精确区分了这两个剂量。更准确的结论是，它选中了正确的近优区间。

Qwen 的 `+8` 选择更明确，但整体排序并不完美：predictor 将 `−4` 排在 `0` 之上，而实际准确率只相差一道题。两个模型的预测分数也都系统性高于实际准确率，因此 absolute probability calibration 没有迁移。

### 5.4 Workpoints Are Usually Regions, Not Single Doses

单一 argmax 容易把抽样波动误写成精确的最优剂量。下表重新检查六条用于建立或解释 workpoint 的曲线：若某剂量与离散 argmax 的配对差异未被检出，则一并纳入近优区间。这些邻点比较是探索性分析，不进入原有 Holm 家族。

**Table 5.5. Observed near-optimal regions on the source curves**

| Model and curve | Discrete argmax | Near-optimal region | Difference from runner-up | Discordant | Exploratory p |
|---|---:|---|---:|---:|---:|
| Llama GSM8K No-CoT | −6 | **{−6, −4}** | +5.00 pp vs −4 | 44/29 | .101 |
| Llama GSM8K CoT | −4 | **{−4}** | +9.67 pp vs −6 | 42/13 | 1.1e−04 |
| Llama MATH No-CoT | −6 | **{−8, −6, −4}** | +3.33 pp vs −4 | 31/21 | .212 |
| Llama MATH CoT | −6 | **{−8, −6, −4}** | +3.67 pp vs −8 | 23/12 | .090 |
| Qwen GSM8K No-CoT | +8 | **{+8}** | +8.00 pp vs +6 | 41/17 | .0022 |
| Qwen GSM8K CoT | +6 | **{+6, +8}** | +2.33 pp vs +8 | 26/19 | .371 |

Qwen No-CoT 这一行描述当初用于确定 workpoint 的冻结主曲线；后续探索性 `+10/+12` 已显示 `+8` 之后是高剂量平台，而非一个已闭合的峰值。

六条曲线中，四条的最佳结果更适合表达为区间。Llama GSM8K CoT 的 `{−4}` 是唯一清晰的单点负向解；Qwen GSM8K No-CoT 的 `+8` 则是冻结主曲线上的单点正向解，但后续高剂量结果将它解释为平台入口。因此，workpoint selection 的合理目标是找到方向正确、regret 较低的近优区间，而不是声称精确命中唯一 argmax。

### 5.5 Supporting Answer-Formation Evidence

Predictor 的有效性与答案形成时序变化相一致。下表使用两个剂量都能定位答案候选的共同题目，比较候选出现前是否已经存在推理内容。

**Table 5.6. Candidate-based answer-formation timing**

| Task and condition | Dose comparison | Accuracy | Shared n | Candidate position | Pre-candidate chars | Reason-first |
|---|---|---:|---:|---:|---:|---:|
| GSM8K No-CoT | 0 → **−6** | .6000 → **.7800** | 281 | .0000 → **.0843** | 0 → **175** | 29.2% → **66.5%** |
| GSM8K CoT | 0 → **−4** | .6900 → **.8500** | 252 | .0021 → **.1117** | 5 → **234.5** | 46.0% → **76.2%** |
| GSM-Hard No-CoT | 0 → **−6** | .1800 → **.2433** | 267 | .0000 → **.0937** | 0 → **201** | 25.8% → **61.0%** |
| GSM-Hard CoT | 0 → **−6** | .2000 → **.2600** | 248 | .0000 → **.1012** | 0 → **231.5** | 40.3% → **63.7%** |

四组的 reason-first 比例均明显提高（McNemar `p<1e−8`）。较好的工作点通常伴随更多候选前推理和更晚出现的答案候选。

完整剂量曲线也显示相同关联：

**Table 5.7. Association between accuracy and commitment timing**

| Model | Dose curve | `accuracy ~ posN` | `accuracy ~ early-candidate%` |
|---|---|---:|---:|
| Llama3.1-8B | 9 doses | ρ=**+.941**, p=.0002 | — |
| Qwen2.5-7B | 11 doses | ρ=**+.863**, p=.0006 | ρ=**−.804**, p=.0029 |

Llama 的九档曲线没有与 Qwen 完全相同的 frozen early-candidate 指标，因此该格保留为空，不能用历史 `premature` 指标替代。

这些关系不能解释为“答案越晚越好”。Qwen 在 `+8` 后准确率已经进入平台，但 `posN` 仍由 `.754` 上升至 `.802`。更准确的说法是：较好的工作点通常使模型摆脱过早回答；进入较稳定的区间后，继续推迟答案不会持续提高准确率。

所有 timing 指标都是 α 干预后的输出结果，并且部分指标只在 committed 或 candidate-covered 子集中定义，因此属于关联证据，不构成因果中介证明。

### 5.6 Conclusion

Commitment features 能预测 GSM8K 未见题目的正确率，也能为 MATH 和 GSM-Hard 提供有用的剂量排序信息。

其中，MATH 是回顾性的 locked transfer；GSM-Hard 才是前瞻性盲测。两者都表明 commitment predictor 更适合选择方向和低 regret 的近优工作区间，而不是精确命中唯一 argmax，也不是直接预测新任务的绝对准确率。

## 6. Fixed-Workpoint Transfer, Local Stability, and Task Boundaries

> 本节检验：不在目标任务上重新搜索 α，直接使用 GSM8K 确定的 workpoint，是否仍能提高准确率？

冻结的迁移点为：

- Llama3.1-8B：`α=−6`
- Qwen2.5-7B：`α=+8`

这与 §5 的 workpoint selection 不同：selection 根据目标任务的完整剂量曲线重新选择 α；fixed-workpoint transfer 则必须沿用 GSM8K 已经确定的剂量。

后续补充剂量只用于检查固定点附近的局部稳定性，不改变原有迁移检验。本文所称的 near-optimal region，是指**在已测离散剂量中，与观察最佳点未被显著区分的集合**；它不是连续区间，也不代表已经证明这些剂量统计等效。

### 6.1 Core Transfer Results

**Table 6.1. GSM8K-derived fixed-workpoint transfer on MATH and GSM-Hard**

| Target | Condition | Model | Frozen α | acc(0) | acc(α) | Δ | Main inference | 95% CI | Result |
|---|---|---|---:|---:|---:|---:|---|---|---|
| GSM-Hard | No-CoT | Llama | −6 | .1800 | .2433 | **+6.33 pp** | raw `p=.00661` | — | Positive |
| GSM-Hard | No-CoT | Qwen | +8 | .3400 | .5033 | **+16.33 pp** | raw `p=1.41e−08` | — | Positive |
| GSM-Hard | CoT | Llama | −6 | .2000 | .2600 | **+6.00 pp** | Holm `p_adj=.00393` | [+2.33, +10.00] | Positive |
| GSM-Hard | CoT | Qwen | +8 | .3800 | .5133 | **+13.33 pp** | Holm `p_adj=9.42e−06` | [+8.00, +19.00] | Positive |
| MATH | No-CoT | Llama | −6 | .3667 | .4333 | **+6.67 pp** | Holm `p_adj=.0489` | [+1.00, +12.33] | Positive |
| MATH | No-CoT | Qwen | +8 | .6067 | .6333 | +2.67 pp | Holm `p_adj=.3581` | [−2.33, +7.67] | Not detected |
| MATH | CoT | Llama | −6 | .4200 | .4900 | **+7.00 pp** | Holm `p_adj=.0225` | — | Positive |
| MATH | CoT | Qwen | +8 | .6300 | .6400 | +1.00 pp | Holm `p_adj=1.0000` | — | Not detected |

各行来自不同的预先定义统计家族，表中的 p 值不能跨行直接比较。完整的统计家族、raw p 和运行记录保留在 `CLAUDE.md`。

GSM-Hard 提供了最稳定的迁移结果：不重新选择 α，两个模型在 No-CoT 和 CoT 条件下均获得准确率提升。No-CoT 结果同时承担 §5.3 的前瞻性 blind workpoint-selection 验证，因此不是一项独立重复证据。

MATH 则表现出模型差异。Llama 的固定 `−6` 在 No-CoT 和 CoT 下均提高准确率；Qwen 的固定 `+8` 在两个条件下都只有较小的点估计增益，且均未被检出。§5 的 commitment predictor 在完整 Qwen MATH 曲线上选中的是 `+6`，说明目标任务重新选择 workpoint 与直接迁移固定点支持的是不同结论。

### 6.2 From Fixed Points to Near-Optimal Regions

七个预先声明的 workpoint-stability 补充格共同组成独立的 Holm `m=7` 家族。下表同时保留新增剂量的主要结果，以及用于描述局部区域的邻点比较。

**Table 6.2. Workpoint-stability results and observed near-optimal regions**

| Curve | Frozen/reference point | Stability cell(s) versus α=0 | Observed near-optimal region | Key neighbour comparison |
|---|---|---|---|---|
| Llama GSM8K CoT | `−6`: 75.33% | `−2`: 74.00%, Δ=+5.00 pp, `p_adj=.174`, CI=[+0.33,+9.67] | **{−4}** | `−4` vs `−6`: +9.67 pp, `p=.000114`; vs `−2`: +11.00 pp, `p=1.07e−06` |
| Llama MATH No-CoT | `−6`: 43.33% | `−8`: 39.33%, Δ=+2.67 pp, `p_adj=.403`, CI=[−3.00,+8.00] | **{−8,−6,−4}** | `−6` vs `−8`: +4.00 pp, `p=.126`; vs `−4`: +3.33 pp, `p=.212` |
| Llama MATH CoT | `−6`: 49.00% | `−8`: 45.33%, Δ=+3.33 pp, `p_adj=.328`, CI=[−0.67,+7.67] | **{−8,−6,−4}** | `−6` vs `−8`: +3.67 pp, `p=.0895`; vs `−4`: +4.00 pp, `p=.104` |
| Llama GSM-Hard CoT | `−6`: 26.00% | `−4`: 30.00%, Δ=+10.00 pp, `p_adj=1.36e−06`, CI=[+6.33,+14.00] | **{−6,−4}** | `−4` vs `−6`: +4.00 pp, `p=.065` |
| Qwen GSM-Hard No-CoT | `+8`: 50.33% | `+10`: 50.33%, Δ=+16.33 pp, `p_adj=9.90e−08`, CI=[+11.00,+21.67] | **{+8,+10}** | `+10` vs `+8`: 0.00 pp, `p=1.000` |
| Qwen GSM-Hard CoT | `+8`: 51.33% | `+6`: 49.00%, Δ=+11.00 pp, `p_adj=.000270`, CI=[+5.67,+16.33]; `+10`: 50.33%, Δ=+12.33 pp, `p_adj=8.46e−05`, CI=[+7.00,+18.00] | **{+6,+8,+10}** | `+8` vs `+6`: +2.33 pp, `p=.371`; vs `+10`: +1.00 pp, `p=.664` |

表中的 neighbour comparison 是看到原始曲线后设计的探索性比较，使用未校正 p 值，不进入 Holm `m=7` 家族，也不能重新定义冻结工作点。

这些结果说明：

- **Llama GSM8K CoT 是清晰的例外。** `−4` 同时高于 `−6` 和 `−2`，构成目前唯一明确的单点负向局部峰。
- **Llama MATH 的峰区较宽。** No-CoT 和 CoT 都支持 `{−8,−6,−4}`，因此 `−6` 更适合解释为宽峰区间内的稳健工作点，而不是精确峰值。
- **Llama GSM-Hard CoT 支持 `{−6,−4}`。** `−4` 的点估计更高，但与固定 `−6` 的邻点差异未达到显著。
- **Qwen GSM-Hard 呈现高剂量平台。** 固定 `+8` 位于 No-CoT 的 `{+8,+10}` 和 CoT 的 `{+6,+8,+10}` 中；在已测范围内尚未观察到平台回落，因此右侧边界仍未闭合。

因此，固定 workpoint 可以是一个稳定、低 regret 的选择，但不必是唯一最佳点。不同任务和模型的近优区域并不相同，原始 α 也不能作为跨模型共享的剂量尺度。

### 6.3 Exploratory Task Boundaries

LogiQA 2.0、BBH object counting 和 CRUXEval-O 用于探索 fixed-workpoint transfer 的任务边界。它们改变了推理内容、答案空间或生成与评分接口，因此单独报告，不用于重新定义 GSM8K workpoint，也不与 MATH/GSM-Hard 合并为统一成功率。

**Table 6.3. Fixed-workpoint transfer on exploratory boundary tasks**

| Target and main metric | Model | Frozen α | acc(0) | acc(α) | Δ | Holm p_adj | 95% CI | Sensitivity |
|---|---|---:|---:|---:|---:|---:|---|---|
| LogiQA 2.0, LAST | Llama | −6 | .5633 | .5200 | −4.33 pp | .107 | [−8.33, −0.33] | FIRST: −3.33 pp |
| LogiQA 2.0, LAST | Qwen | +8 | .6400 | .6500 | +1.00 pp | .801 | [−4.00, +6.33] | FIRST: +0.67 pp |
| BBH object counting, FIRST | Llama | −6 | .4160 | .4080 | −0.80 pp | 1.000 | [−6.80, +5.20] | LAST: .412→.408 |
| BBH object counting, FIRST | Qwen | +8 | .5520 | .5760 | +2.40 pp | 1.000 | [−4.00, +8.40] | LAST: .560→.564 |
| CRUXEval-O, FIRST | Llama | −6 | .3467 | .3100 | −3.67 pp | .1352 | [−8.00, +0.67] | LAST: −2.33 pp |
| CRUXEval-O, FIRST | Qwen | +8 | .2933 | .3767 | **+8.33 pp** | **.0045** | [+3.33, +13.67] | LAST: −6.00 pp |

#### LogiQA 2.0

两个模型的固定 workpoint 均未通过 Holm 校正，FIRST sensitivity 与 LAST 主结果方向一致。因此，该结果不能简单归因于 final-answer parser 或答案的后续修改。

LogiQA 同时改变了推理内容和答案接口，无法判断具体是哪一个因素限制了迁移。它只能说明 GSM8K workpoint 没有稳定迁移到当前的生成式多选逻辑推理设置。

#### BBH Object Counting

BBH 恢复了与 GSM8K 更接近的整数答案和 `####` 提交形式，但两个模型仍未检出准确率提升。恢复输出接口不足以恢复迁移，因此不能把 LogiQA 的 null 单独归因于多选题形式。

探索性 reverse-dose 结果也没有提供一致的剂量排序：

- Llama：`0 (.416) > −6 (.408) > +4 (.328)`；
- Qwen：`+8 (.576) > −6 (.568) > 0 (.552)`。

Steering 确实改变了部分输出行为，但这些变化没有稳定转化为准确率提升，因此只支持行为 signature 的部分迁移，不支持跨任务机制已经得到验证。

#### CRUXEval-O

CRUXEval-O 要求生成任意 Python 字面量。主结果使用预先规定的 FIRST marker，并通过 Python 字面量解析与对象相等进行评分，不等同于官方 exec-based pass@1。

Llama 的固定 `−6` 未检出提升。其另外两个探索性剂量也接近 baseline：`−4=.3333`、`+4=.3367`，相较 `α=0` 的 `.3467` 均未形成稳定增益。

Qwen 在 FIRST 主口径下呈现完整的点估计排序：

`−6 (.2400) < 0 (.2933) < +6 (.3033) < +8 (.3767)`。

其中固定 `+8` 提高 8.33 pp，并通过 Holm 校正。不过，LAST sensitivity 从 `.2833` 降至 `.2233`，方向变为 −6.00 pp。这说明结果依赖预先规定的答案提交口径：它支持 FIRST marker 下的固定点迁移，但不能概括为不受解析方式影响的整体能力提升。

完整的运行配置、统计家族、reverse/neighbor 检验、解析器限制、哈希与输出行为诊断保留在 `CLAUDE.md`。

### 6.4 Conclusion

1. **固定工作点在相近数学推理任务上具有有限迁移能力。** GSM-Hard 上两个模型在 No-CoT 和 CoT 下均获得提升；MATH 上只有 Llama 的固定点获得稳定支持。
2. **工作点通常更适合解释为任务相关的近优区域。** Llama MATH 位于宽负向峰区，Qwen GSM-Hard 位于尚未闭合的正向平台；Llama GSM8K CoT 的 `−4` 则是条件特异的局部峰。
3. **迁移不会自动扩展到所有任务。** LogiQA 和 BBH 均未检出稳定提升；CRUXEval-O 只在 Qwen 的 FIRST 主口径下得到正向结果，并对答案提交口径敏感。
4. **不存在跨模型、跨任务统一的最佳 α。** 冻结点仍用于原有迁移检验，near-optimal region 只用于描述已测剂量中的局部稳定性。
5. **这些结果属于模型输出与准确率层面的证据。** 它们不证明生物多巴胺、通用 wanting 轴或因果中介机制。

## References

**神经科学（次要旁证）：多巴胺 → 焦虑 / 警觉 / 威胁高估**（§2.2 / §2.3 的机制**旁**锚。注意本项目主机制锚已改为 **VTA→NAcc wanting 过载 → 冲动 + 固著**，见 §0.2；下列 DA→anxiety 文献有通路特异性（VTA→IPN），列此仅表明 DA 亦有独立焦虑下游，但**非**本数据 +α 端的主要解释——我们观测到的是抢答 / 固著，而非回避 / freezing）
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
- Yerkes, R. M., & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation. *(倒 U 型 arousal–performance；§1.1 acc 峰在 α=−6、两端崩的 framing 来源)*
