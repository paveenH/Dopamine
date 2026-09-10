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

本节使用统一的 candidate/marker 检测口径，比较 α 对答案候选出现顺序、正式提交和后续生成的影响。每个条件均包含 300 题，`first_acc` 仅作为行为变化的性能参照。

“答案候选（candidate）”与“正式答案标记（marker）”是两个不同事件。Candidate 是输出中最早出现的答案形态数值或表达式，可能是最终答案，也可能只是中间结果；marker 则是 `####` 等正式提交格式。因此：

- `Early candidate`、`Reason first` 和 candidate 前后字符数描述答案候选的出现顺序。
- `Marker position` 描述正式答案标记在全文中的位置，不代表答案在模型内部形成的时间。
- `Conditional acc` 只在存在有效正式答案标记的样本中计算，不能替代总体准确率。

#### No-CoT Condition

| α | First acc | Valid submission | Conditional acc | Early candidate | Reason first | Pre-candidate chars | Post-candidate chars | Marker position | Multiple markers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 40.33% | 60.67% | 23.63% | 76.67% | 45.39% | 5 | 2,259 | 0.0000 | 9.67% |
| **−6** | **78.00%** | 60.67% | **79.67%** | **18.67%** | **66.55%** | **175** | 1,875 | 0.2613 | 19.00% |
| −4 | 73.00% | 58.33% | 78.29% | 30.33% | 32.07% | 0 | 1,899 | 0.2134 | 21.67% |
| −2 | 69.00% | 63.00% | 76.19% | 41.33% | 24.07% | 0 | 1,963 | 0.1659 | 21.67% |
| 0 | 60.00% | 62.67% | 68.62% | 48.00% | 30.85% | 0 | 1,989 | 0.1772 | 21.00% |
| +2 | 57.00% | 53.00% | 66.04% | 59.67% | 26.30% | 0 | 1,992 | 0.1825 | 17.67% |
| +4 | 55.33% | 49.00% | 63.95% | 71.00% | 18.44% | 0 | 2,126 | 0.1455 | 14.33% |
| +6 | 55.00% | 45.33% | 62.50% | 75.67% | 16.79% | 0 | 2,172 | 0.1121 | 16.00% |
| +8 | 53.67% | 53.00% | 58.49% | 70.67% | 27.27% | 0 | 2,167 | 0.1204 | 21.33% |

`α=−6` 同时具有最高总体准确率和最高 conditional accuracy。它的 early-candidate rate 只有 18.67%，reason-first rate 为 66.55%，首个候选答案之前的字符数中位数为 175，说明该剂量更常在答案候选出现前生成可见的推理文本。

从 `α=0` 向正向移动时，early-candidate rate 整体由 48.00% 上升至 70% 以上，reason-first rate 则整体下降。与此同时，候选答案后的字符数由 1,989 增加至约 2,100–2,200。因此，正向 α 的典型输出模式不是更快完成，而是：

> 答案候选更早出现，但候选出现后仍继续生成较长文本。

在 `α=−6 → +8` 区间内，conditional accuracy 从 79.67% 逐步下降至 58.49%。Valid submission rate 和 multiple-marker rate 则没有相同的单调趋势，因此总体准确率变化不能简单归因于答案格式是否有效或 marker 数量。

`α=−8` 是一个不同的边界：early-candidate rate 达到 76.67%，marker position 的中位数为 0，但 conditional accuracy 只有 23.63%。其具体的答案切换与提交不稳定将在 §2.2 讨论。

#### CoT Condition

| α | First acc | Valid submission | Conditional acc | Early candidate | Reason first | Pre-candidate chars | Post-candidate chars | Marker position | Multiple markers |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −6 | 75.33% | 37.67% | 57.52% | **18.67%** | **79.45%** | 228 | 1,861 | 0.2291 | 15.33% |
| **−4** | **85.00%** | 31.67% | **81.05%** | 19.00% | 75.77% | **236** | 1,834 | 0.2732 | 11.67% |
| −2 | 74.00% | **40.67%** | 71.31% | 31.67% | 60.92% | 190 | 1,842 | 0.2878 | 13.67% |
| 0 | 69.00% | 37.67% | 67.26% | 50.00% | 46.04% | 5 | 1,862 | 0.2461 | 12.33% |
| +4 | 59.67% | 29.00% | 59.77% | **90.33%** | **13.09%** | 0 | **2,040** | 0.1952 | 13.00% |

CoT 下，`α=−4` 的准确率最高。该条件只有 19.00% 的输出较早出现答案候选，75.77% 在候选前已经出现推理文本，候选前字符数中位数为 236。

`α=−6` 的 early-candidate rate 和 reason-first rate与 `−4` 接近，但总体准确率和 conditional accuracy都更低。这说明减少提前回答、增加候选前推理通常与较好表现同时出现，却不足以单独解释 `−4` 与 `−6` 的性能差异。

从 `α=0` 移动到 `+4` 时：

- Early-candidate rate：50.00% → 90.33%
- Reason-first rate：46.04% → 13.09%
- Pre-candidate characters：5 → 0
- Post-candidate characters：1,862 → 2,040
- First accuracy：69.00% → 59.67%

因此，CoT 没有消除正向 α 下的提前回答模式。`α=+4` 更常在推理文本之前出现答案候选，并在候选出现后继续生成较长内容。

Valid submission rate 仅为 29.00%–40.67%，且没有随准确率同步变化。Marker position 和 multiple-marker rate 同样没有呈现清晰的性能曲线。相比之下，candidate ordering 更稳定地反映了 α 对可见输出顺序的影响。

总体而言，负向有效区域通常伴随更少的提前答案候选和更多候选前推理，正向 α 则呈现相反方向。但这些指标都是干预后的输出行为，只能作为相关证据，不能证明答案出现顺序是准确率变化的因果中介。

### 2.2 Two Distinct Failure Regimes

准确率曲线两端都会失效，但统一指标显示，它们对应两种不同的输出模式。正向高剂量主要表现为答案候选提前出现后仍持续生成；`α=−8` 则更常在开头正式提交答案，随后在多个候选值之间切换。

下表将统一的 candidate 指标与答案切换、marker-first 和语义重复诊断合并。所有计数的分母均为 300；同一个样本可能同时属于多个重复类型。

| α | Early candidate | Post-candidate chars | Marker at start | ≥2 answer switches | Any repetition | Self-doubt | Format fixation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **−8** | **76.67%** | **2,259** | **171** | **41** | **115** | **99** | **58** |
| **−6** | **18.67%** | 1,875 | 17 | **3** | **23** | **14** | **13** |
| −4 | 30.33% | 1,899 | 9 | 4 | 34 | 24 | 19 |
| −2 | 41.33% | 1,963 | 18 | 10 | 61 | 51 | 28 |
| 0 | 48.00% | 1,989 | 20 | 9 | 77 | 61 | 33 |
| +2 | 59.67% | 1,992 | 19 | 6 | 82 | 72 | 38 |
| +4 | 71.00% | 2,126 | 12 | 6 | 91 | 75 | 46 |
| +6 | **75.67%** | **2,172** | 19 | 12 | **94** | **85** | **54** |
| +8 | 70.67% | 2,167 | 20 | 10 | 88 | 73 | **54** |

`Marker at start` 是一个比 early candidate 更窄的指标：它只统计去除空白后直接以正式 `####` 标记开头的输出。`Self-doubt` 和 `Format fixation` 是基于生成文本的表面模式分类，可以重叠；它们表示反复推翻答案或纠结提交格式，不对应临床焦虑或强迫症诊断。

#### Positive α: Early Candidate Without Stopping

在 `α=+4/+6/+8` 下，70%以上的输出较早出现答案候选，但候选之后仍继续生成约 2,100–2,200 个字符。语义重复也由 baseline 的 77 个样本增加至 88–94 个，其中 self-doubt 和 format fixation 是主要类型。

这一模式并不是简单的“快速回答”。更准确的描述是：模型较早给出一个答案候选，但没有随之结束，而是继续检查、重算、推翻或重复提交。

正向端的 marker-at-start 只有 12–20 个样本，至少两次答案切换的样本为 6–12 个。因此，它的主要特征不是频繁在多个答案之间振荡，而是答案候选出现后仍无法及时停止。

#### Extreme Negative α: Premature Submission and Candidate Instability

`α=−8` 同样具有很高的 early-candidate rate，但其具体形态明显不同：

- 171/300 个输出直接以正式 `####` 答案开头；
- 41 个输出发生至少两次答案切换；
- Conditional accuracy 只有 23.63%；
- 候选答案之后仍继续生成 2,259 个字符；
- 115 个样本出现语义重复。

因此，`α=−8` 不是单纯的短答、拒答或动力不足。它更接近：过早正式提交一个答案，但随后无法稳定保持该候选值。

例如，一个输出可能先提交 `#### 55`，随后在正文中算出正确答案 40，之后又在 55 和 40 之间反复切换。正确候选有时已经出现，但没有被稳定保留为最终提交。

#### Near-Optimal Negative Region

`α=−6/−4` 的输出模式与两端都不同：

- Early-candidate rate 较低；
- Conditional accuracy 较高；
- 答案切换只有 3–4 个样本；
- Full-text repetition 只有 23–34 个样本。

其中 `α=−6` 的重复、答案切换和提前候选均处于低位，与其最高准确率一致。`α=−4` 的结果略弱，但仍明显优于 baseline 和正向高剂量。

总体而言，两端失效不能统一概括为“答案出现得太早”：

1. **正向高剂量**：答案候选提前出现，但之后仍持续检查和重复。
2. **极端负向剂量 `−8`**：正式答案过早提交，随后在多个候选之间振荡。
3. **`−6/−4` 区域**：候选前推理更多，答案切换和语义重复更少。

这些结论描述的是生成文本中的可观测行为，不能确定其内部计算原因，也不能证明这些行为是准确率变化的因果中介。更细的 loop-conditioned 统计、稀有重复子类型和完整案例记录保留在 `CLAUDE.md`。

### 2.3 How CoT Changes the Output Pattern

CoT 的主要作用不是单纯延长回答，而是改变输出的组织方式。为避免与 §2.1 重复，本节只比较 No-CoT 与 CoT 共有的 `α=−4、0、+4` 条件，重点观察 candidate timing、step structure、repetition 和 answer revision。每个条件均包含 300 个样本。:codex-annotation{index="1"}

| Metric | `−4`: No-CoT → CoT | `0`: No-CoT → CoT | `+4`: No-CoT → CoT |
| --- | ---: | ---: | ---: |
| First accuracy | 73.00% → **85.00%** | 60.00% → **69.00%** | 55.33% → **59.67%** |
| Early candidate rate | 30.33% → **19.00%** | 48.00% → 50.00% | 71.00% → **90.33%** |
| Reason-first rate | 32.07% → **75.77%** | 30.85% → **46.04%** | 18.44% → 13.09% |
| Pre-candidate chars, median | 0 → **236** | 0 → 5 | 0 → 0 |
| Post-candidate chars, median | 1899 → 1834 | 1989 → 1862 | 2126 → 2040 |
| Outputs with ≥2 step markers | 73 → **261** | 25 → **220** | 31 → **227** |
| Outputs with full-text repetition | 34 → **8** | 77 → **36** | 91 → **52** |
| First–last accuracy gap | +4.67 → **+0.33 pp** | +4.67 → **+0.67 pp** | +2.66 → **+0.67 pp** |

CoT 在三个剂量下都明显增加了 step structure，同时减少 full-text repetition，并缩小 first–last accuracy gap。这说明加入 CoT 后，回答通常更有组织，后续内容也较少改坏第一次提交的答案。

但 CoT 对 candidate timing 的影响取决于剂量：

- 在 `α=−4` 下，early candidate rate 从 30.33% 降至 19.00%，reason-first rate 从 32.07% 升至 75.77%，first accuracy 同时提高 12.00 pp。这是 CoT 改善输出组织最明显的条件。
- 在 `α=0` 下，CoT 增加了可见的分步推理并提高准确率，但 candidate timing 基本不变。
- 在 `α=+4` 下，虽然 step markers 明显增加、full-text repetition 减少，但 early candidate rate 反而从 71.00% 升至 90.33%。因此，表面上存在分步推理，并不代表模型避免了过早形成答案。

`α=−6` 是一个补充性的异常点：其 first accuracy 为 75.33%，last accuracy 为 78.00%。在包含有效正式答案的 113 个输出中，有 46 个（40.7%）先提交答案、再展开推理；部分样本会先给出错误的 `####` 答案，随后推导出正确结果。该现象与 `−6` 的 first accuracy 低于 last accuracy 相符，但它只是干预后的输出特征，不能单独作为准确率变化的因果解释。

结论：CoT 通常能增加 step structure、减少 repetition 并提高答案稳定性，但不能在所有剂量下阻止 candidate 过早出现。它在适中的负向剂量下最有帮助；在正向剂量下，模型仍可能先形成答案，再补充大量推理。

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

### 4.1 Performance and Dose-Dependent Output Behavior

本节统一使用 `first_acc` 作为主要性能指标，`last_acc` 用于观察后续答案修订。`early_cand_rate` 和 `reason_first_rate` 描述第一个 answer candidate 前后的可见输出顺序；`posN_med` 则表示正式答案 marker 在全文中的位置。candidate 与 marker 不是同一事件，相关指标均为干预后的描述性读数。

所有条件均包含 300 个样本。`cond_acc` 只在存在有效正式答案 marker 的样本中计算，因此用于检查输出格式和条件准确率，不替代总体 `first_acc`。

#### GSM8K

**Table 4.1. Qwen GSM8K performance and output behavior**

| Condition | α | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-CoT | −8 | 60.33% | 71.00% | 84.00% | 56.35% | 95.67% | 0.00% | 0 | 1127 | 0.4702 | 45.00% |
| No-CoT | −6 | 64.00% | 68.67% | 83.67% | 60.96% | 95.67% | 0.00% | 0 | 1098 | 0.5655 | 40.67% |
| No-CoT | −4 | 68.67% | 75.33% | 79.67% | 63.60% | 97.33% | 0.00% | 0 | 1082 | 0.6599 | 35.00% |
| No-CoT | −2 | 71.00% | 79.00% | 79.67% | 66.95% | 96.33% | 0.00% | 0 | 1146 | 0.8099 | 34.67% |
| No-CoT | 0 | 68.00% | 73.33% | 81.00% | 65.02% | 96.33% | 0.00% | 0 | 1130 | 0.8077 | 31.67% |
| No-CoT | +2 | 70.33% | 75.00% | 81.00% | 68.31% | 95.33% | 0.33% | 0 | 1099 | 0.8405 | 28.67% |
| No-CoT | +4 | 71.67% | 78.67% | 78.67% | 69.49% | 92.67% | 3.00% | 0 | 1108 | 0.8224 | 30.33% |
| No-CoT | +6 | 78.00% | 76.33% | 92.00% | 76.45% | 45.33% | 51.67% | 24 | 648 | 0.7213 | 27.67% |
| No-CoT | +8 | 86.00% | 80.33% | 97.00% | 86.25% | 5.00% | 98.00% | 140 | 452 | 0.7632 | 25.33% |
| No-CoT | +10 | 88.33% | 84.33% | 98.33% | 88.14% | 2.67% | 99.00% | 168 | 472 | 0.8024 | 24.33% |
| No-CoT | +12 | **88.67%** | 86.33% | 98.33% | 88.47% | 3.67% | 98.67% | 182 | 491 | 0.8051 | 21.67% |
| CoT | −8 | 80.33% | 79.00% | 79.00% | 78.06% | 99.00% | 0.00% | 0 | 1220 | 0.8280 | 45.33% |
| CoT | −6 | 77.67% | 79.33% | 77.33% | 75.86% | 98.67% | 0.00% | 0 | 1168 | 0.8686 | 37.00% |
| CoT | −4 | 79.00% | 79.33% | 80.33% | 78.42% | 99.67% | 0.00% | 0 | 1167 | 0.8655 | 34.33% |
| CoT | −2 | 79.00% | 81.00% | 78.67% | 77.97% | 99.33% | 0.00% | 0 | 1210 | 0.8615 | 36.67% |
| CoT | 0 | 76.33% | 76.67% | 77.67% | 75.11% | 97.33% | 1.00% | 0 | 1152 | 0.8720 | 33.33% |
| CoT | +2 | 74.33% | 75.00% | 79.67% | 73.22% | 96.33% | 1.67% | 0 | 1136 | 0.8620 | 33.67% |
| CoT | +4 | 78.33% | 81.00% | 76.67% | 79.57% | 91.67% | 6.33% | 0 | 1120 | 0.8700 | 29.67% |
| CoT | +6 | **88.33%** | **89.00%** | 93.00% | **89.61%** | 34.33% | 69.33% | 136 | 626 | 0.8458 | 30.33% |
| CoT | +8 | 86.00% | 84.00% | 100.00% | 86.00% | 7.00% | 93.33% | 173 | 450 | 0.8255 | 25.00% |

GSM8K 的主要变化集中在 `+6` 至 `+8`。在 No-CoT 下，`early_cand_rate` 从 α=0 的 96.33% 降至 `+8` 的 5.00%，`reason_first_rate` 从 0.00% 升至 98.00%，同时 `first_acc` 从 68.00% 提高到 86.00%。继续增加至 `+10/+12` 后，准确率只小幅提高至 88.33%–88.67%，说明当前观察到的是上升后的高剂量平台，而不是明确的单点峰值。

原九档 No-CoT 检验中，`+6`（Holm `p_adj=.016`）和 `+8`（`p_adj<1e−4`）显著高于 α=0；其余剂量未通过校正。`+10/+12` 是观察原九档结果后追加的探索性剂量，不属于该 Holm family，因此主要用于判断平台是否延续。

CoT 下也在 `+6/+8` 出现明显的 candidate-ordering transition。`+6` 的 `first_acc` 为 88.33%，`+8` 为 86.00%，二者的探索性配对比较不显著（`p=.371`），因此更适合将其描述为 `{+6,+8}` 的近优区域。相对 α=0，`+6`（Holm `p_adj=.0002`）和 `+8`（`p_adj=.0030`）均显著提高准确率。

需要注意的是，`posN_med` 与 candidate 指标并不完全同步。例如，No-CoT 从 α=0 到 `+8` 时，`early_cand_rate` 大幅下降，但正式 marker 的位置只从 0.8077 变为 0.7632。这说明正向 α 主要改变的是第一个 answer candidate 之前的输出顺序，不能仅凭正式 marker 的位置判断答案何时形成。

#### MATH

**Table 4.2. Qwen MATH performance and output behavior**

| Condition | α | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-CoT | −8 | 54.00% | 50.00% | 98.00% | 55.10% | 85.00% | 0.67% | 0 | 1516 | 0.9772 | 21.00% |
| No-CoT | −6 | 57.00% | 54.00% | 98.00% | 56.80% | 79.33% | 0.33% | 0 | 1573 | 0.9778 | 22.00% |
| No-CoT | −4 | 58.00% | 55.00% | 98.33% | 57.97% | 74.33% | 1.00% | 0 | 1524 | 0.9776 | 17.00% |
| No-CoT | −2 | 59.00% | 55.33% | 98.33% | 59.32% | 71.33% | 5.03% | 0 | 1478 | 0.9771 | 17.67% |
| No-CoT | 0 | 60.67% | 60.00% | 98.00% | 60.88% | 67.33% | 11.11% | 0 | 1438 | 0.9770 | 16.67% |
| No-CoT | +2 | 60.00% | 58.67% | 97.33% | 61.30% | 59.67% | 21.96% | 0 | 1334 | 0.9767 | 14.33% |
| No-CoT | +4 | 63.33% | 61.33% | 98.67% | 63.18% | 46.33% | 36.64% | 0 | 1258 | 0.9749 | 14.00% |
| No-CoT | +6 | **68.33%** | **67.67%** | 98.33% | **69.15%** | 19.33% | 77.59% | 224 | 860 | 0.9691 | 14.00% |
| No-CoT | +8 | 63.33% | 63.67% | 99.67% | 63.21% | 10.33% | 86.16% | 230 | 727 | 0.9602 | 16.67% |
| CoT | −8 | 59.67% | 57.67% | 99.33% | 60.07% | 86.67% | 1.00% | 0 | 1578 | 0.9646 | 34.00% |
| CoT | −6 | 59.33% | 57.33% | 99.00% | 59.93% | 80.00% | 0.67% | 0 | 1633 | 0.9658 | 34.67% |
| CoT | −4 | 65.00% | 61.67% | 100.00% | 65.00% | 76.67% | 0.33% | 0 | 1570 | 0.9635 | 32.00% |
| CoT | −2 | 63.00% | 58.67% | 99.33% | 63.42% | 74.00% | 1.35% | 0 | 1628 | 0.9659 | 30.67% |
| CoT | 0 | 63.00% | 57.67% | 99.00% | 63.64% | 71.33% | 5.05% | 0 | 1567 | 0.9695 | 28.33% |
| CoT | +2 | 62.00% | 58.33% | 99.67% | 62.21% | 69.00% | 12.50% | 0 | 1486 | 0.9700 | 25.00% |
| CoT | +4 | 63.00% | 59.00% | 98.67% | 63.85% | 58.33% | 24.83% | 0 | 1393 | 0.9727 | 21.00% |
| CoT | +6 | **66.00%** | **65.67%** | 99.67% | **66.22%** | 29.67% | 62.50% | 164 | 970 | 0.9677 | 22.67% |
| CoT | +8 | 64.00% | 64.33% | 99.67% | 64.21% | 10.33% | 85.96% | 252 | 816 | 0.9640 | 27.67% |

MATH 也表现出清晰的 output-ordering transition。No-CoT 的 `early_cand_rate` 从 α=0 的 67.33% 降至 `+6` 的 19.33% 和 `+8` 的 10.33%，`reason_first_rate` 则从 11.11% 升至 77.59% 和 86.16%。但是，`first_acc` 在 `+6` 达到 68.33% 后，于 `+8` 回落至 63.33%。因此，减少 early candidate 并不保证准确率持续提高。

No-CoT 中只有 `+6` 相对 α=0 通过 Holm 校正（`p_adj=.0087`）；`+6` 与 `+8` 的探索性配对比较为 `p=.040`。这支持一个描述性的右臂：适度推迟 candidate 与准确率提升同时出现，但进一步增加剂量后，输出顺序继续变化，性能却开始下降。

CoT 下呈现相似但更弱的曲线：`first_acc` 从 α=0 的 63.00% 升至 `+6` 的 66.00%，随后在 `+8` 降至 64.00%。各剂量相对 α=0 的比较均为 Holm `p_adj=1.000`。逐剂量比较 CoT 与 No-CoT 时，也没有结果通过校正；最大差异出现在 `α=−4`（+7.00 pp，raw `p=.0065`，Holm `p_adj=.0581`）。

MATH 的 `valid_sub_rate` 始终约为 97%–100%，所以 `cond_acc` 与总体准确率接近。与此同时，`posN_med` 在所有条件下都接近 1，说明正式的 `\boxed{}` marker 几乎总在输出末尾。相较之下，candidate-based 指标能够更清楚地显示剂量变化，因此 MATH 的 answer ordering 应主要依据 `early_cand_rate`、`reason_first_rate` 和 `pre_cand_chars_med` 判断。

**Conclusion.** Qwen 在 GSM8K 与 MATH 上都出现了由正向 α 驱动的 output-ordering transition，但性能曲线不同：GSM8K 在高剂量进入平台，MATH 则在 `+6` 后出现回落。由此可见，推迟 answer candidate 与增加 reason-first output 可以伴随准确率提升，但不能单独保证更好的任务表现。

**Table 4.3. Output reordering from baseline to the main workpoint**

| Task | Condition | Dose comparison | first_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | No-CoT | `0 → +8` | 68.00% → **86.00%** | 96.33% → **5.00%** | 0.00% → **98.00%** | 0 → 140 | 1130 → 452 |
| GSM8K | CoT | `0 → +6` | 76.33% → **88.33%** | 97.33% → **34.33%** | 1.00% → **69.33%** | 0 → 136 | 1152 → 626 |
| MATH | No-CoT | `0 → +6 → +8` | 60.67% → **68.33%** → 63.33% | 67.33% → 19.33% → **10.33%** | 11.11% → 77.59% → **86.16%** | 0 → 224 → 230 | 1438 → 860 → 727 |
| MATH | CoT | `0 → +6 → +8` | 63.00% → **66.00%** → 64.00% | 71.33% → 29.67% → **10.33%** | 5.05% → 62.50% → **85.96%** | 0 → 164 → 252 | 1567 → 970 → 816 |

#### Commitment Reordering across Tasks

GSM8K 与 MATH 都出现了明显的 output reordering：随着正向 α 增加，`early_cand_rate` 下降，`reason_first_rate` 上升，更多可见推理被移到第一个 candidate 之前。

但这种变化与准确率的关系因任务而异。GSM8K 的 reordering 与性能提升同时出现，并在 `+6/+8` 附近进入近优平台；MATH 的准确率在 `+6` 达到较高点后回落，而 candidate ordering 在 `+8` 仍继续变化。这说明减少 early candidate 可能与更好的输出状态相关，但不是提高准确率的充分条件。

MATH 的 `\boxed{}` 通常位于输出末尾，因此 `posN_med` 难以区分答案形成顺序。本节主要依据 candidate-based metrics 描述 output ordering。所有指标均为干预后的输出读数，不能作为因果中介证据。

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

下面内容可直接替换现有第 5、6 节。

---

## 5. Commitment-Based Prediction and Workpoint Selection

本节检验两个问题：

1. Commitment behavior 能否预测未见题目的正确率？
2. 冻结的 commitment predictor 能否在新的剂量曲线上找到较好的 steering workpoint？

这里的 workpoint selection 是根据目标任务的多个剂量选择 α。下一节则汇总所有跨任务结果，并区分“直接沿用 GSM8K 工作点”与“在目标任务上重新扫描剂量”两种设计。

训练、特征编码、数据清单、统计检验和产物校验记录于 `CLAUDE.md`。主要准确率指标为离线重新计算的 `first_acc`。

### 5.1 Held-Out Correctness Prediction on GSM8K

Predictor 使用 early candidate、commit state、标准化 commit position（`posN`）及其可观测性预测每道题是否正确。Raw α 不作为输入特征。同一道题的所有剂量始终位于同一个交叉验证 fold，避免同题信息泄漏。

**Table 5.1. Held-out correctness prediction on GSM8K**

| Model | Commitment-only AUROC | Entry-only AUROC | Commitment − Entry | Combined − Commitment | Calibration slope |
|---|---:|---:|---:|---:|---:|
| Llama3.1-8B | **.687** [.656, .719] | .548 [.526, .571] | **+.139** [+.104, +.172] | −.001 [−.004, +.002] | .95 |
| Qwen2.5-7B | **.749** [.710, .787] | .628 [.601, .654] | **+.121** [+.084, +.156] | +.002 [−.002, +.007] | .98 |

Commitment features 在两个模型上都能预测未见 GSM8K 题目的正确率，并且明显优于只使用 entry gain。在 commitment features 基础上加入 entry gain，没有带来可检测的额外提升。

这说明答案形成和提交行为包含与正确率有关的信息，但不证明这些行为造成了正确率变化，也不证明 entry gain 没有机制作用。

### 5.2 Cross-Task Workpoint Selection

冻结的 GSM8K predictor 随后应用于 MATH 和 GSM-Hard。预测分数只用于排列剂量，不用于估计新任务的绝对准确率。

**Table 5.2. Commitment-based workpoint selection across tasks**

| Evaluation | Model | Available curve | Predicted direction | Spearman ρ | Selected α | Observed best α | Near-optimal region | Regret |
|---|---|---|---|---:|---:|---:|---|---:|
| MATH, retrospective | Llama | Original 3 doses | Negative | +1.000 | −4 | −4 | {−4, 0} | 0.00 pp |
| MATH, retrospective | Qwen | 9 doses | Positive | **+.962** | **+6** | +6 | {+4, +6} | **0.00 pp** |
| GSM-Hard, prospective blind | Llama | Frozen No-CoT curve | Negative | **+1.000** | **−6** | −6 | {−6, −4} | **0.00 pp** |
| GSM-Hard, prospective blind | Qwen | Frozen No-CoT curve | Positive | +.600 | **+8** | +8 | {+8} | **0.00 pp** |

MATH 是规则冻结后的回顾性迁移。Qwen 的完整曲线提供了较强的排序检验：predictor 正确选中 `+6`，也识别出 `+8` 的准确率回落。Llama 当时只有 `−4/0/+4`，因此只能证明局部方向正确，不能回填成 predictor 已经在完整曲线上选中后来补测的 `−6`。

GSM-Hard 是真正的前瞻性盲测。Predictor 在查看 accuracy 之前选中 Llama `−6` 和 Qwen `+8`，两个选择的 regret 均为零。不过，Llama 的 `−6/−4` 预测分数和实际准确率都非常接近，因此更准确的说法是 predictor 找到了近优区域，而不是精确区分了唯一最佳点。

两个任务上的 predicted score 都系统性高于实际准确率，说明迁移的是剂量排序，而不是绝对概率校准。

### 5.3 Prospective GSM-Hard Dose Curves

**Table 5.3. Predicted and observed GSM-Hard No-CoT curves**

| Model | Metric | −8 | −6 | −4 | 0 | +4 | +6 | +8 | +10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | Predicted score | .5554 | **.68834** | .68828 | .6303 | .5770 | — | — | — |
|  | Observed `first_acc` | .1100 | **.2433** | .2400 | .1800 | .1700 | — | — | — |
| Qwen2.5-7B | Predicted score | — | — | .6959 | .7038 | .6794 | .7182 | **.8552** | .8547 |
|  | Observed `first_acc` | — | — | .3433 | .3400 | .3467 | .4033 | **.5033** | .5033 |

原始盲测不包含 Qwen `+10`；该邻点为后补结果。加入后，predictor 仍选择 `+8`，实际 `+8/+10` 并列，因此 regret 保持为零。

两条实际剂量曲线均可区分：Llama 的最小 Holm-adjusted p 为 `2.29×10⁻⁷`，Qwen 为 `1.35×10⁻⁷`。不过，predictor 对曲线中每一个局部排序并不完全准确，因此其主要价值是判断方向并找到低-regret 区域。

### 5.4 Post-Hoc Extensions and Boundary Checks

下表合并后续增加的条件迁移、邻点和边界检查。这些结果均发生在主要分析之后，不能作为新的盲测证据。

**Table 5.4. Post-hoc predictor checks**

| Dataset and condition | Model | α | Predicted score | Observed accuracy | Status |
|---|---|---:|---:|---:|---|
| GSM8K CoT | Llama | −6 | .7202 | 75.33% | Added later |
| GSM8K CoT | Llama | −4 | **.7283** | **85.00%** | Observed peak |
| GSM8K CoT | Llama | −2 | .6887 | 74.00% | Added later |
| GSM8K CoT | Llama | 0 | .6282 | 69.00% | Baseline |
| GSM8K CoT | Llama | +4 | .5110 | 59.67% | Added later |
| MATH No-CoT | Llama | −8 | .7137 | 39.33% | Added later |
| MATH No-CoT | Llama | −6 | **.7421** | **43.33%** | Added later; observed best |
| MATH No-CoT | Llama | −4 | .7143 | 40.00% | Original |
| MATH No-CoT | Llama | 0 | .6605 | 36.67% | Original baseline |
| MATH No-CoT | Llama | +4 | .5889 | 33.00% | Original |
| GSM-Hard CoT | Llama | −4 | **.7183** | **30.00%** | Added later |
| GSM-Hard CoT | Llama | −6 | .6797 | 26.00% | Frozen workpoint |
| GSM-Hard CoT | Llama | 0 | .6454 | 20.00% | Baseline |
| GSM-Hard CoT | Qwen | 0 | .7860 | 38.00% | Baseline |
| GSM-Hard CoT | Qwen | +6 | .8387 | 49.00% | Added later |
| GSM-Hard CoT | Qwen | +8 | .8830 | **51.33%** | Frozen workpoint; observed best |
| GSM-Hard CoT | Qwen | +10 | **.8863** | 50.33% | Added later |
| BBH No-CoT | Llama | −6 | **.5143** | 40.80% | Predictor-selected |
| BBH No-CoT | Llama | 0 | .4816 | **41.60%** | Observed best |
| BBH No-CoT | Llama | +4 | .4701 | 32.80% | Reverse diagnostic |
| BBH No-CoT | Qwen | −6 | .7132 | 56.80% | Reverse diagnostic |
| BBH No-CoT | Qwen | 0 | **.7331** | 55.20% | Predictor-selected |
| BBH No-CoT | Qwen | +8 | .7267 | **57.60%** | Observed best |

GSM8K CoT 的预测排序与观察排序一致，均为 `−4 > −6 > −2 > 0 > +4`。但这些特征来自不同生成批次，因此这是跨批次的条件迁移压力测试，不是新的 held-out correctness 验证。

Llama MATH 补充 `−8/−6` 后，predictor 在五点曲线上选中 `−6`，与 observed best 一致。由于这两格是在原始分析后增加的，不能回填成 predictor 当时已经完成了这一选择。

GSM-Hard CoT 下，Llama predictor 选中 `−4`，与 observed best 一致；Qwen predictor 选中 `+10`，而 observed best 为 `+8`，regret 为 1.00 pp。由于 `{+6,+8,+10}` 均属于观察到的近优区域，这一偏差属于区间内误差。

BBH 是重要的边界案例：两个模型的 predictor 都没有命中 observed argmax，但各剂量之间的准确率差异本身未被显著区分。因此，这不是强预测失败，而是表明：当目标任务不存在可检测的 steering 效果时，predictor 的剂量排序也缺乏明确的验证信号。

### 5.5 Supporting Answer-Formation Evidence

Predictor 的有效性与答案形成位置的变化相一致。下表只比较两个剂量下都能定位答案候选的共同题目。

**Table 5.5. Candidate-based answer-formation timing**

| Task and condition | Dose comparison | Accuracy | Shared n | Candidate position | Pre-candidate chars | Reason-first |
|---|---|---:|---:|---:|---:|---:|
| GSM8K No-CoT | 0 → **−6** | .6000 → **.7800** | 281 | .0000 → **.0843** | 0 → **175** | 29.2% → **66.5%** |
| GSM8K CoT | 0 → **−4** | .6900 → **.8500** | 252 | .0021 → **.1117** | 5 → **234.5** | 46.0% → **76.2%** |
| GSM-Hard No-CoT | 0 → **−6** | .1800 → **.2433** | 267 | .0000 → **.0937** | 0 → **201** | 25.8% → **61.0%** |
| GSM-Hard CoT | 0 → **−6** | .2000 → **.2600** | 248 | .0000 → **.1012** | 0 → **231.5** | 40.3% → **63.7%** |

四组的 reason-first 比例均明显提高（McNemar `p<1×10⁻⁸`）。较好的工作点通常伴随更多候选前推理，以及更晚出现的答案候选。

完整剂量曲线也显示相同关联：

**Table 5.6. Association between accuracy and answer timing**

| Model | Dose curve | `accuracy ~ posN` | `accuracy ~ early-candidate%` |
|---|---|---:|---:|
| Llama3.1-8B | 9 doses | ρ=**+.941**, `p=.0002` | — |
| Qwen2.5-7B | 11 doses | ρ=**+.863**, `p=.0006` | ρ=**−.804**, `p=.0029` |

Llama 的九档曲线没有与 Qwen 完全相同的 frozen early-candidate 指标，因此该格保留为空，不能用历史指标替代。

这些结果不能解释为“答案越晚越好”。Qwen 在 `+8` 后准确率已经进入平台，但 `posN` 仍由 `.754` 上升至 `.802`。更准确的说法是，较好的工作点通常使模型摆脱过早回答；进入稳定区域后，继续推迟答案不会持续提高准确率。

所有 timing 指标都是 α 干预后的输出结果，因此只能作为关联证据，不构成因果中介证明。

### 5.6 Conclusion

Commitment features 能预测 GSM8K 未见题目的正确率，也能为 MATH 和 GSM-Hard 提供有用的剂量排序。MATH 是回顾性验证，GSM-Hard 则是前瞻性盲测。

总体而言，predictor 更适合判断 steering 方向并找到低-regret 的近优区域，而不是精确命中唯一 argmax，也不能直接预测新任务的绝对准确率。当目标任务本身没有可检测的剂量效应时，predictor 的排序也缺乏明确的验证依据。

---

## 6. Cross-Task Workpoint Evaluation: Transfer, Dose Sweeps, and Boundaries

本节汇总 steering workpoint 在不同任务上的表现。实验分为两类：

1. **Fixed-workpoint transfer**：直接使用 GSM8K 冻结的工作点，不在目标任务上重新选择剂量。
2. **Task-specific dose sweep**：在目标任务上测试多个剂量，用于寻找该任务自己的 workpoint 或失败边界。

固定迁移点为：

- Llama3.1-8B：`α=−6`
- Qwen2.5-7B：`α=+8`

两种模型使用不同的 steering mask、层范围和激活尺度，因此 raw α 不能作为跨模型共同剂量比较。

### 6.1 Fixed-Workpoint Transfer Across Tasks

**Table 6.1. GSM8K-derived fixed-workpoint transfer**
原表只覆盖严格的 fixed-workpoint transfer。下面补入 GSM-Symbolic、ProofWriter、ZebraLogic 和 FinQA，并增加 `Evaluation type`，避免把完整剂量扫描误写成预先冻结的迁移检验。

| Task | Evaluation type | Condition | Llama `−6` | Qwen `+8` | Verdict |
|---|---|---|---|---|---|
| MATH | Fixed transfer | No-CoT | 36.67% → 43.33%<br>**Δ=+6.67 pp**, `p_adj=.0489`<br>CI=[+1.00,+12.33] | 60.67% → 63.33%<br>Δ=+2.67 pp, `p_adj=.3581`<br>CI=[−2.33,+7.67] | Llama only |
| MATH | Fixed transfer | CoT | 42.00% → 49.00%<br>**Δ=+7.00 pp**, `p_adj=.0225` | 63.00% → 64.00%<br>Δ=+1.00 pp, `p_adj=1.000` | Llama only |
| GSM-Hard | Fixed transfer | No-CoT | 18.00% → 24.33%<br>**Δ=+6.33 pp**, raw `p=.00661` | 34.00% → 50.33%<br>**Δ=+16.33 pp**, raw `p=1.41×10⁻⁸` | Both models |
| GSM-Hard | Fixed transfer | CoT | 20.00% → 26.00%<br>**Δ=+6.00 pp**, `p_adj=.00393`<br>CI=[+2.33,+10.00] | 38.00% → 51.33%<br>**Δ=+13.33 pp**, `p_adj=9.42×10⁻⁶`<br>CI=[+8.00,+19.00] | Both models |
| BBH object counting | Fixed transfer | No-CoT | 41.60% → 40.80%<br>Δ=−0.80 pp, `p_adj=1.000` | 55.20% → 57.60%<br>Δ=+2.40 pp, `p_adj=1.000` | Neither |
| BBH object counting | Fixed transfer | CoT | 40.80% → 56.80%<br>**Δ=+16.00 pp**, `p_adj=2.25×10⁻⁴`<br>CI=[+8.80,+23.20] | 52.80% → 66.80%<br>**Δ=+14.00 pp**, `p_adj=2.25×10⁻⁴`<br>CI=[+7.60,+20.40] | Both models |
| CRUXEval-O | Fixed transfer | No-CoT | 34.67% → 31.00%<br>Δ=−3.67 pp, `p_adj=.1352` | 29.33% → 37.67%<br>**Δ=+8.33 pp**, `p_adj=.0045` | Qwen only |
| CRUXEval-O | Fixed transfer | CoT | 34.67% → 34.00%<br>Δ=−0.67 pp, `p_adj=.9656`<br>CI=[−5.00,+3.67] | 34.67% → 54.00%<br>**Δ=+19.33 pp**, `p_adj=2.63×10⁻⁹`<br>CI=[+13.67,+25.00] | Qwen only |
| CRUXEval-O | Task-specific sweep / Chat | No-CoT / CoT | No-CoT: 45.67% → 46.67% (`−6`), Δ=+1.00 pp, `p_adj=.7111`; CoT: 52.00% → 51.33% (`−6`), Δ=−0.67 pp, `p_adj=1.000` | — | Llama: neither condition |
| LogiQA 2.0 | Fixed transfer | No-CoT | 56.33% → 52.00%<br>Δ=−4.33 pp, `p_adj=.107` | 64.00% → 65.00%<br>Δ=+1.00 pp, `p_adj=.801` | Neither |
| LogiQA 2.0 | Fixed transfer | CoT | 46.33% → 44.00%<br>Δ=−2.33 pp, `p_adj=.9656`<br>CI=[−8.00,+3.33] | 66.33% → 61.00%<br>Δ=−5.33 pp, `p_adj=.1677`<br>CI=[−10.67,−0.33] | Neither |
| GSM-Symbolic | Task-specific sweep / same-family robustness | No-CoT | 48.11% → 57.56%<br>**Δ=+9.44 pp**, `p_adj=.0003`<br>CI=[+5.78,+13.11] | 53.33% → 66.89%<br>**Δ=+13.56 pp**, `p_adj=.0003`<br>CI=[+8.44,+18.56] | Both models |
| GSM-Symbolic | Task-specific sweep / same-family robustness | CoT | 55.56% → 56.67%<br>Δ=+1.11 pp, `p_adj=.527`<br>CI=[−2.44,+4.56] | 52.89% → 65.00%<br>**Δ=+12.11 pp**, `p_adj=.0003`<br>CI=[+7.11,+17.11] | Qwen only |
| ProofWriter-OWA | Task-specific sweep | CoT, Bare | 10.33% → 14.33%<br>Δ=+4.00 pp, `p_adj=.3100`<br>CI=[−1.00,+9.00] | 46.33% → 52.00%<br>Δ=+5.67 pp, `p_adj=.2571`<br>CI=[−0.33,+11.67] | Neither at `−6/+8` |
| ProofWriter-OWA | Task-specific sweep | CoT, Chat | 33.00% → 39.33%<br>Δ=+6.33 pp, `p_adj=.2441`<br>CI=[−0.33,+13.00] | 41.00% → 47.67%<br>**Δ=+6.67 pp**, `p_adj=.0303`<br>CI=[+1.67,+11.67] | Qwen only |
| ZebraLogic-Easy | Task-specific sweep | Task prompt | 36.79% → 35.71%<br>Δ=−1.07 pp, `p_adj=.749` | 34.64% → 23.93%<br>**Δ=−10.71 pp**, `p_adj=.0004` | No positive effect; Qwen `+8` harmful |
| FinQA | Task-specific sweep | CoT | 14.33% → 8.67%<br>**Δ=−5.67 pp**, `p_adj=.0190` | 20.67% → 25.67%<br>Δ=+5.00 pp, `p_adj=.1539` | No positive effect; Llama `−6` harmful |

> **Reading note.** `Fixed transfer` 表示 `−6/+8` 在查看目标任务结果前已经由 GSM8K 冻结。`Task-specific sweep` 表示目标任务测试了完整剂量曲线；表中这里只抽取其中的 `−6/+8` 方便横向比较，不能将这些行重新解释为预先注册的 fixed-workpoint transfer。ProofWriter Bare 中另有 Llama `+4` 的显著结果，但该提升主要伴随有效答案提交增加；ProofWriter Chat 中只有 Qwen `+8` 建立了显著正向 workpoint。CRUXEval-O Chat 行是 Llama-only 的目标任务四点扫描；Qwen 没有运行 Chat 对照，因此不能把空缺解释为 null。

### 6.2 Task-Specific Dose Sweeps

以下实验在目标任务上测试多个剂量。它们可以识别任务自己的 workpoint 或失败边界，但不属于严格的 fixed-workpoint transfer。

#### 6.2.1 GSM-Symbolic

GSM-Symbolic 在 `main`、`p1`、`p2` 各使用 300 题，并沿用 GSM8K 的 first-marker/fallback 评分。它是 GSM8K 同任务家族中的扰动鲁棒性检查，不是独立的跨领域迁移验证。

**Table 6.2. GSM-Symbolic No-CoT and CoT dose sweeps**

每个 config（`main`/`p1`/`p2`）使用 300 个实例；Main/P1/P2 三列各自的准确率。正式推断（Primary Δ / 95% CI / Holm `p_adj`）以 `original_id` 为 cluster 做 paired cluster bootstrap，对 `main`/`p1`/`p2` 三个 config 等权（不按行数加权），并在每个模型、每种 CoT 条件内对三个非零剂量分别执行 Holm `m=3` 校正。

| Condition | Model | α | Main | P1 | P2 | Primary Δ | 95% CI | Holm `p_adj` | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **No-CoT** | Llama3.1-8B | 0 | 57.33% | 53.33% | 33.67% | — | — | — | baseline |
| **No-CoT** | Llama3.1-8B | −6 | 71.00% | 60.67% | 41.00% | **+9.44 pp** | [+5.78,+13.11] | **.0003** | **positive** |
| **No-CoT** | Llama3.1-8B | −4 | 62.67% | 56.67% | 37.67% | **+4.22 pp** | [+0.78,+7.78] | **.0164** | **positive** |
| **No-CoT** | Llama3.1-8B | +4 | 50.67% | 39.33% | 26.67% | **−9.22 pp** | [−13.00,−5.56] | **.0003** | significant degradation |
| **No-CoT** | Qwen2.5-7B | 0 | 65.33% | 56.33% | 38.33% | — | — | — | baseline |
| **No-CoT** | Qwen2.5-7B | −6 | 59.33% | 54.67% | 36.67% | −3.11 pp | [−6.56,+0.44] | .0802 | not significant |
| **No-CoT** | Qwen2.5-7B | +6 | 73.00% | 64.00% | 45.00% | **+7.33 pp** | [+3.56,+11.33] | **.0003** | **positive** |
| **No-CoT** | Qwen2.5-7B | +8 | 78.00% | 71.67% | 51.00% | **+13.56 pp** | [+8.44,+18.56] | **.0003** | **positive** |
| CoT | Llama3.1-8B | 0 | 62.67% | 58.67% | 45.33% | — | — | — | baseline |
| CoT | Llama3.1-8B | −6 | 64.00% | 59.00% | 47.00% | +1.11 pp | [−2.44,+4.56] | .527 | not significant |
| CoT | Llama3.1-8B | −4 | 70.67% | 63.67% | 41.33% | +3.00 pp | [−0.11,+6.22] | .122 | not significant |
| CoT | Llama3.1-8B | +4 | 54.00% | 41.33% | 25.33% | **−15.33 pp** | [−20.00,−10.90] | **.0003** | significant degradation |
| CoT | Qwen2.5-7B | 0 | 63.33% | 56.33% | 39.00% | — | — | — | baseline |
| CoT | Qwen2.5-7B | −6 | 66.00% | 64.33% | 39.00% | **+3.56 pp** | [+0.11,+6.89] | **.044** | **positive** |
| CoT | Qwen2.5-7B | +6 | 79.33% | 68.33% | 47.67% | **+12.22 pp** | [+8.11,+16.33] | **.0003** | **positive** |
| CoT | Qwen2.5-7B | +8 | 81.67% | 66.67% | 46.67% | **+12.11 pp** | [+7.11,+17.11] | **.0003** | **positive** |

**No-CoT 结果已修复 sample_id 冲突后重新计算。** 早期版本的 No-CoT 统计使用了冲突的 `sample_id`（`"{config}:{id}"`，其中官方 `id` 字段实为 cluster 级别的 `original_id`，导致每个 300 题的 config 在按 `sample_id` 去重时只保留了 100（`main`/`p1`）或 50（`p2`）行），据此计算的准确率、cluster、CI 和 Holm 判定均已作废。唯一有效的 `sample_id` 为 `{config}:{original_id}:{instance}`；本表 No-CoT 部分为修复后的正式结果。CoT 部分自始至终使用正确的复合 ID，数字未变。

No-CoT 下，Llama `−6` 和 `−4` 均显著提高准确率（`−6`：+9.44 pp；`−4`：+4.22 pp），`+4` 显著降低准确率；Qwen `+6` 和 `+8` 均显著提高准确率（`+6`：+7.33 pp；`+8`：+13.56 pp），`−6` 未达到显著。

CoT 下，Qwen 的 `−6/+6/+8` 三个非零剂量均显著提高准确率；Llama 没有任何正向显著结果（`−6/−4` 均未显著，`+4` 显著降低表现）。说明同一任务家族内的迁移仍然依赖模型与提示条件。

#### 6.2.2 ProofWriter-OWA

ProofWriter-OWA 使用显式 CoT、固定的单个 Unknown 示例和 first-answer 评分。早期 Bare 条件存在严重的循环、截断和答案提交问题，因此后续增加 Chat 条件检查这些结果是否主要来自接口失效。

Bare 与 Chat 使用相同题目、评分规则和剂量，仅生成接口不同。两者均属于目标任务上的剂量扫描，不是 GSM8K fixed-workpoint transfer。

**Table 6.3. ProofWriter-OWA Bare and Chat dose sweeps (N=300 per cell)**

| Interface | Model | α | Accuracy | Last-answer | Answered-only | No answer | Multiple markers | First ≠ last | Loop | Truncation | Δ vs 0 | Holm `p_adj` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bare | Llama3.1-8B | −6 | .1433 | .1033 | .5513 (n=78) | .740 | .2167 | .615 (n=65) | .933 | 1.000 | +4.00 pp | n.s. |
| Bare | Llama3.1-8B | −4 | .1200 | .0867 | .5373 (n=67) | .777 | .1900 | .509 (n=57) | .900 | 1.000 | +1.67 pp | n.s. |
| Bare | Llama3.1-8B | 0 | .1033 | .0600 | .5000 (n=62) | .793 | .1767 | .396 (n=53) | .933 | 1.000 | — | — |
| Bare | Llama3.1-8B | +4 | **.2167** | .1867 | .5462 (n=119) | .603 | .3533 | .406 (n=106) | .950 | 1.000 | **+11.33 pp** | **2.27×10⁻⁴** |
| Chat | Llama3.1-8B | −6 | **.3933** | .3933 | .5339 (n=221) | .263 | .000 | N/A | .170 | .210 | +6.33 pp | .244 |
| Chat | Llama3.1-8B | −4 | .3400 | .3400 | .5730 (n=178) | .407 | .000 | N/A | .267 | .297 | +1.00 pp | 1.000 |
| Chat | Llama3.1-8B | 0 | .3300 | .3300 | .5470 (n=181) | .397 | .000 | N/A | .257 | .297 | — | — |
| Chat | Llama3.1-8B | +4 | .3167 | .3167 | .5723 (n=166) | .447 | .000 | N/A | .267 | .290 | −1.33 pp | 1.000 |
| Bare | Qwen2.5-7B | −6 | .4933 | .4900 | .5461 (n=271) | .097 | .1967 | .017 (n=59) | .143 | .147 | +3.00 pp | n.s. |
| Bare | Qwen2.5-7B | 0 | .4633 | .4633 | .4649 (n=299) | .003 | .3467 | .000 (n=104) | .343 | .343 | — | — |
| Bare | Qwen2.5-7B | +6 | .4967 | .4967 | .4967 (n=300) | .000 | .3200 | .000 (n=96) | .320 | .320 | +3.34 pp | n.s. |
| Bare | Qwen2.5-7B | +8 | **.5200** | .5200 | .5200 (n=300) | .000 | .4867 | .027 (n=146) | .473 | .473 | +5.67 pp | .257 |
| Chat | Qwen2.5-7B | −6 | .0067 | .0067 | .6667 (n=3) | .990 | .000 | N/A | .000 | .000 | **−40.33 pp** | **2.26×10⁻³⁶** |
| Chat | Qwen2.5-7B | 0 | .4100 | .4100 | .4100 (n=300) | .000 | .007 | .000 (n=2) | .000 | .000 | — | — |
| Chat | Qwen2.5-7B | +6 | .3900 | .3900 | .3913 (n=299) | .003 | .000 | N/A | .000 | .000 | −2.00 pp | 1.000 |
| Chat | Qwen2.5-7B | +8 | **.4767** | .4767 | .4783 (n=299) | .003 | .000 | N/A | .000 | .000 | **+6.67 pp** | **.0303** |

Chat template 大幅减少了循环、截断和多答案问题，确认 Bare 条件下的部分低分来自接口与答案提交失败。

在更稳定的 Chat 条件下，Llama 的 `−6` 数值最高，但未通过 Holm，因此只能视为方向一致的趋势。Qwen 的 `+8` 显著提高准确率，而 `−6` 几乎完全退化为不带规定 marker 的裸标签输出。

Bare 条件下 Llama `+4` 虽然显著，但其提升伴随无答案率明显下降，因此更适合解释为有效提交增加，而不是已经证明推理能力改善。总体而言，ProofWriter 支持 steering 效果，但该效果高度依赖模型和提示接口。

#### 6.2.3 ZebraLogic-Easy

ZebraLogic-Easy 使用 280 题和 first-answer JSON 主评分。缺少完整答案的样本计错，并在每个模型内对三个非零剂量执行 Holm 校正。

**Table 6.4. ZebraLogic-Easy task-specific dose sweep**

| Model | α | Puzzle accuracy | Δ vs 0 | No answer | Raw p | Holm `p_adj` |
|---|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | 0 | **36.79%** | — | 25.4% | — | — |
| Llama3.1-8B | −6 | 35.71% | −1.07 pp | 24.3% | .749 | .749 |
| Llama3.1-8B | −4 | 33.93% | −2.86 pp | 25.0% | .256 | .512 |
| Llama3.1-8B | +4 | 32.14% | −4.64 pp | 25.0% | .066 | .198 |
| Qwen2.5-7B | 0 | 34.64% | — | 0.0% | — | — |
| Qwen2.5-7B | −6 | 30.00% | −4.64 pp | 0.0% | .079 | .158 |
| Qwen2.5-7B | +6 | **36.07%** | +1.43 pp | 0.4% | .678 | .678 |
| Qwen2.5-7B | +8 | 23.93% | **−10.71 pp** | **60.7%** | **1.3×10⁻⁴** | **4.0×10⁻⁴** |

两个模型均未检测到显著的正向 workpoint。Llama 的所有非零剂量均未改善准确率；Qwen `+6` 只有很小的正向点估计，而 `+8` 显著降低表现并使无答案率升至 60.7%。

因此，ZebraLogic-Easy 没有提供正向迁移证据，但明确显示了 Qwen 的高剂量失败边界。

#### 6.2.4 FinQA

FinQA 使用显式 CoT 和直接数字答案评分，在每个模型内对三个非零剂量执行 Holm 校正。这里报告的是自定义的数字答案准确率，不等同于官方 FinQA program/DSL execution 指标。

**Table 6.5. FinQA task-specific dose sweep (N=300 per cell)**

| Model | α | First accuracy | Last accuracy | Answered-only | No answer | Loop | Truncation | Δ vs 0 | Holm `p_adj` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | −6 | 8.67% | 8.00% | 8.70% (n=299) | 0.33% | 93.33% | 100.00% | **−5.67 pp** | **.0190** |
| Llama3.1-8B | −4 | 9.00% | 7.67% | 9.06% (n=298) | 0.67% | 95.33% | 100.00% | **−5.33 pp** | **.0112** |
| Llama3.1-8B | 0 | **14.33%** | 12.67% | 14.38% (n=299) | 0.33% | 96.67% | 100.00% | — | — |
| Llama3.1-8B | +4 | 15.33% | 13.67% | 16.20% (n=284) | 5.33% | 93.33% | 100.00% | +1.00 pp | .7552 |
| Qwen2.5-7B | −6 | 10.00% | 9.67% | 10.91% (n=275) | 8.33% | 19.00% | 28.67% | **−10.67 pp** | **2.83×10⁻⁶** |
| Qwen2.5-7B | 0 | 20.67% | 20.67% | 22.30% (n=278) | 7.33% | 15.33% | 20.67% | — | — |
| Qwen2.5-7B | +6 | 20.67% | 20.33% | 21.38% (n=290) | 3.33% | 14.00% | 18.33% | 0.00 pp | 1.000 |
| Qwen2.5-7B | +8 | **25.67%** | 24.33% | 26.28% (n=293) | 2.33% | 12.33% | 16.67% | +5.00 pp | .1539 |

两个模型都没有建立有效的正向 workpoint。Llama `−6/−4` 显著降低准确率，`+4` 的小幅正向变化未被检出；Qwen `+8` 提高 5.00 pp，但没有通过 Holm，只能视为正向趋势。

Llama 各剂量都存在严重的生成循环与截断，因此其结果需要谨慎解释。不过，主指标取第一次合法答案，尾部循环不会改写已经提交的 first answer。

#### 6.2.5 CRUXEval-O Chat Interface

CRUXEval-O Chat 实验只改变 Llama3 的 prompt wrapper，并分别在 CoT 与 No-CoT 下运行相同的四点剂量 `{−6,−4,0,+4}`。两种条件各自在模型内部以 `α=0` 为基线执行 McNemar 检验和 Holm `m=3` 校正。

**Table 6.6. CRUXEval-O Chat No-CoT and CoT dose sweeps (N=300 per cell)**

| Condition | α | First / last accuracy | No marker | Nonliteral | Degenerate | Truncation | Median chars / tokens | Δ vs 0 | Holm `p_adj` | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-CoT | 0 | 45.67% / 45.67% | 1.7% | 1.3% | 1.7% | 2.0% | 458 / 132 | — | — | — |
| No-CoT | −6 | 46.67% / 46.67% | 1.0% | 1.3% | 0.7% | 1.3% | 468 / 135 | +1.00 pp | .7111 | [−2.67,+4.33] |
| No-CoT | −4 | 43.33% / 43.33% | 2.0% | 1.0% | 1.7% | 2.0% | 486 / 134 | −2.33 pp | .5299 | [−6.00,+1.00] |
| No-CoT | +4 | 41.00% / 41.00% | 2.7% | 0.3% | 2.3% | 2.7% | 207 / 70 | −4.67 pp | .2939 | [−10.00,+0.67] |
| CoT | 0 | 52.00% / 52.00% | 2.3% | 0.0% | 1.3% | 2.3% | 1005 / 283 | — | — | — |
| CoT | −6 | 51.33% / 51.33% | 3.0% | 0.7% | 3.0% | 3.0% | 1003 / 295 | −0.67 pp | 1.0000 | [−5.00,+3.67] |
| CoT | −4 | 49.33% / 49.33% | 2.7% | 0.0% | 1.7% | 2.7% | 1024 / 289 | −2.67 pp | .9667 | [−7.33,+2.00] |
| CoT | +4 | 50.33% / 50.33% | 2.7% | 0.3% | 1.3% | 3.0% | 711 / 199 | −1.67 pp | 1.0000 | [−6.33,+3.00] |

1. Chat 接口下，两种条件的格式与可评分性都保持健康，但所有非零剂量均未显著优于各自的 `α=0`，因此没有检测到有效 workpoint。
2. `+4` 在 CoT 和 No-CoT 下都明显缩短生成，却没有提高准确率，说明输出缩短不等于推理改善。CoT 的 `α=0` 点估计高于 No-CoT（52.00% vs 45.67%），但未进行跨条件检验，只作描述。

### 6.3 Local Stability and Near-Optimal Regions

单一 argmax 容易把抽样波动误写成精确的最佳剂量。本文所称的 near-optimal region，是指在已测离散剂量中，与 observed best 未被显著区分的集合。它不是连续区间，也不代表这些剂量已经被证明统计等效。

后补邻点只用于检查固定工作点附近的稳定性，不能重新定义原有 workpoint。

**Table 6.7. Observed near-optimal regions and local stability**

| Curve | Reference point | Added stability cell(s) | Observed near-optimal region | Key neighbour comparison |
|---|---|---|---|---|
| Llama GSM8K No-CoT | Best `−6` | — | **{−6,−4}** | `−6` vs `−4`: +5.00 pp, `p=.101` |
| Llama GSM8K CoT | Best `−4` | `−2`: 74.00%, Δ=+5.00 pp, `p_adj=.174` | **{−4}** | `−4` vs `−6`: +9.67 pp, `p=1.1×10⁻⁴` |
| Qwen GSM8K No-CoT | Frozen best `+8` | Later high-dose extension | **{+8}** on the frozen curve; later plateau beyond `+8` | `+8` vs `+6`: +8.00 pp, `p=.0022` |
| Qwen GSM8K CoT | Best `+6` | — | **{+6,+8}** | `+6` vs `+8`: +2.33 pp, `p=.371` |
| Llama MATH No-CoT | Frozen `−6` | `−8`: 39.33%, Δ=+2.67 pp, `p_adj=.403`, CI=[−3.00,+8.00] | **{−8,−6,−4}** | `−6` vs `−8`: +4.00 pp, `p=.126`; vs `−4`: +3.33 pp, `p=.212` |
| Llama MATH CoT | Frozen `−6` | `−8`: 45.33%, Δ=+3.33 pp, `p_adj=.328`, CI=[−0.67,+7.67] | **{−8,−6,−4}** | `−6` vs `−8`: +3.67 pp, `p=.0895`; vs `−4`: +4.00 pp, `p=.104` |
| Llama GSM-Hard No-CoT | Frozen best `−6` | — | **{−6,−4}** | Difference only +0.33 pp |
| Llama GSM-Hard CoT | Frozen `−6` | `−4`: 30.00%, Δ=+10.00 pp, `p_adj=1.36×10⁻⁶`, CI=[+6.33,+14.00] | **{−6,−4}** | `−4` vs `−6`: +4.00 pp, `p=.065` |
| Qwen GSM-Hard No-CoT | Frozen `+8` | `+10`: 50.33%, Δ=+16.33 pp, `p_adj=9.90×10⁻⁸`, CI=[+11.00,+21.67] | **{+8,+10}** | `+10` vs `+8`: 0.00 pp, `p=1.000` |
| Qwen GSM-Hard CoT | Frozen `+8` | `+6`: 49.00%, `p_adj=.000270`;<br>`+10`: 50.33%, `p_adj=8.46×10⁻⁵` | **{+6,+8,+10}** | `+8` vs `+6`: +2.33 pp, `p=.371`;<br>vs `+10`: +1.00 pp, `p=.664` |

多数曲线的最佳结果更适合表达为一个局部区域，而不是唯一剂量。Llama GSM8K CoT 的 `−4` 是较清晰的单点局部峰；Llama MATH 则表现为宽负向区域；Qwen GSM-Hard 和高剂量 GSM8K 结果更接近正向平台。

因此，workpoint selection 的合理目标是找到方向正确、regret 较低的区域，而不是声称精确命中唯一 argmax。

### 6.4 Exploratory Output-Pattern Diagnostics

为检查准确率变化是否伴随回答位置变化，我们使用 `early_candidate_rate`（`ec`）进行描述性分析。该指标判断首行是否提前出现裸数字；`ec` 下降只表示模型较少立即输出数字答案，不能直接等同于内部 commitment timing。

**Table 6.8. Output-pattern changes on boundary tasks**

| Task | Model | No-CoT ec (0→α) | CoT ec (0→α) | CoT accuracy Δ | Interpretation |
|---|---|---|---|---:|---|
| BBH object counting | Llama | 95.2% → 84.4% | 97.2% → 63.2% | **+16.00 pp** | `ec` 与准确率方向一致，但输出退化较高 |
| BBH object counting | Qwen | 100.0% → 44.4% | 100.0% → 7.2% | **+14.00 pp** | `ec` 大幅下降，同时准确率提高 |
| CRUXEval-O | Llama | 48.0% → 47.3% | 43.0% → 28.0% | −0.67 pp | `ec` 下降但准确率未改善 |
| CRUXEval-O | Qwen | 45.0% → 19.0% | 36.0% → 0.0% | **+19.33 pp** | `ec` 降至零，同时准确率提高 |
| LogiQA 2.0 | Llama | 0.0% → 13.3% | 0.0% → 4.3% | −2.33 pp | 指标不适用于字母选项 |
| LogiQA 2.0 | Qwen | 0.0% → 0.0% | 0.0% → 0.0% | −5.33 pp | 指标不适用于字母选项 |

BBH-Llama、BBH-Qwen 和 CRUXEval-O-Qwen 的 CoT 准确率收益都伴随 `ec` 下降。然而，CRUXEval-O-Llama 同样出现 `ec` 下降，却没有准确率收益，说明这种输出变化不是性能提升的充分条件。

LogiQA 使用字母选项，现有数字探测器无法判断其回答位置是否发生类似变化。总体上，这些结果只能说明部分准确率收益伴随着更少的提前数字作答，不能证明回答位置变化导致了性能提升。

Llama 的 Chat 四点扫描进一步排除了严重循环与截断作为主要混淆因素：输出已经可以稳定评分，但 CoT 与 No-CoT 均未出现准确率收益，因此该模型上的 CRUXEval-O null 不能仅用 Bare 接口失效解释。

**Table 6.9. GSM-Symbolic commitment / answer-formation timing on key dose comparisons**

主要 commitment 指标是 `early_candidate_rate`（`ec`）和 `reason_first_rate`；两者均为探索性描述统计，不进入 GSM-Symbolic 准确率的 Holm family。`posN`（首个可解析 `####` 标记的归一化字符位置）仅作为辅助格式指标，不要求与准确率同方向——它锚定在 `####` 这一格式事件上，而非答案候选值本身首次出现的位置。`no_answer=0`（本文所有 GSM-Symbolic cell 均如此）只说明冻结 fallback scorer 总能从生成文本中提取出某个可比较的数值，不代表模型都产生了规范的 `####` 提交；是否规范提交需分别参考 `no_marker_rate` 与 `marker_unparsed_rate`。

| Comparison | Accuracy (0→α) | `ec` (0→α) | `reason_first` (0→α) | `posN` (0→α, 辅助) |
|---|---:|---:|---:|---:|
| Llama No-CoT `0→−6` | .4811→.5756 | .2522→**.1789** | .3118→**.6007** | .3093→.3337 |
| Qwen No-CoT `0→+8` | .5333→.6633 | .9622→**.0656** | .0000→**.9844** | .8661→.8287 |
| Qwen CoT `0→+6` | .5289→.6878 | .9911→**.4378** | .0000→**.6389** | .9092→.8801 |
| Qwen CoT `0→+8` | .5289→.6822 | .9911→**.0644** | .0000→**.9633** | .9092→.8549 |
| Llama CoT `0→−6`（反例，`Δ=+1.11 pp`, `p_adj=.527`） | .5556→.5667 | .2833→**.1644** | .5926→**.7929** | .3865→.3294 |

显著改善通常伴随 `ec` 下降和 `reason_first` 上升，Qwen 上变化幅度尤其大；但 Llama CoT 是重要反例——commitment 时序发生了同方向的变化，准确率却没有显著提升。因此，commitment 改变与有效 steering 经常共现，但不是准确率提升的充分条件，这些指标是相关机制证据，而不是因果中介证明。

### 6.5 Cross-Benchmark Summary

**Table 6.10. Where effective workpoints were detected**

| Benchmark | Evaluation type | Llama | Qwen | Main conclusion |
|---|---|---|---|---|
| MATH | Fixed transfer + dose selection | Fixed `−6` supported | Fixed `+8` not detected; task-selected `+6` performs better | Model-specific |
| GSM-Hard | Fixed transfer | No-CoT and CoT supported | No-CoT and CoT supported | Strongest transfer result |
| GSM-Symbolic | Full dose / same-family robustness | No-CoT `−6/−4` supported; CoT not detected | No-CoT `+6/+8` and CoT `−6/+6/+8` supported | Same-family robustness |
| BBH object counting | Fixed transfer | CoT only | CoT only | CoT-dependent |
| CRUXEval-O | Fixed transfer + Chat interface check | Bare 和 Chat 均未检测到有效 workpoint | Bare No-CoT 和 CoT fixed workpoint supported；Chat 未运行 | Model-specific; fixing the interface did not rescue Llama steering |
| LogiQA 2.0 | Fixed transfer | Not detected | Not detected | Double null |
| ProofWriter-OWA | Full dose / interface comparison | Chat `−6` trend only; Bare result submission-sensitive | Chat `+8` supported | Interface-dependent |
| ZebraLogic-Easy | Full dose | No positive workpoint | No positive workpoint; `+8` harmful | High-dose failure boundary |
| FinQA | Full dose | No positive workpoint | `+8` trend only | No effective workpoint |

整体结果可以归纳为三点：

1. **固定工作点可以迁移，但范围有限。** GSM-Hard 的证据最稳定；MATH、BBH 和 CRUXEval-O 均表现出模型或提示条件差异。
2. **完整剂量曲线揭示了更多条件性结果。** GSM-Symbolic 支持同任务家族鲁棒性，ProofWriter 显示接口依赖；ZebraLogic 和 FinQA 没有找到有效的正向 workpoint。CRUXEval-O 展示模型特异性迁移：Qwen 在 Bare fixed-workpoint 条件下有效，Llama 即使使用健康的 Chat 接口仍为 null。不应据此写成"chat template 没有价值"；它改善的是生成稳定性和可评分性，但没有使 steering accuracy effect 出现。
3. **不存在跨模型、跨任务统一的最佳 α。** 更合理的目标是识别每个模型和任务中的有效方向、近优区域与失败边界。

这些结果属于模型输出和准确率层面的证据，不证明生物多巴胺、通用 wanting 轴或 commitment timing 的因果中介机制。完整协议、运行配置、统计家族、敏感性分析和输出诊断保留在 `CLAUDE.md`。

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
