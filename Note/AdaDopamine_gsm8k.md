## 0. Background

### 0.1 Prompt Template Symmetrization

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

交互量定义为：`[Acc(CoT, α) − Acc(CoT, 0)] − [Acc(No-CoT, α) − Acc(No-CoT, 0)]`

| α | Interaction | Bootstrap 95% CI |
|---:|---:|---:|
| −8 | +0.67 pp | — |
| −6 | +0.33 pp | [−6.67, +7.67] |
| −4 | −0.33 pp | [−6.67, +6.00] |
| +4 | +0.33 pp | [−6.67, +7.33] |

`−8` 的描述性点估计也接近 0，与原有三个点的近似平行形状一致。原有三个 CI 均跨 0，且约覆盖 ±7 pp，因此当前结果只能说明：未检出 CoT 明显改变 steering 效应的证据。

这不是等价性证明，也不能证明 CoT 与 steering 机制独立、严格可加或作用于不同内部过程。该交互分析属于 descriptive / exploratory analysis。

### 3.2 Dose-Dependent Output Behavior

本节使用统一口径分析 MATH 输出。`first_acc` 是主要性能指标；`early_cand_rate`、`reason_first_rate` 和 candidate 前后字符数用于描述答案候选与可见推理的先后顺序。`posN_med` 只表示正式 `\boxed{}` marker 的位置，与第一个 answer candidate 不是同一事件。

所有条件均包含 300 个样本。`cond_acc` 只在存在有效正式答案 marker 的样本中计算，不替代总体 `first_acc`。

**Table 3.5. Llama MATH performance and output behavior**

| Condition | α | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-CoT | −8 | 39.33% | 41.00% | 90.67% | 42.65% | 15.33% | 79.86% | 344 | 4308 | 0.2776 | 69.00% |
| No-CoT | −6 | **43.33%** | **44.00%** | 92.67% | **46.40%** | **6.33%** | 79.44% | **360** | **3449** | 0.3142 | 68.67% |
| No-CoT | −4 | 40.00% | 39.67% | 87.33% | 45.80% | 10.33% | 53.79% | 68 | 4646 | 0.2140 | 66.00% |
| No-CoT | 0 | 36.67% | 36.00% | 86.00% | 42.25% | 26.67% | 47.18% | 7 | 5306 | 0.1467 | 63.00% |
| No-CoT | +4 | 33.00% | 34.00% | 85.33% | 37.89% | 58.33% | 30.11% | 0 | 5596 | 0.1561 | 68.67% |
| CoT | −8 | 45.33% | 44.67% | 94.33% | 48.06% | 3.33% | 94.96% | 462 | 3061 | 0.4477 | 63.33% |
| CoT | −6 | **49.00%** | **48.00%** | 92.33% | **53.07%** | **1.00%** | **97.85%** | **493** | 3306 | 0.3413 | 62.00% |
| CoT | −4 | 45.00% | 44.00% | 92.33% | 48.74% | 2.33% | 92.47% | 443 | 3486 | 0.3257 | 63.33% |
| CoT | 0 | 42.00% | 41.00% | 92.67% | 45.32% | 12.00% | 79.36% | 290 | 4815 | 0.2330 | 66.33% |
| CoT | +4 | 38.67% | 38.00% | 87.00% | 44.06% | 45.67% | 44.04% | 0 | 3733 | 0.2791 | 66.00% |

#### Dose-Dependent Candidate Ordering

No-CoT 下，负向 α 通常对应更晚出现的 answer candidate 和更多 candidate 前推理。`α=−6` 的 `early_cand_rate` 最低（6.33%），candidate 前字符数中位数为 360，同时取得最高 `first_acc`（43.33%）。当 α 增加至 `+4` 时，`early_cand_rate` 升至 58.33%，`reason_first_rate` 降至 30.11%，candidate 前字符数降至 0，准确率也降至 33.00%。

candidate 更早出现并不意味着回答更快结束。No-CoT 的 `post_cand_chars_med` 从 `−6` 的 3449 增至 `+4` 的 5596，说明正向 α 更常表现为先出现答案候选、随后继续生成大量内容，而不是立即完成回答。

`−8` 没有延续 `−6` 的准确率提升：虽然其 `reason_first_rate` 和 candidate 前文本仍处于较高水平，但 `first_acc` 降至 39.33%。这与 §3.1 的结论一致——负向端更适合描述为 `{−8,−6,−4}` 的宽近优区域，而不是越负越好。

#### Effect of CoT on Output Organization

CoT 在所有共有剂量下都降低了 `early_cand_rate`，并提高了 `reason_first_rate`。变化在负向条件下最明显：

- `α=−6`：`early_cand_rate` 从 6.33% 降至 1.00%，`reason_first_rate` 从 79.44% 升至 97.85%；
- `α=−4`：`early_cand_rate` 从 10.33% 降至 2.33%，`reason_first_rate` 从 53.79% 升至 92.47%；
- `α=0`：`early_cand_rate` 从 26.67% 降至 12.00%，`reason_first_rate` 从 47.18% 升至 79.36%。

但 CoT 没有改变整体剂量方向。CoT 下仍是 `−6` 表现最好，而 `+4` 同时具有最高的 `early_cand_rate` 和最低的准确率。CoT 因此主要改善输出组织，而不是消除 α 对 candidate ordering 的影响。

#### Submission and Marker Behavior

`valid_sub_rate` 在 No-CoT 下为 85.33%–92.67%，CoT 下为 87.00%–94.33%；`cond_acc` 与总体准确率呈现相似排序。因此，主要曲线不能简单归因于某个剂量无法生成正式答案。

`multi_marker_rate` 在全部条件下都较高（62.00%–69.00%），但没有随准确率呈现一致变化。`posN_med` 同样不与性能稳定对应。这两项更适合作为输出格式和答案修订的辅助指标，不能单独判断答案在内部何时形成。

**Conclusion.** Llama MATH 的较好表现与适度负向 α、较少 early candidate 和更多 candidate 前推理同时出现；正向 α 则更常先形成 candidate，再继续生成较长文本。CoT 能改善输出结构，但不会改变这一总体剂量方向。这些结果属于干预后的输出关联，不构成因果中介证据。

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

本节使用 `first_acc` 作为主要性能指标，`last_acc` 用于观察后续答案修订。`early_cand_rate` 和 `reason_first_rate` 描述第一个 answer candidate 前后的可见输出顺序；`posN_med` 表示正式答案 marker 在全文中的位置。candidate 与 marker 不是同一事件，相关指标均为干预后的描述性读数。

所有条件均包含 300 个样本。`cond_acc` 只在存在有效正式答案 marker 的样本中计算，因此用于检查输出格式和条件准确率，不替代总体 `first_acc`。raw α 只表示对应实验中的干预强度，不应视为跨模型或跨任务的等效剂量。

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

GSM8K 的主要性能变化集中在 `+6/+8`。No-CoT 中，`+6`（Holm `p_adj=.016`）和 `+8`（`p_adj<1e−4`）显著高于 α=0，其余原始剂量未通过校正。继续增加至探索性的 `+10/+12` 后，`first_acc` 仅从 86.00% 小幅升至 88.33%–88.67%，因此当前结果更接近高剂量平台，而不是明确的单点峰值。

CoT 的最高准确率出现在 `+6`，相对 α=0 的 Holm `p_adj=.0002`；`+8` 同样显著（`p_adj=.0030`）。但 `+6` 与 `+8` 没有显著差异（探索性 `p=.371`），因此更适合将 `{+6,+8}` 视为近优区域。

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

MATH No-CoT 的最高准确率出现在 `+6`，相对 α=0 提高 7.67 pp，并通过 Holm 校正（`p_adj=.0087`）。继续增加至 `+8` 后，准确率回落至 63.33%；`+6` 与 `+8` 的探索性配对比较为 `p=.040`，形成描述性的高剂量右臂。

CoT 的曲线更平缓：`first_acc` 从 α=0 的 63.00% 升至 `+6` 的 66.00%，随后在 `+8` 变为 64.00%。各剂量相对 α=0 的比较均为 Holm `p_adj=1.000`。逐剂量比较 CoT 与 No-CoT 时，也没有结果通过校正；最大差异为 `α=−4` 的 +7.00 pp（raw `p=.0065`，Holm `p_adj=.0581`）。

**Table 4.3. Output reordering from baseline to the main workpoint**

| Task | Condition | Dose comparison | first_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K | No-CoT | `0 → +8` | 68.00% → **86.00%** | 96.33% → **5.00%** | 0.00% → **98.00%** | 0 → 140 | 1130 → 452 |
| GSM8K | CoT | `0 → +6` | 76.33% → **88.33%** | 97.33% → **34.33%** | 1.00% → **69.33%** | 0 → 136 | 1152 → 626 |
| MATH | No-CoT | `0 → +6 → +8` | 60.67% → **68.33%** → 63.33% | 67.33% → 19.33% → **10.33%** | 11.11% → 77.59% → **86.16%** | 0 → 224 → 230 | 1438 → 860 → 727 |
| MATH | CoT | `0 → +6 → +8` | 63.00% → **66.00%** → 64.00% | 71.33% → 29.67% → **10.33%** | 5.05% → 62.50% → **85.96%** | 0 → 164 → 252 | 1567 → 970 → 816 |

#### Commitment Reordering across Tasks

GSM8K 与 MATH 都出现了明显的 output reordering：随着正向 α 增加，`early_cand_rate` 下降，`reason_first_rate` 上升，更多可见推理出现在第一个 candidate 之前。

但 reordering 与准确率的关系因任务而异。GSM8K 中，输出顺序变化与性能提升同时出现，并在 `+6/+8` 附近进入近优平台；MATH 中，准确率在 `+6` 达到较高点后回落，但 candidate ordering 在 `+8` 仍继续变化。因此，减少 early candidate 可能与更好的输出状态相关，但不是提高准确率的充分条件。

MATH 的正式 `\boxed{}` marker 通常位于输出末尾，因此 `posN_med` 难以区分答案形成顺序。本节主要依据 candidate-based metrics 描述 output ordering。所有相关指标均为干预后的输出读数，不能作为因果中介证据。

**Conclusion.** 正向 α 会系统性改变 Qwen 的输出顺序，但性能结果具有明显的任务边界：GSM8K 在高剂量进入平台，MATH 则在 `+6` 后出现回落。

### 4.2 High-Dose Boundary Checks

GSM8K 与 MATH 在主要 output-ordering transition 后呈现不同结果。GSM8K No-CoT 的准确率从 `+8` 到 `+12` 保持在 86.00%–88.67%，形成高剂量平台。完整性检查显示，这些条件没有空输出，截断率不超过 1.0%，clean subset accuracy 仍为 84%–88%，因此平台不能简单归因于生成失败或极端 marker 重复。

MATH 则在 `+6` 后出现回落。No-CoT 的 `first_acc` 从 68.33% 降至 63.33%，下降主要集中在 Level 5：准确率由 47.2% 降至 36.0%；CoT 的 Level 5 准确率也由 43.8% 降至 36.0%。与此同时，`early_cand_rate` 仍继续下降，说明 candidate 更晚出现并不一定带来更高准确率。

少量极端 `\boxed{}` 重复主要影响 `last_acc`，不改变以 `first_acc` 为主的结论。完整性检查、post-treatment 分组和 marker repetition 明细记录于 `CLAUDE.md`。

**Conclusion.** Qwen 的高剂量边界具有任务依赖性：GSM8K 表现为平台，MATH 则出现回落。相同方向的 output reordering 可以对应不同的性能结果。

### 4.3 Cross-Model Summary

Llama3.1-8B 与 Qwen2.5-7B 使用相同的分析框架，但二者的有效 α 方向和剂量曲线并不相同。由于模型使用不同的 mask、层带和激活尺度，raw α 不能作为跨模型的共同剂量；可比较的是 performance curve 和 output behavior 的变化形态。

**Table 4.4. Behavioral comparison between Llama and Qwen**

| Dimension | Llama3.1-8B | Qwen2.5-7B |
| --- | --- | --- |
| GSM8K performance curve | No-CoT 在负向区域表现较好；CoT 的 `−4` 是清晰的局部峰 | No-CoT 在 `+8～+12` 进入平台；CoT 的近优区域为 `{+6,+8}` |
| MATH performance curve | `{−8,−6,−4}` 构成宽的负向近优区域 | No-CoT 在 `+6` 达到较高点，`+8` 出现回落；CoT 剂量差异较弱 |
| Main output change | 适度负向 α 通常减少 early candidate、增加 reason-first output；正向 α 更常伴随提前回答和较长的 post-candidate generation | 正向 α 通常减少 early candidate、增加 reason-first output，并压缩 post-candidate generation |
| High-dose boundary | GSM8K 的极端负向 `−8` 形成独立失败模式；正向 α 整体降低表现 | GSM8K 在高剂量饱和，MATH 则在 ordering change 继续增强时出现准确率回落 |
| Effect of CoT | 增加 step structure、减少 repetition，并提高答案稳定性，但不会消除 dose dependence | 改善部分条件的输出结构和准确率，但主要 transition 仍集中在正向区域 |
| Near-optimal region | GSM8K 与 MATH 的有效区域均位于负向 α，但具体范围依任务和 CoT 而变 | GSM8K 的有效区域位于正向高剂量；MATH 的有效范围更窄 |
| Cross-model interpretation | 模型特定的负向响应 | 模型特定的正向响应 |

两个模型在 raw α 上呈现相反方向，但这不表示它们的 baseline 位于相反的内部状态，也不表示各自最佳剂量到达了同一个内部工作点。因此，这里属于 **cross-model analysis**，不能描述为逐点 replication。

尽管剂量方向不同，两种模型在较高表现区域中呈现一个共同的 behavioral signature：第一个 answer candidate 通常更晚出现，candidate 之前包含更多可见推理，candidate 之后的无效延伸相对减少。换句话说，有效工作点通常减少“先报答案、再补过程”的输出模式。

但这一关系不是充分条件。Qwen MATH 在 `+8` 时，`early_cand_rate` 继续下降、`reason_first_rate` 继续上升，准确率却从 `+6` 的较高点回落；Llama 的极端负向条件也表明，进一步推迟或改变答案提交并不一定继续改善表现。output reordering 可以描述较优状态，但不能单独预测准确率。

**Conclusion.** RSN steering 可以在不同模型中重组 reasoning text 与 answer candidate 的输出顺序，但有效方向、近优范围和高剂量边界都依赖模型与任务。共同之处不是某个固定 α，而是较优工作点通常更少出现过早的 answer candidate；这一现象仍是干预后的行为读数，不能直接解释为内部推理机制或因果中介。

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

## 6. Cross-Benchmark Workpoint Performance and Behavioral Boundaries

本节按 benchmark 分别汇总 workpoint performance、local stability 和 output behavior。实验包含两类设计：

1. **Fixed-workpoint transfer**：直接使用 GSM8K 冻结的工作点，不根据目标任务结果重新选择剂量。
2. **Task-specific dose sweep**：在目标任务上测试多个剂量，用于识别该任务的 near-optimal region 或失败边界。

固定迁移点为 Llama3.1-8B 的 `α=−6` 和 Qwen2.5-7B 的 `α=+8`。由于模型使用不同的 mask、层范围和激活尺度，raw α 不能作为跨模型或跨任务的等效剂量。


| 指标                    | 含义                                 |
| --------------------- | ---------------------------------- |
| `α`                   | RSN 干预强度；不同模型间不能直接比较数值大小。          |
| `n`                   | 该条件下的样本数。                          |
| `first_acc`           | 按第一次正式答案判断的准确率，主要性能指标。             |
| `last_acc`            | 按最后一次正式答案判断的准确率，用于观察答案修订。          |
| `fixed_n`             | 第一次答错、最后一次改对的样本数。                  |
| `broke_n`             | 第一次答对、最后一次改错的样本数。                  |
| `valid_sub_rate`      | 能解析出正式答案标记的比例。                     |
| `cond_acc`            | 只在存在有效正式答案的样本中计算的准确率。              |
| `early_cand_rate`     | 输出开头很早就出现答案候选的比例。                  |
| `cand_coverage`       | 能定位到第一个答案候选的样本比例。指标可计算性的覆盖率/质量检查指标 |
| `reason_first_rate` | 首次答案候选前出现 `=`、`Step 1` 或编号步骤等推理文本的比例；属于表面文本代理指标。 |
| `pre_cand_chars_med`  | 首个答案候选之前的字符数中位数。                   |
| `post_cand_chars_med` | 首个答案候选之后继续生成的字符数中位数。               |
| `posN_med`            | 正式答案标记在全文中的归一化位置中位数，范围为 0–1。       |
| `multi_marker_rate`   | 同一输出中出现多个正式答案标记的比例。                |

> **备注：**“答案候选（candidate）”与“正式答案标记（marker）”不是同一事件。candidate 是输出中最早出现的答案形态数值或表达式，例如开头的裸数字、等号右侧结果或 “the answer is …” 后的值；它不要求与正确答案匹配，也可能是中间结果。marker 则是 `####`（GSM8K）或 `\boxed{}`（MATH）等正式提交格式。因此，candidate 通常用于分析答案形成顺序，`posN_med` 用于分析正式提交位置。

本节以各模型在 GSM8K 中确定的 workpoint 或 near-optimal range 为起点，检验其在其他 benchmark 上的迁移表现。除明确标注的 local-stability points 外，不根据目标任务结果重新选择 α。整体 `first_acc` 为主要性能指标；行为指标采用各任务的原生答案格式，只用于描述干预后的输出状态，不作为内部 commitment 或准确率因果机制的直接证据。

### 6.1 GSM-Hard

GSM-Hard 是 fixed-workpoint transfer 中证据最完整的任务。No-CoT 剂量在查看准确率前完成 blind selection；CoT 与新增邻点用于检查 condition transfer 和 local stability。所有条件均包含 300 个样本，主要性能指标为离线计算的 `first_acc`。

`evidence_status` 区分三类证据：

- `prospective_blind_selection`：查看准确率前冻结的盲测；
- `condition_transfer_supplement`：将固定工作点迁移至 CoT 条件；
- `post_hoc_local_stability`：观察主要结果后追加的邻点检查。

#### Llama3.1-8B

##### No-CoT

**Table 6.1. Llama GSM-Hard No-CoT performance and output behavior**

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| −8 | 300 | 11.00% | 11.33% | 67.67% | 6.40% | 66.00% | 27.10% | 5 | 2162 | 0.0000 | 10.33% | prospective_blind_selection |
| −6 | 300 | **24.33%** | **23.67%** | 54.67% | 20.73% | 28.67% | **61.82%** | **204** | 1877 | 0.2740 | 13.67% | prospective_blind_selection |
| −4 | 300 | 24.00% | 23.00% | 53.00% | **28.30%** | **28.00%** | 38.19% | 0 | 1920 | 0.2351 | 15.67% | prospective_blind_selection |
| 0 | 300 | 18.00% | 17.33% | 54.33% | 20.25% | 45.67% | 26.64% | 0 | 1952 | 0.2161 | 13.33% | prospective_blind_selection |
| +4 | 300 | 17.00% | 17.33% | 44.67% | 19.40% | 60.00% | 15.79% | 0 | 1990 | 0.1274 | 11.33% | prospective_blind_selection |

##### CoT

**Table 6.2. Llama GSM-Hard CoT performance and output behavior**

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| −6 | 300 | 26.00% | 27.00% | 52.00% | 17.95% | 30.33% | 63.74% | 234 | 1928 | 0.0000 | 17.00% | condition_transfer_supplement |
| −4 | 300 | **30.00%** | **29.67%** | 45.67% | **27.01%** | **19.00%** | **74.63%** | **292** | 1864 | 0.3294 | 11.67% | post_hoc_local_stability |
| 0 | 300 | 20.00% | 20.67% | 41.33% | 19.35% | 43.67% | 41.29% | 0 | 1930 | 0.2810 | 11.00% | condition_transfer_supplement |

No-CoT 盲测中，GSM8K 冻结工作点 `−6` 将 `first_acc` 从 18.00% 提高至 24.33%（Δ=+6.33 pp，raw `p=.00661`）。`−4` 的准确率为 24.00%，与 `−6` 仅相差 0.33 pp，因此 Llama No-CoT 的近优区域为 `{−6,−4}`，而不是唯一 workpoint。

CoT 中，固定工作点 `−6` 将准确率从 20.00% 提高至 26.00%（Δ=+6.00 pp，`p_adj=.00393`，95% CI=[+2.33,+10.00]）。后补的 `−4` 达到 30.00%，相对 baseline 提高 10.00 pp（`p_adj=1.36×10⁻⁶`），但与 `−6` 的直接差异未达到显著（+4.00 pp，`p=.065`）。因此，CoT 的近优区域同样记为 `{−6,−4}`。

行为指标与性能变化大体一致。No-CoT 从 `0` 到 `−6` 时，`early_cand_rate` 从 45.67% 降至 28.67%，`reason_first_rate` 从 26.64% 升至 61.82%，candidate 前字符数中位数从 0 墠至 204。CoT 的 `−4` 进一步表现出更少的 early candidate（19.00%）和更高的 reason-first rate（74.63%）。

`−8` 是明显的负向边界：准确率降至 11.00%，`early_cand_rate` 升至 66.00%，`posN_med` 降至 0。说明负向 α 并非越大越好；超过近优区域后，输出重新转向过早出现 candidate 或 marker。

#### Qwen2.5-7B

##### No-CoT

**Table 6.3. Qwen GSM-Hard No-CoT performance and output behavior**

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| −4 | 300 | 34.33% | 36.67% | 79.33% | 34.03% | 94.67% | 0.00% | 0 | 1321 | 0.8062 | 35.00% | prospective_blind_selection |
| 0 | 300 | 34.00% | 34.67% | 72.67% | 36.24% | 93.67% | 0.00% | 0 | 1266 | 0.7680 | 29.00% | prospective_blind_selection |
| +4 | 300 | 34.67% | 36.00% | 78.00% | 35.90% | 92.00% | 0.67% | 0 | 1264 | 0.6791 | 39.33% | prospective_blind_selection |
| +6 | 300 | 40.33% | 40.33% | 86.33% | 42.47% | 58.67% | 33.67% | 0 | 1057 | 0.5969 | 39.33% | prospective_blind_selection |
| +8 | 300 | **50.33%** | **48.33%** | **98.33%** | 50.51% | 6.00% | **98.00%** | 204 | **614** | 0.7765 | 34.00% | prospective_blind_selection |
| +10 | 300 | **50.33%** | 46.33% | **98.33%** | **51.19%** | **4.67%** | **98.00%** | **238** | 698 | 0.7987 | 32.33% | post_hoc_local_stability |

##### CoT

**Table 6.4. Qwen GSM-Hard CoT performance and output behavior**

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 300 | 38.00% | 36.67% | 72.00% | 37.96% | 96.00% | 0.33% | 0 | 1456 | 0.8611 | 30.67% | condition_transfer_supplement |
| +6 | 300 | 49.00% | 47.33% | 91.00% | 50.92% | 48.00% | 59.33% | 108 | 812 | 0.8359 | 34.33% | post_hoc_local_stability |
| +8 | 300 | **51.33%** | **50.33%** | **98.67%** | **52.03%** | 8.33% | 97.33% | 226 | 582 | 0.8286 | 31.00% | condition_transfer_supplement |
| +10 | 300 | 50.33% | 49.33% | **98.67%** | 51.01% | **2.67%** | **98.33%** | **228** | **580** | 0.8251 | 30.00% | post_hoc_local_stability |

No-CoT 盲测中，GSM8K 冻结工作点 `+8` 将 `first_acc` 从 34.00% 提高至 50.33%（Δ=+16.33 pp，raw `p=1.41×10⁻⁸`）。后补的 `+10` 同样为 50.33%，与 `+8` 完全相同（`p=1.000`），因此近优区域为 `{+8,+10}`。

CoT 中，固定工作点 `+8` 将准确率从 38.00% 提高至 51.33%（Δ=+13.33 pp，`p_adj=9.42×10⁻⁶`，95% CI=[+8.00,+19.00]）。后补的 `+6` 和 `+10` 分别达到 49.00% 和 50.33%，与 `+8` 均未显著分开（`p=.371/.664`），因此近优区域为 `{+6,+8,+10}`。

Qwen 的 output reordering 集中在正向高剂量。No-CoT 从 `0` 到 `+8` 时，`early_cand_rate` 从 93.67% 降至 6.00%，`reason_first_rate` 从 0.00% 升至 98.00%，candidate 前字符数从 0 增至 204，candidate 后字符数从 1266 降至 614。CoT 从 `0` 到 `+8` 也呈现相同变化：`early_cand_rate` 从 96.00% 降至 8.33%，`reason_first_rate` 从 0.33% 升至 97.33%。

继续增加至 `+10` 后，candidate ordering 仍有小幅变化，但准确率没有进一步提高。这说明减少 early candidate 与进入较优工作区间相关，但在进入平台后，继续推迟 candidate 不会持续增加准确率。

#### Summary

GSM-Hard 是 fixed-workpoint transfer 最稳定的正向结果：Llama 的负向工作点和 Qwen 的正向工作点在 No-CoT 与 CoT 下都提高了准确率。邻点分析进一步表明，更合理的描述是近优区域——Llama 为 `{−6,−4}`，Qwen No-CoT 为 `{+8,+10}`，Qwen CoT 为 `{+6,+8,+10}`。

两个模型的有效 raw α 方向相反，但较优区域具有相似的 behavioral signature：更少的 early candidate、更多的 reason-first output，以及更多 candidate 前文本。该模式描述的是干预后的输出重组，不能证明内部推理顺序或因果中介关系。

### 6.2 GSM-Symbolic

GSM-Symbolic 汇总 `main`、`p1` 和 `p2` 三个配置，每个配置包含 300 个实例，因此每个剂量共有 900 个样本。它与 GSM8K 属于同一任务家族，主要用于检验 arithmetic workpoint 对题目扰动的稳健性，不构成独立的跨领域迁移证据。

推断统计以 `original_id` 为 cluster，对三个配置等权，并在每个模型和 CoT 条件内执行 Holm `m=3` 校正。完整的数据修复、cluster bootstrap 和校验记录于 `CLAUDE.md`。

`candidate_posN_med` 表示第一个 answer candidate 的归一化位置，`posN_med` 表示正式答案 marker 的位置。前者更直接描述 output ordering；两者均为干预后的输出读数。

#### Llama3.1-8B

##### No-CoT
**Table 6.5. Llama GSM-Symbolic No-CoT performance and output behavior**

| α | n | first_acc | Main | P1 | P2 | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 900 | **57.56%** | 71.00% | 60.67% | 41.00% | 55.56% | 50.89% | 53.98% | 17.89% | 60.07% | 167 | 1906 | 0.0751 | 0.3429 | 15.33% |
| −4 | 900 | 52.34% | 62.67% | 56.67% | 37.67% | 50.33% | 55.44% | 52.81% | 19.78% | 27.92% | 0 | 1908 | 0.0000 | 0.3042 | 18.00% |
| 0 | 900 | 48.11% | 57.33% | 53.33% | 33.67% | 45.78% | 61.67% | 54.04% | 25.22% | 31.18% | 0 | 1891 | 0.0000 | 0.2988 | 18.66% |
| +4 | 900 | 38.89% | 50.67% | 39.33% | 26.67% | 38.00% | 47.00% | 44.25% | 49.67% | 13.72% | 0 | 2012 | 0.0000 | 0.2088 | 14.78% |

##### CoT

**Table 6.6. Llama GSM-Symbolic CoT performance and output behavior**

| α | n | first_acc | Main | P1 | P2 | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 900 | 56.67% | 64.00% | 59.00% | 47.00% | 58.89% | 35.11% | 43.76% | 16.45% | 79.29% | 291 | 1895 | 0.1356 | 0.3204 | 13.11% |
| −4 | 900 | **58.56%** | 70.67% | 63.67% | 41.33% | 59.33% | 31.00% | 50.05% | 14.00% | 70.35% | 272 | 1899 | 0.1267 | 0.3756 | 10.67% |
| 0 | 900 | 55.56% | 62.67% | 58.67% | 45.33% | 55.56% | 34.78% | 51.84% | 28.33% | 59.26% | 209 | 1917 | 0.0951 | 0.3835 | 12.67% |
| +4 | 900 | 40.22% | 54.00% | 41.33% | 25.33% | 39.11% | 28.67% | 45.52% | 71.66% | 11.38% | 0 | 2186 | 0.0000 | 0.2092 | 7.56% |

No-CoT 下，`−6` 将 `first_acc` 从 48.11% 提高至 57.56%（Δ=+9.44 pp，95% CI=[+5.78,+13.11]，Holm `p_adj=.0003`）；`−4` 同样显著提高表现（Δ=+4.22 pp，95% CI=[+0.78,+7.78]，`p_adj=.0164`）。相反，`+4` 使准确率下降 9.22 pp（95% CI=[−13.00,−5.56]，`p_adj=.0003`）。

`−6` 的提升伴随 output reordering：相对 α=0，`early_cand_rate` 从 25.22% 降至 17.89%，`reason_first_rate` 从 31.18% 升至 60.07%，candidate 前字符数从 0 墠至 167。`−4` 虽然也提高准确率，但其 `reason_first_rate` 和 candidate position 变化较弱，说明单一行为指标不能解释全部性能增益。

CoT 下，`−6` 和 `−4` 的点估计分别提高 1.11 pp 和 3.00 pp，但均未通过 Holm 校正（`p_adj=.527/.122`）。二者仍表现出更少的 early candidate 和更多 reason-first output，说明 output reordering 可以在没有显著准确率提升时出现。`+4` 则使准确率下降 15.33 pp（95% CI=[−20.00,−10.90]，`p_adj=.0003`），同时 `early_cand_rate` 升至 71.66%。

因此，Llama 的 GSM8K 固定工作点 `−6` 只在 GSM-Symbolic No-CoT 下建立了显著正向迁移；CoT 下未检测到相同增益。正向 `+4` 在两种条件下均形成明确的失败方向。

#### Qwen2.5-7B

##### No-CoT

**Table 6.7. Qwen GSM-Symbolic No-CoT performance and output behavior**

| α | n | first_acc | Main | P1 | P2 | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 900 | 50.22% | 59.33% | 54.67% | 36.67% | 54.22% | 73.78% | 46.43% | 97.22% | 0.00% | 0 | 1593 | 0.0000 | 0.8368 | 34.33% |
| 0 | 900 | 53.33% | 65.33% | 56.33% | 38.33% | 56.11% | 69.11% | 51.23% | 96.22% | 0.00% | 0 | 1599 | 0.0000 | 0.8624 | 28.67% |
| +6 | 900 | 60.67% | 73.00% | 64.00% | 45.00% | 60.89% | 83.11% | 63.06% | 53.44% | 46.22% | 0 | 1036 | 0.0000 | 0.8226 | 25.11% |
| +8 | 900 | **66.89%** | 78.00% | 71.67% | 51.00% | 65.33% | 98.33% | 67.49% | 6.56% | 98.44% | 170 | 706 | 0.1753 | 0.8318 | 27.22% |

##### CoT

**Table 6.8. Qwen GSM-Symbolic CoT performance and output behavior**

| α | n | first_acc | Main | P1 | P2 | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 900 | 56.44% | 66.00% | 64.33% | 39.00% | 57.78% | 79.89% | 55.37% | 99.33% | 0.00% | 0 | 1626 | 0.0000 | 0.9007 | 38.67% |
| 0 | 900 | 52.89% | 63.33% | 56.33% | 39.00% | 53.11% | 85.22% | 52.52% | 99.11% | 0.00% | 0 | 1606 | 0.0000 | 0.9113 | 38.56% |
| +6 | 900 | **65.11%** | 79.33% | 68.33% | 47.67% | 64.22% | 92.56% | 66.57% | 43.78% | 63.89% | 114 | 918 | 0.1216 | 0.8793 | 31.45% |
| +8 | 900 | 65.00% | 81.67% | 66.67% | 46.67% | 64.56% | 98.89% | 64.99% | 6.44% | 96.33% | 112 | 556 | 0.1859 | 0.8534 | 28.11% |

No-CoT 下，`+6` 和 `+8` 分别将准确率提高 7.33 pp 和 13.56 pp，两者均通过 Holm 校正（`p_adj=.0003`）。固定工作点 `+8` 同时具有最高的 `first_acc`、最高的 `valid_sub_rate` 和最低的 `early_cand_rate`。`−6` 的准确率下降 3.11 pp，但未达到显著（`p_adj=.0802`）。

CoT 下，`+6` 和 `+8` 分别提高 12.22 pp 和 12.11 pp（均为 `p_adj=.0003`），两者的准确率几乎相同，形成正向高剂量平台。`−6` 也产生较小但显著的提升（Δ=+3.56 pp，95% CI=[+0.11,+6.89]，`p_adj=.044`）。

正向高剂量的主要行为变化是从 answer-first 转向 reason-first。No-CoT 从 α=0 到 `+8` 时，`early_cand_rate` 从 96.22% 降至 6.56%，`reason_first_rate` 从 0.00% 升至 98.44%；CoT 中相应变化为 99.11%→6.44% 和 0.00%→96.33%。candidate 后字符数也明显减少。

但是，CoT 从 `+6` 到 `+8` 时，`early_cand_rate` 继续从 43.78% 降至 6.44%，准确率却基本不变（65.11% vs 65.00%）。此外，`−6` 在没有改善 candidate ordering 的情况下仍产生小幅显著增益。这两项结果共同表明，reason-first output 与较高表现经常同时出现，但既不是准确率提升的必要条件，也不是充分条件。

#### Summary

GSM-Symbolic 提供了同任务家族内的部分迁移证据：

- Llama No-CoT 的 `−6/−4` 显著提高准确率，但 CoT 下没有检测到相同增益；`+4` 在两种条件下均显著降低表现。
- Qwen No-CoT 的 `+6/+8` 和 CoT 的 `−6/+6/+8` 均显著提高准确率，其中主要增益集中在正向高剂量。
- Qwen `+6/+8` 的高表现通常伴随明显的 output reordering，但进一步减少 early candidate 不会继续提高准确率。

**Conclusion.** GSM-Symbolic 支持 workpoint 在相近任务家族中的有限迁移，但结果仍依赖模型与 CoT 条件。candidate ordering 是有用的 behavioral signature，却不能单独解释或预测全部准确率变化。

### 6.3 BBH Object Counting

BBH Object Counting 每个条件包含 250 个样本。这里主要检验 GSM8K 固定工作点——Llama 的 `−6` 和 Qwen 的 `+8`——能否迁移到物体计数任务。`first_acc` 是主要性能指标；`valid_sub_rate` 和 `cond_acc` 用于检查正式答案的提交质量。

#### Llama3.1-8B

##### No-CoT

**Table 6.9. Llama BBH Object Counting No-CoT performance and output behavior**

| α | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 40.80% | 40.80% | 82.00% | 47.80% | 84.40% | 9.20% | 0 | 1922 | 0.0875 | 51.60% |
| 0 | **41.60%** | **41.20%** | 86.80% | **46.54%** | 95.20% | 2.00% | 0 | 1983 | 0.0633 | 63.60% |
| +4 | 32.80% | 32.80% | 83.60% | 35.89% | 99.20% | 0.00% | 0 | 2002 | 0.0963 | 58.00% |

##### CoT

**Table 6.10. Llama BBH Object Counting CoT performance and output behavior**

| α | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | **56.80%** | **56.80%** | 92.00% | **54.35%** | **63.20%** | **39.92%** | 5 | 1790 | 0.0011 | 52.00% |
| −4 | 44.80% | 44.80% | **97.20%** | 43.62% | 88.00% | 8.80% | 0 | 1790 | 0.0011 | 77.20% |
| 0 | 40.80% | 40.80% | 95.60% | 39.33% | 97.20% | 1.60% | 0 | 1790 | 0.0011 | 80.80% |
| +4 | 32.00% | 32.00% | 90.00% | 30.67% | 99.20% | 0.00% | 0 | 1790 | 0.0011 | 78.00% |

No-CoT 下，固定工作点 `−6` 的准确率为 40.80%，与 baseline 的 41.60% 基本相同（Δ=−0.80 pp，Holm `p_adj=1.000`），因此没有检测到正向迁移。`+4` 的准确率进一步降至32.80%。

CoT 下结果明显不同。`−6` 将 `first_acc` 从 40.80% 提高至 56.80%（Δ=+16.00 pp，95% CI=[+8.80,+23.20]，Holm `p_adj=2.25×10⁻⁴`）。该提升也出现在 `cond_acc`，说明结果不能简单归因于更多样本生成了正式答案。

行为上，CoT `−6` 的 `early_cand_rate` 从 baseline 的 97.20% 降至 63.20%，`reason_first_rate` 从 1.60% 升至 39.92%。相比之下，No-CoT `−6` 虽然也有较小的 ordering change，但准确率没有提高。这说明 output reordering 与有效 workpoint 相关，但不能单独保证性能提升。

#### Qwen2.5-7B

##### No-CoT

**Table 6.11. Qwen BBH Object Counting No-CoT performance and output behavior**

| α | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 56.80% | 57.60% | 86.40% | 53.24% | 100.00% | 0.00% | 0 | 418 | 0.7428 | 20.00% |
| 0 | 55.20% | 56.00% | 86.80% | 52.53% | 100.00% | 0.00% | 0 | 422 | 0.7568 | 15.60% |
| +8 | **57.60%** | 56.40% | **97.20%** | **56.38%** | **44.40%** | **37.40%** | **50** | **150** | 0.6986 | 30.80% |

##### CoT

**Table 6.12. Qwen BBH Object Counting CoT performance and output behavior**

| α | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 45.60% | 46.00% | 43.20% | 62.04% | 100.00% | 0.00% | 0 | 20 | 0.6497 | 8.80% |
| 0 | 52.80% | 54.00% | 38.40% | 62.50% | 100.00% | 0.00% | 0 | 18 | 0.3227 | 8.40% |
| +6 | 43.20% | 44.00% | 52.40% | 53.44% | 96.80% | 3.20% | 0 | 89 | 0.6673 | 23.60% |
| +8 | **66.80%** | **66.40%** | **97.60%** | **68.44%** | **7.20%** | **80.32%** | **211** | 143 | 0.6868 | 23.60% |

No-CoT 下，固定工作点 `+8` 将准确率从 55.20% 提高至 57.60%，但差异没有通过校正（Δ=+2.40 pp，Holm `p_adj=1.000`）。尽管 `early_cand_rate` 从 100.00% 降至 44.40%，`reason_first_rate` 从 0.00% 升至 37.40%，明显的 output reordering 并未转化为可检测的准确率提升。

CoT 下，`+8` 将 `first_acc` 从 52.80% 提高至 66.80%（Δ=+14.00 pp，95% CI=[+7.60,+20.40]，Holm `p_adj=2.25×10⁻⁴`）。与此同时，`early_cand_rate` 从 100.00% 降至 7.20%，`reason_first_rate` 从 0.00% 升至 80.32%。

Qwen CoT 的提升还伴随 `valid_sub_rate` 从 38.40% 升至 97.60%，而 `cond_acc` 从 62.50% 升至 68.44%。这说明总体增益同时包含正式答案提交恢复和 answered-sample accuracy 的改善，不能全部归因于推理能力变化。具体的 marker parsing 诊断记录于 `CLAUDE.md`。

#### Summary

BBH Object Counting 的 steering 效果明显依赖 CoT：

- No-CoT 下，Llama `−6` 和 Qwen `+8` 均未显著提高准确率。
- CoT 下，Llama `−6` 和 Qwen `+8` 分别提高 16.00 pp 和 14.00 pp，且均通过 Holm 校正。
- 两个有效 CoT workpoint 都伴随更少的 early candidate 和更多 reason-first output。
- Qwen No-CoT `+8` 已经产生明显的 output reordering，却没有显著提高准确率，再次说明该行为不是充分条件。

**Conclusion.** BBH Object Counting 支持 CoT 条件下的 fixed-workpoint transfer，但不支持 No-CoT 下的稳定迁移。行为重组与有效结果相伴，但输出格式恢复和任务条件同样重要。

### 6.4 CRUXEval-O

CRUXEval-O 使用 Python literal 作为答案，因此本节将整体 `first_acc` 作为主要性能指标，`last_acc` 用于观察答案修订。Bare 与 Chat 使用不同的输出接口，结果分别报告，不合并为同一统计比较。

> **Metric note.** `answer_first_rate` 和 `pre_marker_chars_med` 均以正式 `####` marker 为基准，不等同于答案 candidate 的形成时间。`marker_present_rate` 只表示输出中出现过 marker，`valid_sub_rate` 则要求其内容能够被成功解析。`nonliteral_rate` 用于识别 marker 后内容的格式失败。`truncated_rate` 仅在 Chat 数据中可用；Bare 中缺少该元数据，不能据此判断是否发生截断。

#### Bare Performance

**Table 6.13. CRUXEval-O Bare performance**

| Model | Condition | α | n | first_acc | last_acc | valid_sub_rate | cond_acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama3.1-8B | No-CoT | −6 | 300 | 31.00% | 28.67% | 96.33% | 32.18% |
| Llama3.1-8B | No-CoT | −4 | 300 | 33.33% | 30.33% | 96.67% | 34.48% |
| Llama3.1-8B | No-CoT | 0 | 300 | 34.67% | 31.00% | 96.67% | 35.86% |
| Llama3.1-8B | No-CoT | +4 | 300 | 33.67% | 30.67% | 97.67% | 34.47% |
| Llama3.1-8B | CoT | −6 | 300 | 34.00% | 32.33% | 97.00% | 35.05% |
| Llama3.1-8B | CoT | −4 | 300 | 35.67% | 34.33% | 95.67% | 37.28% |
| Llama3.1-8B | CoT | 0 | 300 | 34.67% | 37.00% | 97.00% | 35.74% |
| Llama3.1-8B | CoT | +4 | 300 | 32.33% | 36.00% | 97.33% | 33.22% |
| Qwen2.5-7B | No-CoT | −6 | 300 | 24.00% | 23.67% | 77.67% | 30.90% |
| Qwen2.5-7B | No-CoT | 0 | 300 | 29.33% | 28.33% | 87.33% | 33.59% |
| Qwen2.5-7B | No-CoT | +6 | 300 | 30.33% | 25.00% | 85.33% | 35.55% |
| Qwen2.5-7B | No-CoT | +8 | 300 | **37.67%** | 22.33% | 86.67% | 43.46% |
| Qwen2.5-7B | CoT | −6 | 300 | 34.67% | 31.00% | 85.67% | 40.47% |
| Qwen2.5-7B | CoT | 0 | 300 | 34.67% | 28.67% | 83.67% | 41.43% |
| Qwen2.5-7B | CoT | +6 | 300 | 43.00% | 24.67% | 85.00% | 50.59% |
| Qwen2.5-7B | CoT | +8 | 300 | **54.00%** | 29.00% | 95.67% | 56.45% |

Llama 在 Bare 接口下没有表现出稳定收益。No-CoT 中各 α 均未超过 baseline；CoT 的最高点为 `−4`，但只比 baseline 高 1.00 pp，不支持稳定的 workpoint transfer。

Qwen 的结果不同。No-CoT 在 `+8` 时，`first_acc` 从 29.33% 提高至 37.67%，增加 8.33 pp（`p_adj=.0045`）；CoT 在 `+8` 时从34.67%提高至54.00%，增加19.33 pp（`p_adj=2.63×10⁻⁹`）。因此，CRUXEval-O 的正向 transfer 主要出现在 Qwen，且 CoT 条件下效果更强。

不过，Qwen 的 `last_acc` 没有同步提高：No-CoT `+8` 为22.33%，CoT `+8` 为29.00%。这说明性能增益主要体现在第一次正式提交，后续答案修订反而经常破坏原本的答案。

#### Qwen CoT Output Reordering

**Table 6.14. Qwen CoT formal-marker ordering**

| α | first_acc | last_acc | answer_first_rate | pre_marker_chars_med | posN_med | multi_marker_rate | gen_chars_med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 34.67% | 31.00% | 4.33% | 47 | 0.1866 | 69.33% | 670 |
| 0 | 34.67% | 28.67% | 7.33% | 46 | 0.3237 | 69.67% | 552 |
| +6 | 43.00% | 24.67% | 0.00% | 474 | 0.9035 | 67.67% | 589 |
| +8 | **54.00%** | 29.00% | 0.00% | 686 | 0.9490 | 69.67% | 804 |

从 α=0 增加到 `+6/+8` 后，正式答案标记明显后移：`answer_first_rate` 降至0%，`pre_marker_chars_med` 从46增加至474和686，`posN_med` 也接近输出末尾。与此同时，`first_acc` 明显提高。

但这组指标描述的是正式 `####` marker 的位置，而不是第一个答案 candidate 的形成时间，不能据此断言模型在内部“更晚决定答案”。此外，约七成输出含有多个 marker，且 `first_acc` 与 `last_acc` 明显分离，说明高剂量同时增加了后续修订的不稳定性。

因此，Qwen CoT 的性能提升与 formal-marker reordering 同时出现，但这种 reordering 只能作为输出行为描述，不能视为准确率提升的因果机制。

#### Llama Chat Interface Control

**Table 6.15. Llama Chat performance and output behavior**

| Condition | α | n | first_acc | last_acc | valid_sub_rate | cond_acc | answer_first_rate | pre_marker_chars_med | posN_med | multi_marker_rate | marker_present_rate | nonliteral_rate | truncated_rate | gen_chars_med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-CoT | −6 | 300 | 46.67% | 46.67% | 97.67% | 47.78% | 0.00% | 449 | 0.9700 | 0.00% | 99.00% | 1.33% | 1.33% | 466 |
| No-CoT | −4 | 300 | 43.33% | 43.33% | 97.00% | 44.67% | 0.00% | 460 | 0.9689 | 0.00% | 98.00% | 1.00% | 2.00% | 483 |
| No-CoT | 0 | 300 | 45.67% | 45.67% | 97.00% | 47.08% | 0.00% | 435 | 0.9684 | 0.00% | 98.33% | 1.33% | 2.00% | 458 |
| No-CoT | +4 | 300 | 41.00% | 41.00% | 97.00% | 42.27% | 0.00% | 186 | 0.9214 | 0.00% | 97.33% | 0.33% | 2.67% | 206 |
| CoT | −6 | 300 | 51.33% | 51.33% | 96.33% | 53.29% | 0.00% | 970 | 0.9853 | 0.00% | 97.00% | 0.67% | 3.00% | 1000 |
| CoT | −4 | 300 | 49.33% | 49.33% | 97.33% | 50.68% | 0.00% | 994 | 0.9855 | 0.00% | 97.33% | 0.00% | 2.67% | 1023 |
| CoT | 0 | 300 | **52.00%** | **52.00%** | 97.67% | 53.24% | 0.00% | 978 | 0.9861 | 0.00% | 97.67% | 0.00% | 2.33% | 1002 |
| CoT | +4 | 300 | 50.33% | 50.33% | 97.00% | 51.89% | 0.00% | 682 | 0.9769 | 0.00% | 97.33% | 0.33% | 3.00% | 710 |

Chat 接口下，`valid_sub_rate` 与 `marker_present_rate` 均超过 96%，而 `nonliteral_rate` 和 `truncated_rate` 很低，说明输出接口整体健康。Llama 的 null result 因而不能简单归因于格式失败。

行为上，负向 α 没有引起明显的 output reordering；`−6/−4` 的正式答案位置和生成长度与 baseline 基本接近。相比之下，`+4` 会明显缩短输出并使正式 marker 提前，但没有提高准确率。No-CoT 的最佳干预点 `−6` 仅比 baseline 高 1.00 pp（Holm-adjusted `p=.7111`）；CoT 的所有干预点均未超过 baseline，差异也不显著。

**Conclusion.** CRUXEval-O 呈现清晰的 model boundary：Qwen，尤其在 CoT 条件下，能够从正向 α 中获得显著的 `first_acc` 提升；Llama 在 Bare 和健康的 Chat 接口下均没有稳定收益。Qwen 的增益伴随正式答案后移，但 `last_acc` 没有同步改善，说明后续答案修订仍不稳定。与此同时，Llama `+4` 虽然改变了输出长度和 marker 位置，却没有带来性能收益，再次说明 output reordering 本身并不足以保证准确率提高。

### 6.5 ProofWriter-OWA

ProofWriter-OWA 要求模型使用严格格式 `#### True/False/Unknown` 提交答案。Bare 接口中存在明显的循环生成和截断，因此该接口只用于记录性能与输出健康度，不用于解释 commitment-related behavior。行为分析学分析主要基于 Chat 接口。

#### Bare Interface Results

**Table 6.16. ProofWriter-OWA Bare performance and interface health**

| Model | α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | loop_rate | truncated_rate | gen_chars_med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama3.1-8B | −6 | 300 | 14.33% | 10.33% | 26.00% | 74.00% | 55.13% | 93.33% | 100.00% | 4070 |
| Llama3.1-8B | −4 | 300 | 12.00% | 8.67% | 22.33% | 77.67% | 53.73% | 90.00% | 100.00% | 4164 |
| Llama3.1-8B | 0 | 300 | 10.33% | 6.00% | 20.67% | 79.33% | 50.00% | 93.33% | 100.00% | 4165 |
| Llama3.1-8B | +4 | 300 | 21.67% | 18.67% | 39.67% | 60.33% | 54.62% | 95.00% | 100.00% | 4310 |
| Qwen2.5-7B | −6 | 300 | 49.33% | 49.00% | 90.33% | 9.67% | 54.61% | 14.33% | 14.67% | 850 |
| Qwen2.5-7B | 0 | 300 | 46.33% | 46.33% | 99.67% | 0.33% | 46.49% | 34.33% | 34.33% | 558 |
| Qwen2.5-7B | +6 | 300 | 49.67% | 49.67% | 100.00% | 0.00% | 49.67% | 32.00% | 32.00% | 611 |
| Qwen2.5-7B | +8 | 300 | 52.00% | 52.00% | 100.00% | 0.00% | 52.00% | 47.33% | 47.33% | 894 |

Llama Bare 的所有剂量均发生 100% 截断，并伴随 90% 以上的循环生成。整体准确率因而主要受到能否完成严格提交的限制，不能用于可靠的行为解释。Qwen Bare 的有效提交率较高，但循环和截断仍随剂量变化，因此也只作为接口对照。

#### Llama3.1-8B Chat CoT

**Table 6.17. Llama Chat CoT performance and output behavior**

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | loop_rate | truncated_rate | gen_chars_med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 300 | **39.33%** | **39.33%** | 73.67% | 26.33% | 53.39% | 0.00% | 1368 | 0.9915 | 0.00% | 17.00% | 21.00% | 1605 |
| −4 | 300 | 34.00% | 34.00% | 59.33% | 40.67% | 57.30% | 0.00% | 1202 | 0.9902 | 0.00% | 26.67% | 29.67% | 1527 |
| 0 | 300 | 33.00% | 33.00% | 60.33% | 39.67% | 54.70% | 0.00% | 1078 | 0.9897 | 0.00% | 25.67% | 29.67% | 1358 |
| +4 | 300 | 31.67% | 31.67% | 55.33% | 44.67% | 57.23% | 0.00% | 929 | 0.9877 | 0.00% | 26.67% | 29.00% | 1203 |

Chat 接口显著缓解了 Llama Bare 中的格式失败。与 baseline 相比，`−6` 的 `first_acc` 从 33.00% 提高至 39.33%，`valid_sub_rate` 从 60.33% 提高至 73.67%，同时循环和截断比例下降。

输出行为呈现连续的长度变化：随着 α 从 `−6` 增加到 `+4`，`pre_marker_chars_med` 从 1368 降至 929，`gen_chars_med` 从 1605 降至 1203。负向 α 因而生成更长的 marker 前文本，但所有条件的 `marker_posN_med` 都接近 1，说明正式答案始终主要位于输出末尾。这更适合解释为生成长度变化，而不是明确的 commitment reordering。

#### Qwen2.5-7B Chat CoT

**Table 6.18. Qwen Chat CoT performance and output behavior**

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | loop_rate | truncated_rate | gen_chars_med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 300 | 0.67% | 0.67% | 1.00% | 99.00% | 66.67% | 0.00% | 500 | 0.9785 | 0.00% | 0.00% | 0.00% | 5 |
| 0 | 300 | 41.00% | 41.00% | 100.00% | 0.00% | 41.00% | 50.33% | 0 | 0.0000 | 0.67% | 0.00% | 0.00% | 231 |
| +6 | 300 | 39.00% | 39.00% | 99.67% | 0.33% | 39.13% | 96.32% | 0 | 0.0000 | 0.00% | 0.00% | 0.00% | 12 |
| +8 | 300 | **47.67%** | **47.67%** | 99.67% | 0.33% | 47.83% | 4.01% | 549 | 0.9778 | 0.00% | 0.00% | 0.00% | 563 |

Qwen Chat 对 α 表现出明显的非线性接口响应。`−6` 几乎完全无法产生严格答案，`valid_sub_rate` 只有 1.00%，因此该点属于格式崩溃，不能作为正常的性能或行为状态解释。

`+6` 仍能稳定提交答案，但 96.32% 的有效输出直接以 marker 开始，生成长度中位数只有 12 个字符，准确率也未超过 baseline。到 `+8` 时，输出重新转为先生成较长文本、再在末尾提交答案：`answer_first_rate` 降至 4.01%，`pre_marker_chars_med` 增至 549，`first_acc` 同时从 41.00% 提高至 47.67%。

这一变化不是随 α 平滑移动的单一 output-reordering curve，而是多个输出状态之间的切换。`+8` 的性能提升与更长的 marker 前文本同时出现，但 `+6` 和 `−6` 的反例说明，单独使用 marker timing 无法预测准确率。

> **Metric note.** `answer_first_rate`、`pre_marker_chars_med` 和 `marker_posN_med` 均以严格答案 marker 为基准，只描述可观察的输出顺序，不等同于答案 candidate 的形成时间或内部 commitment。长度和位置指标还会受到循环、截断及无有效提交的影响。

**Conclusion.** Chat 接口修复了 Llama Bare 中的大部分输出问题，并在 `−6` 获得较高的准确率和有效提交率；Qwen 则在 `+8` 获得最高准确率。两种模型的有效方向与 GSM8K-derived workpoint 一致，但 ProofWriter-O​​WA 的行为变化高度依赖模型与接口，不能概括为统一的“先思考、再回答”效应。

### 6.6 FinQA

FinQA 的答案形式比 GSM8K 更复杂。`first_acc` 仍作为主要性能指标；candidate-based metrics 使用 GSM8K candidate detector，因此只用于探索性描述，不作为 FinQA 内部答案形成时间的直接测量。

#### Llama3.1-8B-Instruct

**Llama3.1-8B-Instruct — FinQA CoT**

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate | loop_rate | truncated_rate | gen_chars_med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 300 | 8.67% | 8.00% | 99.67% | 8.70% | 34.67% | 3.97% | 5 | 828 | 0.0053 | 0.0000 | 10.33% | 93.33% | 100.00% | 868 |
| −4 | 300 | 9.00% | 7.67% | 99.33% | 9.06% | 75.33% | 2.53% | 0 | 804 | 0.0000 | 0.0057 | 19.67% | 95.33% | 100.00% | 868 |
| 0 | 300 | 14.33% | 12.67% | 99.67% | 14.38% | 89.67% | 3.46% | 0 | 826 | 0.0000 | 0.0077 | 21.67% | 96.67% | 100.00% | 847 |
| +4 | 300 | **15.33%** | **13.67%** | 94.67% | 16.20% | 81.33% | 10.49% | 0 | 872 | 0.0000 | 0.0113 | 30.67% | 93.33% | 100.00% | 891 |

Llama 在所有剂量下都达到 100% `truncated_rate`，并伴随超过 93% 的 `loop_rate`。因此，较大的 `post_cand_chars_med` 和 `gen_chars_med` 主要反映循环生成，而不能解释为更长或更充分的推理。

性能上，GSM8K-derived workpoint `−6` 的 `first_acc` 从 baseline 的 14.33% 降至 8.67%；`−4` 同样下降。虽然 `−6` 的 `early_cand_rate` 明显降低，但准确率没有改善，而且输出几乎全部截断。这一结果不能支持“candidate 后移带来更好推理”的解释。

`+4` 的点估计略高于 baseline，但提升仅为 1.00 pp，同时有效提交率下降、多个 marker 的比例上升。整体来看，Llama 在 FinQA 上没有表现出可靠的 fixed-workpoint transfer。

#### Qwen2.5-7B-Instruct

**Qwen2.5-7B-Instruct — FinQA CoT**

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate | loop_rate | truncated_rate | gen_chars_med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 300 | 10.00% | 9.67% | 91.67% | 10.91% | 100.00% | 0.00% | 0 | 742 | 0.0000 | 0.5400 | 42.67% | 19.00% | 28.67% | 746 |
| 0 | 300 | 20.67% | 20.67% | 92.67% | 22.30% | 99.33% | 0.00% | 0 | 512 | 0.0000 | 0.3773 | 36.33% | 15.33% | 20.67% | 516 |
| +6 | 300 | 20.67% | 20.33% | 96.67% | 21.38% | 54.00% | 37.37% | 5 | 288 | 0.0122 | 0.6611 | 34.33% | 14.00% | 18.33% | 516 |
| +8 | 300 | **25.67%** | **24.33%** | 97.67% | 26.28% | 43.33% | 49.15% | 150 | 259 | 0.4138 | 0.2812 | 48.67% | 12.33% | 16.67% | 554 |

Qwen 的正向 α 产生了较清晰的 output reordering。相对于 baseline：

- `+6` 将 `early_cand_rate` 从 99.33% 降至 54.00%，`reason_first_rate` 从 0.00% 升至 37.37%，但 `first_acc` 保持在 20.67%。
- `+8` 进一步将 `early_cand_rate` 降至 43.33%，并将 `reason_first_rate` 提高至 49.15%；`first_acc` 同时提高至 25.67%，点估计增加 5.00 pp。
- `loop_rate` 和 `truncated_rate` 随正向 α 下降，说明这些行为变化不能完全归因于更严重的循环或截断。

不过，`+6` 已产生明显的 output reordering，却没有提高准确率，说明这种变化本身并不充分。`+8` 的 `multi_marker_rate` 升至 48.67%，且 `last_acc` 低于 `first_acc`，也表明重复提交和后续修订仍可能破坏答案。

> **Metric note.** `candidate_posN_med` 使用 GSM8K candidate detector，在 FinQA 上属于探索性指标；`posN_med` 测量的是第一个合法 `#### <value>` marker。两者检测的对象和统计子集不同，不能直接比较。长度指标还会受到循环与截断影响。

**Conclusion.** FinQA 没有表现出稳定的跨任务 workpoint transfer。Llama 的负向 workpoint 降低了准确率，并受到严重循环和截断影响；Qwen `+8` 的准确率点估计有所提高，并伴随较少的 early candidate 和更多的 reason-first output，但 `+6` 的行为反例说明 output reordering 不能单独预测性能。
    
### 6.4 ZebraLogic-Easy

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

### 6.7 ZebraLogic-Easy

ZebraLogic-Easy 的答案是完整逻辑网格，因此行为指标围绕 solution grid 的出现位置和修订情况定义。行为指标采用 gold-free 分析；准确率来自冻结的正式评测，未由本次行为脚本重新计算。每个条件均包含 280 个样本。

#### Llama3.1-8B-Instruct

**Llama3.1-8B-Instruct — ZebraLogic-Easy**

| α | first_puzzle_acc | last_puzzle_acc | first_cell_acc | last_cell_acc | valid_sub_rate | full_grid_rate | reason_before_solution_rate | pre_solution_chars_med | solution_posN_med | grid_agreement_med | grid_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 35.71% | 33.93% | 48.55% | 41.91% | 75.71% | 71.07% | 75.71% | 544 | 0.1295 | 1.0000 | 9.47% | 90.71% | 100.00% | 4192 |
| −4 | 33.93% | 32.86% | 47.36% | 42.09% | 75.00% | 69.64% | 75.00% | 515 | 0.1236 | 1.0000 | 7.94% | 93.21% | 100.00% | 4162 |
| 0 | **36.79%** | **37.14%** | **48.73%** | **48.68%** | 74.64% | 67.86% | 74.64% | 520 | 0.1270 | 1.0000 | 10.19% | 95.00% | 100.00% | 4118 |
| +4 | 32.14% | 30.00% | 46.27% | 38.64% | 75.00% | 70.71% | 75.00% | 522 | 0.1234 | 1.0000 | 9.34% | 90.00% | 100.00% | 4168 |

Llama 的各项行为指标基本保持不变。`valid_sub_rate` 约为 75%，`pre_solution_chars_med` 维持在 515–544，`solution_posN_med` 也集中在 0.12 附近，没有出现清晰的 dose-dependent reordering。

准确率同样没有改善。baseline 的 `first_puzzle_acc` 为 36.79%，`−6`、`−4` 和 `+4` 均低于该点；`last_puzzle_acc` 与 `last_cell_acc` 也没有显示有效 transfer。

所有条件的 `truncated_rate` 均为 100%，`loop_rate` 超过 90%，且生成 token 数中位数均达到上限 1024。因此，较长的输出不能解释为更充分的推理，solution position 和生成长度也需要谨慎解读。

#### Qwen2.5-7B-Instruct

**Qwen2.5-7B-Instruct — ZebraLogic-Easy**

| α | first_puzzle_acc | last_puzzle_acc | first_cell_acc | last_cell_acc | valid_sub_rate | full_grid_rate | reason_before_solution_rate | pre_solution_chars_med | solution_posN_med | grid_agreement_med | grid_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| −6 | 30.00% | 16.07% | 58.32% | 35.14% | 100.00% | 92.50% | 100.00% | 508 | 0.1669 | 0.1111 | 67.27% | 6.07% | 60.36% | 3906 |
| 0 | 34.64% | 15.36% | 64.05% | 28.45% | 100.00% | 93.57% | 100.00% | 506 | 0.1492 | 0.0714 | 76.34% | 7.50% | 72.86% | 3967 |
| +6 | **36.07%** | 12.14% | **65.64%** | 35.95% | 99.64% | 96.79% | 99.64% | 546 | 0.1614 | 0.2000 | 60.65% | 9.29% | 68.57% | 3956 |
| +8 | 23.93% | **24.64%** | 23.73% | 24.18% | 39.29% | 38.93% | 39.29% | 479 | 0.3411 | 1.0000 | 2.75% | 62.50% | 63.57% | 3070 |

Qwen 在 `−6`、baseline 和 `+6` 下几乎总能生成答案，并且 `reason_before_solution_rate` 接近 100%。该指标在这些条件下已经饱和，无法区分不同剂量的输出组织方式。

`+6` 的 `first_puzzle_acc` 从 34.64% 小幅提高至 36.07%，`first_cell_acc` 从 64.05% 提高至 65.64%，但 `last_puzzle_acc` 反而下降。因此，这只能视为有限的点估计变化，不能支持稳定的 workpoint transfer。

`+8` 则出现明显的输出崩溃：`valid_sub_rate` 从 100.00% 降至 39.29%，`full_grid_rate` 降至 38.93%，`first_puzzle_acc` 和 `first_cell_acc` 也分别降至 23.93% 和 23.73%。此时较低的 grid disagreement 主要来自能够产生完整答案的选择性子集，不能解释为答案修订变得更加稳定。

所有 Qwen 条件的生成 token 数中位数同样达到上限 1024，且截断率较高。因此，solution timing 和修订指标仍会受到截断及有效提交率变化的影响。

> **Metric note.** `reason_before_solution_rate` 和 `solution_posN_med` 描述可观察的 solution grid 位置，不代表内部 commitment。`grid_agreement_med` 只在存在可比较网格的样本中计算；当 `valid_sub_rate` 明显变化时，不应直接跨条件比较。`no_answer_rate` 是 `100% − valid_sub_rate`，因此未在合并表中重复列出。

**Conclusion.** ZebraLogic-Easy 没有显示稳定的 fixed-workpoint transfer。Llama 的性能和行为指标整体接近 baseline；Qwen `+6` 只有有限的 first-answer 点估计提升，而 `+8` 出现明显的格式与性能崩溃。该任务进一步说明，reason-before-solution 行为可能已经饱和，不能单独预测准确率。

### 6.9 Cross-Benchmark Summary

**Table Cross-benchmark workpoint outcomes**

| Benchmark | Llama3.1-8B | Qwen2.5-7B | Main boundary |
| --- | --- | --- | --- |
| GSM-Hard | Supported in No-CoT and CoT | Supported in No-CoT and CoT | Workpoint extends to a nearby hard arithmetic task |
| GSM-Symbolic | Supported in No-CoT; not stable in CoT | Supported in No-CoT and CoT | Transfer depends on model and prompting condition |
| BBH Object Counting | CoT only | CoT only | CoT-dependent transfer; No-CoT reordering is insufficient |
| CRUXEval-O | No stable gain | Supported mainly under CoT | Clear model boundary; revision remains unstable |
| ProofWriter-OWA | Descriptive gain at `−6` under Chat | Descriptive gain at `+8` under Chat | Strong interface dependence; Bare results are format-limited |
| LogiQA 2.0 | No stable gain | No stable gain | Multiple-choice output does not show reliable transfer |
| ZebraLogic-Easy | No stable gain | No stable gain; `+8` collapses | Reason-before-solution is saturated and not predictive |
| FinQA | No stable gain | Modest descriptive gain at `+8` | Reordering appears in Qwen but does not establish stable transfer |

跨任务结果不支持一个对所有任务都有效的单点 workpoint。更合适的结论是：每个模型存在一个在部分相近任务中可迁移的 workpoint range，但其效果受到 benchmark、prompting condition 和输出接口的共同限制。

行为结果同样没有形成统一的“先思考、再回答”规律。有效 α 经常伴随更少的 early candidate、更多的 reason-first output 或更晚的正式提交，但这些变化既不是性能提升的必要条件，也不是充分条件。部分任务出现明显的 output reordering 却没有准确率收益，另一些任务则受到截断、循环、格式崩溃或行为指标饱和的影响。

**Conclusion.** Fixed-workpoint transfer 在 GSM-Hard、GSM-Symbolic、BBH Object Counting 和 Qwen CRUXEval-O 上获得不同程度的支持，但不能推广到所有任务。RSN 能够调节 commitment-related output behavior，但行为方向及其与准确率的关系具有明显的 model、task、condition 和 interface dependence。