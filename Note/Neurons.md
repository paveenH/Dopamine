# Neural Representations and Functional Relationships

## 1. Scope and Main Findings

本文分别整理 Role–Confidence 与 Role–Chat 两组关系。前者同时包含 representation 与 intervention-based functional evidence；后者目前主要是 hidden-state structural evidence，Chat 的行为 dose response 单独见 `ReasoningChat.md`。hidden-state projection、cosine、top-coordinate overlap 与 transition subspace 都只能描述表征关系，不能单独证明共享因果机制。

Role neurons（RSNs）与 Confidence neurons（CSNs）在表征上明显相关，但目前没有证据表明两者具有相同功能。

- **Representation:** 两个方向在 Layer 11–19 的 cosine similarity 为 **0.6063**，并共享 **46/180（25.6%）** 个 top-neuron positions。
- **Organization:** 整体 alignment 主要来自广泛分布的非 top positions；共享 top neurons 数量少，但单位贡献高度富集。
- **MMLU-E:** 小剂量 CSN steering 能减少 E（“I am not sure”）选择，并在 confident prompt 下保持或小幅提高准确率；较大剂量则明显损害答案选择。
- **GSM8K:** CSN steering 会改变提前输出、规范提交和重复行为，但没有复现 RSN 的有效推理工作区间。

最简洁的结论是：

> **RSNs 与 CSNs 共享一部分表征基础，但不是可直接互换的功能机制。**

## 2. Role–Confidence Relationship

### 2.1 Direction and Layerwise Alignment

两个方向定义为：

$$
d_{\mathrm{role}}=\mu_{\mathrm{expert}}-\mu_{\mathrm{nonexpert}},
\qquad
d_{\mathrm{confidence}}=\mu_{\mathrm{confident}}-\mu_{\mathrm{unconfident}}.
$$

| Direction Metric | Result |
|---|---:|
| Concatenated cosine similarity, Layer 11–19 | **0.6063** |
| Concatenated Pearson correlation, Layer 11–19 | **0.6063** |
| Mean layerwise cosine | 0.4991 |
| Median layerwise cosine | 0.4752 |
| Minimum layerwise cosine | 0.1584 (Layer 11) |
| Maximum layerwise cosine | **0.7641 (Layer 19)** |
| Role-direction L2 norm | 4.4395 |
| Confidence-direction L2 norm | 12.5108 |
| Confidence/Role norm ratio | **2.82** |

两种方向从 Layer 11 起总体更接近，并在 Layer 19 达到最高相似度。Confidence direction 的整体幅度约为 Role direction 的 **2.82 倍**，因此相同 raw α 不能被视为等强度干预。

Confident 与 unconfident 条件的分化也主要出现在这一中后层区间：

| Layerwise Metric | Layer 1–10 | Layer 11–19 | Change |
|---|---:|---:|---:|
| Confident–Unconfident correlation | 0.9886 | **0.8783** | Decreased |
| Derived divergence, $1-r$ | 0.0114 | **0.1217** | About 10× higher |
| Confidence-direction norm | 0.4571 | **3.9009** | About 8.5× higher |
| Minimum in-band correlation | — | **0.8192 (Layer 19)** | — |
| Maximum in-band direction norm | — | **About 6.4 (Layer 19)** | — |

这说明显式 confidence 条件与原 expert/non-expert 条件在相近层段形成差异，但层级位置相近本身不等于功能相同。

### 2.2 Top-Neuron Overlap

在 Layer 11–19，每层分别选取 20 个 Role top neurons 和 20 个 Confidence top neurons。随机情况下，每层期望重叠仅为 $20\times20/4096\approx0.098$。

| Layer | Shared Positions | Overlap Rate | Jaccard | Sign Agreement | One-Sided $p$ |
|---:|---:|---:|---:|---:|---:|
| 11 | 2/20 | 10% | 0.053 | 50.0% | $4.08\times10^{-3}$ |
| 12 | 0/20 | 0% | 0.000 | — | 1.00 |
| 13 | 3/20 | 15% | 0.081 | 66.7% | $1.08\times10^{-4}$ |
| 14 | 5/20 | 25% | 0.143 | 100% | $2.40\times10^{-8}$ |
| 15 | 5/20 | 25% | 0.143 | 100% | $2.40\times10^{-8}$ |
| 16 | 8/20 | 40% | 0.250 | 100% | $7.88\times10^{-15}$ |
| 17 | 6/20 | 30% | 0.176 | 83.3% | $2.21\times10^{-10}$ |
| **18** | **10/20** | **50%** | **0.333** | **100%** | $9.21\times10^{-20}$ |
| 19 | 7/20 | 35% | 0.212 | 100% | $1.52\times10^{-12}$ |
| **Layer 11–19** | **46/180** | **25.6%** | **0.147** | — | — |

除 Layer 12 外，各层重叠都高于随机预期，Layer 18 的重叠最高。Layer 14–19 的共享 neurons 也几乎都具有相同符号。两组 neurons 因而共享一个稳定核心，但 **46/180** 的总重叠也清楚表明它们并非同一组 neurons。

### 2.3 Distribution of Alignment

Layer 11–19 的 36,864 个 neuron positions 可分为四组：

| Neuron Group | Count | Signed Dot Share | Absolute Alignment Share | Role Energy | Confidence Energy | Relative Contribution per Position |
|---|---:|---:|---:|---:|---:|---:|
| Shared top | 46 | **8.0%** | **6.5%** | 5.63% | 5.25% | **1.0× (reference)** |
| Role only | 134 | 2.9% | 2.5% | 3.75% | 1.13% | About 1/8× |
| Confidence only | 134 | 2.8% | 2.5% | 0.96% | 4.61% | About 1/8× |
| Neither top | 36,550 | **86.3%** | **88.6%** | **89.65%** | **89.01%** | About 1/74× |

从总量看，alignment 主要来自 neither-top positions；从单位密度看，46 个 shared-top positions 的贡献约为 role-only 或 confidence-only 的 **8 倍**，约为 neither-top 的 **74 倍**。

因此，最符合数据的结构是：

> **广泛分布的表征对齐背景，加上一个稀疏、方向一致且高度富集的共享核心。**

这并不意味着所有非 top neurons 都均匀贡献；大量 alignment 仍可能集中在刚好未进入 top-20 的较高排名 neurons 中。

### 2.4 Functional Evidence on MMLU-E

#### 2.4.1 RSN Steering under Confidence Prompts

在 confident prompt 下，RSN steering 能明显减少 E 选择，同时基本保持准确率。

| Condition | STEM Acc. | STEM E-rate | Humanities Acc. | Humanities E-rate | Social Sciences Acc. | Social Sciences E-rate | Other Acc. | Other E-rate | Task-Macro Acc. | Task-Macro E-rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Confident baseline | 55.03% | 2.72% | 68.16% | 3.34% | 73.65% | 1.60% | 69.06% | 1.54% | 65.14% | 2.36% |
| RSN +3, Layer 11–19 | 56.17% | 0.40% | 69.39% | 0.93% | 74.10% | 0.85% | 68.93% | 0.59% | **65.87%** | 0.66% |
| RSN +3, Layer 11–19, t4 | 55.70% | 0.23% | **69.81%** | 0.39% | **74.01%** | 0.61% | 68.69% | 0.34% | 65.74% | 0.37% |
| RSN +4, Layer 11–19 | 55.14% | **0.09%** | 69.38% | **0.34%** | 73.68% | **0.54%** | 68.70% | **0.27%** | 65.38% | **0.28%** |

Task-macro E-rate 从 **2.36%** 降至 **0.66%、0.37% 和 0.28%**，准确率维持在 **65.38%–65.87%**。这说明 RSNs 能调节不确定性表达，而不必同步损害知识选择。

#### 2.4.2 CSN Dose Response

CSN steering 对 confident 与 unconfident prompts 均有因果影响，但小剂量与大剂量的行为不同。

| α | Prompt | Task-Macro Acc. | E-rate | Wrong Non-E |
|---:|---|---:|---:|---:|
| 0 | Confident | 65.20% | 2.31% | 32.49% |
| +0.5 | Confident | 66.62% | 0.75% | 32.62% |
| +1 | Confident | 66.68% | 0.29% | 33.03% |
| +2 | Confident | 56.99% | 1.51% | 41.49% |
| +4 | Confident | 32.39% | 0.02% | 67.60% |
| 0 | Unconfident | 0.32% | 99.39% | 0.28% |
| +0.5 | Unconfident | 2.57% | 96.08% | 1.36% |
| +1 | Unconfident | 12.13% | 81.58% | 6.30% |
| +2 | Unconfident | 53.62% | 9.04% | 37.33% |
| +4 | Unconfident | 27.42% | 0.02% | 72.56% |

Conditional accuracy 衡量模型没有选择 E 时的正确率。它有助于区分“少弃权”与“非 E 答案质量”，但不同剂量下进入该子集的样本会变化，因此不能单独证明知识能力提高。

| α | Confident Conditional Acc. (95% CI) | Δ vs 0 (95% CI) | Unconfident Conditional Acc. (95% CI) | Δ vs 0 (95% CI) |
|---:|---:|---:|---:|---:|
| 0 | 66.56% [65.78, 67.33] | — | 49.46% [39.21, 59.60] | — |
| +0.5 | 67.19% [66.40, 67.95] | +0.63 pp [+0.29, +0.98] | 60.35% [56.60, 64.12] | **+10.89 pp [+1.60, +20.22]** |
| +1 | 67.09% [66.31, 67.87] | +0.53 pp [+0.03, +1.02] | 60.22% [58.54, 61.91] | **+10.76 pp [+0.97, +21.14]** |
| +2 | 56.31% [55.50, 57.13] | −10.25 pp [−11.11, −9.41] | 58.24% [57.39, 59.07] | +8.78 pp [−1.42, +19.16] |
| +4 | 31.83% [31.07, 32.61] | −34.73 pp [−35.80, −33.64] | 27.36% [26.61, 28.09] | −22.10 pp [−32.41, −11.99] |

结果可以概括为：

- **α=+0.5/+1:** confident prompt 的准确率与 conditional accuracy 保持或小幅提高，同时 E-rate 降低；这是当前最干净的低代价区间。
- **Unconfident prompt:** α=+1 时 E-rate 已从 **99.39%** 降至 **81.58%**。Conditional accuracy 同时上升，但 baseline 的非 E 子集很小、跨剂量子集组成也不同，因此应解释为“回答倾向和条件正确率共同变化”，而不是知识被恢复。
- **α=+2:** unconfident accuracy 的大幅上升主要伴随 E-rate 从 **99.39%** 降至 **9.04%**；其 conditional-accuracy 差异区间跨 0，不能确认额外的答案质量提升。Confident accuracy 已下降 **8.21 pp**。
- **α=+4:** 两种 prompt 都进入过度 steering；E-rate 接近 0，但 confident 与 unconfident conditional accuracy 分别降至 **31.83%** 和 **27.36%**。

#### 2.4.3 RSN–CSN Comparison under the Confident Prompt

| Steering | α | Task-Macro Acc. | Δ Acc. | E-rate | Δ E-rate |
|---|---:|---:|---:|---:|---:|
| RSN baseline | 0 | 65.14% | — | 2.36% | — |
| RSN | +3 | 65.87% | +0.73 pp | 0.66% | −1.70 pp |
| RSN | +4 | 65.38% | +0.24 pp | 0.28% | −2.08 pp |
| CSN baseline | 0 | 65.20% | — | 2.31% | — |
| CSN | +0.5 | 66.62% | +1.42 pp | 0.75% | −1.56 pp |
| CSN | +1 | 66.68% | +1.48 pp | 0.29% | −2.02 pp |
| CSN | +2 | 56.99% | −8.21 pp | 1.51% | −0.80 pp |
| CSN | +4 | 32.39% | −32.81 pp | 0.02% | −2.29 pp |

两种 masks 都能降低不确定性表达，但功能曲线不同：RSN 在已测试剂量下基本保留准确率；CSN 只有较小正向剂量保持稳定，剂量继续增加后会快速损害 A–D 答案选择。

这两组结果来自不同实验链，且 masks 未做 norm matching。因此只能比较行为形态，不能用相同 raw α 比较绝对强弱。

### 2.5 Functional Evidence on GSM8K

#### 2.5.1 CSN Steering on GSM8K

在 GSM8K 的 300 道题上，我们对 Confidence-neuron steering（CSN）测试了九个剂量。准确率采用冻结的 offline `first_acc` 口径；行为指标沿用统一分析模板。

| α | First acc. | Last acc. | Valid submission | Conditional acc. | Early candidate | Candidate coverage | Reason first | Median `posN` | Multiple markers | Median gen. chars | Δ First acc. vs 0（95% CI） | McNemar raw p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −4 | 51.00% | 49.67% | 36.67% | 63.64% | 72.33% | 99.33% | 10.74% | 0.1876 | 16.00% | 2,289 | — | — | — |
| −2 | 60.33% | 56.33% | 55.00% | 61.21% | 60.33% | 94.67% | 34.51% | 0.1697 | 19.67% | 2,118 | — | — | — |
| **−1** | **66.67%** | **63.33%** | 64.33% | 73.06% | **41.00%** | 98.00% | 35.71% | 0.2336 | 22.33% | 2,107 | **+6.33 pp** `[+1.00,+11.67]` | 0.0248 | 0.0993 |
| −0.5 | 63.33% | 60.67% | 62.33% | 71.66% | 48.33% | 97.67% | 30.72% | 0.1934 | 21.33% | 2,126 | +3.00 pp `[−1.33,+7.33]` | 0.2221 | 0.2221 |
| 0 | 60.33% | 55.67% | 62.00% | 66.67% | 48.33% | 98.00% | 30.95% | 0.1743 | 22.33% | 2,164 | Reference | — | — |
| +0.5 | 65.33% | 61.33% | 60.33% | **73.48%** | 46.33% | 97.00% | 28.52% | 0.1861 | 19.67% | **2,078** | +5.00 pp `[+0.33,+9.67]` | 0.0489 | 0.1438 |
| +1 | 65.67% | 61.67% | **65.00%** | 69.74% | 44.00% | 96.33% | 34.60% | 0.1921 | 26.00% | 2,090 | +5.33 pp `[+0.33,+10.33]` | 0.0479 | 0.1438 |
| +2 | 55.67% | 53.33% | 53.33% | 63.75% | 51.00% | 95.33% | **51.05%** | 0.1619 | 21.00% | 2,229 | — | — | — |
| +4 | 46.33% | 46.00% | 57.33% | 48.26% | 69.33% | 98.00% | 24.83% | 0.2026 | 17.00% | 2,273 | — | — | — |

小剂量区域的准确率高于 α=0：α=−1 和 +1 分别达到 **66.67%** 和 **65.67%**，对应原始提升 **6.33 pp** 和 **5.33 pp**。但四组 small-dose 比较经过 Holm 校正后均未达到显著水平（最小 `p_adj=0.0993`），因此这些结果应解释为正向趋势，而不是已经确认的准确率提升。探索性比较也无法区分 α=−1 与 +1（差异 −1.00 pp，McNemar `p=0.818`），尚不支持正负方向具有不同功能。

行为上，α=−1 同时表现出较低的 early-candidate rate（**41.00%**，baseline 为 48.33%）、较高的 valid-submission rate（64.33%）和 conditional accuracy（73.06%），方向上与“减少过早候选输出、提高答案质量”一致。不过这些关系并不单调：例如 α=+2 的 reason-first rate 最高（51.05%），但准确率下降至 55.67%。因此，更多前置推理或更晚输出答案本身并不足以保证正确率提高。

`Conditional accuracy` 和 `posN` 仅在产生有效正式提交的样本中定义；`reason-first` 使用 candidate-covered subset。`posN` 表示正式 `####` 标记在输出文本中的相对位置，不代表模型内部的 commitment 时间。上述行为变化均为干预后的描述性共现，不能解释为准确率变化的因果中介。

#### 2.5.2 RSN–CSN Functional Comparison

九点结果修正了早期五点粗网格下“CSN 没有高于 baseline 的剂量点”这一判断：CSN 在 α≈±1 附近确实出现了约 **5–6 pp** 的小剂量高响应区域。但这一提升尚未通过多重比较校正，而且正负剂量表现近似，未呈现明确的方向特异性。

相比之下，同一 GSM8K 条件下，RSN steering 从 α=0 的约 **60%** 提高到 α=−6 的 **78.0%**，形成了幅度更大、方向更明确的有效工作点，并伴随更系统的候选输出顺序变化。CSN 则表现为小剂量下可能改善、较大正负剂量均损害性能的局部响应，其最佳观测点 α=−1 为 **66.67%**，仍明显低于 RSN 的峰值。

因此，当前结果支持两种机制具有部分相关但不完全相同的功能：

> **CSN 可以调节答案提交与候选输出行为，并可能在较小剂量下改善推理表现；但尚未复现 RSN 那种方向明确、幅度较大且统计稳健的 reasoning-calibration workpoint。**

两种 masks 的方向范数不同，且尚未进行 norm matching，因此相同 raw α 不代表相同干预强度。这里能够比较的是行为曲线与功能特征，而不是两种 neurons 的绝对效应强弱。

### 2.6 Role–Confidence Summary

表征证据支持 RSNs 与 CSNs 具有共同的中后层方向、显著的 top-neuron overlap，以及一个高度富集的共享核心。但功能实验显示，两者不能简单等同：

- RSN steering 可以在降低不确定性表达的同时保留 MMLU-E 准确率，并在 GSM8K 形成有效工作点。
- CSN steering 的小剂量区间可以较低代价地减少 E 选择，但较大剂量迅速损害答案质量。
- CSN 在 GSM8K 上改变多种 commitment-related outputs，却没有带来对应的准确率提升。

因此，当前证据支持“**结构相关、功能有别**”。是否存在真正可互换的因果通路，仍需要 norm-matched cross-steering 与相应控制实验验证。

## 3. Role–Chat Relationship

### 3.1 Analysis Scope

本节检验 Role direction 与 Chat–Bare transition 之间的表征关系。分析覆盖 Llama3 与 Qwen2.5，并使用 GSM8K、MATH 与 GSM-Hard 的 paired Chat/Bare hidden states。对应的 Role directions 来自相同 reasoning tasks 的 Expert/Non-expert 对比。

MMLU-E Role direction 单独保留，因为其构建包含基于答案的样本筛选，不能与未经筛选的 reasoning Role directions 直接合并。行为层面的 Chat dose response 已在 `ReasoningChat.md` 中报告，本节不再重复。

### 3.2 Chat–Bare Hidden-State Direction

定义每个任务的平均状态差为：

$$d_{\mathrm{chat}}=\operatorname{mean}(h_{\mathrm{chat}}-h_{\mathrm{bare}}).$$

GSM8K 与 GSM-Hard 的 Chat–Bare 方向高度一致：Llama3 cosine 为 **0.983**，Qwen2.5 为 **0.988**。MATH 与两个 GSM 类任务的相似度较低，但仍显示出结构化的接口相关 hidden-state shift。该结果仅表明 native Chat template 改变了表征状态；平均方向本身不足以说明它与 Role direction 是否相同。

### 3.3 Dense Direction Alignment

所有 cosine 均从 raw、sign-fixed vectors 计算；没有 L2 normalization 或自动翻转符号。Reasoning Role average 与 Reasoning Chat average 在各模型 RSN band 内均接近零：

| Model | Reasoning Role–Chat Mean Cosine | GSM8K Matched | MATH Matched | GSM-Hard Matched | MMLU-E Role–Reasoning Role |
|---|---:|---:|---:|---:|---:|
| Llama3 | **−0.0286** | −0.0923 | +0.0976 | −0.0785 | +0.1777 |
| Qwen2.5 | **−0.1036** | −0.1721 | +0.0236 | −0.1617 | +0.1343 |

Task-matched cosine 的符号并不一致，MATH 与 GSM 类任务的模式不同。因此，Role direction 与 Chat–Bare mean direction 不是同一条 dense axis；MMLU-E 的构建口径也不支持将其与 reasoning Role direction 直接混合。

### 3.4 Sparse Coordinate Overlap

尽管 dense cosine 很低，Role 与 Chat 的 absolute top-0.5% coordinates 存在远高于随机的重合。下表报告 task-matched comparisons；Holm 显著性是 coordinate-overlap random-mask null，而不是样本级总体推断。

| Model | Task | Mean Enrichment over Random | Mean Overlap Coefficient | Holm-Significant Band Layers |
|---|---|---:|---:|---:|
| Llama3 | GSM8K | **34.13×** | 0.167 | 7/9 |
| Llama3 | MATH | **48.92×** | 0.239 | 9/9 |
| Llama3 | GSM-Hard | **37.55×** | 0.183 | 7/9 |
| Qwen2.5 | GSM8K | **57.15×** | 0.287 | 6/6 |
| Qwen2.5 | MATH | **49.78×** | 0.250 | 6/6 |
| Qwen2.5 | GSM-Hard | **57.15×** | 0.287 | 6/6 |

在全层范围，Llama3 有 **19/33** 层、Qwen2.5 有 **13/29** 层达到 Holm-adjusted significance；对应 mean enrichment 分别为 **32.89×** 与 **34.33×**。在 RSN band 内的 enrichment 分别为 **42.10×** 与 **47.93×**，高于 band 外的 **29.44×** 与 **30.78×**。Role energy captured by Chat top-0.5% mask 也高于 random-mask reference（Llama3 0.0434 vs 0.0049；Qwen2.5 0.0684 vs 0.0050），但 containment 本身是非对称描述，不能被读成 axis equivalence 或共享机制。

Role directions 本身具有 task structure：GSM8K–GSM-Hard cosine 为 Llama3 **0.956**、Qwen2.5 **0.976**；MATH 与 GSM8K/GSM-Hard 的 Role cosine 分别为 Llama3 **0.162/0.146**、Qwen2.5 **0.348/0.355**。这与 MATH 的不同 sparse pattern 一致，但不构成因果解释。

### 3.5 Chat–Bare Transition Subspace

将三个任务的 900 组 paired Chat–Bare differences 合并后，在每层拟合 linear PCA transition subspace。主分析采用 within-task centering 与 150/150 split cross-fitting；投影能量为：

$$R^2(k)=\frac{\|V_k^\top r\|^2}{\|r\|^2}.$$

| k | Llama3 Centered R² | Qwen2.5 Centered R² |
|---:|---:|---:|
| 1 | 0.0286 | 0.0278 |
| 2 | 0.0353 | 0.0417 |
| 5 | 0.0631 | 0.1011 |
| 10 | **0.0788** | **0.1261** |
| 20 | 0.1126 | 0.1423 |
| 50 | 0.1901 | 0.2052 |

| Model | k=10 Centered R² | Uncentered R² | Isotropic Null Mean / p97.5 | Permutation Null Mean / p97.5 | Enrichment | Held-Out Reconstruction, k=50 |
|---|---:|---:|---:|---:|---:|---:|
| Llama3 | **0.0788** | 0.0744 | 0.0024 / 0.0030 | 0.0024 / 0.0030 | **32.25× / 32.30×** | **0.4381** |
| Qwen2.5 | **0.1261** | 0.1085 | 0.0028 / 0.0036 | 0.0028 / 0.0035 | **45.29× / 45.61×** | **0.4768** |

Centered projection 在移除任务平均 shift 后仍远高于两个 TRUE band-level null 的 p97.5；不同 `k` 的结果也没有仅随 random-subspace 基线增长。前 10 维捕获 Role 总能量的 **7.9–12.6%**，前 50 维捕获 **19.0–20.5%**。这支持 Role direction 与 Chat–Bare transition subspace 存在非随机几何关系；该 subspace 是线性 PCA 对 900 个样本差分的近似，不是完整 Chat manifold。

### 3.6 Role–Chat Summary

Role direction 与 Chat–Bare mean direction 几乎正交，因此二者不是同一条 dense axis。尽管如此，它们共享显著高于随机水平的 sparse coordinates，而且 Role direction 对 Chat–Bare transition subspace 的投影明显超过随机基线。

因此，当前证据支持一种部分包含关系：Role/RSN 不是完整的 Chat state，但与 Chat 引起的多维状态变化共享非随机的稀疏结构和子空间成分。该结果属于 representational evidence，不能单独证明二者具有相同的因果机制。

## 4. MMLU-E RSN–Reasoning RSN Relationship

### 4.1 Analysis Scope

本节直接比较两类 Role directions：

- **MMLU-E RSN（MRSN）**：MMLU-E 的 Expert−Non-expert direction；
- **Reasoning RSN（RRSN）**：GSM8K、MATH 与 GSM-Hard 三个 Expert−Non-expert directions 的未归一化算术平均。

所有方向均保持原始符号，不做自动翻转；RRSN 在聚合前不进行逐任务 L2 normalization。主比较沿用已冻结的 MRSN band：Llama3 `[11,20)`，Qwen2.5 `[16,22)`。因此，本节回答的是同一层段内两类 Role directions 的表征关系，而不是它们各自最优的 causal steering layer。

### 4.2 Dense Direction Alignment

RRSN 与 MRSN 在两个模型中均呈弱正相关，但不构成同一条 dense axis。

| Model | Band | Mean Layerwise Cosine | Flattened-Band Cosine | Per-Layer Range |
|---|---:|---:|---:|---:|
| Llama3 | `[11,20)` | **0.1777** | 0.1400 | 0.0240–0.2536 |
| Qwen2.5 | `[16,22)` | **0.1343** | 0.1307 | 0.1107–0.1488 |

Llama3 的 alignment 在 band 后段明显下降，而 Qwen2.5 在 band 内较为稳定。两者的 cosine 均为正，但远低于 GSM8K–GSM-Hard Role direction 的跨任务 cosine（Llama3 0.956；Qwen2.5 0.976）。因此，更准确的描述是：MRSN 与 RRSN 存在部分共享结构，但其大部分 dense direction 仍是 task-conditioned 的。

### 4.3 Task-Level Sparse Overlap

现有 sparse analysis 分别比较了三个 reasoning task 的 Role direction 与 MRSN。下表报告同一 MRSN band 内的平均结果；随机富集以相同 hidden size 下两个等规模 top-coordinate sets 的期望重叠为基线。

| Model | Reasoning Role | Mean Dense Cosine | Mean Top-Coordinate Overlap | Mean Enrichment over Random | Shared-Dimension Sign Agreement |
|---|---|---:|---:|---:|---:|
| Llama3 | GSM8K | 0.1848 | 3.67/20 | **37.55×** | 0.870 |
| Llama3 | MATH | 0.0573 | 2.11/20 | **21.62×** | 0.543 |
| Llama3 | GSM-Hard | 0.1628 | 3.44/20 | **35.27×** | 0.889 |
| Qwen2.5 | GSM8K | 0.0984 | 2.50/18 | **27.65×** | 0.833 |
| Qwen2.5 | MATH | 0.1507 | 2.67/18 | **29.50×** | 0.833 |
| Qwen2.5 | GSM-Hard | 0.1058 | 2.67/18 | **29.50×** | 0.847 |

重叠的绝对数量不高，但明显超过随机预期，并且大多数 shared dimensions 具有相同符号。Llama3 上，MATH 与 MRSN 的 dense alignment 和符号一致性均弱于两个 GSM 类任务；Qwen2.5 则没有出现相同排序。这说明共享 sparse coordinates 具有跨任务稳定性，但其强度与任务构成仍具有模型依赖性。

Qwen2.5 的这组既有 sparse statistics 使用旧分析中的 top-18 定义，而当前 canonical NMD 规则为 `int(3584×0.5%)=17`。因此，这些数值只作为已有的 task-level descriptive evidence；最终 RRSN mean mask 与 MRSN mask 的 exact-NMD comparison 仍应使用 top-17 重新计算。

### 4.4 Current Interpretation and Evidence Boundary

现有结果共同支持：

> **MRSN 与 RRSN 不是统一的 dense Role axis，但共享一个显著高于随机的 sparse coordinate core。**

这一结论仍有三个边界。第一，现有 sparse table 比较的是三个 task-specific Role directions 与 MRSN，而不是最终三任务平均 RRSN mask 与 MRSN mask。第二，当前只有 mean matrices，尚无 split-half reliability 或 sample-level bootstrap。第三，静态 cosine、overlap 与 sign agreement 均不能证明两组 neurons 在功能上可互换。

因此尚待补充的直接分析包括：final RRSN–MRSN exact-NMD overlap、Jaccard、weighted cosine、双向 energy containment 与逐层 projection。因果层面则仍需在相同 band、neuron count 与 norm-matched intervention strength 下完成 `RRSN → GSM8K` 自身正对照和 `RRSN → MMLU-E` 反向迁移。

## 5. Cross-Relationship Summary

**Table 5.1. Summary of Representational and Functional Relationships**

| Relationship | Dense Alignment | Sparse Alignment | Subspace Evidence | Functional Evidence | Current Interpretation |
|---|---|---|---|---|---|
| Role–Confidence | Moderate | Enriched shared top neurons | Not tested | Related but distinct dose responses | Shared representation with functional separation |
| Role–Chat | Near-zero mean-direction cosine | Strong top-coordinate enrichment | Non-random projection into Chat transition subspace | Related output-ordering effects reported separately | Partial structural relationship, not axis equivalence |
| MRSN–RRSN | Weak positive alignment | Enriched task-level top-coordinate overlap; final aggregate mask pending | Not tested | MRSN transfers to GSM8K; RRSN causal tests pending | Task-conditioned dense directions with a shared sparse core |

raw α、direction norm 与模型间 activation scale 都不能直接比较；尤其不同模型的 hidden sizes、architecture、mask band 与激活尺度不同，表中的模式不能转换为跨模型的绝对效应排序。

## 6. Conclusions and Evidence Boundaries

Role–Confidence 结果显示中等方向对齐与 enriched shared neurons，但其 functional dose responses 不同。Role–Chat 结果则显示 mean directions 不同，却共享 sparse coordinates，并对 Chat–Bare transition subspace 有非随机投影。MRSN–RRSN 比较进一步显示，来自不同任务族的 Role directions 只有弱 dense alignment，但仍保留显著高于随机的 sparse overlap。三组结果共同说明 RSN 与其他状态方向及不同任务条件下的 Role representations 存在部分表征联系，但不能把 Role、Confidence、Chat 或不同来源的 RSN 视为同一机制。

这些证据不支持人类 dopamine system 的生物学同源性，也不能从结构相似性推出因果通路。需要 norm-matched cross-steering、针对性的 intervention 和适当 controls，才可能检验功能或因果上的重叠。

## Figure Index

| Figure | Purpose | File or Directory |
|---|---|---|
| Main Role Projection | Role projection into the Chat–Bare transition subspace | `5.chat_role_transition_subspace`|
| Layerwise Heatmap | Layerwise Role–Chat subspace relationship | `5.chat_role_transition_subspace` |
| Enrichment Analysis | Projection enrichment against null distributions | `5.chat_role_transition_subspace` |
| Dense and Sparse Analysis | Role–Chat dense cosine and sparse-coordinate figures | `4.full_role_chat_direction` |
| Transition-Subspace Artifacts | Reports, results and plotting sources | `5.chat_role_transition_subspace` |
| MRSN–RRSN Dense Alignment | Direct mean-direction and layerwise comparison | `3.dense_direction_similarity` |
| MRSN–Task RSN Sparse Alignment | Cross-task Role overlap, enrichment and sign agreement | `4.full_role_chat_direction` |
