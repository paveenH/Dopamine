# Role and Confidence Neurons: Representation and Function

## 1. Main Findings

Role neurons（RSNs）与 Confidence neurons（CSNs）在表征上明显相关，但目前没有证据表明两者具有相同功能。

- **Representation:** 两个方向在 Layer 11–19 的 cosine similarity 为 **0.6063**，并共享 **46/180（25.6%）** 个 top-neuron positions。
- **Organization:** 整体 alignment 主要来自广泛分布的非 top positions；共享 top neurons 数量少，但单位贡献高度富集。
- **MMLU-E:** 小剂量 CSN steering 能减少 E（“I am not sure”）选择，并在 confident prompt 下保持或小幅提高准确率；较大剂量则明显损害答案选择。
- **GSM8K:** CSN steering 会改变提前输出、规范提交和重复行为，但没有复现 RSN 的有效推理工作区间。

最简洁的结论是：

> **RSNs 与 CSNs 共享一部分表征基础，但不是可直接互换的功能机制。**

## 2. Representational Alignment

### 2.1 Direction and Layerwise Separation

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

## 3. Functional Evidence on MMLU-E

### 3.1 RSN Steering under Confidence Prompts

在 confident prompt 下，RSN steering 能明显减少 E 选择，同时基本保持准确率。

| Condition | STEM Acc. | STEM E-rate | Humanities Acc. | Humanities E-rate | Social Sciences Acc. | Social Sciences E-rate | Other Acc. | Other E-rate | Task-Macro Acc. | Task-Macro E-rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Confident baseline | 55.03% | 2.72% | 68.16% | 3.34% | 73.65% | 1.60% | 69.06% | 1.54% | 65.14% | 2.36% |
| RSN +3, Layer 11–19 | 56.17% | 0.40% | 69.39% | 0.93% | 74.10% | 0.85% | 68.93% | 0.59% | **65.87%** | 0.66% |
| RSN +3, Layer 11–19, t4 | 55.70% | 0.23% | **69.81%** | 0.39% | **74.01%** | 0.61% | 68.69% | 0.34% | 65.74% | 0.37% |
| RSN +4, Layer 11–19 | 55.14% | **0.09%** | 69.38% | **0.34%** | 73.68% | **0.54%** | 68.70% | **0.27%** | 65.38% | **0.28%** |

Task-macro E-rate 从 **2.36%** 降至 **0.66%、0.37% 和 0.28%**，准确率维持在 **65.38%–65.87%**。这说明 RSNs 能调节不确定性表达，而不必同步损害知识选择。

### 3.2 CSN Dose Response

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

### 3.3 RSN–CSN Comparison under the Confident Prompt

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

## 4. Functional Evidence on GSM8K

### 4.1 CSN Output Behavior

在 GSM8K 的 300 道题上，CSN steering 改变了多种输出行为，但未产生高于 baseline 的准确率点。

| Metric | −4 | −2 | 0 | +2 | +4 |
|---|---:|---:|---:|---:|---:|
| **First accuracy** | 51.0% | **60.3%** | **60.3%** | 55.7% | 46.3% |
| Last accuracy | 49.7% | 56.3% | 55.7% | 53.3% | 46.0% |
| **Committed accuracy** | 63.6% | 61.2% | **66.7%** | 63.8% | 48.3% |
| Commit rate | 36.7% | 55.0% | **62.0%** | 53.3% | 57.3% |
| Median `####` position | 20% | 20% | 18% | 17% | 18% |
| Mean `####` position | 25% | 24% | 24% | 25% | 21% |
| Premature, leading digit | 266 | 167 | 199 | 133 | 188 |
| **Premature, either rule** | **266 (88.7%)** | 196 (65.3%) | 206 (68.7%) | **139 (46.3%)** | 220 (73.3%) |
| `####` at text start | 8 | 31 | 23 | 9 | 58 |
| At least two answer switches | 14 | 21 | 10 | 10 | 12 |
| At least three candidate values | 3 | 7 | 5 | 7 | 3 |
| Median generation length | 2,289 | 2,118 | 2,163 | 2,229 | 2,273 |
| Loop samples | 212 | 231 | 227 | 222 | 226 |
| At least two `Step` markers | 26 | 55 | 23 | 36 | 24 |
| Stuck loops | 15 | 16 | 27 | 27 | 36 |
| Median equation count | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 |
| **Full-text compulsive repetition** | **104** | 59 | 73 | 75 | **102** |

`Committed accuracy` 和 `####` position 只在产生可解析正式提交的样本中定义，不能替代总体准确率。其余计数的分母均为 300。

主要现象如下：

- First accuracy 为 **51.0% → 60.3% → 60.3% → 55.7% → 46.3%**。强正、负干预都降低准确率，没有出现高于 baseline 的工作点。
- First 与 last accuracy 的差异均不超过 **4.7 pp**，且多数剂量下 first accuracy 更高；后续答案修改通常没有改善总体表现。
- Median `####` position 始终为全文的 **17%–20%**，commit rate 也没有一致的剂量方向。
- `Premature, either rule` 在 +2 降至 139，但准确率同时下降到 55.7%；减少 premature output 本身不足以提高准确率。
- Full-text compulsive repetition 在 −4/+4 增至 104/102，与两端性能下降相伴，但普通 loop rate 没有清晰趋势。
- +4 的 committed accuracy 降至 **48.3%**，stuck loops 增至 **36**，说明强干预也降低正式提交质量。

这些结果是描述性点估计；该 pilot 没有对各剂量与 baseline 做正式显著性检验。

### 4.2 RSN–CSN Comparison

| α | RSN First Acc. | CSN First Acc. | RSN Commit Rate | CSN Commit Rate | RSN Premature | CSN Premature |
|---:|---:|---:|---:|---:|---:|---:|
| −4 | **73.0%** | 51.0% | 58.3% | 36.7% | 195 | 266 |
| −2 | **69.0%** | 60.3% | 63.0% | 55.0% | 223 | 196 |
| 0 | 60.0% | 60.3% | 62.7% | 62.0% | 206 | 206 |
| +2 | 57.0% | 55.7% | 53.0% | 53.3% | 215 | 139 |
| +4 | **55.3%** | 46.3% | 49.0% | 57.3% | 232 | 220 |

RSN 在完整曲线的 **α=−6** 达到 **78.0%** accuracy，同时 premature output 降至 **94**，committed accuracy 达到 **79.7%**。CSN 没有复现这一“较低 premature、较高提交质量、准确率同步提升”的状态。

同样需要注意：两种 masks 未做 norm matching，相同 raw α 不是相同干预剂量。这里比较的是曲线与行为签名，而不是绝对效应强弱。

## 5. Conclusion and Evidence Boundary

表征证据支持 RSNs 与 CSNs 具有共同的中后层方向、显著的 top-neuron overlap，以及一个高度富集的共享核心。但功能实验显示，两者不能简单等同：

- RSN steering 可以在降低不确定性表达的同时保留 MMLU-E 准确率，并在 GSM8K 形成有效工作点。
- CSN steering 的小剂量区间可以较低代价地减少 E 选择，但较大剂量迅速损害答案质量。
- CSN 在 GSM8K 上改变多种 commitment-related outputs，却没有带来对应的准确率提升。

因此，当前证据支持“**结构相关、功能有别**”。是否存在真正可互换的因果通路，仍需要 norm-matched cross-steering 与相应控制实验验证。
