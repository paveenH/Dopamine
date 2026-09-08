# Role Neurons 与 Confidence Neurons 的表征关系

## 1. 研究问题

我们希望判断：原 RSN 研究中通过 `expert/non-expert` 条件识别的 role neurons，与通过 `confident/unconfident` 条件识别的 confidence neurons，是否对应相同或相近的内部表征机制。

分析集中在 RSN 的干预层 Layer 11–19，并依次比较：

1. 两类条件产生的整体表征方向；
2. confident 与 unconfident 表征的逐层分化；
3. 两组 top neurons 的位置重叠；
4. 不同 neuron 集合对整体方向一致性的贡献。

## 2. 整体表征方向

Role direction 与 Confidence direction 分别定义为：

$$
d_{\mathrm{role}}
=
\mu_{\mathrm{expert}}
-
\mu_{\mathrm{nonexpert}}
$$

$$
d_{\mathrm{confidence}}
=
\mu_{\mathrm{confident}}
-
\mu_{\mathrm{unconfident}}
$$

两个方向在 Layer 11–19 表现出明显的正向对齐。

| 指标 | Layer 11–19 结果 |
|---|---:|
| 拼接 cosine similarity | **0.6063** |
| 拼接 Pearson correlation | **0.6063** |
| 逐层 cosine 均值 | 0.4991 |
| 逐层 cosine 中位数 | 0.4752 |
| 逐层最低值 | 0.1584（Layer 11） |
| 逐层最高值 | **0.7641（Layer 19）** |
| Role direction L2 norm | 4.4395 |
| Confidence direction L2 norm | 12.5108 |
| Confidence/Role norm ratio | 2.82 |

两种方向的相似度从 Layer 11 开始总体增强，并在 Layer 19 达到约 **0.76**。这说明角色身份与显式自信提示虽然不是同一种干预，但它们在中后层引起了方向相近的内部表征变化。

Confidence direction 的整体幅度约为 Role direction 的 **2.82 倍**，说明显式 confident/unconfident 提示产生了更强的表征位移。

## 3. Confident 与 Unconfident 的逐层分化

除了比较 Role 与 Confidence 的差分方向，我们还直接比较了 confident 与 unconfident 条件下的平均 hidden states。

| 指标 | Layer 1–10 | Layer 11–19 | 变化 |
|---|---:|---:|---:|
| Confident–Unconfident correlation | 0.9886 | **0.8783** | 明显下降 |
| Derived divergence \(1-r\) | 0.0114 | **0.1217** | 约增至10倍 |
| Confidence direction norm | 0.4571 | **3.9009** | 约增至8.5倍 |
| 区间极值 | — | \(r=0.8192\) | Layer 19 |
| 区间内最大 direction norm | — | 约6.4 | Layer 19 |

在 Layer 1–10，confident 与 unconfident 的平均表征几乎完全一致，平均相关性为 **0.9886**。进入 Layer 11–19 后，两者的相关性明显下降，平均降至 **0.8783**，并在 Layer 19 达到最低值 **0.8192**。

与此同时，表征差异幅度从 Layer 10–11 附近开始快速增加。Layer 11–19 的平均 divergence 为 **0.1217**，约为前段的10倍；direction norm 也从平均 **0.4571** 上升至 **3.9009**。

因此，显式 confidence 条件的主要分化同样出现在 RSN 所关注的中后层区间，呈现出与原 expert/non-expert 分化相似的层级位置。

## 4. Top-neuron support overlap

在 Layer 11–19，每层分别选取20个 Role top neurons 和20个 Confidence top neurons，共计每组180个 neuron-position。随机选择时，每层的期望重叠数仅为：

\[
\frac{20\times20}{4096}\approx0.098
\]

实际重叠结果如下：

| Layer | 共享数量 | Overlap rate | Jaccard | 符号一致率 | 单尾 \(p\) |
|---:|---:|---:|---:|---:|---:|
| 11 | 2/20 | 10% | 0.053 | 50.0% | \(4.08\times10^{-3}\) |
| 12 | 0/20 | 0% | 0.000 | — | 1.00 |
| 13 | 3/20 | 15% | 0.081 | 66.7% | \(1.08\times10^{-4}\) |
| 14 | 5/20 | 25% | 0.143 | 100% | \(2.40\times10^{-8}\) |
| 15 | 5/20 | 25% | 0.143 | 100% | \(2.40\times10^{-8}\) |
| 16 | 8/20 | 40% | 0.250 | 100% | \(7.88\times10^{-15}\) |
| 17 | 6/20 | 30% | 0.176 | 83.3% | \(2.21\times10^{-10}\) |
| **18** | **10/20** | **50%** | **0.333** | **100%** | \(9.21\times10^{-20}\) |
| 19 | 7/20 | 35% | 0.212 | 100% | \(1.52\times10^{-12}\) |
| **Layer 11–19** | **46/180** | **25.6%** | **0.147** | — | — |

两组 top neurons 在 Layer 11–19 共共享 **46/180** 个位置，overlap rate 为 **25.6%**，Jaccard similarity 为 **0.147**。

除 Layer 12 外，其余各层的重叠均显著高于随机预期。重叠程度总体随网络深度增强，并在 Layer 18 达到最高：20个 top neurons 中有 **10个共享**，即一半的神经元位置相同。

Layer 14–19 的共享 neurons 还表现出很高的方向一致性，绝大多数层的符号一致率为 **100%**。这表明共享 neurons 不仅位置相同，其在 Role 和 Confidence 两个方向中的变化符号也基本一致。

不过，46/180 的重叠也说明两组 neurons 并不完全相同。更准确地说，Role 与 Confidence 共享一个显著高于随机的核心子集，同时各自保留了大量特异 neurons。

## 5. Shared neurons 对整体方向一致性的贡献

为了判断整体 cosine similarity 是否主要由共享 top neurons 产生，我们将 Layer 11–19 的所有36,864个 neuron-position 分为四组：

| Neuron 组 | 数量 | Signed dot 占比 | Absolute alignment 占比 | Role energy | Confidence energy |
|---|---:|---:|---:|---:|---:|
| Shared-top | 46 | **8.0%** | **6.5%** | 5.63% | 5.25% |
| Role-only | 134 | 2.9% | 2.5% | 3.75% | 1.13% |
| Confidence-only | 134 | 2.8% | 2.5% | 0.96% | 4.61% |
| Neither-top | 36,550 | **86.3%** | **88.6%** | **89.65%** | **89.01%** |

从总量看，Role–Confidence alignment 主要位于 top-20 support 之外：neither-top 集合贡献了 **86.3%** 的 signed dot product 和 **88.6%** 的 absolute alignment。

但是，shared-top 只有46个 neuron-position，却贡献了 **8.0%** 的 signed dot product，说明其贡献密度非常高。

| 单位贡献比较 | 约数 |
|---|---:|
| Shared-top / Role-only | 8倍 |
| Shared-top / Confidence-only | 8倍 |
| Shared-top / Neither-top | **74倍** |

因此，这些共享 top neurons 虽然不能单独解释整体 cosine similarity，但构成了一个高度富集的局部核心。整体结果更符合以下结构：

> **广泛分布的表征对齐背景，加上少量单位贡献显著更高的共享核心 neurons。**

需要注意的是，neither-top 集合包含绝大多数 neuron-position。当前结果只能说明大部分 alignment 总量位于 top-20 之外，尚不能证明所有非 top neurons 都在均匀贡献；这部分信号也可能集中在刚好未进入 top-20 的次高排名 neurons 中。

## 6. 总结

MMLU-E 上的结果表明：

1. **Role 与 Confidence 的整体表征方向明显相关。**  
   Layer 11–19 的拼接 cosine 为 **0.6063**，逐层最高达到 **0.7641**。

2. **Confidence 的主要表征分化发生在与 RSN 相近的中后层。**  
   Confident–Unconfident correlation 从 Layer 1–10 的 **0.9886** 降至 Layer 11–19 的 **0.8783**，最低在 Layer 19 达到 **0.8192**。

3. **Role neurons 与 Confidence neurons 部分重叠，但并非同一组。**  
   两组 top neurons 共享 **46/180（25.6%）**，Jaccard 为 **0.147**，显著高于随机预期。

4. **共享 neurons 构成一个稀疏但高度富集的核心。**  
   Shared-top neurons 的单位贡献约为 role-only/confidence-only neurons 的 **8倍**，约为 neither-top neurons 的 **74倍**。

总体而言，Role 与显式 Confidence 可能共享一部分中后层表征基础，但两者并不是完全相同的神经元机制。当前结果支持一种简洁的解释：

> **Role 与 Confidence 共享一个方向一致、贡献高度集中的稀疏核心，同时伴随更广泛的分布式表征对齐。**

这些结果目前证明的是结构上的关联。两组 neurons 是否具有可互换的功能，仍需通过后续 cross-steering 实验验证。