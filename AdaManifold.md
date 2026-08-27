# Manifold Geometry Analysis
## 0. Executive Summary

**核心问题。** RSN steering 究竟只是沿固定方向改变增益，还是会重组 hidden-state 的移动方向？这种几何差异能否进一步解释 Llama 的 `α=−6` 峰值，以及 Qwen 正端的高剂量平台？

**Llama 核心发现（last-prefill，decoder 18）。** RSN steering 呈现出一种 **piecewise geometry**，不能由单一的全局 scalar-gain 模型完整解释：

- **负剂量近似构成一维标量族。** `α=−8` 与 `α=−6` 的位移高度共线：`cos=0.989`，scalar-fit residual 仅为 `2.1%`，且 300/300 道题的方向符号一致。拟合得到的缩放系数 `k=1.379`，也与独立计算的位移幅度比 `24.63/17.67=1.394` 高度吻合。换言之，`−8` 基本上是沿着 `−6` 的方向继续放大，而不是进入完全不同的状态。

- **正剂量并非负剂量的简单镜像。** `α=+6` 与 `α=−6` 部分反向对齐（`cos=−0.662`），但二者并不共线：共享轴只能解释约 `cos²≈44%` 的位移能量，其余约 `56%` 位于正交方向。因此，`+6` 既不是较弱的 `−6`，也不是其完全反向版本，而包含明显的方向重组。

这一结果为 Llama 的非对称行为曲线提供了一个简洁的几何解释：**`α=−6` 到达较优的 working region，而 `α=−8` 沿同一方向继续推进并发生 overshoot。** 因此，性能崩溃不一定意味着模型进入了完全不同的状态，也可能是沿有效方向移动过度。

**证据边界。** 上述结论仅成立于 **last-prefill、decoder 18**——即 steering 的注入位置，也是不同 α 之间唯一能够严格按题、按 token 配对的位置。目前尚未检测到稳定的增量行为预测价值，且该 last-prefill 几何模式没有稳定延伸到 commit-aligned decode。因此，这些结果不能被表述为完整的推理轨迹机制，也不能证明状态已经“离开 natural manifold”。

**当前定位。** Llama 的主要分析已经完成，manifold 结果应定位为 **last-prefill explanatory geometry**：它为 `−6` 最优、`−8` overshoot 和正负剂量不对称提供了机制性线索，但不是独立的因果机制或行为预测主线。Llama–Qwen 是否共享同一套解释，仍需由预先登记的 Qwen last-prefill analysis 检验。

---

## 1. Research Questions

关于 α 对 hidden state 做了什么，有三种互相竞争的解释：

1. **Scalar gain** — steering 只改变沿固定方向的幅度。
2. **On-manifold retiming** — 轨迹仍留在 natural manifold 内，但速度、phase occupancy 或 commitment timing 改变。
3. **Directional reorganisation / off-subspace deviation** — 极端剂量让轨迹转向，同时伴随 accuracy 下降、loop 或 format failure。

**Cross-model question。** Llama 呈 asymmetric peaked response（最优 α=−6，−8 崩溃），Qwen 呈 high-dose plateau。若两者接受的都是 gain-like steering，候选解释是 **Llama 沿自己的轴 overshoot，而 Qwen 的位移 saturate**——这样就能从几何而非仅从行为解释 peak-vs-plateau。

**关于 prediction。** 我们进一步探索了 manifold features 能否在 `s_t`/`Z_t` 与 commitment-related variables 之外提供增量预测信息（§3.5、§3.6）。这项分析是一项辅助性的 *value check*，而不是判断几何解释是否成立的必要条件。个体题目的 correctness 主要受到题目难度影响，commit position 也具有较高噪声；此外，一维信号与高维几何描述回答的并不是同一个问题。因此，目前未检测到稳定的增量预测价值，并不否定下文的描述性几何结果。

---

## 2. Methods

### 2.1 Data and conditions

Llama-3.1-8B，GSM8K，300 题，除特别说明外为 No-CoT。只使用已存储的 hidden states——本分析没有重新生成任何模型输出。

| condition | role |
|---|---|
| α=0 | natural baseline; the basis is fit here |
| α=−6 | best working point |
| α=−8 | negative-arm collapse |
| α=+6 | positive-arm damage representative |
| CoT α=0 / −4 | conditional validation set (§3.6) |

Band `[11,20)` 对应 decoder layers 10–18（L=9）。**Primary layer 为 decoder 18**（export slot `8`），decoder 10（slot `0`）用于 sensitivity analysis，其余层作为补充。跨条件样本依据 `question_idx` 配对，以确保同一道题在不同 α 下保持对应关系。

### 2.2 Frozen question split

60/20/20 **按题**划分，实际落为 **185/55/60**，版本 `manifold-split-v1`。

- **Train** — 拟合 α=0 PCA basis。
- **Validation** — 评估 `k = 5/10/20` 下结论的稳健性。
- **Test** — 报告最终剂量比较与 exploratory prediction results。

所有条件共用同一个 split，以避免剂量差异与题目难度混淆。划分单位为 question，而非 token，因为同一道题内的 hidden states 具有相关性。基于 hash threshold 得到的实际样本数为 185/55/60；保留这一结果可使既有题目的归属在数据集扩展后保持稳定。

**PCA dimension。** 主分析使用 `k = 20`，并报告 `k = 5/10/20` 的 sensitivity；若三个维度下的结果方向一致，则视为对 k 的选择稳健。Top-30 spectrum 用于观察谱尾，其中前 20 个分量构成分析 basis。由于 validation NRE 会随 k 单调下降，k 并不依据剂量效应进行优化，而是预先固定并通过 sensitivity analysis 检查结论是否依赖单一维度。

### 2.3 Alpha=0 PCA reference geometry

PCA basis 仅由 **α=0 的 train questions** 拟合。Validation 用于维度敏感性分析，test 用于最终比较；steered conditions 不参与 reference geometry 的学习。四个 phase 分别拟合独立 basis，共得到 9 layers × 4 phases = 36 个 PCA bases。

两个关键设定如下：

- **Per-question equal weighting**（行按 `1/√n_i` 缩放），避免长轨迹因 token 数较多而对 PCA basis 产生更大权重。该处理尤其重要，因为生成长度本身与 α 相关。
- **统一中心化。** 所有条件都使用 α=0 training set 的均值 `mu`，从而保留 steering 引起的条件间位移。

### 2.4 Token phases and pairing rules

| phase | span | rows/question | role |
|---|---|---|---|
| `prefill` | last prefill token only | 1 | 不同 α 下 prompt 与 token 完全一致，可进行严格配对的 displacement analysis |
| `pre_commit` | `[c−20, c)` | ≤20 | event-aligned distribution comparison |
| `post_commit` | `[c, c+20)` | ≤20 | 同上；commit 本身是第一行 |
| `decode_all` | whole decode, per-question row mean | 1 | level sensitivity；同时包含 no-commit 样本 |

- 对于在 token 20 之前 commit 的样本，使用其实际可用的短窗口。这样可以保留 fast-commitment trajectories；Llama α=0 中约 23% 的样本在 token 20 前 commit，且该比例随 α 改变。
- No-commit 样本不进入 commit-aligned phases，但保留在 `decode_all` 中；各条件的 coverage 在结果中单独报告。
- Decode 阶段采用 event-aligned distribution comparison。由于 α 会改变生成文本，不同条件下不存在逐 token 的严格对应关系，因此不在这些阶段估计 paired state displacement。

### 2.5 Geometry metrics and matched nulls

**Normalized reconstruction error (NRE)。**
`NRE(α) = mean(RE_α) / mean(RE_{α=0, held-out})`，按 layer × phase 计算。主指标采用 **ratio of cohort means**；相较于逐题计算 ratio 后取平均，这一口径对 α=0 residual 接近零的样本更稳定。按 hidden-state norm 归一化仅作为 sensitivity，因为该归一化可能同时移除待检验的 scalar-gain 效应。

**PCA-subspace alignment。** 本分析使用按 phase 拟合的 global PCA basis，因此衡量的是 PCA-subspace alignment，而非 local tangent alignment。对 prefill，位移 `d = h(α) − h(0)` 被分解为 α=0 top-k 子空间内及其正交补空间中的能量。主指标为 **energy-pooled ratio** `Σ‖W_k d‖² / Σ‖d‖²`，使不同题目的贡献随其位移能量加权。

**Cross-dose scalar fit。** 使用最小二乘模型 `d_a ≈ k·d_b`，报告 cosine、缩放系数 `k` 与 residual。在最小二乘解处，`residual ≡ 1 − cos²`；因此 cosine 与 residual 描述相同的方向关系，而 `k` 额外刻画沿共享方向的幅度变化。Per-question 同号比例用于检验 pooled cosine 是否由方向不同的样本亚群混合而成。

**Commitment-centroid distance。** 到 α=0 TRAIN `post_commit` centroid 的距离，在 train 上定义、在 test 上评估。

**Matched isotropic null。** Null distribution 与对应 phase 匹配 `m`、`nq`、`dim` 和 per-question weighting，并采用相同的 Gram decomposition。该参照用于区分真实的 spectral concentration 与 `m ≪ dim` 时由有限采样产生的低秩结构。对于位移方向，各向同性参照为 `k/dim`；在 4096 维空间中，随机位移在任意 20-D 子空间中的期望能量占比为 `20/4096 = 0.488%`。

Shuffled-question 不改变 pooled PCA 的总体分布，因此不作为有效 null。采用的几何参照包括 matched isotropic spectrum、相同 k 的 random orthonormal subspace，以及用于 speed/curvature 的 trajectory-order shuffle。

### 2.6 Statistical analysis and scope of inference

- Bootstrap 与 cluster inference 均以 **question** 为统计单位。
- 预先设定的剂量对比为 `−8 vs 0`、`−6 vs 0` 和 `+6 vs 0`。
- Holm correction 在各 metric family 内实施，不跨 family 合并。
- 对 exploratory prediction analysis，只有当同一 metric pair 的两个指标均改善时，才将该剂量记为 improved；stable improvement 进一步要求三个剂量方向一致。其余结果报告为 mixed 或 not detected。

解释范围限定如下：

- PCA 证明的是 **linear low-rank**，不是 nonlinear manifold。
- `k = 20` 是预先设定的 analysis cap，不代表 intrinsic dimension。
- Top-k 子空间外的能量不能直接解释为 "off-manifold"；k=20 只覆盖 α=0 约 50% 的方差，因此正交补空间仍包含大量自然变异。
- Null ratio 仅在相同 phase 内具有可比性。
- 结果描述的是 **computational geometry of RSN steering**，不构成生物多巴胺证据或因果证据。

---

## 3. Results

### 3.1 Data integrity, accuracy and coverage

四个 primary conditions 均完成全量数据验收：每个条件 n=300，`stored_layer_indices = [10..18, 31]`，band `[11,20)`。

- 由原始 hidden states 复算的 projection 与已存结果完全一致，四个条件的误差均为 **0.00e+00**。
- 与 lightweight batch 比较时，四个条件在三个字段上的逐题 agreement 均为 **1.000**。这一数值描述本批次数据的一致性。
- Accuracy 为 **79.67 / 60.00 / 51.67 / 40.67**（α = −6 / 0 / +6 / −8），在同一批次中复现了 α=−6 的性能峰值。该批次在后续分析中保持独立，不与另一批次的 dose table 进行逐题合并。

**Commit coverage** 作为描述性行为指标报告：

| cell | α | coverage |
|---|---|---|
| `nocot` | 0 | 297/300 = .990 |
| `nocot_aneg6` | −6 | 298/300 = .993 |
| `nocot_aneg8` | −8 | 294/300 = .980 |
| `nocot_a6` | +6 | **281/300 = .937** |
| `cot` | 0 | 297/300 = .990 |
| `cot_aneg4` | −4 | 296/300 = .987 |

+6 的 coverage 相比 α=0 下降约 5 个百分点，与此前观察到的正 α format degradation 方向一致。因此，+6 的 commit-aligned analysis 少包含 16 道题，并可能受到 intervention-dependent selection 的影响。

Fit-phase 行数：prefill / `decode_all` m=185；`pre_commit` m=2940（nq=**150**）；`post_commit` m=3576（nq=183）。`pre_commit` 的 nq 更低，是因为在 decode step 0 就 commit 的样本有 post 窗口而没有 pre 窗口。

### 3.2 Alpha=0 spectral concentration

累计解释方差，实测 vs matched isotropic null（20 draws，median [2.5, 97.5]）：

| phase | m | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| prefill | 185 | .257 vs .039 | .365 vs .076 | **.499 vs .148** [.1477, .1489] |
| pre_commit | 2940 | .243 vs .006 | .344 vs .011 | **.446 vs .022** [.0222, .0223] |
| post_commit | 3576 | .296 vs .005 | .397 vs .010 | **.484 vs .020** [.0200, .0201] |
| decode_all | 185 | .641 vs .039 | .730 vs .076 | .812 vs .148 |

Null 区间的相对宽度小于 1%，20 draws 已收敛。三个 k 值方向一致。

结果支持 **low-rank spectral concentration relative to a matched isotropic null**。不同 phase 的 null level 受到样本结构影响，因此倍数仅在各 phase 内解释：prefill 的 `m = nq = 185`，其 null 在 k=20 时已达到 14.8%，所以 prefill 的 3.4× 与 commit windows 的约 20× 不宜直接比较。谱呈缓慢衰减且没有明确 elbow，因此主分析固定 k=20，并通过 k=5/10/20 报告 sensitivity。

`decode_all` 被归约为每题一个均值向量，并在约 185 个点上拟合。因此，其谱形同时反映这种聚合方式，不应单独解释为 manifold stability 的证据。

### 3.3 Exact last-prefill PCA-subspace analysis

在 decoder 18 的 ambient space 中计算位移分解。300 道题依据 `question_idx` 跨条件配对，并统一投影到 α=0 training basis。定义 `f_k = ‖W_k d‖²/‖d‖²`，主结果采用 energy-pooled ratio。

**Primary (TEST split, k=20)：**

| α | mean‖d‖ | inside [95% CI] | outside |
|---|---|---|---|
| −8 | 24.63 | **21.4%** [20.9, 21.8] | 78.6% |
| −6 | 17.67 | **21.2%** [20.8, 21.7] | 78.8% |
| +6 | 13.23 | **9.8%** [9.2, 10.5] | 90.2% |

**k sensitivity (TEST)：** 当 k 从 5 增至 10 和 20 时，−8 的 inside ratio 为 9.6 → 17.0 → 21.4，−6 为 9.5 → 16.8 → 21.2；两者均增加 **11.8pp**，且 per-dimension profile 高度相似。+6 则为 5.8 → 8.1 → 9.8，仅增加 **4.0pp**，表明其位移能量较少分布在 α=0 的前 20 个主成分上。

**Split agreement at k=20**（train / val / test）：−8 为 21.8 / 21.1 / 21.4，−6 为 21.8 / 21.0 / 21.2，+6 为 10.7 / 9.5 / 9.8。三个 split 的结果接近，说明观察到的差异并非仅由用于拟合 basis 的 training questions 驱动。Pooled ratio 与 per-question mean 相差 <0.1pp，也未显示少数大位移样本主导总体结果。

**随机参照：** 各向同性位移在任意 20-D 子空间中的期望能量占比为 `20/4096 = 0.488%`。因此，+6 的 9.8% 约为随机参照的 **20×**，−6/−8 的约 21.2% 为 **43×**。三个剂量的位移能量均明显高于各向同性参照，但正剂量在 α=0 主方向上的集中程度约为负剂量的一半。

位移幅度与 inside ratio 呈现不同的剂量关系：三个条件的 mean‖d‖ 为 24.63 / 17.67 / 13.23，而 inside ratio 并不按相同比例变化。由于两个位移即使具有相近的 inside ratio，也可能在同一子空间中指向不同方向，具体的方向关系由 §3.4 的 cross-dose scalar fit 评估。

### 3.4 Cross-dose direction and scalar fit

最小二乘 `d_a ≈ k·d_b`，TEST split：

| pair | cos [95% CI] | k | residual | same-signed |
|---|---|---|---|---|
| −8 \| −6 | **0.989** [0.989, 0.990] | 1.379 | 0.021 | 100.0% |
| −8 \| +6 | −0.657 [−0.667, −0.647] | −1.222 | 0.569 | 0.0% |
| −6 \| +6 | −0.662 [−0.674, −0.650] | −0.884 | 0.562 | 0.0% |

四个 split 的结果小数点后三位一致。

- **The negative arm is approximately a one-dimensional scalar family**：cos 0.989，residual 2.1%，300/300 同号，且 `k = 1.379` 与独立算出的幅度比 24.63/17.67 = 1.394 吻合。
- **The positive arm is partially anti-aligned with a substantial orthogonal residual**：cos ≈ −0.66，共同轴 `cos² ≈ 44%`，正交约 56%。100% 同号意味着这是 *一致的部分*反向，而不是若干亚群的混合。
- 因此 steering **并不**在所有剂量上共享一条直线。

Inside ratio（§3.3）与 cosine 提供相互一致但概念不同的证据：前者衡量位移能量在 α=0 top-k PCA subspace 中的比例，后者直接比较两个剂量的位移方向。现有分析尚未确定 +6 相对于负端轴的正交分量是否主要位于 top-k 子空间之外。

### 3.5 Incremental prediction

**Result: no stable incremental behavioural predictive value was detected。**

Round 1 使用 TEST split 评估 correctness（Z-only baseline）。Round 2 的设计发生在观察 round 1 结果之后，因此属于 **post-hoc analysis**，不作为 confirmatory evidence。后续结果均在这一 provenance 下解释。

**Commit position** — baseline 只含生成前特征 `[Z_prefill, prefill confidence]`；commitment behaviour 不可入选，因为它 *就是* outcome：

| α | baseline R² / MAE / ρ | +geometry R² / MAE / ρ |
|---|---|---|
| −8 | .003 / 73.1 / .129 | .019 / 72.3 / .192 |
| −6 | .074 / 74.1 / .276 | .104 / 72.5 / .359 |
| +6 | .004 / 67.8 / .033 | −.004 / 68.3 / −.010 |

两个负剂量在 R²、MAE 与 ρ 上均改善，而 +6 在三项指标上均下降。根据预先设定的跨剂量一致性标准，该结果为 **mixed**。

**Correctness** — 在该分析中，commitment-related variables 可作为 predictors。AUC 为 .700 / .386 / .547 → .688 / .497 / .443，log-loss 为 .6294 / .4990 / .6917 → .6429 / .4882 / .7247。结果在指标和剂量之间并不一致：−6 的 AUC 从 .386 提升至 .497，而 −8 与 +6 的 AUC 下降。

Round 1 存档（correctness，Z-only baseline）：AUC .485 / .479 / .570 → .503 / .617 / .458，一升两降。

这些结果有三项限制：TEST split 已用于多轮探索；每个条件的 test sample size 为 n=60，估计不确定性较大；此外，所检验的 outcome 未必是评估几何解释价值的最佳目标。因此，结论限定为 **未检测到稳定增量预测价值**，而不是几何信息已被证伪。

### 3.6 CoT negative-arm conditional confirmation

H1 在 CoT projection 之前预先设定：*加入 prefill geometry 会改善负 α 的 commit-position prediction，而不会改善正 α。* CoT conditions 使用与原分析相同的模型设定，并投影到 **既有的 No-CoT α=0 basis**，以评估该几何表示在新条件下的迁移性。

**CoT α=−4, commit position：**

| | R² | MAE | ρ |
|---|---|---|---|
| baseline (Z, conf) | −0.101 | 56.5 | −0.091 |
| +geometry | **−0.056** | **53.9** | **0.121** |

R²、MAE 与 ρ 均朝预期方向变化，因此 **H1 的负端满足预设的方向性判据**。

**但绝对预测力仍然较弱**：加入几何后 R² 仍为负，表示模型的预测性能仍低于以 training mean 作为预测值；ρ 也接近零。因此，该结果应解释为 **a reproducible weak directional signal, not strong predictive evidence**。

同一 cell 上的 correctness 明显变差：AUC .531 → .462，log-loss .3298 → .4242。

该分析仅覆盖 H1 的 **负端**，因为 CoT 数据不包含正剂量。α=−4 也不属于最初形成 H1 的 −8/−6 剂量，因此这一结果同时检验了向未测剂量的外推。由于 CoT states 投影在 No-CoT α=0 basis 上，数值描述的是 CoT 状态相对于 **No-CoT reference geometry** 的位置。

### 3.7 Minimal pre/post-commit decode analysis

Decoder 18，k=20，TEST split，仅 event-aligned distributions。

| phase | α | n | NRE | speed | centroid dist. |
|---|---|---|---|---|---|
| pre_commit | −8 | 28 | 1.087 | 6.791 | 2.708 |
| | −6 | 55 | 0.988 | 6.846 | 2.464 |
| | 0 | 47 | 1.000 | 7.096 | 2.617 |
| | +6 | 45 | 1.044 | 6.801 | 2.801 |
| post_commit | −8 | 60 | 1.409 | 5.319 | 4.883 |
| | −6 | 59 | 0.836 | 6.903 | 4.392 |
| | 0 | 60 | 1.000 | 6.112 | 4.252 |
| | +6 | 57 | 1.050 | 6.323 | 4.770 |

预设的判断标准包括：(a) −6 与 −8 呈现一致的几何 profile，(b) +6 与负剂量稳定分离，以及 (c) 该模式同时出现在 pre-commit 与 post-commit。结果未满足上述三项标准：

- 在 `post_commit` 中，−8 明显高于 α=0 参照（NRE 1.409）而 −6 明显低于（0.836）；两个负剂量并未归为一组。
- +6 的 NRE 为 1.044 / 1.050，接近 α=0；在 decode analysis 中偏离更明显的是 −8，而 last-prefill analysis 中差异最明显的是 +6。
- `pre_commit` 的 NRE 在四个剂量上只跨 0.988–1.087，所以结构只出现在一个 phase 中。

此外，`pre_commit` 的样本数不平衡（−8: 28；−6: 55）。由于 α 会改变 commit timing，进入 pre-commit analysis 的样本集合本身也随剂量变化，因此这些比较可能受到 intervention-dependent selection 的影响。

**Conclusion: last-prefill geometry did not stably extend to commit-aligned decode。** 这并不证伪 manifold；它界定了干净结构存在的范围。

这一结果与 last-prefill finding 的差异并不构成直接矛盾。Pre-commit coverage 随 α 改变，post-commit 位于 `####` 之后并受到答案格式影响，而 steering 在 last-prefill 注入后，各条件的生成轨迹已经开始分叉。此外，NRE 衡量的是 reconstruction-error magnitude；高于或低于 α=0 并不等价于位移方向相反。

---

## 4. Interpretation

### 4.1 Supported findings

1. **α=0 states carry low-rank spectral concentration relative to a matched isotropic null** —— commit 窗口中 k=20 处为 null 的 20–24×。
2. **The negative arm is approximately a one-dimensional scalar family**，幅度沿共享轴由 −6 增长到 −8。
3. **The positive arm is partially anti-aligned with a substantial orthogonal residual** —— 既非镜像，也不只是更小的位移。
4. **Steering is piecewise, not one global scalar gain。**
5. 因此，**α=−6 抵达 working region，而 α=−8 沿同一条轴 overshoot** —— 崩溃不必意味着到达一个完全不同的状态。这是本条线最有价值的单项解释性结果。
6. **正负行为不对称有几何对应物**：这种不对称存在于 hidden-state geometry 本身，而不仅存在于 accuracy 数字里。

以上六条全部是 **last-prefill** 的陈述。

### 4.2 Relation to Dopamine and the Thinking Curve

这些结果只能支持 **computational-level** 的多巴胺类比：

- 负端的共享方向表现得像一条稳定的 **gain-control axis**。
- −6 → −8 是沿该轴从有效剂量进入过量区——结构上就是 Yerkes–Dodson optimum → overdose 的形状。
- +6 的方向性重组提示：过度或反向调节可能进入一个 **不同的计算 regime**，而不只是一个更小或反向的 regime。
- Cross-model 的候选解释是：**Llama 的有效位移持续增长并最终 overshoot，而 Qwen 的位移可能受到压缩或逐渐饱和**。预先登记的 Qwen analysis 将直接检验这一假设。

**α 的正负不等于生物多巴胺的增加或减少。** Manifold 描述的是 RSN steering 的计算几何，不是一种神经递质。

### 4.3 Unsupported claims and limitations

当前证据不能支持以下主张：

- 任何 **nonlinear manifold** 主张 —— PCA 只能确立 linear low-rank。
- 把 `k = 20` 当作 **intrinsic dimension** —— 它是 analysis cap。
- 把 top-k 子空间外的能量称为 **"Off-manifold"** —— k=20 只覆盖 α=0 约 50% 的方差，补空间大部分是普通变异。
- 任何形式的 **causal** 主张。本研究是对已存 hidden states 的离线几何分析；因果检验需要实施新的 random/orthogonal injection 并重新采集数据。
- 跨模型比较 **raw α、PC axes 或 hidden-state values** —— mask、层数（L=9 vs L=6）与激活尺度都不同，所以相等的 α 不是相等的干预。
- **General behavioural predictive value** —— §3.5 未发现跨剂量一致的增量，§3.6 中满足方向性判据的模型仍具有负 R²。该结果仅说明当前 features 与 outcomes 下未检测到稳定增量，不用于裁决几何解释本身的价值。

结构性限制：

- **本批次没有独立的第二个 α=0 cell**，因此 α=0 reference geometry 的稳定性只能通过 train/validation subsampling 或 bootstrap 估计，无法由两个独立 α=0 批次交叉验证。
- **TEST 已用于探索性分析**，round 1 之后设计的检验均视为补充性结果。
- Commit-aligned 队列 **由 manipulation 自身的结果筛选**（coverage 与 pre-commit 可用性都随 α 变动）。
- **预测失败不抹掉描述性几何结果**；同样地，描述性几何也不授权任何预测或因果主张。

---

## 5. Status and Next Step

**Llama analysis: COMPLETE。** 当前证据将其定位为 **last-prefill explanatory geometry**：它为非对称剂量响应提供几何解释，但尚未形成完整的因果机制或稳定的行为预测模型。

当前 Llama pilot 不进一步扩展至 TLE、UMAP/t-SNE、全层扫描或新的 correctness-prediction analyses。Minimal decode analysis 用于界定 last-prefill finding 的适用范围。

**下一步为预先登记的 Qwen last-prefill analysis。** `PREREG_qwen_prefill.md` 将分析限定于 last-prefill，并使用 Qwen 自身的 α=0 basis、band `[16,22)` 与 commit locator。主要问题包括：正端剂量是否共享一个方向、位移幅度是否趋于饱和，以及位移相对于模型自身 PCA subspace 的 inside ratio。

Qwen 的结果将检验该几何解释能否扩展到跨模型差异。若 Llama 的位移沿共享轴增长并发生 overshoot，而 Qwen 的位移沿自身共享轴逐渐饱和，则这一对比可为 peak-versus-plateau 提供候选解释。若 Qwen 不呈现方向共享或幅度饱和，则当前结论仍限定为 Llama 的 last-prefill geometry。

其余剂量（No-CoT ±2/±4/+8）可用于 continuity analysis。由于这些条件使用相同题目和同一 basis，它们用于描述剂量曲线的连续性，而不构成独立验证集。

---

## Appendix. Artifact and provenance index

实现细节、执行命令、validation guards、tests 与故障记录见 `CLAUDE.md` § *Manifold pilot*。以下仅列出分析所使用的主要代码与数据产物。

**Scripts**（in the Dopamine repo）

| file | role |
|---|---|
| `check_hs_llama.py` | §3.1 H5 acceptance (server, read-only) |
| `manifold/split_manifest.py` + `.json` | §2.2 frozen split |
| `manifold_fit.py` | §2.3–2.4 basis fit + projection |
| `manifold_prefill_exact.py` | §3.3 ambient displacement decomposition |
| `manifold_prefill_direction.py` | §3.4 cross-dose cosine and scalar fit |
| `run_manifold_pilot.sh` | launcher |
| `test_check_hs_llama.py`, `manifold/test_split_manifest.py`, `test_manifold_fit.py` | guard suites |

**Offline analysis**（`RoleAnswer/manifold/`，不在 git 中）

`incremental.py`（§3.5 round 1）· `incremental2.py`（§3.5 round 2）·
`confirm_cot.py`（§3.6）· `decode_minimal.py`（§3.7）

**Pre-registrations**（`RoleAnswer/manifold/`）

`PREREG_incremental.md` · `PREREG_negative_arm_confirm.md` ·
`PREREG_decode_minimal.md` · `PREREG_qwen_prefill.md`

**Data artifacts**

- Server: `components/llama3/manifold/phase1b_eot/`（basis + 四个 No-CoT cell），
  `components/llama3/manifold/phase1b_eot_cot/`（CoT cells，复用同一 basis）
- Local: `RoleAnswer/llama3/dopamine/manifold/` —— `basis.npz`、
  `basis_meta.json`、`manifold_*.json`、`prefill_exact.json`、
  `prefill_direction.json`
- Source H5: `components/hidden_states/gsm8k/phase1b_eot/`
