# Manifold Geometry Analysis
## 0. Executive Summary

**核心问题。** RSN steering 究竟只是沿固定方向改变增益，还是会重组 hidden-state 的移动方向？这种几何差异能否进一步解释 Llama 的 `α=−6` 峰值，以及 Qwen 正端的高剂量平台？

**Llama 核心发现（last-prefill，decoder 18）。** RSN steering 呈现出一种 **piecewise geometry**，不能由单一的全局 scalar-gain 模型完整解释：

- **负剂量近似构成一维标量族。** `α=−8` 与 `α=−6` 的位移高度共线：`cos=0.989`，scalar-fit residual 仅为 `2.1%`，且 300/300 道题的方向符号一致。拟合得到的缩放系数 `k=1.379`，也与独立计算的位移幅度比 `24.63/17.67=1.394` 高度吻合。换言之，`−8` 基本上是沿着 `−6` 的方向继续放大，而不是进入完全不同的状态。

- **正剂量并非负剂量的简单镜像。** `α=+6` 与 `α=−6` 部分反向对齐（`cos=−0.662`），但二者并不共线：共享轴只能解释约 `cos²≈44%` 的位移能量，其余约 `56%` 位于正交方向。因此，`+6` 既不是较弱的 `−6`，也不是其完全反向版本，而包含明显的方向重组。

这一结果为 Llama 的非对称行为曲线提供了一个简洁的几何解释：**`α=−6` 到达较优的 working region，而 `α=−8` 沿同一方向继续推进并发生 overshoot。** 因此，性能崩溃不一定意味着模型进入了完全不同的状态，也可能是沿有效方向移动过度。

**证据边界。** 上述结论仅成立于 **last-prefill、decoder 18**——即 steering 的注入位置，也是不同 α 之间唯一能够严格按题、按 token 配对的位置。目前尚未检测到稳定的增量行为预测价值，且该 last-prefill 几何模式没有稳定延伸到 commit-aligned decode。因此，这些结果不能被表述为完整的推理轨迹机制，也不能证明状态已经“离开 natural manifold”。

**当前定位（2026-08-28 关闭）。** manifold 结果应定位为 **last-prefill explanatory geometry**：它为 `−6` 最优、`−8` overshoot 和正负剂量不对称提供了机制性线索，但不是独立的因果机制或行为预测主线。预先登记的 Qwen last-prefill 检验**已完成**（§3.8）：两模型的 entry displacement 都平滑、线性、单轴，因此该几何**不能**解释 Llama peak 与 Qwen plateau，差异定位到下游 commitment / decode dynamics——后续结果见 `AdaptiveThinking.md` §5.8。跨层 sensitivity（§3.9）进一步显示，正负两臂在注入点重合、随深度单调分离，故末层的分段几何是**层间传播的涌现结构**，而非注入本身的性质。

## 1. Research Questions

关于 α 对 hidden state 做了什么，有三种互相竞争的解释：

1. **Scalar gain** — steering 只改变沿固定方向的幅度。
2. **On-manifold retiming** — 轨迹仍留在 natural manifold 内，但速度、phase occupancy 或 commitment timing 改变。
3. **Directional reorganisation / off-subspace deviation** — 极端剂量让轨迹转向，同时伴随 accuracy 下降、loop 或 format failure。

**Cross-model question。** Llama 呈 asymmetric peaked response（最优 α=−6，−8 崩溃），Qwen 呈 high-dose plateau。若两者接受的都是 gain-like steering，候选解释是 **Llama 沿自己的轴 overshoot，而 Qwen 的位移 saturate**——这样就能从几何而非仅从行为解释 peak-vs-plateau。

**关于 prediction。** 我们进一步探索了 manifold features 能否在 `s_t`/`Z_t` 与 commitment-related variables 之外提供增量预测信息（§3.5、§3.6）。这项分析是一项辅助性的 *value check*，而不是判断几何解释是否成立的必要条件。个体题目的 correctness 主要受到题目难度影响，commit position 也具有较高噪声；此外，一维信号与高维几何描述回答的并不是同一个问题。因此，目前未检测到稳定的增量预测价值，并不否定下文的描述性几何结果。

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

**Cumulative explained variance**

| phase | m | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| prefill | 185 | .257 vs .039 | .365 vs .076 | **.499 vs .148** [.1477, .1489] |
| pre_commit | 2940 | .243 vs .006 | .344 vs .011 | **.446 vs .022** [.0222, .0223] |
| post_commit | 3576 | .296 vs .005 | .397 vs .010 | **.484 vs .020** [.0200, .0201] |
| decode_all | 185 | .641 vs .039 | .730 vs .076 | .812 vs .148 |


- **实测**：真实 α=0 hidden states 的累计解释方差。
- **matched isotropic null**：生成与真实数据具有相同 `m`、维度、题目数和加权方式，但各方向均匀随机的数据。
- **20 draws**：独立生成并计算了 20 组随机数据。
- **median [2.5, 97.5]**：报告这 20 次随机结果的中位数，以及 2.5%–97.5% 分位区间，作为近似 95% null 区间。
- `m` 是该 phase 用来拟合 PCA 的**总行数（hidden-state vectors）**。
  - `prefill: m=185`：185 个 train questions，每题取 last-prefill 的 1 个向量。
  - `pre_commit: m=2940`：150 道有 pre-commit 窗口的题，每题最多取 20 个 token 向量。
  - `post_commit: m=3576`：183 道题，每题最多取 20 个 token 向量。
  - `decode_all: m=185`：每道 train question 的全部 decode states 先求均值，因此每题只留下 1 个向量。
- 表中的 `.257/.365/...` 实际是**累计解释方差比例**，不是 NRE。这里标题建议改成：

前20个主成分解释了 **44.6%** 的方差；随机数据通常只能解释约 **2.2%**，而且20次结果都集中在约 2.22%–2.23%。因此，这说明真实 hidden states 的方差明显集中在少数方向上，具有显著的 **low-rank linear structure**；但仅凭 PCA 还不能称为完整的非线性 manifold。


### 3.3 Exact last-prefill PCA-subspace analysis

在 decoder 18 的 ambient space 中计算位移分解。300 道题依据 `question_idx` 跨条件配对，并统一投影到 α=0 training basis。定义 `f_k = ‖W_k d‖²/‖d‖²`，主结果采用 energy-pooled ratio。

**Primary (TEST split, k=20)：**

| α | mean‖d‖ | inside [95% CI] | outside |
|---|---|---|---|
| −8 | 24.63 | **21.4%** [20.9, 21.8] | 78.6% |
| −6 | 17.67 | **21.2%** [20.8, 21.7] | 78.8% |
| −4 | 10.66 | **19.0%** [18.6, 19.5] | 81.0% |
| −2 | 4.89 | **15.9%** [15.3, 16.4] | 84.1% |
| +2 | 4.63 | **11.9%** [11.3, 12.6] | 88.1% |
| +4 | 9.02 | **10.8%** [10.1, 11.5] | 89.2% |
| +6 | 13.23 | **9.8%** [9.2, 10.6] | 90.2% |
| +8 | 17.39 | **9.2%** [8.6, 9.8] | 90.8% |

1. 对每道题，在 decoder 18 取最后一个 prompt token 的 hidden state：
   - 基线：`h(0)`
   - steering 后：`h(α)`

2. 计算同一道题的精确位移：`dα = h(α) − h(0)`

3. 将这个位移分解为：

   - **inside**：落在 α=0 top-k PCA 子空间内的能量
   - **outside**：落在该子空间正交方向上的能量

- `mean‖d‖`：steering 把状态推了多远
- `inside ratio`：位移有多少沿 α=0 的主要变化方向
- `outside ratio`：有多少不在这些 top-k 方向内
- `k=5/10/20 sensitivity`：结论是否依赖 PCA 维数

**k sensitivity (TEST)：** 当 k 从 5 增至 10 和 20 时，−8 的 inside ratio 为 9.6 → 17.0 → 21.4，−6 为 9.5 → 16.8 → 21.2；两者均增加 **11.8pp**，且 per-dimension profile 高度相似。+6 则为 5.8 → 8.1 → 9.8，仅增加 **4.0pp**，表明其位移能量较少分布在 α=0 的前 20 个主成分上。

**Dose trend at k=20 (TEST)：** inside ratio 在两臂上单调变化，且方向相反——负端 15.9 → 19.0 → 21.2 → 21.4（−2 → −8），正端 11.9 → 10.8 → 9.8 → 9.2（+2 → +8）。inside ratio 是一个比例，对纯缩放不变，因此这一变化**不能**归因于位移幅度的增长。它提示两臂在剂量增大时都存在小幅方向漂移：负端逐渐更贴合 α=0 的主方向（在 −6/−8 之间趋于饱和），正端则逐渐更少落在其中。该漂移不改变 §3.4 中两臂各自近似单轴的结论。

**Split agreement at k=20**（train / val / test）：−8 为 21.8 / 21.1 / 21.4，−6 为 21.8 / 21.0 / 21.2，+6 为 10.7 / 9.5 / 9.8。三个 split 的结果接近，说明观察到的差异并非仅由用于拟合 basis 的 training questions 驱动。Pooled ratio 与 per-question mean 相差 <0.1pp，也未显示少数大位移样本主导总体结果。

**随机参照：** 各向同性位移在任意 20-D 子空间中的期望能量占比为 `20/4096 = 0.488%`。因此，+6 的 9.8% 约为随机参照的 **20×**，−6/−8 的约 21.2% 为 **43×**。三个剂量的位移能量均明显高于各向同性参照，但正剂量在 α=0 主方向上的集中程度约为负剂量的一半。

这一节说明：−8 和 −6 对 α=0 主结构的对齐程度非常接近，而 +6 的对齐程度明显更低。但这一节本身还不能证明 `−8` 和 `−6` 位移方向完全相同；这需要下一节的 cosine 与 scalar fit 才能确认。

### 3.4 Cross-dose direction and scalar fit

最小二乘 `d_a ≈ k·d_b`，
- `d_a = h(α_a) − h(0)`：剂量 `α_a` 造成的状态位移
- `d_b = h(α_b) − h(0)`：剂量 `α_b` 造成的状态位移
- `k`：寻找一个最合适的缩放倍数，使 `k·d_b` 尽可能接近 `d_a`

| pair | cos [95% CI] | k | residual | same-signed |
|---|---|---|---|---|
| −8 \| −6 | **0.989** [0.989, 0.990] | 1.379 | 0.021 | 100.0% |
| −4 \| −6 | **0.973** [0.971, 0.975] | 0.587 | 0.053 | 100.0% |
| −2 \| −6 | **0.889** [0.884, 0.895] | 0.246 | 0.209 | 100.0% |
| +2 \| −6 | **−0.710** [−0.720, −0.699] | −0.186 | 0.496 | 0.0% |
| +4 \| −6 | **−0.672** [−0.683, −0.660] | −0.343 | 0.549 | 0.0% |
| +6 \| −6 | **−0.662** [−0.673, −0.650] | −0.496 | 0.562 | 0.0% |
| +8 \| −6 | **−0.673** [−0.683, −0.662] | −0.662 | 0.548 | 0.0% |
| −8 \| +6 | −0.657 [−0.667, −0.647] | −1.222 | 0.569 | 0.0% |

- **The negative arm is approximately a one-dimensional scalar family**：cos 0.989，residual 2.1%，300/300 同号，且 `k = 1.379` 与独立算出的幅度比 24.63/17.67 = 1.394 吻合。加入 −4/−2 后该结构在四个剂量上一致：相邻 cos 为 0.955 (−4\|−2)、0.972 (−6\|−4)、0.989 (−8\|−6)，六对全部 100% 同号。
- **The positive arm is partially anti-aligned with a substantial orthogonal residual**：cos ≈ −0.66，共同轴 `cos² ≈ 44%`，正交约 56%。100% 同号意味着这是 *一致的部分*反向，而不是若干亚群的混合。该 frozen wording 描述的是**正臂相对负臂**的关系，不涉及正臂内部结构。
- **正臂内部同样近似单轴，且比负臂更共线**（ALL, n=300）：相邻 cos 为 0.990 (+2\|+4)、0.994 (+4\|+6)、0.997 (+6\|+8)，六对全部 100% 同号，`k_ls` 与独立幅度比的偏差仅 0.003–0.012（负臂为 0.015–0.718）。跨臂 cos 在 +2/+4/+6/+8 上恒定为 −0.710/−0.673/−0.664/−0.675，说明两臂夹角是与剂量无关的固定几何关系。
- 因此 steering **并不**在所有剂量上共享一条直线。其结构不是「一条干净的轴加上一团方向重组」，而是**两条各自近似一维的轴，彼此保持约 131° 的固定夹角**。

**主图：** `RoleAnswer/llama3/dopamine/manifold/fig_llama_nine_dose.png` 四联展示 accuracy、`‖dα‖`、相对 `d_−6` 的 cosine、`in_k20`。ALL n=300，post-hoc descriptive evidence（TEST 已在 §3.5 用尽）。Split consistency 与 28 对 pairwise 见 Supplement S1/S2。

Inside ratio（§3.3）与 cosine 提供相互一致但概念不同的证据：前者衡量位移能量在 α=0 top-k PCA subspace 中的比例，后者直接比较两个剂量的位移方向。现有分析尚未确定 +6 相对于负端轴的正交分量是否主要位于 top-k 子空间之外。

### 3.5 Incremental prediction

**Result: no stable incremental behavioural predictive value was detected。**

#### Commit position
- 预测目标：`commit position` 是模型生成过程中第一次出现答案提交标记 `####` 的 decode token 位置。例如第 80 个生成 token 出现 `####`，目标值就是约 80。它是一个连续数值，不是在预测具体哪个 token。

- 预测模型：**Ridge linear regression（λ=1）**，每个 α 单独建模：
  - 用 TRAIN questions 拟合
  - 在 TEST questions 上评估
  - 输入特征先按 TRAIN 的均值和标准差归一化

- 输入：
  - 基线模型：`commit_position ~ Z_prefill + prefill_confidence`
  - 增强模型：`commit_position ~ Z_prefill + prefill_confidence + geometry_features`

- geometry features：
  - prefill 状态的幅度
  - 落在 α=0 top-20 PCA 子空间内的比例
  - 子空间外的残差比例
  - `R²` 越高越好：解释了多少位置差异
  - `MAE` 越低越好：平均预测位置相差多少 token
  - `ρ` 越高越好：能否正确排列哪些题 commit 较早或较晚

| α | baseline R² / MAE / ρ | +geometry R² / MAE / ρ |
|---|---|---|
| −8 | .003 / 73.1 / .129 | .019 / 72.3 / .192 |
| −6 | .074 / 74.1 / .276 | .104 / 72.5 / .359 |
| +6 | .004 / 67.8 / .033 | −.004 / 68.3 / −.010 |

两个负剂量在 R²、MAE 与 ρ 上均改善，而 +6 在三项指标上均下降。根据预先设定的跨剂量一致性标准，该结果为 **mixed**。

#### Correctness

预测的是题目是否答对，使用的是 **logistic regression**。它与这张 commit-position 表是两个独立任务。

在该分析中，commitment-related variables 可作为 predictors。AUC 为 .700 / .386 / .547 → .688 / .497 / .443，log-loss 为 .6294 / .4990 / .6917 → .6429 / .4882 / .7247。结果在指标和剂量之间并不一致：−6 的 AUC 从 .386 提升至 .497，而 −8 与 +6 的 AUC 下降。

Round 1 存档（correctness，Z-only baseline）：AUC .485 / .479 / .570 → .503 / .617 / .458，一升两降。


### 3.6 CoT negative-arm conditional confirmation

H1 在 CoT projection 之前预先设定：*加入 prefill geometry 会改善负 α 的 commit-position prediction，而不会改善正 α。* CoT conditions 使用与原分析相同的模型设定，并投影到 **既有的 No-CoT α=0 basis**，以评估该几何表示在新条件下的迁移性。


**CoT α=−4, commit position：**

| | R² | MAE | ρ |
|---|---|---|---|
| baseline (Z, conf) | −0.101 | 56.5 | −0.091 |
| +geometry | **−0.056** | **53.9** | **0.121** |

R²、MAE 与 ρ 均朝预期方向变化，因此 **H1 的负端满足预设的方向性判据**。

**但绝对预测力仍然较弱**：加入几何后 R² 仍为负，表示模型的预测性能仍低于以 training mean 作为预测值；ρ 也接近零。因此，该结果应解释为 **a reproducible weak directional signal, not strong predictive evidence**。同一 cell 上的 correctness 明显变差：AUC .531 → .462，log-loss .3298 → .4242。该分析仅覆盖 H1 的 **负端**，因为 CoT 数据不包含正剂量。α=−4 也不属于最初形成 H1 的 −8/−6 剂量，因此这一结果同时检验了向未测剂量的外推。由于 CoT states 投影在 No-CoT α=0 basis 上，数值描述的是 CoT 状态相对于 **No-CoT reference geometry** 的位置。

### 3.7 Minimal pre/post-commit decode analysis

**模型在 commit 前后20个 token 的生成轨迹，是否延续 last-prefill 发现的几何规律。**

Decoder 18，k=20，TEST split，event-aligned distributions

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

- `phase`
  - `pre_commit`：`####` 出现前最多20个 token
  - `post_commit`：从 `####` 开始后的最多20个 token

- `n`：TEST split 中实际具有该阶段数据的题数。`pre_commit` 的 n 较少，是因为部分题在第一个 decode token 就 commit，没有 commit 前窗口。

- `NRE`：相对于 α=0 的 PCA 重建误差。
  - `1.0`：与 α=0 相当
  - `>1`：更难被 α=0 PCA 子空间重建
  - `<1`：比 α=0 更集中在该子空间附近  
  它表示重建误差大小，不能直接理解为方向或“离开 manifold”。

- `speed`：相邻 token 在 top-20 PCA 坐标中的平均移动距离。
  - 较大：hidden state 每一步变化较快
  - 较小：轨迹移动较慢

- `centroid dist.`：每道题该阶段的平均 PCA 坐标，到 α=0 TRAIN 同阶段中心的距离。
  - 较大：整体状态离 α=0 的典型区域更远
  - 只能在同一个 phase 内比较


**Commit 前：几乎没有清晰差异。**

- NRE 都接近 1：`0.988–1.087`
- speed 也很接近：`6.79–7.10`
- centroid distance 差异较小

说明 steering 在 commit 前窗口没有形成稳定、清晰的剂量分组。

**Commit 后：出现差异，但不是预期的正负分组。**

- `−8`：NRE 最高 `1.409`，speed 最低 `5.319`
- `−6`：NRE 最低 `0.836`，speed 最高 `6.903`
- `+6`：NRE `1.050`，接近 α=0

也就是说，last-prefill 时 `−8` 和 `−6` 几乎沿同一方向；但进入 post-commit 后，两者的轨迹表现反而明显不同。+6 也没有稳定成为最特殊的一组。

> 因此结论是：**last-prefill 的“负端共享轴、正端方向重组”没有稳定延伸到 commit-aligned decode。** 同时 `pre_commit` 的 n 随 α 明显变化，存在选择偏差，所以这一部分更适合作为边界说明，而不是强机制结论。

**Conclusion: last-prefill geometry did not stably extend to commit-aligned decode。** 


### 3.8 Qwen last-prefill cross-model check

这一分析检验 Qwen 的行为 plateau 是否来自 last-prefill 状态位移的饱和或方向改变。分析位置为 decoder 20（steering band 的最后一层）,并使用 Qwen 自己的 α=0 PCA basis。可用的 No-CoT 条件为 `−8 / 0 / +6 / +8 / +12`。完整运行配置、pre-registration 与验收细节见 `CLAUDE.md` 的 *Manifold pilot* 章节。

#### Complete results

**Displacement magnitude and inside ratio (TEST, k=20):**

| α | mean‖d‖ | ‖d‖/\|α\| | inside [95% CI] | outside |
|---|---|---|---|---|
| −8 | 119.50 | 14.94 | **6.59%** [6.15, 7.03] | 93.41% |
| +6 | 89.11 | 14.85 | **4.70%** [4.37, 5.06] | 95.30% |
| +8 | 120.68 | 15.09 | **5.62%** [5.27, 5.96] | 94.38% |
| +12 | 181.33 | 15.11 | **4.83%** [4.57, 5.09] | 95.17% |

**Cross-dose direction and scalar fit (ALL, n=300):**

| pair | cos [95% CI] | k | residual | same-signed |
|---|---|---|---|---|
| +6 \| +8 | **0.983** [0.982, 0.984] | 0.726 | 0.033 | 100.0% |
| +8 \| +12 | **0.980** [0.980, 0.981] | 0.654 | 0.039 | 100.0% |
| +6 \| +12 | **0.965** [0.964, 0.966] | 0.475 | 0.068 | 100.0% |
| +6 \| −8 | −0.782 [−0.787, −0.776] | −0.584 | 0.389 | 0.0% |
| +8 \| −8 | −0.754 [−0.759, −0.749] | −0.763 | 0.431 | 0.0% |
| +12 \| −8 | −0.802 [−0.805, −0.798] | −1.217 | 0.357 | 0.0% |

Split 一致性良好:cos 的 test 与 all 差异 ≤0.004,`in_k20` 的 test/all 差异 ≤0.4pp。

#### Main findings

**1. Qwen 的正剂量共享一条近似一维的方向。** +6、+8 和 +12 之间的 cosine 为 **0.965–0.983**,所有题目均为同号位移,scalar coefficient 也与独立计算的幅度比一致。因此,提高正剂量主要是在同一条轴上走得更远,而不是不断切换方向。

**2. 位移没有饱和,而是随 α 继续线性增长。** +6、+8 和 +12 的 `‖d‖/|α|` 分别为 **14.85、15.09 和 15.11**,几乎完全恒定；+12 处没有出现幅度压缩。与此同时,Qwen 的 accuracy 在 +8/+10/+12 已经进入平台（86.00 / 88.33 / 88.67,高剂量两两差异不显著）。也就是说：**内部位移仍在增加,但行为改善已经停止。** 这排除了“行为 plateau 是因为 entry-state displacement 已经饱和”这一解释。

**3. 位移与自然状态的主要 PCA 方向存在稳定对齐。** Qwen 正臂有 **4.70%–5.62%** 的位移能量落在 α=0 top-20 PCA subspace 内,约为各向同性随机参照 `20/3584 = 0.558%` 的 **8.4–10.1×**。因此这不是随机方向；但它低于 Llama 相对自身基线的倍数。由于两个模型使用不同 basis、band、mask 和 hidden size,这些绝对比例不能直接作为模型强弱比较。

**4. 正臂与现有的 −8 条件并非简单镜像。** 三个正剂量与 −8 的 cosine 为 −0.754 至 −0.802,对应约 141° 的稳定夹角。不过,Qwen 负臂只有 −8 一个剂量,因此只能描述正臂与这个负剂量的关系,不能判断 Qwen 负臂本身是否也是单轴结构。

#### Conclusion

> **Qwen 的正剂量在 last-prefill 上沿同一条轴线性扩张,没有出现方向转折或幅度饱和；但其 accuracy 已经平台化。**

因此,Qwen 的高剂量 plateau 不是由 entry-state saturation 造成的。结合 Llama 的结果,Llama 两臂与 Qwen 正臂在 last-prefill 都表现出平滑、近似单轴且随剂量增长的位移,行为却分别呈现 peak 与 plateau。**Last-prefill geometry 无法解释这种行为差异,差异更可能产生于后续的 commitment 与 decode dynamics。**

当前能够支持的范围是：Llama 的正负两臂与 Qwen 的正臂均呈单轴线性结构；Qwen 负臂因只有一个剂量点,尚不能判断。


### 3.9 Cross-layer sensitivity

前面的主要分析集中在 steering band 的最后一层。为了确认结论不是由单层选择造成的,这里进一步检查 Llama decoder 10–18 和 Qwen decoder 15–20 的全部 15 个 layer slot。每层都使用该层自己的 α=0 PCA basis；最后一层仍是预先指定的 primary layer。ALL n=300 用于描述跨层趋势,TEST 只用于核对趋势是否稳定。完整实现与验收细节见 `CLAUDE.md` 的 *Manifold pilot* 章节。

#### Complete per-layer results

**Llama — mean‖d‖ (TEST) 与臂内线性拟合 `‖d‖ = β·|α|`:**

| decoder | −8 | −6 | −4 | −2 | +2 | +4 | +6 | +8 | β⁻ | R²⁻ | β⁺ | R²⁺ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1.18 | 0.89 | 0.60 | 0.31 | 0.31 | 0.60 | 0.89 | 1.18 | 0.15 | 0.999 | 0.15 | 0.999 |
| 11 | 2.22 | 1.67 | 1.12 | 0.57 | 0.58 | 1.15 | 1.72 | 2.30 | 0.28 | 1.000 | 0.29 | 1.000 |
| 12 | 2.97 | 2.23 | 1.49 | 0.75 | 0.75 | 1.49 | 2.24 | 2.98 | 0.37 | 1.000 | 0.37 | 1.000 |
| 13 | 3.87 | 2.89 | 1.91 | 0.96 | 0.95 | 1.86 | 2.76 | 3.66 | 0.48 | 1.000 | 0.46 | 1.000 |
| 14 | 6.16 | 4.57 | 3.02 | 1.50 | 1.46 | 2.87 | 4.24 | 5.59 | 0.76 | 1.000 | 0.70 | 0.999 |
| 15 | 10.01 | 7.32 | 4.72 | 2.32 | 2.27 | 4.45 | 6.55 | 8.57 | 1.23 | 0.997 | 1.09 | 0.998 |
| 16 | 15.07 | 11.08 | 7.20 | 3.52 | 3.35 | 6.51 | 9.52 | 12.38 | 1.86 | 0.998 | 1.57 | 0.997 |
| 17 | 20.07 | 14.61 | 9.35 | 4.51 | 4.23 | 8.17 | 11.91 | 15.52 | 2.46 | 0.996 | 1.97 | 0.997 |
| **18** | 24.63 | 17.67 | 10.66 | 4.89 | 4.63 | 9.02 | 13.23 | 17.39 | 2.96 | 0.985 | 2.20 | 0.998 |

**Llama — cosine vs `d_−6` (ALL n=300),末列为 split 稳定性:**

| decoder | −8 | −4 | −2 | +2 | +4 | +6 | +8 | max\|all−test\| |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.999 | 0.998 | 0.972 | −0.888 | −0.953 | −0.970 | −0.977 | 0.0032 |
| 11 | 0.999 | 0.998 | 0.986 | −0.949 | −0.963 | −0.959 | −0.949 | 0.0013 |
| 12 | 0.998 | 0.998 | 0.987 | −0.950 | −0.950 | −0.936 | −0.920 | 0.0024 |
| 13 | 0.997 | 0.996 | 0.981 | −0.929 | −0.914 | −0.887 | −0.856 | 0.0050 |
| 14 | 0.997 | 0.996 | 0.984 | −0.943 | −0.928 | −0.907 | −0.885 | 0.0009 |
| 15 | 0.988 | 0.979 | 0.933 | −0.856 | −0.835 | −0.819 | −0.808 | 0.0043 |
| 16 | 0.988 | 0.979 | 0.935 | −0.847 | −0.821 | −0.807 | −0.803 | 0.0028 |
| 17 | 0.990 | 0.981 | 0.928 | −0.821 | −0.796 | −0.789 | −0.793 | 0.0045 |
| **18** | 0.989 | 0.972 | 0.886 | −0.710 | −0.673 | −0.664 | −0.675 | 0.0033 |

**Qwen — mean‖d‖ (TEST)、`‖d‖/|α|` 与 cosine vs `d_+8` (ALL):**

| decoder | −8 | +6 | +8 | +12 | /\|α\| +6 | +8 | +12 | cos +6 | +12 | −8 | dev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 14.03 | 10.52 | 14.03 | 21.04 | 1.75 | 1.75 | 1.75 | 1.000 | 1.000 | −1.000 | 0.0000 |
| 16 | 25.42 | 19.08 | 25.51 | 38.44 | 3.18 | 3.19 | 3.20 | 1.000 | 0.999 | −0.975 | 0.0012 |
| 17 | 39.06 | 29.53 | 39.65 | 60.53 | 4.92 | 4.96 | 5.04 | 0.999 | 0.996 | −0.958 | 0.0012 |
| 18 | 52.74 | 39.82 | 53.87 | 82.26 | 6.64 | 6.73 | 6.86 | 0.996 | 0.992 | −0.883 | 0.0016 |
| 19 | 81.85 | 61.93 | 85.48 | 128.16 | 10.32 | 10.68 | 10.68 | 0.988 | 0.987 | −0.774 | 0.0009 |
| **20** | 119.50 | 89.11 | 120.68 | 181.33 | 14.85 | 15.08 | 15.11 | 0.983 | 0.980 | −0.754 | 0.0011 |

#### Main findings

**1. 位移随深度放大,但每一层仍保持线性剂量关系。** Llama 各层的 `‖d‖ = β·|α|` 拟合均接近完全线性（R² = **0.985–1.000**）；Qwen 在同一层内的 `‖d‖/|α|` 也基本恒定。与此同时,位移幅度沿模型深度持续增大：Llama 的 α=−8 从 1.18 增至 24.63（约 **21×**）,Qwen 的 α=+12 从 21.04 增至 181.33（约 **8.6×**）。换句话说,剂量作用没有在中间层突然改变或饱和,而是在传播过程中逐层放大。

**2. 正负方向起初几乎完全相反,随后逐层分开。**

| | 第一个 steered layer | primary layer |
|---|---|---|
| Llama 跨臂 cos(−6 vs +8) | −0.977(dec 10) | **−0.675**(dec 18) |
| Qwen 跨臂 cos(+8 vs −8) | −0.99996(dec 15) | **−0.754**(dec 20) |

在第一个 steered layer,正负 α 只是沿同一个 steering direction 向相反方向移动,因此 Qwen 的跨臂 cosine 几乎等于 −1。随着信息经过后续层,跨臂 cosine 单调升高至 Llama 的 −0.675 和 Qwen 的 −0.754,说明两条路线逐渐不再互为镜像。这个趋势在 train、validation 和 test 中一致；Llama 第一层在极小位移下的轻微偏差属于数值精度影响,不改变整体趋势。

**3. 同一侧的不同剂量始终沿着近似同一方向。** Llama 正、负两臂的层内 cosine 分别保持在约 0.98 和 0.97 以上,Qwen 正臂保持在 0.98 以上,并且各层均为 100% 同号。因此,随深度改变的主要是**正臂与负臂之间的关系**,而不是每条臂内部的单轴结构。

#### Conclusion

> **RSN 注入起初只是同一方向上的“推”和“拉”。经过模型后续层的传播后,正负路线逐渐分开,最终形成 band 末端看到的两条轴。**

因此,§3.4 和 §3.8 中的固定跨臂夹角不是 steering input 本身的性质,而是层间传播形成的涌现结构。Llama 与 Qwen 都呈现这一规律,说明跨层分化具有跨模型一致性；但两者最终分别表现为 accuracy peak 和 plateau,所以该几何结构仍不足以解释行为曲线的差异。

主图见 `fig_xlayer_crossarm.png`。上面的表格保留全部 15 个 layer slot；逐层 PCA inside-ratio 数据保存在完整结果产物中,primary layer 的 k=20 结果仍见 §3.3 和 §3.8。

### 3.10 Inside-ratio sensitivity across k and layers (cross-model)

前面的分析使用 `k=20` 和 steering band 的最后一层。这里进一步检查结论是否依赖 PCA 维数或特定层，比较 **k=5/10/20** 以及 Llama 和 Qwen 的全部 15 个 steered-layer slots。

我们报告三种互补口径：

- **inside%**：位移能量落在 α=0 top-k PCA 子空间内的比例；
- **enrichment** `inside/(k/H)`：相对于同维度随机方向的倍数，用于校正两个模型不同的 hidden size；
- **natural-variance alignment** `inside/EV`：相对于该 top-k 子空间在 α=0 状态中所解释方差的比例。

两模型的 α=0 top-20 explained variance 都较稳定：Llama 为 **49.9–54.6%**，Qwen 为 **56.3–58.6%**。完整计算和验收细节见 `CLAUDE.md` 的 *Manifold pilot* 章节。

**Primary layer（Llama dec 18 / Qwen dec 20），TEST split。每格三个数：inside% / enrichment `inside/(k/H)` / alignment `inside/EV`：**

| model | cell | ‖d‖ | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| Llama | −8 | 24.63 | 9.6% / 78.4× / 0.373 | 17.0% / 69.4× / 0.464 | 21.4% / 43.7× / 0.428 |
| Llama | −6 | 17.67 | 9.5% / 77.6× / 0.369 | 16.8% / 69.0× / 0.461 | 21.2% / 43.5× / 0.426 |
| Llama | +6 | 13.23 | 5.8% / 47.9× / 0.228 | 8.1% / 33.1× / 0.222 | 9.8% / 20.2× / 0.197 |
| Qwen | −8 | 119.50 | 2.5% / 17.8× / 0.078 | 4.7% / 16.8× / 0.105 | 6.6% / 11.8× / 0.112 |
| Qwen | +6 | 89.11 | 1.7% / 12.3× / 0.054 | 3.4% / 12.1× / 0.076 | 4.7% / 8.4× / 0.080 |
| Qwen | +8 | 120.68 | 2.2% / 15.8× / 0.070 | 4.2% / 15.1× / 0.095 | 5.6% / 10.1× / 0.096 |
| Qwen | +12 | 181.33 | 1.7% / 12.4× / 0.054 | 3.7% / 13.2× / 0.082 | 4.8% / 8.7× / 0.082 |

#### Primary-layer result

三个 k 得到相同排序：**Llama 负臂 > Llama 正臂 > Qwen**。在 k=20 时，Llama 负臂约有 21% 的位移能量落在 top-20 子空间内，Llama `+6` 为 9.8%，Qwen 则集中在 4.7–6.6%。

Qwen 的 inside ratio 在不同剂量下变化很小。这与 §3.8 的单轴结果一致：提高剂量主要让状态沿同一方向移动得更远，并未明显改变该方向与 α=0 PCA 子空间的关系。

**跨层稳健性（dose-matched，取各模型层内最大值）：**

| k | −8：Llama max / Qwen max | +6：Llama max / Qwen max |
|---|---|---|
| 5 | 92.9× / 28.5× （3.3×） | 81.9× / 28.6× （2.9×） |
| 10 | 81.3× / 16.8× （4.8×） | 63.0× / 16.1× （3.9×） |
| 20 | 54.1× / 18.3× （3.0×） | 43.9× / 18.3× （2.4×） |

#### Cross-layer result

跨层结果与 primary layer 一致。在 layer-matched 比较中，Llama 的 enrichment 是 Qwen 的 **2.4–4.8 倍**；使用 `inside/EV` 后，差距仍为 **2.4–5.0 倍**。因此，这个差异不依赖某一个 k，也不是最后一层单独造成的。

不过，两模型未经层匹配的范围仍有重叠：k=20 enrichment 为 Llama **6.1–55.9×**、Qwen **4.7–18.3×**；alignment 为 Llama **0.055–0.538**、Qwen **0.047–0.175**。Llama 的低层可以低于 Qwen 的高层，所以不能概括为“Llama 在所有位置都更高”。

#### Conclusion

> **在相同相对层位置的比较下，Llama 的 entry displacement 比 Qwen 更贴近各自 α=0 状态的主要 PCA 方向。**

这是一项稳定的描述性差异，但不是 off-manifold 证据。Top-20 PCA 只覆盖约一半自然方差，落在其外的成分仍可能属于正常状态变化。更重要的是，两模型的 entry displacement 都保持平滑、线性和近似单轴，而行为分别形成 peak 与 plateau。因此，inside-ratio 差异不能解释行为曲线，后续分析仍应转向 commitment 与 decode dynamics。

## 4. Interpretation

### 4.1 Natural hidden states exhibit structured variation

α=0 hidden states 具有明显的低秩线性结构。前 20 个 PCA directions 在 commit 前后解释约 **45%–48%** 的方差，而匹配条件下的随机基线约为 **2%**。

这说明自然状态的变化集中在少数主要方向上，但不能据此断言存在完整的非线性 manifold，也不能把 `k=20` 当作模型的内在维度。

### 4.2 Steering produces piecewise rather than globally scalar geometry

Llama 在 last-prefill、decoder 18 呈现清楚的分段结构：

- **负臂内部近似单轴。** `−6` 与 `−8` 的 cosine 为 **0.989**，scalar-fit residual 为 **2.1%**；`−8` 主要是沿 `−6` 的方向继续走远。
- **正臂内部也近似单轴。** 相邻剂量 cosine 为 **0.990–0.997**，说明增加剂量主要改变幅度，而非不断改变方向。
- **正负两臂不是简单镜像。** 跨臂 cosine 约为 **−0.66**，对应约 **131°** 的夹角和约 **56%** 的正交残差。

因此：

> **Llama 的 last-prefill response 是 piecewise scalar，而不是 global scalar。负、正剂量分别沿两条近似一维轴随剂量扩张；两条轴之间保持约 131° 的固定夹角。**

这个结构随层深逐渐形成：在第一个 steered layer，正负剂量近似沿同一方向推拉；经过后续层传播后，两条路线才逐渐分开。末层的分段几何因此是模型内部传播的结果，而不是 steering mask 本身的性质。

行为上，`−6` 的 accuracy 为 **79.67%**，`−8` 则降至 **40.67%**，但两者仍沿几乎相同的轴。因此，Llama 的峰值不是 entry direction 突然改变造成的，更符合沿负端轴 **overshoot** 后被下游非线性放大的解释。

### 4.3 The clean entry geometry does not persist unchanged during decoding

清晰的分段结构主要出现在 last-prefill。进入生成后：

- commit 前各剂量的 NRE、trajectory speed 和 centroid distance 差异很小；
- commit 后虽然出现差异，但 `−6` 与 `−8` 不再保持相同分组；
- `+6` 也没有稳定成为最特殊的条件。

这不表示 decode states 没有结构，而是说明 **last-prefill 的跨剂量关系没有原样延续**。生成文本、commitment、答案格式、循环和自回归反馈会让轨迹逐渐分叉。

> **Steering 在 entry boundary 形成清晰、可控的初始位移；decode dynamics 则是对这个初始条件的非线性展开。**

### 4.4 Relation to Dopamine and the Thinking Curve

Manifold 为 Thinking Curve 提供的是一个边界条件：**entry gain 很简单，行为转换并不简单。**

- Llama 的 `−6 → −8` 沿同一条轴继续增长，行为却由最佳点转为崩溃。
- Qwen 的 `+6 → +8 → +12` 位移持续线性增长，accuracy 却进入平台。
- 两模型的正负路线都会随层深逐渐分化，但这种共同的跨层结构仍对应不同的行为曲线。

因此，Qwen plateau 不是 entry-state saturation，Llama peak 也不是 entry direction 转折。更合适的跨模型框架是：

> **Entry gain is smooth and near-linear, whereas the downstream dose–response function is model-dependent.**

这把主要问题定位到 commitment timing、early-token selection 和 decode dynamics。Layer-matched 条件下，Llama 与自身主要 PCA 方向的对齐强于 Qwen（§3.10），但该差异同样不能区分 peak 与 plateau。

这里的 dopamine 只是一种功能类比：适度 gain 可能有益，过度 gain 可能造成 overshoot。α 的正负不能直接对应生物 dopamine 的增减，PCA 方向也不是生物神经回路。

#### 跨模型总结图

主图 `RoleAnswer/llama3/dopamine/manifold/fig_crossmodel_summary.png` 汇总四项结果：

- **A：** Llama accuracy 形成 peak，Qwen 形成 plateau；
- **B：** 两模型各剂量臂的 entry displacement 都近似线性增长；
- **C：** 同一臂内相邻剂量方向高度一致（全部 `|cos| ≥ 0.95`）；
- **D：** 几何幅度与行为结果明显解耦。

最直接的例子是 Llama `−6` 与 `−8`：两者沿几乎同一方向，归一化幅度为 **1.00 / 1.39**，accuracy change 却为 **+19.7 / −19.3pp**。Qwen `+8 → +12` 的位移增加约 50%，accuracy 只增加 2.3pp。因此：

> **两个模型的 entry displacement 都平滑且近似线性，但 Llama 的行为形成 peak、Qwen 的行为形成 plateau。位移的方向与幅度都不足以预测这一差异。**

图中每个模型都使用自己的 α=0 basis、layer band 和归一化参考；raw α 与位移绝对大小不作跨模型比较。

### 4.5 Scope of the conclusion

当前结果支持一条简洁的计算链：

> **对称 steering 注入 → 随层深产生正负路线分化 → last-prefill 形成 piecewise-scalar geometry；Llama peak 与 Qwen plateau 未由这段 entry geometry 解释，差异需要在下游 commitment/decode dynamics 中寻找。**

其中能够确定的是：

- Llama 正、负两臂与 Qwen 正臂均呈近似单轴、线性增长；
- Qwen plateau 不是 entry-state saturation；
- Llama `−8` collapse 更符合沿 `−6` 轴 overshoot，而不是进入新方向；
- last-prefill 的清晰几何没有稳定延伸到 commit-aligned decode。

证据边界如下：

- PCA 只证明低秩线性结构；top-k 之外不能直接称为 off-manifold；
- Qwen 负臂只有 `−8` 一个剂量，是否单轴无法判断；
- decode 比较受到轨迹分叉和 commit-window coverage 差异影响；
- 离线 hidden-state 分析不能替代 random/orthogonal direction 的真实因果注入；
- layer-matched inside-ratio 差异是描述性结果，不能解释 peak 与 plateau。

因此，manifold pilot 的最终定位是：

> **Last-prefill explanatory geometry：它约束了可能的机制解释，并把跨模型差异定位到下游动力学；它不是独立预测模型，也不是完整的因果机制。**


## Supplement

### S1. Split consistency of the direction analysis

相对参考轴 `d_−6` 的 pooled cosine，四个 split 分别列出。差异 ≤0.011，说明 §3.4 的方向结论并非由拟合 basis 所用的 training questions 驱动。

| α | train (185) | val (55) | test (60) | all (300) |
|---|---|---|---|---|
| −8 | 0.990 | 0.989 | 0.989 | 0.989 |
| −4 | 0.972 | 0.971 | 0.973 | 0.972 |
| −2 | 0.886 | 0.882 | 0.889 | 0.886 |
| +2 | −0.712 | −0.701 | −0.710 | −0.710 |
| +4 | −0.675 | −0.665 | −0.672 | −0.673 |
| +6 | −0.666 | −0.658 | −0.662 | −0.664 |
| +8 | −0.676 | −0.670 | −0.673 | −0.675 |

`in_k20` 的 split 一致性同样成立（§3.3，差异 ≤0.7pp）。

### S2. All 28 pairwise dose comparisons

八个剂量的全部两两比较（ALL, n=300）。主叙事只使用其中相对 `d_−6` 的一列（§3.4）与臂内相邻对；本表完整保留以供核对，**不作为主叙事**。`residual ≡ 1 − cos²`，因此 `k` 是三者中唯一独立的数值。

| pair | cos | 95% CI | k | residual | same-signed |
|---|---|---|---|---|---|
| −8 \| −6 | 0.989 | [0.989, 0.990] | 1.375 | 0.021 | 100.0% |
| −8 \| −4 | 0.942 | [0.941, 0.943] | 2.168 | 0.113 | 100.0% |
| −8 \| −2 | 0.858 | [0.855, 0.861] | 4.319 | 0.264 | 100.0% |
| −8 \| +2 | −0.699 | [−0.703, −0.695] | −3.744 | 0.511 | 0.0% |
| −8 \| +4 | −0.667 | [−0.671, −0.663] | −1.834 | 0.555 | 0.0% |
| −8 \| +6 | −0.661 | [−0.664, −0.657] | −1.237 | 0.564 | 0.0% |
| −8 \| +8 | −0.672 | [−0.676, −0.669] | −0.957 | 0.548 | 0.0% |
| −6 \| −4 | 0.972 | [0.971, 0.973] | 1.610 | 0.055 | 100.0% |
| −6 \| −2 | 0.886 | [0.884, 0.889] | 3.210 | 0.215 | 100.0% |
| −6 \| +2 | −0.710 | [−0.714, −0.705] | −2.733 | 0.497 | 0.0% |
| −6 \| +4 | −0.673 | [−0.678, −0.668] | −1.330 | 0.547 | 0.0% |
| −6 \| +6 | −0.664 | [−0.668, −0.659] | −0.894 | 0.559 | 0.0% |
| −6 \| +8 | −0.675 | [−0.678, −0.671] | −0.691 | 0.545 | 0.0% |
| −4 \| −2 | 0.955 | [0.953, 0.956] | 2.088 | 0.089 | 100.0% |
| −4 \| +2 | −0.784 | [−0.788, −0.779] | −1.823 | 0.386 | 0.0% |
| −4 \| +4 | −0.741 | [−0.746, −0.736] | −0.885 | 0.450 | 0.0% |
| −4 \| +6 | −0.727 | [−0.731, −0.722] | −0.591 | 0.472 | 0.0% |
| −4 \| +8 | −0.731 | [−0.735, −0.727] | −0.452 | 0.466 | 0.0% |
| −2 \| +2 | −0.904 | [−0.907, −0.901] | −0.961 | 0.183 | 0.0% |
| −2 \| +4 | −0.862 | [−0.865, −0.858] | −0.470 | 0.257 | 0.0% |
| −2 \| +6 | −0.838 | [−0.841, −0.834] | −0.312 | 0.298 | 0.0% |
| −2 \| +8 | −0.830 | [−0.833, −0.827] | −0.235 | 0.311 | 0.0% |
| +2 \| +4 | 0.990 | [0.990, 0.990] | 0.508 | 0.020 | 100.0% |
| +2 \| +6 | 0.971 | [0.971, 0.972] | 0.340 | 0.056 | 100.0% |
| +2 \| +8 | 0.955 | [0.954, 0.956] | 0.254 | 0.088 | 100.0% |
| +4 \| +6 | 0.994 | [0.994, 0.994] | 0.677 | 0.012 | 100.0% |
| +4 \| +8 | 0.983 | [0.983, 0.983] | 0.509 | 0.034 | 100.0% |
| +6 \| +8 | 0.997 | [0.997, 0.997] | 0.758 | 0.007 | 100.0% |

跨臂的 cos 随 |α| 变化（−2\|+2 的 −0.904 到 −6\|+6 的 −0.664），因为低剂量位移较小、受各题噪声影响更大；四个「同幅度」对中的三个（±4/±6/±8）稳定在 −0.66…−0.74。

### S3. Nine-point accuracy, same H5 batch

由九个 cell 的 `generated` 文本经冻结的 GSM8K offline extractor 重算（`analyze_first_last_acc` 的 `all_hash` / `norm_gsm8k` / `fallback_gsm8k`，import 而非重写）。九格与各 cell 的 stored accuracy 差值均为 0.00，且九格为同一批 300 题。

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| first_acc | 40.67 | **79.67** | 74.33 | 68.33 | 60.00 | 55.33 | 51.33 | 51.67 | 49.67 |

这是 184 机器、bs=1 的 HS 批次自身的 accuracy；依 184-vs-182 规则，**不得**与 182 生产剂量表逐题混用。

### S4. Inside-ratio enrichment, all 15 layer slots

`inside / (k/H)`，TEST split，H = 4096 (Llama) / 3584 (Qwen)。§3.10 的稳健性判断即基于本表。

**Llama, k=20:**

| α | dec10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | **18** |
|---|---|---|---|---|---|---|---|---|---|
| −8 | 6.1 | 10.1 | 12.0 | 25.3 | 37.0 | 37.1 | 50.7 | 54.1 | **43.7** |
| −6 | 6.1 | 10.1 | 11.4 | 23.5 | 35.5 | 36.8 | 51.6 | 55.7 | **43.5** |
| −4 | 6.3 | 10.2 | 10.7 | 21.4 | 33.9 | 37.3 | 52.5 | 55.9 | **39.0** |
| −2 | 7.2 | 10.6 | 10.0 | 19.2 | 32.3 | 37.2 | 51.5 | 53.3 | **32.5** |
| +2 | 8.1 | 11.2 | 9.6 | 15.0 | 29.0 | 36.2 | 47.2 | 45.7 | **24.4** |
| +4 | 6.8 | 10.7 | 8.9 | 13.2 | 27.9 | 35.6 | 45.3 | 42.7 | **22.2** |
| +6 | 6.5 | 10.5 | 8.5 | 11.8 | 26.9 | 35.2 | 43.9 | 40.7 | **20.2** |
| +8 | 6.4 | 10.2 | 8.2 | 11.0 | 26.2 | 34.7 | 42.7 | 39.2 | **18.8** |

**Qwen, k=20:**

| α | dec15 | 16 | 17 | 18 | 19 | **20** |
|---|---|---|---|---|---|---|
| −8 | 18.3 | 6.5 | 4.7 | 7.1 | 10.4 | **11.8** |
| +6 | 18.3 | 5.7 | 4.8 | 6.5 | 9.2 | **8.4** |
| +8 | 18.3 | 5.7 | 5.1 | 7.1 | 10.7 | **10.1** |
| +12 | 18.3 | 5.8 | 5.6 | 7.6 | 10.2 | **8.7** |

**Natural-variance alignment `inside / α=0 explained variance`, k=20**（各层 α=0 explained variance：Llama 49.9–54.6%、Qwen 56.3–58.6%，均随层高度稳定）：

| model | α | 各 steered layer（左→右为由浅至深，粗体为 primary） |
|---|---|---|
| Llama | −8 | 0.055 0.094 0.113 0.231 0.339 0.345 0.490 0.521 **0.428** |
| Llama | −6 | 0.055 0.094 0.107 0.215 0.326 0.343 0.499 0.537 **0.426** |
| Llama | +6 | 0.058 0.097 0.080 0.108 0.247 0.328 0.424 0.392 **0.197** |
| Llama | +8 | 0.057 0.095 0.077 0.100 0.241 0.324 0.413 0.378 **0.184** |
| Qwen | −8 | 0.175 0.064 0.047 0.070 0.100 **0.112** |
| Qwen | +6 | 0.175 0.056 0.048 0.064 0.088 **0.080** |
| Qwen | +8 | 0.175 0.056 0.051 0.070 0.103 **0.096** |
| Qwen | +12 | 0.175 0.057 0.055 0.075 0.098 **0.082** |

k=5 与 k=10 的完整表格保存在 `xlayer/exact_*.json` 中；三个 k 的排序结论一致（§3.10）。Qwen 第一个 steered layer 四个剂量读数完全相同（18.3），是解析必然：该层位移即 `α·mask`，方向与剂量无关，故 inside ratio 与 α 无关。

## Appendix. Artifact and provenance index

分析所用的全部代码、数据产物路径、执行命令、validation guards、tests 与故障记录，统一收录在
`CLAUDE.md` § *Manifold pilot*（scripts 与 guard suites 见其正文与 **Local checks** 一节；
数据产物路径见 *Where the artifacts live*；offline 分析脚本与四份 pre-registration 位于
`RoleAnswer/manifold/`，不在本 git repo 内）。

本文件只承载研究叙事与结果；实现层面的单一事实来源是 `CLAUDE.md`，两处不一致时以其为准。
