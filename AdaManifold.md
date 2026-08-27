# Manifold Geometry Analysis
## 0. Executive Summary

**核心问题。** RSN steering 究竟只是沿固定方向改变增益，还是会重组 hidden-state 的移动方向？这种几何差异能否进一步解释 Llama 的 `α=−6` 峰值，以及 Qwen 正端的高剂量平台？

**Llama 核心发现（last-prefill，decoder 18）。** RSN steering 呈现出一种 **piecewise geometry**，不能由单一的全局 scalar-gain 模型完整解释：

- **负剂量近似构成一维标量族。** `α=−8` 与 `α=−6` 的位移高度共线：`cos=0.989`，scalar-fit residual 仅为 `2.1%`，且 300/300 道题的方向符号一致。拟合得到的缩放系数 `k=1.379`，也与独立计算的位移幅度比 `24.63/17.67=1.394` 高度吻合。换言之，`−8` 基本上是沿着 `−6` 的方向继续放大，而不是进入完全不同的状态。

- **正剂量并非负剂量的简单镜像。** `α=+6` 与 `α=−6` 部分反向对齐（`cos=−0.662`），但二者并不共线：共享轴只能解释约 `cos²≈44%` 的位移能量，其余约 `56%` 位于正交方向。因此，`+6` 既不是较弱的 `−6`，也不是其完全反向版本，而包含明显的方向重组。

这一结果为 Llama 的非对称行为曲线提供了一个简洁的几何解释：**`α=−6` 到达较优的 working region，而 `α=−8` 沿同一方向继续推进并发生 overshoot。** 因此，性能崩溃不一定意味着模型进入了完全不同的状态，也可能是沿有效方向移动过度。

**证据边界。** 上述结论仅成立于 **last-prefill、decoder 18**——即 steering 的注入位置，也是不同 α 之间唯一能够严格按题、按 token 配对的位置。目前尚未检测到稳定的增量行为预测价值，且该 last-prefill 几何模式没有稳定延伸到 commit-aligned decode。因此，这些结果不能被表述为完整的推理轨迹机制，也不能证明状态已经“离开 natural manifold”。

**当前定位。** Llama 的主要分析已经完成，manifold 结果应定位为 **last-prefill explanatory geometry**：它为 `−6` 最优、`−8` overshoot 和正负剂量不对称提供了机制性线索，但不是独立的因果机制或行为预测主线。Llama–Qwen 是否共享同一套解释，仍需由预先登记的 Qwen last-prefill analysis 检验。

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


下面这版可以直接替换现有的 `## 4. Interpretation`。我把重点改成“发现了什么—如何理解—适用范围”，减少规则式表述。

## 4. Interpretation

### 4.1 Natural hidden states exhibit structured variation

α=0 的 hidden states 并不是在4096维空间中均匀变化。前20个 PCA directions 在 pre-commit 和 post-commit 阶段解释了约45%–48%的方差，而匹配样本规模和维度的随机基线只能解释约2%。这说明模型的自然推理状态具有明显的 **low-rank linear structure**。

这一结果表明 PCA 提取到的并非单纯由高维、小样本造成的随机结构。但它只证明方差集中在少数线性方向上，不能进一步推断存在一个确定的非线性 manifold，也不能把 `k=20` 解释为模型状态的内在维度。

### 4.2 Steering produces piecewise rather than globally scalar geometry

最清晰的跨剂量结果出现在 **last-prefill、decoder 18**，即 steering 注入完成、模型尚未开始生成的位置。

在负剂量一侧，−6 与 −8 的状态位移几乎完全共线：

- cosine similarity 为 **0.989**；
- scalar-fit residual 仅为 **2.1%**；
- 300道题的位移方向全部同号；
- `d_{−8} ≈ 1.379·d_{−6}`，与两者独立计算的位移幅度比基本一致。

因此，−8 并不是进入了一个全新的状态方向，而主要是沿着 −6 已经使用的方向继续移动。结合行为结果，α=−6 对应最高准确率，而 α=−8 出现性能崩溃，这与一种 **overshoot** 解释一致：−6 到达有效工作区域，−8 则沿同一条轴移动过远。

+6 呈现不同的几何关系。它与负剂量位移部分反向对齐，cosine 约为 −0.66，但仍有约56%的能量无法由负端轴的反向缩放解释。因此，+6 既不是 −6 的简单镜像，也不是同一方向上幅度更小的状态，而是包含明显的额外方向成分。

补齐 ±2/±4/+8 后（§3.3、§3.4），这一图景变得更明确：**正臂内部同样近似单轴，且比负臂更共线**（相邻 cos 0.990/0.994/0.997，全部 100% 同号），而跨臂 cosine 在四个正剂量上恒定约 −0.66。因此额外的方向成分并不是随剂量出现的重组，而是两臂之间与剂量无关的固定夹角。

综合来看，RSN steering 不能用一条覆盖所有剂量的全局 scalar-gain axis 描述。更合适的解释是：

> **Llama 的 last-prefill response 是 piecewise scalar，而不是 global scalar。负、正剂量分别沿两条近似一维轴随剂量扩张；两条轴之间保持约 131° 的固定夹角。**

需要区分三个层次：

- **臂内**：近似 scalar family，方向基本稳定，剂量主要改变幅度（两臂 ‖d‖ 对 |α| 均线性，R² = 0.998 / 0.9999，且均未见饱和）。
- **跨臂**：不是同一轴的正反镜像，cos ≈ −0.66，存在约 56% 的正交残差。
- **行为对应**：−6 → −8 的几何仍然平滑线性，cos 0.989、residual 2.1%、幅度落在线性外推上，而 accuracy 从 79.67 骤降至 40.67。因此行为峰值**不是**由 entry geometry 的方向转折造成的；更符合的解释是沿负端轴 overshoot 之后被下游非线性放大。

这为 Llama 的非对称行为曲线提供了对应的几何解释：几何在两臂内都是平滑的，行为的非对称性因此需要在 entry geometry 之外寻找来源。

### 4.3 The clean entry geometry does not persist unchanged during decoding

上述剂量结构主要成立于 last-prefill。进入生成阶段后，结果发生变化：

- pre-commit 阶段各剂量的 NRE、trajectory speed 和 centroid distance 差异较小；
- post-commit 阶段虽然出现几何差异，但 −6 与 −8 不再表现为同一组；
- +6 也不再是与其他剂量分离最明显的条件。

因此，不能把 last-prefill 的“负端共享轴、正端方向重组”直接延伸到整条生成轨迹。

这不表示 decode hidden states 没有几何结构。α=0 在 commit 前后仍然表现出显著的低秩谱集中；缺失的是**跨剂量关系的稳定延续**。一个合理的解释是，steering 首先在输入边界形成清晰的状态位移，随后不同条件生成不同文本，并受到 commitment、答案格式、循环和自回归反馈的共同影响，使轨迹逐渐分叉和重组。

因此，manifold 结果与此前 `G/Z` 分析共同指向一个更一般的模式：

> **Steering 在 entry boundary 产生清晰、可控的变化，但 decode dynamics 并不是该 entry effect 的简单线性传播。**

### 4.4 Relation to Dopamine and the Thinking Curve

这些结果为 Thinking Curve 提供了一个计算层面的解释。

负剂量共享轴可以被理解为一条稳定的 **gain-control axis**。沿这条轴增加位移，模型先到达 α=−6 的有效工作区域，随后在 α=−8 发生 overshoot。这与“适度调节有益、过度调节有害”的曲线形态一致。

但是，这只是计算几何上的类比。α 的正负不能直接等同于生物多巴胺的增加或减少，PCA 方向也不是生物神经回路。当前结果说明的是 RSN steering 如何重组模型状态，而不是模型内部存在真实的多巴胺机制。

对于跨模型差异，一个值得继续检验的假设是：

- Llama 的有效位移在负端持续增大，最终发生 overshoot；
- Qwen 的正端位移可能逐渐压缩或饱和，因此行为表现形成 plateau。

这一假设需要在 Qwen 自身的 α=0 PCA basis、layer band 和剂量范围内独立检验。不同模型之间不能直接比较 raw α、PCA axes 或 hidden-state 数值。

### 4.5 Scope of the conclusion

当前证据支持的核心结论是：

> **Llama 的 RSN steering 在 last-prefill 形成了清晰的分段几何：负剂量主要沿共享轴缩放，−8 相对 −6 表现为沿轴 overshoot；+6 则包含显著的方向重组。该结构为非对称准确率曲线提供了解释性几何，但没有稳定延伸到 commit-aligned decoding。**

这一结论的范围需要保持明确：

- PCA 证明的是低秩线性结构，而不是完整的非线性 manifold。
- Top-20 子空间外的能量不能直接称为 off-manifold，因为该子空间只覆盖约一半的 α=0 方差。
- 精确的跨剂量方向结论目前以 last-prefill、decoder 18 为主。
- 增量预测分析没有检测到稳定结果，但预测 correctness 或 commit position 并不是几何解释成立的必要条件。
- 这些结果来自离线 hidden-state 分析，不构成 steering 方向具有因果特异性的直接证据。
- Commit-aligned 样本覆盖率随 α 改变，因此 decode 比较同时受到生成轨迹分叉和样本选择的影响。

因此，manifold pilot 最合适的定位是：

> **它是对 Llama entry-state steering 的解释性几何分析，补充了 Thinking Curve 的行为结果；它不是独立的预测模型，也不是完整的因果机制。**


## Appendix. Artifact and provenance index

分析所用的全部代码、数据产物路径、执行命令、validation guards、tests 与故障记录，统一收录在
`CLAUDE.md` § *Manifold pilot*（scripts 与 guard suites 见其正文与 **Local checks** 一节；
数据产物路径见 *Where the artifacts live*；offline 分析脚本与四份 pre-registration 位于
`RoleAnswer/manifold/`，不在本 git repo 内）。

本文件只承载研究叙事与结果；实现层面的单一事实来源是 `CLAUDE.md`，两处不一致时以其为准。
