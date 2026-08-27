# Manifold Geometry Analysis
## 0. Executive Summary

**核心问题。** RSN steering 到底是沿单一方向的 scalar gain、是 natural manifold 内部的 retiming，还是偏离该 manifold？同一套几何解释能否说明为什么 Llama 峰值在 α=−6，而 Qwen 在正端呈 plateau？

**Llama 核心发现（last-prefill, decoder 18）。** Steering 是 *piecewise* 的，不是一个全局的 scalar gain：

- **The negative arm is approximately a one-dimensional scalar family**：cos(−8, −6) = 0.989，scalar-fit residual 2.1%，300/300 题同号，而 least-squares `k = 1.379` 与独立算出的幅度比 24.63/17.67 = 1.394 吻合。所以 −8 就是沿同一条轴走得更远的 −6。
- **The positive arm is partially anti-aligned with a substantial orthogonal residual**：cos(−6, +6) = −0.662，共同轴只承载 `cos² ≈ 44%` 的能量，约 56% 是正交的。+6 既不是镜像，也不是更小的 −6。

由此得到本条线最有价值的解释性结果：**α=−6 抵达一个 working region，而 α=−8 是沿同一条轴 overshoot**——崩溃不必意味着到达了一个完全不同的状态。

**证据边界。** 以上全部成立于 **last-prefill**，即注入位置，也是唯一严格 α-matched 的位置。**No stable incremental behavioural predictive value was detected**（按冻结的三剂量判据），且 **last-prefill geometry did not stably extend to commit-aligned decode**。

**当前定位。** Llama 分析 **已完成**；manifold 被定位为 **last-prefill explanatory geometry**，不是机制主线，也不是预测线。

**下一步。** 只有一件事：冻结的 Qwen last-prefill 分析（`PREREG_qwen_prefill.md`）。manifold 是否能以超出 supplement 的身份进论文，由它的结果决定。

---

## 1. Research Questions

关于 α 对 hidden state 做了什么，有三种互相竞争的解释：

1. **Scalar gain** — steering 只改变沿固定方向的幅度。
2. **On-manifold retiming** — 轨迹仍留在 natural manifold 内，但速度、phase occupancy 或 commitment timing 改变。
3. **Directional reorganisation / off-subspace deviation** — 极端剂量让轨迹转向，同时伴随 accuracy 下降、loop 或 format failure。

**Cross-model question。** Llama 呈 asymmetric peaked response（最优 α=−6，−8 崩溃），Qwen 呈 high-dose plateau。若两者接受的都是 gain-like steering，候选解释是 **Llama 沿自己的轴 overshoot，而 Qwen 的位移 saturate**——这样就能从几何而非仅从行为解释 peak-vs-plateau。

**关于 prediction。** 计划中有一条约束性条件：manifold 必须提供超出 `s_t`/`Z_t` + commitment behaviour 的信息，否则降为 auxiliary visualization。该检验已执行（§3.5、§3.6）且 **未通过**。但 incremental prediction 始终只是一个 *value check*，从来不是目的：区分上述三种解释并不需要预测哪一题会答对。**预测失败不抹掉描述性几何结果。**

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

Band `[11,20)` → decoder layers 10–18 (L=9)。**Primary layer: decoder 18**（export slot `8`）；sensitivity：decoder 10（slot `0`）；其余层为补充。配对一律按 `question_idx`，绝不按行序。

### 2.2 Frozen question split

60/20/20 **按题**划分，实际落为 **185/55/60**，版本 `manifold-split-v1`。

- **Train** — 拟合 α=0 PCA basis。只做拟合，不选 k。
- **Validation** — `k = 5/10/20` robustness。
- **Test** — 最终剂量比较与 prediction checks。

规则：所有 cell 共用同一个 split（per-cell split 会把剂量效应和题目难度混淆）；按题而非按 token 划分（同一题内的 token 相关，token split 会泄漏）；计数 **不**重新平衡为 180/60/60（阈值落在 hash 值上，排序切片会让每个分配依赖整个集合，破坏扩展稳定性）。

**k rule (frozen)。** 主分析 `k = 20`；`k = 5/10/20` 作为 sensitivity 报告，只有三者方向一致才可宣称 robust；top-30 spectrum 仅作 diagnostic tail，永不进入 basis。**k 不由剂量效应选出**——validation NRE 随 k 单调下降，无法产生极值，所以"在 val 上选 k"本身不是一个自洽的程序。

### 2.3 Alpha=0 PCA reference geometry

只在 **α=0 TRAIN questions** 上拟合——不用 val（它检验 k），不用 test（它承载结论数字），不用任何 steered cell（那正是被检验的对象）。四个 phase 各有 **独立的 basis**（9 layers × 4 phases = 36），彼此不共享任何东西。

两个 load-bearing 的设定：

- **Per-question equal weighting**（行按 `1/√n_i` 缩放）。否则一条 20-token 轨迹会以 7:1 压过一条 3-token 的，basis 会变成 *慢样本的* manifold——而长度本身与 α 相关。
- **每个 cell 都用同一个 α=0 `mu` 做中心化。** 若用 steered cell 自己的均值中心化，就等于把待检验的位移本身减掉了。

### 2.4 Token phases and pairing rules

| phase | span | rows/question | role |
|---|---|---|---|
| `prefill` | last prefill token only | 1 | **the only strictly α-matched position**（同 prompt、同 token）——唯一允许做 displacement claim 的 phase |
| `pre_commit` | `[c−20, c)` | ≤20 | event-aligned distribution comparison |
| `post_commit` | `[c, c+20)` | ≤20 | 同上；commit 本身是第一行 |
| `decode_all` | whole decode, per-question row mean | 1 | level sensitivity；唯一能容纳 no-commit 样本的 phase |

- **在 token 20 之前 commit 的样本予以保留，用它实际的短窗口。** 丢弃 `c < 20` 会系统性删除 fast commitment——而这恰恰是 α 所改变的行为（Llama α=0 约 23% 的样本在 token 20 前 commit，且该比例本身依赖 α）。截断窗口，绝不截断样本。
- **No-commit 样本只在 aligned phases 中排除**，仍进入 `decode_all`；coverage 按 cell 报告，且从不作为 gate。
- **Decode phases 只支持 event-aligned distribution comparison。** α 改变了生成文本，各 cell 的 token 不同：那里没有 per-token pairing，也不做 per-state displacement claim。

### 2.5 Geometry metrics and matched nulls

**Normalized reconstruction error (NRE)。**
`NRE(α) = mean(RE_α) / mean(RE_{α=0, held-out})`，按 layer × phase 计——是 **ratio of cohort means**，绝不是 per-question ratio 的均值（后者会在 α=0 residual 极小的题上爆炸）。按 hidden-state norm 归一化只作 sensitivity：那会把待检验的 scalar-gain 效应除掉。

**PCA-subspace alignment。**（原名 "local tangent alignment" 已更名：这是一个 *global per-phase* basis，除非真的实现了 α=0 kNN/local PCA，否则不得使用 `local tangent` 一词。）对 prefill，`d = h(α) − h(0)` 分解为落在 α=0 top-k 子空间内的能量与其余部分；primary 是 **energy-pooled ratio** `Σ‖W_k d‖² / Σ‖d‖²`，而非 per-question ratio 的均值——后者会让一个近零位移与一个大位移获得同等权重。

**Cross-dose scalar fit。** 最小二乘 `d_a ≈ k·d_b`，报告 cos、`k` 与 residual。在最小二乘 `k` 处 `residual ≡ 1 − cos²` 精确成立（已验证到 1.1e-16），因此 **`k` 是三者中唯一独立的数**，必须与 residual 并列报告。pooled cosine 旁附 per-question 同号比例，因为 pooled 值可能掩盖一部分同向、一部分反向的混合。

**Commitment-centroid distance。** 到 α=0 TRAIN `post_commit` centroid 的距离，在 train 上定义、在 test 上评估。

**Matched isotropic null。** 每张 spectrum 旁必须并列：null 必须匹配该 phase 的 `m`、`nq`、`dim` **以及** per-question weighting，并走同一条 Gram 路径。否则无法把 low-rank structure 与 `m ≪ dim` 的采样必然性区分开。位移的随机参照是 `k/dim`——各向同性位移在任意 20-D 子空间中只占 `20/4096 = 0.488%` 的能量。

已弃用的 null：**shuffled-question**，对 pooled PCA 一般是无意义的。有效的几何 negative control 是 matched isotropic spectrum、同 k 的 random orthonormal subspace，以及 trajectory-order shuffle（仅用于 speed/curvature）。

### 2.6 Statistical rules and claim boundaries

- Bootstrap/cluster 单位自始至终是 **question**。
- 剂量对比预先冻结：`−8 vs 0`、`−6 vs 0`、`+6 vs 0`。
- Holm 在 metric family 内校正；family 之间不合并。
- 一个剂量算 improved，必须其 metric pair 的 **两个成员**都朝正确方向移动；"stable" 要求 **三个剂量**方向一致。达不到就报告为 mixed / not detected，绝不报告为 positive。

Frozen claim boundaries：

- PCA 证明的是 **linear low-rank**，不是 nonlinear manifold。
- `k = 20` 是 analysis cap，**绝不是 intrinsic dimension**。
- top-k 子空间 **外部**的能量 **不是** "off-manifold"——k=20 只覆盖 α=0 约 50% 的方差，所以补空间大部分是普通变异。
- Null 倍数只能 **在 phase 内**比较。
- 结果描述的是 **computational geometry of RSN steering**；它们不是生物多巴胺证据，且 **不作任何 causal claim**。

---

## 3. Results

### 3.1 Data integrity, accuracy and coverage

四个 primary cell 在 **full probe**（非抽样）下通过验收：各 n=300，`stored_layer_indices = [10..18, 31]`，band `[11,20)`。

- Projection reproduction 在四个 cell 中均读到 **exactly 0.00e+00**。
- 与 lightweight batch 的 per-question agreement 在四个 cell 的三个字段上均实测 **1.000**——这是本批次的实测属性。
- Accuracy **79.67 / 60.00 / 51.67 / 40.67**（α = −6 / 0 / +6 / −8）**同批次**复现了 −6 峰值。这是 184 bs=1 批次，**永远不可**与 182 dose table 做 per-question 混用。

**Commit coverage**（作为行为结果报告，绝不作 gate）：

| cell | α | coverage |
|---|---|---|
| `nocot` | 0 | 297/300 = .990 |
| `nocot_aneg6` | −6 | 298/300 = .993 |
| `nocot_aneg8` | −8 | 294/300 = .980 |
| `nocot_a6` | +6 | **281/300 = .937** |
| `cot` | 0 | 297/300 = .990 |
| `cot_aneg4` | −4 | 296/300 = .987 |

+6 下降约 5pp，依赖 α，方向与已记录的 +α format degradation 一致。后果是：+6 的任何 commit-aligned 数字都少了 16 题，而且这些题是被 manipulation 自身的结果所筛选的。

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

**Frozen wording: low-rank spectral concentration relative to a matched isotropic null。** 倍数只能在 phase 内比较——prefill 的 3.4× 与 commit 窗口的 ~20× 不可通约，因为 prefill 有 `m = nq = 185`，光是 null 本身就已到 14.8%。谱是一条没有 elbow 的缓慢尾巴，这正是把 k 固定为 20 并报告 sensitivity 的原因。

`decode_all` 必须按 phase 读：它被压缩为每题一行，只在约 185 个点上拟合，所以那里谱平坦是压缩移除了题内变异，**而不是** manifold 不稳定。

### 3.3 Exact last-prefill PCA-subspace analysis

Ambient-space 分解，decoder 18，300 题严格按 `question_idx` 配对，同一 α=0 train basis，`f_k = ‖W_k d‖²/‖d‖²`，primary = energy-pooled ratio。

**Primary (TEST split, k=20)：**

| α | mean‖d‖ | inside [95% CI] | outside |
|---|---|---|---|
| −8 | 24.63 | **21.4%** [20.9, 21.8] | 78.6% |
| −6 | 17.67 | **21.2%** [20.8, 21.7] | 78.8% |
| +6 | 13.23 | **9.8%** [9.2, 10.5] | 90.2% |

**k sensitivity (TEST)：** −8 走 9.6 → 17.0 → 21.4，−6 走 9.5 → 16.8 → 21.2（两者从 k=5 到 k=20 都是 **+11.8pp**，per-dimension profile 几乎相同），而 +6 走 5.8 → 8.1 → 9.8（**+4.0pp**）——增加维度并不能把 +6 的能量捞回来。

**Split agreement at k=20**（train / val / test）：−8 21.8 / 21.1 / 21.4；−6 21.8 / 21.0 / 21.2；+6 10.7 / 9.5 / 9.8。没有对 basis 自身的 train 题过拟合。Pooled 与 per-question 均值差 <0.1pp，所以没有哪道大位移的题在主导结果。

**随机参照，必须并列报告：** 各向同性位移在任意 20-D 子空间中占 `20/4096 = 0.488%` 的能量。所以 +6 的 9.8% 是随机的 **20×**，−6/−8 的 21.2% 是 **43×**——**三个剂量都与 α=0 主结构强对齐**，彼此只差两倍。没有这个参照，9.8% 会被误读成"几乎不对齐"。

幅度随 |α| 单调（24.63 / 17.67 / 13.23），而 inside ratio 不是。**方向结论不能由 inside ratio 推出**——两个位移可以同等程度地填满同一个 top-k 子空间，却指向不同方向。方向由 §3.4 定案。

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

Inside ratio（§3.3）与这里的 cosine 是 **两个彼此吻合的观察，不是同一件事**。要证明它们是一件事，需要检验 +6 的正交分量是否落在 top-k 之外。

### 3.5 Incremental prediction

**Verdict: no stable incremental behavioural predictive value was detected。**

出处说明：round 1（correctness，Z-only baseline）已按其既定目的用掉 TEST split 并耗尽它。Round 2 是在看到 round 1 之后才设计的，因此是 **post-hoc**，不是 confirmatory。用一次 test set 本身不是错误；看完结果后再回头调模型才是。

**Commit position** — baseline 只含生成前特征 `[Z_prefill, prefill confidence]`；commitment behaviour 不可入选，因为它 *就是* outcome：

| α | baseline R² / MAE / ρ | +geometry R² / MAE / ρ |
|---|---|---|
| −8 | .003 / 73.1 / .129 | .019 / 72.3 / .192 |
| −6 | .074 / 74.1 / .276 | .104 / 72.5 / .359 |
| +6 | .004 / 67.8 / .033 | −.004 / 68.3 / −.010 |

两个负剂量在三项指标上都改善；+6 三项都变差。按冻结的"三个剂量必须一致"规则，这是 **mixed**。

**Correctness** — 这里 commitment *可以*入选（作为 predictor，不是 outcome）：AUC .700 / .386 / .547 → .688 / .497 / .443；log-loss .6294 / .4990 / .6917 → .6429 / .4882 / .7247。**结果在指标之间与剂量之间都不一致**——−6 的 AUC 改善（.386 → .497），而 −8 与 +6 下降，所以这不能写成"三个剂量全部变差"。

Round 1 存档（correctness，Z-only baseline）：AUC .485 / .479 / .570 → .503 / .617 / .458，一升两降。

随每个数字同行的 caveats：TEST 已耗尽；n=60，抽样误差大于观测到的差异；措辞是 **"not detected"**，绝不是 "disproved"——baseline 接近随机并不使检验失效，因为几何本可以独立地在其之上带来改善，而它没有。

### 3.6 CoT negative-arm conditional confirmation

H1，在任何 CoT 投影之前冻结：*加入 prefill geometry 会改善负 α 的 commit-position 预测，而对正 α 不会。* 冻结模型原封不动地迁移；CoT 投影到 **已有的 No-CoT α=0 basis** 上（重新拟合会使它变成一个新模型，而不是确认）。

**CoT α=−4, commit position：**

| | R² | MAE | ρ |
|---|---|---|---|
| baseline (Z, conf) | −0.101 | 56.5 | −0.091 |
| +geometry | **−0.056** | **53.9** | **0.121** |

三项都朝预测方向移动，所以 **H1 的负端通过其预设的方向性判据**。

**但绝对预测力很弱**：加入几何后 R² 仍为 *负*，意味着模型仍不如直接用训练均值；ρ 仍然很小。诚实的读法是 **a reproducible weak directional signal, not strong predictive evidence**——它只是从"比均值还差"改善到"没那么差"。

同一 cell 上的 correctness 明显变差：AUC .531 → .462，log-loss .3298 → .4242。

预先冻结的适用范围限制：只有 H1 的 **负端**可检验（CoT 没有正剂量，而正端既不得报告为已确认、也不得悄悄略去）；**α=−4 不在 H1 推导所用的剂量之内**（−8/−6），所以这是向一个未测量剂量的外推；这些数字描述的是 CoT 状态相对于 **No-CoT** natural manifold 的位置。

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

对照预设规则——(a) −6 与 −8 彼此一致，(b) +6 稳定分离，(c) 在 **两个** phase 中都可见——三条都不成立：

- 在 `post_commit` 中，−8 明显高于 α=0 参照（NRE 1.409）而 −6 明显低于（0.836）；两个负剂量并未归为一组。
- +6 落在 1.044 / 1.050，接近 α=0 参照，反而是 −8 成为离群 cell——与 prefill 的图景相反，那里 +6 才是特殊的那个。
- `pre_commit` 的 NRE 在四个剂量上只跨 0.988–1.087，所以结构只出现在一个 phase 中。

此外 `pre_commit` 的 n 严重不平衡（−8: 28 vs −6: 55），因为 α 改变 commit timing，从而改变有 pre-commit 窗口的样本比例。这是叠加在其余问题之上的 selection effect，所以那里连描述性比较也要打折扣。

**Conclusion: last-prefill geometry did not stably extend to commit-aligned decode。** 这并不证伪 manifold；它界定了干净结构存在的范围。

值得说明为何分叉是 **预期之内**而非矛盾：pre-commit coverage 随 α 移动，post-commit 位于 `####` 之后（状态反映答案格式），而 steering 注入在 last-prefill——所以文本在注入之后立刻分叉。另外 NRE 是幅度比，**一个高值和一个低值并不代表方向相反**。

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
- Cross-model 的候选解释：**Llama 的有效位移持续增长并最终 overshoot；Qwen 的可能被压缩或饱和**，这样就能解释 peak 与 plateau 的差别。这正是冻结的 Qwen 分析要检验的。

**α 的正负不等于生物多巴胺的增加或减少。** Manifold 描述的是 RSN steering 的计算几何，不是一种神经递质。

### 4.3 Unsupported claims and limitations

这里的任何内容都不支持：

- 任何 **nonlinear manifold** 主张 —— PCA 只能确立 linear low-rank。
- 把 `k = 20` 当作 **intrinsic dimension** —— 它是 analysis cap。
- 把 top-k 子空间外的能量称为 **"Off-manifold"** —— k=20 只覆盖 α=0 约 50% 的方差，补空间大部分是普通变异。
- 任何形式的 **causal** 主张。这是对已存状态的离线再投影；causal test 需要 random/orthogonal *injection* 并重新采集。
- 跨模型比较 **raw α、PC axes 或 hidden-state values** —— mask、层数（L=9 vs L=6）与激活尺度都不同，所以相等的 α 不是相等的干预。
- **General behavioural predictive value** —— §3.5 与 §3.6 都没找到，而唯一通过的判据所依托的模型 R² 为负。

结构性限制：

- **本批次没有独立的第二个 α=0 cell**，所以 α=0 manifold 的稳定性只能靠 train/val 子采样或 bootstrap 估计，无法用两个独立 α=0 批次交叉验证。这削弱了"manifold 是稳定的"这一前提，并且是一个预先登记的 stop condition。
- **TEST 已耗尽** —— round 1 之后的每个数字都是补充性的。
- Commit-aligned 队列 **由 manipulation 自身的结果筛选**（coverage 与 pre-commit 可用性都随 α 变动）。
- **预测失败不抹掉描述性几何结果**；同样地，描述性几何也不授权任何预测或因果主张。

---

## 5. Status and Next Step

**Llama analysis: COMPLETE。** 定位为 **last-prefill explanatory geometry** —— 一项机制解释性的补充，不是机制主线，也不是预测线。

**不再扩展**（这是决定，不是待办）：TLE、UMAP/t-SNE、全层扫描、追加剂量，以及任何进一步的 correctness-prediction 工作。decode 检验是最后一次扩展，它终止了这条线。

**唯一剩余步骤：冻结的 Qwen last-prefill 分析。** 范围固定在 `PREREG_qwen_prefill.md` —— 仅 last-prefill，Qwen 自己的 α=0 basis、自己的 band `[16,22)`、自己的 commit locator，三个问题（正端是否共享一个方向；位移幅度是否饱和；inside ratio 相对于各自模型自身的子空间），失败条件预先写定。

**manifold 是否能以超出 supplement 的身份进论文，由该结果决定。** 若 Llama 沿一条轴增长直至 overshoot，而 Qwen 的幅度沿一条轴趋平，则几何为 peak-versus-plateau 提供了一个候选解释。若 Qwen 的正端并不共享一个方向，或其幅度在 +12 仍继续增长，或两个模型看起来根本相似，则如实报告，这条线以 Llama-only supplement 收尾。

其余剂量（No-CoT ±2/±4/+8）可用，但 **仅作 continuity checks** —— 同样的题、同样的 basis —— 绝不作为独立验证集。

---

## Appendix. Artifact and provenance index

实现细节、精确命令、guards、tests 与失败出处见 `CLAUDE.md` § *Manifold pilot*。本索引只列出存在什么、在哪里。

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
