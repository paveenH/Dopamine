下面是一份可直接放进 `TODO.md` 的 Manifold Plan。原则是：先在 Llama 上冻结方法，再原样迁移到 Qwen；只使用已有 H5，不重新生成模型输出。

## Manifold Pilot Plan

### 执行状态（2026-08-26）

| 阶段 | 状态 |
|---|---|
| §1 H5 验收 | **已完成** — `check_hs_llama.py --n_probe 0` 全量探针，四个 cell 全过 |
| §2 划分冻结 | **已冻结** — `manifold/split_manifest.json`，185/55/60，digest `64af9b38…` |
| §3 拟合+投影 | **已完成** — 36 个 basis，四个 cell 已导出到本地 |
| §4 起 | 未开始 |

产物位于 `RoleAnswer/llama3/dopamine/manifold/`（`basis.npz`、`basis_meta.json`、
四个 `manifold_*.json`），不进 git。服务器端产物在
`components/llama3/manifold/phase1b_eot/`。

**§3 的求解器实现细节**：BLAS 线程在 `import numpy` 前锁定为 1（服务器
OpenBLAS 在多核下 `eigh` 自旋，实测 957s vs 8.2s，117 倍反向加速）；PCA 用
`scipy.linalg.eigh(subset_by_index=...)` 精确求 top-30（前 20 作 basis，后 10
仅诊断谱尾）；`total_var` 用 Frobenius 恒等式 `‖A‖_F²/nq` 精确补回，因为部分
求解拿不到全谱和，而它是 explained ratio 的分母。等价性已验证：`total_var`
相对误差 1.2e-16，前 20 特征值与全谱一致到 1.8e-15。

---

### 0. 研究问题

区分三个竞争解释：

1. **Scalar gain**：RSN steering 主要改变固定方向上的幅度。
2. **On-manifold retiming**：轨迹仍在自然推理流形内，但速度、停留阶段或 commitment timing 改变。
3. **Off-manifold deviation**：极端剂量使轨迹转向或偏离自然流形，并对应性能下降、循环或格式失效。

Manifold 必须提供超过 `s_t/Z_t + commitment behavior` 的信息，否则只作为辅助可视化。

---

### 1. 数据与条件

#### Llama 主 pilot

- `α=0`：自然基线
- `α=−6`：最佳工作点
- `α=−8`：负端崩溃
- `α=+6`：正端损害代表

第二阶段再加入：

- CoT `α=0/−4`
- 其他已有剂量作完整曲线检查

#### 数据检查

- [ ] 核对每个 H5 的样本数、题号、层索引、α、CoT 与生成文本。
- [ ] 检查 `prefill_hs`、`decode_hs` 和 projection 长度一致。
- [ ] 用现有 RSN 投影代码复算少量样本，确认与已发布 `Z_t/s_t` 一致。
- [ ] 所有 pairing 使用 `question_idx`，不依赖 H5 行顺序。

可复用 [track_hidden_states.py](/Users/paveenhuang/Downloads/Dopamine/track_hidden_states.py)、[extract_signal_json.py](/Users/paveenhuang/Downloads/Dopamine/extract_signal_json.py) 的数据口径。

---

### 2. 冻结数据划分

按 question 固定划分：

- Train：拟合 α=0 PCA basis。**只做拟合，不选 k。**
- Validation：检查 `k=5/10/20` 的稳健性。
- Test：最后才打开，只做最终剂量比较与预测检验。

**k 规则（冻结）**：primary 固定 `k=20`；`k=5/10/20` 作 sensitivity，三者方向
一致才算稳健；top-30 spectrum 仅用于诊断谱尾衰减，不进 basis。
**不得依据剂量效应挑 k** —— val 上的 NRE 对 k 单调下降，不会产生极值点，
"用 val 选 k" 这个说法本身不成立。

实际划分：185/55/60（**不是** 180/60/60）。阈值加在 hash 值上，排序取整会让
每个题的归属重新依赖全集，破坏扩展稳定性。

要求：

- [ ] 同一道题的所有 α condition 必须进入同一个 split。
- [ ] 依据 question hash 固定 split，并保存 manifest。
- [ ] 不根据 accuracy 或预期效应重新划分。

---

### 3. 定义自然流形

每层独立分析，不直接拼接不同层。

只使用 `α=0` training questions。**四个 phase 各自独立拟合一个 PCA basis**
（9 层 × 4 phase = 36 个），互不共享基：

| phase | 范围 | 每题行数 | 角色 |
|---|---|---|---|
| `prefill` | 最后一个 prefill token，仅此一个 | 1 | **唯一严格 α 对齐的位置**（同 prompt、同 token），因此是唯一支持位移 claim 的 phase |
| `pre_commit` | `[c−20, c)`，至多 20 步 | ≤20 | 事件对齐分布比较 |
| `post_commit` | `[c, c+20)`，至多 20 步 | ≤20 | 同上；commit 本身落在首行 |
| `decode_all` | 整个 decode 段，题内均值 | 1 | level sensitivity（option A）；无 commit 样本唯一能进入的 phase |

两条冻结规则：

- **commit 早于 token 20 的样本保留，按其实际短窗口计算**，绝不丢弃。丢弃
  `c<20` 会系统性删掉快速 commit —— 而那正是 α 所改变的行为（Llama α=0 已有
  约 23% 样本在 token 20 前 commit，且该比例本身随 α 变化）。截断窗口，不截断
  样本。
- **无 commit 的样本只排除 aligned phases**，仍进入 `decode_all`；coverage
  逐 cell 报告，不作门槛。

PCA 设置：

- [x] 每题等权（行按 `1/√n_i` 缩放），否则 20 token 的轨迹以 7:1 压过 3 token
      的，basis 会变成"慢样本的流形" —— 而长度与 α 相关。
- [x] 所有 cell 用**同一个** α=0 的 `mu` 中心化；用各自均值会减掉待测的位移。
- [ ] TLE 仅作 intrinsic-dimension sensitivity。
- [ ] UMAP/t-SNE 只展示，不用于统计结论。

**`decode_all` 的谱须按 phase 判读**：它归约到每题 1 行后只在 ~185 个点上拟合，
所以谱平不等于流形不稳，而是归约去掉了题内变异。

---

### 3b. 有限样本随机基线（已执行，结果见下）

随机基线**必须逐 phase 匹配 `m`、`nq`、`dim` 和每题等权**，并走同一条 Gram
路径。否则无法区分"低秩结构"与"`m ≪ dim` 时的采样必然"。

累积解释方差，实测 vs 各向同性高斯（20 draws，中位数 [2.5, 97.5]）：

| phase | m | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| prefill | 185 | .257 vs .039 | .365 vs .076 | **.499 vs .148** [.1477,.1489] |
| pre_commit | 2940 | .243 vs .006 | .344 vs .011 | **.446 vs .022** [.0222,.0223] |
| post_commit | 3576 | .296 vs .005 | .397 vs .010 | **.484 vs .020** [.0200,.0201] |
| decode_all | 185 | .641 vs .039 | .730 vs .076 | .812 vs .148 |

Null 区间极窄（相对宽度 <1%），20 draws 已收敛。

**措辞（冻结）**：这证明的是 **low-rank spectral concentration relative to a
matched isotropic null**。

- **不得**直接称为"低维 manifold" —— PCA 只能证线性低秩，不能证非线性流形。
- **不得**把 `k=20` 当作内在维度；它是分析用的截断。
- **倍数只在 phase 内比较**：`prefill` 的 3.4× 与 commit 窗口的 20× 不可横比，
  因为 `prefill` 的 `m = nq = 185`，随机基线本身就能拿到 14.8%。
- 谱是缓降拖尾、无肘部，所以不存在"天然正确"的 k —— 这正是 k 规则固定为 20
  且报 sensitivity 的原因。

---

### 3c. commit cohort 的选择偏差（冻结）

`pre/post_commit` 的可用样本随 α 变化，实测 commit coverage：

| cell | α | coverage |
|---|---|---|
| `nocot` | 0 | 297/300 = .990 |
| `nocot_aneg6` | −6 | 298/300 = .993 |
| `nocot_aneg8` | −8 | 294/300 = .980 |
| `nocot_a6` | +6 | **281/300 = .937** |

三层口径，缺一不可：

1. **coverage 本身先作为行为结果报告** —— +6 掉约 5pp 且方向符合已知的
   +α 格式退化，这是结果，不是数据问题。
2. **available cohort** 做阶段分布比较。
3. **common-question intersection** 作配对 sensitivity。

2 和 3 **都不能写成严格因果位移**。只有 `prefill` 支持严格同题位移。

---

### 3d. 主层与多重比较（冻结）

- **Primary：decoder 18**（导出 slot `8`）。Sensitivity：decoder 10（slot `0`）。
  其余层进 supplement。
- 导出的 `per_layer` 键是**存储下标 0–8**，对应 decoder 10–18；`31` 是最终层，
  不在 band 内、无 basis。
- 主要剂量 contrasts 提前冻结：`−8 vs 0`、`−6 vs 0`、`+6 vs 0`。
- Bootstrap 单位是 **question**，与全项目一致；multiplicity family 在每个
  指标族内做 Holm，不跨族合并。

---

### 3e. Prefill 位移分解（已执行，精确环境空间）

`manifold_prefill_exact.py`，decoder 18（slot 8），300 题按 `question_idx` 严格
配对，同一个 α=0 train basis，`d = h(α) − h(0)`，`f_k = ‖W_k d‖²/‖d‖²`，
primary 为能量池化比（非逐题比的均值）。

**Primary（test split，k=20）:**

| α | mean‖d‖ | inside [95% CI] | outside |
|---|---|---|---|
| −8 | 24.63 | **21.4%** [20.9, 21.8] | 78.6% |
| −6 | 17.67 | **21.2%** [20.8, 21.7] | 78.8% |
| +6 | 13.23 | **9.8%** [9.2, 10.5] | 90.2% |

**幅度与方向是两个独立的轴。** 幅度随剂量单调（24.63/17.67/13.23），但方向不是:
−8 与 −6 的 inside 几乎相同且 CI 大幅重叠,而 **+6 只有一半,CI 完全不重叠**。
负剂量从 −6 到 −8 是沿同一方向走得更远；+6 是另一个方向,不只是反向或更小。

**k sensitivity（test）** 支持同一读法：−8/−6 从 k=5 到 k=20 都涨 **+11.8pp**
（9.6→17.0→21.4 / 9.5→16.8→21.2，逐维分布几乎重合），+6 只涨 **+4.0pp**
（5.8→8.1→9.8）——给 +6 更多维度也捡不回能量。

**三个 split 一致**（−8: 21.8/21.1/21.4，−6: 21.8/21.0/21.2，+6: 10.7/9.5/9.8），
排除对 basis 训练集的过拟合。pooled 与逐题均值差 <0.1pp，无大位移题支配。

**随机参照必须同报**：各向同性位移落进任意 20 维子空间的期望能量占比是
`20/4096 = 0.488%`。所以 +6 的 9.8% 是随机的 **20 倍**，−6/−8 的 21.2% 是
**43 倍** —— **三个剂量都与 α=0 主结构显著对齐**，只是程度差一倍。缺了这个
参照，9.8% 会被误读为"几乎不对齐"。

**措辞**：energy inside/outside the α=0 top-k PCA subspace。**不写 off-manifold**
—— k=20 只张成 α=0 方差的约 50%，outside 必然含大量普通自然变异。

---

### 4. 几何指标

#### Primary

1. **Normalized reconstruction error（NRE，定义已冻结）**

   Primary：`NRE(α) = mean(RE_α) / mean(RE_{α=0, held-out})`，分母按
   layer × phase 各自计算，**在同一 cohort 上取均值之比**。

   - hidden-state norm 标准化只作 sensitivity —— 它会把待测的 scalar-gain
     效应一并除掉。
   - **不使用逐题 ratio 再平均**：α=0 残差极小的题会让比值爆炸，均值被少数
     样本支配。

2. **PCA-subspace alignment**（原名 "Local tangent alignment"，已更名）

   当前实现是 **phase 级的全局 PCA basis，不是 local tangent space**。除非
   将来真的实现 α=0 的 kNN / local PCA，否则不得写作 local tangent。

   - `prefill` 支持严格同题配对位移分析：`Δh = h(α) − h(0)`，分解为落在
     α=0 前 k 个主方向内的切向能量与其余的法向能量。
   - `pre/post_commit` 已产生不同文本，**只能做事件对齐的分布比较**，不能
     写成逐状态因果位移。
   - 报告切向占比时必须同时给 `k=5/10/20`：k=20 只覆盖 α=0 方差的约 50%，
     所以"法向"里包含大量本属自然变异、只是未进前 20 维的方向。正确措辞是
     **"位移主要不落在 α=0 前 k 个主方向张成的子空间内"**，而不是"离开流形"。

3. **Commitment-centroid distance**
   在 training questions 上定义 α=0 successful/stable commitment centroid，检验 held-out conditions 到该区域的距离。

#### Secondary

- participation ratio / PCA spectrum
- trajectory speed
- curvature / turning angle
- pre-commit → commit → post-commit 的方向变化
- projected path length
- α=0 manifold coverage

所有指标按层、阶段和 condition 报告，不先跨层平均掩盖异质性。

---

### 5. 稳定性检查

- [ ] 按 question 做 split-half/bootstrap。
- [ ] 检查 PCA subspace principal angles 或 projection-matrix similarity。
- [ ] 检查 reconstruction error、tangent alignment 的剂量排序是否跨 bootstrap 稳定。
- [ ] 检查结论是否对 PCA 维数和窗口长度稳健。
- [ ] **几何负控制（shuffled-question 已废弃 —— 对 pooled PCA 通常无意义）**：
      (a) matched isotropic spectrum null（下节）；
      (b) random orthonormal subspace，同 k；
      (c) trajectory-order shuffle，**仅用于 speed/curvature 的 null**。
- [ ] 不重新解释已经完成的 random/orthogonal remask；它是 readout control，不是 causal injection control。

如果 α=0 manifold 本身不稳定，则停止，不进入剂量解释。

---

### 6. 统计与增量解释

所有推断以 question 为 bootstrap/cluster 单位。

建立两组模型：

#### A. Commitment readout

预测：

- commit step
- early-candidate
- post-commit continuation/loop

基线：

- `G_prefill/Z_t`
- `s_t`
- early confidence
- condition fixed effects

加入 manifold features，比较 held-out `ΔR²`、log loss或相关性。

#### B. Correctness/stable completion

基线：

- `s_t/Z_t`
- confidence
- generation length
- commit position
- marker/format状态
- condition fixed effects

再加入 reconstruction error、tangent alignment、speed、curvature、centroid distance。

这里只检验增量预测，不声称 manifold feature 导致正确。

---

### 7. 预设判读

- **流形内、tangent 稳定、只有幅度变化**  
  → 支持 scalar gain。

- **流形内，但 speed/phase occupancy/commitment-centroid distance 改变**  
  → 支持 on-manifold retiming。

- **`−8/+6` 的 normal displacement 或 curvature 增加，并关联失败**  
  → 支持 off-manifold over-steering。

- **`−6` 更接近 successful centroid，但没有明显法向偏移**  
  → 支持最佳点是自然流形内的有效工作区。

- **几何指标不能超过 `s_t + commitment` 基线**  
  → manifold 降为可视化补充，不扩成论文主线。

---

### 8. 跨模型阶段

只有 Llama pipeline 完全冻结后才运行 Qwen：

- [ ] **冻结分析算法与选择规则**，而非机械复制 Llama 的 token 级定义。
      Qwen 必须使用它自己的 commit locator（`####`-only）、自己的 layer band
      `[16,22)` 和模型内标准化 —— 照搬 Llama 的 locator 会悄悄重定义事件本身。
- [ ] Qwen 使用自己的 α=0 manifold 和模型内标准化。
- [ ] 比较几何机制，不比较 raw α 或原始坐标值。
- [ ] 检验 Qwen 高剂量平台属于 scalar compression、on-manifold retiming，还是不同轨迹机制。
- [ ] 最后才讨论 Llama `−6` 与 Qwen `+8` 是否靠近相似的功能工作区。

### 9. 停止条件

满足任一条件即不继续扩大：

- α=0 manifold 在 held-out/bootstraps 中不稳定。
- 结果严重依赖 PCA 维数或窗口定义。
- 几何指标不能提供超过一维信号与 commitment 指标的增量解释。
- Llama 没有可重复的主效应，则不开展复杂跨模型空间对齐。

最终执行顺序：

> **H5验收 → Llama α=0 manifold → 四条件 held-out pilot → 增量预测 → 冻结方法 → Qwen复用 → 决定是否需要 causal direction injection。**