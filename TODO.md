## Daily
办理公证需要的下料

## TO DO
0. 测试一下qwen中间一点的mask ✖
1. Qwen25-7B ICG实验结果整理 ✔ 
2. 整理行为学的结果 ✔
3. Qwen GSM8k实验以及结果分析 ✔
4. Qwen MATH ✔
5. Qwen High-Dose in GSM8K ✔ 
6. MATH cot ✔ 
7. 复现qwen的thinking curve signal部分，没有存HS ✔
8. qwen的thinking curve 存HS ✔
9. 重新梳理一下AdaptiveThinking的文档 ✔
9. qwen的具体分析 ✔
10. Qwen 的 output decisiveness: 从现有 7 个 H5 cell 提取 entropy/log(V)、top1、margin ✔
11. manifold llama3 实验以及结果整理 ✔
11. manifold 补齐 Llama 全 α 曲线 ✔
12. manifold Qwen25 实验以及结果整理 ✔
13. manifold sentiity ✔
14. 对比
---
## 重新编列后的 TODO

### P0. 收尾 Manifold

- [ ] 完成 inside-ratio sensitivity：比较 `k=5/10/20`、不同层及相对随机基线 enrichment。
- [ ] 完成 Llama–Qwen 几何总结图。
- [ ] 冻结结论：两个模型的 entry geometry 都平滑、近似 piecewise scalar，但不能解释 peak 与 plateau。
- [ ] 关闭 manifold 扩展：不再做 prediction、UMAP/TLE 或复杂 decode manifold。

### P1. 完成跨模型 Thinking Curve（最高优先级）

比较功能状态，而非 raw α：

- Llama：`0 → 最佳 −6 → overshoot −8`
- Qwen：`0 → 改善 +6 → plateau +8/+12`

统一分析：

- [ ] entry gain
- [ ] early-answer / candidate timing
- [ ] commit position
- [ ] commit-aligned `s_t/Z_t`
- [ ] confidence、entropy、margin
- [ ] reasoning length、loop
- [ ] post-commit release

目标结论：

> 两个模型的 entry gain 都近似线性，但下游 commitment/decode 转换函数不同，因此产生 Llama peak 与 Qwen plateau。

### P2. 补 causal direction control

- [ ] 构造与 RSN 匹配 norm、sparsity 和注入层的 random directions。
- [ ] 构造 orthogonal-to-RSN directions。
- [ ] 实际注入模型，比较 accuracy、commitment 与 Thinking Curve。

这一步回答的是“效果是否来自 RSN 方向本身”；现有 remask 只能支持 readout specificity。

### P3. 整理 Behaviour evidence

- [ ] Betting：作为稳定正向证据。
- [ ] CGT/IGT：保留有效结果与 construct boundary。
- [ ] Bandit：作为 recognition–action dissociation 的边界证据。
- [ ] 不重跑 Qwen 高剂量 CGT/IGT。
- [ ] 如需跨模型行为确认，只选择一个预先确定、已有稳定效应的任务。

目标是整理证据层级，不再增加大量行为任务。

### P4. 增加一个外部 reasoning task

- [ ] 首选 GSM-Hard。
- [ ] Llama 检验负向 working region 是否迁移。
- [ ] Qwen 检验 `+6～+8` commitment 转折是否迁移。
- [ ] 固定少量、事先确定的剂量，不重新搜索最佳 α。
- [ ] 只有 GSM-Hard 给出明确价值时，才考虑 SVAMP/ASDiv。

措辞使用“迁移性检验”，不预设两个模型必须一致。

### P5. 同步推进论文

主线固定为：

`RSN discovery → GSM8K calibration → behavioural generality/boundary → Thinking Curve mechanism → cross-model difference`

同步完成：

- [ ] 论文 section skeleton
- [ ] 跨模型 Thinking Curve 主图
- [ ] causal control 图
- [ ] behaviour evidence 汇总表
- [ ] limitations 与 evidence boundary
- [ ] reproducibility / protocol appendix

总体优先顺序：

> **Manifold 收尾 → Thinking Curve → causal control → Behaviour 整理 → GSM-Hard → 论文整合**

其中论文结构和主图不必等实验全部结束，可以与 Thinking Curve 同步推进。

---
1. **收尾并关闭 Manifold**
   - 完成 `k=5/10/20` inside-ratio sensitivity。
   - 完成 Llama–Qwen 几何总结图。
   - 冻结结论：entry geometry 平滑且近似 piecewise scalar，但不能单独解释 Llama peak 与 Qwen plateau。
   - 不再扩展 prediction、UMAP/TLE 或复杂 decode manifold。

2. **完成跨模型 Thinking Curve 与 boundary-state comparison**
   - 不比较 raw `α`，全部转换为模型内部标准化指标。
   - 比较：
     - Llama：`0 → −6 → −8`
     - Qwen：`0 → +6 → +8/+12`
   - 统一分析 `Z_prefill`、`decode[0]` 回弹、early candidate、commit position、`s_t/Z_t`、confidence、entropy、margin、长度、loop 和 post-commit release。
   - 判断两模型是到达相似 working state，还是仅表现出相似功能但内部状态不同。

3. **做 blind held-out working-state matching**
   - 用训练题定义 Llama `−6` 的目标状态。
   - 不查看 Qwen accuracy，仅根据内部与输出指标选择最接近的 Qwen 剂量。
   - 在 held-out questions 检验 accuracy、commitment 和 stopping。
   - 若无法对齐，就接受“共同 commitment-calibration 功能、不同模型实现”的结论。
   - CCA/Procrustes 只在这一步出现明确对齐信号后再考虑。

4. **增加一个外部 reasoning benchmark**
   - 首选 GSM-Hard。
   - 固定少量预先确定的剂量，不重新搜索最优 α。
   - Llama 检验负向 working region 是否迁移；Qwen 棧验 `+6～+8` commitment 转折是否迁移。
   - 重点报告 accuracy、early candidate、commit timing、长度、loop 和格式稳定性。
   - 目标是证明 reasoning-task transfer 和实际应用潜力；只有结果明确时才考虑 SVAMP/ASDiv。

5. **补 causal direction control**
   - 构造 norm、sparsity 和注入层匹配的 random directions。
   - 构造 orthogonal-to-RSN directions。
   - 实际注入代表性条件，比较 accuracy、commitment 与 Thinking Curve。
   - 回答效果是否具有 RSN 方向特异性，而不只是任意 hidden-state 扰动。

6. **整理 Behaviour evidence**
   - Betting：稳定正向证据。
   - CGT/IGT：重新按 `evidence → recognition → utilization → commitment → outcome` 梳理。
   - Bandit：作为 recognition–action dissociation 和能力边界证据。
   - 不重跑 Qwen 高剂量 CGT/IGT，也不继续增加大量行为任务。
   - 如需跨模型确认，只选择一个已有稳定效应的行为任务。

7. **同步推进论文与主图**
   - 主线固定为：  
     `RSN discovery → GSM8K calibration → reasoning transfer → behavioural generality/boundary → Thinking Curve mechanism → cross-model difference`
   - 同步完成 Thinking Curve 主图、GSM-Hard 迁移图、causal control 图、Behaviour 汇总表及 limitations。
   - 不必等所有实验结束才开始写作。

8. **暂缓扩张**
   - 暂不加入第三个模型。
   - 暂不大规模增加 reasoning benchmarks。
   - 暂不重开复杂 manifold。
   - 第三个模型仅作为论文初稿完成后的审稿风险储备。

总体顺序：

> **Manifold 收尾 → Thinking Curve/状态对齐 → held-out matching → GSM-Hard → causal control → Behaviour 整理 → 论文定稿**
---

## TO DO — ACL ARR

### 2. Cross-Model Positive-Result Replication

**目标：** 验证至少一个稳定的 RSN 行为效应不局限于 Llama3-8B，而不是复制全部 benchmark。

- [ ] 在第 1 步完成后选择一个最稳定的正结果任务。
- [ ] 重新验证 Qwen 的 tokenizer、anchor、candidate IDs、mask 与 steering fires。
- [ ] 使用模型自身定位的 direction 与校准后的 α / `ΔG_prefill`，不直接搬用 Llama 的 mask 或 raw dose。
- [ ] 复现该任务的主指标、方向性与必要的 validity checks。
- [ ] 若无法复现，将跨模型主张明确限定为当前模型范围；不以 Bandit 负结果作为跨模型 gate。

### 3. Direct Causal-Specificity Control

**目标：** 区分 RSN direction 的效应与一般 hidden-state perturbation。

- [ ] 整理 Thinking Curve 已有 random / orthogonal controls 及其可支持的结论。
- [ ] 选择与第 2 步相同或同等稳定的代表性行为任务。
- [ ] 构造在范数、层、support 与 α 上匹配的 random / orthogonal direction control。
- [ ] 比较行为主指标并冻结 direction-specificity 的判定口径。

### 4. Paper Positioning and Preparation（与实验并行）

- [ ] 将主贡献定位为 `role-conditioned behavioral gain control`。
- [ ] 将 dopamine 定位为 `selective dopamine-like functional analogy`，明确不声称 RSN 实现完整 dopamine / RPE 系统。
- [x] 将 Bandit 定位为 directed exploration 的边界证据，不用于估计第三个 working point。
- [ ] 整理社会角色—dopamine、wanting 与 decision policy 文献。
- [ ] 制作统一主结果图与机制示意图。
- [ ] 汇总模型、seed、prompt、steering 与统计规格。
- [ ] 完成 Limitations、Responsible NLP 与可复现性清单。
- [ ] 撰写 ACL ARR 长文初稿。

# Brain

我觉得有价值，但建议拆成“现在做”与“后续做”。

1. **现在做：Steingroever 健康常模对齐**

这条成本低，而且公开数据确实包含 617 名参与者的逐 trial 选择、收益与损失。不过数据混合了 95/100/150 trials 和三种 payoff scheme，因此必须只选与我们 IGT 协议完全匹配的子集。[Steingroever et al. 数据说明](https://openpsychologydata.metajnl.com/articles/jopd.ak)

建议检验：

- `net_block1–5` 学习曲线；
- net score 分布；
- deck preference；
- Wasserstein distance / RMSE；
- 按原始 study 做 held-out，而不是把所有参与者混在一起挑最佳 α。

它能支持的结论是：

> 某个 RSN 条件产生的 IGT 行为最接近健康人常模。

但不能写成：

> α=0 等于正常 dopamine 水平。

行为相似不能识别神经递质水平。因此建议称为 **human behavioural calibration**，而不是 dopamine-axis calibration。

2. **已有药理学方向对照：保留即可**

现在的定性文献结论已经够用了：

- α+ 与较早承诺、较高下注或 reward seeking 的方向相似；
- α− 与较保守行为的方向相似。

把它作为 Discussion 中的 correspondence table，不必继续寻找无法获得的三组相关系数。措辞停在：

> RSN manipulation exhibits behavioural correspondence with reported dopaminergic pharmacology effects.

不要上升为神经机制同源。

3. **fMRI RSA：有潜力，但暂不作为当前投稿必做项**

NARPS 确实有 108 名参与者、四个 mixed-gamble runs、trial timing、BIDS 数据和公开预处理结果，技术上适合开展共享刺激分析。[NARPS 数据说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602933/)

但原计划需要两处修正：

- 必须把 NARPS 中完全相同的 gain/loss gamble 输入 LLM，才能构建可比较的 RDM；现有 MCQ/Betting 刺激不能直接与 fMRI trial 做 RSA。
- 即使 vmPFC/striatum RSA 显著，也只能说明 **representational correspondence**，不能“直接证明 RSN 操纵 reward representation”。
- Broca/Wernicke 不显著也不能证明“不是语言表征”；ROI 功效、刺激语言量和噪声都会造成 null。
- 还需要控制 gain、loss、accept/reject、RT 等 nuisance RDM，否则相关性可能只是两个系统都编码金额。

因此这会变成一个独立而完整的 brain–LLM 项目，不太可能只是附加分析。

我的建议优先级是：

1. 完成 Thinking Curve、GSM-Hard 和 causal direction control。
2. 并行做 Steingroever 健康常模的小型定量对齐。
3. 将药理学方向一致性放进 Discussion。
4. NARPS RSA 放入 future work；如果前面的行为对齐很漂亮，再考虑扩成后续论文。

所以这条线不应该删除。**最值得现在做的是健康人 IGT 行为对齐；fMRI RSA 很有价值，但不应成为当前 ACL ARR 的阻塞项。**