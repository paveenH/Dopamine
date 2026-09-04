#### Tmux
conda activate dopamine
conda activate roleplaying
conda deactivate

tmux new -s da
tmux attach -t da

cd /data1/paveen/Dopamine

git pull origin main
watch -n 1 nvidia-smi
export CUDA_VISIBLE_DEVICES=3

top -u $USER

##### 清理缓存
rm -rf /home/nas/d12922004/.cache/huggingface/hub
rm -rf /home/nas/d12922004/.hf_cache/huggingface/hub

##### 182/184/185/177/178
rsync -avzP d12922004@140.112.31.182:/data1/paveen/Dopamine/components/llama3 /Users/paveenhuang/Downloads

rsync -avzP d12922004@140.112.31.182:/data1/paveen/Dopamine/components/benchmark/cruxeval_p4c_formal.json /Users/paveenhuang/Downloads

rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

---
Daily
09.04 Lab聚餐、整理衣橱、选课
09.07 生理期推迟
09.11 seminar
09.19 台北-杭州萧山 机票 ✔
09.20 杭州逛逛
09.21 回家 高铁*1 - Helene ⏸
09.22、09.23、09.24 在家 需要去办理公证 + 爸妈护照
09.24 全曜回家 高铁*1  ⏸
09.25 - 10.02 武夷山-成都-丽江 <川滇之间>
10.02 丽江
10.02晚上-10.03 成都市区 机票 ✔ 住宿 ⏸
10.04 成都-武夷山Flight Home
10.04-10.10 Home
10.11 Flight Taipei

办理公证需要的下料
成都的住宿（10.03）
购买徒步需要携带的东西
准备多益考试
看完瑜伽视频
看完徐玉兰视频
---
Dopamine.Nature2026.[Endocannabinoids facilitate reward engagement through retrograde gain control.](https://doi.org/10.1038/s41586-026-10967-w) 该研究发现，伏隔核 D2R–Penk 神经元通过释放内源性大麻素 2-AG，逆向抑制 aPVT→NAc 的谷氨酸输入，从而以通路特异的增益控制维持奖励追求中的行为投入。该机制与 RSN 调节 engagement/commitment gain 的功能解释高度相关，也位于接受多巴胺调节的伏隔核奖赏回路中；但论文直接验证的是 `2-AG→CB1R` 通路，而非 dopamine，因此适合作为 neuromodulatory engagement gain control 的生物学参照，而不能作为 RSN≈dopamine 的直接证据。

---
Agent1: task coding (claude)
Agent2: Task design & check (GPT)
Agent3: claude.md mataining (claude)
Agent4: Document (GPT)
Agent5: Total Design (GPT)
---
组会内容（08.31）：
1）弄清楚neurons的差异：confident & unconfident // thinking & answer directly (先推理再提交、先提交再推理)
2）Manifold -> 不同方法找到的neurons之间的差异

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
14. cross-model thinking curve + 再次整理thinking.md -> commitment regime ✔
15. GSM-Hard -> 1）最佳工作点可以复制；2）可以通过回答情况看来预测是否最优 ✔
15. GSM-Hard COT + alpha Vs. COT ✔
--- 
16. MATH补充完整 ✔ 
17. LogiQA working point ✖ 目前做不出来，不确定是因为选择题的形式问题还是逻辑推理无法迁移
18. BBH counting task ✔
19. 精简claude.md的内容 ✔
20. 整理GSM8K文档 ✔
21. 补充一下llama3-gsm8k-cot working point的结果 ✔
22. 思考关于working point，补充一下其余的点 (GSM 185/ Llama GSM8K 182/ LLama Math 182) ✔ 
23. CRUXEval Design ✔ 
24. 优化文档GSM8K ✔
25. 补充BBH CRUX LogiQA cot的结果 -> cot会有效果 ✔
25. 顺便丰富一下prediction的结果 ✔
26. ZebraLogic WP测试 ⏸
---
16. Ada-GSM8K部分需要一个同一的指标 （reason-first）
15. commitment regime 作为预测标的（直接预测调整的方向）

---

| 优先级 | 任务 | 最能回答什么 | Llama3 表现 / 风险预期 |
|---|---|---|---|
| 4 | **ZebraLogic** | 非选择题逻辑约束求解的更强测试 | **exact solve 偏低**；格式与能力地板风险高，cellwise 可较可读 |
| 5 | **ProofWriter / RuleTaker** | 移除选项内容、保留离散答案空间的机制控制 | **中高**；规则推导相对适合 8B，但仍是标签提交 |
| 6 | **FinQA** | 开放数值答案的跨数据域验证 | **中等**；表格读取与程序式计算是额外难点 |
| 7 | **TruthfulQA-Gen** | 去除选项后的自由生成行为是否改变 | **不宜写单一 accuracy**；可生成，但要面对 judge 评分与 Llama 尾部 loop |
| 8 | **FOLIO** | 自然语言逻辑 | **中低到中等**；三值判断、语义歧义，小样本 |
| 9 | **GSM-Symbolic** | GSM8K 接口鲁棒性 | **中高**；最接近 GSM8K，因此独立性弱 |
| 10 | **LiveCodeBench output prediction** | 程序输出预测 | **偏低到中等**；题目版本、代码能力与执行环境成本较高 |

---

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
### 从 AdaptiveThinking.md §5.8 Open items 迁出（2026-08-28）

- [ ] **§5.6 逐层口径扩展到 −8 与 CoT**（原 Open 3，范围已收窄）
      AdaManifold.md §3.9 的 cross-layer 已在「位移幅度 + 方向余弦」尺度覆盖全部 15 个
      layer slot（含 Qwen −8 与 CoT），但 §5.6 的 **scalar-compression residual /
      null specificity** 是另一套口径，仍只跑过正臂高剂量。HS 已采集，属分析工作。

- [ ] **因果方向控制：真正注入 random / orthogonal directions**（原 Open 4 → 即 P2）
      §5.6.3 的 null 是 remask（对同一批 hidden states 重投影），只能支持
      readout specificity。要主张 steering direction 有因果特异性，必须实际注入并
      重新采集。与 P2 第 53–55 行、ACL ARR 清单第 3 项是同一件事，勿重复列。

- [ ] **Llama 五格 H5 provenance 验收**（非阻塞）
      九点剂量曲线依赖的 nocot_aneg4/aneg2/a2/a4/a8 从未过 check_hs_llama.py。
      只读，全量 probe 十几到几十分钟。不做则 supplement 措辞须写
      「四格经全量验收；其余五格未单独验收，但 stored accuracy 与离线重算完全一致」。

- [ ] **commit-aligned s_t/Z_t 与 post-commit release 的跨模型对称比较**（supplementary，
      结构性受限，非工作量缺口）
      Qwen 可读队列由操纵结果选出（α=0 覆盖率 4.0%，n=12），加样本也造不出匹配参照。

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


---

> **RSN/Thinking Curve 能否成为可预测、可迁移、可用于选择推理状态的指标。**

这会让工作从“有趣的机制现象”提升成“有实际用途的 reasoning calibration framework”。

## 一、预测模型对错：值得做，但要重新定义目标

现有结果其实还不能说 Thinking Curve 可以预测单题对错：

- correct 组通常有更高、更持续的 pre-commit `s_t`；
- 但 `G_prefill`、`p_t`、entropy/top1/margin 都没有稳定区分 correct/incorrect；
- Manifold feature 也没有提供稳定的增量预测价值。

这说明单独拿一个 `s_t` 或 PCA 指标做分类可能不够。[AdaptiveThinking.md](/Users/paveenhuang/Downloads/Dopamine/AdaptiveThinking.md:276) [AdaManifold.md](/Users/paveenhuang/Downloads/Dopamine/AdaManifold.md:231)

但我仍然认为值得继续，因为之前检验的是“单个信号是否直接预测 correctness”，还没有真正检验：

> **完整的 reasoning-state panel 是否能在答案提交前，增量预测这道题最终会不会做对。**

建议定义两个预测时间点：

1. **Early prediction**：生成前 20/50 tokens 后预测最终正确性。
2. **Pre-commit prediction**：模型即将首次提交答案前预测正确性。

特征可以包括：

- `s_t` level、变化量和稳定度；
- `p_t` amplitude；
- entropy、top1、margin；
- early-candidate 是否出现；
- 当前 reasoning length；
- commitment proximity；
- answer switching / instability proxy。

关键基线必须包括：

- 只用题目难度；
- 只用 entropy/top1/margin；
- 只用生成长度；
- 上述基线 + RSN/Thinking Curve features。

真正有价值的结果不是单纯 AUROC 高，而是：

> **加入 RSN dynamics 后，在 held-out questions 上比普通 confidence、difficulty 和 length 基线预测得更好。**

更强的验证是：

> 在 GSM8K 训练预测器，冻结后直接测试 MATH。

如果能跨任务保持预测力，Thinking Curve 才真正具有“reasoning monitor”的价值。

## 二、统一工作点：应该改成“统一功能工作区间”

我不建议假设 GSM8K 和 MATH 存在完全相同的 raw α 最优点，因为现有数据已经显示：

- Llama GSM8K：`−6` 是尖锐峰值；
- Llama MATH：目前只充分支持 `−4 > 0 > +4`，还没有完整覆盖 `−6`；
- Qwen GSM8K：`+8～+12` 平台；
- Qwen MATH：`+6` 左右最好，`+8` 在困难题上回落。

所以“统一 α”大概率不成立。但可能存在：

> **跨任务共享的 functional working-state region。**

例如这个区间可能表现为：

- 不在开头立刻给候选答案；
- 保留足够的 pre-commit computation；
- output distribution 已经足够明确；
- 但没有进入过度延迟、重复、改答案或计算压缩；
- commit 后能够正常 release 和停止。

也就是说，统一的不是剂量，而是：

> **commitment readiness 与 remaining computation 之间的平衡状态。**

## 三、最有价值的实验：用 GSM8K 的状态目标预测 MATH 的最佳剂量

我建议下一步直接做一个“跨任务工作点迁移”实验。

### 阶段 A：先用现有输出做零成本 pilot

在 GSM8K 上定义一个不使用 accuracy 的 functional-state score，例如组合：

- normalized commitment position；
- pre-commit reasoning span；
- early-candidate rate；
- answer switching；
- loop/stopping；
- output decisiveness。

然后冻结这个定义，直接应用到现有 MATH 各 α cell：

- 检查 GSM8K 的优良状态区间，在 MATH 上是否也对应较高 accuracy；
- 检查过早 commit 和过度 processing 是否在两个任务上都对应失败区。

这一步主要用现有文本结果，可以先判断假说有没有希望。

### 阶段 B：再补 MATH hidden states

如果文本层 pilot 支持，就只收集关键剂量的 MATH hidden states，不做完整大 sweep：

- Llama：`−8 / −6 / −4 / 0 / +4`
- Qwen：`0 / +4 / +6 / +8 / +12`

然后完成真正的迁移检验：

1. 在 GSM8K 上冻结 state representation 和目标区间；
2. 不看 MATH accuracy，只根据 MATH calibration subset 的内部状态选择最接近目标区间的 α；
3. 在独立 MATH test subset 上揭示 accuracy；
4. 比较 state-selected α、固定 α=0，以及直接照搬 GSM8K 最佳 raw α。

如果 state-selected α 能迁移，而 raw α 不能，这会是非常强的结果：

> **RSN 的价值不是提供一个通用剂量，而是提供一个可以跨任务识别和校准的 reasoning working state。**

## 最终可以形成的新故事

现在的故事是：

> RSN steering 改变 commitment dynamics，并产生模型与任务依赖的 performance curve。

下一阶段可以升级成：

> **Thinking Curve 提供一个在线 reasoning-state readout；该状态能够预测推理成功与失败，并允许我们不依赖目标任务标签、通过功能状态匹配将 GSM8K 上发现的工作区间迁移到 MATH。**

我会把优先级排成：

1. 现有 GSM8K/MATH 输出上的跨任务 functional-state pilot；
2. GSM8K 内部的 held-out correctness prediction；
3. GSM8K→MATH frozen predictor transfer；
4. 只有前三项出现稳定信号，才补 MATH hidden-state cells。

这条路线比继续证明干预有效更有“价值”，而且能够明确回答两个更大的问题：**能否提前知道模型会不会做错，以及能否把一个任务上的最佳推理状态迁移到另一个任务。**


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
