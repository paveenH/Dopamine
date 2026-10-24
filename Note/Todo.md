### Document writing
要求：1）细节部分放到Claude.md不要在文档中出现 2）数据尽量保持完整 3）尽量合并表格（但是不要勉强，可以合并的合并） 4）内容和章节都可以重构或者合并 5）结论简洁 通俗易懂 目标是提升可读性 6）标题和表格写英文
给我md版本的文字就好 我会自己去替换
尽量不要删除原始数据，可以合并

---

### Note
Dopamine.Nature2026.[Endocannabinoids facilitate reward engagement through retrograde gain control.](https://doi.org/10.1038/s41586-026-10967-w) 该研究发现，伏隔核 D2R–Penk 神经元通过释放内源性大麻素 2-AG，逆向抑制 aPVT→NAc 的谷氨酸输入，从而以通路特异的增益控制维持奖励追求中的行为投入。该机制与 RSN 调节 engagement/commitment gain 的功能解释高度相关，也位于接受多巴胺调节的伏隔核奖赏回路中；但论文直接验证的是 `2-AG→CB1R` 通路，而非 dopamine，因此适合作为 neuromodulatory engagement gain control 的生物学参照，而不能作为 RSN≈dopamine 的直接证据。

Bandit.NatureCommunications2026.[Foraging models explain human exploration in uncertain tasks.](https://doi.org/10.1038/s41467-026-75773-4) 该研究发现，人类在动态 Bandit 中更接近 compare-to-threshold 策略：主要追踪当前选项是否仍值得继续，而非持续比较所有候选价值。这与 PV10 中模型反复采样 incumbent、却不响应低样本替代臂的行为相似，为 incumbent persistence 提供了“局部阈值决策”的替代解释；但我们尚未进行相应的模型拟合，且任务设定不同，因此只能视为行为结构上的参照，不能断言 LLM 使用了相同的 foraging-RL 机制。

---

### Daily

10.21 周三 看学弟的實驗結果；看科技新聞；重新构思Paper的思路
10.22 周四 继续收尾看有什么需要补充的实验；办理居留证
10.24 周六 补充模型

准备多益考试
看完瑜伽视频
看完徐玉兰视频

---
### TO DO

1. 重新看看willingness实验，统一模板，优化负向表达 -> v2，出现相反的结果 ✖
2. 优化willingness的表述，再做一版本的结果 -> v3 还是很难解释 ✖
3. 从MMLUE开始完善，弄清E选项降低的原因 -> 看起来结果还是很乱 ✖
4. 重新整理思路 ✔
5. 统一posN Qwen和llama的reasoning行为学分析指标 #### 的定义 ✔
6. infoquestion: RSN 改变的是模型“直接作答的倾向”，还是“根据信息是否足够，决定先问清楚再回答”的策略 ✔
6. 重新梳理当前的思路和之前想到的理论之间的关系 ✔
7. 完成 Norm-Matched Orthogonal Injection Controls ✔
8. GSM8K Base ✔
9. llama3-base 1-shot gsm8k -> 1-shot会破坏commit的自然规律 ✔
10. Qwen Base 的 early-candidate 小盲审 ✔
11. Off-Support Orthogonal Control ✔
12. Off-Support Orthogonal MMLUE ✔
13. Qwen2.5 expert vs non-expert ✔
14. Chat-CoT Saturation Check (GSM8K / GSM-Hard) ✔
---
15. ProofWriter-OWA 补充实验 Chat zero-shot No-CoT ✔ -> Llama 未检出工作点收益；Qwen 严格分数上升主要伴随格式恢复，宽松标签正确率未见提升。
16. 复核 ProofWriter-OWA Qwen Chat 1-shot CoT `+8` ✔ -> 冻结严格收益 +6.67 pp（Holm p=.03027）；宽松标签 41.00%→48.00%，+7.00 pp（配对 p=.00985，独立 Holm m=3 p=.02956），不由严格 marker 格式恢复解释；主结果不变，见 ReasoningBare §6.5。
17. ProofWriter-OWA Chat zero-shot CoT ✔ -> Qwen +8 严格 +18.00 pp（Holm p=1.28×10⁻⁹），但宽松标签口径仅 +1.67 pp（p=.487）或 +3.67 pp（p=.0614）；260/300 在解释前给标签。Llama −6 未检出提升。
18. FinQA Chat 0-shot CoT ✔ -> Llama −6 未检出提升；Qwen +8 严格准确率下降 12.33 pp（Holm p=4.74×10⁻⁷），宽松数值评分无净变化，主要受 marker 格式影响。
19. 一些统计分析 ✔
20. Inducing answer-first ordering -> pushy chat GSM8K ✔ -> pushy 未诱发更多 early-candidate
21. Mistral实验补充 ✔ -> Mistral在Bare常无法回答GSM8K 在Chat上有趋势但是没有commitment change
22. Openchat/ Qwen3 GSM8K 实验补充 ✔
23. 把Qwen3 测完整 ⏸
24. New model: OLMo-3-7B-Instruct ✖
25. New Model

---

1. **OpenChat-3.5**：Mistral 架构，可以补上主线缺的第三个家族；数学最强；② 类  ✖ GSM8K提交不了
2. **Qwen3-8B**：技术上最稳（① 类，Bare 下明确先作答）✔ 可用
3. **StableLM-2-12B-chat**：全新家族；② 类；推理能力未知
4. **Nous-Hermes-2-Mistral-7B-DPO**：① 类
5. **OpenHermes-2.5-Mistral-7B**：① 类
6. **Hermes-3-Llama-3.1-8B**：② 类，Llama 家族
7. **Zephyr-7B-beta**：② 类 
8. **Qwen3-14B**：规模对照
9. **Qwen2.5-3B-Instruct**：小规模，role gain 弱
10. **Qwen3-32B**
11. OLMo-2-1124-7B-Instruct（AllenAI）✖ MMLUE都选E 偏置太多
12. Llama-3.1-Tulu-3-8B 和 OLMo-2 是同一套公开配方，但基座是 Llama-3.1。和我们的 Llama-3.1-8B-Instruct 放在一起，是"同基座、不同 post-training"的对照。不增加家族多样性，只在要讨论训练阶段时才有价值。
13. Gemma-2-9B-it 或 Gemma-3-12B-it Google 家族，推理能力强。风险：ModelAnalysis 里 Gemma-7B（第一代）属于 ④ 类，没有 role effect。新一代未必一样，但有失败的先例。
14. Ministral-8B-Instruct 或 Mistral-Nemo-12B-Instruct Mistral 的新版本，推理能力比 7B-v0.3 强。但仍是 Mistral 家族，而 Mistral 已经放在附录。
15. granite-3.3-8b-instruct（2025-04）
16. GLM-4-9B-chat
17. InternLM2.5-7B-chat

---


## 必需

| # | 实验 | 格数 | 目的 |
|---|---|---|---|
| 1 | GSM8K Bare，+8 和 +10，768 token | 2 | 补全正向臂，确定是峰值还是平台；确定工作点 |
| 2 | GSM8K Bare，α=0 和工作点，2048 token | 2 | 检查预算：截断率下降后，收益是否还在 |
| 3 | GSM-Hard Bare，α=0 和工作点 | 2 | 固定工作点迁移。Llama 和 Qwen2.5 在这个任务上都迁移成功，是最直接的对照 |
| 4 | MATH Bare，α=0 和工作点，2048 token，bs=8 | 2 | 第二个迁移任务。Qwen2.5 的 MATH 峰值比 GSM8K 低一档，看 Qwen3 是否也这样 |

**做完这些，Qwen3 就能覆盖主线的三条主张：**
- C2：剂量响应的方向和形状。
- C3：迁移。
- C4：工作点能否跨任务沿用。

---


3．随后优先解决 RSN 与 CSN 的区别。

这里需要修正你之前的一点理解：最新 [Neurons.md](/Users/paveenhuang/Downloads/Dopamine/Note/Neurons.md:162) 显示，**CSN 也会改变候选输出和提交行为**；小剂量有准确率改善趋势，但尚未通过多重比较校正。

因此，下一步应先做**逐层范数匹配、有限剂量范围的功能比较**，再决定是否分解共享方向和剩余方向。现在直接把二者的差异命名为“confidence 与 commitment 的区别”，证据还不够。

对比激活均值来改变行为已有 [CAA 等先例](https://aclanthology.org/2024.acl-long.828/)。我认为我们接下来的贡献应集中在：**用有区分力的预测，说明 RSN 的功能及边界。** 核验工具、新行为任务和更细的 MLP／attention 追踪，可以等这一步有结论后再决定。

---

2. 正交对照补一组 off-support（ortho_gauss_off）
每个模型补 3 个 cell，生成方式和 §7 完全相同，只是把非零位置挪到 RSN 位置之外。
和现有的 same-support 结果放在一起，就能把"位置"和"权重"两个因素拆开。Llama 尤其需要这组数据。
如果 off-support 在 Llama 上也能涨分，那说明 Llama 在这个工作点上对稀疏扰动本身就比较敏感，§7 的结论要相应收窄。如果不涨，就能支持"RSN 的位置也是特殊的"。
生成 mask 的工具 detection/nmd.py 里已经有，代价很小。
是否给 Llama 加更多 seed（比如加到 10 个）可以放到后面再定。3 个 seed 只能说明"RSN 比这三个方向好"，加 seed 可以换来一个百分位式的说法，但不是必需的。

---

1．先比较 Qwen 的 Expert／Non-expert／Neutral 提示。
使用同一批 GSM8K 300 题、Bare No-CoT、α=0，三个提示在同批运行。预先固定模板和检测器，主要看 early-candidate，辅助看首答案准确率。
明确预测：Qwen 的 Expert 提示应比 Non-expert 更少在开头输出答案候选，与正向 RSN 的变化对应。
结果	对解释的帮助
方向一致，差异清楚	支持角色提示与 RSN 存在行为对应
方向相反	削弱“RSN 只是复现专家提示策略”的简单解释
差异很小、区间较宽	这个检验暂时无法区分解释


同时将已有 Llama 角色输出按同一口径离线重算。这里即使发现对应，也只说明行为对应，不能直接证明内部路径相同。

### 第二优先：角色提示与 steering 的对应检验（先做一格，GPU 开销小）

RSN 在定义上是 expert − non-expert 的表征差，所以最直接的检验不预设心理功能，只问：

> 在同一模型、同一任务上，+α 与 −α 的差异，是否复现 expert 与 non-expert 提示之间的差异？

**待检验的解释（"角色策略"）：** +α 把模型推向它自己学到的"专家行为"，而专家在某个情境下做什么是训练决定的。决策任务里两个模型学到的专家都"果断"，所以 Betting / CGT / InfoQuest 方向一致；数学推理里 Qwen2.5 可能因大量推理训练而把"专家"学成先推导再作答，所以只有 Qwen GSM8K 的方向和 Llama 相反。如果成立，"方向相反"不再是矛盾，而是角色效应本身因模型和任务而异。

#### 第一步：只做 Qwen GSM8K（决定性的一格）

整个方向矛盾集中在这里，先用它决定这条线是否值得扩展。

- **设计：** Qwen2.5-7B-Instruct，GSM8K Bare No-CoT，α=0，同一次运行内跑 `an expert` / `a non expert` / `neutral` 三个提示（role 字符串与 Llama `run_gsm8k.sh` 一致），n=300，逐题配对。neutral 同批重跑，不与旧 `mdf_0` 跨批比较。输出写入新目录 `answer_mdf_gsm8k_role/`，不覆盖已有 cell。
- **预先写定的预测（运行前冻结）：** Qwen expert 提示的 `early_candidate_rate` **低于** non-expert 提示，方向与 Qwen 自身 +α 一致、与 Llama 相反。
- **读数：** 主读数 `early_candidate_rate`（`earlycand-v1`，冻结检测器，不重调）；辅助读数 offline `first_acc`；expert vs non-expert 用配对 exact McNemar，并报配对 bootstrap 区间。
- **判定：**

| 结果 | 解读 |
|---|---|
| 方向与预测一致，区间不含 0 | 支持"角色策略"解释，进入第二步 |
| 方向相反 | 解释被否定；方向差异来自 steering 本身而非角色，需另找原因 |
| 差异接近 0 / 区间跨 0 | 提示层面的角色效应太弱，本检验不能下结论，**不计为支持** |

#### Llama 对照：离线重算

此前的 Llama GSM8K 角色数字（expert 58.0% / ec 54.3%；neutral 60.0% / 48.0%；non-expert 68.0% / 41.7%）来自 inline 字段，**不能直接引用**。用冻结抽取器（`analyze_first_last_acc`）和 `earlycand-v1` 在 `llama3/gsm8k/bare/mdf_0` 上重算后，与 Qwen 用同一个分析脚本出对照表。纯离线。

#### 第二步（仅当第一步支持时）：扩展到行为任务

- Betting 和 CGT 各补 expert、non-expert 提示，两个模型，只跑 α=0。
- **这些都是新协议，不是旧结果的补格：** Betting 的冻结 prompt 没有 role 位置，加入角色句会改变 prompt；Qwen CGT v5 已 CLOSED。两者都只能与同批重跑的 neutral 比较，不能与 §3.1 / §3.3 的冻结结果比较。
- 可选：在 Chat 接口下看角色提示是否同样不再改变输出组织，用于解释表 7a 的接口依赖。

**限制：** 角色提示与 steering 方向一致只说明两者对应，是 consistent-with 证据，不证明 steering 的效应"经由"角色表征产生；提示改变的是整个上下文，steering 只改一组神经元。

### 第三优先：草稿末尾新加的"核验工具"实验，暂缓

它的思路很好：把"是否发起核验"和"能否利用核验结果"分开。但我有三点顾虑：

1. **接口风险。** 调用工具基本需要 Chat 接口，而表 7a 显示 Llama 在 Chat 下基线已经先推理后作答，RSN 在那里没有效果。Llama 很可能直接得到无效结果。Qwen 才可能有信号。
2. **前车之鉴。** Fixed-Answer、Willingness、InfoQuest 这三条新接口实验，都在建立测量接口阶段卡住了。这个实验需要新的工具接口、新的注入位置和新的评分方式，是同类风险。
3. **它不解决你最关心的问题。** 草稿自己也写了：即使成功，也只建立推理内部的联系，不能把 Betting、CGT 连起来。

所以建议：只有在角色对应检验失败、而我们仍然需要一座行为与推理之间的桥时，再启动它。届时先在 Qwen 上做 α=0 的小规模试点。

---
2．随后完成一次有限的 RSN–CSN 等范数比较。
你指出 CSN 在 GSM8K 上变化小、缺少清楚规律，这正是值得检验的差异。匹配实际稀疏注入向量的逐层范数后，比较双方的候选输出顺序和准确率，才能更有依据地回答：
RSN 的清楚效果，是否能够由 confident–unconfident 方向替代？

这一步应限定比较范围，避免不断为 CSN 寻找更好的剂量。

---

## 顺带的小事

- 草稿里的回退子集准确率还是反推出来的近似值，用冻结抽取器逐题重算一遍，大约十分钟。
- 先提交这一轮指标统一的文档改动。
- C1 里 Qwen CGT 降为附录里的边界证据，主文的行为证据以 Betting 为主。
---

**我觉得还能做，但下一步应该优先区分解释，而不是继续增加“RSN 在某个任务上有效”的例子。** 我会按下面顺序推进。

**1．完成正交注入对照，先确认具体方向是否重要。**

这是已经准备好的实验，最值得先收尾。关键看 **RSN 与各个等范数正交方向的直接差异**，不能只看 RSN 显著、对照不显著。

它回答的是：收益是否依赖这组特定权重，而非相近规模的扰动。即使支持，也只限定于这些对照和原工作点数据，不能据此命名心理机制。

**2．做一个小型“主动核验—反馈利用”实验，把行为和收益连接起来。**

[TMLR1005.md](/Users/paveenhuang/Downloads/Dopamine/Note/TMLR1005.md:340) 文末已经提出这个方向，我仍认为它比继续研究 marker 位置更有信息量：

给不同干预条件**完全相同的解题草稿**，包含正确和错误两类，设置两个分支：

| 分支 | 要区分的问题 |
|---|---|
| 工具可用，模型自主决定是否核验 | RSN 是否改变发起检查的行为？ |
| 直接提供相同的正确核验反馈 | RSN 是否改变利用反馈、修正答案的能力？ |

重点看错误草稿是否更容易触发核验、错误能否纠正，以及正确答案是否被改错。这样可以区分“检查更多”“检查更有针对性”和“拿到信息后更会使用”。

先做 **20–30 题的 α=0 pilot**，确认工具调用和反馈利用确实可测，再决定是否开展干预。外部反馈值得单独控制，因为已有研究发现，仅要求模型自行反思未必能稳定改善推理。[Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798)

**3．如果要直接回应“只是更像专家”，加入显式专家提示作为参照。**

在上述同一任务里比较中性提示、专家提示和 RSN steering，观察它们是否产生相同的核验与纠错模式。

如果效果接近，就应认真考虑：RSN 的实际用途是否主要是实现某类角色提示效应。如果出现分离，例如都增加检查，但只有一种改善针对性，就能进一步缩小功能解释。**相似或不同都不能单独证明内部机制相同或不同，但比“专家方向”这个名字更具体。**

GSM8K Base 可以保留为后续方向，不过它主要回答模型训练阶段的依赖性，对“究竟调节什么”的区分较间接。

**我的优先选择是：先完成正交对照，再只开一个核验 pilot。** 目标是获得一个能同时观察行为选择、信息利用和最终收益的实验，而不是再寻找一个容易出现正结果的任务。

---

## Vector
- 提取位置：当前 RSN 是 Role 提示在任务入口引起的综合状态差异。可以比较“身份描述结束处”和“生成入口”的 HS，区分身份编码与任务整合后的影响；目前还不能确定换位置会更好。
- 与 confidence 的差异方向：可以提取 Role 中无法被 confidence 方向解释的 residual。它可能帮助定位额外的功能成分，但不能直接叫“更纯粹的 Role”，因为仍可能包含其他提示和任务效应。
- Manifold reasoning Chat & MMLUE RSN

## Representation

| 层次 | 要回答的问题 | 推荐指标 |
|---|---|---|
| 几何结构 | 两组 activation 的空间布局是否相似？ | centered CKA、mCCA/SVCCA、正交 Procrustes residual、子空间夹角 |
| 可读出功能 | 在 A 中线性可读的信息，B 中是否仍可读？ | 同一标签/状态的线性 probe；跨条件训练-测试（train A → test B） |
| 因果功能 | 两个表征是否能产生同一种干预效果？ | norm-matched cross-steering / cross-ablation 矩阵 |

| Layer | Input | Core metrics | 回答的问题 |
| --- | --- | --- | --- |
| Mean Direction | 每个实验的全量均值差 $r=\mu_{\text{expert}}-\mu_{\text{non-expert}}$ | signed cosine、norm、逐层 cosine | 两个角色方向是否指向相近的位置？ |
| Sparse Coordinates | 均值方向的逐层 top-$k$ 坐标 | overlap、Jaccard、enrichment、符号一致性、双向能量包含率 | 是否涉及相同神经元？共享部分有多重要？ |
| Role-Transition CKA | 两个实验在相同题目上的逐题配对差 $d_i=h_{i,\text{expert}}-h_{i,\text{non-expert}}$ | **Centered linear CKA**；题目配对打乱作为 null | 不同题目的 role 变化之间，整体几何关系是否相似？ |
| Role-Transition Subspace | 各实验的逐题配对差 $d_i$ | 对 $d_i-\bar d$ 拟合 PCA；比较子空间夹角、交叉重建率 | 除平均方向外，role 对不同题目的影响模式是否相似？ |
| Direction–Subspace | 一个实验的均值方向 $r$ 与另一个实验的 role-transition PCA 子空间 $V_k$ | 投影比例 $R^2=\lVert V_k^\top r\rVert^2/\lVert r\rVert^2$，及随机基线 | 一个实验的平均角色方向在多大程度上落入另一个实验的变化空间？ |

## Brain

**Steingroever 健康常模对齐**

公开数据确实包含 617 名参与者的逐 trial 选择、收益与损失。不过数据混合了 95/100/150 trials 和三种 payoff scheme，因此必须只选与我们 IGT 协议完全匹配的子集。[Steingroever et al. 数据说明](https://openpsychologydata.metajnl.com/articles/jopd.ak)

建议检验：

- `net_block1–5` 学习曲线；
- net score 分布；
- deck preference；
- Wasserstein distance / RMSE；
- 按原始 study 做 held-out，而不是把所有参与者混在一起挑最佳 α。

它能支持的结论是：某个 RSN 条件产生的 IGT 行为最接近健康人常模。行为相似不能识别神经递质水平。因此建议称为 **human behavioural calibration**，而不是 dopamine-axis calibration。


**已有药理学方向对照：保留即可**

现在的定性文献结论已经够用了：

- α+ 与较早承诺、较高下注或 reward seeking 的方向相似；
- α− 与较保守行为的方向相似。

把它作为 Discussion 中的 correspondence table，不必继续寻找无法获得的三组相关系数。措辞停在：

> RSN manipulation exhibits behavioural correspondence with reported dopaminergic pharmacology effects.

**fMRI RSA：有潜力，但暂不作为当前投稿必做项**

NARPS 确实有 108 名参与者、四个 mixed-gamble runs、trial timing、BIDS 数据和公开预处理结果，技术上适合开展共享刺激分析。[NARPS 数据说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602933/)

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

---
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
9. 重新梳理一下ThinkingCurve的文档 ✔
9. qwen的具体分析 ✔
10. Qwen 的 output decisiveness: 从现有 7 个 H5 cell 提取 entropy/log(V)、top1、margin ✔
11. manifold llama3 实验以及结果整理 
11. manifold 补齐 Llama 全 α 曲线 ✔
12. manifold Qwen25 实验以及结果整理 ✔
13. manifold sentiity ✔
14. cross-model thinking curve + 再次整理thinking.md -> commitment regime ✔
15. GSM-Hard -> 1）最佳工作点可以复制；2）可以通过回答情况看来预测是否最优 ✔
15. GSM-Hard COT + alpha Vs. COT ✔
--- 
16. MATH 补充完整 ✔ 
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
26. multi-hop ProofWwiter OWA 无法得到有效的格式的答案 ✖
27. ZebraLogic WP 测试 ✖
28. 1-shot multi-hop ProofWwiter OWA ✖
29. FinQA ✖
30. GSM-Symbolic cot & non-cot ✔
31. ProofWriter check是不是格式问题导致Llama失效 -> 确认是格式问题 -> 是格式问题，用chat趋势ok ✔
32. 也修改一下CRUXEval的chat版本 cot & non-cot ✔
33. GSM-Symbolic行为学特征统计 ✔
34. Confidence neurons ✔
   1) 相关性分析：role neurons & confidence neurons; 相关性在11-19层上升 -> 实际上相关性非常的高，最高可以达到0.7左右
   2) confident & unconfident相关性分析：相关性在11-19最低，类似RSN
   3) overlap：band 内仅共享 46/180=25.6%，Jaccard 为 0.14；但是是显著高于随机
   4) shared-top 的单位贡献大约是 role-only/confidence-only 的 8 倍；是 neither-top 的 74 倍。
   5) overlap分析：整体 alignment 是广泛分布的，而非集中在极少数高贡献 neurons
35. cross-steering: MMLUE ✔
36. cross-steering: GSM8K ✔
---
37. 统一各个任务的行为统计指标 ✔
38. 缩小confidence alpha MMLUE: ACC + Behaivour ✔
39. 缩小confidence alpha GSM8K: : ACC + Behaivour ✔
38. 确认Loop的问题 -> Llama3补充chat template + steering的结果
   1) GSM8K ✔
   2) GSM-hard ✔
   3) MATH ✔
   4) 整理一份当前的结果 ✔
39. 试着理解chat template的影响的原因：注入点切回到answer <><><> Answer: MATH + GSM8K ✔
40. 确认Qwen上是不是存在相同的chat 现象 -> 基本可以确认RSN和Chat是类似的function ✔
41. chat - bare llama3 & qwen2.5 GSM8K/MATH/GSMHard -> 以目前的结果分析不出来 ✖
42. 采一下GSM8K/ MATH/ GSMHard/ MMLUE 上面的expert vs. non-expert HS ✔
43. 分析Role Matrix之间的关系 ✖ 除了部分top neurons重叠之外 几乎正交 没有相关性 
44. Manifold分析 reasoning RSN & Chat -> 类似top neurons之间的关系 ✔
45. 确认Reasoning RSN(RRSN)的expert与non-expert的关系；RRSN与MRSN之间的相关性 -> 还是和之前一样 核心重叠，相似度很低；✔
46. Steering RSM8K with RRSN ✔ 
47. Steering MMLUE with RRSN ✔ -> 结果很乱，和MRSN非常不一致
48. 看是不是RRSN的提取需要增加Abstention的提示 -> ARRSN 结果没有比较好 ✖ 
49. 换成全量的GSM8K RRSN 作为固定的RRSN ✔
50. 换成全量的GSM8K Chat 作为固定的ChatSN ✔
51. 删除之前所有关于RRSN的内容 ✔
52. GSM8K with anstention expert & non-expert performance ✔
   1) 当前的prompt控制不住输出，那么HS应该也是不对的 ✖
   2) 优化prompt v2 -> 有差异但是都会回答 ✖
53. 全量 1319 题 GSM8K Role 方向：GRSN & AGRSN ✔
54. 相似度 GRSN & AGRSN -> 相似度也不是很高 ✖
