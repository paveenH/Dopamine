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
11. manifold llama3 实验以及结果整理 ⏸ 
12. Manifold Qwen25 实验以及结果整理
---
1）现在只用 `−8/−6/0/+6` 能证明两端不同，但无法知道方向变化是突然发生，还是随 α 连续旋转。建议做一个很小的 Llama continuity check：只分析 **last-prefill、decoder 18**，加入已有的 `−4/−2/+2/+4/+8`，不扩展到所有层或 decode。
每个 α 画三条曲线：
- 位移幅度：`‖hα−h0‖`
- 相对负端主轴的 cosine / scalar coefficient
- 偏离该轴的 residual：`1−cos²`

再与 accuracy curve 并排展示。这样可以直接回答：
- 负端是否始终沿同一轴增长；
- `−6→−8` 是否确实是连续 overshoot；
- 正端从哪个 α 开始发生方向重组；
- 几何转折是否对应行为曲线的峰值和损害。
这会显著增强当前结论，但应定位为**同批数据上的描述性剂量曲线**，不是新的独立验证。
2）
---
1. **增加 reasoning task**：先做 GSM-Hard，再考虑 SVAMP/ASDiv。目标是检验 Qwen 的 `+6～+8` commitment 转折，以及 Llama 的负向工作区能否迁移；措辞是“检验迁移性”，不是预设一致性。
3. **不一致问题**：作为统领前两项的科学问题，不需要再单独堆一轮 α 曲线。
4. **暂不把 Qwen CGT/IGT 拉到更高 α**：reasoning 的高工作点不能直接迁移到行为任务。Qwen CGT 在 `+2` 已出现 knowing/label-bias 问题，`+4/+6` 是格式或构念失效；继续到 `+8` 更可能放大崩溃，不能说明“以前拉得不够”。可以重分析现有数据，但不建议先重跑。
3. **只增加一个外部 reasoning task**：优先 GSM-Hard，验证 commitment 转折能否迁移，而不是大规模铺 benchmark。
4. **必要时补 causal random/orthogonal injection**：如果正文要强调 RSN 方向本身具有机制特异性，这比继续增加行为任务更能回应 ACL 审稿。
5. **不重开高剂量 CGT/IGT**：它容易让论文重新散开，而且格式/knowing 已经先于剂量不足成为限制，可放在 boundary evidence。

---
### 1. 先比较共同的 boundary state

不要先用 pre-commit `s_t`，因为 Qwen 低剂量几乎立即提交，cohort 不可比。优先使用所有样本都有的指标：

- `Z_prefill`
- `entropy/log(V)`
- top1 probability
- probability margin
- `decode[0]` 的回弹幅度
- early-candidate / 首个答案位置

把每项转成各模型 α=0 基线下的标准分或百分位，再比较：

- Llama：`α=0 → −6`
- Qwen：`α=0 → +8`

关键问题是：两者的最佳剂量是否把这些指标推向相似的“高果断但不过度提交”的区域。若只是准确率提高、内部指标并不汇合，就不支持共同 working state。

### 2. 做严格的 held-out state matching

更有说服力的检验是：

1. 用训练题确定 Llama `α=−6` 的目标状态；
2. 只根据内部指标，选择距离该状态最近的 Qwen 剂量，不能查看 Qwen accuracy；
3. 在 held-out questions 上检验该剂量是否同时改善 accuracy、commit timing 和 loop；
4. 用 bootstrap 检验“最佳状态距离”是否显著小于 α=0 和极端剂量。

如果内部状态盲选出的剂量正好接近 Qwen `+8`，才能真正支持：

> 两模型从不同方向到达相似的功能工作区。

不过这一步最好以 output-distribution 指标为主；模型内标准化的 RSN 投影只能作为辅助，因为两条 RSN 轴并不天然是同一坐标。

### 3. 再做 hidden-state / manifold 对齐

如果要进一步解释“为什么方向相反”，可以使用已存 HS：

- 用两模型 α=0、相同题目的 hidden states 学习 PCA；
- 按相对层深逐层匹配，不直接拼接不同 layer band；
- 用 CCA 或 orthogonal Procrustes，在训练题上建立跨模型空间映射；
- 映射冻结后，在 held-out 题上比较各剂量到“successful commitment region”的距离；
- 同时比较 on-manifold reconstruction error、trajectory speed、切向方向和 commit centroid distance。

判别结果：

- `Llama −6` 与 `Qwen +8` 映射后靠近：支持不同基线、共同工作区；
- 最佳状态不靠近，但各自都改善行为：更可能是不同模型通过不同内部机制达到相似行为；
- 极端剂量偏离自然流形：说明性能下降或平台可能与轨迹失配有关。

我建议下一步先做第 1 步的 **boundary-state alignment**。它利用刚提取好的 output decisiveness，没有 pre-commit cohort 问题，成本最低；结果成立后再投入完整 manifold。

---

我建议下一步不要追求把两条曲线“整理成一致”，而是把“不一致”本身升级为研究问题：

> RSN 是否提供共同的 adaptive-calibration 功能，但不同模型通过不同工作区实现它？

简单分三步：

1. **先排除剂量口径问题**
   不再比较 raw `α`。改用各模型内部标准化的有效干预量，例如 `ΔG_prefill`，再比较 commit position、early candidate、top1/margin、loop 等共同读数。先判断“方向相反”是不是仅由 mask、层数和 activation scale 不同造成。

2. **做 blind working-state matching**
   用 Llama `−6` 的内部与输出状态定义目标区，不看 Qwen accuracy，选择最接近的 Qwen 剂量；然后在 held-out questions 检验它是否也带来较好的 accuracy、commitment 和 stopping。
   
   - 若接近 Qwen `+8`：支持“不同方向到达相似功能工作区”。
   - 若状态仍不接近：接受“两模型使用不同内部机制产生改善”。

3. **再做小规模 manifold pilot**
   直接用已有 H5，检查高剂量是沿自然流形移动、发生转向，还是单纯 scalar compression。只有几何指标能在一维 RSN 指标之外提供增量解释，才继续扩大。

我的优先顺序是：

> **标准化剂量与 boundary-state comparison → held-out matching → manifold pilot。**

暂时不建议加第三个模型，也不急着扩新 benchmark。先把“跨模型共同机制”究竟应表述为共同工作点，还是共同功能、不同实现，回答清楚。即使最后无法对齐，也不是复现失败，而是一个更有价值且更诚实的结论：**RSN 的 commitment-calibration 功能可迁移，但工作区具有模型依赖性。**
---
1. **Thinking Curve：最高优先级**
   直接回答核心问题：为什么 Qwen 需要正向、Llama 却可能需要负向调节。比较的应是早期候选、commitment、推理压缩、循环等行为状态，而不是 raw α。

2. **把 manifold 作为 Thinking Curve 的机制补充**
   看 RSN steering 是沿着自然推理流形移动，还是把状态推离流形；以及它能否额外预测 commitment/正确率。先做小型 pilot，不单独扩成大工程。

3. **增加一个 reasoning benchmark**
   建议优先 GSM-Hard，用预先固定的 `0/+4/+6/+8`，检验已发现的 `+6` 转折能否迁移到更难算术。不要宣称“一个工作点永久通用”；当前更合理的是：**commitment 转折可能迁移，但最佳准确率点仍受任务难度影响。**

4. **重分析 CGT/IGT**
   这项成本低，可以用  
   `evidence → recognition → utilization → commitment → outcome`  
   重新定位 Qwen 到底卡在哪里。但它主要补充行为边界，不要试图强行“救活”任务结果。

5. **第三个模型暂缓**
   它会显著增加校准、解释和篇幅成本。除非论文初稿完成后发现审稿主张必须依赖第三模型，否则两模型的方向差异本身已经很有信息。

另外，我认为有一项比第三模型更重要：在 Qwen 的代表性 `+6` 条件加入**等范数随机／正交方向控制**。它能说明 reasoning 改善来自 RSN 方向本身，而不只是任意 hidden-state 扰动。

一句话路线：

> **先解释 Llama–Qwen 的方向差异 → 用一个困难 reasoning benchmark 检验迁移 → 用 CGT/IGT 补行为边界 → 然后开始写论文；第三模型留作审稿风险储备。**

---
6. Reanalysize IGT&CGT based on the personality of qwen25 [@Dopamine0819]
7. 需要证明 working point吗？
   选择题并非完全不能用，但必须要求模型先自由推理、最后再给选项；这样又会引入 CoT prompt 的脚手架效应。所以接下来应优先选择自然开放生成、答案可自动核验的任务：
   - SVAMP / ASDiv：验证数学题内部迁移；
   - GSM-Hard：验证更困难算术是否需要不同工作点；
   - ProofWriter：观察非数学的规则推理和证明过程；
   - 带上下文的多跳问答：观察模型是否先整合证据再提交答案。
   - 之前qwen做不了会不会是因为不够大力？看起来至少要+6？同理llama3?

5. Behaviour: 测一下和人类的行为学对齐关系
6. Model:
   HumanLLM
   Meta 新模型 GGUF 量化版（含 17GB 版、视觉 projector、DFlash drafter，llama.cpp 直接用）
   https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
   Inkling-Small
   MiniCPM5-2B

7. 有什么是等待可以收益的task?
8. Manifold

---
### 8. 最终产出

- [ ] 一张 Llama–Qwen effective-state 对齐图，横轴不使用 raw α。
- [ ] 一张 task-entry → commitment trajectory 图。
- [ ] 一张 working-state 与失败区域图。
- [ ] 一张 RSN vs random/orthogonal causal control 图。
- [ ] 完整保留所有 dose、null、失败结果和非显著指标。
- [ ] 将结果写入 `AdaptiveThinking.md` 的 Qwen replication 新节。
- [ ] 运行配置、路径、metadata 和分析器口径写入 `CLAUDE.md`。
- [ ] Manifold 只有在提供额外解释力时，才进入正文机制主张。
- **尚未完成**：
  - Expert / Non-Expert Persona 分析；
  - 完整 11 档 confidence 曲线；
  - 真正的 **Llama–Qwen working-state alignment** 与 held-out 验证；
  - random/orthogonal direction 的**实际因果注入**；目前只有离线重投影；
  - 对应的跨模型 effective-state 图和 causal-control 图。
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

### 5. Optional Extensions（不阻塞首轮投稿）

- [ ] Base–Instruct：选择同架构 checkpoints，分别 self-localize RSN，比较 projection、direction overlap、behavioral working point 与 `Checkpoint × α` interaction。
- [ ] 只讨论 post-training 是否建立或增强该控制轴，不将 Base 解释为“低多巴胺”。
- [ ] HumanLLM / 人类行为相似性分析。
- [ ] 其他模型（Mistral、Muse-Glimmer、Inkling-Small）仅在 Qwen 结果明确后考虑。
- [ ] 人脑、fMRI/EEG、commit prediction 与动态 controller 留待主论文证据闭合后。

# Follow-up

0. Adaptive CoT router：只用 prefill 或 very early decode features 預測要不要 think。
   - RSN features: x_prefill, RSN projection mean / variance, middle-layer RSN activation, role-sensitive direction projection, first 5-10 decode token RSN slope
   - uncertainty features: MSP, entropy, constrained entropy, logit margin, E-option logit / abstention probability
   - frequency/dynamic features（參考 ICLR2026 Balanced Thinking）: step-level confidence variance, local fluctuation，用來區分 overthinking / underthinking
   - info-theoretic features（參考 NIPS2025 Think or Not）: InfoBias, InfoGain，先作 diagnostic / baseline，不急著當主控制器
   - baselines: entropy threshold, MSP threshold, answer logit margin, question length, random routing, always CoT, always No-CoT

1. 推理過程中 Dopamine curve 與 Thinking curve 的關係：在 reasoning model 的 `<think>` trace 裡對齊 backtrack / first-commit / hedging / verification marker。

# Brain

## B1. 理論框架

**RSA（Representational Similarity Analysis）核心邏輯：**
1. 給模型和人腦看同樣刺激
2. 分別產生 N×N 相似矩陣（RDM）
3. 比較兩個 RDM 的 Spearman 相關

**我們的預測**：RSN Δh 方向上的 RDM 應與 **ventral striatum / vmPFC** 相關，而與語言區（Broca / Wernicke）不相關。這直接說明 RSN 操縱的是 reward 表徵，不是語言表徵。

## B2. 兩條執行路徑

### 路徑 A：公開 fMRI 數據 + RSA（1–2 個月）

**推薦數據集：**

| 數據集 | 來源 | 優點 | 適用性 |
|---|---|---|---|
| **NARPS Mixed Gambles（ds001734）** | OpenNeuro | 108人，vmPFC+striatum，已預處理，BIDS | 最直接：gambling 行為 ↔ 我們 Betting 的 RDM 比對 |
| **Tom et al. Mixed-Gambles（ds000005）** | OpenNeuro | Poldrack lab 2007 Science，OFC+striatum 乾淨 | Reviewer 熟悉，說服力強 |
| **MID Task（多個數據集）** monetary incentive delay | OpenNeuro 搜尋 | Wanting 最直接的 fMRI 範式（reward anticipation） | 對應 Betting 的 incentive salience | 

**執行步驟：**
1. 提取我們的 LLM 在不同 α 條件下，layers 11–20 的 hidden states → 構建 RDM
2. 從公開數據集提取 ventral striatum ROI 的 trial-level 激活向量 → 構建腦區 RDM
3. 計算兩個 RDM 的 Spearman 相關（需設計共享的刺激結構）
4. 對照組：同一套分析在語言區（IFG / STG）的相關應接近零

**關鍵挑戰**：LLM 刺激（MCQ 題目）和 fMRI 刺激（賭注任務）的對齊——需要設計一批「LLM 和人腦都能做」的共享刺激集。

### 路徑 B：行為層次對齊（1–2 週，不需要 fMRI）

找已發表的人類行為數據（帕金森 vs 健康人 vs DA-agonist 組的 Cambridge Gamble / Iowa Gambling Task），與我們的 α 劑量比對：

| 組別 | 人腦 DA 狀態 | 預測對應 α |
|---|---|---|
| 帕金森未服藥 | 低 tonic DA | α<0 |
| 健康控制 | 正常 DA | α≈0 |
| DA 激動劑 / L-Dopa ON | 高 DA | α>0 |

#### 文獻偵察結論（2026-07-23，已查）

**① 方向學高度一致（可寫，定性）。** 人類 DA 藥理學在 gambling task 上的方向與我們的 α 預測完全對齊：
- α+ (高DA) → 更衝動 / 早承諾 / delay aversion↑：Multiple Modes 2013（高 levodopa 劑量 → delay aversion↑）；Cools 2003（L-Dopa ON → 理性決策但衝動下注）；Riba/Pizzagalli 2008（pramipexole → boost 後保守傾向消失，更 reward-seeking）。**直接對上我們 CGT-Sequential 的 α+ → accept_step↓ / DAI 展寬。**
- α− (低DA / 未服藥帕金森) → risk-averse / 保守；DA 治療才轉向 risk-taking。

**② 嚴格「三組行為向量相關係數」做不了。** 查過的三篇關鍵文獻都沒有可對齊的 trial-level 或乾淨組均值表：Riba 2008 只報聚合百分比（placebo 47% vs pramipexole 49%，n.s.，核心效應在 boost 試次，無 mean bet/SD）；Multiple Modes 2013 只有 patients-vs-controls 總體、無 ON/OFF 分組、無 CGT 子量表 mean/SD；Cools 2003 結論為文字性。→ 缺三組可比數值向量，原設想的相關表無法計算。

**③ 落地形態改為「藥理學方向定性對齊」**：一段 correspondence-to-human-DA-pharmacology 敘述 + 一張定性對照表（未服藥帕金森↔α−；agonist/L-Dopa↔α+；我們的 α 在 CGT-seq / IGT 上再現人類 DA 方向）。夠撐 EMNLP/NeurIPS discussion 一節，但為定性方向複現，非定量 RSA。措辭停在行為層，勿跳神經層。嚴格定量須寫信向作者（Cools / Djamshidian 組）要 raw data（合作級，非 1–2 週自辦）。

#### ★ 可做的定量版本：用 Steingroever 617人常模校準 α 軸零點

反過來用「只有健康人常模」這個限制：掃 α∈{−8…+8}，看哪個 α 的 IGT 行為**分布**與 617 名健康人最接近。
- 若最佳對齊 α≈0 → **數據驗證**出 α=0 baseline = 人類正常 DA 水平（比「假設 α=0=健康人」強），順勢錨定 α−=帕金森方向、α+=agonist 方向，間接補上缺的兩組。
- 對齊在分布層做：主讀數 = 逐 block 學習曲線（net_block1–5，形狀 RMSE/相關）或 net_score 整體分布（KS / Wasserstein）。指標口徑與 `analyze_igt.py` 一致（net_score / net_block / deck preference = IGT 標準指標，與 Steingroever 可比）。
- **關鍵前提（待確認）**：須挑 Steingroever 中用**經典 Bechara 100-trial payoff scheme** 的子集（我們的 IGT 是這個），否則學習曲線不可比；並確認其提供 trial-level 選牌序列（才能算 block 曲線）。
- 零 GPU（IGT 各 α 數據已有）、零合作門檻（公開可下載）。措辭：「α axis 的行為零點與人類健康常模一致」，勿 over-claim 到神經層。

**可用公開數據 / 文獻**：
- [617人 Iowa Gambling Task 數據（Steingroever et al.）](https://openpsychologydata.metajnl.com/articles/jopd.ak)（純行為，完全公開；健康常模，用於零點校準）
- Riba, Krämer, Heldmann, Richter, Münte (2008) *Dopamine Agonist Increases Risk Taking but Blunts Reward-Related Brain Activity*, PLOS ONE — [PMC2423613](https://pmc.ncbi.nlm.nih.gov/articles/PMC2423613/)
- *Multiple Modes of Impulsivity in Parkinson's Disease* (2013), PLOS ONE — [10.1371/journal.pone.0085747](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085747)
- Cools, Barker, Sahakian, Robbins (2003) — L-Dopa ON/OFF CGT（理性決策但衝動下注）
- 帕金森 CGT/IGT 文獻（Djamshidian et al. 2010/2011；DBS ON/OFF delay aversion, PMC3439437）— 多為論文表格均值，非 raw data


# PLAN

### Phase 1（2 週內，低成本或 zero-cost）

| # | 實驗 | 成本 | 目的 |
|---|---|---|---|
| 1 | Bandit 逐輪收斂曲線（現有數據） | 0 | 可視化證據 |
| 2 | Known-correct subset analysis（現有 log） | 0 | 純 wanting 證據 |
| 3 | 行為層次腦對齊（路徑 B，文獻比對） | 0 | 最快的腦連結 |
| 4 | Cambridge Gamble Task 設計 + 跑 Llama3 | ~1 GPU day | 排除 Betting 的信心 confound |
| 5 | Betting alpha sweep α∈{−8..+8} | ~1 GPU day | 畫出完整倒 U |

### Phase 2（1 個月內，核心 claim）

| # | 實驗 | 成本 | 目的 |
|---|---|---|---|
| 6 | **Tülu-3 SFT vs DPO on Betting** | ~2 GPU days | RLHF punchline |
| 7 | **Llama3-Base self-localized mask** | ~3 GPU days | 排除 mask 適配問題 |
| 8 | Random/PCA/Prompt baselines | ~1 GPU day | Reviewer 必問 |
| 9 | 公開 fMRI RSA 初版（NARPS ds001734） | 1–2 週 | 腦對應量化 |

### Phase 3（如有空間）

| # | 實驗 | 目的 |
|---|---|---|
| 10 | Qwen3 + Mistral Betting 跨模型 | 廣度 |
| 11 | Loss-aversion framing on Betting | Prospect theory 連結 |
| 12 | 共享刺激集設計 + neural encoding | 登頂級期刊的路 |
| 13 | **Pressure × Confidence Dissociation** | 區分 DA-like commitment vs confidence |
| 14 | **Task Difficulty × RSN Activation（現有數據）** | DA effort/uncertainty 對應 |
