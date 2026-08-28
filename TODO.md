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
