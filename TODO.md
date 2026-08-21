
## TO DO
0. 测试一下qwen中间一点的mask ✖
1. Qwen25-7B ICG实验结果整理 ✔ 
2. 整理行为学的结果 ✔
3. Qwen GSM8k实验以及结果分析 ✔
4. Qwen MATH ✔
5. Qwen High-Dose in GSM8K ✔ 
5. MATH cot 真正需要补的只有 MATH，并应等待 MATH-CoT 后一次完成：
   - commit 前后字符和等式；
   - +8 难题回落是否来自提交前推理被压缩；
   - \boxed{} 极端重复尾部；
   - No-CoT 与 CoT 是否改变峰位或有效区间。
6. Reanalysize IGT&CGT based on the personality of qwen25 [@Dopamine0819]
7. 需要证明 working point吗？
   选择题并非完全不能用，但必须要求模型先自由推理、最后再给选项；这样又会引入 CoT prompt 的脚手架效应。所以接下来应优先选择自然开放生成、答案可自动核验的任务：
   - SVAMP / ASDiv：验证数学题内部迁移；
   - GSM-Hard：验证更困难算术是否需要不同工作点；
   - ProofWriter：观察非数学的规则推理和证明过程；
   - 带上下文的多跳问答：观察模型是否先整合证据再提交答案。

5. Behaviour: 测一下和人类的行为学对齐关系
6. Model:
   HumanLLM
   Meta 新模型 GGUF 量化版（含 17GB 版、视觉 projector、DFlash drafter，llama.cpp 直接用）
   https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
   Inkling-Small
7. 有什么是等待可以收益的task?
8. Manifold

守规矩、会分析、善于解释，但行动层面偏僵硬；外部推动合适时能变得果断，过强时又容易机械化或越界。
反应快、凭直觉、愿意行动，但自我控制较弱；外部推动合适时能更坚定，过强时容易抢答、固著和反复纠缠，推动不足时又会犹豫、失去行动稳定性。
Qwen：先控制、后行动，问题是过度规则化。
Llama：先行动、后控制，问题是冲动与难以收住。
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

执行顺序：

```text
现有行为证据与主张冻结
→ 选择一个稳定正结果任务
→ Qwen 跨模型复现
→ 同一代表任务的直接方向控制
→ 主结果图、全文与复现材料
→ Optional extensions
```

行文参考：*Hippocampo-neocortical interaction as compressive retrieval-augmented generation*（Nature Communications, 2026）。

# Follow-up

0. Adaptive CoT router：只用 prefill 或 very early decode features 預測要不要 think。
   - RSN features: x_prefill, RSN projection mean / variance, middle-layer RSN activation, role-sensitive direction projection, first 5-10 decode token RSN slope
   - uncertainty features: MSP, entropy, constrained entropy, logit margin, E-option logit / abstention probability
   - frequency/dynamic features（參考 ICLR2026 Balanced Thinking）: step-level confidence variance, local fluctuation，用來區分 overthinking / underthinking
   - info-theoretic features（參考 NIPS2025 Think or Not）: InfoBias, InfoGain，先作 diagnostic / baseline，不急著當主控制器
   - baselines: entropy threshold, MSP threshold, answer logit margin, question length, random routing, always CoT, always No-CoT

1. 推理過程中 Dopamine curve 與 Thinking curve 的關係：在 reasoning model 的 `<think>` trace 裡對齊 backtrack / first-commit / hedging / verification marker。



# Reference: candidate anxiety / mental-health benchmarks

| Benchmark / Scale | # Items / Samples | What It Tests | Why It May Be Useful Here | Source |
|---|---:|---|---|---|
| **STAI-s LLM Anxiety Protocol** | 20 STAI-state items; paper repeats administrations across conditions | Anxiety-like **state self-report** in LLMs under baseline / trauma-induction / relaxation prompts | Best direct fit for testing whether α steering changes anxiety-like questionnaire scores | [npj Digital Medicine paper](https://www.nature.com/articles/s41746-025-01512-6) · [GitHub](https://github.com/akjagadish/gpt-trauma-induction) |
| **STAI full** | 40 items: 20 state + 20 trait | State anxiety + trait anxiety | Could separate temporary α-induced state from stable persona-style trait responses | [STAI overview](https://www.ebsco.com/research-starters/health-and-medicine/state-trait-anxiety-inventory-stai) |
| **GAD-7** | 7 items | Generalized anxiety symptom severity | Very lightweight anxiety probe; easy pilot, but short and human-symptom framed | [AHRQ GAD-7](https://integrationacademy.ahrq.gov/resources/7336) |
| **DASS-42** | 42 items: Depression 14 / Anxiety 14 / Stress 14 | Depression, anxiety, and stress dimensions | Good next probe after STAI-s because it can test whether α+ specifically raises anxiety/stress rather than all negative affect | [DASS-42 overview](https://www.sralab.org/rehabilitation-measures/depression-anxiety-stress-scale) |
| **PROMIS Anxiety Item Bank** | 29 anxiety items | Anxiety symptoms across a broader item bank | More anxiety-specific than DASS; useful if we want more than 20 anxiety items | [PROMIS Anxiety item bank reference](https://www.sciencedirect.com/science/article/pii/S0022399926000954) |
| **PHQ-9** | 9 items | Depression symptom severity | Short depression contrast; useful as a negative-control affect dimension, but too short for main α curve | [PHQ-9 overview](https://www.apa.org/depression-guideline/patient-health-questionnaire.pdf) |
| **SCL-90-R** | 90 items | Broad symptom checklist: depression, anxiety, phobic anxiety, obsessive-compulsive, etc. | Large multi-domain probe, but copyright/commercial-use concerns make it less convenient | [SCL-90-R overview](https://www.pearsonclinical.com/psychology/products/100000645/symptom-checklist-90-revised-scl-90-r.html) |
| **MentalBench** | 24,750 synthetic clinical cases | DSM-grounded psychiatric diagnosis and differential diagnosis | Tests mental-health reasoning, including anxiety-disorder recognition; not a model-state anxiety probe | [Hugging Face dataset](https://huggingface.co/datasets/hysong/MentalBench) |
| **SMHD** | Large Reddit user-level dataset; includes anxiety and depression diagnosis labels | Mental-health condition classification from user posts | Useful if we want anxiety/depression recognition from naturalistic text, not self-report state | [SMHD resource](https://ir.cs.georgetown.edu/resources/smhd.html) |
| **IMHI / MentaLLaMA benchmark** | 100K+ instruction-style mental-health samples | Mental-health intent / risk / support / diagnosis-style tasks | Useful for testing whether α changes mental-health reasoning or safety behavior | [MentaLLaMA paper/project](https://arxiv.org/abs/2309.13567) |
| **eRisk** | Yearly shared-task datasets; size varies by task/year | Early risk detection for depression, self-harm, anorexia, etc. | Good for longitudinal mental-health detection, but less directly tied to anxiety-like model state | [eRisk overview](https://erisk.irlab.org/) |

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
