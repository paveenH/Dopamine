**整体判断**

三份文档已经构成一条比较完整的证据链：

```text
RSN neurons
→ α 可线性操纵 task-entry gain
→ 非线性改变 commitment / engagement state
→ 在不同任务上产生不同的行为 working point
→ 功能上类似 dopaminergic adaptive calibration
```

### 目前已有的结果

1. **行为学层最强**
   - Betting 中 α 显著改变下注，但 accuracy 基本稳定，是目前最干净的 **wanting–knowing dissociation**。
   - Bandit 显示非负 α 的宽平台与 `+8` 过载崩溃。
   - IGT 提供较弱的探索、history integration 和 punishment sensitivity 证据，应作为 boundary condition。
   - HaluEval 表明 α 调节的是 challenge/verification engagement，而不是 hallucination capacity。

2. **GSM8K 提供最完整的生成行为画像**
   - nominal 最佳点为 `α=-6`，accuracy 从 baseline 60% 升至 78%；`-8` 则崩溃。[AdaDopamine_gsm8k.md](/Users/paveenhuang/Downloads/Dopamine/AdaDopamine_gsm8k.md:64)
   - 正 α 对应更早 commit、较低 committed accuracy，以及答案后继续生成的 letting-go failure。[AdaDopamine_gsm8k.md](/Users/paveenhuang/Downloads/Dopamine/AdaDopamine_gsm8k.md:161)
   - anxiety-like 重复在 `-6/-4` 最低，两端升高，但负端和正端的失败机制不同。[AdaDopamine_gsm8k.md](/Users/paveenhuang/Downloads/Dopamine/AdaDopamine_gsm8k.md:195)
   - CoT 降低这些破坏，但保留 α 的方向。[AdaDopamine_gsm8k.md](/Users/paveenhuang/Downloads/Dopamine/AdaDopamine_gsm8k.md:279)

3. **内部信号层找到了输入到行为的转换**
   - `G_prefill` 随 α 近乎线性。
   - 行为不是由 `G_prefill` 绝对值直接决定，而是经过非线性的 pre-commit `s_t` state。
   - `early_s_t` 与 dose-level accuracy 同步，但目前只是 9 个剂量点的 covariation，不能称为中介机制。[AdaptiveThinking.md](/Users/paveenhuang/Downloads/Dopamine/AdaptiveThinking.md:603)
   - `p_t` 目前只是 fast residual，尚不能称为 phasic dopamine。
   - 当前最合适的总论断仍是 **controllable latent gain mechanism / computational analogue**。[AdaptiveThinking.md](/Users/paveenhuang/Downloads/Dopamine/AdaptiveThinking.md:742)
---
## TODO
1. 完成random的分析
2. 抽样samples看看实际提交的回答是什么样子
3. 实际的samples如果有发现什么问题 后面提交的结果可能要重新计算
3. 思考一下这个差异可以用来做什么

---
### 1. RSN Direction Specificity（最高优先级，zero/low GPU）

**目标：** 确认当前信号集中在 NMD/RSN 方向，而不是任意稀疏方向都会出现。

- 用当前 `G/Z` 坐标重算 NMD vs random mask，覆盖 CoT/No-CoT、Expert/Non-Expert 和 α-dose。
- 每个 mask 使用自己的 neutral reference、norm 和 standard deviation，避免因尺度不同产生假优势。
- 主读数：`G_prefill`、pre-commit `s_t mean/slope`、`p_t abs_mean/std`；commit-centered 图只作时间定位。
- 优先生成至少 10 个 norm/sparsity-matched random masks，构成 random null distribution；单一 random mask 只作初步检查。
- 将 **offline re-projection** 与真正的 **random-direction causal steering** 分开命名。

**完成标准：** RSN 主效应稳定高于 random null，且不只由少数 layer 或单一 condition 驱动；否则将结论降为一般 sparse-state effect。

### 2. Slow-State Behavioral Validation

**目标：** 判断 `s_t` 表示 ramping/vigor，还是较一般的 slow engagement / commitment state。

- 在 common-valid questions 上，检验 `s_t level/slope` 与 commit position、generation length、premature commitment、answer oscillation、post-commit loop 的 per-sample 关系。
- 使用 item-level regression，同时纳入 difficulty/correctness、entropy/top1、response length 和 commit-marker availability。
- 分开分析 pre-commit level、pre-commit slope 与 post-commit release，不以整段平均替代 event-centered readout。
- 在 held-out questions 上验证预测方向。

**完成标准：** slope 在控制 length/confidence 后仍稳定预测推进速度，才保留 **ramping/vigor**；若只有 level 稳定，则统一改称 **slow engagement / commitment state**。

### 3. α-Steering Anxiety-Scale Experiment

**目标：** 测试 α 是否改变 anxiety-like state self-report，并排除通用数字选择或极端作答偏置。

```text
STAI-State 20 items × α∈{−8,−6,−4,−2,0,+2,+4,+6,+8}
× 5 item-order permutations
```

- 每个 item 独立 prompt、独立 prefill steering，不累积 conversation history；固定输出为 `Response: 1/2/3/4`。
- 同时加入 STAI-Trait、neutral non-affective Likert control，以及原量表的 reverse-keyed items。
- 主读数：总分、anxiety-present / anxiety-absent 分量、reverse-key consistency。
- 诊断：extreme-response rate、acquiescence bias、invalid rate、response variance、各数字选择比例。
- 将 dose-level scale profile 与 GSM8K anxiety-repeat、answer oscillation、commit timing 和 loop 指标对齐；不把自评量表单独当作“模型真的焦虑”的证据。
- 实施前确认量表条目的使用与公开发布条件。

**完成标准：** `α effect on STAI-State > STAI-Trait > neutral control`，reverse-key 后方向一致，并与至少一个非自评 anxiety-like behavior 同步；否则解释为 response bias。

Barratt 冲动性量表 (BIS-11)
轻躁狂症状清单 (HCL-32)
自评量表：强迫症问卷修订版 (OCI-R) 或 帕多瓦量表 (PI-WSUR)
行为测试（硬指标）：威斯康星卡片分类测验 (WCST, Wisconsin Card Sorting Test) 的文本变体
Wisconsin Card Sorting Test (WCST-64) 参考-Cognitive.RolePlaying.2025.Visual Large Language Models Exhibit Human-Level Cognitive Flexibility in the Wisconsin Card Sorting Test / icmlworkhop2024.Cognitive Flexibility of Large Language Models (不建议)

### 4. Direction-Specific Causal Controls

**目标：** 证明行为 working point 来自 RSN 方向，而不是一般 hidden-state perturbation。

- 选择需求相反的两个任务：GSM8K（偏负 working point）与 Betting 或 Bandit（非负 working region）。
- 比较 RSN、至少 10 个 random directions、PCA direction 和一个非 wanting 功能方向。
- 按实际 intervention norm 或 `ΔG_prefill` 匹配剂量，不直接比较相同 raw α。
- GSM8K 主读数：accuracy、committed accuracy、commit position、oscillation/anxiety；Betting/Bandit 主读数：bet/OptFrac 与 accuracy/invalid control。

**完成标准：** RSN 在两个任务中产生预测方向相反但 task-appropriate 的 working-point shift，且效应显著超出 matched-control distribution。

### 5. Cross-Model and Post-Training Replication

**目标：** 判断该 latent gain mechanism 是否可泛化，以及 post-training 是创造还是 sharpen 它。

- 在同一模型家族的 Base → SFT → DPO/Instruction-tuned checkpoints 上分别 self-localize RSN。
- 比较 neuron/layer overlap、direction similarity、`G_prefill` gain、behavioral working point 和 steering sensitivity。
- 先用 Betting + GSM8K 两个代表任务；主结论稳定后再扩展 Qwen/Mistral，不立即复制全部 benchmark。
- 所有模型使用各自定位的方向与各自校准的 α/`ΔG_prefill`，避免直接搬用 Llama 的 mask 和 raw dose。

**完成标准：** 至少一个独立模型或同家族训练阶段复现 direction-specific、task-dependent working point；若只在 Llama3-IT 成立，则将结论限定为 model-specific mechanism。

人脑、fMRI/EEG、commit prediction 与动态 controller 暂时放到以上验证之后。当前执行顺序：

```text
analysis freeze
→ RSN specificity
→ slow-state behavioral validation
→ α × anxiety scale
→ direction-specific causal controls
→ cross-model/post-training replication
```


---
# TODO
0. Trajectory：看一些没有平均的raw samples的曲线是什么样子 关注关键点（commit/answer）
1. Trajectory：直接看Cot在neurons上的改变 copy COT的曲线
2. Trajectory：预测commit，然后再commit前后进行干预；需要
2. Trajectory：signal部分补充random-mask的对比
3. Behaviour: 测一下焦虑和心理学量表
3. Behaviour: 测一下和人类的行为学对齐关系
4. RLHF: RLHF 和 dopamine 出現的關係：整理 Notion `Model Analysis; Hallucination & Origin -> 15. Origin Analysis`，看 post-training 是否 sharpen 了 decisiveness axis。

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

