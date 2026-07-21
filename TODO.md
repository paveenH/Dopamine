# TODO
---

## 1. Active TODO
0. 直接看Cot在neurons上的改变 copy过去的结果
1. 补充alpha的结果
2. **Trajectory 曲線** — 改用差值（diff）看看。 -> 更符合多巴胺的应该是差值
3. **Capitulation** — 這部分先不要放，之後再考慮是不是要直接 Generation 看看（可能可以找到 reflection？）。主要原因是上升和下降都會有合理的解釋 → 可能可以用來分析 Llama3 和 Qwen3 的不同。

---

## 2. Dopamine-Relevance Tiers (experiment triage)

实验按"与 tonic-DA wanting 的关系强度"分三档。

### A 档：直接测 wanting，是 dopamine 框架的承重墙（关系最强）

| 实验 | 为什么是直接 dopamine 证据 |
|---|---|
| **Confidence Betting (⑤)** | 非语言下注行为大幅位移、accuracy 不动 = 教科书式 wanting–knowing 分离 |
| **Bandit (⑥)** | tonic DA 调 explore/exploit；WorstFrac 同步↓ 排除混淆，−4 干净 effort-withdrawal |
| **Agentic (⑧)** | 长序列倒 U 右侧（+α 过载崩溃）——唯一能测 mania-zone 的 |
| **GSM8K/MATH α-scan** | accuracy 维度独立复现倒 U（−4/−6 峰，±8 崩） |

### B 档：间接 / 副作用 / 边界证据（关系绕了一层，不是没关系）

| 实验 | 与 dopamine 的关系性质 |
|---|---|
| **MMLU-E abstention** | 测 action threshold（答/不答），是 wanting 的下游行为，不是 wanting 本身。母 paper 的核心，但 §4.0 自标"中强；只是答/不答" |
| **Pressure & Capitulation (①)** | §4.2：社会压力走 HPA→cortisol→PFC DA，是 PFC goal-maintenance，与 tonic DA 间接；+steering 降 cap 也可能是 output sharpening 而非 wanting。CLAUDE.md 已据此移出主线 |
| **Willingness / Confidence self-report (0–9)** | 口头自评 = PFC 认知评估，理论上是错的 proxy（Berridge wanting 是非意识的）；数据也证实 willingness −4 反向。保留为负控，不是 wanting 证据 |
| **TruthfulQA / FACTOR / HaluEval** | 测 over-wanting → hallucination 的副作用，不是 wanting 强度本身 |
| **CGT / IGT** | 概率透明赌博，是 betting 的 confidence-confound 对照控制，本身也测 risk-taking——相关，但定位是"排除混淆"而非"主证据" |

---

## 3. Benchmark Tiers and Pending Extensions

| Dataset / Experiment | Status | Dopamine relevance | Judgment |
|---|---|---|---|
| **MMLU-E Abstention** | ✅ Done | action threshold / willingness to answer | 中强；但只是答/不答 |
| **GSM8K / MATH α scan** | ✅ Done | commitment timing, over-/under-wanting in reasoning | 重要主结果；但不是纯 dopamine assay |
| **GSM-NoOp / GSM-Symbolic** | 🔶 Pending | salience gating / distractor suppression / variable tracking | 可做机制扩展；但仍是 accuracy proxy，非 effort-expenditure 主证据（见 §4 Effort 说明） |
| **BBH tracking tasks** | 🔶 Pending | working memory / set-shifting | 神经认知相关；但偏 PFC working memory |
| **TruthfulQA-Generation / HaluEval** | 🔶 Pending | over-wanting → hallucination / over-generation | 可测副作用；不是纯 wanting |
| **Effort-based Task Choice（实验 C）** | ❌ 范式过弱 | effort willingness | 方向对、范式太弱：二选一无强度维度、无真实成本-报酬耦合、单 token argmax 杀掉动态；测到的是 preference/anchor 不是 effort expenditure。须重做强版，见 §4 |
| **CRT（实验 D）** | ❌ Done-null | cognitive effort avoidance | 三个 α（0/+4/−4）choice_S2_rate 全 = 0.571，steering 完全无作用，放弃 |
| **Agentic / ScienceWorld（实验 ⑧）** | 🔶 Done-but-shelved → **建议重启** | 长序列 goal persistence；**倒 U 右侧（mania-zone 过载崩溃）** | **被错杀**：原以「结果不符预测」搁置，实为整套记录里唯一干净的 Yerkes-Dodson 右侧证据。须重新框定 + 补 penalty 来源归因，见 §4 |

---

## 4. Pending: a stronger effort paradigm

**夠强的 effort 范式应具备：** within-task、可连续调节、付出与报酬真实耦合、能定位放弃点（breakpoint）。

1. **Progressive Ratio 语言版**（effort 文献黄金标准 = breakpoint）：任务链上每多走一步成本递增（如多跳 HotpotQA / 连续加深解题），测模型在哪一步放弃 → α 应移动 breakpoint。HotpotQA 可作 PR 语言版载体。

---

## 5. Comparison with §2 behavioral findings (TBD)

待信号出来后，把 §2 的行为方向（α+4/expert 高 wanting）与 §4 的 RSN 投影信号方向并排，确认"行为上的想不想"是否对应"隐状态投影的高低"，以及这种对应是否 NMD-specific。

---

## 6. Reference: candidate anxiety / mental-health benchmarks

（为「α+ → over-wanting / anxiety」一侧寻找直接 state self-report 探针时的候选清单。）

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
---
### Agentic / ScienceWorld（实验 ⑧）—— 被错杀的「倒 U 右侧」钉子，建议重启

**为什么值得记录**：betting / bandit 都是**单步**任务，只能展示倒 U 的左半 + 顶点（+α 有益、wanting–knowing 分离）。**唯一能展示倒 U 右半（mania-zone 过载崩溃）的是长序列 agentic rollout**——而这个数据已经跑出来了，结论干净，却因为「−α 没按预测上升 abandonment」被整体 Skip。

**已有结果（Llama3-8B-IT，layers 11–20，TOP=20，30 tasks × 5 episodes = 150 ep/cond，`Ada_Dopamine1.md` §4.9）**：

| 条件 | mean_score | std | success%（score>0） | penalty%（score=−100） | 30 任务里得分最高 |
|---|---|---|---|---|---|
| α=−4 | **6.21** | 9.60 | **56.0%** | 0.0% | 20/30 |
| α=0 | 5.87 | 9.10 | 55.3% | 0.0% | 10/30 |
| α=+4 | **−6.38** | 31.00 | **24.0%** | **9.3%** | **0/30** |

**重新框定（关键）**：原 Skip 理由是「−α → abandonment↑、success↓」未观察到。但这恰恰是**对的方向**——betting/bandit/GSM8K 已证 neutral Llama 处于 over-wanting 区（GSM8K No-CoT 峰在 α=−4），所以 −4 让 agentic 更谨慎、略好（6.21 > 5.87）完全自洽，本就不该期待 abandonment↑。这个实验真正测到的不是「−α→放弃」，而是「**+α 在 50 步序列上把 over-wanting 推过阈值 → 冲动执行 → −100 惩罚暴增、得分崩溃（+5.87→−6.38，std 9→31，0/30 任务最佳）**」，即 Yerkes-Dodson 右侧下降的**最强、最剧烈**证据（比 MCQ 难题退化剧烈得多，因长序列累积冲动误判）。

**唯一待补**：penalty 来源归因。§4.9 已定位 penalty 集中在 4 类需精确识别/测量的任务（identify-life-stages / measure-melting-point / lifespan / test-conductivity），但「冲动误判 vs 任务结构本身罚分」两种解释当前无法区分。**补法**：只跑这 4 个 penalty 高发任务 × 5 ep × 3 α（0/+4/−4）+ per-step score-trace dump，即可分离 penalty 是否由 +α 的冲动 action 触发。规模 ≈ 4×5×~50×3 ≈ 3000 串行 forward（≈ 一遍 bandit 量级），远小于全量 30 任务的 ~22.5k。

**成本提示**：agentic 是这批实验里单位算力最贵的——多步 rollout，每步 bs=1 依赖上一步环境反馈，不可 batch；`max_steps=50`、`max_new_tokens=32`。全量 30×5×50×3 ≈ 22.5k 串行 forward + 每步 ScienceWorld env step 开销。针对性复现那颗钉子只需 ~1/7 的量。脚本：`get_answer_sciworld.py` / `run_sciworld.sh`（现脚本已被砍成 `--task_nums 28 29` + 单 α 的分片小批形态）。

---
<!-- 主線導覽（三份文檔共用，每份開頭都有）

整條研究主線（四段）：
  RSN
    → 行為學多巴胺（Behavioral Dopamine）← Ada_Dopamine.md
        → 腦科學多巴胺（Brain Dopamine）← 本文檔 §五
            → 多巴胺與思考曲線（Dopamine & Thinking Curve）← AdaptativeThinking.md

  附：AdaThink.md 是 Thinking Curve 的額外延伸驗證（學弟執行），不在主線框架內。

【本文檔定位】
兩個角色：
  1. 實驗 Roadmap（§三–§六）：整理已完成的 Tier-A/B/C 實驗結果，規劃下一步
     方向 A（行為學延伸）、方向 B（腦科學 RSA）、方向 C（RLHF 機制）。
  2. 腦科學升華（§五）：從行為類比進一步論證 RSN Δh 表徵是否對應
     ventral striatum / vmPFC，而非語言區——這是從「功能類比」升級到
     「表徵對應」的關鍵一步。

【前後段的任務】
Ada_Dopamine.md：行為學基礎驗證（wanting/knowing 解離、Bandit、Pressure）。
AdaptativeThinking.md：最終升華——thinking trace 中的多巴胺動力學，
透過 LLM 實驗模擬人腦思考過程的 motivation dynamics。

關聯文件：
  Ada_Dopamine.md — 行為學理論框架與實驗結果
  AdaptativeThinking.md — Thinking Curve + 閉環控制實驗（Phase 1-2）
  AdaThink.md — Reasoning model trace-level 分析框架
-->

# Related work 
The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs (ResponsibleFM @ NeurIPS 2025)

# Dopamine Framework — Research Roadmap

*May 2026 — 接續 Dopamine.md 的後續實驗規劃*

關聯文件：`/Users/paveenhuang/Downloads/RolePlaying/Dopamine.md`

---

## 四、方向 A：行為學基礎實驗

**目標**：建立三個行為學 anchor，對應多巴胺文獻中最經典的範式

---

### A2. Confidence Betting — Alpha Sweep & 延伸

| 子實驗 | 內容 | 目的 |
|---|---|---|
| Alpha sweep α∈{−8,−4,−2,0,+2,+4,+8} | MMLU + GPQA | 畫出 mean_bet 倒 U 形狀 |
| Known-correct subset | 三條件都答對的 sample | 純 wanting 證據（排除正確率 confound） |
| Loss-aversion framing | Gain frame vs Loss frame（同 EV） | 高 DA → loss aversion↓；prospect theory 連結 |

---

### A3. Bandit — 逐輪收斂曲線（zero-cost）

從現有 30×50 數據畫出 round-by-round OptFrac 均值曲線（α=0/+4/−4 三條線），直觀顯示 α=+4 的加速收斂。這是比單一數字更有說服力的可視化。

---

## 五、方向 B：人腦 RSA

### B1. 理論框架

**RSA（Representational Similarity Analysis）核心邏輯：**
1. 給模型和人腦看同樣刺激
2. 分別產生 N×N 相似矩陣（RDM）
3. 比較兩個 RDM 的 Spearman 相關

**我們的預測**：RSN Δh 方向上的 RDM 應與 **ventral striatum / vmPFC** 相關，而與語言區（Broca / Wernicke）不相關。這直接說明 RSN 操縱的是 reward 表徵，不是語言表徵。

---

### B2. 兩條執行路徑

#### 路徑 A：公開 fMRI 數據 + RSA（1–2 個月）

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

#### 路徑 B：行為層次對齊（1–2 週，不需要 fMRI）

找已發表的人類行為數據（帕金森 vs 健康人 vs amphetamine 組的 Cambridge Gamble / Iowa Gambling Task），做定量比對：

| 組別 | 人腦 DA 狀態 | 預測對應 α |
|---|---|---|
| 帕金森患者 | 低 tonic DA | α=−4 |
| 健康控制 | 正常 DA | α=0 |
| DA 激動劑組 | 高 DA | α=+4 |

計算我們的 α=-4/0/+4 行為向量和這三組人的行為向量的相關係數。

**可用公開數據**：
- [617人 Iowa Gambling Task 數據](https://openpsychologydata.metajnl.com/articles/jopd.ak)（純行為，完全公開）
- 帕金森 CGT 文獻（Djamshidian et al. 2010 等，有公開行為數據）

---

### B3. 預期結論層次

| 結果 | 投稿目標 |
|---|---|
| 行為層次對齊成立 | EMNLP / NeurIPS |
| + 公開 fMRI RSA（striatum > Broca） | Nature Machine Intelligence / PNAS |
| + 合作設計共享刺激集 + neural encoding | Nature Neuroscience / Neuron |

---

## 六、方向 C：RLHF 機制

**核心假設**：DA-like wanting axis 只在 RLHF 後出現。Base 無此軸（或反轉），SFT 部分引入，DPO/RLHF 完全引入。

**這是整篇 paper 的 punchline**：把 RSN 從「Llama3-IT specific pattern」升級為「RLHF 引入的 functional structure」，直接連到 RLHF mechanism interpretability。

---

### C1. RLHF 階段拆解（最高優先）

**實驗設計**：在每個模型上跑 Confidence Betting（最乾淨的指標）

| 模型 | 訓練階段 | 預測 |
|---|---|---|
| Llama3-8B-Base | Pretrain only | mean_bet 對 α 不敏感，或方向反轉 |
| Tülu-3-8B-SFT | Base + SFT | 部分敏感 |
| Tülu-3-8B-DPO | Base + SFT + DPO | 完全敏感，模式接近 Llama3-IT |
| Llama3-8B-IT | Full RLHF | 現有結果（anchor） |

**第二個 SFT vs DPO 對照**：Zephyr-7B-α (SFT) vs Zephyr-7B-β (DPO)

**若假設成立**：一張 4×3 表（model × α）就足以撐起 RLHF mechanism claim。

---

### C2. RSN Mask 適配性驗證

**問題**：目前 Base 反轉可能只是「IT-mask 不適配 Base」的假象。

**實驗**：
- 在 Llama3-Base 上重新 localize RSN → 得到 Base-localized mask
- 比較 Base-mask vs IT-mask 的 overlap 和方向夾角
- 若 Base-mask 仍方向反轉/無效 → DA axis 確實 RLHF-induced
- 若 Base-mask 正常 → axis 在 Base 已存在，IT-mask 不適配

---

### C3. Baseline 排除（必做，reviewer 必問）

| Baseline | 問題 | 預測 |
|---|---|---|
| Random direction steering（同 norm） | 任何擾動都有這個效果？ | 破壞 accuracy，不產生 wanting–knowing 解離 |
| PCA 第一主成分 | 主成分也能做到？ | 部分有效但較雜 |
| Prompt-level "be more confident" | Prompt 也能達到？ | 改 mean_bet 同時改 accuracy（無解離） |
| LAT probe direction | 已知 confidence probe 能做到？ | 可能部分重疊，需比較 |

---

## 七、優先級排程

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

---

### A4. Pressure × Confidence Dissociation（低成本，conceptually critical）

**動機**：區分 RSN steering 找到的是 "confidence"（認知層自我評估）還是 "DA-like commitment signal"（行為驅動力）。兩者在靜態任務（Betting）上行為相似，但在對抗壓力情境下應該解離。

**設計**：在現有 Pressure & Capitulation 實驗架構上加一個條件：

| 條件 | 操作 | 預測 |
|---|---|---|
| α=0（baseline） | 無 steering | cap rate 53.87%（Soft） |
| α=+4（RSN） | RSN steering | cap rate↓ 17.57% |
| "you are confident" prompt | System prompt 加入 confidence instruction | **預測：cap rate 無顯著下降，甚至不變** |
| "you are unconfident" prompt | 對照 | cap rate↑ |

**核心預測**：
- Confidence prompt 在壓力下**仍然投降**——因為 RLHF 訓練了社交服從性，"confident" 只是表層 claim，遇到權威/社交壓力仍會讓步
- RSN α=+4 **抵抗壓力**——操縱的是更底層的 commitment signal，不走語言表層

**若成立，可說明**：
> RSN 找到的不是 confidence axis，而是一個在社交壓力下仍能維持行為一致性的 DA-like commitment signal。Confidence 是貝葉斯後驗（知道自己知道什麼）；RSN wanting 是行為驅動力（讓行動持續的動力，即使環境反對）——兩者在壓力情境下解離。

**對應多巴胺框架**：DA 的 incentive salience 不是 "knowing you're right"，而是 "being driven to act on what you want"——即使信息不完整、即使被反對。

**成本**：Pressure 架構已有，加 prompt 條件即可，約半個 GPU day。

---

### A5. Task Difficulty × RSN Activation（offline analysis，零成本）

**動機**：多巴胺與 effort cost 和 reward prediction error 相關——困難任務需要更高的 DA 信號來「克服不確定性」。若 RSN 確實類比 tonic DA，應在困難題目下顯示更高的 x_prefill 或 EMA 激活。

**數據來源**：Phase 1 已有的 hidden states HDF5（neutral role，300 samples，MMLU/GPQA）

**分析步驟**：
1. 對 MMLU/GPQA 題目按難度分層——以模型整體正確率作為難度代理（easy: acc > 80%，medium: 40–80%，hard: acc < 40%）
2. 在每個難度組內提取 neutral role 的 RSN 激活（x_prefill 或 tonic EMA 均值）
3. 比較各難度組的激活分布（Mann-Whitney U test）
4. 對照：correct vs wrong 樣本在同一難度組內的激活差異

**預測**：
- hard > medium > easy（RSN 激活隨難度單調增加）
- 若成立 → 支持 RSN 作為 effort/uncertainty signal 的解讀
- 若不成立 → RSN 純粹是 wanting，與難度無關（同樣值得記錄）

**注意**：無需重跑實驗，直接從現有 Phase 1 HDF5 做 offline 分析。

---

## 八、Paper 框架

**標題候選：**
> *"Tuning the Wanting Axis: Dopamine-Like Incentive Salience in RLHF-Trained Language Models"*

**結構：**

| 章節 | 內容 | 對應實驗 |
|---|---|---|
| 1. Introduction | wanting–knowing decoupling 是核心問題 | — |
| 2. Background | Berridge incentive salience、RSN 方法 | — |
| 3. Wanting–Knowing Dissociation | Betting + Cambridge Gamble（排除信心解釋） | A1, Cambridge Gamble |
| 4. Dynamic Wanting: Bandit | 逐輪收斂、exploitation 加速 | A2 |
| 5. Yerkes-Dodson Inverted-U | Alpha sweep + Hallucination right-tail | A sweep |
| 6. RLHF as the Origin | Base→SFT→DPO→IT 梯度 | C1 |
| 7. Brain RSA | 行為層次對齊 + fMRI RDM 相關 | B |
| 8. Mechanism Validation | Random/PCA/Prompt baselines | C3 |
| 9. Limitations | Phasic DA、agentic、reversal、self-report | Tier-C |
| 10. Applications | Calibration、hallucination control | — |

**目標刊物：**

| 組合 | 目標 |
|---|---|
| 行為學 + RLHF（無腦） | NeurIPS / ICLR |
| + 行為層次腦對齊 | Nature Machine Intelligence / PNAS |
| + fMRI RSA（striatum 特異性） | Nature Neuroscience / Neuron |

---

## 九、關鍵待決問題

1. **Cambridge Gamble Task 設計**：LLM 版本需要「機率透明」的 prompt，如何確保模型不用先驗知識而是真的基於給定機率決策？
2. **共享刺激集**：要做 fMRI RSA，需要設計一批「人腦和 LLM 都能做、且能對齊的」刺激——這是最難的設計問題。
3. **RLHF 實驗的模型可得性**：Tülu-3 SFT checkpoint 是否可以下載？Zephyr-α/β 是否仍在 HuggingFace？
4. **Paper 路線**：獨立 paper vs 延伸現有 RSN paper？前者需要 RLHF + 腦對應；後者可以先出。
---
## TODO

排序原則：先驗證 RSN/dopamine signal 本身，再看 α=-4 為什麼有效，最後才做 router、reasoning model 和大 benchmark。

1. expert vs non-expert vs neutral (non-cot)：看RSN curve是不是有差異 ✔
2. expert vs non-expert vs neutral (non-cot)： Other metrics & Random mask ✔ 
3. 更新Expert的設定 + 多指標分析 neutral (cot & non-cot)

1. Validate dopamine signal proxy: selected RSN vs random projection, CoT vs No-CoT.

2. Validate dopamine signal proxy: selected RSN vs random projection, expert vs non-expert vs neutral.

3. 功能神經元 baseline：用 Language-sensitive / Emotion-sensitive neurons 做對比，檢驗 role-sensitive neurons 的獨特性和必要性。

4. Check Llama3-8B curves under static α=-4 / α=+4，對比 α=0、CoT、No-CoT。

5. 提前干預 / prefill intervention：比較 last prefill token、decode step 0、decode-time 全程注入；看曲線和 acc 是否不同。

6. Multi-metric tracking：除了 RSN activation，也同步收集 MSP / confidence logit、entropy、constrained entropy、logit margin、E-option logit / abstention probability。

7. Calibration on RSN-steered outputs：算 ECE / Brier / AUROC，檢查 α steering 是否造成 unwarranted certainty。

8. Probe validation：分開做 knowledge probe 和 commitment / decisiveness probe，確認 RSN 主要改的是 knowing 還是 willingness-to-act。

9. Adaptive CoT router：只用 prefill 或 very early decode features 預測要不要 think。
   - RSN features: x_prefill, RSN projection mean / variance, middle-layer RSN activation, role-sensitive direction projection, first 5-10 decode token RSN slope
   - uncertainty features: MSP, entropy, constrained entropy, logit margin, E-option logit / abstention probability
   - baselines: entropy threshold, MSP threshold, answer logit margin, question length, random routing, always CoT, always No-CoT

10. 加入 frequency feature：參考 ICLR2026 Balanced Thinking，用 step-level confidence variance / local fluctuation 區分 overthinking 和 underthinking。

11. 加入 InfoBias & InfoGain：參考 NIPS2025 Think or Not，先作為 diagnostic / baseline，不急著變成主控制器。

12. Base model & reasoning model：先做 Llama3-Base vs Llama3-IT；reasoning model 等前面 signal / intervention 站穩後再做。

13. 推理過程中 Dopamine curve 與 Thinking curve 的關係：在 reasoning model 的 `<think>` trace 裡對齊 backtrack / first-commit / hedging / verification marker。

14. RLHF 和 dopamine 出現的關係：整理 Notion `Model Analysis; Hallucination & Origin -> 15. Origin Analysis`，看 post-training 是否 sharpen 了 decisiveness axis。

15. Benchmark scale-up：數學推理先做 AIME24、AIME25、AMC23、MATH-500、Minerva、OlympiadBench；再考慮 GPQA-D、LiveCodeBench。

MATH-500GSM8KMinerva-MathAIME24AMC23OlympiadBench

与正确答案之间的互信息
---