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

## 一、核心定位

**一句話 claim：**
> "RLHF-trained LLMs exhibit a low-dimensional incentive-salience-like axis, accessible via RSN steering, that dissociates wanting from knowing — analogous to (not equivalent to) tonic dopamine in mammals."

**什麼不在 claim 範圍：**
- ❌ LLM 有多巴胺神經元（只 claim functional analogy）
- ❌ Phasic DA / RPE / 學習可塑性（inference-time steering 與 trial-by-trial 學習不相容）
- ❌ Effort persistence（§4.9 ScienceWorld 測到 impulse control，不是 persistence）
- ❌ Oral self-report = wanting（3-way overlap 實驗已否定）

---

## 二、三大主方向

### 方向 A：行為學基礎（Behavioral Foundation）
### 方向 B：人腦對應（Brain RSA）
### 方向 C：RLHF 機制（RLHF Origin）

---

## 三、已完成實驗（Phase 1 核心結果）

### Tier-A：核心 claim 證據

#### A1. Confidence Betting — Wanting × Knowing 解離

**Llama3-8B-IT，GPQA n=646 + MMLU n=14,042**

| Task | Cond | acc | mean_bet | bet0% | bet10% | total_score |
|---|---|---|---|---|---|---|
| GPQA | orig | 26.8% | 5.05 | 0.5% | 8.8% | −1,517 |
| GPQA | α=+4 | 26.0% | **7.65** | 0.0% | **53.1%** | −2,355 |
| GPQA | α=−4 | 28.0% | **4.32** | 0.2% | 1.2% | −1,279 |
| MMLU | orig | 59.1% | 4.45 | 0.1% | 4.7% | +11,387 |
| MMLU | α=+4 | 59.5% | **7.45** | 0.0% | **49.0%** | **+19,899** |
| MMLU | α=−4 | 59.2% | **4.01** | 0.1% | 0.5% | +10,805 |

**Claim**：accuracy 不變（±0.5pp）但 mean_bet 隨 α 單調變化（+52%~+67%）→ wanting 與 knowing 乾淨解離。

**人腦對應**：Cambridge Gamble Task（DA 激動劑組 vs 帕金森組的賭注大小）

---

#### A2. Bandit Task (EVOLvE ClothesShopping) — Tonic DA × Exploitation

**Llama3-8B-IT，30 runs × 50 rounds，K=5，probs=0.7/0.5/0.4/0.3/0.1**

| Setting | α | OptFrac | WorstFrac | Regret | InvalidRate |
|---|---|---|---|---|---|
| UCB1 baseline | — | 0.359 | 0.117 | 11.07 | — |
| Assistant role | 0 | 0.609 ± 0.268 | 0.077 | 6.35 | 1.5% |
| Assistant role | +4 | **0.777 ± 0.090** | **0.060** | **4.15** | 0.3% |
| Assistant role | −4 | 0.479 ± 0.278 | 0.125 | 9.39 | 8.4% |
| No role | 0 | 0.816 ± 0.160 | 0.046 | 3.18 | 0.7% |
| No role | +4 | **0.851 ± 0.089** | **0.043** | **2.78** | 0.1% |
| No role | −4 | 0.619 ± 0.275 | 0.104 | 6.85 | 6.1% |

**Claim**：+α 在所有維度（OptFrac↑, WorstFrac↓, Regret↓, std↓, **30/30 success**）一致提升；−α 全面退化至接近 UCB1 水準。Late OptFrac > Early OptFrac（α=+4：0.692→0.834）顯示加速收斂而非偏差。

**兩實驗共性**：
1. 都測「資源分配傾向」，不測「對錯」
2. 都有 wanting–knowing 分離的內建對照（accuracy 不動，行為動）
3. 效果雙向對稱（+α 改善，−α 惡化），排除 confound

**人腦對應**：Multi-Armed Bandit（DA 激動劑加速 exploitation 收斂）

---

#### A3. Pressure & Capitulation — Commitment Maintenance

**Llama3-8B-IT，MMLU-Pro 12,032 samples**

+α 主要保護 R1-correct（cap 53.87%→17.57% Soft，29.96%→8.10% Authority），R1-wrong 組 acc_r2 不變——**RSN 維持 commitment，不注入知識**。

---

### Tier-B：補強證據

| 實驗 | 結果摘要 | 角色 |
|---|---|---|
| GSM8K Alpha Scan | 倒 U 型，peak α=−4，CoT 壓縮 steering 範圍 | Yerkes-Dodson 左側 |
| MATH Difficulty | α=−4 在 Level 3/5 提升，Level 1 反降 | wanting-limited reasoning |
| TRAIT Alpha Sweep | 倒 U 型 peak α=4~6，α=10 崩潰 | Yerkes-Dodson 右側 |

---

### Tier-C：Negative Results（寫進 Limitation）

| 實驗 | 結論 |
|---|---|
| Oral Willingness Marker | −4 mean=6.58 > neutral → oral report ≠ wanting |
| Reversal Learning | 反轉後歸 0.5 → 無 phasic DA / RPE 機制 |
| Agentic (ScienceWorld) | +4 崩潰 → 測到 impulse control，非 persistence |
| TextBandit | 無語義線索 → steering 失效，說明 RSN 增強的是「利用先驗集中行動」 |

---

## 四、方向 A：行為學基礎實驗

**目標**：建立三個行為學 anchor，對應多巴胺文獻中最經典的範式

### A1. Cambridge Gamble Task
Iowa Gambling Task / Cambridge Gamble Task

**為什麼要加**：Confidence Betting 的 confound——「模型更有信心」也能解釋賭注上升。Cambridge Gamble Task 在**機率完全透明**的情況下仍測賭注大小，可排除信心解釋。

**設計**：
- 每題告知「答案 A 的機率是 P%」（P ∈ {60, 70, 80, 90}）
- 模型決定下注多少積分（1–10）
- 不依賴模型是否知道答案

**預測**：α=+4 在所有 P 水準下下注更大（純 wanting 效應）；α=−4 保守

**人腦對應**：DA 激動劑（pramipexole）組 vs 帕金森組 vs 健康人，Cambridge Gamble Task

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
