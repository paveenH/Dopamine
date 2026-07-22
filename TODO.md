# TODO
---


## 1. Active TODO
0. 直接看Cot在neurons上的改变 copy过去的结果
1. 测一下焦虑和心理学量表

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

