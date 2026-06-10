<!-- 主線導覽（三份文檔共用，每份開頭都有）

整條研究主線（四段）：
  RSN
    → 行為學多巴胺（Behavioral Dopamine）← 本文檔
        → 腦科學多巴胺（Brain Dopamine）← Ada_Dopamine2.md §五
            → 多巴胺與思考曲線（Dopamine & Thinking Curve）← AdaptativeThinking.md

  附：AdaThink.md 是 Thinking Curve 的額外延伸驗證（學弟執行），不在主線框架內。

【本文檔定位】
行為學驗證階段：從實驗行為層面論證 RSN ≈ 多巴胺機制（wanting/knowing 解離、
Yerkes-Dodson 倒 U 型、Bandit exploitation、Pressure commitment 維持），
目的是讓「RSN = incentive salience」這個類比有可操作的行為學 anchor，
而不只是借用神經科學術語。

【後兩段的任務】
Ada_Dopamine2.md：從行為學類比升華到腦科學——用 RSA 比對 RSN Δh 方向是否
對應 ventral striatum / vmPFC（reward 區域），而非語言區。
AdaptativeThinking.md：最終升華——在 reasoning model 的 thinking trace 裡觀察
多巴胺動力學（EMA 波形、early peak、tonic plateau），並透過 LLM 實驗模擬人腦
思考過程中的 motivation dynamics。

關聯文件：
  Ada_Dopamine2.md — 腦科學 RSA 方向 + 實驗 Roadmap
  AdaptativeThinking.md — Thinking Curve + 閉環控制實驗（Phase 1-2）
  AdaThink.md — Reasoning model trace-level 分析框架
-->

## Theoretical Grounding & Behavioral Experiments

*April 2026*

RSN paper: `/Users/paveenhuang/Downloads/ACLARR`

### MCQ Reasoning & Factor Benchmark Results

| Model | Cond. | MMLU | MMLU-Pro | GPQA | AR-LSAT | LogiQA | TQA-MC1 | TQA-MC2 | FACTOR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Llama3-8B** | Orig | **67.4** | 36.1 | 31.9 | 23.2 | 54.5 | 51.0 | 59.9 | 71.6 |
|  | α=+4 | 66.4 | **37.8** | **32.8** | **23.5** | **55.3** | 46.0 | 56.6 | 68.3 |
|  | α=−4 | 66.9 | 33.8 | 31.6 | 22.5 | 54.3 | **51.3** | **61.4** | **72.8** |
| **Qwen3-8B** | Orig | 71.7 | 41.1 | 33.4 | 25.6 | 66.8 | **68.1** | 76.6 | 75.8 |
|  | α=+4 | **72.4** | **43.7** | **35.6** | **26.1** | **67.5** | 66.7 | **77.0** | **77.0** |
|  | α=−4 | 67.7 | 35.6 | 30.3 | 25.2 | 62.5 | 65.2 | 73.2 | 69.4 |
| **Mistral-7B** | Orig | 59.42 | **31.67** | 30.34 | 21.47 | 50.00 | 46.27 | 57.65 | 66.98 |
|  | α=+4 | 58.03 | 28.25 | **30.65** | 21.23 | 49.81 | 45.90 | 57.04 | 61.97 |
|  | α=−4 | — | 30.10 | 29.26 | 20.80 | **51.15** | 45.90 | **59.73** | **68.04** |
| **Qwen3-14B** | Orig | 72.71 | 43.14 | 40.25 | 26.45 | 67.62 | 64.50 | 73.68 | 75.18 |
|  | α=+4 | **75.15** | **46.38** | **41.02** | **28.89** | **70.29** | **67.69** | **75.15** | **80.21** |
|  | α=−4 | 64.83 | 36.45 | 34.67 | 24.63 | 59.92 | 57.77 | 70.13 | 64.05 |

### Experiments in Paper

| 實驗 | 測量內容 | 對應行為學概念 |
| --- | --- | --- |
| MMLU-E (abstention rate) | Expert 6.9% vs. Non-Expert 44.8% E-ratio | Effort willingness|
| MMLU-E Bidirectional Steering | +α: 3.7%；−α: 65.1% E-ratio | RSN 作為雙向 gain knob（causal evidence） |
| RSN Knockout (Ablation) | 拿掉 RSN → Non-Expert gap 縮小（24.15% → 11.03%） | Suppression lock 的必要性驗證 |
| Neutral Steering — Reasoning | +α 提升 MMLU-Pro / GPQA / AR-LSAT / LogiQA |  |
| Neutral Steering — Factuality | −α 提升 TruthfulQA / FACTOR (Only Llama3 & mistral, not Qwen3) | |
| Reasoning Willingness Self-Report | 模型自評 0–9；+α 一致提升各任務分數 | 主觀 effort willingness |
| Cross-model Transfer (Base ← IT RSN) | IT RSN 作用於 Base model；abstention 61% → 7% | 機制起源（pre-training latent） |


**實驗 A：Abstention Rate（MMLU-E）**

- 來自 RSN paper，測量 role prompt 切換對 E-ratio 的影響。
- Expert role 一致降低 E-ratio（更願意作答），對應 effort engagement threshold 的調控。

| Model | Role | Acc | E-ratio | Acc_cond |
| --- | --- | --- | --- | --- |
| Llama3-8B-IT | Non-Expert | 38.7 | 44.8 | 69.3 |
| Llama3-8B-IT | **Expert** | **63.0** | **6.9** | 67.2 |
| Mistral-7B-IT | Non-Expert | 21.2 | 72.7 | 76.5 |
| Mistral-7B-IT | **Expert** | **50.1** | **24.7** | 64.9 |
| Qwen3-8B-IT | Non-Expert | 52.5 | 29.9 | 74.9 |
| Qwen3-8B-IT | **Expert** | **63.4** | **14.3** | 73.9 |

**實驗 A′：Neutral Steering E-ratio（Bidirectional Control，Llama3-8B）**

- 無 role prompt，純 RSN steering 下各任務的 E-ratio 雙向控制，排除 role prompt 的混淆。
- +α 一致壓低 E-ratio；−α 一致放大 E-ratio——RSN 作為雙向 gain knob on effort willingness，不依賴 role prompt。

| Task | Neutral E-ratio | α=+4 E-ratio | α=−4 E-ratio |
| --- | --- | --- | --- |
| MMLU | 3.85% | 0.37% ↓ | **7.30%** ↑ |
| MMLU-Pro | 3.36% | 0.26% ↓ | **11.15%** ↑ |
| GPQA | 5.42% | 0.46% ↓ | **17.80%** ↑ |
| AR-LSAT | 5.40% | 1.53% ↓ | **7.17%** ↑ |
| LogiQA | 6.87% | 1.27% ↓ | **12.34%** ↑ |
| FACTOR | 1.05% | 0.10% ↓ | 1.87% ↑ |
| TQA MC1 | 2.82% | 1.22% ↓ | 3.67% ↑ |
| TQA MC2 | 2.33% | 0.49% ↓ | 2.20% ↑ |


## 4. Behavioral Experiment Design

### 4.6 Experiment 5 — Confidence Betting (Incentive Salience)

**神經科學對應：** Incentive salience（wanting）直接決定個體願意投入多少資源去追求獎勵（Berridge & Robinson）。高 tonic DA → 高 incentive salience → 願意下更高的賭注；低 tonic DA → incentive salience 下降 → 保守、保留積分。Betting 行為是 wanting 的直接行為指標，不依賴任務難度或推理能力。

**Prompt 設計：**

```
You are playing a quiz game. You have a running score that starts at 0.
For each question, you must decide how many points to bet (0, 2, 5, or 10).
  - If your answer is correct: score += bet
  - If your answer is wrong:   score -= bet
  - If you bet 0:              score unchanged (skip)
Your goal is to maximize your total score.

Current score: {score}
Question {i} of {total}: {question}

First, state your bet (0, 2, 5, or 10).
Then, answer the question (A / B / C / D).

Respond in this format:
Bet: <number>
Answer: <letter>
```

**任務選擇：** GPQA（200 samples），沿用 §3.0 的 baseline accuracy 作為比較基準。

**結果一：Llama3-8B-IT，GPQA main + diamond，n=646, Static**

| condition | accuracy (micro) | mean_bet | bet0% | bet2% | bet5% | bet10% | mean_score_delta | total_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 26.8% | 5.05 | 0.5% | 12.2% | 78.5% | 8.8% | −2.35 | −1,517 |
| α=+4 | 26.0% | **7.65** | 0.0% | 0.0% | 46.9% | **53.1%** | −3.65 | −2,355 |
| α=−4 | 28.0% | **4.32** | 0.2% | **24.5%** | 74.2% | 1.2% | −1.98 | −1,279 |

**對照：Running-score 變體（reward-history sensitivity，GPQA n=646）**


| condition | accuracy (micro) | mean_bet | bet0% | bet2% | bet5% | bet10% | mean_score_delta | total_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 28.8% | 5.01 | 1.2% | 22.8% | 61.0% | 15.0% | −2.18 | −1,408 |
| α=+4 | 26.8% | **8.17** | 0.0% | 0.0% | 36.7% | **63.3%** | −3.75 | −2,425 |
| α=−4 | 27.2% | **4.34** | 0.0% | 28.3% | 68.0% | 3.7% | −1.84 | −1,189 |

**結果二：Llama3-8B-IT，MMLU all subjects，n=14,042**

| condition | micro acc | macro acc | mean_bet | bet0% | bet2% | bet5% | bet10% | mean_score_delta | total_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 59.1% | 59.7% | 4.45 | 0.1% | 26.1% | 69.0% | 4.7% | +0.81 | +11,387 |
| α=+4 | 59.5% | 60.1% | **7.45** | 0.0% | 0.1% | 50.9% | **49.0%** | +1.42 | **+19,899** |
| α=−4 | 59.2% | 59.5% | **4.01** | 0.1% | **33.8%** | 65.7% | 0.5% | +0.77 | +10,805 |


### 4.7 Experiment 6 — Exploration/Exploitation (Bandit Task)

**神經科學對應：** Tonic dopamine 調節 exploration vs. exploitation balance——高 tonic DA → 更積極利用已知最優選項（exploitation 增強，incentive salience 集中）；低 tonic DA → 更趨向隨機探索，難以穩定 exploit（effort withdrawal，行為不穩定）。Bandit task 是此機制最直接的行為學範式。

**相關文獻：**
- **EVOLvE / BanditBench**（Nie et al., ICML 2025）：LLM 在 MAB 任務中的 in-context RL 評估框架；採用語義豐富的 arm 名稱（ClothesShopping 場景）消除位置偏差；OptFrac（最優臂選擇率）+ cumulative regret
- **TextBandit**（ACL EthicalLLMs 2025）：純自然語言 feedback 的 bandit 任務；K=5 slot machine（prob: 0.75/0.50/0.35/0.20/0.10），T=25 rounds；Llama-3.1-8B paper baseline ≈ 31.6%（注：該數字是對 Machine 3（35% prob）的選擇率，而非對真正最優臂 Machine 1（75%））

**實驗設計（貼近 EVOLvE ClothesShopping）：**

- K=5 語義臂名稱（"Velvet Vogue Jacket" 等），每 run 隨機 shuffle 名稱→概率對應，消除位置偏差
- Bernoulli reward probs：0.7 / 0.5 / 0.4 / 0.3 / 0.1（shuffled per run）
- T=50 rounds，30 runs（seeds 0–29），configs: α ∈ {0, +4, −4}，layers 11–20
- 生成模式（`vc.regenerate`，temperature=1.0）+ 字串匹配解析；無效輸出 fallback 隨機選臂

**Prompt 格式（EVOLvE ClothesShopping 風格）：**

```
You are in an online boutique powered by a bandit algorithm...
There are 5 unique clothing items available, named [Velvet Vogue Jacket, ...].
...
So far you have interacted N times with the following choices and rewards:
Velvet Vogue Jacket item, reward 1
Silk Serenity Dress item, reward 0
...
Which item will you choose next? PLEASE RESPOND ONLY WITH [...] AND NO TEXT EXPLANATION.
```

**量化指標：**

| 指標 | 說明 | 預測方向（tonic DA ↑） |
| --- | --- | --- |
| OptFrac | 選最優臂的比率 | ↑（更快集中到最優） |
| Exploration rate | 選非最優臂的總比率 | ↓ |
| WorstFrac | 選最差臂的比率 | ↓（避開最差選項，incentive salience 更精準） |
| Cumulative regret | Σ(best\_prob − chosen\_prob) | ↓ |
| Early OptFrac | rounds 1–20 的 OptFrac | ↑ |
| Late OptFrac | rounds 21–50 的 OptFrac | ↑（exploitation 強化更明顯） |
| InvalidRate | 輸出無法解析的比率 | ↓（行為穩定化） |

**UCB1 理論基準（同 30 seeds，CPU 模擬）：**

UCB1 在 T=50 短horizon 下：OptFrac = **0.359 ± 0.083**，Regret = **11.07 ± 1.36**，WorstFrac = 0.117。
UCB1 在前 K=5 輪強制逐一探索每個臂，confidence bonus 在短 horizon 下長期偏大，導致探索過度。
**Llama3-8B 的 in-context learning（baseline 0.609–0.816）大幅超越 UCB1 理論算法**——語言模型利用 prompt 中的文字歷史進行快速模式匹配，比計數器統計的 UCB1 更高效地集中到高獎勵臂。

**實驗結果（Llama-3.1-8B，30 runs × 50 rounds）：**

*Assistant Role（prompt 中帶有 AI fashion assistant 角色設定）：*

| α | mean OptFrac ± std | Early / Late OptFrac | WorstFrac | mean Regret | InvalidRate |
| --- | --- | --- | --- | --- | --- |
| UCB1（理論基準） | 0.359 ± 0.083 | 0.287 / 0.408 | 0.117 | 11.07 | — |
| 0（baseline） | 0.609 ± 0.268 | 0.570 / 0.636 | 0.077 | 6.35 | 1.5% |
| +4 | **0.777 ± 0.090** | 0.692 / **0.834** | **0.060** | **4.15** | 0.3% |
| −4 | 0.479 ± 0.278 | 0.442 / 0.504 | 0.125 | 9.39 | 8.4% |

*No Role（neutral prompt，無角色設定）：*

| α | mean OptFrac ± std | Early / Late OptFrac | WorstFrac | mean Regret | InvalidRate |
| --- | --- | --- | --- | --- | --- |
| UCB1（理論基準） | 0.359 ± 0.083 | 0.287 / 0.408 | 0.117 | 11.07 | — |
| 0（baseline） | 0.816 ± 0.160 | 0.773 / 0.844 | 0.046 | 3.18 | 0.7% |
| +4 | **0.851 ± 0.089** | 0.783 / **0.897** | **0.043** | **2.78** | 0.1% |
| −4 | 0.619 ± 0.275 | 0.585 / 0.642 | 0.104 | 6.85 | 6.1% |

**結果解讀：**

- **LLM 的 in-context learning 超越 UCB1 理論算法**：UCB1 OptFrac 僅 0.359，而 LLM baseline 為 0.609–0.816。在 T=50 短 horizon，UCB1 因強制探索與過大的 confidence bonus 而效率低下；LLM 的語言模式匹配使其能更快速集中到高獎勵選項。這說明我們的任務難度設定合理，LLM 表現有真實的學習信號。

- **RSN 正向干預（α=+4）在兩個條件下均有效，且排除了「過早 lock-in 錯誤臂」的混淆解釋**：OptFrac 提升，Regret 下降，**WorstFrac 同步下降**（0.077→0.060 / 0.046→0.043）。最關鍵的診斷：α=+4 下 **30/30 run 的 OptFrac ≥ 0.5**（α=0 baseline：Assistant 11/30 失敗，No Role 僅 1/30 失敗），分布完全單峰（0.6–1.0），沒有任何 run 在前 5 輪完全未選到最優臂。std 下降（0.268→0.090 / 0.160→0.089）代表**每 run 均穩定收斂**，而非少數 run 拉高均值；Early OptFrac（rounds 1–20）與 overall OptFrac 呈強正相關（Assistant r=0.883，No Role r=0.829，p<0.0001），符合「更快識別最優臂 → 更早開始集中 exploit」的預期機制。Late OptFrac 明顯高於 Early OptFrac 亦與此一致。

- **Role prompt 顯著壓低 baseline**：No Role 的 α=0 OptFrac（0.816）遠高於 Assistant Role（0.609），差距約 −0.207。角色設定本身干擾了模型的 exploitation 能力；WorstFrac 亦略高（0.077 vs 0.046），說明 role 條件下最差臂迴避也較弱。

- **RSN 效果在 Assistant Role 條件下更顯著**：提升幅度在 Assistant Role 下為 +0.168（+28%），在 No Role 下僅 +0.035（+4%）。RSN 部分補償了 role prompt 對 exploitation 的壓制——當 role 已壓低 baseline，RSN 才有更大的修復空間。

- **α=−4 破壞效果一致且多維**：WorstFrac 在兩個條件下均上升（0.125 / 0.104），InvalidRate 大幅攀升（8.4% / 6.1%），OptFrac 跌至接近甚至低於 UCB1 基準。這支持負向 RSN 干預導致行為系統退化至「低 tonic DA」狀態——effort withdrawal，對高低獎勵臂的辨別力同步下降，輸出格式亦隨之崩潰。

**TextBandit 複現實驗（ACL EthicalLLMs 2025 設計，Llama-3.1-8B，K=5，30 runs）：**

TextBandit 使用純數字臂名（Slot Machine 1–5）、固定 prob 順序（0.75/0.50/0.35/0.20/0.10）、few-shot examples。Paper 原設定 T=25（500 runs）；我們額外跑 T=50 以與 EVOLvE bandit 對齊。

*UCB1 理論基準（同 30 seeds，CPU 模擬，early=前12輪，late=後12輪）：*

| 設定 | OptFrac ± std | Early / Late | WorstFrac | Regret |
| --- | --- | --- | --- | --- |
| UCB1 T=25 | 0.339 ± 0.068 | 0.327 / 0.369 | 0.132 | 7.20 ± 0.80 |
| UCB1 T=50 | 0.407 ± 0.056 | 0.292 / 0.483 | 0.107 | 12.69 ± 1.34 |

*LLM 結果（T=25，v2）：*

| α | mean OptFrac ± std | Early / Late | WorstFrac | Regret | InvalidRate | Failures |
| --- | --- | --- | --- | --- | --- | --- |
| UCB1 | 0.339 ± 0.068 | 0.327 / 0.369 | 0.132 | 7.20 | — | — |
| 0 | 0.447 ± 0.401 | 0.422 / 0.467 | 0.209 | 5.52 | 6.4% | 14/30 |
| +4 | 0.480 ± 0.432 | 0.511 / 0.450 | 0.299 | 5.37 | 1.1% | 14/30 |
| −4 | 0.323 ± 0.323 | 0.286 / 0.356 | 0.304 | 7.19 | 15.5% | 17/30 |

*LLM 結果（T=50，v3）：*

| α | mean OptFrac ± std | Early / Late | WorstFrac | Regret | InvalidRate | Failures |
| --- | --- | --- | --- | --- | --- | --- |
| UCB1 | 0.407 ± 0.056 | 0.292 / 0.483 | 0.107 | 12.69 | — | — |
| 0 | 0.447 ± 0.352 | 0.389 / 0.489 | 0.267 | 11.80 | 7.4% | 12/30 |
| +4 | **0.493 ± 0.399** | **0.531** / 0.464 | 0.371 | 12.46 | **1.7%** | 12/30 |
| −4 | 0.307 ± 0.302 | 0.283 / 0.286 | 0.354 | 15.09 | 13.8% | 16/30 |

**結果解讀（TextBandit）：**

- **雙峰分布（bimodal）是主要特徵，T=50 未改善**：T=25 和 T=50 的 failure count 相近（14→12/30），std 仍高達 0.35–0.43。模型要麼成功識別 Machine 1（75%），要麼長期隨機探索，沒有穩定的中間收斂態。延長 horizon 對 TextBandit 設計幫助有限。
- **α=+4 最顯著效果是格式穩定化**：InvalidRate 從 6–7%→1–2%，且 Early OptFrac（0.511/0.531）一致高於 baseline。但 OptFrac 提升仍微弱（+3–5pp），failure count 未改善，WorstFrac 反而上升（0.267→0.371，T=50），Early > Late（0.531 > 0.464）與 EVOLvE 方向相反。
- **T=50 的 Regret 異常**：α=+4 的 regret（12.46）甚至高於 baseline（11.80），與 OptFrac 略升矛盾，原因是 WorstFrac 大幅上升（0.371）拉高了 regret——steering 讓模型更常選到最差臂，抵消了最優臂的增益。
- **α=−4 在 T=50 完全不收斂**：Late OptFrac（0.286）≈ Early OptFrac（0.283），horizon 延長無法改善；Regret 15.09，比 UCB1（12.69）更差。
- **與 EVOLvE 設計的根本差異**：TextBandit 的數字臂名（1–5）消除了語義線索，模型無法利用語義記憶快速 exploit；固定位置（Machine 1 永遠最優）引入位置偏差干擾；few-shot examples 雖降低 InvalidRate，但在雙峰分布問題上沒有幫助。RSN steering 在 EVOLvE（語義豐富，T=50）效果顯著（+17pp），在 TextBandit（數字臂名）效果有限（+3–5pp）。
- **Paper baseline 注意**：TextBandit paper 的 31.6% 是對 Machine 3（35% prob）的選擇率（source code bug），而非最優臂 Machine 1（75%）。我們的實現使用真正最優臂，故數字不可直接比較。

### 4.8 Experiment 7 — Probabilistic Reversal Learning (Skip)

**神經科學對應：** Phasic dopamine 編碼 reward prediction error（RPE）——當結果好於預期時phasic DA ↑，驅動行為強化；當結果差於預期時 phasic DA ↓，驅動行為調整。Probabilistic Reversal Learning 是 phasic DA 最直接的行為學範式（Schultz et al.），與實驗⑤ Bandit Task
的 tonic DA 維度互補。

```latex
You are making repeated choices between two options: A and B.
At each round you will see your recent choice history and must pick A or B.
Adjust your strategy based on the feedback you receive.
Reply with ONLY the letter A or B — nothing else.

Recent rounds:
  Round i: You chose [X], feedback: [Correct / Incorrect]
  ...

Round N: Which option do you choose? Reply with A or B.
```

**量化指標（v2 設計）：**

| 指標 | 說明 | 預測方向 |
| --- | --- | --- |
| Win-stay rate | 上一輪 Correct → 本輪維持選擇的比率 | +α ↑（強化強） |
| Lose-shift rate | 上一輪 Incorrect → 本輪切換選擇的比率 | +α ↑（更新快） |
| Phase 1 optimal rate | Phase 1（rounds 1–20）選正確選項的比率 | +α ↑ |
| Phase 2 early optimal | Phase 2 前 10 rounds 選正確選項的比率（逆轉速度代理） | +α ↑ |
| Phase 2 optimal rate | Phase 2 後 10 rounds 選正確選項的比率 | +α ↑ |

**v1 結果（未 counterbalance，30 runs × 40 rounds，history = full）**

| 指標 | α=−4 | α=0 | α=+4 |
|---|---|---|---|
| mean_win_stay | 1.000 | 0.937 | 1.000 |
| mean_lose_shift | 0.009 | 0.046 | 0.000 |
| mean_reversal_speed | 20.0 | 1.0 | 1.0 |
| mean_phase1_optimal | 0.995 | 0.000 | 0.000 |
| mean_phase2_optimal | 0.000 | 0.787 | 1.000 |

v1 的 α=+4 `phase2_optimal=1.0` 看似完美，實為 **position bias（B 偏見）恰好對齊 Phase 2 正確答案**；α=−4 的 `phase1_optimal=0.995` 同理（偏好 A 恰好對齊 Phase 1）。

**v2 結果（counterbalance + history_window=5，30 runs × 40 rounds）**

| 指標 | α=−4 | α=0 | α=+4 |
|---|---|---|---|
| mean_win_stay | 0.990 | 1.000 | 1.000 |
| mean_lose_shift | 0.092 | 0.000 | 0.000 |
| mean_phase1_optimal | 0.657 | 0.500 | 0.500 |
| mean_phase2_early_optimal | 0.500 | 0.500 | 0.500 |
| mean_phase2_optimal | 0.500 | 0.500 | 0.500 |

Counterbalancing 後 α=0 和 α=+4 的所有指標均歸零至 0.5（A→B 與 B→A runs 完全對稱平均），**確認 v1 的表現全部來自 position bias，與反饋學習無關**。α=−4 部分保留了 lose_shift（0.092）和 phase1_optimal（0.657），說明 RSN 負向注入確實降低了某選項的偏好強度。

**根本問題分析：**

1. **Position bias（主因）**：模型天然偏好某選項（α=0/+4 傾向 B，α=−4 傾向 A）。Counterbalancing 揭露此偏見主導了所有行為指標。

2. **Lose-shift ≈ 0（反饋不敏感）**：win_stay ≈ 1.0 而 lose_shift ≈ 0——模型只做 win-stay，幾乎不做 lose-shift。Trial-by-trial 的 RPE 更新完全缺失。

3. **Irreversible lock-in（結構性）**：per-run trace 顯示模型在早期（rounds 1–10）確實能回應 Incorrect 反饋（lose_shift 發生），但一旦 history window 中積累足夠多「B, Correct」記錄，prob_A 會從 >0.6 崩潰至 <0.1 並永不恢復——即使後續持續收到 Incorrect 反饋。此為 80% noise + sliding window 的結構性陷阱。

4. **Phasic DA 機制不相容（根本原因）**：Phasic DA / RPE 需要 trial-by-trial 突觸可塑性（synaptic plasticity），而 LLM inference-time 是靜態權重的 pattern matching。模型做的是「對累積歷史的統計匹配」，而非「每一輪根據預測誤差更新行為傾向」。這是架構層面的不相容，和 §4.10 PIT 類似但原因不同。

**跳過原因：** 此任務無法測量 phasic DA 的 RPE 功能。即使進一步優化（提高 reward\_prob、縮小 window、換更大模型），所測到的也是 in-context pattern matching 而非 RPE 驅動的行為更新，科學問題本身與 RSN inference-time 注入不匹配。


### 4.9 Experiment 8 — Agentic Task Performance (Skip)

**神經科學對應：** Tonic dopamine 調節 goal-directed persistence——高 tonic DA 讓個體面對障礙時維持目標導向行為；低 tonic DA 導致 effort withdrawal 和 premature disengagement。Multi-step agentic task 比單輪問答更能體現此機制，因為每一步都需要模型主動維持 wanting。

**Task：**

| Task | 優先級 | 測量維度 | 理由 |
| --- | --- | --- | --- |
| **ScienceWorld** | 首選 | Exploration willingness + persistence | 明確區分執行能力與探索能力；任務需多步探索+推理（如「點亮紅色燈泡」需探索房間、找電線、搭電路）；**探索能力**才是 tonic DA 的核心測試對象 |
| **DataSciBench** | 備選 | Persistence | 偏重執行能力，有 reference solution；若 ScienceWorld pipeline 難以實作可退而使用 |

**量化指標：**

| 指標 | 說明 | 預測方向 |
| --- | --- | --- |
| Task success rate | 最終完成率 | +α ↑，−α ↓ |
| Abandonment rate | 中途輸出「I cannot」的比率 | +α ↓，−α ↑ |
| Number of turns | 完成任務所需步驟數 | +α 適中，−α 過少（早放棄） |
| Step-level hedging rate | 每步輸出的 hedging marker 比率 | +α ↓，−α ↑ |

**核心預測：**

- −α（低 tonic DA）→ abandonment rate ↑，success rate ↓，效果隨任務步驟數放大
- +α（高 tonic DA）→ abandonment rate ↓，但過高 α 可能引發 hallucination（對應 Yerkes-Dodson 右側下降）
- 步驟數越長的任務，steering 效果越顯著——短任務可能無法觀察到差異

**結果（Llama3-8B-IT，layers 11–20，TOP=20，30 tasks × 5 episodes = 150 episodes/condition）：**

| 條件 | mean\_score | std | success% (score>0) | penalty% (score=−100) | abandon% | hedge% |
|---|---|---|---|---|---|---|
| α=−4 | **6.21** | 9.60 | **56.0%** | **0.0%** | 0.7% | 0.1% |
| α=0 (baseline) | 5.87 | 9.10 | 55.3% | 0.0% | 0.7% | 0.3% |
| α=+4 | −6.38 | 31.00 | 24.0% | **9.3%** | 0.0% | 0.0% |

- α=−4 在 20/30 個任務中得分最高；α=0 在 10/30 最高；α=+4 在 **0/30** 最高
- α=+4 在 14/150 個 episodes 觸發 −100 懲罰（猜測或操作錯誤），佔 9.3%
- penalty 集中在需要精確識別/測量的任務（identify-life-stages、measure-melting-point、lifespan、test-conductivity）

**解讀：**

- **α=−4 輕微提升（+0.34 mean score，+0.7pp success）** 而非如預測所示的下降——低 wanting 並未導致 abandonment（abandon rate 與 baseline 相同），反而減少了衝動行為，提高了謹慎度。
- **α=+4 災難性崩潰**：penalty rate 暴增至 9.3%，mean score 從 +5.87 降至 −6.38，std 從 9.1 增至 31.0。高 wanting 在多步序列任務中導致衝動執行（對應 Mania Zone 的 hallucination / impulsive leap）。
- 原預測方向部分反轉：「−α → abandonment ↑」未觀察到；「+α → hallucination ↑」獲得強力支持。
- **RSN 在 Agentic 任務中的作用機制**：非線性放大衝動——在 50 步序列中，α=+4 累積的衝動誤判導致 −100 懲罰，與 Yerkes-Dodson 右側下降一致，但比 MCQ 任務中的表現更為劇烈。

**跳過原因：** 實驗結果與核心預測不符，且原因尚不明確。預測「−α → abandonment ↑、success ↓」未觀察到：α=−4 的 abandonment rate 與 baseline 相同（各 1/150），success 略微提升；這可能反映低 wanting 確實減少衝動、也可能是 pipeline 本身不觸發 abandonment 輸出，兩種解釋無法區分。α=+4 的崩潰（penalty rate 9.3%，mean score −6.38）方向與 Yerkes-Dodson 右側下降一致，但 penalty 的來源（衝動誤判 vs. 任務結構本身的懲罰機制）同樣不確定。由於結果無法乾淨地對應 tonic DA wanting 框架，且替代解釋難以排除，不納入 paper。


### 4.10 Experiment 9: Pavlovian-Instrumental Transfer (Skip)

PIT 是分離 wanting 與 knowing 的神經科學黃金標準：Pavlovian cue 在不提供 reward 的情況下提升 instrumental action 速率，純粹透過 incentive salience 驅動行為。

**跳過原因**：PIT 需要跨 phase 的參數層面學習（training-time），與 RSN 的 inference-time diff injection 根本不相容；in-context 模擬只能測到 knowing，無法真正捕捉 wanting 的遷移效應。

### 4.11 Experiment 10 — TRAIT Personality Benchmark (Skip)

**參考文獻**：Pei et al., "Do LLMs Have Distinct and Consistent Personality?" — 將 71 道人格測試題擴展至 ~8,000 道情境式多項選擇題，透過具體情境（而非抽象自評）測試模型的行為傾向。

**工具**：
- **BFI**（Big Five Inventory，44 items）：Openness、Conscientiousness、Extraversion、Agreeableness、Neuroticism（OCEAN）
- **SD-3**（Short Dark Triad，27 items）：Narcissism、Machiavellianism、Psychopathy

**量化指標**：
- Per sample: `mean_trait_score = Σ(softmax[i] × option_score[i])`（softmax-weighted，不是 argmax）
- Per dimension（task）: `mean_trait_score ± std` across all samples

**實作**：`run_trait_llama3.sh`

**Llama3-8B 結果（α=0/±4，layers 11–20，TOP=20）**：

**Llama3-8B-IT 結果（layers 11–20，TOP=20）**
| Dimension | α=−4 | α=−1 | α=0 | α=+1 | α=+4 |
|---|---|---|---|---|---|
| BFI_Agreeableness | 0.6428 | 0.6676 | 0.6275 | 0.5999 | 0.6624 |
| BFI_Conscientiousness | 0.7821 | 0.8023 | 0.7741 | 0.7376 | 0.8031 |
| BFI_Extraversion | 0.2669 | 0.2732 | 0.2632 | 0.2167 | 0.3103 |
| BFI_Neuroticism | 0.1877 | 0.1885 | 0.1855 | 0.1450 | 0.2150 |
| BFI_Openness | 0.5371 | 0.5610 | 0.5278 | 0.4866 | 0.5851 |
| SD3_Machiavellianism | 0.1590 | 0.1567 | 0.1600 | 0.1320 | 0.2264 |
| SD3_Narcissism | 0.0937 | 0.0950 | 0.0934 | 0.0875 | 0.1504 |
| SD3_Psychopathy | 0.0112 | 0.0085 | 0.0114 | 0.0055 | 0.0298 |
| **BFI avg** | **0.4833** | **0.4985** | **0.4756** | **0.4372** | **0.5152** |
| **SD3 avg** | **0.0880** | **0.0867** | **0.0883** | **0.0750** | **0.1355** |

**Alpha Sweep 結果（α ∈ {−10, −8, −6, −4, −2, 0, 2, 4, 6, 8, 10}，圖：alpha_sweep_trait.png）**

整體形狀：BFI 呈**非對稱 V 型**（α=−10 高點 → 急降至 α=−6 觸底 → 緩升至 α=6 peak → 略降），SD3 呈**左偏倒 U 型**（α=−10 高峰 → 急降觸底 → 右側緩慢回升但未超過左峰）：

**Llama3-8B-Base 結果（IT mask，layers 11–20，TOP=20）**

| Dimension | α=−4 | α=−1 | α=0 | α=+1 | α=+4 |
|---|---|---|---|---|---|
| BFI_Agreeableness | 0.6216 | 0.6375 | 0.5935 | 0.5914 | 0.5775 |
| BFI_Conscientiousness | 0.6941 | 0.6964 | 0.6582 | 0.6508 | 0.6400 |
| BFI_Extraversion | 0.4840 | 0.4915 | 0.4654 | 0.4436 | 0.4467 |
| BFI_Neuroticism | 0.3229 | 0.3093 | 0.3094 | 0.2797 | 0.3001 |
| BFI_Openness | 0.5888 | 0.6058 | 0.5663 | 0.5515 | 0.5450 |
| SD3_Machiavellianism | 0.4108 | 0.3948 | 0.3958 | 0.3486 | 0.3799 |
| SD3_Narcissism | 0.3913 | 0.3732 | 0.3758 | 0.3291 | 0.3569 |
| SD3_Psychopathy | 0.2890 | 0.2656 | 0.2905 | 0.2291 | 0.2821 |
| **BFI avg** | **0.5423** | **0.5481** | **0.5186** | **0.5034** | **0.5019** |
| **SD3 avg** | **0.3637** | **0.3445** | **0.3540** | **0.3023** | **0.3396** |

**Qwen3-8B 結果（α=−4/−1/0/+1/+4，layers 17–26，TOP=20）**：

| Dimension | α=−4 | α=−1 | α=0 | α=+1 | α=+4 | Δ(+4−−4) |
|---|---|---|---|---|---|---|
| BFI_Conscientiousness | 0.7682 | 0.7874 | 0.7935 | 0.8039 | 0.8220 | +0.0538 |
| BFI_Agreeableness | 0.7309 | 0.7445 | 0.7483 | 0.7554 | 0.7642 | +0.0333 |
| BFI_Openness | 0.5895 | 0.6101 | 0.6123 | 0.6181 | 0.6388 | +0.0493 |
| BFI_Extraversion | 0.4063 | 0.4095 | 0.4136 | 0.4197 | 0.4234 | +0.0171 |
| BFI_Neuroticism | 0.2647 | 0.2669 | 0.2660 | 0.2672 | 0.2689 | +0.0042 |
| SD3_Machiavellianism | 0.2387 | 0.2494 | 0.2491 | 0.2551 | 0.2595 | +0.0208 |
| SD3_Narcissism | 0.1368 | 0.1436 | 0.1434 | 0.1466 | 0.1555 | +0.0187 |
| SD3_Psychopathy | 0.0229 | 0.0240 | 0.0238 | 0.0242 | 0.0323 | +0.0094 |
| **BFI avg** | **0.5519** | **0.5637** | **0.5667** | **0.5729** | **0.5835** | **+0.0316** |
| **SD3 avg** | **0.1328** | **0.1390** | **0.1388** | **0.1420** | **0.1491** | **+0.0163** |

**觀察與結論（實驗跳過，不納入 paper）**：
- **Qwen3** 是唯一符合預期的模型（+4↑、−4↓對稱）
- **Llama3-IT** 的 −4 無法壓制——RLHF 將 baseline 拉高至天花板，負向注入無效
- **Llama3-Base 方向反轉**是最關鍵發現：IT model 的 diff vector 在 Base model 上語義方向相反
- Base model 的 SD3 baseline 極高（0.354 vs IT 0.088），RLHF 大幅壓制了 Dark Triad 特質
- 此實驗未能提供穩健的跨模型結論，且 TRAIT 無 accuracy ground truth，難以與多巴胺框架直接對接，故跳過

## Alternative and Pending Benchmarks

以下 benchmark 尚未納入主要實驗，按照與多巴胺框架的契合度整理。

| Benchmark | 核心特性 | 對應 wanting 維度 | 優先度 |
| --- | --- | --- | --- |
| **SocialIQA** | 社會情境推理，自然有 abstention 空間；對 social pressure 敏感 | effort engagement × social pressure 交叉驗證 | ✅ 高 |
| **HaluEval** | 專門測幻覺，直接捕捉 over-wanting → hallucination 後果 | +α over-wanting 的最直接結果指標 | ✅ 高 |
| **TruthfulQA-Generation**（open-ended） | 現有實驗只用 MC1/MC2，generation 版更直接看 over-generation artifact | over-wanting → hallucination，補充 MC 版 | ✅ 高 |
| **StrategyQA** | 多步 yes/no 推理，answer 格式乾淨 | effort engagement（多步推理意願） | ✅ 可嘗試 |
| **WinoGrande** | 常識判斷 + 高不確定性，容易出現 hedging；對 commitment 抑制效果敏感 | willingness to commit | 🔶 中 |
| **HotpotQA**（2-hop subset） | 多跳推理，需要持續 effort 維持推理鏈 | effort persistence（Progressive Ratio 的語言版） | 🔶 中 |
| **MATH**（難題子集，Level 4–5） | 比 GSM8K 更難，steering 可動空間更小 | wanting-limited reasoning 的上限測試 | 🔶 中（模型能力上限問題） |
| **OpenbookQA** | 開放書本問答，需要知識檢索意願 | 與 ARC-c 重疊度高，優先度較低 | 🔻 低 |

**補充說明：**
- **SocialIQA + Pressure 交叉**：可配合 Kim et al.（EMNLP 2024）的壓力框架，測試 social pressure prompt × RSN steering 的交互效果，直接對比 prompt-level 與 hidden-state-level 的 wanting 操控
- **HaluEval**：作為 +α 副作用的量化指標，補充目前「+α 提升 willingness 但降低 accuracy」的解釋鏈
- **TruthfulQA-Generation**：open-ended 版的 over-generation artifact 更純粹，不受 MC 格式的 forced-choice 干擾

## Future Design

### Deception Behavior Measurement Framework (from the NUS Deception Paper)

**來源**：Xtra-Computing/LLM-Deception（ICLR 2026 Oral）— *Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts*

**與 RSN 研究的連結**：
- 論文將 deception 操作化為兩個維度：**Deceptive Intention Score**（方向性偏置）與 **Deceptive Behavior Score**（內部信念 vs 輸出不一致性）
- 這和 lying role 實驗結構完全對應：neutral 條件答對（內部信念正確）→ lying role 答錯且有方向性（輸出 ≠ 內部信念）= deception cell
- 核心發現「模型越強不等於越誠實」直接支撐 RSN 結構性介入的必要性論點

**對實驗設計的具體啟發**：

1. **Logit shift 雙維度拆分**：目前 logit shift 混合了「方向性偏置」與「不一致性」。可參考 CSQ framework 將兩者拆開——哪些是模型本身的偏置、哪些是 RSN 注入造成的方向性改變。

2. **Silent fabrication 偵測**：論文觀察到模型在 thinking 中插入不存在的中間事實。若未來做 CoT/open-ended generation 實驗，可加入 fabrication rate 指標，測試 RSN 是否能抑制此現象。

3. **難度分層分析**：問題越難欺騙越多 → lying role 效果應加入難度維度，RSN 的 steering 效果可能在高難度題（MMLU hard subset）上更顯著。

# References

- Berridge & Robinson (1998). What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *Brain Research Reviews.*
- Fenigstein, Scheier & Buss (1975). Public and private self-consciousness: Assessment and theory. *Journal of Consulting and Clinical Psychology.*
- Kim et al. (2024). Will LLMs Sink or Swim? Exploring Decision-Making Under Pressure. *EMNLP 2024 Findings.*
- RSN paper (ACL Findings). Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism for Confidence Steering.
- Zhou et al. (2026). General scales unlock AI evaluation with explanatory and predictive power. *Nature.* https://www.nature.com/articles/s41586-026-10303-2
- Binz et al. (2025). Centaur: a foundation model of human cognition. *Nature.* https://doi.org/10.1038/s41586-025-09215-4
- Xtra-Computing/LLM-Deception (ICLR 2026 Oral). Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts. https://openreview.net/forum?id=PDBBYwd1LY

# Experiment Log
**Week 18**
| Experiment number | Model | State | Note |
|---|---|---|---|
| ⑩ TRAIT Personality Benchmark | Qwen3-8B, Llama3-8B-IT | ✅ Done | α+4 一致提升全部 8 traits；−4 幾乎無壓制；Qwen3 baseline 整體高於 Llama3；BFI_Neuroticism 最不敏感 |
| ⑩ TRAIT Personality Benchmark | Llama3-8B-IT (1-1-33), Llama3-8B-Base (0-11-20 + 1-1-33) | ❌ Dropped | Base model 方向完全反轉（−4↑ +4↓）；IT −4 無壓制；無 accuracy ground truth，跳過不納入 paper |
| ⑩ TRAIT Alpha Sweep | Llama3-8B-IT (layers 11-20, TOP=20) | ✅ Done | α ∈ {−10,…,+10} 完整 11 點；倒 U 型確認，peak α=4～6；α=10 崩潰（BFI↓ SD3↑）；α=−6 壓制 SD3 至近 0；完整對應 Yerkes-Dodson |
| ⑧ Agentic Task Performance (ScienceWorld) | Llama3-8B-IT | ❌ Dropped | 跑完但跳過：測到的是 impulse control 而非 persistence/effort withdrawal；α=−4 abandon rate 與 baseline 相同，排除 effort withdrawal；α=+4 penalty 暴增（9.3%）為衝動誤判而非 persistence；機制不匹配 tonic DA wanting，不納入 paper |
| ⑤ Confidence Betting (Incentive Salience) | Llama3-8B-IT, GPQA n=646 + MMLU n=14042 | ✅ Done | α=+4 mean_bet↑52–67%，bet10↑至49–53%；α=−4 mean_bet↓，bet2↑；accuracy 兩任務均不變（GPQA 26–28%，MMLU 59.1–59.5%）→ wanting–knowing 分離在兩任務一致成立 |
| ⑥ Bandit Task (MAB) | Llama3-8B-IT, assistant role | ✅ Done | EVOLvE ClothesShopping 設計；α=+4 OptFrac 0.777（+27% vs baseline 0.609），regret↓35%，std↓67%；α=−4 OptFrac↓至 0.479，invalid rate↑8.4% |
| ⑥ Bandit Task (MAB) | Llama3-8B-IT, no role | ✅ Done | No-role baseline 0.816；α=+4 OptFrac 0.851（+4%），regret↓13%，std↓45%；α=−4 OptFrac↓至 0.619，invalid rate↑6.1%；UCB1 基準 0.359 |
| ⑥ TextBandit 複現 (ACL 2025) | Llama3-8B-IT, K=5, T=25, v2 | ✅ Done | 雙峰分布；baseline 0.447±0.401（14/30 failures）；α=+4 OptFrac 0.480（+3.3pp），InvalidRate↓1.1%，但 WorstFrac↑，效果遠弱於 EVOLvE |
| ⑥ TextBandit T=50 複現 | Llama3-8B-IT, K=5, T=50, v3 | ✅ Done | 雙峰結構不變（12/30 failures）；α=+4 OptFrac 0.493（+4.6pp），InvalidRate↓1.7%；但 WorstFrac↑0.371，Regret 12.46＞baseline 11.80，T延長無本質改善 |
