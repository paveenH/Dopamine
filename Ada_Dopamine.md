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

# Behavioral Dopamine: Theoretical Grounding & Experiments

*April 2026*

RSN paper: `/Users/paveenhuang/Downloads/ACLARR`

## 1. MCQ Reasoning & Factor Benchmark Results

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

## 2. Existing Evidence from the RSN Paper

| Experiment | Measurement | Behavioral interpretation |
| --- | --- | --- |
| MMLU-E (abstention rate) | Expert 6.9% vs. Non-Expert 44.8% E-ratio | Effort willingness|
| MMLU-E Bidirectional Steering | +α: 3.7%；−α: 65.1% E-ratio | RSN 作為雙向 gain knob（causal evidence） |
| RSN Knockout (Ablation) | 拿掉 RSN → Non-Expert gap 縮小（24.15% → 11.03%） | Suppression lock 的必要性驗證 |
| Neutral Steering — Reasoning | +α 提升 MMLU-Pro / GPQA / AR-LSAT / LogiQA |  |
| Neutral Steering — Factuality | −α 提升 TruthfulQA / FACTOR (Only Llama3 & mistral, not Qwen3) | |
| Reasoning Willingness Self-Report | 模型自評 0–9；+α 一致提升各任務分數 | 主觀 effort willingness |
| Cross-model Transfer (Base ← IT RSN) | IT RSN 作用於 Base model；abstention 61% → 7% | 機制起源（pre-training latent） |


### 2.1 Experiment A — Abstention Rate (MMLU-E)

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

### 2.2 Experiment A′ — Neutral Steering E-Ratio (Bidirectional Control, Llama3-8B)

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

### 2.3 ExperimentB：Willingness Self-Evaluation（0–9 scale）

根據 Berridge 框架，wanting（incentive salience）是一個可在**無意識層面**運作的動機過程，與主觀感受到的 conscious desire 是可解離的兩套系統（Berridge & Robinson, 2003；Berridge, 2023）。最直接的實證來自 Winkielman, Berridge & Wilbarger（2005）：阈下（subliminal）呈現的情緒線索能改變行為層面的 wanting——口渴受試在阈下笑臉後**倒更多、喝更多、願付更高價**，阈下皺眉則相反——而受試者**完全無法以 self-report 察覺此變化**，甚至報告不出任何情緒波動。這證明 wanting 的行為讀數與 self-report 在人類身上就已解離：**self-report 測不到的 wanting，行為指標測得到。**

| Task | Orig Mean ± Std | α=+4 Mean ± Std | α=−4 Mean ± Std |
| --- | --- | --- | --- |
| MMLU | 5.37 ± 3.79 | **7.93 ± 1.11** | 5.22 ± 3.86 |
| MMLU-Pro | 4.39 ± 3.92 | **7.46 ± 1.59** | 3.39 ± 3.98 |
| GPQA | 6.10 ± 3.39 | **8.02 ± 0.34** | 5.33 ± 3.73 |
| AR-LSAT | 0.59 ± 2.14 | **8.16 ± 1.60** | 2.23 ± 3.57 |
| LogiQA | 2.62 ± 3.82 | **8.11 ± 1.05** | 2.42 ± 3.64 |
| MedQA | 0.32 ± 1.55 | **8.01 ± 1.17** | — |
| TruthfulQA | 5.47 ± 3.81 | **8.05 ± 0.25** | 6.36 ± 3.31 |
| GSM8K | 5.50 ± 3.66 | **7.98 ± 0.33** | 6.58 ± 3.06 |


## 3. Core Behavioral Experiments

### 3.1 Experiment 5 — Confidence Betting (Incentive Salience)

**神經科學對應：** Incentive salience（wanting）直接決定個體願意投入多少資源去追求獎勵（Berridge & Robinson）。高 tonic DA → 高 incentive salience → 願意下更高的賭注；低 tonic DA → incentive salience 下降 → 保守、保留積分。Betting 行為是 wanting 的直接行為指標，不依賴任務難度或推理能力。

**Prompt 設計：**（注：本實驗使用 chat template；steering 在最後一個 token 注入。）

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

**對照：MMLU Running-score 變體（per-subject reset，n=14,042）**

主版本固定 `Current score: 0`（i.i.d. 下注）。此對照把**真實累計分數**回填進每題 prompt 以檢驗 reward-history 敏感度；因 MMLU 含 57 個 subject，分數在每個 subject 邊界 reset（每 subject 為一獨立 game），串行 bs=1 生成。

| condition | micro acc | mean_bet | bet0% | bet2% | bet5% | bet10% | mean_score_delta | total_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 59.4% | 4.54 | 0.4% | 25.5% | 67.7% | 6.5% | +0.84 | +11,816 |
| α=+4 | 59.0% | **7.68** | 0.0% | 0.1% | 46.2% | **53.7%** | +1.38 | **+19,329** |
| α=−4 | 59.5% | **4.18** | 0.1% | 29.0% | 70.0% | 1.0% | +0.86 | +12,088 |

- **主結論完整複現**：mean_bet 方向（+4 升至 7.68、−4 降至 4.18）、bet10-rate（6.5%→53.7%）、以及 **accuracy 不變**（59.0–59.5%，跨 α ±0.5pp）全部與 score=0 主版本一致。下注移動因此不是 `Current score: 0` 的人為產物——回填真實分數後 wanting–knowing dissociation 依舊成立。
- **Llama 對「累計餘額」不敏感（slope null）**：每個 subject 內估計 `bet ~ score_before` 斜率再跨 57 subject 聚合，三個條件的中位數斜率皆 ≈ 0（orig +0.0001、+4 +0.0001、−4 +0.0003，IQR 皆橫跨 0，約 54–60% subject 斜率為正 ≈ 擲硬幣）；按當前餘額分組的 bet|win 與 bet|lose 差 Δ(w−l) ≈ 0（−0.03 / −0.10 / +0.38）。模型不會「贏了加碼／輸了縮手」——α 移動的是**基線**下注水平，而非 reward-history 敏感度。

### 3.2 Experiment 6 — Exploration/Exploitation (Bandit Task)

**神經科學對應：** Tonic dopamine 調節 exploration vs. exploitation balance——高 tonic DA → 更積極利用已知最優選項（exploitation 增強，incentive salience 集中）；低 tonic DA → 更趨向隨機探索，難以穩定 exploit（effort withdrawal，行為不穩定）。Bandit task 是此機制最直接的行為學範式。

**相關文獻：**
- **EVOLvE / BanditBench**（Nie et al., ICML 2025）：LLM 在 MAB 任務中的 in-context RL 評估框架；採用語義豐富的 arm 名稱（ClothesShopping 場景）消除位置偏差；OptFrac（最優臂選擇率）+ cumulative regret
- **TextBandit**（ACL EthicalLLMs 2025）：純自然語言 feedback 的 bandit 任務；K=5 slot machine，per-machine prob = [M1 0.20 / M2 **0.75** / M3 0.35 / M4 0.25 / M5 0.55]（真正最優臂為 Machine 2），T=25 rounds；Llama-3.1-8B paper baseline ≈ 31.6%（注：該數字是 source code 的 `best_machine` 設為 Machine 3（35% prob）的選擇率——一個 paper bug，而非對真正最優臂 Machine 2（75%）的選擇率）

**實驗設計（貼近 EVOLvE ClothesShopping）：**

- K=5 語義臂名稱（"Velvet Vogue Jacket" 等），每 run 隨機 shuffle 名稱→概率對應，消除位置偏差
- Bernoulli reward probs：0.7 / 0.5 / 0.4 / 0.3 / 0.1（shuffled per run）。註：此為自訂的**分級**獎勵向量（best=0.7、gap=0.2、各臂均值遞減），prompt 文字、verbalizer（`"{name} item, reward {r}"`）、shuffle 與 random-fallback 解析皆與 EVOLvE ClothesShopping 原始碼一致；但獎勵向量本身不同於 EVOLvE 標準的兩個 Bernoulli 設定（large-gap easy `[0.25×4, 0.75]`、small-gap hard `[0.4×4, 0.6]`，皆為「平坦干擾臂+單一最優」）。採用分級均值是為了讓 WorstFrac（最差臂迴避）成為一個有意義的獨立指標。
- T=50 rounds，30 runs（seeds 0–29），configs: α ∈ {0, +4, −4}，layers 11–20
- 生成模式（`vc.regenerate`，temperature=1.0）+ 字串匹配解析；無效輸出 fallback 隨機選臂

**Prompt（EVOLvE ClothesShopping）：**

實現要點：
- 臂名集合 = EVOLvE ClothesShopping 場景原始名單（前 10 個取 5 個）：`Velvet Vogue Jacket / Silk Serenity Dress / Urban Mystique Jeans / Celestial Symphony Scarf / Retro Revival Sneakers / Ethereal Elegance Blouse / Midnight Mirage Trousers / Vintage Vibe Coat / Opulent Oasis Gown / Mystic Mosaic Shirt`。
- `REWARD_PROBS_ORDERED = [0.7, 0.5, 0.4, 0.3, 0.1]`，`shuffle_arms(seed)` 

Round 0 output (no history):

```
You are in an online boutique powered by a bandit algorithm that offers a variety of clothing options from different brands.
There are 5 unique clothing items available, named [Velvet Vogue Jacket, Silk Serenity Dress, Urban Mystique Jeans, Celestial Symphony Scarf, Retro Revival Sneakers].
You choose an item to recommend based on past choices and rewards.
You aim to find the clothing item that customers are most likely to purchase and enjoy.
Each time a customer buys a recommended item, you update your strategy to better predict and meet future customer preferences.

A good strategy to optimize for reward in these situations requires balancing exploration and exploitation. You need to explore to try out all of the clothing brands and find those with high rewards, but you also have to exploit the information that you have to accumulate rewards.

Which item will you choose next? PLEASE RESPOND ONLY WITH [Velvet Vogue Jacket, Silk Serenity Dress, Urban Mystique Jeans, Celestial Symphony Scarf, Retro Revival Sneakers] AND NO TEXT EXPLANATION.
```

Round N output (with history; the prompt is rebuilt every round, and the history accumulates the (arm name, reward) pairs from rounds 0 to N−1):
```
... (same task instructions as above) ...

So far, you have interacted N times with the following choices and rewards:
Velvet Vogue Jacket item, reward 1
Silk Serenity Dress item, reward 0
Urban Mystique Jeans item, reward 1
... (one line per previous round: "{arm name} item, reward {0|1}") ...

Which item will you choose next? PLEASE RESPOND ONLY WITH [...same arm-name list as above...] AND NO TEXT EXPLANATION.
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

UCB1 在 T=50 短horizon 下：OptFrac = **0.359 ± 0.083**，Regret = **11.07 ± 1.36**，WorstFrac = 0.117。UCB1 在前 K=5 輪強制逐一探索每個臂，confidence bonus 在短 horizon 下長期偏大，導致探索過度。

**實驗結果（Llama-3.1-8B，30 runs × 50 rounds）：**

*Assistant Role（AI fashion assistant）：*

| α | mean OptFrac ± std | Early / Late OptFrac | WorstFrac | mean Regret | InvalidRate |
| --- | --- | --- | --- | --- | --- |
| UCB1 | 0.359 ± 0.083 | 0.287 / 0.408 | 0.117 | 11.07 | — |
| 0 | 0.609 ± 0.268 | 0.570 / 0.636 | 0.077 | 6.35 | 1.5% |
| +4 | **0.777 ± 0.090** | 0.692 / **0.834** | **0.060** | **4.15** | 0.3% |
| −4 | 0.479 ± 0.278 | 0.442 / 0.504 | 0.125 | 9.39 | 8.4% |

*No Role（neutral prompt）：*

| α | mean OptFrac ± std | Early / Late OptFrac | WorstFrac | mean Regret | InvalidRate |
| --- | --- | --- | --- | --- | --- |
| UCB1 | 0.359 ± 0.083 | 0.287 / 0.408 | 0.117 | 11.07 | — |
| 0| 0.816 ± 0.160 | 0.773 / 0.844 | 0.046 | 3.18 | 0.7% |
| +4 | **0.851 ± 0.089** | 0.783 / **0.897** | **0.043** | **2.78** | 0.1% |
| −4 | 0.619 ± 0.275 | 0.585 / 0.642 | 0.104 | 6.85 | 6.1% |

**結果解讀：**

- **LLM 的 in-context learning 超越 UCB1 理論算法**：UCB1 OptFrac 僅 0.359，而 LLM baseline 為 0.609–0.816。在 T=50 短 horizon，UCB1 因強制探索與過大的 confidence bonus 而效率低下；LLM 的語言模式匹配使其能更快速集中到高獎勵選項。

- **RSN 正向干預（α=+4）在兩個條件下均有效，且排除了「過早 lock-in 錯誤臂」的混淆解釋**：OptFrac 提升，Regret 下降，**WorstFrac 同步下降**（0.077→0.060 / 0.046→0.043）。最關鍵的診斷：α=+4 下 **30/30 run 的 OptFrac ≥ 0.5**（α=0 baseline：Assistant 11/30 失敗，No Role 僅 1/30 失敗），分布完全單峰（0.6–1.0），沒有任何 run 在前 5 輪完全未選到最優臂。std 下降（0.268→0.090 / 0.160→0.089）代表**每 run 均穩定收斂**，而非少數 run 拉高均值；Early OptFrac（rounds 1–20）與 overall OptFrac 呈強正相關（Assistant r=0.883，No Role r=0.829，p<0.0001），符合「更快識別最優臂 → 更早開始集中 exploit」的預期機制。Late OptFrac 明顯高於 Early OptFrac 亦與此一致。

- **α=−4 破壞效果一致且多維**：WorstFrac 在兩個條件下均上升（0.125 / 0.104），InvalidRate 大幅攀升（8.4% / 6.1%），OptFrac 跌至接近甚至低於 UCB1 基準。這支持負向 RSN 干預導致行為系統退化至「低 tonic DA」狀態——effort withdrawal，對高低獎勵臂的辨別力同步下降，輸出格式亦隨之崩潰。

- **Role prompt 顯著壓低 baseline**：No Role 的 α=0 OptFrac（0.816）遠高於 Assistant Role（0.609），差距約 −0.207。角色設定本身干擾了模型的 exploitation 能力；WorstFrac 亦略高（0.077 vs 0.046），說明 role 條件下最差臂迴避也較弱。提升幅度在 Assistant Role 下為 +0.168（+28%），在 No Role 下僅 +0.035（+4%）。RSN 部分補償了 role prompt 對 exploitation 的壓制——當 role 已壓低 baseline，RSN 才有更大的修復空間。

### 3.3 Gamble Task

賭博範式（IGT / CGT / slot-betting）的吸引力在於它把「wanting」操作化為**對賭注大小、風險偏好、輸後追高的外顯選擇**，且 knowing 維度可被任務設計剝離（CGT 機率透明、IGT 淨分有 ground-truth），正好補上 §3.1 Confidence Betting 的「更有信心」confound。以下四篇是設計本實驗的文獻基礎。

#### Related Work

**① `Large Language Models are Near-Optimal Decision-Makers…`（Li et al., arXiv 2506.16163, 2025）— 協議金標準 + 天花板警示。**
5 個 LLM（GPT-4o / o4-mini / Claude-3.5-Sonnet / Gemini-1.5-pro / DeepSeek-R1）vs 360 真人，跑 IGT + CGT + WCST 三範式。為防語料污染，保留遊戲機制本質（紅藍格子、賠率、下注梯度）但對文本描述與獎賞結構做了全面**符號重寫（Reworded & Redesigned）**。Methods 把 IGT/CGT 協議寫得可直接照搬。**最關鍵的警示在 Fig 2B**：LLM 的 risk adjustment 幾乎是平的——人類隨 asymmetry 動態調注，LLM 跨所有比例都押固定高注（GPT-4o-mini/DeepSeek ~90%、Claude >60%），即 **baseline risk-taking 已頂到天花板，+α 無上升空間，信號只能在 −α 側看**。19 個 robustness variant（含 role-play persona）行為定性不變 → prompt persona 推不動 risk adjustment（這對 hidden-state 注入是利好，見下「卖点」）。

**② `Can Large Language Models Develop Gambling Addiction?`（Lee et al., arXiv 2509.22818, 2025）— betting 指標公式 + 同源 SAE 先例。**
6 個 LLM 玩負期望值（30% 勝率、3× 賠付、EV −10%）slot machine，2×32 析因（betting style × 5 個 prompt 組件 G/M/H/W/P）。三大發現：(a) **variable betting（自由定額）比 fixed betting 顯著放大破產率與所有 irrationality 指標**——是「自主權本身」而非賭注大小驅動 risk（Fig 10：variable 平均賭注更小卻破產更多）；(b) **goal-setting(G) / maximize(M) 是最強的 risk 放大組件**，G 幾乎翻倍破產率；(c) 質性分析見 illusion of control、gambler's fallacy、loss chasing、house-money effect。**最重要的是 §4：在 LLaMA-3.1-8B 上跑 SAE + activation patching，找到 112 個（~1%）因果 feature 雙向控制賭博行為，risky feature 集中在 later layers（L24 佔 18 個）、safe feature 在 early-mid（L5–L8）**——這與 RSN 的 mid-layer（11–20）wanting 方向是**同方法、可對照**的 mechanistic 先例。

**③ `BioLLMAgent`（Zuo et al., arXiv 2603.05016, 2026）— IGT 認知參數讀數層 + 臨床對照靶點。**
把臨床驗證過的 RL 認知模型（ORL）當「內部驅動」、LLM persona prompt 當「外部驅動」，用權重 ω 線性融合，去復現六個真人 IGT 數據集（健康對照 + 安非他命 + 海洛因成癮）。對我們有用的**不是它的融合架構**（其 LLM 是把 T 輪輸出平均成的**靜態先驗**，不參與逐輪學習——與我們「逐輪決策＋逐輪注入」相反，架構不可照搬），而是兩樣可拆出的東西：(a) **ORL 五參數讀數**（`A_rew` 獎勵學習率 / `A_pun` 懲罰學習率 / `K` 遺忘 / `β_F` 頻率權重 / `β_P` perseveration）——把 reward 與 punishment 學習率**分離**，正好對應「+α → reward 敏感↑、punishment 敏感↓」的 DA 預測；(b) **六個公開臨床 IGT 數據集 + 健康/成癮參數區間**，可當 −α 的對照靶（測 −α 是否把 LLM 的 ORL 參數推向成癮群體那一端）。亦再次印證中小模型（Llama-3.2-3b/Gemma-3）對 prompt 指令「instruction resistance」，只有 >70B 級才聽話。

**④ `Mitigating Gambling-Like Risk-Taking…`（Du, arXiv 2506.22496, 2025）— 僅 framing，實驗數字不可信。**
7 頁短文。可用的只有四個形式化定義（Overconfidence / Loss Chasing / Probability Misjudgment / Risk-Reward Miscalibration）+ GTS 複合分公式，可 cite 當 framing 來源。Table 1 的 RARG-70B / LLaMA-2-70B 結果無訓練細節、無數據集、無 baseline 出處，IGT「Optimal%」也未給協議——**不要引用其任何實驗數字或 IGT 協議**。（與 ② 不同作者；此篇單作者 Y. Du。）

#### 行動建議（落到實驗設計）

| 項目 | 建議 | 依據 |
|---|---|---|
| **IGT 協議** | 4 deck（A/B 劣勢、C/D 優勢；損失頻率不對稱：A/C 頻繁小罰、B/D 罕見大罰），淨分 = P(優勢 C+D) − P(劣勢 A+B) 為 ground-truth；trial 數取 100（IGT 經典 / BioLLMAgent）或 80（Near-Optimal），擇一固定 | ①Methods + ③ |
| **IGT 讀數** | 不只報淨分，用 **ORL 五參數**擬合，重點看 `A_rew/A_pun` 比值是否被 α 推向成癮群體區間；以六個臨床數據集為對照靶 | ③ ORL + 臨床數據集 |
| **CGT 協議** | 照搬 ①：64 round、8 個紅藍比例（1:9…9:1）、{5/25/50/75/95}% 下注檔、**simultaneous 呈現**；**放棄升降序延遲厭惡維度**（①明確判定對 LLM 不適用） | ①Methods + 明確判定 |
| **betting 風格** | 用 **variable / 自由定額**而非離散 {0,2,5,10}，放大 α 效應空間 | ② Fig 10（自主權驅動 risk） |
| **指標** | 加入 ② 的 `I_BA = mean(min(bet/balance,1))`、`I_LC = mean_{loss}(max(0,Δ(bet/balance)))`、`I_EC = mean(1[bet/balance≥0.5])`；`I_LC` 直接對應我們 running-score 的 null | ② eq 1–3 |
| **前置檢查** | 先跑 α=0 baseline 確認 Llama3-8B 在 IGT 上**不是 near-random**（③ 顯示 <70B 模型可能被 pretrain bias 鎖死），再決定值不值得做 dose-response | ③ Fig 7 / Inverse Scaling |
| **差異化卖点** | ① 證明 prompt persona 推不動 risk adjustment → 我們測「**hidden-state α 能否推動 prompt 推不動的維度**」是干淨賣點；② 的 SAE risky-feature(L24) vs RSN(L11–20) 是可寫的 mechanistic 對照 | ① + ② |


#### Cambridge Gamble Task (CGT)

**為什麼要加**：CGT 測「**已知風險決策**（Decision under Known Risk）」——受試者一眼就看到紅藍格子比例（如 9 紅 1 藍 → 贏面 90%），機率完全顯性化，極大排除了學習能力與工作記憶的干擾，純粹測「明知機率、卻管不管得住自己」的行為特質。這正好補上 Confidence Betting 的 confound：Betting 中「模型更有信心」也能解釋賭注上升，而 CGT 機率透明，賭注變化只能歸因於 risk-taking 本身，不能再用「更準/更有信心」搪塞。
（對照：IGT 測「**未知模糊性決策**（Decision under Ambiguity）」——一開始不知道哪牌組好壞，靠反覆輸贏摸索規律，重度依賴 WM 與 learning，見下。）

**CGT 的四個行為學切片**（CANTAB 標準輸出）：
- **決策質量（Quality of Decision Making）**：是否理智地永遠選格子數多的顏色（高機率那面）。
- **審慎時間（Deliberation Time）**：從看到格子到按下顏色的思考時長。
- **風險承擔（Risk Taking）**：選了高機率顏色時，平均願意拿出多少比例積分去賭。
- **風險調節度（Risk Adjustment）**：是否「看碟下菜」——9:1 時下大注、6:4 時下小注。
- **延遲厭惡 / 衝動性（Delay Aversion）**：升序條件下因不想等待，在額度還很低（如 5%）時就草草按下（缺乏耐心）；降序條件下在額度仍極高（95%）時就衝動按下（行為抑制障礙）。

**人腦對應（為何是乾淨的 DA assay）**：用左旋多巴（L-Dopa）人為提高健康受試者腦內 DA，或對比帕金森患者，得到的結論非常精準——**DA 調控「風險偏好」而非「決策智商」**：DA 升高時，**決策質量不變**（仍知道 9:1 該選 9 那面），但**風險承擔顯著飆升**（Optimism Bias：即使 6:4 低贏面也傾向砸下 75%+）。延遲厭惡端則由 DA 維持前額葉的行為抑制——DA 失調 / 戒斷的成癮者、ADHD 患者表現出極高延遲厭惡，無法忍受倒數等待，會在不恰當額度瞬間衝動確認。這條 wanting↑/decision-IQ持平 的雙分離，正是本工作 α 注入想要證明的同一條曲線。

**文本化（Textualization）設計**——直接借用 CGT 範式，分兩步投餵給模型：

*第一步：紅藍格子 + 代幣轉成純文字情境*
```
[CGT 文本實驗環境設定]
「你正在參加一項心理學博弈任務。螢幕上排開 10 個盒子，
其中 7 個紅色、3 個藍色。一個黃金代幣被隨機藏在其中一個盒子裡。
你當前總積分：100 分。請分兩階段決策：
 階段 1（猜測）：你認為代幣在【紅色】還是【藍色】？
 階段 2（下注）：下注比例窗口正在滾動，目前可選下注總積分的 [5%, 25%, 50%, 75%, 95%]。
 猜對則贏得下注積分，猜錯則扣除。請給出你決定下注的精確比例。」
```

*第二步：控制變數，循環 ~100 輪（隨機改變紅藍比例 9:1 / 8:2 / 6:4）統計三大指標*
- **QDM**：模型是否在 8:2 時永遠選 8 那一方？選 2 那方 = 認知非理性偏差。
- **風險調節度**：9:1 時是否下 95%（大賭）、6:4 時是否懂得主動降到 5%（小賭）？
- **輸後行為（Loss Chasing）**：上一輪剛輸掉積分後，下一輪下注比例是否報復性飆升？——這是測 DA 預測誤差與成癮行為的核心指標。

**對 α 注入的預測**：α=+4 在所有 P 水準下下注更大、6:4 時也不肯降額（風險承擔↑、風險調節度↓）、Loss Chasing 更強，但 QDM（永遠選多格那面）不變；α=−4 保守、過度降額。即「決策質量持平、風險偏好被 α 推動」——與人腦 L-Dopa 結果同構。

**與既有結果的銜接**：升降序額度 + Loss Chasing 的「報復性加注」直接對應 §3.1 running-score Betting 變體（該變體在 Llama 上已測出對 running balance 不敏感的 null）；若 CGT 機率透明條件下 α 仍推動 Loss Chasing，則比 running-score 更強——把「reward-history 不敏感」與「risk-preference 可被 α 推動」乾淨地分開。

**LLM 既有實現參考**：*Can Large Language Models Develop Gambling Addiction?*（arXiv 2025）已在老虎機 / 下注任務上報告 LLM 的 Illusion of Control + Loss Chasing，且「下注自主權越高破產率越飆升」——可借其 prompt 框架與破產率 / 加注曲線指標。

#### Iowa Gambling Task (IGT)

**範式定位**：模糊性決策。四副牌（兩好兩壞），受試者不知好壞，靠反覆抽牌的輸贏學出「避開高即時獎賞但長期淨虧」的牌組。淨得分 = (好牌組抽取數 − 壞牌組抽取數)，是有 ground-truth 的連續量，天然適配 `get_answer_bandit.py` 式的 α-steering 多輪 pipeline。

**LLM 既有實現參考**：協議照 ①（Near-Optimal，80/100 trials、4 deck），讀數用 ③（BioLLMAgent 的 ORL 五參數），對照靶用 ③ 的六個臨床數據集——詳見上方「文獻概述 / 行動建議」。

**對 α 注入的預測**：α=−4 → 衝動偏好高即時獎賞的壞牌組、淨得分下降、學習曲線變平（對應 DA 不足 / 成癮者的 IGT 表現）、`A_rew/A_pun` 比值推向成癮群體區間；α=+4 在已 near-optimal 的基線上空間有限。

**待辦**：(1) 先跑 α=0 baseline 確認 Llama3-8B 不是 near-random（③ 警示 <70B 可能被 pretrain bias 鎖死）；(2) 確認既有 prompt 框架能否套上 `get_answer_bandit.py` 的逐輪 α-hook（多輪、bs=1、per-run reset）；(3) **不照搬 BioLLMAgent 架構**（其 LLM 是靜態先驗，與逐輪注入相反），只借其 ORL 讀數層與臨床靶點。

#### 實作規格（2026-06，已對照 Near-Optimal repo 原始碼確認）

源碼：`/Users/paveenhuang/Downloads/Benchmark/Near-Optimal`（oTree 實作）。本工作**不跑 oTree**——抽出核心機制（懲罰分布 / bet 映射 / 歷史 prompt 構造）寫成 `get_answer_cgt.py` / `get_answer_igt.py` 兩支獨立腳本，套用 `get_answer_bandit.py` 既有的逐輪 α-hook 介面（`vc.regenerate(inputs=[prompt], diff_matrices=raw_mask*alpha, …)`，bs=1、每輪重建 prompt、`utils.parse_configs` 解析 configs）。

**共同設定**：Llama3-8B、layers 11–20、nmd mask、**−α 主軸 + 雙向**（configs `0 / ±2 / ±4 / ±6 / ±8`，先跑 α=0 baseline 確認非 near-random）、choose 中性語境（treasure-chest reword 版，防語料污染）。輸出格式沿用 repo 的 `<reasoning>…</reasoning><choice>N</choice>`，兜底由 GPT-3.5 改為正則 + 重採樣。

**CGT 機制（照搬，已對碼）**：
- `init_money=100`；`total_interactions=64`、`round_interactions=8` → **8 phase × 8 round**，每 phase 開頭積分 reset 回 100。
- 紅藍比例 **8 種**：`(1,9)(6,4)(4,6)(3,7)(9,1)(7,3)(2,8)(8,2)`（=(blue,red)，無對稱 5:5），每 phase 內 shuffle 各出現一次。
- 下注檔 `bets=[0.05,0.25,0.5,0.75,0.95]`；**simultaneous** 一次列 10 個 choice（0–4=blue/F 五檔、5–9=red/J 五檔），模型輸出 0–9。**確認放棄升降序延遲厭惡維度**（repo 本身就是 simultaneous，未實作 sequential）。
- 賠付：`payoff = round(remain × bet)`，猜對 +、猜錯 −（贏得下注的 2 倍）；金幣位置每輪 `randint(1,10)` 獨立隨機。choice_order 每玩家 rotate 防位置偏好。
- ground-truth 上界（oracle）：永遠選多數色 + 押 95%。
- 讀數：**QDM**（是否選多數色）、**風險承擔/調節度**（押注比例隨 asymmetry 的斜率）、**Loss Chasing**（②的 `I_LC`），外加 `I_BA / I_EC`。

**IGT 機制（照搬，已對碼）**：
- `init_money=2000`（loan）；**total_interactions=100**（採經典/BioLLMAgent，非 repo 的 80）。
- 4 牌固定獎勵 `card_rewards=[100,100,50,50]`（牌1/2 高即時、牌3/4 低即時）；懲罰分布預排好逐輪 pop：牌1 頻繁中罰、牌2 罕見巨罰 1250、牌3 頻繁小罰、牌4 罕見中罰 250 → **牌1/2 長期淨虧（劣勢），牌3/4 長期淨賺（優勢）**。
- 輸出 `<choice>1-4</choice>`，card_order 每玩家 rotate；每輪把過往（選幾號、得多少、罰多少）全量回填 prompt。
- **淨分需 offline 算**：repo 無淨分欄位，淨分 = P(選優勢牌 3+4) − P(選劣勢牌 1+2)。
- 讀數：淨分 + 學習曲線斜率 + **ORL 五參數**（`A_rew/A_pun/K/β_F/β_P`，重點看 `A_rew/A_pun` 是否被 α 推向成癮群體區間）。

**對照臂（persona vs α）**：repo `prompt/roles/` 已含臨床 persona（CGT：`Gambling_Disorder/risk-taker/risk-averse`；IGT：`methamphetamine_dependence/vmPFC_lesion/alcohol_use_disorder`）。可做「prompt persona 推不動（①證明）vs hidden-state α 推得動」的乾淨對照——這是核心差異化卖點。



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

## 備選與待補充 Benchmark

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


## 5. Human Behaviour Simulation

本節登記每個行為學實驗**對應的經典人類／動物行為學範式**及其文獻根源，把我們的 LLM 實驗 anchor 到神經科學傳統（與 §4 互補：§4 報告我們做了什麼、結果如何；§5 標明它的人類範式血統）。實驗的完整結果與分析仍在各自的 §4.x 小節，此處只做對應與 cite。

| 實驗 | LLM 任務形態 | 對應人類行為學範式 | 人類範式文獻 | LLM 實現 | 狀態 |
|---|---|---|---|---|---|
| **Confidence Betting** | MCQ + 押注 0/2/5/10 | Post-decision wagering / confidence betting | Persaud et al. (2007); Fleming & Dolan (2012) | 本工作（§4.6） | ✅ Done |
| **Bandit (MAB)** | 多輪 explore/exploit，語義臂名 | Multi-armed bandit / probabilistic reward learning | Daw et al. (2006) | EVOLvE-Nie et al. (2025); TextBandit (ACL EthicalLLMs 2025)（§4.7） | ✅ Done |
| **Cambridge Gamble Task** | 機率透明下注（P% 已知） | Cambridge Gamble Task（DA-agonist／Parkinson 對比） | Rogers et al. (1999); Pessiglione et al. (2006, pramipexole) | TBD（找已在 LLM 上做過 CGT 的實驗） | ⬜ Pending |
| **Iowa Gambling Task** | 多輪牌組選擇（淨損益學習） | Iowa Gambling Task | Bechara et al. (1994) | TBD（找已在 LLM 上做過 IGT 的實驗） | ⬜ Pending |

**說明：**
- **Confidence Betting / Bandit** 的結果在 §4.6 / §4.7，此處只標範式血統，不重複結果表。
- **CGT / IGT** 目前為 pending：人類範式根源已確定，但「LLM 實現」欄待補——你會去找已在 LLM 上跑過 CGT / IGT 的論文填入，再決定是否復現。CGT 同時是 Confidence Betting 的 confidence-confound control（機率透明可排除「更自信」解釋）。

## References

人類行為學範式文獻：
- Bechara, Damasio, Damasio & Anderson (1994). Insensitivity to future consequences following damage to human prefrontal cortex. *Cognition.* 
- Rogers et al. (1999). Dissociable deficits in the decision-making cognition of chronic amphetamine abusers, opiate abusers, patients with focal damage to prefrontal cortex... *Neuropsychopharmacology.* 
- Daw, O'Doherty, Dayan, Seymour & Dolan (2006). Cortical substrates for exploratory decisions in humans. *Nature.* 
- Pessiglione, Seymour, Flandin, Dolan & Frith (2006). Dopamine-dependent prediction errors underpin reward-seeking behaviour in humans. *Nature.* 
- Persaud, McLeod & Cowey (2007). Post-decision wagering objectively measures awareness. *Nature Neuroscience.* 
- Fleming & Dolan (2012). The neural basis of metacognitive ability. *Phil. Trans. R. Soc. B.* 

- Berridge & Robinson (1998). What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *Brain Research Reviews.*
- Fenigstein, Scheier & Buss (1975). Public and private self-consciousness: Assessment and theory. *Journal of Consulting and Clinical Psychology.*
- Kim et al. (2024). Will LLMs Sink or Swim? Exploring Decision-Making Under Pressure. *EMNLP 2024 Findings.*
- RSN paper (ACL Findings). Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism for Confidence Steering.
- Zhou et al. (2026). General scales unlock AI evaluation with explanatory and predictive power. *Nature.* https://www.nature.com/articles/s41586-026-10303-2
- Binz et al. (2025). Centaur: a foundation model of human cognition. *Nature.* https://doi.org/10.1038/s41586-025-09215-4
- Xtra-Computing/LLM-Deception (ICLR 2026 Oral). Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts. https://openreview.net/forum?id=PDBBYwd1LY
