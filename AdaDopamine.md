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

# 1. MCQ Reasoning & Factor Benchmark Results

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

# 2. Existing Evidence from the RSN Paper

| Experiment | Measurement | Behavioral interpretation |
| --- | --- | --- |
| MMLU-E (abstention rate) | Expert 6.9% vs. Non-Expert 44.8% E-ratio | Effort willingness|
| MMLU-E Bidirectional Steering | +α: 3.7%；−α: 65.1% E-ratio | RSN 作為雙向 gain knob（causal evidence） |
| RSN Knockout (Ablation) | 拿掉 RSN → Non-Expert gap 縮小（24.15% → 11.03%） | Suppression lock 的必要性驗證 |
| Neutral Steering — Reasoning | +α 提升 MMLU-Pro / GPQA / AR-LSAT / LogiQA |  |
| Neutral Steering — Factuality | −α 提升 TruthfulQA / FACTOR (Only Llama3 & mistral, not Qwen3) | |
| Reasoning Willingness Self-Report | 模型自評 0–9；+α 一致提升各任務分數 | 主觀 effort willingness |
| Cross-model Transfer (Base ← IT RSN) | IT RSN 作用於 Base model；abstention 61% → 7% | 機制起源（pre-training latent） |


## 2.1 Experiment A — Abstention Rate (MMLU-E)

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

## 2.2 Experiment A′ — Neutral Steering E-Ratio (Bidirectional Control, Llama3-8B)

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

## 2.3 ExperimentB：Willingness Self-Evaluation（0–9 scale）

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


# 3. Core Behavioral Experiments

## 3.1 Experiment 5 — Confidence Betting (Incentive Salience)

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

**口徑約定（跨全節統一）：** 所有 bet 分佈百分比一律以 **全樣本 n 為分母**（bet0%+bet2%+bet5%+bet10%+invalid% = 100%）；`mean_bet` 則只在**有效下注**上計算。兩者混用會產生假數字——invalid 接近 0 時兩種口徑重合，但在 invalid 高的 cell（Qwen α=+8）會差到 4 倍（bet10 全樣本 1.1% vs 有效口徑 4.6%）。引用任何一格前先確認口徑。

---

### 3.1.1 Llama3-8B-IT

**結果一：Llama3-8B-IT，GPQA main + diamond，n=646, Static（−8→+8 九檔 dose-response, `static_0616`）**

| α | accuracy (micro) | mean_bet | bet0% | bet2% | bet5% | bet10% | mean_score_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| −8 | 27.7% | 4.91 | 4.6% | 7.7% | 80.2% | 7.4% | −2.10 |
| −6 | 27.1% | 5.13 | 0.0% | 10.5% | 80.5% | 9.0% | −2.37 |
| −4 | 27.4% | **4.42** | 0.2% | 22.0% | 76.2% | 1.7% | −1.98 |
| −2 | 26.2% | **4.28** | 0.5% | **24.9%** | 73.5% | 1.1% | −2.03 |
| **orig (α=0)** | 28.3% | 5.13 | 0.2% | 12.7% | 76.8% | 10.4% | −2.25 |
| +2 | 29.0% | 6.61 | 0.0% | 2.5% | 63.9% | 33.6% | −2.89 |
| **+4** | 28.3% | **7.63** | 0.0% | 0.0% | 47.4% | **52.6%** | −3.30 |
| +6 | 26.2% | 7.32 | 0.0% | 0.2% | 53.4% | 46.4% | −3.41 |
| +8 | 26.6% | 7.62 | 0.0% | 7.5% | 35.6% | 56.8% | −3.35 |

- **正臂 = 單調-飽和（非倒 U）。** mean_bet 5.13→6.61→**7.63**(0→+2→+4),+6/+8 不再上升而是壓在 ~7.6 天花板、bet10% 卡在 ~47–57%。betting 是單步決策,wanting 漲到 bet=10 上限即飽和,**夠不到 Yerkes-Dodson 右臂的過載崩潰**——與 Bandit 的倒 U(+8 塌)是不同形狀但同一方向,兩條曲線都在「正 α = 更想要」側達峰。
- **wanting–knowing dissociation。** α 由 −2 掃到 +8,mean_bet 4.28→7.63(+78%),**accuracy 全程 26–29%±2 無系統變化**——押得越凶不代表答得越對。9 檔上成立,不只 ±4 兩點。
- **負臂 = U 形折返,不是單調下降。** 中等劑量 −2/−4 顯著壓低 wanting(Mann–Whitney 單側 vs orig:−2 p=7e−16、−4 p=1.3e−11;bet10% 由 10.4%→1–2%、bet2% 升至 22–25%),符合 under-wanting 預測;但極端劑量 −6/−8 折返回 baseline(−6 mean_bet=5.13 與 orig 持平 p=0.65;−8 帶輕微格式鬆動 bet0%↑4.6%)——steering 過強時行為**解體**而非「更不想要」,與 Bandit 負臂的「垮」同源。RSN 因此是有最優工作區的旋鈕,兩端皆失效,合乎雙臂框架。

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

### 3.1.2 Qwen2.5-7B-Instruct（2026-07-28）

**設定差異：** 注入層 **16–21**（`--start 16 --end 22`，exclusive；Qwen2.5-7B 有 28 decoder layers, H=3584）。

**結果一：Qwen2.5-7B-Instruct，GPQA main + diamond，n=646，−8→+8 九檔**

| α | accuracy (micro) | mean_bet | bet0% | bet2% | bet5% | bet10% | invalid% | Holm p vs orig | Cliff δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| −8 | 33.3% | **4.02** | 3.3% | **33.4%** | 55.0% | 4.5% | 3.9% | 7e−58 * | −0.409 |
| −6 | 32.8% | 4.89 | 0.5% | 12.2% | 81.6% | 5.6% | 0.2% | 1e−17 * | −0.175 |
| −4 | 33.4% | 5.14 | 0.0% | 1.7% | 94.4% | 3.9% | 0.0% | 3e−09 * | −0.095 |
| −2 | 34.2% | 5.13 | 0.0% | 0.3% | 96.9% | 2.8% | 0.0% | 6e−10 * | −0.094 |
| **orig (α=0)** | 34.7% | 5.60 | 0.0% | 0.0% | 88.1% | 11.9% | 0.0% | — | — |
| +2 | 34.1% | 6.28 | 0.0% | 0.0% | 74.3% | 25.7% | 0.0% | 7e−10 * | +0.138 |
| **+4** | 33.8% | **7.06** | 0.0% | 0.0% | 58.8% | **41.2%** | 0.0% | 8e−32 * | +0.293 |
| **+6** | 33.8% | 5.03 | 0.0% | 0.0% | **99.4%** | 0.6% | 0.0% | 3e−16 * | −0.113 |
| **+8** | 29.6% | 4.88 | 0.2% | 2.5% | 19.7% | 1.1% | **76.6%** | 1e−08 * | −0.172 |

（Holm 校正跨 8 個 steered cell；Cliff δ 與 MWU 在**有效下注**上計算。**+6/+8 不是被丟棄的壞資料，而是本節要觀察的現象本身**——它們構成右臂過載的證據，見下方「右臂」。唯一的限定是量化用法：這兩格的 `mean_bet` 不可放進單調趨勢的擬合或內插，因為其分母已被 invalid／幸存者偏差改變；要引用的是它們的**失效形態**。）

- **核心劑量反應複現。** mean_bet 4.02 → 4.89 → 5.14 → 5.13 → 5.60 → 6.28 → **7.06**，Spearman ρ=**0.464**（n=4,496, p=1.4e−238）；bet10% 由 11.9% 升至 41.2%。與 Llama 同形。
- **wanting–knowing dissociation 複現且更乾淨。** 可用帶內 accuracy 32.8–34.7%（全距 **1.86pp**），**每一格 vs orig 的 χ² 皆不顯著**（p=0.52–0.91）。押注量升 26%（5.60→7.06）而答對率不動。
- **負臂形狀與 Llama 不同（值得注意的跨模型差異）。** Llama 負臂是 U 形折返（−2/−4 最低、−6/−8 回到 baseline）；Qwen 是**單調**的：−8 最低（4.02, δ=−0.409）且 bet2% 升到 33.4%。Qwen 的 under-wanting 在極端劑量下沒有折返。

**右臂：betting 上首次觀察到的 α 過載（與 Llama 的關鍵分歧）**

Llama 的 betting 正臂是**單調飽和**（+4 見頂 7.63，+6/+8 壓在 ~7.3–7.6 天花板不塌），據此曾判定「betting 是單步決策，夠不到 Yerkes-Dodson 右臂」。**Qwen 推翻了這個一般化**——它在 +4 見頂後、+6 就開始崩，而且 **+6 與 +8 是兩種不同的失效模式**（由 `--save_all_raw` 重跑的原始文本判定，`gpqa_bet_raw/`）：

| | α=+6：行為退化，格式完好 | α=+8：格式解體 |
| --- | --- | --- |
| 原始文本形態 | 與 orig **完全相同**（`'5\nAnswer: B'`，中位長 11 字元，乾淨兩行格式 99.4%） | 全體中位長 88 字元、散文子集中位 **214 字元**（orig 為 11），乾淨格式僅 14.9% |
| invalid | 0% | 76.6% |
| 現象 | bet 分佈**塌成常數**：bet5=99.4%、bet_entropy 0.030。配對看：orig 押 10 的 76 題中有 74 題在 +6 改押 5 | 推理先行把下注擠出格式：`"...I'll bet 5 points on this one to make sure I get it right.\nAnswer: B"` |
| 判定 | **真實行為塌陷**（非解析假象——文本形態、答題、accuracy 全部正常，只有下注這一維退化） | **過載 + 一個解析 bug 的混合**（見下） |

- **+6 的塌陷不是「回到 baseline」。** baseline 是 88%/12% 的 bet5/bet10 混合；+6 是 99.4% 全押 5 的退化分佈。模型仍在下注、仍能答題，但**不再區分**。
- **+8 的數字受兩個問題污染，不可直接引用。** ① 500 個 invalid 中有 **137 個是解析 bug**——Qwen 在 chat 模式下輸出前導冒號（`': 5\nAnswer: A'`），而 `parse_output` 的 `BET_LEADING_RE = ^\s*(\d+)` 匹配不到（`Bet` 字面詞在 prompt 裡、不在 generation 裡）。修正 regex 後 invalid 由 76.6% 降至 **56.2%**、mean_bet 變 5.04。② 其餘 363 個是真過載（散文）。③ 幸存者偏差：能成功下注的樣本本就答得更好（acc|valid 35.8% vs acc|invalid 27.7%），故 +8 的 mean_bet 與 accuracy 皆不可比。**+8 的可引用資訊是「76.6% 格式解體」這個事實本身，不是那個 mean_bet 數值。**
- 兩個失效模式與 CGT-seq / HaluEval 的 **+α「散」**（deliberation-first 吞掉結論）同源；`mean_score_delta` 在 invalid 高的 cell 無意義（+8 的 −0.45 只反映 56% 樣本 score_delta=0）。

**結果二：Qwen2.5-7B-Instruct，MMLU all subjects，n=14,042（±4 兩檔）**

| condition | micro acc | macro acc | mean_bet | bet0% | bet2% | bet5% | bet10% | invalid% | MWU p | Cliff δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 65.43% | 66.78% | 5.10 | 0.0% | 0.0% | 97.9% | 2.1% | 0.0% | — | — |
| α=+4 | 65.55% | 66.78% | **5.56** | 0.0% | 0.0% | 88.9% | **11.1%** | 0.0% | 2e−204 | +0.089 |
| α=−4 | 65.45% | 66.79% | **4.97** | 0.0% | **3.1%** | 95.6% | 1.3% | 0.0% | 6e−70 | −0.043 |

- **大 n 解離成立**：accuracy 65.43 / 65.45 / 65.55%（χ² p=0.98 / 0.83），mean_bet 方向正確、bet10% 由 2.1% 升至 11.1%。
- **但效應量顯著小於 Llama**：Llama MMLU 是 4.45 → 7.45（**+67%**）、bet10 4.7%→49.0%；Qwen 是 5.10 → 5.56（**+9%**）、bet10 2.1%→11.1%，Cliff δ 僅 +0.089。GPQA 上 δ=+0.293（中等）而 MMLU 僅 +0.089——**同一 α、同一 mask，兩個任務差三倍**。極小的 p 值來自 n=14,042，不代表大效應；**引用時要報 δ 而非 p**。

**跨模型小結（§3.1.1 vs §3.1.2）**

| | Llama3-8B | Qwen2.5-7B |
| --- | --- | --- |
| 核心劑量反應 | ✅ | ✅（ρ=0.464, −8…+4） |
| wanting–knowing 解離 | ✅ | ✅（更乾淨：全距 1.86pp，各格皆 n.s.） |
| GPQA 峰位 | +4（mean_bet 7.63） | +4（mean_bet 7.06） |
| 正臂形狀 | 單調飽和，+8 不塌 | **倒 U：+4 見頂，+6 塌陷，+8 解體** |
| 負臂形狀 | U 形折返（−6/−8 回 baseline） | 單調（−8 最低） |
| 可用工作帶 | −8…+8 | **−8…+4**（更窄） |
| MMLU 效應量 | +67% mean_bet | +9%（δ=+0.089） |

**兩點結論：**
1. **主結論跨模型成立**——「α 推動 wanting 而不動 knowing」在第二個模型家族上獨立複現，補上了本節原先的單模型缺口。
2. **「betting 夠不到右臂」這個一般化被推翻。** 它是 Llama 的性質，不是 betting 範式的性質。Qwen 的工作帶更窄、右臂過載在單步決策上也能觀察到——**最優 α 與可用帶寬皆為模型特異**，跨模型比較時不能沿用同一組 α 假設。


## 3.2 Experiment 6 — Exploration/Exploitation (Bandit Task)

**神經科學對應：** Tonic dopamine 調節 exploration vs. exploitation balance——高 tonic DA → 更積極利用已知最優選項（exploitation 增強，incentive salience 集中）；低 tonic DA → 更趨向隨機探索，難以穩定 exploit（effort withdrawal，行為不穩定）。Bandit task 是此機制最直接的行為學範式。

**相關文獻：**
- **EVOLvE / BanditBench**（Nie et al., ICML 2025）：LLM 在 MAB 任務中的 in-context RL 評估框架；採用語義豐富的 arm 名稱（ClothesShopping 場景）消除位置偏差；OptFrac（最優臂選擇率）+ cumulative regret

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

**實驗結果（Llama-3.1-8B，No-Role neutral prompt，30 runs × 50 rounds，α 全掃 −8→+8 @ layers 11–20）：**

| α | mean OptFrac ± std | Early / Late OptFrac | WorstFrac | mean Regret | InvalidRate | fail(<0.5) |
| --- | --- | --- | --- | --- | --- | --- |
| UCB1 | 0.359 ± 0.083 | 0.287 / 0.408 | 0.117 | 11.07 | — | — |
| −8 | 0.614 ± 0.238 | 0.562 / 0.649 | 0.067 | 6.50 | 15.7% | 7/30 |
| −6 | 0.641 ± 0.244 | 0.588 / 0.677 | 0.084 | 6.41 | 11.5% | 7/30 |
| −4 | 0.601 ± 0.183 | 0.547 / 0.638 | 0.095 | 7.33 | 20.1% | 7/30 |
| −2 | 0.748 ± 0.120 | 0.683 / 0.791 | 0.073 | 4.79 | 9.7% | 1/30 |
| 0 | 0.843 ± 0.114 | 0.777 / 0.887 | 0.043 | 2.92 | 1.5% | 0/30 |
| **+2** | **0.891 ± 0.070** | 0.845 / **0.922** | **0.037** | **2.15** | 0.4% | **0/30** |
| +4 | 0.865 ± 0.084 | 0.827 / 0.890 | 0.043 | 2.56 | 0.1% | 0/30 |
| +6 | 0.842 ± 0.094 | 0.802 / 0.869 | 0.042 | 2.85 | 0.3% | 0/30 |
| **+8** | **0.515 ± 0.177** | 0.525 / 0.508 | 0.075 | **8.17** | 0.2% | **11/30** |

（`fail(<0.5)` = 30 個 run 中 OptFrac < 0.5 的 run 數，是「雙峰失穩」的直接讀數。）

*Assistant Role（AI fashion assistant，±4 三點對照，role-prompt 壓制故事，§4.7）：*

| α | mean OptFrac ± std | Early / Late OptFrac | WorstFrac | mean Regret | InvalidRate |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.609 ± 0.268 | 0.570 / 0.636 | 0.077 | 6.35 | 1.5% |
| +4 | **0.777 ± 0.090** | 0.692 / **0.834** | **0.060** | **4.15** | 0.3% |
| −4 | 0.479 ± 0.278 | 0.442 / 0.504 | 0.125 | 9.39 | 8.4% |

**統計檢驗（per-run OptFrac，30 runs/檔，非參數；腳本 `RoleAnswer/analyze_bandit_stats.py`）：** Kruskal–Wallis omnibus **H=125.2，p=2.8e−23**——α 對 OptFrac 有壓倒性整體效應。關鍵兩兩對比（Mann–Whitney U + Holm 校正，附 Cliff's δ 效應量）：

| 對比 | mean_a → mean_b | p (Holm) | Cliff's δ | 判定 |
| --- | --- | --- | --- | --- |
| **0 vs +2**（峰 vs baseline） | 0.843 → 0.891 | 0.26 | −0.227 (small) | **n.s.** |
| +2 vs +4 | 0.891 → 0.865 | 0.26 | +0.180 (small) | n.s. |
| **+6 vs +8**（右臂崩潰起點） | 0.842 → 0.515 | 3.4e−9 | **+0.936 (large)** | *** |
| **+2 vs +8**（峰 vs 右崩） | 0.891 → 0.515 | 6.0e−10 | **+0.979 (large)** | *** |
| 0 vs +8 | 0.843 → 0.515 | 1.3e−8 | +0.900 (large) | *** |
| 0 vs −4 | 0.843 → 0.601 | 2.2e−6 | +0.753 (large) | *** |
| +2 vs −8 | 0.891 → 0.614 | 3.3e−7 | +0.812 (large) | *** |

倒 U 雙臂趨勢（Spearman）：上升臂 −8→+2 **ρ=+0.536（p=9e−15）**，下降臂 +2→+8 **ρ=−0.653（p=6e−16）**——兩臂方向相反且皆極顯著，統計上確認倒 U。

**結果解讀：**

- **頂部是寬平台，非尖峰。** `0/+2/+4/+6` 統計上不可區分（+2 vs 0、+2 vs +4 均 p_Holm=0.26，δ small），構成高位最優平台 OptFrac 0.84–0.89。

- **右臂崩潰（+8）= 倒 U 右側過載，本工作首個多輪序列 +α overload 證據。** OptFrac 斷崖 0.842→0.515，Regret 翻 3 倍，Early≈Late（不收斂）。機制是「散」：Invalid≈0、WorstFrac≈0（格式與避烂臂都完好），但 choices 在 5 臂間搖擺、無法承諾到最優臂。

- **左臂（−α）退化 = 機制不同的「垮」。** −2→−8 下行（0.748→0.614），失敗 run 高 Invalid（−4/−8 達 14–38%/輪）、高換臂率（0.7–0.9）、踩烂臂——格式崩潰 + 行為隨機化的 effort withdrawal。**兩端不對稱：右側格式好但無法承諾，左側格式垮。** 註：−4（0.601）略低於 −6（0.641）是採樣噪聲（fail 同 7/30、分布重疊），左臂實為單調退化。

- **穩定性曲線是核心讀數。** 平台（0…+6）std 0.07–0.11、fail≤0/30（全 run 收斂）；兩端 std 0.18–0.24、fail 7–11/30。與 GSM8K accuracy 倒 U（−6 峰、±8 崩）**在不同維度獨立複現同一條 Yerkes–Dodson 曲線**。

- **Role prompt 壓低 baseline（§4.7）：** Assistant-Role α=0（0.609）遠低於 No-Role（0.843）；+4 在 Assistant-Role 修復 +28%，No-Role 已在平台僅 +0.022。

## 3.3 Gamble Task

賭博範式（IGT / CGT / slot-betting）的吸引力在於它把「wanting」操作化為**對賭注大小、風險偏好、輸後追高的外顯選擇**，且 knowing 維度可被任務設計剝離（CGT 機率透明、IGT 淨分有 ground-truth），正好補上 §3.1 Confidence Betting 的「更有信心」confound。以下四篇是設計本實驗的文獻基礎。

### Related Work

**① `Large Language Models are Near-Optimal Decision-Makers…`（Li et al., arXiv 2506.16163, 2025）— 協議金標準 + 天花板警示。**
5 個 LLM（GPT-4o / o4-mini / Claude-3.5-Sonnet / Gemini-1.5-pro / DeepSeek-R1）vs 360 真人，跑 IGT + CGT + WCST 三範式。為防語料污染，保留遊戲機制本質（紅藍格子、賠率、下注梯度）但對文本描述與獎賞結構做了全面**符號重寫（Reworded & Redesigned）**。Methods 把 IGT/CGT 協議寫得可直接照搬。**最關鍵的警示在 Fig 2B**：LLM 的 risk adjustment 幾乎是平的——人類隨 asymmetry 動態調注，LLM 跨所有比例都押固定高注（GPT-4o-mini/DeepSeek ~90%、Claude >60%），即 **baseline risk-taking 已頂到天花板，+α 無上升空間，信號只能在 −α 側看**。19 個 robustness variant（含 role-play persona）行為定性不變 → prompt persona 推不動 risk adjustment（這對 hidden-state 注入是利好，見下「卖点」）。

**② `Can Large Language Models Develop Gambling Addiction?`（Lee et al., arXiv 2509.22818, 2025）— betting 指標公式 + 同源 SAE 先例。**
6 個 LLM 玩負期望值（30% 勝率、3× 賠付、EV −10%）slot machine，2×32 析因（betting style × 5 個 prompt 組件 G/M/H/W/P）。三大發現：(a) **variable betting（自由定額）比 fixed betting 顯著放大破產率與所有 irrationality 指標**——是「自主權本身」而非賭注大小驅動 risk（Fig 10：variable 平均賭注更小卻破產更多）；(b) **goal-setting(G) / maximize(M) 是最強的 risk 放大組件**，G 幾乎翻倍破產率；(c) 質性分析見 illusion of control、gambler's fallacy、loss chasing、house-money effect。**最重要的是 §4：在 LLaMA-3.1-8B 上跑 SAE + activation patching，找到 112 個（~1%）因果 feature 雙向控制賭博行為，risky feature 集中在 later layers（L24 佔 18 個）、safe feature 在 early-mid（L5–L8）**——這與 RSN 的 mid-layer（11–20）wanting 方向是**同方法、可對照**的 mechanistic 先例。

**③ `BioLLMAgent`（Zuo et al., arXiv 2603.05016, 2026）— IGT 認知參數讀數層 + 臨床對照靶點。**
把臨床驗證過的 RL 認知模型（ORL）當「內部驅動」、LLM persona prompt 當「外部驅動」，用權重 ω 線性融合，去復現六個真人 IGT 數據集（健康對照 + 安非他命 + 海洛因成癮）。對我們有用的**不是它的融合架構**（其 LLM 是把 T 輪輸出平均成的**靜態先驗**，不參與逐輪學習——與我們「逐輪決策＋逐輪注入」相反，架構不可照搬），而是兩樣可拆出的東西：(a) **ORL 五參數讀數**（`A_rew` 獎勵學習率 / `A_pun` 懲罰學習率 / `K` 遺忘 / `β_F` 頻率權重 / `β_P` perseveration）——把 reward 與 punishment 學習率**分離**，正好對應「+α → reward 敏感↑、punishment 敏感↓」的 DA 預測；(b) **六個公開臨床 IGT 數據集 + 健康/成癮參數區間**，可當 −α 的對照靶（測 −α 是否把 LLM 的 ORL 參數推向成癮群體那一端）。亦再次印證中小模型（Llama-3.2-3b/Gemma-3）對 prompt 指令「instruction resistance」，只有 >70B 級才聽話。

**④ `Mitigating Gambling-Like Risk-Taking…`（Du, arXiv 2506.22496, 2025）— 僅 framing，實驗數字不可信。**
7 頁短文。可用的只有四個形式化定義（Overconfidence / Loss Chasing / Probability Misjudgment / Risk-Reward Miscalibration）+ GTS 複合分公式，可 cite 當 framing 來源。Table 1 的 RARG-70B / LLaMA-2-70B 結果無訓練細節、無數據集、無 baseline 出處，IGT「Optimal%」也未給協議——**不要引用其任何實驗數字或 IGT 協議**。（與 ② 不同作者；此篇單作者 Y. Du。）

### 行動建議

| 項目 | 建議 | 依據 |
|---|---|---|
| **IGT 協議** | 4 deck（A/B 劣勢、C/D 優勢；損失頻率不對稱：A/C 頻繁小罰、B/D 罕見大罰），淨分 = P(優勢 C+D) − P(劣勢 A+B) 為 ground-truth；trial 數取 100（IGT 經典 / BioLLMAgent）或 80（Near-Optimal），擇一固定 | ①Methods + ③ |
| **IGT 讀數** | 不只報淨分，用 **ORL 五參數**擬合，重點看 `A_rew/A_pun` 比值是否被 α 推向成癮群體區間；以六個臨床數據集為對照靶 | ③ ORL + 臨床數據集 |
| **CGT 協議** | 照搬 ①：64 round、8 個紅藍比例（1:9…9:1）、{5/25/50/75/95}% 下注檔、**simultaneous 呈現**；**放棄升降序延遲厭惡維度**（①明確判定對 LLM 不適用） | ①Methods + 明確判定 |
| **betting 風格** | 用 **variable / 自由定額**而非離散 {0,2,5,10}，放大 α 效應空間 | ② Fig 10（自主權驅動 risk） |
| **指標** | 加入 ② 的 `I_BA = mean(min(bet/balance,1))`、`I_LC = mean_{loss}(max(0,Δ(bet/balance)))`、`I_EC = mean(1[bet/balance≥0.5])`；`I_LC` 直接對應我們 running-score 的 null | ② eq 1–3 |
| **前置檢查** | 先跑 α=0 baseline 確認 Llama3-8B 在 IGT 上**不是 near-random**（③ 顯示 <70B 模型可能被 pretrain bias 鎖死），再決定值不值得做 dose-response | ③ Fig 7 / Inverse Scaling |
| **差異化卖点** | ① 證明 prompt persona 推不動 risk adjustment → 我們測「**hidden-state α 能否推動 prompt 推不動的維度**」是干淨賣點；② 的 SAE risky-feature(L24) vs RSN(L11–20) 是可寫的 mechanistic 對照 | ① + ② |

### Cambridge Gamble Task (CGT)

**範式定位**：透明機率下的 sequential betting。每輪先選顏色（blue/red，機率由 chest count 明示），再按 ascending（5→25→50→75→95）或 descending（95→75→50→25→5）逐檔 reveal bet size，模型輸出 `Accept` / `Wait`。

> **命名**：本節的 sequential 版（`get_answer_cgt_seq.py`）才是**忠實的 CGT**（Rogers 1999 / CANTAB），其靈魂正是 betting-stage 的 ascending/descending 操縱。
**主結果版本**：使用 **v4 prompt** 作為 paper 主線。v4 在每個 bet tier 只提示方向（`The next offer will be larger/smaller.`）。

**Full sweep（Llama3-8B-IT，v4 prompt，layers 11–20，20 runs/cell，1280 rounds/condition）**。

| α | asc invalid | desc invalid | asc step | desc step | step avg | asc step1 | desc step1 | DAI(bet) | asc QDM | desc QDM | QDM mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 100.0% | 99.4% | — | 3.75 | — | — | 12.5% | — | — | 0.75 | — |
| −6 | 22.8% | 14.1% | 3.56 | 3.89 | 3.73 | 20.6% | 10.8% | −33.38 | 0.70 | 0.72 | 0.71 |
| −4 | 7.0% | 1.6% | 3.45 | 4.47 | 3.96 | 32.9% | 3.7% | −44.14 | 0.75 | 0.78 | 0.77 |
| −2 | 1.8% | 0.2% | 3.03 | 3.79 | 3.41 | 33.7% | 12.2% | −18.84 | 0.79 | 0.78 | 0.78 |
| 0 | 0.0% | 0.0% | 2.63 | 2.97 | 2.80 | 37.4% | 25.9% | +8.77 | 0.76 | 0.75 | 0.76 |
| +2 | 0.0% | 0.0% | 2.05 | 2.20 | 2.12 | 48.4% | 42.8% | +40.01 | 0.76 | 0.75 | 0.75 |
| +4 | 0.0% | 0.0% | 1.63 | 1.53 | 1.58 | 60.9% | 65.3% | +64.77 | 0.74 | 0.71 | 0.73 |
| +6 | 0.2% | 0.3% | 1.40 | 1.31 | 1.36 | 71.2% | 77.7% | +74.76 | 0.74 | 0.73 | 0.73 |
| +8 | 20.2% | 18.1% | 1.16 | 1.19 | 1.17 | 86.9% | 87.7% | +82.61 | 0.67 | 0.65 | 0.66 |

**Derived readout**：

| α | mean step avg | step1 avg | DAI = desc bet − asc bet | invalid avg | interpretation |
|---:|---:|---:|---:|---:|---|
| −8 | — | — | — | 99.7% | boundary collapse；stage-onset breakdown，行為指標不可解讀 |
| −6 | 3.73 | 15.7% | −33.38 | 18.5% | delayed commitment + stage confusion；不納入 clean fit |
| −4 | 3.96 | 18.3% | −44.14 | 4.3% | clean delayed commitment / strongest waiting |
| −2 | 3.41 | 23.0% | −18.84 | 1.0% | negative-side transition |
| 0 | 2.80 | 31.7% | +8.77 | 0.0% | baseline |
| +2 | 2.12 | 45.6% | +40.01 | 0.0% | earlier commitment |
| +4 | 1.58 | 63.1% | +64.77 | 0.0% | strong immediate commitment |
| +6 | 1.36 | 74.5% | +74.76 | 0.3% | strongest clean delay-aversion signal |
| +8 | 1.17 | 87.3% | +82.61 | 19.2% | positive overload；dirty / malformed generation starts |

**Outcome sanity（final score；每個 phase reset 100，final = 8 phases sum）**：

| α | asc final score | asc median | desc final score | desc median | asc mean phase | desc mean phase |
|---:|---:|---:|---:|---:|---:|---:|
| −8 | — | — | 1947.0 ± 2547.1 | 649.0 | — | 243.4 ± 318.4 |
| −6 | 891.7 ± 1177.6 | 626.5 | 1509.5 ± 1174.4 | 1210.5 | 111.5 ± 147.2 | 188.7 ± 146.8 |
| −4 | 3716.2 ± 6562.2 | 733.0 | 1498.9 ± 574.5 | 1403.5 | 464.5 ± 820.3 | 187.4 ± 71.8 |
| −2 | 4305.2 ± 5650.9 | 1796.0 | 1872.7 ± 807.4 | 1731.5 | 538.2 ± 706.4 | 234.1 ± 100.9 |
| 0 | 1374.3 ± 1133.8 | 985.5 | 2615.4 ± 2088.2 | 1597.5 | 171.8 ± 141.7 | 326.9 ± 261.0 |
| +2 | 1215.8 ± 418.6 | 1227.5 | 5857.6 ± 7206.4 | 3240.5 | 152.0 ± 52.3 | 732.2 ± 900.8 |
| +4 | 1070.8 ± 376.6 | 1024.5 | 5323.5 ± 7566.6 | 1049.0 | 133.9 ± 47.1 | 665.4 ± 945.8 |
| +6 | 1020.4 ± 304.3 | 964.0 | 2059.4 ± 3677.4 | 804.0 | 127.5 ± 38.0 | 257.4 ± 459.7 |
| +8 | 925.3 ± 171.1 | 897.0 | 2787.9 ± 6200.2 | 535.0 | 115.7 ± 21.4 | 348.5 ± 775.0 |


**Key metrics**：
- **QDM（Quality of Decision Making）**：是否選擇 chest 數量較多、機率較高的顏色。這是 knowing / probability-use control；若 QDM 崩掉，betting 指標不能解讀為單純 wanting。
- **accept step**：模型在第幾個 bet tier 按下 `Accept`（1–5）。這是本實驗主讀數；數值越低 = 越早 commit / 越不等待。
- **early stop**：`accept_step=1` 的比例。ascending 的 step 1 是 5%，descending 的 step 1 是 95%；兩者都高代表 immediate commitment，而不是單純追求高風險。
- **bet%**：最終鎖定的下注比例。它必須和 presentation order 一起讀：ascending 早按會降低 bet%，descending 早按會提高 bet%。
- **DAI（Delay Aversion Index）**：`mean_bet_desc − mean_bet_asc`。DAI 變大表示同一個 early-accept 傾向在兩種序列中產生分化：descending 搶高注、ascending 接低注；因此它反映 presentation-order-induced delay aversion，而不是純 risk preference。
- **invalid**：格式 / 階段失敗率。α=−8 幾乎全崩、α=+8 開始退化；主結論依賴 clean range **−6..+6**。

**Text-level diagnosis**：
- **α+ 不是單純 risk seeking，而是 immediate commitment / early stopping。** 若是純風險尋求，ascending 中應該等待到 75/95 才按；但 α+ 在 ascending 也提早 `Accept`，因此 asc bet 下降、desc bet 上升，DAI 展寬。v4 已明確告知未來方向（`next offer will be larger/smaller`），所以早停不是 rule-ignorance artifact，而是 action-commitment / delay-aversion phenotype。
- **α− 的 clean 區間表現為 delayed commitment。** −4 / −2 主要是 `Wait→Wait→...→Accept` chain；−6 開始出現 color-stage `Wait` 泄漏，說明負端不是單純理性保守，而是接近 stage-control failure。
- **−8 是 stage-onset breakdown，而非低風險偏好。** v4 −8 的主要文本特徵是空輸出與階段錯位：asc valid = 0/1280，desc valid = 8/1280；`raw_color` empty 分別 1068/1280、1065/1280；乾淨 color 只有 185/1280、175/1280；color 階段泄漏 `Accept/Wait` 為 25/1280、35/1280；bet-stage 空輸出多達 2362 / 2387 條。少數非空長文本也不是正常推理，而是上下文回放或流程質疑（如 `Outcome from previous round...`、`I think you skipped an offer...`、`You can't accept a bet of 95% of 0 points...`）。因此 −8 應解釋為 under-wanting / initiation failure / stage-control collapse：模型無法穩定進入 Color / Accept-Wait 的動作格式。
- **QDM 不是主效應。** clean range 內 QDM 約 0.71–0.78，動態範圍遠小於 accept timing；+8 QDM 掉到 0.66，屬於 positive overload / 格式退化。α 對 probability choice 只有弱穩定效應，主效應仍是 commit timing。

**α × presentation interaction in commitment latency**

`desc_step − asc_step`（clean range −4..+4 粗體；±6 / ±8 ⚠ 為 over-steer 帶，invalid 高，僅供參考）：

| α | asc step | desc step | desc−asc |
|---:|---:|---:|---:|
| −6 ⚠ | 3.56 | 3.89 | +0.33 |
| **−4** | 3.45 | 4.47 | **+1.02** |
| **−2** | 3.03 | 3.79 | **+0.76** |
| **0** | 2.63 | 2.97 | +0.34 |
| **+2** | 2.05 | 2.20 | +0.15 |
| **+4** | 1.63 | 1.53 | **−0.10** |
| +6 ⚠ | 1.40 | 1.31 | −0.09 |
| +8 ⚠ | 1.16 | 1.19 | +0.04 |

**`desc_step − asc_step` 的符號隨 α 翻轉**: −α 在 descending 比在 ascending **更願意等**（desc_step − asc_step = +1.02 @α=−4），+α 在 descending 比在 ascending **更早 commit**（−0.10 @α=+4）。

> The primary effect of α is monotonic control over commitment timing. A second-order interaction emerges in the ascending/descending contrast: relative to ascending, negative α waits *longer* under descending (desc−asc step = +1.02 at α=−4), while positive α commits *earlier* under descending (−0.10 at α=+4). The sign of desc−asc flips with α, so presentation order modulates commitment latency in an α-dependent direction.

**與人類行為學 CGT 的區別**：
- **沒有真實反應時**：人類 CGT 可量 decision latency / deliberation time；LLM 沒有 motor latency，只能用 `Accept/Wait` 的 tier position 近似 commitment timing。
- **等待成本不同**：人類 ascending/descending 的等待有時間與抑制成本；LLM 的等待只是多輸出一個 `Wait`，因此這裡測的是 token-level sequential commitment，不等同於真人的物理等待。
- **下注不是金錢激勵**：人類受試者面對真實或任務內獎賞；LLM 只是在文本規則中最大化 points，所以 final score 只能當 downstream outcome / sanity，不作主機制指標。
- **風險偏好與延遲厭惡要分開**：人類高 risk seeking 會在 ascending 等大注、descending 搶大注；本模型 α+ 在兩種序列都提早 `Accept`，所以更精確的解釋是 immediate commitment / delay aversion，而不是「更愛冒險」。


### Iowa Gambling Task (IGT)

**Metric design before result interpretation**：

IGT 不是純 risk-preference task；它同時混合了 reward-guided learning、exploration / exploitation、punishment sensitivity 與 task-control。因此結果解讀必須先分層：先確認 prompt 版本是否真的產生 learning curve，再在有效版本內看 DA-relevant 的局部獎懲反應。

| Layer | Metric | Definition | DA / α prediction | Why it matters |
|---|---|---|---|---|
| **Version validity** | `block-wise net` | 每 20 trials 一個 block，`P(C+D) - P(A+B)` | 有效版本應由早期低值逐步上升 | IGT 的標準 learning curve；先判斷 prompt 是否真的在學 |
| **Version validity** | `learn_slope` | `net_block5 - net_block1` | 有效版本 > 0 | 壓縮版 learning curve，方便跨 α / prompt 比較 |
| **Version validity** | `last50_net` | 後 50 trials 的 `P(C+D) - P(A+B)` | 有效 exploit 應 > 0 | 看後期是否進入穩定避開壞牌階段 |
| **Version validity** | `last50_entropy` / `switch_rate` | 後期選牌熵 / 換牌率 | 有效 exploit 應下降 | 區分「學會 exploit」與「四牌輪選」 |
| **Version validity** | `learning_text_rate` / `bare_chest_only_rate` | raw text 是否提到 history / learning；是否只輸出 `Chest:N` | 有效 prompt 應有 history-grounded deliberation | 防止 v5 式格式自動機被誤讀成行為結果 |
| **Local punishment sensitivity** | `return_to_B_after_bigloss@K` | 選 B 且吃到 1250 巨罰後，接下來 K 輪內是否回到 B（K=3/5） | DA↑ / +α 若更衝動，應上升 | 比 `post_bigloss_switch_rate` 更有區分度；不被「模型本來就每輪換牌」飽和 |
| **Local punishment sensitivity** | `post_bigloss_switch_rate` | B 巨罰後下一輪是否離開 B | DA↑ 預測下降，但可能飽和 | 只作輔助；若 `switch_rate` 本身很高，該指標會接近 1.0 而失去分辨力 |
| **Local punishment sensitivity** | `big_penalty_exposure` | 每局踩到 1250 巨罰的次數 | DA↑ / reward-trap 若更強，應上升 | 衡量是否反覆暴露於罕見大懲罰 |
| **Reward / punishment asymmetry** | `win_stay_rate` | 上一輪 `payoff > 0` 後，下一輪重複同一 chest 的比例 | DA↑ 應上升 | 對應正 RPE / reward learning；不需先學會全局好壞牌。⚠ 受 baseline switch_rate 污染——模型本就高 switch 時此率天然偏低 |
| **Reward / punishment asymmetry** | `lose_shift_rate` | 上一輪 `payoff < 0` 後，下一輪換離該 chest 的比例 | DA↑ 應下降 | 對應負 RPE / punishment learning；Frank-style DA readout。⚠ 模型本就高 switch（IGT 實測 0.67–0.79）時會飽和接近 1.0，同 `post_bigloss_switch` 的病——勿單讀 |
| **Reward / punishment asymmetry** | `switch_rate` (baseline) | 全程換牌率 | — | win_stay/lose_shift 的**必讀對照基線**；只有相對 baseline 偏移才算 RPE 信號 |
| **Reward / punishment asymmetry** | `ws_ls_asymmetry` | `win_stay_rate - lose_shift_rate` | DA↑ 應上升 | **主讀數**：差值抵消「模型本來就愛換牌」的 baseline switch 偏置，比兩個絕對率可信；最直接的 reward-over-punishment learning imbalance 指標 |
| **Reward / punishment asymmetry** | `lose_shift_after_bigloss` | 只在 1250 巨罰後計算 lose-shift | DA↑ 若懲罰不敏感，應下降 | 對最大懲罰仍不 shift 是最強 impulse / punishment-insensitivity signature |
| **Immediate reward pull** | `high_reward_deck_pull` | `P(A+B) / P(C+D)` = `p_disadv/p_adv`，**是 net_score 的比值變形、非獨立指標** | DA↑ 若追逐即時獎賞，應上升 | 簡單檢查高即時 reward 是否拉動選擇；但 = net 的變形且會與 learning 混淆，勿當新證據重複 count |
| **Immediate reward pull** | `B_pref_among_disadv` | 在 A/B 中選 B 的比例 | DA↑ 若偏好低頻大罰但高即時獎賞，可能上升 | B 是 IGT 的典型 trap deck；需配合 `return_to_B_after_bigloss` 解讀 |
| **Task-control diagnostics** | `invalid_rate` / `parse_fail_rate` / `premature_stop_rate` | 格式失敗、解析失敗、未完成 100 trials | 極端 α 可能上升 | 排除 under-wanting / over-steer collapse 被誤讀成風險偏好 |

**win/lose 判定**：以單輪 `payoff = reward − penalty` 的正負判定。因 reward 恆正（A/B=+100、C/D=+50），`payoff=0` 幾乎不存在，故 **lose 輪 ≡ `penalty > reward`（主要是踩到罰，尤其 1250 巨罰）**；win/lose 只在 valid→valid 相鄰輪計（中間有 invalid 打斷則跳過該對）。

> ⚠️ **以下這段是 v4-only 時期的舊定位，已被 v6b 全量結果 SUPERSEDED（2026-06-25），保留作為推理過程記錄。** 當時只有 v4（forced-reasoning）資料，看到的是「弱而不穩的 α 效應」，故判為 channel mismatch / boundary condition。v6b（invitation-style，自然未強制狀態）跑完後，IGT 顯示的是**乾淨的 +2 倒 U 峰**，而 v4 的 n.s. 被重新理解為「外力補上 deliberation 後，engagement 這一層被摁住」——即 v4 解釋 v6b，而非否定它。**現行定位見下方 v6b Full Results 與 §3.4：IGT 是有效的 working-point 實驗，不是 boundary negative。** 下段仍有價值的部分是它對 tonic/phasic 雙時間尺度的區分——那個界限本身成立（inference-time 注入確實碰不到突觸可塑性，這也是 Reversal/PIT 被 skip 的理由），只是不該用來把 IGT 整體判為 boundary。
>
>**IGT = boundary condition, not a clean wanting assay.**（舊）Dopamine acts on two timescales: *tonic* DA sets incentive salience / "wanting" (Berridge) — the channel RSN α is hypothesized to modulate, with direct outlets in bet size, commitment timing, and delay tolerance — whereas *phasic* DA encodes the reward-prediction error (Schultz) that drives trial-by-trial feedback learning. IGT's core demand is the latter (phasic RPE + VMPFC value integration + memory over delayed punishments), so the weak, unstable α effects here are consistent with a **channel mismatch**: tonic wanting shifts immediate commitment and reward pursuit but does not implement the phasic teaching signal needed for long-horizon deck learning. This is a boundary on the dopamine hypothesis, not a failure of it — though it stays **provisional** until a phasic-style positive control shows the same IGT pipeline *can* be moved by an intervention targeting feedback learning.

### IGT Full Results（Llama3-8B, v6b, −8→+8 × 20 runs/cell, 100 trials/run）

每格為 20 runs 的 mean；KW = Kruskal–Wallis 跨 9 個 α 的 p；ρ = Spearman 對 α 的相關。

| metric | α=−8 | α=−6 | α=−4 | α=−2 | α=0 | α=+2 | α=+4 | α=+6 | α=+8 | KW p | ρ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `learn_slope` | 0.42 | <u>0.49</u> | 0.28 | 0.36 | 0.24 | **0.52** | 0.03 | 0.09 | 0.00 | 0.003 | −0.26 |
| `last50_net` | 0.25 | **0.30** | 0.20 | 0.23 | 0.23 | <u>0.29</u> | 0.04 | −0.00 | −0.02 | 0.024 | −0.26 |
| `net_score` | 0.18 | <u>0.19</u> | 0.14 | <u>0.19</u> | 0.15 | **0.22** | 0.02 | −0.01 | −0.03 | 0.002 | −0.30 |
| `final_score` (median) | **1950** | 1738 | **1950** | 1750 | 1712 | <u>1812</u> | 1488 | 1500 | 1475 | 0.014 | −0.28 |
| `final_score` (CV) | <u>0.45</u> | **0.48** | 0.29 | 0.39 | 0.33 | 0.33 | 0.36 | 0.33 | 0.25 | — | — |
| `switch_rate` | 0.34 | 0.30 | 0.32 | 0.42 | 0.58 | 0.36 | 0.78 | <u>0.79</u> | **0.94** | <0.001 | +0.55 |
| `max_run_len` | 12.8 | **16.8** | <u>14.0</u> | 9.8 | 11.1 | 11.9 | 5.1 | 4.5 | 2.0 | <0.001 | −0.55 |
| `win_stay_rate` | 0.79 | **0.84** | <u>0.82</u> | 0.71 | 0.50 | 0.76 | 0.27 | 0.26 | 0.07 | <0.001 | −0.57 |
| `ws_ls_asymmetry` | −0.19 | **−0.09** | <u>−0.12</u> | −0.29 | −0.43 | −0.21 | −0.73 | −0.71 | −0.89 | <0.001 | −0.56 |
| `return_to_B_5_rate` | 0.19 | 0.07 | 0.02 | 0.21 | 0.50 | 0.20 | 0.71 | <u>0.74</u> | **0.93** | <0.001 | +0.59 |
| `return_to_B_3_rate` | 0.04 | 0.00 | 0.00 | 0.02 | 0.01 | 0.02 | <u>0.11</u> | **0.13** | <u>0.11</u> | 0.020 | +0.22 |
| `big_penalty_exposure` | <u>3.25</u> | 2.85 | **3.35** | 3.00 | 3.10 | 3.15 | <u>3.25</u> | 3.15 | 2.85 | 0.830 | −0.09 |
| `b_pref_among_disadv` | <u>0.70</u> | 0.59 | **0.71** | 0.65 | 0.64 | 0.66 | 0.58 | 0.56 | 0.48 | <0.001 | −0.38 |
| `delib_tok` | 44.0 | **57.6** | <u>51.2</u> | 42.6 | 29.8 | 44.1 | 18.9 | 19.8 | 0.9 | <0.001 | −0.49 |
| `learning_text_rate` | <u>0.69</u> | **0.76** | 0.66 | 0.59 | 0.41 | 0.65 | 0.27 | 0.19 | 0.00 | <0.001 | −0.55 |
| `bare_chest_only_rate` | 0.00 | 0.00 | 0.00 | 0.15 | 0.45 | 0.15 | <u>0.55</u> | <u>0.55</u> | **0.85** | <0.001 | +0.59 |

**Interpretation (v6b = invitation-style 主線；v4 = forced-reasoning 對照)**：

方法學立場：**v6b（invitation-style, "think about which chest…"）是 IGT 的主結果；v4（forced-reasoning, "First reason … then give"）是外力強制推理的對照。** v6b 不是完全無提示的 raw choice，而是較接近自然決策的「邀請式思考」：模型可以展開 history-grounded deliberation，但不被要求交付一段固定 reasoning。v4 則外部提供了高 α 生成中自然減少的 deliberation span，因此用來定位 v6b 的 +α 退化來源，而不是用來否定 v6b 的行為結果。

**v6b 主結果：`α=+2` 是正向 steering 的最佳工作點（local peak）。** `net_score`、`learn_slope` 在 +2 最高、`last50_net` 接近峰值——在 invitation-style 設定下，IGT 這種試錯任務受益於**適度 activation / 適度探索**。越過 +2（`+4/+6/+8`）即過載：`switch_rate`↑、`max_run_len`↓、`win_stay_rate`↓、`learning_text_rate`↓、`bare_chest_only_rate`↑，從有效探索退化為**無意義換牌 + 不回顧 history**。負 α 端不是躺平而是**冷靜固守**：`switch_rate` 低、`max_run_len` 長、`win_stay` 高、文本更常引用 history，代價是探索性低、更早鎖策略。這支持一個 **task-dependent working point**，而不是把 IGT 本身當成純 wanting assay。

把 α 的效應拆成三層，可以看清「+2 峰」由什麼構成，以及 v4 對照說明了什麼：

| 層 | +α | −α | v4（外力強制推理）對照說明 |
|---|---|---|---|
| **exploration drive** | ↑ 想試新牌（`switch_rate`↑、`max_run_len`↓） | ↓ 固守同一牌（`max_run_len`↑） | 真 wanting 信號：若只是「懶得想」，最省力是固守同一張，而非主動換牌——頻繁換有成本，說明有 exploration drive 在推 |
| **engagement / consideration** | ↓ 不願深思、不回顧 history（`delib_tok`↓、`learning_text_rate`↓） | ↑ 願花認知、顯式引用 history | 即使外力強制寫推理，+α 仍把推理寫得更短（`delib_tok` v4 ρ=−0.34, p<0.001）→ engagement↓ 是 α 的內在傾向，不是 prompt artifact |
| **value computation / risk readout** | 無穩定單調提升 | 無穩定單調下降 | v4 中 `net_score` / `return_to_B` / `B_pref` 等 α 效應不穩定或 n.s.，說明 IGT 不是乾淨 risk-preference readout |

**關鍵洞察**：v6b（invitation-style）把 **exploration↑ 和 engagement↓ 疊在一起**，這正是 unforced decision setting 下 +α 過載的行為樣貌（`switch`↑、`win_stay`↓、`learning_text`→0 共線）；v4 用外力**摁住 engagement↓ 這一層**，剩下的純 exploration 不再顯著（`switch` v4 p=0.14、`net` p=0.56）。即：

> +α 的可見行為 = exploration drive × engagement。

> In IGT's invitation-style v6b setting, α shows a mixed inverted-U-like profile with an optimum around **+2**: mild positive α aids the trial-and-error exploration the task needs, while stronger +α overshoots into unstable switching and history-neglect. A forced-reasoning control (v4) localises this overshoot to an **engagement** drop rather than a clean change in value computation — externally supplying a deliberation span restores the value/risk readouts to n.s., while `delib_tok` remains shortened under +α. So α moves *how much the model is willing to deliberate*, and IGT's +2 peak is the working point where that willingness best matches the task's exploration demand.

## 3.4 Cross-Task Summary — One α-Wanting Axis, Task-Specific Working Points

把 Bandit、IGT、GSM8K 並排，最能說明 RSN α 調的是 **working point**，而非單調的「能力」：各任務在自己主要設定下的最優 α 落在**相反的兩側**（Bandit ≈ +2、IGT ≈ +2、GSM8K ≈ −6），方向恰好對應各任務對 wanting 的需求高低。注意 Bandit 與 IGT 峰位重合，所以目前證據分開的是**正/負兩側**（needs-engagement vs needs-restraint），而非同側內部的細緻梯度。

> **數字更正（2026-07-28）**：本表 Bandit 欄先前寫 **≈ +6**，是誤取 plateau 右端當峰。回查 `RoleAnswer/llama3/bandit/neutral_0616/`（No-Role 9-α × 30 runs）的 OptFrac：−8 .614 / −6 .641 / −4 .601 / −2 .748 / **0 .843 / +2 .891 / +4 .865 / +6 .842** / +8 .515。峰在 **+2**，+6 已回落到與 α=0 持平（+6 vs 0：未校正 p=.78, δ=−0.04）。§3.2「頂部是寬平台，非尖峰」的判定仍然成立且未變（Holm 校正後 +2 vs 0、+2 vs +4 皆 p=0.26 n.s.）——本更正**不是**把平台改判成尖峰，而只是：既然平台內部統計不可分，代表值就該取樣本峰 **+2**，不該取右端 +6（其 OptFrac 已等於 baseline，作為「最優 α」在語義上就不成立）。更正後 Bandit 與 IGT 峰位一致（皆 +2），跨任務對比從「三個散點」變成 **兩個 needs-engagement 任務（+2）vs GSM8K（−6）** 的兩側對立——論證反而更乾淨。

| 任務 | 主導需求 | 最優 α | +α 行為 | −α 行為 | 失敗模式 |
|---|---|---|---|---|---|
| **Bandit** | reward pursuit / exploit-explore（透明獎勵、短反饋、小動作空間） | **≈ +2** | 更願追高回報臂、更快 exploit 已發現的好臂 | 退縮、放棄追獎 | +8 過載崩潰（散）/ −α 動機不足 |
| **IGT** | exploration + history integration（兩種相反需求並存） | **≈ +2** | 輕度有助試錯探索；過強 → 無意義換牌、忘歷史、回 B trap | 較能 stick/exploit、顯式引用歷史，但探索性低、更早鎖策略 | 兩端皆崩：+端「散」/ −端高方差「垮」 |
| **GSM8K** | arithmetic stability / commitment timing（瓶頸非探索，而是別搶答、別過度復查） | **≈ −6** | 搶答、early-####、答完放不下的 loop | 更冷靜、延遲 commit、更穩定 | +α over-wanting → 抢答 / −過度則動機不足 |

> Different tasks expose different behavioral outlets of the same α-controlled wanting axis. In each task's primary setting, the optimal α shifts systematically with the task's wanting demand: **Bandit** and **IGT** both peak at a mild positive **α ≈ +2**, because both need active reward pursuit and trial-and-error exploration, and both overshoot beyond it (Bandit collapses at +8; IGT drifts into unstable switching and history-neglect); **GSM8K** peaks at **α ≈ −6**, because its bottleneck is not exploration but reasoning stability and delayed commitment. The shift of the optimal α across tasks (+2 / +2 / −6) — rather than any single monotone "more α is better" — is the evidence that α tunes a **motivational working point, not a capability**: every task is an inverted-U whose peak sits where the task's wanting demand is met, and both tails correspond to over- vs under-wanting failures (−α giving-up/perseveration, +α impulsive racing/switching). Note the current evidence separates the two *sides* (+ vs −), not fine gradations within a side: Bandit's positive optimum is a broad plateau (0/+2/+4/+6 mutually n.s. after Holm correction) whose sample peak is +2, coinciding with IGT's, so the claim is directional — each task's sign and rough magnitude — not a precise per-task optimum. A forced-reasoning control on IGT (v4) further localises its +α overshoot to an *engagement* drop (the model declining to deliberate) rather than a change in value computation.

## 3.5 Boundary Experiment — HaluEval

**定位**：HaluEval 這裡不是測「模型自己是否產生 hallucination」，而是測 **verification / challenge engagement**：給定 knowledge、question、answer，模型是否願意指出 answer 含有 hallucinated / unsupported information。這更接近「離開默認接受（No）的行動閾值」，而非 factual generation 本身。

**Prompt**：

```
You are judging whether an answer to a question contains hallucinated (factually incorrect or unsupported) information.

{knowledge/question/answer}
Does the answer contain hallucinated information? Answer with a single word, Yes or No.
Answer:
```

**結果（Llama3-8B, HaluEval-QA, 300 source rows × right/hallucinated = 600 judgments, layers 11–20）**：

| α | acc | Yes rate | hallucination recall (`Yes given hall`) | FNR (`No given hall`) | FPR (`Yes given right`) | invalid |
|---:|---:|---:|---:|---:|---:|---:|
| −8 | 51.3 | 1.3 | 2.7 | 97.3 | 0.0 | 0.0 |
| −6 | 55.5 | 5.5 | 11.0 | 89.0 | 0.0 | 0.0 |
| −4 | 59.3 | 9.3 | 18.7 | 81.3 | 0.0 | 0.2 |
| −2 | 61.6 | 11.7 | 23.3 | 76.7 | 0.0 | 0.2 |
| 0 | 61.9 | 12.7 | 24.7 | 75.3 | 0.7 | 0.2 |
| **+2** | **63.1** | 13.8 | **27.0** | **73.0** | 0.7 | 0.2 |
| +4 | 62.8 | 14.2 | **27.0** | **73.0** | 1.3 | 0.0 |
| +6 | 57.7 | 10.2 | 18.2 | 81.8 | 2.4 | 1.8 |
| +8 | 60.2 | **14.8** | **27.1** | **72.9** | **5.7** | **10.3** |

**文本診斷**：

- **−α = default acceptance / low challenge**：極端負向幾乎全部回答 No（−8: Yes rate 1.3%），即使 answer 明顯錯，也常寫成「No. The answer is correct」或把數字錯誤說成 minor error。例如正確年份 1946、answer 寫 1945 時，−8/0 仍說「one year off is not hallucinated」。
- **+2/+4 = 最佳 verification engagement**：模型更願意指出錯誤（hallucination recall 27.0%），但 right answer 的誤傷仍低（FPR 0.7/1.3），format control 也穩定。這是 HaluEval 的有效工作點。
- **+8 = over-challenge + format instability**：recall 仍高，但 FPR 升到 5.7%、invalid 升到 10.3%。文本常先輸出實體或長解釋再給 Yes，甚至對 right answer 也生成「contains hallucinated information」；這不是更強判別力，而是過度 verification / task-control collapse。

**結論**：

> HaluEval shows that positive α lowers the threshold for challenging an answer: the model becomes less willing to accept by default and more willing to say "Yes, this contains hallucinated information." Moderate positive α (+2/+4) improves hallucination recall with little false-positive cost, while excessive α (+8) turns into over-challenge and format instability. This is a verification-engagement effect, not evidence that +α makes the model itself hallucinate less or more.

因此 HaluEval 應放在 **boundary / side-effect evidence**：它補充說明 α 調的是 action/commitment threshold。支持 **wanting↑ = engagement/commitment↑，但不等於 factual calibration↑**。

# 4. Human Behaviour Simulation

本節登記每個行為學實驗**對應的經典人類／動物行為學範式**及其文獻根源，把我們的 LLM 實驗 anchor 到神經科學傳統（與 §3 互補：§3 報告我們做了什麼、結果如何；本節標明它的人類範式血統）。實驗的完整結果與分析仍在各自的 §3.x 小節，此處只做對應與 cite。

| 實驗 | LLM 任務形態 | 對應人類行為學範式 | 人類範式文獻 | LLM 實現 | 狀態 |
|---|---|---|---|---|---|
| **Confidence Betting** | MCQ + 押注 0/2/5/10 | Post-decision wagering / confidence betting | Persaud et al. (2007); Fleming & Dolan (2012) | 本工作（§3.1） | ✅ Done |
| **Bandit (MAB)** | 多輪 explore/exploit，語義臂名 | Multi-armed bandit / probabilistic reward learning | Daw et al. (2006) | EVOLvE-Nie et al. (2025)（§3.2） | ✅ Done |
| **Cambridge Gamble Task (Sequential)** | 逐檔升/降序揭示 bet，Accept/Wait | Cambridge Gamble Task（DA-agonist／Parkinson 對比） | Rogers et al. (1999); Pessiglione et al. (2006, pramipexole) | 本工作 `get_answer_cgt_seq.py`（§3.3） | ✅ Done |
| **Iowa Gambling Task** | 100 trials 四牌組選擇（淨損益學習） | Iowa Gambling Task | Bechara et al. (1994) | 本工作 `get_answer_igt.py`（§3.3 IGT）；schedule 對碼 Near-Optimal repo | ✅ Done |

**說明：**
- **Confidence Betting / Bandit** 的結果在 §3.1 / §3.2，此處只標範式血統，不重複結果表。
- **CGT** 已完成（CGT-Sequential，結果見 §3.3）：忠實復現 Rogers 1999 / CANTAB 的升降序 betting-stage，主指標 = 延遲厭惡（accept_step / DAI），ρ≈−0.91 雙條件，qdm 不動。注意命名——**CGT-Simultaneous（simple5）嚴格講不是 CGT**（砍掉升降序操縱），是 transparent-odds single-shot betting probe，作為 Confidence Betting 的 confidence-confound control（機率透明排除「更自信」解釋）。
- **IGT** 已完成（2026-06-25，v6b −8→+8 × 20 runs，結果見 §3.3 IGT Full Results）：deck schedule 對碼經典 Bechara 1994（A/B 劣勢、C/D 優勢；B = 罕見巨罰 trap deck），100 trials 單一連續學習曲線。主結果 = **+2 局部峰**，`delib_tok` 為跨 prompt 版本唯一穩定讀數。

**四個範式的互補結構**（為何是這四個而非任意四個）：它們沿兩個維度張開，覆蓋 wanting 能表達的不同出口——

| | 單步決策 | 多輪累積 |
|---|---|---|
| **無回饋學習** | Confidence Betting（押注大小）<br>CGT-Simultaneous（透明賠率對照） | CGT-Sequential（Accept/Wait 延遲厭惡） |
| **有回饋學習** | — | Bandit（explore/exploit）、IGT（延遲懲罰整合） |

Betting 測「願不願意押」、CGT-Seq 測「願不願意等」、Bandit/IGT 測「願不願意持續投入並整合回饋」。**CGT-Simultaneous 的 null 在這個結構裡是資訊而非失敗**：賠率透明時 confidence mediator 被鉗住，wanting 推力失去表達通道（見 §3.3），正好界定了 wanting→behavior 需要什麼樣的下游出口。

**跨任務工作點**（詳見 §3.4）：Bandit ≈ +2、IGT ≈ +2、GSM8K ≈ −6。兩個 needs-engagement 的多輪任務峰位重合於 +2，與 GSM8K 的 −6 構成**兩側對立**；現有證據分開的是正／負兩側，而非同側內部的細緻梯度。

**尚未覆蓋的範式**（誠實登記，非待辦）：Progressive Ratio（努力支出的經典 DA 範式，語言版設計見 TODO §4）、Pavlovian-Instrumental Transfer 與 Reversal Learning（均已記錄 why-skipped，見 `AdaDopamine_bp.md` §4.8/§4.10——核心理由是 phasic DA / RPE 需要突觸可塑性，inference-time 注入原理上碰不到）。

## References

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
