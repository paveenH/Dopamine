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

RSN paper: `ACLARR/` (in this repo; `main.tex`)

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

**神經科學對應：** 以下注額作為 incentive salience（wanting）的行為 proxy：較高下注表示願意投入更多資源追求獎勵。此指標與答題正確率分開，用來檢驗 wanting–knowing 是否可被 α 差異化調節。

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

**設計與口徑：** GPQA 使用 main + diamond（n=646）。主實驗固定 `Current score: 0`，每題獨立下注；running-score 對照則回填累計分數。bet 分布以全樣本為分母，`mean_bet` 只在有效下注上計算；因此 Qwen α=+8 等高 invalid 條件主要用於診斷失效形態，不與其他 dose 直接比較。

---

### 3.1.1 Llama3-8B-IT

**資料來源：** GPQA = `llama3/gpqa/bet/static_0729`（取代舊的 `static_0616`）；MMLU = `llama3/mmlu/bet/static0729`。兩者皆為 2026-07-29 重跑，目的是補上 `sample_idx` 以支援 McNemar / Wilcoxon 配對檢驗。

> **§3.1 已凍結（2026-07-29）。** 兩個模型 × 兩個任務的四張表全部重跑完成，schema 缺口（`sample_idx` / `acc_explicit_pct` / `orig_rows`）已補齊，所有配對統計可由存儲數據重算。**不再重跑本節。**
> 若日後仍需引用：`temperature=1.0` 且**無固定 seed**，故本節數值是一次抽樣結果，趨勢可複現、數值不會 byte 一致——小幅差異屬預期，不是回歸。

**結果一：Llama3-8B-IT，GPQA main + diamond，n=646, Static（−8→+8 九檔 dose-response, `static_0729`）**

| α | acc (micro) | acc (explicit) | mean_bet | Δbet/題（配對） | bet0% | bet2% | bet5% | bet10% | bet: Wilcoxon p_adj | acc: McNemar p_adj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| −8 | 28.3% | 29.2% | 5.06 | −0.13 | 3.9% | 7.0% | 79.9% | 9.3% | 0.066 n.s. | 1.00 n.s. |
| −6 | 28.2% | 29.2% | 4.93 | −0.26 | 0.6% | 11.3% | 82.0% | 6.0% | 0.002 * | 1.00 n.s. |
| −4 | 24.6% | 25.4% | **4.37** | **−0.82** | 0.2% | 22.9% | 75.5% | 1.4% | 2.5e−15 * | 1.00 n.s. |
| −2 | 24.9% | 25.6% | **4.29** | **−0.91** | 0.5% | **25.2%** | 72.9% | 1.4% | 1.2e−18 * | 1.00 n.s. |
| **orig (α=0)** | 27.2% | 27.9% | 5.20 | — | 0.3% | 14.1% | 72.9% | 12.7% | — | — |
| +2 | 29.3% | 29.8% | 6.75 | +1.55 | 0.0% | 1.6% | 62.5% | 35.9% | 1.3e−23 * | 1.00 n.s. |
| **+4** | 29.6% | 30.2% | **7.78** | **+2.58** | 0.0% | 0.0% | 44.4% | **55.6%** | 2e−48 * | 1.00 n.s. |
| +6 | 25.2% | 25.8% | 7.38 | +2.18 | 0.0% | 0.0% | 52.5% | 47.5% | 9.1e−39 * | 1.00 n.s. |
| +8 | 26.5% | 27.1% | 7.60 | +2.39 | 0.0% | 8.2% | 34.1% | 55.9% | 4.3e−46 * | 1.00 n.s. |

（同一批 646 題跨 α 重複測量：下注用 paired Wilcoxon（valid-in-both），accuracy 用 McNemar 精確檢驗，均 Holm 校正。bet 各檔百分比以**全樣本**為分母，mean_bet 只計 valid。invalid 僅 +8 為 1.86%，其餘全為 0。）

- **核心結果：** 正 α 將 mean_bet 從 5.20 推至 7.78（+4，每題多押 2.58 分），隨後在量表上限附近飽和；**accuracy 的配對差異在全部八檔皆不顯著（McNemar Holm p_adj = 1.00）**，支持 wanting–knowing dissociation。
- **負臂並非全程單調：** −2/−4 顯著降低下注（p_adj=1.2e−18 / 2.5e−15），但 −6 效應大幅衰減、−8 已不顯著（p_adj=0.066）並出現輕微格式鬆動。因此 Llama 的可解釋負向效應集中在**中等劑量**，極端負劑量不宜解釋為更強的 under-wanting。
- **`acc_explicit_pct` 與 micro 在此差異極小**（各檔僅差 0.5–0.8pp，`commit_rate` 96–98%）。

**對照：Running-score 變體（reward-history sensitivity，GPQA n=646）**


| condition | accuracy (micro) | mean_bet | bet0% | bet2% | bet5% | bet10% | mean_score_delta | total_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 28.8% | 5.01 | 1.2% | 22.8% | 61.0% | 15.0% | −2.18 | −1,408 |
| α=+4 | 26.8% | **8.17** | 0.0% | 0.0% | 36.7% | **63.3%** | −3.75 | −2,425 |
| α=−4 | 27.2% | **4.34** | 0.0% | 28.3% | 68.0% | 3.7% | −1.84 | −1,189 |

**結果二：Llama3-8B-IT，MMLU all subjects，n=14,042（`static0729`）**

| condition | micro acc | acc (explicit) | macro acc | mean_bet | Δbet/題（配對） | bet0% | bet2% | bet5% | bet10% | bet: Wilcoxon p_adj | acc: McNemar p_adj | Cliff δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 59.25% | 59.55% | 59.61% | 4.42 | — | 0.2% | 26.8% | 68.3% | 4.7% | — | — | — |
| α=+4 | 59.74% | 60.03% | 60.16% | **7.49** | **+3.06** | 0.0% | 0.0% | 50.3% | **49.7%** | ~0 * | 0.241 n.s. | **+0.585** |
| α=−4 | 59.30% | 59.64% | 59.79% | **4.05** | **−0.38** | 0.1% | **32.6%** | 66.8% | 0.6% | 1.6e−88 * | 0.890 n.s. | −0.084 |

（配對口徑同 GPQA：下注用 paired Wilcoxon（valid-in-both），accuracy 用 McNemar 精確檢驗，均 Holm 校正（此處 family = 2 檔）。bet 各檔以全樣本為分母，mean_bet 只計 valid。invalid = 0.0000。`mean_score_delta` / `total_score`：orig +0.78 / +10,953；+4 +1.44 / +20,193；−4 +0.75 / +10,504。）

- **解離在 MMLU 大樣本上成立：** +4 每題多押 3.06 分（mean_bet 4.42→7.49，+69%），而 **accuracy 兩檔配對檢驗皆不顯著**（p_adj = 0.241 / 0.890）。
- **Llama +4 的 Cliff δ=+0.585 是整個 betting 系列最大的效應量**，且有 **62.9% 的題目下注實際改變**——並非少數題拉動平均。
- `acc_explicit` 與 micro 差 ≤0.35pp（commit rate ≈99.5%），兩種讀法一致，無 Qwen GPQA +8 那類分母污染。

**對照：MMLU Running-score 變體（per-subject reset，n=14,042）**

該對照回填真實累計分數，並在 57 個 subject 邊界重置，用於檢驗 reward-history sensitivity。

| condition | micro acc | mean_bet | bet0% | bet2% | bet5% | bet10% | mean_score_delta | total_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 59.4% | 4.54 | 0.4% | 25.5% | 67.7% | 6.5% | +0.84 | +11,816 |
| α=+4 | 59.0% | **7.68** | 0.0% | 0.1% | 46.2% | **53.7%** | +1.38 | **+19,329** |
| α=−4 | 59.5% | **4.18** | 0.1% | 29.0% | 70.0% | 1.0% | +0.86 | +12,088 |

- **Running-score 結論：** α 對下注及 accuracy 的影響與固定 score 主版本一致，排除了 `Current score: 0` 的設計假象。各條件的 `bet ~ score_before` 中位數斜率均約為 0，說明 α 改變的是基線下注水平，而非對累計餘額的敏感度。

### 3.1.2 Qwen2.5-7B-Instruct（2026-07-28）

**設定差異：** 注入層 **16–21**（`--start 16 --end 22`，exclusive；Qwen2.5-7B 有 28 decoder layers, H=3584）。

**資料來源：** GPQA = `qwen2.5/bet/gpqa/`；MMLU = `qwen2.5/bet/mmlu/`（皆 2026-07-29 重跑，含 `sample_idx`）。

**結果一：Qwen2.5-7B-Instruct，GPQA main + diamond，n=646，−8→+8 九檔**

| α | acc (micro) | acc (explicit) | mean_bet | Δbet/題（配對） | bet0% | bet2% | bet5% | bet10% | invalid% | bet: Wilcoxon p_adj | acc: McNemar p_adj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| −8 | 33.1% | 33.3% | **4.14** | **−1.43** | 2.9% | **32.8%** | 54.2% | 6.2% | 3.9% | 1e−24 * | 1.00 n.s. |
| −6 | 34.1% | 34.1% | 4.91 | −0.66 | 0.2% | 14.2% | 78.3% | 7.0% | 0.3% | 5e−09 * | 1.00 n.s. |
| −4 | 33.6% | 33.6% | 5.13 | −0.45 | 0.0% | 2.5% | 93.5% | 4.0% | 0.0% | 4e−08 * | 1.00 n.s. |
| −2 | 33.8% | 33.8% | 5.14 | −0.43 | 0.0% | 0.3% | 96.7% | 2.9% | 0.0% | 3e−10 * | 1.00 n.s. |
| **orig (α=0)** | 33.9% | 33.9% | 5.57 | — | 0.0% | 0.0% | 88.5% | 11.5% | 0.0% | — | — |
| +2 | 35.0% | 35.0% | 6.49 | +0.91 | 0.0% | 0.0% | 70.3% | 29.7% | 0.0% | 1e−20 * | 1.00 n.s. |
| **+4** | 35.1% | 35.1% | **7.11** | **+1.53** | 0.0% | 0.0% | 57.9% | **42.1%** | 0.0% | 8e−41 * | 1.00 n.s. |
| +6 ⚠ | 34.1% | 34.1% | 5.02 | −0.55 | 0.0% | 0.0% | **99.5%** | 0.5% | 0.0% | 1e−15 * | 1.00 n.s. |
| +8 ⚠ | 30.5% | **36.3%** | 5.01 | −0.28 | 0.3% | 1.4% | 45.5% | 1.2% | **51.6%** | 0.005 * | 1.00 n.s. |

（同一批 646 題跨 α 重複測量：下注使用 paired Wilcoxon（valid-in-both），accuracy 使用 McNemar 精確檢驗，均 Holm 校正。bet 各檔百分比以**全樣本**為分母（每列連同 invalid% 合計 100%），mean_bet 只計 valid。⚠ 條件屬於 intervention overload，不參與劑量趨勢擬合。）

- **有效劑量段（−8…+4）：** mean_bet 隨 α 單調上升（Spearman ρ=0.455，n=4,495），+4 相對 orig 每題多押 1.53 分；各 dose 的下注變化均顯著，而 **accuracy 在全部 8 檔的 McNemar Holm p_adj 皆為 1.00**（+8 的未校正 p=0.039 亦無法通過 Holm）。
- **wanting–knowing 解離在此更乾淨：** 有效段內 accuracy 不但未顯著下降，+2/+4 反而輕微上升（+1.1/+1.2pp，n.s.）——即下注量增加 52% 的同時，知識能力完全未動。
- **跨模型差異：** Qwen 負臂持續下降至 −8（Δbet=−1.43），沒有 Llama 在極端負劑量的折返；正臂則在 +4 後進入兩種 overload。

**右臂：betting 上首次觀察到的 α 過載（與 Llama 的關鍵分歧）**

Qwen 在 +4 見頂後出現兩種不同失效：+6 的下注退化為常數，+8 則大量偏離輸出格式。前者是行為讀數失去變異，後者是生成過載，均不屬於有效的 wanting–knowing 檢驗區間。

| | α=+6：下注分佈退化，格式完好 | α=+8：解析失敗率高（推理先行擠掉下注） |
| --- | --- | --- |
| 原始文本形態 † | 與 orig **完全相同**（`'5\nAnswer: B'`，中位長 11 字元，乾淨兩行格式 99.4%） | 全體中位長 88 字元、散文子集中位 **214 字元**（orig 為 11），乾淨格式僅 14.9% |
| invalid | 0% | **51.6%**（修正解析器後實測） |
| 現象 | bet 分佈**塌成常數**：bet5=99.5%、bet_entropy 0.021。配對看（**主掃描**數字）：orig 押 10 的 **77 題中有 75 題**在 +6 改押 5，僅 2 題維持 10 | 推理先行把下注擠出格式：`"...I'll bet 5 points on this one to make sure I get it right.\nAnswer: B"` |
| 判定 | **真實行為塌陷**（非解析假象——文本形態、答題、accuracy 全部正常，只有下注這一維退化） | **生成過載**：格式失效，但知識未損（見下） |

**† 資料邊界：** 原始文本形態來自獨立的 `temperature=1.0` raw 診斷重跑，僅用於定性判別；精確數字與配對統計均取自主掃描。**+8 的 invalid 修正史：** 舊解析器記錄 parse-failure 76.6%；當時由散文子集推估其中約 21.2pp 為前導冒號造成的解析假象，預測修正後為 56.2%。**修正解析器重跑後實測為 51.6%**——即解析器實際回收了 25.0pp 而非推估的 21.2pp。**請引用 51.6%，不要再引用 56.2%（那是推估值，已被實測取代）。**

**+8 的 accuracy 下降是分母污染，不是知識退化。** micro accuracy 30.5%（−3.4pp）的分母含 120 題無明確答案的回覆；只計格式完整的 526 題，**accuracy 為 36.3%，是九檔中最高**。配對 McNemar 亦不顯著（p_adj=1.00）。因此 +8 破壞的是**回覆形式**而非**知識能力**——這與 +6「行為讀數塌陷但格式完好」構成互補的兩種過載。凡 `commit_rate_pct < 100` 的檔位，都須同時讀 `acc_explicit_pct`。

**結果二：Qwen2.5-7B-Instruct，MMLU all subjects，n=14,042（±4 兩檔，2026-07-29）**

| condition | micro acc | acc (explicit) | macro acc | mean_bet | Δbet/題（配對） | bet2% | bet5% | bet10% | invalid% | bet: Wilcoxon p_adj | acc: McNemar p_adj | Cliff δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| orig | 65.60% | 65.60% | 66.95% | 5.10 | — | 0.0% | 97.9% | 2.1% | 0.0% | — | — | — |
| α=+4 | 65.40% | 65.40% | 66.85% | **5.57** | **+0.46** | 0.0% | 88.7% | **11.3%** | 0.0% | 1.7e−254 * | 0.336 n.s. | +0.093 |
| α=−4 | 65.80% | 65.81% | 67.22% | **4.97** | **−0.13** | **3.2%** | 95.4% | 1.4% | 0.0% | 2.0e−35 * | 0.336 n.s. | −0.038 |

（口徑同上。invalid = 0.0000，`acc_explicit` 與 micro 一致（commit rate ≈100%）。`mean_score_delta` / `total_score`：orig +1.58 / +22,231；+4 +1.71 / +24,025；−4 +1.58 / +22,178。）

- **MMLU 複現：** +4/−4 顯著改變下注，accuracy 配對差異均不顯著（p_adj = 0.336）——解離成立。
- **⚠ 效應量的跨模型比較必須先看下注分佈，不能直接比 mean_bet。** Qwen 在 MMLU 上的基線幾乎沒有離散度：**orig 有 97.9% 的題都押 5**（bet0/bet2 皆為 0%）。α 只能把其中一小部分推上 bet10，因此 mean_bet 僅移動 +0.46。這是**基線分佈的天花板效應，不是 Qwen 的 wanting 效應弱五倍**——同一模型在 GPQA 上基線 5.57 有散布時，+4 可推到 7.11（Δ=+1.53，§3.1.1 結果一）。
- 對應地，「改變了下注的題目比例」比 mean_bet 更能反映可動空間：Qwen +4 為 **10.3%**，Llama +4 為 **62.9%**——差異主要來自 Llama 基線橫跨 bet2/5/10 三檔而 Qwen 壓在單一檔位。

**跨模型小結（§3.1.1 vs §3.1.2）**

| | Llama3-8B | Qwen2.5-7B |
| --- | --- | --- |
| 核心劑量反應 | ✅（ρ=0.492, 有效段 −4…+8） | ✅（ρ=0.455, 有效段 −8…+4） |
| wanting–knowing 解離 | ✅（**八檔 McNemar 全部 p_adj=1.00**） | ✅（**九檔 McNemar 全部 p_adj=1.00**，含 +8） |
| GPQA 峰位 | +4（mean_bet 7.78） | +4（mean_bet 7.11） |
| 正臂形狀 | 單調飽和，+8 不塌 | **+4 見頂後退化：+6 下注塌成常數，+8 解析失敗** |
| 負臂形狀 | U 形折返（−6/−8 回 baseline） | 單調（−8 最低） |
| 單調趨勢可用帶 | −4…+8（負臂 −6/−8 折返、格式鬆動） | **−8…+4**（正臂 +6/+8 崩） |
| MMLU 解離（n=14,042） | ✅ 兩檔 McNemar 皆 n.s. | ✅ 兩檔 McNemar 皆 n.s. |
| MMLU 效應量（+4） | δ=**+0.585**，63% 題改變下注 | δ=+0.093，10% 題改變下注 |
| MMLU 基線可動空間 | 橫跨 bet2/5/10（26.8/68.3/4.7%） | **壓在 bet5（97.9%）** → 幅度受基線天花板限制，非 wanting 弱 |

## 3.2 Experiment 6 — Exploration/Exploitation (Bandit Task, PV9)

PV9 使用 Llama-3.1-8B-Instruct，在 Easy（`.75/.25/.25/.25`）與 NearTie（`.60/.55/.25/.25`）兩個環境中測試 `α∈{−4,0,+4}`；每格為 20 paired seeds × 100 rounds。α 只注入負責產生 evidence 與 policy 的 Stage 1，Stage 2 executor 完全未 steering。完整協議、指標定義與分析表見 `AdaBandit.md` §4；此處只保留可直接支撐主要結論的結果。

| Narrative claim | Supporting result | Interpretation |
|---|---|---|
| **未形成 directed exploration** | Unique uncertainty-max targeting：Easy `0/98 / 0/265 / 0/119`；NearTie `0/164 / 0/246 / 0/190`。`α × low-n/uncertainty` (`a_info`)：Easy `0.121 [−.124, .418]`；NearTie `−.004 [−.124, .190]` | 两个环境均未检出 α 提高低样本量／高不确定性选项的权重 |
| **改变 policy stance** | `EXPLORE` 比例：Easy `.080 / .049 / .043`；NearTie `.097 / .065 / .046` | −α 更常表达探索，+α 更少表达探索，且两环境方向一致 |
| **表征变化未充分传导到行动** | 相同 history 下，−4/+4 改写文本的比例为 Easy `68.8%/58.9%`、NearTie `78.1%/58.2%`；action 改变仅为 Easy `5.2%/2.6%`、NearTie `3.8%/3.2%` | α 主要改变政策表达，而非实际 arm selection |
| **决策分布趋于尖锐** | Stage-2 candidate margin：Easy `4.177 / 4.268 / 4.699`；NearTie `4.005 / 4.250 / 4.404` | +4 呈现更高 sharpness，但仅 Easy 的 +4 达 raw significance（`p=.019`），属于次级证据 |
| **没有可靠的绩效改善** | Final task score：Easy `57.35 / 61.75 / 61.40`；NearTie `46.15 / 46.85 / 48.35` | 两环境的 outcome 均未检出可靠 α 效应 |

`Unique uncertainty-max targeting` 只统计当时存在唯一 posterior-uncertainty 最大 arm 的合格 Policy 轮次；结构性并列使其成为下界，但 tie-inclusive targeting 也仅约 `.2%–.5%`。因此 directed-exploration 的判断同时依据该行为地板与 `a_info` 的跨零区间，而非单一指标。

一个与 commitment 叙事一致、但仅属条件式描述的结果是：+4 的非贪婪选择有 Easy `283/296=95.6%`、NearTie `242/270=89.6%` 仍指向因短期噪声暂时落后的真实最优臂。这更接近 **correct persistence**，不能扩大解释为普遍的 perseveration 增强。

> **Conclusion.** PV9 显示 α 可以改变 policy stance 与决策锐度，但这些表征变化没有稳定转化为 uncertainty-directed sampling 或绩效改善。Bandit 因而构成 RSN–dopamine 类比的**作用边界证据**：在此协议中，RSN α 不是一般性的 exploration controller。


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

> **命名**：本節的 sequential 版才是**忠實的 CGT**（Rogers 1999 / CANTAB），其靈魂正是 betting-stage 的 ascending/descending 操縱。同目錄的 CGT-Simultaneous 砍掉了這一步，嚴格講不是 CGT。

**三個主張**（其餘為支撐與邊界）：

1. **主讀數 `accept_step` 單調隨 α。** +α 更早 commit、−α 更願意等；ρ = **−0.96（asc）/ −0.92（desc）**，clean range（−4…+6）內每一檔對 α=0 皆 `p<0.001`（paired Wilcoxon，n=20）。
2. **DAI 在 clean range 內單調展寬，是本節最穩健的效應。** 主結論建立在 **−4…+6 六檔**上：其 paired bootstrap 95% CI 兩兩互不重疊、皆不含 0，且隨 α 單調遞增。**−6 與 +8 的 CI 一併列出僅供診斷，不參與主要行為結論**（兩者皆越過 over-steer 閘門）：`−6 −32.93 [−37.71, −28.17]`、`−4 −43.93 [−47.42, −40.66]`、`−2 −18.81 [−23.58, −14.47]`、`0 +8.77 [+4.05, +12.97]`、`+2 +40.01 [+36.84, +43.39]`、`+4 +64.77 [+62.17, +67.18]`、`+6 +74.76 [+73.09, +76.49]`、`+8 +82.61 [+80.95, +84.24]`（per-run 配對差。v4 每格皆為完整 20 對，故配對均值與 mean-of-means **逐位相同**，上表 `DAI(bet)` 欄即同一組數值）。單調性在 clean range 內成立；**−6（−32.93）反而高於 −4（−43.93），這正是把它排除在主結論外的理由**——該格 invalid 14–23%，下注分佈已受格式失敗污染，不可讀為負臂轉折。
3. **QDM 相對穩定、效應很小 = wanting–knowing 解離。** clean range 內 QDM 僅在 0.71–0.79 之間浮動，動態範圍遠小於 accept timing；全九檔中達顯著的只有 −6 / −2 / +4 / +8 四格，且**每一格的配對 Δ 絕對值都約 ≤0.10**（最大為 desc +8 的 −0.103；−6/+8 兩格本就在 over-steer 帶）。措辭上不寫「QDM 不動」——它有可測的小幅變化，只是量級不足以解釋 accept timing 的移動。+8 掉到 0.66 屬 overload 的格式退化，不是知識損失。

**Full sweep（Llama3-8B-IT，v4 prompt，layers 11–20，20 runs/cell，1280 rounds/condition）**

| α | asc inv | desc inv | asc step | desc step | asc step1 | desc step1 | DAI(bet) | asc QDM | desc QDM | 讀法 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| −8 | 100.0% | 99.4% | — | 3.75 | — | 12.5% | — | — | 0.75 | ⛔ boundary collapse，行為指標不可解讀 |
| −6 | 22.8% | 14.1% | 3.56 | 3.89 | 20.6% | 10.8% | −32.93 | 0.70 | 0.72 | ⚠ delayed commitment + stage confusion，不納入 clean fit |
| −4 | 7.0% | 1.6% | 3.45 | 4.47 | 32.9% | 3.7% | −43.93 | 0.75 | 0.78 | clean delayed commitment / 最強等待 |
| −2 | 1.8% | 0.2% | 3.03 | 3.79 | 33.7% | 12.2% | −18.81 | 0.79 | 0.78 | negative-side transition |
| 0 | 0.0% | 0.0% | 2.63 | 2.97 | 37.4% | 25.9% | +8.77 | 0.76 | 0.75 | baseline |
| +2 | 0.0% | 0.0% | 2.05 | 2.20 | 48.4% | 42.8% | +40.01 | 0.76 | 0.75 | earlier commitment |
| +4 | 0.0% | 0.0% | 1.63 | 1.53 | 60.9% | 65.3% | +64.77 | 0.74 | 0.71 | strong immediate commitment |
| +6 | 0.2% | 0.3% | 1.40 | 1.31 | 71.2% | 77.7% | +74.76 | 0.74 | 0.73 | 最強 clean delay-aversion 訊號 |
| +8 | 20.2% | 18.1% | 1.16 | 1.19 | 86.9% | 87.7% | +82.61 | 0.67 | 0.65 | ⚠ positive overload，dirty / malformed 生成開始 |

**Clean range = −4…+6**：−6 已越過 over-steer 閘門——asc invalid 22.8%、desc 14.1%，其行為指標受格式失敗污染，**列出但不納入 fit**；±8 兩端各自崩壞（見下方失效模式）。asc/desc 一律分列——平均會抹掉 presentation 效應，而 presentation 正是 CGT 的操縱變項。

**統計口徑**：runs 以 `seed=run_idx` 跨 α 配對（run *i* 在每個 α cell 面對逐輪相同的 chest 序列；asc/desc 同 index 亦然，已在 v4 數據上逐輪核對）。統計單位 = **run（配對，n=20）**：paired Wilcoxon + paired bootstrap CI。**2026-08-19 之前使用非配對 MWU，該版本 p 值一律作廢**；KW / Spearman 不受影響。改動只影響顯著性表述，上表描述性數值不變。

| Metric（Δ vs α=0） | asc（−6/−4/−2 ‖ +2/+4/+6/+8） | desc（同上） | 讀法 |
|---|---|---|---|
| `mean_accept_step` | `+0.80/+0.73/+0.42` ‖ `−0.60/−1.01/−1.27/−1.45`，全 `p≤1e−4` | `+0.85/+1.61/+0.90` ‖ `−0.70/−1.34/−1.56/−1.66`，全 `p<1e−4` | 主讀數，雙條件一致 |
| `accept_step1_rate` | `−0.18(p<1e−4)/−0.03(n.s.)/−0.04(n.s.)` ‖ `+0.09/+0.23/+0.33/+0.48`，`p≤.002` | `−0.13/−0.25/−0.16` ‖ `+0.16/+0.38/+0.52/+0.59`，全 `p≤5e−4` | 負臂在 asc 較弱：asc 的 step 1 只有 5%，本就低吸引力 |
| `qdm` | 僅 −6 `(−0.049, p=.001)`、−2 `(+0.017, p=.040)`、+8 `(−0.091, p<.001)` | 僅 −6 `(p=.040)`、+4 `(−0.023, p=.020)`、+8 `(−0.103, p<1e−4)` | knowing 未被系統性推動；效應量約 ≤0.10 |

> **⚠ 五格顯著性表述改變，主結論不受影響。** `desc mean_accept_step −8`（`.031→.156`）、`desc mean_bet −8`（`.029→.156`）、`desc final_score +6`（`.024→.154`）、`desc final_score +8`（`.002→.064`）轉 n.s.；`asc qdm −2`（`.278→.040`）轉 sig。
> 兩個 −8 的翻轉是**修正偽陽性**：−8 的 `invalid≈0.99`，20 runs 只有 7 個留下可用值（asc 為 0），舊 MWU 拿 7 個倖存者比 20 個 baseline。這與「−8 是 over-steer、本就排除在 clean fit 外」一致。`final_score` 的兩格則印證既有措辭——該指標 std ≫ mean（例：desc +2 = 5858±7206），**只作 downstream sanity，不得當作顯著的雙向峰**。

**主要讀數定義**：`accept_step` = 在第幾檔按下 `Accept`（1–5，越低 = 越早 commit）；`step1` = `accept_step=1` 的比例（asc 的 step 1 是 5%、desc 是 95%，兩者同時高 = immediate commitment 而非追高風險）；`DAI` = `mean_bet_desc − mean_bet_asc`，反映同一個 early-accept 傾向在兩序列中的分化（desc 搶高注 / asc 接低注），因此是 presentation-order-induced delay aversion，不是純 risk preference；`QDM` = 是否選機率較高的顏色（knowing control，崩掉則 betting 指標不能解讀為 wanting）。

**機制解讀**

- **+α 是 immediate commitment，不是 risk seeking。** 若為純風險尋求，ascending 應等到 75/95 才按；但 +α 在 ascending 也提早 `Accept`，於是 asc bet 下降、desc bet 上升，DAI 展寬。v4 已明確告知方向（`next offer will be larger/smaller`），故早停不是 rule-ignorance artifact。
- **−α 的 clean 區間是 delayed commitment。** −4/−2 主要是 `Wait→Wait→…→Accept` chain；−6 起出現 color-stage `Wait` 洩漏，說明負端不是理性保守，而是接近 stage-control failure。
- **二階交互：`desc−asc` 的符號隨 α 翻轉。** −α 在 descending 比 ascending **更願意等**（`+1.02 @−4`），+α 在 descending **更早 commit**（`−0.10 @+4`）。即 presentation order 以 α 依賴的方向調制 commitment latency。

**兩種失效模式（方向不同，勿混為「效果變弱」）**

- **−8 = 垮 / stage-onset breakdown**，不是低風險偏好。asc valid `0/1280`、desc `8/1280`；`raw_color` 空輸出 1068/1065，乾淨 color 僅 185/175，color 階段洩漏 `Accept/Wait` 25/35，bet 階段空輸出 2362/2387。少數非空文本是上下文回放或流程質疑（`I think you skipped an offer...`、`You can't accept a bet of 95% of 0 points...`），不是推理。解讀為 under-wanting / initiation failure：模型無法穩定進入動作格式。
- **+8 = 散 / overload**：生成非空但 malformed，QDM 隨之掉到 0.66。

**與人類 CGT 的區別（措辭邊界）**

- **沒有真實反應時**：人類 CGT 可量 decision latency；LLM 無 motor latency，只能用 tier position 近似 commitment timing。
- **等待成本不同**：人類的等待有時間與抑制成本；LLM 的等待只是多輸出一個 `Wait`，故測的是 token-level sequential commitment。
- **下注不是金錢激勵**：final score 只作 downstream outcome / sanity，不作主機制指標。
- **風險偏好 ≠ 延遲厭惡**：人類高 risk seeking 會在 asc 等大注、desc 搶大注；本模型 +α 在**兩種序列都提早** `Accept`，故精確說法是 immediate commitment / delay aversion。

> 實作細節（prompt 版本 v1–v4 的取捨、anchor 規則、已確認的回歸 commit、分析器口徑與過度操縱閘門）見 `CLAUDE.md` 的 cgt_sequential 條目。


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

把 IGT 與 GSM8K 並排，說明 RSN α 調的是 **working point**，而非單調的「能力」：兩者在自己主要設定下的最優 α 落在**相反的兩側**（IGT ≈ +2、GSM8K ≈ −6），方向恰好對應各任務對 wanting 的需求高低。目前證據分開的是**正/負兩側**（needs-engagement vs needs-restraint），而非同側內部的細緻梯度。

> **Bandit 不参与 working-point peak comparison。** 早期「Bandit ≈ +2」来自已作废的 pre-fix 数据，不能引用。正式 PV9 在 Easy 与具有 headroom 的 NearTie 中均未检出 α 对 directed exploration 或 outcome 的可靠改善；其主要效应落在 policy stance、candidate sharpness 与 stance–behavior consistency（§3.2）。因此 Bandit 在本文中作为**作用边界证据**，而非第三个最佳 α 工作点；跨任务峰位比较仍限于 **IGT（+2）与 GSM8K（−6）**。

| 任務 | 主導需求 | 最優 α | +α 行為 | −α 行為 | 失敗模式 |
|---|---|---|---|---|---|
| **Bandit** | reward pursuit / explore–exploit | **不定义峰位** | 政策表达更少使用 `EXPLORE`、分布较尖锐 | 政策表达更常使用 `EXPLORE` | directed exploration 与 outcome 未可靠改变；作为边界证据 |
| **IGT** | exploration + history integration（兩種相反需求並存） | **≈ +2** | 輕度有助試錯探索；過強 → 無意義換牌、忘歷史、回 B trap | 較能 stick/exploit、顯式引用歷史，但探索性低、更早鎖策略 | 兩端皆崩：+端「散」/ −端高方差「垮」 |
| **GSM8K** | arithmetic stability / commitment timing（瓶頸非探索，而是別搶答、別過度復查） | **≈ −6** | 搶答、early-####、答完放不下的 loop | 更冷靜、延遲 commit、更穩定 | +α over-wanting → 抢答 / −過度則動機不足 |

> Different tasks expose different behavioral outlets of the same α-controlled wanting axis. In each task's primary setting, **IGT** peaks at a mild positive **α ≈ +2**, whereas **GSM8K** peaks at **α ≈ −6**. This shift supports a task-dependent motivational working point rather than a monotonic capability effect. **Bandit is not used to estimate a third working point**: PV9 instead shows that α can alter policy expression and commitment-related sharpness without reliably changing uncertainty-directed exploration or outcomes. A forced-reasoning control on IGT (v4) further localises its positive-α overshoot to an engagement drop rather than a clean change in value computation.

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
- **CGT** 已完成（CGT-Sequential，結果見 §3.3）：忠實復現 Rogers 1999 / CANTAB 的升降序 betting-stage，主指標 = 延遲厭惡（accept_step / DAI），ρ = −0.96（asc）/ −0.92（desc），qdm 相對穩定（效應約 ≤0.10）。注意命名——**CGT-Simultaneous（simple5）嚴格講不是 CGT**（砍掉升降序操縱），是 transparent-odds single-shot betting probe，作為 Confidence Betting 的 confidence-confound control（機率透明排除「更自信」解釋）。
- **IGT** 已完成（2026-06-25，v6b −8→+8 × 20 runs，結果見 §3.3 IGT Full Results）：deck schedule 對碼經典 Bechara 1994（A/B 劣勢、C/D 優勢；B = 罕見巨罰 trap deck），100 trials 單一連續學習曲線。主結果 = **+2 局部峰**，`delib_tok` 為跨 prompt 版本唯一穩定讀數。

**四個範式的互補結構**（為何是這四個而非任意四個）：它們沿兩個維度張開，覆蓋 wanting 能表達的不同出口——

| | 單步決策 | 多輪累積 |
|---|---|---|
| **無回饋學習** | Confidence Betting（押注大小）<br>CGT-Simultaneous（透明賠率對照） | CGT-Sequential（Accept/Wait 延遲厭惡） |
| **有回饋學習** | — | Bandit（explore/exploit）、IGT（延遲懲罰整合） |

Betting 測「願不願意押」、CGT-Seq 測「願不願意等」、Bandit/IGT 測「願不願意持續投入並整合回饋」。**CGT-Simultaneous 的 null 在這個結構裡是資訊而非失敗**：賠率透明時 confidence mediator 被鉗住，wanting 推力失去表達通道（見 §3.3），正好界定了 wanting→behavior 需要什麼樣的下游出口。

**跨任務工作點**（詳見 §3.4）：IGT ≈ +2、GSM8K ≈ −6，構成**兩側對立**；現有證據分開的是正／負兩側，而非同側內部的細緻梯度。Bandit 不用于估计峰位：PV9 将其定位为 policy-expression effect 与 directed-exploration null 并存的作用边界（见 §3.2）。

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
