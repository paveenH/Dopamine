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

RSN paper: `ACLARR/main.tex`

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


## 2.1 Abstention Rate (MMLUE)

#### Expert & Non-expert

- From RSN paper，測量 role prompt 切換對 E-ratio 的影響。
- Expert role 一致降低 E-ratio，對應 effort engagement threshold 的調控。

| Model | Role | Acc | E-ratio | Acc_cond |
| --- | --- | --- | --- | --- |
| Llama3-8B-IT | Non-Expert | 38.7 | 44.8 | 69.3 |
| Llama3-8B-IT | **Expert** | **63.0** | **6.9** | 67.2 |
| Mistral-7B-IT | Non-Expert | 21.2 | 72.7 | 76.5 |
| Mistral-7B-IT | **Expert** | **50.1** | **24.7** | 64.9 |
| Qwen3-8B-IT | Non-Expert | 52.5 | 29.9 | 74.9 |
| Qwen3-8B-IT | **Expert** | **63.4** | **14.3** | 73.9 |

#### Neutral Steering E-Ratio (Bidirectional Control, Llama3-8B)

- 無 role prompt
- +α 一致壓低 E-ratio；−α 一致放大 E-ratio 
- RSN 作為雙向 gain knob on effort willingness，不依賴 role prompt。

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

## 2.2 Willingness Self-Evaluation（0–9 scale）

让模型用 0–9 分评价自己“有多愿意推理”。结果显示，正向 RSN 调节普遍提高了自评分数，说明 RSN 会影响模型对推理意愿的表达。但“说自己愿意”不等于“实际投入更多”，这个分数也可能受到答题信心或表达习惯的影响。

Berridge 框架提醒我们，动机过程可以在无意识层面运作，不一定都能通过自评反映，但并不意味着自评没有价值（[Berridge，2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10527990/)）。因此，我们将这个实验作为意愿表达的辅助证据，再结合实际选择、投入和核验行为，判断 RSN 究竟改变了什么；仅凭自评分数，还不能证明它具有类似多巴胺的调节功能。

neutral prompt ：
Here is a question: {context}
When dealing with the question, your self-evaluation of
"reasoning willingness" from [0,9] is:

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

## 3.1 Confidence Betting (Incentive Salience)

模型先选择下注 0、2、5 或 10 分，再回答选择题：答对获得下注分数，答错扣除相同分数。主实验每题独立，初始分数固定为 0；另设累积分数对照。

**主要结果：两个模型在 GPQA 和 MMLU 上都能被 α 调节下注大小，而答题准确率的配对差异均未达到显著。** 这说明下注行为的变化可以与答题表现分开，但下注仍混合了信心、风险偏好和输出习惯，不能直接等同于纯粹的 wanting。

### 3.1.1 GPQA Dose Response

GPQA main + diamond，N=646。Llama 指 Llama3-8B-IT，Qwen 指 Qwen2.5-7B-Instruct。

**Table 1. GPQA Accuracy and Betting Across Doses**

| Model | α | Micro accuracy | Explicit-answer accuracy | Mean bet | Paired Δbet | Bet 0 (%) | Bet 2 (%) | Bet 5 (%) | Bet 10 (%) | Invalid bet (%) | Bet p_adj | Accuracy p_adj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | −8 | 28.3% | 29.2% | 5.06 | −0.13 | 3.9 | 7.0 | 79.9 | 9.3 | 0 | 0.066 | 1.00 |
| Llama | −6 | 28.2% | 29.2% | 4.93 | −0.26 | 0.6 | 11.3 | 82.0 | 6.0 | 0 | 0.002 | 1.00 |
| Llama | −4 | 24.6% | 25.4% | 4.37 | −0.82 | 0.2 | 22.9 | 75.5 | 1.4 | 0 | 2.5e−15 | 1.00 |
| Llama | −2 | 24.9% | 25.6% | 4.29 | −0.91 | 0.5 | 25.2 | 72.9 | 1.4 | 0 | 1.2e−18 | 1.00 |
| Llama | **0** | 27.2% | 27.9% | **5.20** | — | 0.3 | 14.1 | 72.9 | 12.7 | 0 | — | — |
| Llama | +2 | 29.3% | 29.8% | 6.75 | +1.55 | 0.0 | 1.6 | 62.5 | 35.9 | 0 | 1.3e−23 | 1.00 |
| Llama | **+4** | 29.6% | 30.2% | **7.78** | **+2.58** | 0.0 | 0.0 | 44.4 | **55.6** | 0 | 2e−48 | 1.00 |
| Llama | +6 | 25.2% | 25.8% | 7.38 | +2.18 | 0.0 | 0.0 | 52.5 | 47.5 | 0 | 9.1e−39 | 1.00 |
| Llama | +8 | 26.5% | 27.1% | 7.60 | +2.39 | 0.0 | 8.2 | 34.1 | 55.9 | 1.86 | 4.3e−46 | 1.00 |
| Qwen | −8 | 33.1% | 33.3% | 4.14 | −1.43 | 2.9 | 32.8 | 54.2 | 6.2 | 3.9 | 1e−24 | 1.00 |
| Qwen | −6 | 34.1% | 34.1% | 4.91 | −0.66 | 0.2 | 14.2 | 78.3 | 7.0 | 0.3 | 5e−09 | 1.00 |
| Qwen | −4 | 33.6% | 33.6% | 5.13 | −0.45 | 0.0 | 2.5 | 93.5 | 4.0 | 0.0 | 4e−08 | 1.00 |
| Qwen | −2 | 33.8% | 33.8% | 5.14 | −0.43 | 0.0 | 0.3 | 96.7 | 2.9 | 0.0 | 3e−10 | 1.00 |
| Qwen | **0** | 33.9% | 33.9% | **5.57** | — | 0.0 | 0.0 | 88.5 | 11.5 | 0.0 | — | — |
| Qwen | +2 | 35.0% | 35.0% | 6.49 | +0.91 | 0.0 | 0.0 | 70.3 | 29.7 | 0.0 | 1e−20 | 1.00 |
| Qwen | **+4** | 35.1% | 35.1% | **7.11** | **+1.53** | 0.0 | 0.0 | 57.9 | **42.1** | 0.0 | 8e−41 | 1.00 |
| Qwen | +6† | 34.1% | 34.1% | 5.02 | −0.55 | 0.0 | 0.0 | 99.5 | 0.5 | 0.0 | 1e−15 | 1.00 |
| Qwen | +8† | 30.5% | 36.3% | 5.01 | −0.28 | 0.3 | 1.4 | 45.5 | 1.2 | 51.6 | 0.005 | 1.00 |

下注比例及 invalid 比例以全样本为分母，mean bet 只计有效下注；paired Δbet 只计两个条件均有效的题目，因此不一定等于表中两个均值之差。Explicit-answer accuracy 只计有明确答案的回复。Bet p_adj 为配对 Wilcoxon 检验，Accuracy p_adj 为精确 McNemar 检验，均经 Holm 校正。† 为高剂量失效条件，不纳入有效剂量趋势分析。

**+4 在两个模型上都提高下注。** Llama 的 mean bet 从 5.20 升至 7.78（约 +50%），Qwen 从 5.57 升至 7.11（约 +28%）；bet 10 的比例分别从 12.7% 升至 55.6%、从 11.5% 升至 42.1%。两个模型的全部八个非零剂量，accuracy p_adj 均为 1.00。

**剂量曲线具有模型差异。** Llama 在 −2/−4 降低下注，但 −6/−8 又接近基线；正向剂量在 +4 后维持较高下注水平。Qwen 在 −8…+4 内的 mean bet 随 α 上升，随后出现高剂量失效。所选区间的下注趋势分别为 Llama ρ=0.492（−4…+8）、Qwen ρ=0.455（−8…+4，n=4,495 个有效下注）。

### 3.1.2 MMLU Cross-Model Results

MMLU all subjects，N=14,042。

**Table 2. MMLU Accuracy, Betting, and Score Outcomes**

| Model | α | Micro accuracy | Explicit-answer accuracy | Macro accuracy | Mean bet | Paired Δbet | Bet 0 (%) | Bet 2 (%) | Bet 5 (%) | Bet 10 (%) | Invalid bet (%) | Bet p_adj | Accuracy p_adj | Cliff’s δ | Mean score change | Total score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | **0** | 59.25% | 59.55% | 59.61% | **4.42** | — | 0.2 | 26.8 | 68.3 | 4.7 | 0 | — | — | — | +0.78 | +10,953 |
| Llama | **+4** | 59.74% | 60.03% | 60.16% | **7.49** | **+3.06** | 0.0 | 0.0 | 50.3 | **49.7** | 0 | ~0 | 0.241 | **+0.585** | +1.44 | +20,193 |
| Llama | −4 | 59.30% | 59.64% | 59.79% | 4.05 | −0.38 | 0.1 | 32.6 | 66.8 | 0.6 | 0 | 1.6e−88 | 0.890 | −0.084 | +0.75 | +10,504 |
| Qwen | **0** | 65.60% | 65.60% | 66.95% | **5.10** | — | 0.0 | 0.0 | 97.9 | 2.1 | 0.0 | — | — | — | +1.58 | +22,231 |
| Qwen | **+4** | 65.40% | 65.40% | 66.85% | **5.57** | **+0.46** | 0.0 | 0.0 | 88.7 | **11.3** | 0.0 | 1.7e−254 | 0.336 | +0.093 | +1.71 | +24,025 |
| Qwen | −4 | 65.80% | 65.81% | 67.22% | 4.97 | −0.13 | 0.0 | 3.2 | 95.4 | 1.4 | 0.0 | 2.0e−35 | 0.336 | −0.038 | +1.58 | +22,178 |

统计及分母口径同 Table 1；每个模型的 Holm 校正包含 ±4 两个对照。~0 表示数值下溢，不表示概率严格为零。

**两个模型均呈现下注变化与准确率变化的分离。** ±4 的下注差异均显著，accuracy 差异均不显著。Llama +4 的 mean bet 增加约 69%，Cliff’s δ=+0.585，是本组 Betting 结果中最大的效应量；62.9% 的题目改变了下注。Qwen +4 的对应数值为约 +9%、δ=+0.093，以及 10.3%。

**跨模型比较需要同时看基线分布。** Llama 的基线分散在 bet 2/5/10，Qwen 则有 97.9% 的题目押 5。分布差异限制了直接比较 mean bet 增幅的解释：现有结果不能据此判断 Qwen 的 wanting 调节更弱，也不能单凭基线集中程度确定效应差异的原因。

### 3.1.3 Running-Score Controls

Llama 的对照将真实累积分数回填到下一题；MMLU 在每个 subject 开始时重置分数。

**Table 3. Running-Score Controls in Llama**

| Task | N | α | Micro accuracy | Mean bet | Bet 0 (%) | Bet 2 (%) | Bet 5 (%) | Bet 10 (%) | Mean score change | Total score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPQA | 646 | 0 | 28.8% | 5.01 | 1.2 | 22.8 | 61.0 | 15.0 | −2.18 | −1,408 |
| GPQA | 646 | +4 | 26.8% | **8.17** | 0.0 | 0.0 | 36.7 | **63.3** | −3.75 | −2,425 |
| GPQA | 646 | −4 | 27.2% | **4.34** | 0.0 | 28.3 | 68.0 | 3.7 | −1.84 | −1,189 |
| MMLU | 14,042 | 0 | 59.4% | 4.54 | 0.4 | 25.5 | 67.7 | 6.5 | +0.84 | +11,816 |
| MMLU | 14,042 | +4 | 59.0% | **7.68** | 0.0 | 0.1 | 46.2 | **53.7** | +1.38 | **+19,329** |
| MMLU | 14,042 | −4 | 59.5% | **4.18** | 0.1 | 29.0 | 70.0 | 1.0 | +0.86 | +12,088 |

**回填累积分数后，下注变化方向保持一致：+4 提高下注，−4 降低下注。** 该现象不依赖于固定 `Current score: 0`。各条件下，下注对先前分数的中位数斜率均约为 0，未显示明显的余额敏感度变化。

更高下注也不保证更高收益：+4 在 GPQA 上扩大亏损，在 MMLU 上增加总分，收益取决于任务中的答题表现。

### 3.1.4 High-Dose Failure in Qwen

Qwen GPQA 的 +6 与 +8 呈现两种不同失效，应与有效剂量下的下注调节分开解释。

**Table 4. Qwen GPQA High-Dose Diagnostics**

| Measure | α=+6 | α=+8 |
| --- | --- | --- |
| Bet 5 (%) | 99.5 | 45.5 |
| Invalid bet (%) | 0.0 | 51.6 |
| Bet entropy | 0.021 | — |
| Paired change from baseline | Of 77 baseline bet-10 questions, 75 changed to bet 5 and 2 remained at bet 10 | — |
| Micro accuracy | 34.1% | 30.5% |
| Explicit-answer accuracy | 34.1% | 36.3% |
| Explicit-answer count | — | 526/646 |
| Accuracy p_adj | 1.00 | 1.00 |
| Median response length† | 11 characters | 88 characters overall; 214 in the prose subset |
| Clean two-line format† | 99.4% | 14.9% |
| Failure pattern | Betting distribution collapses to an almost constant response | Prose responses frequently omit or displace the required bet format |

† 回复长度及格式比例来自独立的原始文本诊断批次；其余数值来自主扫描，二者不用于交叉配对。

**+6 几乎总是押 5，下注读数失去变化；+8 则主要表现为输出格式失效。** +8 有 120/646 个回复没有明确答案，micro accuracy 为 30.5%，在其余 526 个有明确答案的回复中为 36.3%。两个准确率的分母不同，既不能将 micro accuracy 的下降直接解释为知识退化，也不能用筛选后子集的较高准确率证明知识能力完全不受影响。

### 3.1.5 Conclusion

**在可解释的剂量范围内，α 能明显改变模型愿意押多少分，而准确率未检测到显著变化。** 这一结果在两个模型、两个任务中均出现，支持下注行为与答题表现可以分开调节。

效应大小和有效剂量范围依赖模型及任务。Confidence Betting 是激励与信心表达的行为证据，但尚不足以证明一个纯粹、通用的 wanting 机制。

## 3.2 Experiment 6 — Exploration/Exploitation (Bandit Task, PV9)

PV9 使用 Llama-3.1-8B-Instruct，在 Easy（`.75/.25/.25/.25`）與 NearTie（`.60/.55/.25/.25`）兩個環境中測試 `α∈{−4,0,+4}`；每格為 20 paired seeds × 100 rounds。α 只注入負責產生 evidence 與 policy 的 Stage 1，Stage 2 executor 完全未 steering。完整協議、指標定義與分析表見 `Bandit.md` §4；此處只保留可直接支撐主要結論的結果。

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

   **QDM 的成分分解（2026-08-19 新增診斷）：** 總體 QDM 由兩個成分疊加而成——一個恆定的 **red-favouring label 偏好**（`qdm_major_red` 減 `qdm_major_blue`，clean range 內 gap 0.11–0.19，兩子群皆落在 0.58–0.84）與一條**機率使用梯度**（`asym_gradient`，即 asym-8 減 asym-2，按 major color 分層後等權平均，clean range 內 0.15–0.22）。**clean range 內未檢測到兩者隨 α 的系統性變化**（label gap ρ(α)=−0.09 asc / +0.04 desc；asym gradient ρ=+0.19 / +0.12），因此「knowing 相對穩定」不僅成立於平均值，也成立於其兩個成分——α 移動的是 commitment timing，而非顏色判斷策略。n.s. 不構成等效性證明，僅表示在本設計下未檢出。逐格統計見 `analyze_cgt_seq.py` 輸出，此處不展開。

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
- **+8 = 散 / overload**：生成非空但 malformed，QDM 隨之掉到 0.66。該下降是 **major-red 子群單側塌縮**（`qdm_major_red` 0.82→0.59，paired p<.001，asc/desc 皆然），而 `qdm_major_blue` 未動（p=0.198 / 0.294）——即 +8 抹平的是既有的 red 偏好，而非整體判斷力，與「格式退化而非知識損失」一致。

**與人類 CGT 的區別（措辭邊界）**

- **沒有真實反應時**：人類 CGT 可量 decision latency；LLM 無 motor latency，只能用 tier position 近似 commitment timing。
- **等待成本不同**：人類的等待有時間與抑制成本；LLM 的等待只是多輸出一個 `Wait`，故測的是 token-level sequential commitment。
- **下注不是金錢激勵**：final score 只作 downstream outcome / sanity，不作主機制指標。
- **風險偏好 ≠ 延遲厭惡**：人類高 risk seeking 會在 asc 等大注、desc 搶大注；本模型 +α 在**兩種序列都提早** `Accept`，故精確說法是 immediate commitment / delay aversion。

> 實作細節（prompt 版本 v1–v4 的取捨、anchor 規則、已確認的回歸 commit、分析器口徑與過度操縱閘門）見 `CLAUDE.md` 的 cgt_sequential 條目。

#### CGT-Sequential 跨模型：Qwen2.5-7B-Instruct（v5，2026-08-19，**受有效劑量窗限制的部分複製**）

**定位：這是受有效劑量窗限制的部分跨模型複製，不是完整複製。** v5 修復了 Qwen v4 在 α=0 的顏色標籤鎖定，使 baseline 通過 knowing gate；但 Qwen 的有效窗仍遠窄於 Llama（Llama v4 clean range −4…+6），且 desc +2 已在 knowing control 上失效。v5 的 prompt 校準、標籤／位置歸因、pilot 與失效診斷見 `CLAUDE.md`；正文只保留正式結果。

**正式結果（N=20/格，1280 rounds/格，v5，layers 16–21）**

| 條件 | α | invalid | `qdm_blue` | `qdm_red` | `asym_grad` | gate |
|---|---|---|---|---|---|---|
| desc | −2 | .0477 | .982 | .717 | .201 | **PASS** |
| desc | 0 | .0000 | .989 | .617 | .213 | **PASS** |
| desc | +2 | .0000 | .984 | **.5031** | .178 | **FAIL（`qdm_red`）** |
| asc | −2 | .0258 | .979 | .763 | .208 | **PASS** |
| asc | 0 | .0000 | .991 | .811 | .181 | **PASS** |
| asc | +2 | .0000 | .997 | .731 | .200 | **PASS** |

| 條件 | α | `accept_step` | `step1_rate` | `mean_bet%` |
|---|---|---|---|---|
| desc | −2 / 0 / +2 | 3.508 / 1.677 / **1.048** | .279 / .743 / .967 | 37.9 / 79.5 / 94.0 |
| asc | −2 / 0 / +2 | 3.010 / 1.185 / **1.012** | .399 / .938 / .991 | 50.9 / 9.2 / 5.3 |

**可引用結論：**

1. **asc −2/0/+2 是完整通過的三點劑量結果。** `accept_step` 3.010→1.185→1.012，三個配對比較皆 `p≤7.9e−04`（exact paired Wilcoxon，n=20/16）。
2. **desc 只有 −2/0 可正式比較。** `accept_step` 3.508→1.677（`p=1.9e−06`）；+2 雖進一步降至 1.048、且 96.7% 在首檔搶下 95% 高注，但 `qdm_red=.5031` 接近隨機水平並未通過 gate，因此只能作 over-steering 邊界，不得單獨引用為 wanting 效應。
3. **DAI 僅 −2 與 0 可正式引用：** −12.97（95% CI [−19.03, −7.08]）→ +70.21（[+65.93, +74.16]）。+2 的 +88.71 含有失效的 desc +2，只是診斷值，不能支撐完整單調展寬。

整體上，Qwen 複製了 α 推動 immediate commitment 的方向，但只在較窄的構念有效窗內成立；+α 先放大既有 Blue 標籤偏差，再於 desc +2 壓垮 knowing control。故最終定位是**受有效劑量窗限制的部分複製**。實作、全 pilot、歸因分析與 parser 敏感性檢查見 `CLAUDE.md` 的 CGT-Sequential Qwen 條目。

### Iowa Gambling Task (IGT)

IGT 讓模型連續進行 100 次四牌組選擇：A/B 長期不利，C/D 長期有利。學習表現以 `p_adv=P(C+D)` 與 `net_score=P(C+D)−P(A+B)=2p_adv−1` 表示；兩者是同一讀數的不同尺度，**不計為兩份獨立證據**。完整 prompt lineage、指標定義、有效性閘門與分析口徑見 `CLAUDE.md` 的 IGT 條目。

#### Llama3-8B（v6b）：呈現 task-dependent working point

每格為 20 runs 的 mean；KW 為跨 9 個 α 的 Kruskal–Wallis `p`，ρ 為 metric 與 α 的 Spearman correlation。表中保留完整指標面板，**不因未達顯著而刪除讀數**。

| metric | α=−8 | α=−6 | α=−4 | α=−2 | α=0 | α=+2 | α=+4 | α=+6 | α=+8 | KW p | ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
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

1. **`α=+2` 是局部最佳工作點。** `net_score` 與 `learn_slope` 同時達峰；更強的 +α 則轉為高頻換牌、短持續與少回顧 history，學習表現快速下降。
2. **兩端不是同一種失效。** +α 過強是「散」（switch↑、deliberation↓）；−α 則更傾向固守與顯式回顧 history，但探索不足、較早鎖定策略。
3. **v4 forced-reasoning 是機制對照，不是主結果。** 外力補上 deliberation 後，value/risk 指標回到不穩定或 n.s.，但推理仍隨 +α 縮短；因此 v6b 的右臂退化更接近 engagement 與 exploration 失衡，而不是乾淨的價值計算改寫。

#### Qwen2.5-7B-Instruct（v6b）：推理通道複製，學習表現呈局部趨勢

Qwen 的兩個 α=0 批次均通過預先設定的 baseline gate。合併後僅作描述的 40-run 基線為 `net_score=.147`、`p_adv=.573`；兩批按 seed 配對的差異未檢出系統性批次效應（Δnet=+.0996，paired Wilcoxon `p=.207`）。各 α 的正式效應仍與**本批次自己的 α=0**配對。

負臂與正臂分開運行，因此保留各自的 `0ⁿ` / `0ᵖ`，不以合併基線取代原始 cell。下表同樣保留全量指標，不按顯著性篩除。

| metric | −8 | −6 | −4 | −2 | 0ⁿ | 0ᵖ | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `net_score` | .070 | .188 | .248 | .136 | .097 | .197 | .166 | .173 | .241 | −.011 |
| `learn_slope` | .135 | .159 | .374 | .189 | .130 | .372 | .280 | .210 | .197 | .100 |
| `net_block1→5` | −.05→.09 | .06→.22 | −.01→.36 | .06→.25 | −.03→.10 | .03→.40 | −.04→.25 | .02→.23 | .05→.25 | −.01→.85 |
| `p_adv` | .535 | .594 | .624 | .568 | .548 | .598 | .583 | .587 | .621 | .494 |
| `delib_tok` | 5.25 | 4.19 | 18.60 | 27.70 | 22.44 | 24.66 | 4.96 | 0.00 | 2.05 | 2.91 |
| `zero_frac` | .168 | .775 | .347 | .115 | .282 | .277 | .850 | 1.000 | .954 | .903 |
| `avg_raw_len` | 28 | 29 | 95 | 138 | 114 | 124 | 32 | 8 | 17 | 17 |
| `switch_rate` | .745 | .615 | .270 | .204 | .304 | .331 | .650 | .722 | .596 | .270 |
| `cycle_score` | .706 | .652 | .341 | .294 | .404 | .405 | .713 | .782 | .576 | .528 |
| `invalid` | .016 | .001 | .001 | .001 | .000 | .001 | .000 | .000 | .003 | .876 |

- **推理／生成長度呈倒 U，峰在 −2。** `delib_tok` 在負臂隨 α 上升（ρ=+.541），在正臂隨 α 下降（ρ=−.489；均 `p<1e−4`）；−2 為 27.70，而 +4 降至 0。`avg_raw_len` 呈相同形狀（峰值 138，+4 僅 8）。
- **探索與重複結構同步退化。** `switch_rate`、`cycle_score` 在兩端上升、中段最低，顯示推理縮短時模型更接近機械切換；這是同一退化模式的行為側寫，不計為獨立證據。
- **學習表現呈現局部改善，但沒有穩定的全程劑量曲線。** `net_score` 在 −4（.248）與 +6（.241）形成局部高點；最強配對格 −4 相對 neg-0 的 Δnet=+.151、raw `p=.031`，Holm 後 `p_adj=.250`。顯著性用來限制推論強度，而不是刪除這些趨勢。`learn_slope` 與 `net_block1→5` 亦完整列出；除不可讀的 +8 外，各 α 的 block5 均高於 block1。
- **次級模式是優勢牌組內部的 D→C 移動。** +2/+4 的 `p_C` 上升、+2/+4/+6 的 `p_D` 下降；因 C、D 都是優勢牌組，總 `p_adv` 可近乎不變。此模式不等同於整體學習改善，但不能由聚合 `net_score` 看見。
- **+8 必須排除。** `invalid=.876`，多數輸出丟失 `Chest` 前綴並進入 fallback；因此其 `net_block5=.850` / `last50_net=.840` 是格式失效產物，不得解讀。

因此 Qwen 的定位是：**基線會做 IGT；α 對顯式 deliberation 與策略結構的影響清楚，學習表現亦有局部改善趨勢，但尚不足以確認穩定的跨劑量 learning effect。** 這是 engagement/推理通道的複製，不是 Llama `+2` working-point 峰的逐格重現。


## 3.4 Boundary Experiment — HaluEval

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

因此 HaluEval 應放在 **boundary / side-effect evidence**：它補充說明 α 會移動 action/commitment threshold，與候選的 engagement/commitment gain 相容；但它不獨立驗證 wanting，更不等於 factual calibration 提升。

## 3.5 Cross-Task Evidence Summary — α-Sensitive Behavioral Readouts and Boundary Conditions

跨任務最一致的觀察不是一組可排序的「最佳 α」，而是 α 能改變模型的 **engagement、commitment timing、輸出銳度與策略表達**。這些改變有時會伴隨符合預期的行為或局部績效改善，但能否轉化為構念有效的 wanting、directed exploration 或 learning effect，明顯依賴任務、模型、prompt、基線分布與有效劑量窗。因此目前證據與一個候選的 motivational gain mechanism **相容**，但尚不足以證明單一 latent wanting axis，也不足以建立通用的 task-specific optimal α 規律。

| 任務 | 較穩定的觀察 | 尚未建立／主要邊界 | 目前定位 |
|---|---|---|---|
| **Confidence Betting** | 有效劑量窗內，α 可大幅移動下注分布，而 accuracy 相對穩定 | 下注仍混合 confidence、自我報告尺度與 ceiling；極端劑量會常數化或格式崩潰 | incentive/commitment 表達的正向橋接證據，不等同純 wanting |
| **CGT-Sequential** | Llama 的 accept timing 隨 α 移動，clean range 內 knowing control 相對穩定 | Qwen 只在窄劑量窗部分複製，且 label prior 會隨 +α 放大；跨模型完整劑量反應未建立 | 目前最直接的 commitment-timing 證據，但模型與接口依賴明顯 |
| **Bandit（PV9）** | α 改變 policy stance、文字表達與 candidate sharpness | 未可靠改變 uncertainty-directed sampling、穩定探索或 outcome | 明確的作用邊界：表徵／承諾變化不保證資訊獲取 |
| **IGT** | Llama 出現 +2 局部峰；Qwen 的 deliberation 與策略結構明顯隨 α 改變，學習表現有局部高點 | Qwen 未重現 Llama 峰位，學習改善未形成穩定跨劑量曲線；推理縮短也未必改變 net outcome | engagement/strategy 證據較強，learning working point 仍屬模型內、描述性結果 |
| **HaluEval** | 中等 +α 降低 challenge threshold，提高指出錯誤的傾向 | 這是 verification engagement，不是 factual calibration，也不表示模型較少 hallucinate | action-threshold 的邊界／副作用證據 |
| **GSM8K** | α 會移動推理跨度與 commitment timing，特定設定下曾出現負側局部峰 | 它不是 behavioral-economics wanting assay，且不能由其峰位反推其他任務應有的 α | reasoning-task 對照；支持 task dependence，不獨立驗證 wanting |

> **Evidence boundary.** The cross-task pattern is most consistent with an α-sensitive engagement/commitment gain that changes how strongly and how early a model expresses a policy. Its downstream consequences are task- and interface-dependent: some cells show construct-valid behavioral movement, others show only textual or distributional change, and others fail validity or outcome gates. We therefore treat a unified wanting axis and task-specific optimal α values as hypotheses motivated by the data, not conclusions established by the present behavioral suite.


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

**跨任務證據邊界**（詳見 §3.4）：目前較一致的是 α 對 engagement、commitment timing 與策略表達的影響；各任務的局部峰位只作模型內描述，不再組合成統一 wanting axis 或通用 optimal-α 規律。Bandit/PV9 尤其顯示 policy-expression effect 可以與 directed-exploration／outcome 的 null 並存。

**尚未覆蓋的範式**（誠實登記，非待辦）：Progressive Ratio（努力支出的經典 DA 範式，語言版設計見 TODO §4）、Pavlovian-Instrumental Transfer 與 Reversal Learning（均已記錄 why-skipped，見 `Dopamine_backup.md` §4.8/§4.10——核心理由是 phasic DA / RPE 需要突觸可塑性，inference-time 注入原理上碰不到）。

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
