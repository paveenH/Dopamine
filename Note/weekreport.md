
## Slide 1 — What Do RSNs Regulate?

| Question | Evidence This Week |
|---|---|
| Can behavior identify a psychological function? | InfoQuest, willingness ratings, fixed-answer choices |
| Are the effects specific to RSN? | Orthogonal controls, MMLU-E |
| Do the effects transfer across checkpoints? | Base models |
| Does steering reproduce expert behavior? | Expert vs. Non-expert |
| Does the baseline response pattern constrain gains? | Bare / Chat × CoT |

**說明：** 回應上週的核心問題：能否描述這批座標及其干預方向的功能？本週透過不同對照，逐步區分行為效果與機制解釋。

---

## Slide 2 — InfoQuest: Targeted Information Seeking

**Follow-up rate; n = 100 scenarios per condition**

| Model | Information | α = −4 | α = 0 | α = +4 |
|---|---|---:|---:|---:|
| Llama | Incomplete | 49% | 48% | 43% |
| Llama | Complete | 0% | 1% | 1% |
| Qwen | Incomplete | 56% | 45% | 42% |
| Qwen | Complete | 2% | 0% | 0% |

**說明：** 模型主要在資訊缺失時追問，但 RSN 的調節幅度有限，多數效應區間跨零。這刻畫了追問行為，尚不足以定義一般的資訊獲取功能。

*Footnote: Single automated judge; no human validation.*

---

## Slide 3 — Self-Reports and Choices Have Measurement Limits

**Llama GSM8K: expected willingness score, 0–9**

| Prompt Version | α = −4 | α = 0 | α = +4 |
|---|---:|---:|---:|
| V2 | 7.46 | 7.28 | 6.66 |
| V3 | 6.02 | 6.21 | 5.38 |

**Llama MMLU: fixed-answer readouts**

| Readout | Baseline | α = +4 | Main Limitation |
|---|---:|---:|---|
| Correctness rating | 6.869 | 6.857 | Mean rating does not identify internal confidence |
| Conditional submission probability | 0.610 | 0.682 | Extremely low X/Y probability mass |

**說明：** 自評結果隨提示版本改變，提交讀數也受到回答介面限制。目前還不能把評分或選項偏好直接解釋為內部意願或信心。

*Footnote: V2–V3 comparisons are descriptive; prompt and run conditions differ. Submission probabilities are normalized within X/Y.*

---

## Slide 4 — Shared Reasoning Gains, Different Abstention Effects

**Llama: GSM8K and MMLU-E**

| Direction | GSM8K Acc., α = −6 | GSM8K EC, α = −6 | MMLU-E ΔE, α = +4 |
|---|---:|---:|---:|
| Baseline | 60.00% | 48.00% | — |
| RSN | **78.00%** | **18.67%** | **−41.04 pp†** |
| Off-support seed 1 | 75.33% | 23.67% | −0.74 pp |
| Off-support seed 2 | 63.00% | 37.00% | +6.58 pp |
| Off-support seed 3 | 62.67% | 37.67% | +1.96 pp |

**說明：** 其他方向可以部分復現 GSM8K 收益，卻未觀察到同等幅度的棄答調節。能改善推理表現，不代表具備相同的行為功能。

*Footnote: MMLU-E uses task-macro changes. †Historical RSN reference; not a matched comparison. Doses differ across tasks.*

---

## Slide 5 — RSN Transfer to Qwen Base

**GSM8K Bare; Instruct-derived mask**

| α | First Acc. | Δ Acc. | Early-Candidate Rate |
|---:|---:|---:|---:|
| −8 | 47.33% | −15.00 pp | 50.33% |
| 0 | 62.33% | — | 37.33% |
| +8 | **80.00%** | **+17.67 pp** | **7.67%** |

**說明：** Instruct 提取的方向在 Qwen Base 上仍能改變輸出組織與準確率，作用不只出現在 Instruct 檢查點。Llama Base 仍受到生成介面限制，無法作相同判斷。

*Footnote: Checkpoint transfer does not identify pretraining, SFT, or RLHF as the source.*

---

## Slide 6 — Expert Prompts Do Not Consistently Reproduce Steering

**GSM8K Bare, α = 0**

| Model | Role | Early-Candidate Rate | First Acc. |
|---|---|---:|---:|
| Llama | Expert | 54.33% | 58.00% |
| Llama | Non-expert | 41.67% | 68.00% |
| Llama | Neutral | 48.00% | 60.00% |
| Qwen | Expert | 97.00% | 55.33% |
| Qwen | Non-expert | 95.67% | 70.67% |
| Qwen | Neutral | 96.67% | 69.33% |

**說明：** Qwen 的角色提示幾乎沒有改變候選答案順序，但正向 RSN 能大幅改變。因此，「兩個模型學到不同專家策略」尚不能解釋 steering 的方向差異。

*Footnote: Qwen Expert−Non-expert ΔEC = +1.33 pp, 95% CI [−0.33, +3.33].*

---

## Slide 7 — CoT Under Bare Prompts

**GSM8K, α = 0; No-CoT → CoT**

| Model | First Acc. | Early-Candidate Rate |
|---|---:|---:|
| Llama | 60.00% → 69.00% | 48.00% → 50.00% |
| Qwen | 68.00% → 76.33% | 96.33% → 97.33% |

**說明：** Bare 下已有 CoT 實驗。CoT 可以提高準確率，卻未必減少早期候選，因此不能把 CoT 收益完全等同於答案延後。

---

## Slide 8 — CoT Under Native Chat

**α = 0; No-CoT → CoT**

| Model | Task | First Acc. | Early-Candidate Rate |
|---|---|---:|---:|
| Llama | GSM8K | 89.67% → 89.67% | 0.00% → 0.00% |
| Llama | GSM-Hard | 31.00% → 34.67% | 0.33% → 0.33% |
| Qwen | GSM8K | **63.33% → 93.00%** | **50.00% → 1.33%** |
| Qwen | GSM-Hard | **40.00% → 57.00%** | **31.00% → 0.33%** |

**說明：** Qwen 的較大 CoT 收益伴隨明顯輸出重排；Llama 原本已幾乎沒有早期候選，變化較小。準確率提升空間與作答方式的調整空間，需要分開討論。

*Footnote: Cross-prompt comparisons are descriptive.*

---

## Slide 9 — Additional Steering Gains Under Chat-CoT

| Model | Task | α | Δ Acc. [95% CI], pp | Holm p |
|---|---|---:|---:|---:|
| Llama | GSM8K | −6 | +0.33 [−2.00, +3.00] | 1.000 |
| Llama | GSM-Hard | −6 | −1.33 [−3.67, +1.00] | 0.388 |
| Qwen | GSM8K | +8 | +1.67 [−0.33, +3.67] | 0.359 |
| Qwen | GSM-Hard | +8 | −2.67 [−5.67, +0.33] | 0.268 |

**說明：** Chat-CoT 下，四項比較均未檢出固定工作點的額外收益。結果與 CoT、RSN 存在部分功能重疊相容，但尚未證明相同機制或內部 commitment 飽和。

*Footnote: Paired comparisons within Chat-CoT; no CoT × steering interaction test.*

---

## Slide 10 — Working Hypothesis: Regulation of Response Strategies

| Observation | Implication |
|---|---|
| Self-reports and choices depend on the interface | Psychological labels require stronger discrimination |
| Random directions reproduce some reasoning gains | Reasoning improvement is not exclusively RSN-specific |
| Qwen Base retains steering effects | Effects can transfer across checkpoints |
| Expert prompts do not reproduce Qwen steering | Expert behavior is not a sufficient explanation |
| CoT and steering gains vary across response conditions | Baseline response patterns may constrain benefits |

**說明：** 整體假設是：**RSN 在任務入口調節模型採用不同作答策略的傾向，收益取決於策略與任務是否匹配。** 目前已能刻畫多種功能效果，但共同內部機制仍待驗證。

---

### Sources

:codex-file-citation{path="/Users/paveenhuang/Downloads/Dopamine/Note/TMLR1005.md" purpose="source"}

:codex-file-citation{path="/Users/paveenhuang/Downloads/Dopamine/Note/BehaviourFail.md" purpose="source"}

:codex-file-citation{path="/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningBare.md" purpose="source"}

:codex-file-citation{path="/Users/paveenhuang/Downloads/Dopamine/Note/ReasoningChat.md" purpose="source"}