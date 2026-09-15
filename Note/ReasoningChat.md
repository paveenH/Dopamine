# Reasoning under Chat Interfaces

## 1. Scope and Evaluation Setup

本节比较 Native Chat、Chat Matched-Anchor 与既有 Bare 参照下的性能和可见输出。Bare 只用于接口参照，不能与 Chat 条件进行样本级统计推断。`first_acc` 是主要性能指标；在提供该指标的实验中，`last_acc` 仅作为敏感性分析。Llama3.1-8B-Instruct 的 Native Chat 是九点 dose response；Qwen2.5-7B-Instruct 只测试 `−8/0/+6/+8`，属于 targeted dose comparison，不构成完整 dose curve。`early_candidate_rate`、`reason_first_rate` 与 `candidate_posN_median` 只描述干预后的可见 output ordering，不能代表内部 commitment 或因果中介。

## 2. Native Chat Dose–Response

### 2.1 Llama3.1-8B-Instruct

#### GSM8K

**Table 2.1. Llama3.1-8B-Instruct on GSM8K under Native Chat**

| α | n | first_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 89.00% | 96.33% | 0.33% | 92.62% | 0.4628 | 0.00% | 99.33% | 0.67% | 566 |
| −6 | 300 | 88.67% | 96.00% | 0.00% | 100.00% | 0.4573 | 0.33% | 99.33% | 0.67% | 559 |
| −4 | 300 | 90.00% | 96.00% | 0.00% | 100.00% | 0.4637 | 0.00% | 99.67% | 0.33% | 541 |
| −2 | 300 | 91.00% | 97.00% | 0.00% | 100.00% | 0.4706 | 0.00% | 99.67% | 0.33% | 533 |
| **0** | 300 | **89.67%** | 96.67% | **0.00%** | 100.00% | **0.4761** | **0.00%** | **99.00%** | 1.00% | **528** |
| +2 | 300 | 90.33% | 97.33% | 0.00% | 100.00% | 0.4790 | 0.33% | 99.67% | 0.33% | 522 |
| +4 | 300 | 89.00% | 97.00% | 0.67% | 100.00% | 0.4775 | 0.33% | 99.00% | 1.00% | 527 |
| **+6** | 300 | 90.33% | 95.67% | **20.67%** | 96.97% | **0.3083** | 0.00% | 99.00% | 1.00% | **384** |
| **+8** | 300 | **78.33%** | 90.67% | **52.00%** | 97.97% | **0.2141** | 1.00% | 99.00% | 1.00% | **292** |

负向 α 没有重现 Bare 的负向工作点。高正向 α 推动 candidate 前移并压缩生成；`+6` 已改变 ordering 但性能仍近 baseline，`+8` 降至 78.33%。

#### MATH

**Table 2.2. Llama3.1-8B-Instruct on MATH under Native Chat**

| α | n | first_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 47.67% | 84.33% | 1.00% | 96.86% | 0.2976 | 2.67% | 84.00% | 16.00% | 945 |
| −6 | 300 | 47.33% | 80.33% | 1.00% | 96.13% | 0.2909 | 5.67% | 80.00% | 20.00% | 1012 |
| −4 | 300 | 47.67% | 85.33% | 0.67% | 94.76% | 0.3272 | 3.67% | 85.67% | 14.33% | 1002 |
| −2 | 300 | 47.67% | 84.33% | 0.67% | 94.77% | 0.3094 | 5.00% | 84.33% | 15.67% | 1004 |
| **0** | 300 | **47.67%** | 85.33% | **0.33%** | 96.43% | **0.3252** | **3.67%** | **85.67%** | **14.33%** | **1035** |
| +2 | 300 | 46.00% | 80.00% | 0.33% | 96.47% | 0.2950 | 4.00% | 79.67% | 20.33% | 1039 |
| +4 | 300 | 48.67% | 81.67% | 2.33% | 95.90% | 0.2991 | 4.00% | 83.00% | 17.00% | 985 |
| **+6** | 300 | 47.33% | 82.33% | **21.67%** | 90.39% | **0.2749** | 3.67% | 83.33% | 16.67% | **830** |
| **+8** | 300 | **42.67%** | **77.00%** | **29.33%** | 93.09% | **0.2279** | **7.67%** | 81.00% | 19.00% | **716** |

负向 α 未形成清晰性能峰。`+6/+8` 提前 candidate 并缩短输出，`+8` 同时降至 42.67%；MATH 的长度与位置必须结合较高的截断率解释。

#### GSM-Hard

**Table 2.3. Llama3.1-8B-Instruct on GSM-Hard under Native Chat**

| α | n | first_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 33.00% | 89.67% | 1.00% | 98.66% | 0.3969 | 1.00% | 95.00% | 5.00% | 687 |
| −6 | 300 | 32.00% | 82.33% | 0.33% | 99.32% | 0.3827 | 1.33% | 91.67% | 8.33% | 698 |
| −4 | 300 | 30.00% | 85.00% | 0.33% | 99.32% | 0.4022 | 2.33% | 90.67% | 9.33% | 673 |
| −2 | 300 | 30.67% | 85.33% | 0.33% | 99.32% | 0.4123 | 1.33% | 91.33% | 8.67% | 665 |
| **0** | 300 | **31.00%** | 86.67% | **0.33%** | 99.32% | **0.4161** | **1.67%** | **92.00%** | **8.00%** | **676** |
| +2 | 300 | 32.00% | 87.33% | 0.33% | 99.32% | 0.4052 | 2.00% | 93.33% | 6.67% | 643 |
| +4 | 300 | 32.00% | 86.00% | 1.00% | 100.00% | 0.4031 | 1.67% | 92.33% | 7.67% | 690 |
| **+6** | 300 | 30.33% | 87.67% | **22.00%** | 97.96% | **0.2935** | 1.00% | 94.67% | 5.33% | **483** |
| **+8** | 300 | **24.33%** | 87.33% | **56.00%** | 98.99% | **0.1429** | 2.00% | 96.67% | 3.33% | **375** |

负向 α 未出现清晰工作点；`+6/+8` 使 candidate 前移并压缩生成，`+8` 的 `first_acc` 降至 24.33%。

### 2.2 Qwen2.5-7B-Instruct

本节只比较 `−8/0/+6/+8`；不与 Llama 的 α 绝对大小比较。

#### GSM8K

**Table 2.4. Qwen2.5-7B-Instruct on GSM8K under Native Chat**

| α | n | first_acc | last_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **−8** | 300 | **47.33%** | 61.00% | 97.00% | **68.67%** | **30.30%** | **0.0167** | 0.00% | 100.00% | 0.00% | 638 |
| **0** | 300 | **63.33%** | 71.00% | 91.33% | **50.00%** | **49.49%** | **0.2179** | 0.00% | 99.67% | 0.33% | 734 |
| **+6** | 300 | **94.67%** | 94.67% | 96.33% | **1.00%** | **100.00%** | **0.4294** | 0.00% | 99.67% | 0.33% | 771 |
| **+8** | 300 | **93.00%** | 93.00% | 98.67% | **2.00%** | **99.67%** | **0.3929** | 0.00% | 100.00% | 0.00% | 585 |

baseline `first_acc=63.33%`。`+6` 和 `+8` 分别达到 94.67% 与 93.00%，均显著提高；`early_candidate_rate` 下降，`reason_first_rate` 接近 100%。`−8` 则推动 candidate 前移并显著降低准确率。性能与 output ordering 同向，但不能据此主张因果中介。

#### MATH

**Table 2.5. Qwen2.5-7B-Instruct on MATH under Native Chat**

| α | n | first_acc | last_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 71.33% | 71.33% | 99.67% | 2.00% | 97.95% | 0.2775 | 0.33% | 99.67% | 0.33% | 1246 |
| **0** | 300 | **70.67%** | 70.67% | 98.67% | **0.00%** | **98.63%** | **0.3205** | 0.67% | 98.67% | 1.33% | 1278 |
| +6 | 300 | 72.67% | 72.67% | 99.33% | 0.33% | 98.97% | 0.3198 | 0.33% | 99.33% | 0.67% | 1245 |
| +8 | 300 | 71.33% | 71.33% | 100.00% | 0.33% | 98.98% | 0.3252 | 0.00% | 100.00% | 0.00% | 1281 |

baseline 已稳定 reasoning-first。`−8/+6/+8` 均未显著改变准确率，行为指标也基本稳定，说明此 baseline state 下进一步调控不一定带来性能收益。

#### GSM-Hard

**Table 2.6. Qwen2.5-7B-Instruct on GSM-Hard under Native Chat**

| α | n | first_acc | last_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_median |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **−8** | 300 | **25.67%** | 34.67% | 94.67% | **56.33%** | **43.29%** | **0.0321** | 0.33% | 99.00% | 1.00% | 851 |
| **0** | 300 | **40.00%** | 44.67% | 86.00% | **31.00%** | **69.05%** | **0.2591** | 0.33% | 99.00% | 1.00% | 914 |
| **+6** | 300 | **54.33%** | 54.33% | 93.33% | **0.33%** | **99.67%** | **0.4088** | 0.00% | 100.00% | 0.00% | 899 |
| **+8** | 300 | **54.67%** | 54.67% | 94.33% | **5.33%** | **99.00%** | **0.3784** | 0.33% | 99.67% | 0.33% | 743 |

`+6/+8` 均显著提高准确率，同时降低 `early_candidate_rate`、提高 `reason_first_rate`；`−8` 方向相反并显著损害准确率，整体与 GSM8K 一致。

#### Statistical Comparison

**Table 2.7. Qwen Native Chat Comparisons against α=0**

| Task | α | Baseline first_acc | Cell first_acc | Δ | Discordant (baseline-only / cell-only) | Raw p | Holm p_adj | Significant |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GSM8K | −8 | 63.33% | 47.33% | −16.00 pp | 52 / 4 | 1.101e−11 | 1.101e−11 | Yes |
| GSM8K | +6 | 63.33% | 94.67% | +31.33 pp | 1 / 95 | 2.449e−27 | 7.346e−27 | Yes |
| GSM8K | +8 | 63.33% | 93.00% | +29.67 pp | 4 / 93 | 4.565e−23 | 9.131e−23 | Yes |
| MATH | −8 | 70.67% | 71.33% | +0.67 pp | 18 / 20 | .8714 | 1.0000 | No |
| MATH | +6 | 70.67% | 72.67% | +2.00 pp | 9 / 15 | .3075 | .9224 | No |
| MATH | +8 | 70.67% | 71.33% | +0.67 pp | 15 / 17 | .8601 | 1.0000 | No |
| GSM-Hard | −8 | 40.00% | 25.67% | −14.33 pp | 50 / 7 | 4.237e−09 | 1.271e−08 | Yes |
| GSM-Hard | +6 | 40.00% | 54.33% | +14.33 pp | 21 / 64 | 3.276e−06 | 3.276e−06 | Yes |
| GSM-Hard | +8 | 40.00% | 54.67% | +14.67 pp | 12 / 56 | 6.209e−08 | 1.242e−07 | Yes |

每个任务各自进行 `m=3` Holm 校正，不跨任务合并。

### 2.3 Cross-Model and Cross-Task Summary

| Model | Task | Dose coverage | Main performance pattern | Output pattern |
| ----- | ---- | ------------- | ------------------------ | -------------- |
| Llama3.1-8B-Instruct | GSM8K | Nine-point Native Chat curve | Stable through `+6`, decline at `+8` | High positive α moves candidates earlier and shortens output |
| Llama3.1-8B-Instruct | MATH | Nine-point Native Chat curve | Flat through `+6`, decline at `+8` | High positive α moves candidates earlier; health is less stable |
| Llama3.1-8B-Instruct | GSM-Hard | Nine-point Native Chat curve | Decline at high positive α | High positive α moves candidates earlier and shortens output |
| Qwen2.5-7B-Instruct | GSM8K | Targeted `−8/0/+6/+8` | Significant gains at `+6/+8`; loss at `−8` | Shifts toward reasoning-first at `+6/+8` |
| Qwen2.5-7B-Instruct | MATH | Targeted `−8/0/+6/+8` | No significant change | Baseline is already reasoning-first |
| Qwen2.5-7B-Instruct | GSM-Hard | Targeted `−8/0/+6/+8` | Significant gains at `+6/+8`; loss at `−8` | Same ordering direction as GSM8K |

Llama Native Chat 未重现 Bare 的负向工作点，强正向 α 主要伴随 candidate 前移、生成压缩和高剂量性能下降。Qwen GSM8K 与 GSM-Hard 在 `+6/+8` 下转向 reasoning-first 并显著改善；MATH baseline 已高度 reasoning-first，未见显著收益。α 的效果取决于 model、task 与 interface，不能提出跨模型统一剂量结论。

## 3. Matched-Anchor Control

### Llama3.1-8B-Instruct under Chat Matched-Anchor

该条件独立于 Native Chat，不能作样本级统计合并。

#### GSM8K

**Table 3.1. Llama3.1-8B-Instruct on GSM8K under Chat Matched-Anchor**

| α | first_acc | last_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 89.33% | 89.33% | 96.67% | 3.00% | 96.97% | 0.4128 | 0.33% | 99.00% | 1.00% | 506 |
| −6 | 88.33% | 88.33% | 95.33% | 1.67% | 94.61% | 0.4565 | 0.00% | 98.33% | 1.67% | 493 |
| −4 | 90.00% | 90.00% | 97.00% | 1.00% | 97.64% | 0.4742 | 0.00% | 99.67% | 0.33% | 499 |
| −2 | 90.00% | 90.00% | 97.00% | 1.00% | 98.65% | 0.4722 | 0.00% | 99.33% | 0.67% | 501 |
| **0** | **88.33%** | 88.33% | 96.33% | 2.33% | 96.96% | 0.4657 | 0.33% | 99.00% | 1.00% | 501 |
| +2 | 87.67% | 87.67% | 95.67% | 4.33% | 95.61% | 0.4493 | 0.33% | 98.33% | 1.67% | 498 |
| **+4** | 84.67% | 84.67% | 89.00% | **25.67%** | **59.26%** | **0.2250** | 0.33% | 97.67% | 2.33% | **424** |
| **+6** | **79.00%** | 79.00% | 89.67% | **54.00%** | **34.56%** | **0.0000** | 0.67% | 97.00% | 3.00% | **297** |
| +8 | 84.67% | 84.67% | 92.00% | 47.67% | 71.38% | 0.1486 | 0.67% | 98.00% | 2.00% | 304 |

负向 α 未形成稳定工作点；相较 Native Chat，正向 transition 更早，`+6` 显著降至 79.00%。

#### MATH

**Table 3.2. Llama3.1-8B-Instruct on MATH under Chat Matched-Anchor**

| α | first_acc | last_acc | valid_sub_rate | early_candidate_rate | reason_first_rate | candidate_posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 44.67% | 44.67% | 86.67% | 18.00% | 79.65% | 0.2500 | 5.33% | 77.33% | 22.67% | 967 |
| −6 | 46.00% | 46.00% | 81.67% | 7.67% | 87.63% | 0.3107 | 4.67% | 81.00% | 19.00% | 922 |
| −4 | 47.00% | 47.00% | 86.00% | 9.67% | 86.36% | 0.3123 | 4.00% | 83.67% | 16.33% | 866 |
| −2 | 45.00% | 45.00% | 84.67% | 10.67% | 85.92% | 0.2819 | 4.67% | 83.00% | 17.00% | 921 |
| **0** | **48.00%** | 48.00% | 85.33% | 13.00% | 82.87% | 0.2872 | 3.67% | 83.33% | 16.67% | 903 |
| **+2** | **43.00%** | 43.00% | 85.67% | **21.67%** | **75.70%** | **0.2302** | 3.33% | 82.00% | 18.00% | 936 |
| **+4** | **34.33%** | 34.33% | 80.33% | **58.00%** | **51.42%** | **0.0237** | 8.33% | 68.67% | 31.33% | 1050 |
| +6 | 33.33% | 34.00% | 66.67% | 60.00% | 58.01% | 0.0073 | 10.00% | 73.00% | 27.00% | 941 |
| +8 | 37.33% | 37.67% | 62.33% | 45.33% | 70.57% | 0.0487 | 7.00% | 75.33% | 24.67% | 858 |

MATH 对 assistant-side anchor 更敏感：正向 α 自 `+2` 开始下降，`+4/+6/+8` 均伴随输出重排与 generation health 恶化。

#### Statistical Comparison

**Table 3.3. Holm-Corrected Comparisons against the Matched-Anchor Baseline**

| Task | α | Δfirst_acc | Discordant (baseline-only / cell-only) | Raw p | Holm p_adj | Significant |
|---|---:|---:|---:|---:|---:|---|
| GSM8K | −8 | +1.00 pp | 8 / 11 | .6476 | 1.0000 | No |
| GSM8K | −6 | 0.00 pp | 6 / 6 | 1.0000 | 1.0000 | No |
| GSM8K | −4 | +1.67 pp | 4 / 9 | .2668 | 1.0000 | No |
| GSM8K | −2 | +1.67 pp | 4 / 9 | .2668 | 1.0000 | No |
| GSM8K | +2 | −0.67 pp | 7 / 5 | .7744 | 1.0000 | No |
| GSM8K | +4 | −3.67 pp | 28 / 17 | .1352 | .8209 | No |
| GSM8K | +6 | −9.33 pp | 42 / 14 | .0002 | .0019 | Yes |
| GSM8K | +8 | −3.67 pp | 26 / 15 | .1173 | .8209 | No |
| MATH | −8 | −3.33 pp | 26 / 16 | .1641 | .5442 | No |
| MATH | −6 | −2.00 pp | 19 / 13 | .3771 | .7542 | No |
| MATH | −4 | −1.00 pp | 15 / 12 | .7011 | .7542 | No |
| MATH | −2 | −3.00 pp | 19 / 10 | .1360 | .5442 | No |
| MATH | +2 | −5.00 pp | 21 / 6 | .0059 | .0296 | Yes |
| MATH | +4 | −13.67 pp | 58 / 17 | 2.00e−06 | 1.70e−05 | Yes |
| MATH | +6 | −14.67 pp | 67 / 23 | 4.00e−06 | 2.70e−05 | Yes |
| MATH | +8 | −10.67 pp | 54 / 22 | .0003 | .0019 | Yes |

每个任务相对自身 `α=0` 作八项 Holm 校正。Matched-Anchor 不是 Native Chat 的等价复现，而是不同的 intervention geometry。

## 4. Interface-Dependent Output Behavior

### 4.1 Bare versus Native Chat


**Table 4.1. Llama Bare and Native Chat Output States**

| Model | Task | Condition | first_acc | early_candidate_rate | reason_first_rate |
|---|---|---|---:|---:|---:|
| Llama3.1-8B-Instruct | GSM8K | Bare α=0 | 60.00% | 48.00% | 30.85% |
| Llama3.1-8B-Instruct | GSM8K | Native Chat α=0 | 89.67% | 0.00% | 100.00% |
| Llama3.1-8B-Instruct | GSM8K | Bare −6 | 78.00% | 18.67% | 66.55% |
| Llama3.1-8B-Instruct | GSM8K | Native Chat −6 | 88.67% | 0.00% | 100.00% |
| Llama3.1-8B-Instruct | MATH | Bare α=0 | 36.67% | 26.67% | 47.18% |
| Llama3.1-8B-Instruct | MATH | Native Chat α=0 | 47.67% | 0.33% | 96.43% |
| Llama3.1-8B-Instruct | MATH | Bare −6 | 43.33% | 6.33% | 79.44% |
| Llama3.1-8B-Instruct | MATH | Native Chat −6 | 47.33% | 1.00% | 96.13% |
| Llama3.1-8B-Instruct | GSM-Hard | Bare α=0 | 18.00% | 45.67% | 26.64% |
| Llama3.1-8B-Instruct | GSM-Hard | Native Chat α=0 | 31.00% | 0.33% | 99.32% |
| Llama3.1-8B-Instruct | GSM-Hard | Bare −6 | 24.33% | 28.67% | 61.82% |
| Llama3.1-8B-Instruct | GSM-Hard | Native Chat −6 | 32.00% | 0.33% | 99.32% |

Native Chat 在三个任务中都显著重排了 Llama 的可见输出：相较 Bare α=0，`early_candidate_rate` 接近地板，`reason_first_rate` 接近上限。Bare 条件下有效的 `−6` 可以使输出向同一方向移动，但其变化幅度小于 Chat 本身；进入 Native Chat 后，再施加 `−6` 几乎不再改变 output ordering。该比较只描述跨接口的行为状态，不作跨接口显著性或因果归因。

**Table 4.2. Qwen Bare and Native Chat Output States**

| Model | Task | Condition | first_acc | early_candidate_rate | reason_first_rate |
|---|---|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | GSM8K | Bare α=0 | 68.00% | 96.33% | 0.00% |
| Qwen2.5-7B-Instruct | GSM8K | Native Chat α=0 | 63.33% | 50.00% | 49.49% |
| Qwen2.5-7B-Instruct | GSM8K | Bare +6 | 78.00% | 45.33% | 51.67% |
| Qwen2.5-7B-Instruct | GSM8K | Native Chat +6 | 94.67% | 1.00% | 100.00% |
| Qwen2.5-7B-Instruct | GSM8K | Bare +8 | 86.00% | 5.00% | 98.00% |
| Qwen2.5-7B-Instruct | GSM8K | Native Chat +8 | 93.00% | 2.00% | 99.67% |
| Qwen2.5-7B-Instruct | MATH | Bare α=0 | 60.67% | 67.33% | 11.11% |
| Qwen2.5-7B-Instruct | MATH | Native Chat α=0 | 70.67% | 0.00% | 98.63% |
| Qwen2.5-7B-Instruct | MATH | Bare +6 | 68.33% | 19.33% | 77.59% |
| Qwen2.5-7B-Instruct | MATH | Native Chat +6 | 72.67% | 0.33% | 98.97% |
| Qwen2.5-7B-Instruct | GSM-Hard | Bare α=0 | 34.00% | 93.67% | 0.00% |
| Qwen2.5-7B-Instruct | GSM-Hard | Native Chat α=0 | 40.00% | 31.00% | 69.05% |
| Qwen2.5-7B-Instruct | GSM-Hard | Bare +8 | 50.33% | 6.00% | 98.00% |
| Qwen2.5-7B-Instruct | GSM-Hard | Native Chat +8 | 54.67% | 5.33% | 99.00% |

该表只描述 output state，不作跨接口显著性推断。

### 4.2 Opening Style across Interfaces

**Table 4.2. Llama Opening Style across Interfaces at α=0**

| Task | Interface | Main opening pattern |
|---|---|---|
| GSM8K | Bare | 208/300 start directly with a number or equation |
| GSM8K | Native Chat | 283/300 start with “To find…” |
| GSM8K | Chat Matched-Anchor | 247/300 start with “To find…”, with more direct calculation |
| MATH | Bare | 139/300 start with a number or formula; 53/300 use `Step 1` |
| MATH | Native Chat | 294/300 start with “To…” |
| MATH | Chat Matched-Anchor | 228/300 start with “To…”; 38/300 start directly with `\boxed{}` |

Bare 的开场较不统一，也更容易先给答案、重复 marker 或出现长尾；Native Chat 通常先说明目标再计算；Matched-Anchor 更容易直接进入数字、公式或正式答案。

### 4.3 Output Ordering and Generation Health

`early_candidate_rate`、`reason_first_rate` 和 `candidate_posN_median` 描述 output ordering；`loop_rate`、`truncation_rate` 和 `natural_eos_rate` 用于判断 generation health。输出重排不一定立即改变准确率，长度缩短也不能自动解释为更有效或更差的推理。

### 4.4 Interpretation Boundaries

candidate 不等于正式 marker，reasoning-first 是表面输出模式。行为指标不是内部 commitment 的直接测量；行为变化与准确率共同出现，不等于行为变化造成准确率变化。

## 5. Conclusions

1. Chat interface 会显著改变模型的基础输出方式。
2. Bare 中发现的 workpoint 不能直接迁移到 Chat。
3. Llama 在 Native Chat 下主要表现为正向高剂量引起的 candidate 前移和性能损失。
4. Qwen 在 GSM8K 与 GSM-Hard 上表现出相反方向：`+6/+8` 促进 reasoning-first，并显著提高准确率。
5. Qwen MATH 没有明显收益，说明干预效果存在 task boundary。
6. Steering 效果取决于 model、task、interface 和 baseline output state。

**Conclusion.** Chat interface 改变了 steering 的剂量响应。有效方向并不跨模型和任务统一：Qwen 在 GSM8K 与 GSM-Hard 上可通过正向 α 获得明显收益，但在已经稳定 reasoning-first 的 MATH 上没有进一步改善；Llama 的结果则主要表现为高正向剂量导致输出提前和性能下降。因此，workpoint 必须在具体的 model–task–interface 条件下确定。


---

# Role–Chat Relationship: Interim Report

## 1. Behavioral Validation on Qwen

我们首先验证 Qwen 是否也存在 Chat interface 与 RSN steering 的功能联系。

Qwen 在 GSM8K 和 GSM-Hard 上呈现了清晰结果：

- Native Chat 相比 Bare 减少提前回答，并增加先推理后提交；
- 在 Native Chat 上继续施加正向 RSN steering，效果可以进一步增强；
- GSM8K 的 α=+6/+8 分别提升 **31.34/29.67 pp**；
- GSM-Hard 的 α=+6/+8 分别提升 **14.33/14.67 pp**；
- MATH 上没有显著 steering 效应。

这说明 RSN 与 Chat 在行为功能上存在相似性：二者都能改变模型的推理参与程度、答案提交顺序和最终表现。但不同模型的有效 α 方向不同，因此不能把某个固定正负方向视为跨模型统一剂量。

**Status: Supported at the functional level.**

## 2. Chat–Bare Hidden-State Direction

我们提取了 Llama3 和 Qwen2.5 在 GSM8K、MATH、GSM-Hard 上的 paired Chat/Bare hidden states，并计算：

$$d_{\text{chat}}=\operatorname{mean}(h_{\text{chat}}-h_{\text{bare}})$$

Chat–Bare shift 本身具有一定跨任务稳定性：

- GSM8K 与 GSM-Hard 的方向高度一致；
- Llama3 cosine = **0.983**；
- Qwen2.5 cosine = **0.988**；
- MATH 与两个 GSM 类任务的相似度较低，但仍保持较高的整体相关。

因此，Chat template 确实引起了结构化、可重复的 hidden-state shift。但仅分析平均 Chat–Bare direction，无法解释它与 Role/RSN 的关系。

**Status: Chat–Bare direction is measurable, but mean-direction analysis alone is insufficient.**

## 3. Expert/Non-expert Hidden States

我们重新采集了两个模型在以下任务上的 Expert/Non-expert hidden states：

- GSM8K
- MATH
- GSM-Hard
- MMLU-E

其中三个 reasoning task 使用相同样本进行 Expert/Non-expert 配对；MMLU-E Role direction 单独保留，因为原始 RSN 方向包含基于答案的筛选，不能与未经筛选的 reasoning Role direction 直接混合。

**Status: Completed.**

## 4. Dense and Sparse Role–Chat Relationship

### Dense Direction

Reasoning Role average 与 Reasoning Chat average 在 RSN band 内的 cosine 为：

| Model | Dense Role–Chat Cosine |
|---|---:|
| Llama3 | **−0.0286** |
| Qwen2.5 | **−0.1036** |

Task-matched comparisons同样接近零，并且 GSM8K/GSM-Hard 多为负，MATH 略偏正。

因此，Role direction 与 Chat–Bare mean direction 几乎正交，不能认为它们是同一个 dense axis。

### Sparse Coordinates

尽管 dense cosine 很低，Role 与 Chat 的绝对值 top-0.5% dimensions 存在显著重合：

- Llama3 task-matched enrichment：**34–49× random**；
- Qwen2.5 task-matched enrichment：**50–57× random**；
- Llama3 有 **19/33 layers** 达到 Holm-adjusted significance；
- Qwen2.5 有 **13/29 layers** 达到显著。

Role directions 本身也表现出明显任务结构：

- GSM8K 与 GSM-Hard Role cosine：**0.96–0.98**；
- MATH 与其他任务明显不同；
- MMLU-E Role 与 reasoning Role 的相似度也较低。

这些结果说明：

> Role 与 Chat 共享非随机的稀疏坐标结构，但不是同一条完整的 dense direction。

**Status: Dense alignment not supported; sparse structural overlap supported.**

## 5. Chat–Bare Transition Subspace

为了检验 Role 是否位于 Chat–Bare 更广泛的状态变化空间中，我们将三个任务的900组 paired differences 合并，在每层进行 PCA，并计算 Role direction 投影到前 \(k\) 个主成分上的能量比例：

$$
R^2(k)=\frac{\|V_k^\top r\|^2}{\|r\|^2}
$$

主分析采用 within-task centering，并通过150/150 split cross-fitting进行评估。

### Main Results

| Model | Centered R², k=10 | Random Mean | Null p97.5 | Enrichment | Held-out Reconstruction, k=50 |
|---|---:|---:|---:|---:|---:|
| Llama3 | **0.0788** | 0.0024 | 0.0030 | **32.3×** | **0.4381** |
| Qwen2.5 | **0.1261** | 0.0028 | 0.0036 | **45.3×** | **0.4768** |

结果在不同 \(k\) 下保持稳定：

| k | Llama3 R² | Qwen2.5 R² |
|---:|---:|---:|
| 1 | 0.0286 | 0.0278 |
| 5 | 0.0631 | 0.1011 |
| 10 | 0.0788 | 0.1261 |
| 20 | 0.1126 | 0.1423 |
| 50 | 0.1901 | 0.2052 |

Centered projection 没有在去除任务平均 shift 后消失，而且明显超过 isotropic 与 coordinate-permutation null。

这说明：

> Role direction 不是 Chat–Bare 的平均转换方向，但它部分嵌入在 Chat–Bare 的多维 transition subspace 中。

前10维捕获 Role 总能量的 **7.9–12.6%**，前50维捕获 **19–21%**。因此这是明显但不完全的几何包含关系。

**Status: Non-random subspace relationship supported.**

## Overall Conclusion

目前结果形成了一个比较一致的层级结构：

1. **Behavioral level:** RSN 与 Chat 都会改变推理参与、提交顺序和任务表现；
2. **Dense direction level:** Role 与 Chat–Bare mean direction 几乎正交；
3. **Sparse level:** 二者共享显著高于随机的 top-neuron coordinates；
4. **Subspace level:** Role direction 对 Chat–Bare transition subspace 具有稳定且远超随机的投影。

最合适的总体表述是：

> Chat induces a broad, multidimensional working-state transition. Role-sensitive directions are not equivalent to its mean transition direction, but form a non-random sparse and subspace-level component of this broader transformation.

通俗地说：

> **Chat 是一次更广泛的工作状态重构；Role/RSN 不是整个 Chat 状态，但可能是其中一条与推理参与和答案提交有关的局部功能轴。**

这些结果仍然属于结构和功能层面的证据，不能单独证明共享因果机制，也不能推出与人类多巴胺系统的生物学同源性。

## Figure Locations

- Role–Chat dense/sparse figures: `/Users/paveenhuang/Documents/RSNResult/RoleHidden/AdaResult/4.full_role_chat_direction/plots/`

- Transition-subspace figures: `/Users/paveenhuang/Documents/RSNResult/RoleHidden/AdaResult/5.chat_role_transition_subspace/plots/`

- Transition-subspace results and plotting sources: `/Users/paveenhuang/Documents/RSNResult/RoleHidden/AdaResult/5.chat_role_transition_subspace/`

其中包括主图 `fig_main_role_projection.png`，以及补充图 `fig_supp_layerwise_heatmap.png` 和 `fig_supp_enrichment.png`。
