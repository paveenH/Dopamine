# Reasoning under Chat Interfaces

## 1. Scope and Evaluation Setup

本节检验 Chat interface 下的 reasoning performance 与输出行为，并比较两种 Chat 设置：

- **Native Chat**：使用模型原生 Chat template。
- **Chat Matched-Anchor**：保留 Chat template，但在 assistant 端加入与 Bare 条件匹配的答案起始 anchor。

所有条件均包含 300 个样本。正文以 offline `first_acc` 为主要性能指标；`last_acc` 用于检查后续答案修改。`early_cand_rate`、`reason_first_rate` 与 `cand_posN_med` 描述答案候选和可见推理文本的排列方式，不能直接代表模型内部何时形成答案。

具体脚本、文件路径、完整 provenance、hash 与交叉检查记录见 `CLAUDE.md`。

## 2. Native Chat Dose–Response

### 2.1 GSM8K

**Table 2.1. Llama3.1-8B-Instruct on GSM8K under Native Chat**

| α | n | first_acc | valid_sub_rate | no_parse_marker_rate | early_cand_rate | reason_first_rate | posN_med | cand_posN_med | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med | first_line_le60 | first_line_has_num |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 89.00% | 96.33% | 3.67% | 0.33% | 92.62% | 0.9867 | 0.4628 | 0.00% | 99.33% | 0.67% | 566 | 11.67% | 18.67% |
| −6 | 300 | 88.67% | 96.00% | 4.00% | 0.00% | 100.00% | 0.9869 | 0.4573 | 0.33% | 99.33% | 0.67% | 559 | 2.67% | 16.33% |
| −4 | 300 | 90.00% | 96.00% | 4.00% | 0.00% | 100.00% | 0.9863 | 0.4637 | 0.00% | 99.67% | 0.33% | 541 | 3.33% | 16.67% |
| −2 | 300 | 91.00% | 97.00% | 3.00% | 0.00% | 100.00% | 0.9864 | 0.4706 | 0.00% | 99.67% | 0.33% | 533 | 4.33% | 16.33% |
| **0** | 300 | **89.67%** | 96.67% | 3.33% | **0.00%** | 100.00% | 0.9862 | **0.4761** | **0.00%** | **99.00%** | 1.00% | **528** | 3.67% | 16.67% |
| +2 | 300 | 90.33% | 97.33% | 2.67% | 0.00% | 100.00% | 0.9859 | 0.4790 | 0.33% | 99.67% | 0.33% | 522 | 3.33% | 17.33% |
| +4 | 300 | 89.00% | 97.00% | 3.00% | 0.67% | 100.00% | 0.9860 | 0.4775 | 0.33% | 99.00% | 1.00% | 527 | 3.00% | 23.33% |
| **+6** | 300 | 90.33% | 95.67% | 4.33% | **20.67%** | 96.97% | 0.9809 | **0.3083** | 0.00% | 99.00% | 1.00% | **384** | 26.67% | 89.33% |
| **+8** | 300 | **78.33%** | **90.67%** | 9.33% | **52.00%** | 97.97% | 0.9747 | **0.2141** | 1.00% | 99.00% | 1.00% | **292** | **61.00%** | 86.67% |

负向 α 没有重现 Bare 条件下以 `−6` 为中心的性能峰。`−8` 至 `+4` 的 `first_acc` 均在 88.67%–91.00% 之间，整体变化很小。

主要变化出现在正向高剂量。到 `+6` 时，`early_cand_rate` 从 0.00% 升至 20.67%，`cand_posN_med` 从 0.4761 降至 0.3083，生成长度中位数从 528 降至 384，但准确率仍为 90.33%。到 `+8` 后，candidate 进一步前移，生成长度降至 292，同时准确率下降至 78.33%。

因此，输出重排先于明显的性能下降出现；candidate 提前并不必然立即造成准确率下降，但更强的变化会与性能损失同时出现。

### 2.2 MATH

**Table 2.2. Llama3.1-8B-Instruct on MATH under Native Chat**

| α | n | first_acc | valid_sub_rate | no_parse_marker_rate | early_cand_rate | reason_first_rate | posN_med | cand_posN_med | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med | first_line_le60 | first_line_has_num |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 47.67% | 84.33% | 15.67% | 1.00% | 96.86% | 0.9845 | 0.2976 | 2.67% | 84.00% | 16.00% | 945 | 4.00% | 45.33% |
| −6 | 300 | 47.33% | 80.33% | 19.67% | 1.00% | 96.13% | 0.9848 | 0.2909 | 5.67% | 80.00% | 20.00% | 1012 | 4.33% | 44.67% |
| −4 | 300 | 47.67% | 85.33% | 14.67% | 0.67% | 94.76% | 0.9849 | 0.3272 | 3.67% | 85.67% | 14.33% | 1002 | 5.33% | 38.67% |
| −2 | 300 | 47.67% | 84.33% | 15.67% | 0.67% | 94.77% | 0.9845 | 0.3094 | 5.00% | 84.33% | 15.67% | 1004 | 4.00% | 38.67% |
| **0** | 300 | **47.67%** | 85.33% | 14.67% | **0.33%** | 96.43% | 0.9841 | **0.3252** | **3.67%** | **85.67%** | **14.33%** | **1035** | 4.00% | 39.00% |
| +2 | 300 | 46.00% | 80.00% | 20.00% | 0.33% | 96.47% | 0.9846 | 0.2950 | 4.00% | 79.67% | 20.33% | 1039 | 3.00% | 41.67% |
| +4 | 300 | 48.67% | 81.67% | 18.33% | 2.33% | 95.90% | 0.9840 | 0.2991 | 4.00% | 83.00% | 17.00% | 985 | 8.33% | 50.00% |
| **+6** | 300 | 47.33% | 82.33% | 17.67% | **21.67%** | 90.39% | 0.9809 | **0.2749** | 3.67% | 83.33% | 16.67% | **830** | 33.33% | 65.33% |
| **+8** | 300 | **42.67%** | **77.00%*** | **22.67%** | **29.33%** | 93.09% | 0.9788 | **0.2279** | **7.67%** | 81.00% | 19.00% | **716** | **43.00%** | 68.67% |

\* `α=+8` 的 `valid_sub_rate` 存在一个样本的冻结定义差异：本表为 77.00%，原分析器的对应结果为 77.33%。该差异不影响主要结论。

MATH 的负向 α 同样没有形成清晰的性能峰；从 `−8` 到 `0`，`first_acc` 基本维持在 47%–48%。

正向高剂量再次改变输出顺序。`early_cand_rate` 从 baseline 的 0.33% 升至 `+6` 的 21.67% 和 `+8` 的 29.33%，生成长度中位数则由 1035 降至 830 和 716。准确率在 `+6` 仍接近 baseline，但在 `+8` 降至 42.67%。

与 GSM8K 一样，candidate 前移和生成压缩可以先于明显的准确率下降出现。不过，MATH 的截断率整体更高，因此长度与位置指标需要结合 generation health 解读。

### 2.3 GSM-Hard

**Table 2.3. Llama3.1-8B-Instruct on GSM-Hard under Native Chat**

| α | n | first_acc | valid_sub_rate | no_parse_marker_rate | early_cand_rate | reason_first_rate | posN_med | cand_posN_med | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med | first_line_le60 | first_line_has_num |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 33.00% | 89.67% | 10.33% | 1.00% | 98.66% | 0.9843 | 0.3969 | 1.00% | 95.00% | 5.00% | 687 | 13.33% | 16.00% |
| −6 | 300 | 32.00% | 82.33% | 17.67% | 0.33% | 99.32% | 0.9838 | 0.3827 | 1.33% | 91.67% | 8.33% | 698 | 3.67% | 15.33% |
| −4 | 300 | 30.00% | 85.00% | 15.00% | 0.33% | 99.32% | 0.9828 | 0.4022 | 2.33% | 90.67% | 9.33% | 673 | 3.33% | 15.67% |
| −2 | 300 | 30.67% | 85.33% | 14.67% | 0.33% | 99.32% | 0.9836 | 0.4123 | 1.33% | 91.33% | 8.67% | 665 | 4.00% | 17.00% |
| **0** | 300 | **31.00%** | 86.67% | 13.33% | **0.33%** | 99.32% | 0.9836 | **0.4161** | **1.67%** | **92.00%** | **8.00%** | **676** | 3.67% | 16.00% |
| +2 | 300 | 32.00% | 87.33% | 12.67% | 0.33% | 99.32% | 0.9827 | 0.4052 | 2.00% | 93.33% | 6.67% | 643 | 4.00% | 16.67% |
| +4 | 300 | 32.00% | 86.00% | 14.00% | 1.00% | 100.00% | 0.9832 | 0.4031 | 1.67% | 92.33% | 7.67% | 690 | 4.00% | 18.33% |
| **+6** | 300 | 30.33% | 87.67% | 12.33% | **22.00%** | 97.96% | 0.9771 | **0.2935** | 1.00% | 94.67% | 5.33% | **483** | 32.33% | 80.67% |
| **+8** | 300 | **24.33%** | 87.33% | 12.67% | **56.00%** | 98.99% | 0.9719 | **0.1429** | 2.00% | 96.67% | 3.33% | **375** | **65.00%** | 84.67% |

GSM-Hard 的结果与 GSM8K 相似。`−8` 至 `+4` 的准确率没有形成清晰的负向工作点，但 `+6/+8` 出现明显的 candidate 前移和生成压缩。

从 baseline 到 `+8`，`early_cand_rate` 从 0.33% 升至 56.00%，`cand_posN_med` 从 0.4161 降至 0.1429，生成长度中位数从 676 降至 375；与此同时，`first_acc` 从 31.00% 降至 24.33%。

### 2.4 Cross-Task Summary

Native Chat 下，三个任务呈现出一致的总体形状：

- 负向 α 没有重现 Bare 条件下的性能峰。
- `+6/+8` 系统性推动 candidate 前移，并压缩输出长度。
- 输出重排在 `+6` 已经出现，但明显的性能损失主要集中在 `+8`。
- GSM8K 与 GSM-Hard 的 generation health 整体稳定；MATH 的截断与循环问题相对更明显。

因此，Native Chat 将模型维持在较稳定的 reasoning-first 输出状态，但足够强的正向 α 仍能推动模型更早进入答案阶段。

## 3. Matched-Anchor Control

Chat Matched-Anchor 在 assistant 端加入答案起始提示，用于检验 Chat template 与注入位置共同变化时，原有剂量效应是否保留。该条件属于独立实验，不能与 Native Chat 的样本级统计直接合并。

### 3.1 GSM8K

**Table 3.1. GSM8K under Chat Matched-Anchor**

| α | first_acc | last_acc | valid_sub_rate | no_parse_marker_rate | early_cand_rate | reason_first_rate | posN_med | cand_posN_med | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med | first_line_le60 | first_line_has_num |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 89.33% | 89.33% | 96.67% | 3.33% | 3.00% | 96.97% | 0.9853 | 0.4128 | 0.33% | 99.00% | 1.00% | 506 | 19.00% | 30.67% |
| −6 | 88.33% | 88.33% | 95.33% | 4.67% | 1.67% | 94.61% | 0.9851 | 0.4565 | 0.00% | 98.33% | 1.67% | 493 | 8.67% | 25.33% |
| −4 | 90.00% | 90.00% | 97.00% | 3.00% | 1.00% | 97.64% | 0.9854 | 0.4742 | 0.00% | 99.67% | 0.33% | 499 | 5.67% | 24.33% |
| −2 | 90.00% | 90.00% | 97.00% | 3.00% | 1.00% | 98.65% | 0.9853 | 0.4722 | 0.00% | 99.33% | 0.67% | 501 | 5.67% | 25.33% |
| **0** | **88.33%** | 88.33% | 96.33% | 3.67% | 2.33% | 96.96% | 0.9853 | 0.4657 | 0.33% | 99.00% | 1.00% | 501 | 8.00% | 26.33% |
| +2 | 87.67% | 87.67% | 95.67% | 4.33% | 4.33% | 95.61% | 0.9851 | 0.4493 | 0.33% | 98.33% | 1.67% | 498 | 10.67% | 29.00% |
| +4 | 84.67% | 84.67% | 89.00% | 11.00% | 25.67% | 59.26% | 0.9815 | 0.2250 | 0.33% | 97.67% | 2.33% | 424 | 31.00% | 70.33% |
| +6 | 79.00% | 79.00% | 89.67% | 10.33% | 54.00% | 34.56% | 0.9733 | 0.0000 | 0.67% | 97.00% | 3.00% | 297 | 58.00% | 95.33% |
| +8 | 84.67% | 84.67% | 92.00% | 8.00% | 47.67% | 71.38% | 0.9743 | 0.1486 | 0.67% | 98.00% | 2.00% | 304 | 58.00% | 86.00% |

负向 α 仍未形成稳定工作点：`−6` 与 baseline 的 `first_acc` 完全相同，其他负向剂量的差异也很小。

与 Native Chat 相比，正向 transition 更早出现。`+4` 已表现出明显的 candidate 前移和 `reason_first_rate` 下降；`+6` 的 `first_acc` 降至 79.00%，相对 baseline 下降 9.33 pp，并在 Holm 校正后显著。

### 3.2 MATH

**Table 3.2. MATH under Chat Matched-Anchor**

| α | first_acc | last_acc | valid_sub_rate | no_parse_marker_rate | early_cand_rate | reason_first_rate | posN_med | cand_posN_med | loop_rate | natural_eos_rate | truncation_rate | gen_chars_med | first_line_le60 | first_line_has_num |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | 44.67% | 44.67% | 86.67% | 13.00% | 18.00% | 79.65% | 0.9777 | 0.2500 | 5.33% | 77.33% | 22.67% | 967 | 22.67% | 56.00% |
| −6 | 46.00% | 46.00% | 81.67% | 18.33% | 7.67% | 87.63% | 0.9801 | 0.3107 | 4.67% | 81.00% | 19.00% | 922 | 12.00% | 53.00% |
| −4 | 47.00% | 47.00% | 86.00% | 14.00% | 9.67% | 86.36% | 0.9795 | 0.3123 | 4.00% | 83.67% | 16.33% | 866 | 13.67% | 53.33% |
| −2 | 45.00% | 45.00% | 84.67% | 15.33% | 10.67% | 85.92% | 0.9797 | 0.2819 | 4.67% | 83.00% | 17.00% | 921 | 13.67% | 55.33% |
| **0** | **48.00%** | 48.00% | 85.33% | 14.67% | 13.00% | 82.87% | 0.9784 | 0.2872 | 3.67% | 83.33% | 16.67% | 903 | 16.00% | 56.33% |
| +2 | 43.00% | 43.00% | 85.67% | 14.33% | 21.67% | 75.70% | 0.9764 | 0.2302 | 3.33% | 82.00% | 18.00% | 936 | 25.67% | 61.00% |
| +4 | 34.33% | 34.33% | 80.33% | 19.67% | 58.00% | 51.42% | 0.9483 | 0.0237 | 8.33% | 68.67% | 31.33% | 1050 | 61.33% | 84.33% |
| +6 | 33.33% | 34.00% | 66.67% | 33.33% | 60.00% | 58.01% | 0.9755 | 0.0073 | 10.00% | 73.00% | 27.00% | 941 | 67.33% | 86.00% |
| +8 | 37.33% | 37.67% | 62.33% | 37.67% | 45.33% | 70.57% | 0.9778 | 0.0487 | 7.00% | 75.33% | 24.67% | 858 | 63.33% | 73.00% |

MATH 对 assistant-side anchor 更敏感。负向 α 仍未产生收益，但正向剂量从 `+2` 开始下降；`+4/+6/+8` 均出现较大的准确率损失。

这些变化同时伴随 `early_cand_rate` 上升、candidate 前移，以及 `valid_sub_rate` 和 generation health 恶化。因此，MATH 的高剂量损失不仅涉及回答顺序，也包含输出完整性下降。

### 3.3 Statistical Comparison

**Table 3.3. Holm-Corrected Comparisons against the Matched-Anchor Baseline**

| Task | α | Δfirst_acc | Discordant (baseline-only / cell-only) | Raw p | Holm p_adj | Significant |
|---|---:|---:|---:|---:|---:|---|
| GSM8K | −8 | +1.00 pp | 8 / 11 | .6476 | 1.0000 | No |
| GSM8K | −6 | 0.00 pp | 6 / 6 | 1.0000 | 1.0000 | No |
| GSM8K | −4 | +1.67 pp | 4 / 9 | .2668 | 1.0000 | No |
| GSM8K | −2 | +1.67 pp | 4 / 9 | .2668 | 1.0000 | No |
| GSM8K | +2 | −0.67 pp | 7 / 5 | .7744 | 1.0000 | No |
| GSM8K | +4 | −3.67 pp | 28 / 17 | .1352 | .8209 | No |
| GSM8K | +6 | **−9.33 pp** | 42 / 14 | .0002 | **.0019** | **Yes** |
| GSM8K | +8 | −3.67 pp | 26 / 15 | .1173 | .8209 | No |
| MATH | −8 | −3.33 pp | 26 / 16 | .1641 | .5442 | No |
| MATH | −6 | −2.00 pp | 19 / 13 | .3771 | .7542 | No |
| MATH | −4 | −1.00 pp | 15 / 12 | .7011 | .7542 | No |
| MATH | −2 | −3.00 pp | 19 / 10 | .1360 | .5442 | No |
| MATH | +2 | **−5.00 pp** | 21 / 6 | .0059 | **.0296** | **Yes** |
| MATH | +4 | **−13.67 pp** | 58 / 17 | 2.00e−06 | **1.70e−05** | **Yes** |
| MATH | +6 | **−14.67 pp** | 67 / 23 | 4.00e−06 | **2.70e−05** | **Yes** |
| MATH | +8 | **−10.67 pp** | 54 / 22 | .0003 | **.0019** | **Yes** |

每个任务分别以自身的 `α=0` 为 baseline，并在八个比较内进行 Holm 校正。

统计结果与剂量曲线一致：负向 α 在 GSM8K 和 MATH 上均未产生显著收益；正向 α 则在 GSM8K 的 `+6` 以及 MATH 的 `+2` 至 `+8` 造成显著下降。

### 3.4 Native Chat versus Matched-Anchor

两种 Chat 设置保留了相同的总体方向，但 transition 的位置和强度不同：

- 两种设置都没有重现 Bare 条件下的负向工作点。
- 两种设置都在正向 α 下出现 candidate 前移和生成压缩。
- Native Chat 的明显性能下降主要出现在 `+8`。
- Matched-Anchor 将 transition 提前：GSM8K 在 `+4/+6` 已明显变化，MATH 从 `+2` 开始出现显著损失。
- GSM8K 在两种设置下的 loop、truncation 与 natural EOS 整体健康；MATH 在 Matched-Anchor 下的 generation health 更差。

因此，Chat template 决定了整体输出模式，而 assistant-side anchor 会进一步改变模型进入答案阶段的敏感度。Matched-Anchor 不是 Native Chat 的等价复现，而是一种具有不同 intervention geometry 的接口条件。

## 4. Interface-Dependent Output Behavior

### 4.1 Opening Style across Interfaces

对 `α=0` 的 300 个输出进行开场风格统计后，三种接口表现出明显差异。

**Table 4.1. Main Opening Style at α=0**

| Task | Interface | Main opening pattern |
|---|---|---|
| GSM8K | Bare | 208/300 start directly with a number or equation |
| GSM8K | Native Chat | 283/300 start with “To find…” |
| GSM8K | Chat Matched-Anchor | 247/300 start with “To find…”, with more direct calculation |
| MATH | Bare | 139/300 start with a number or formula; 53/300 use `Step 1` |
| MATH | Native Chat | 294/300 start with “To…” |
| MATH | Chat Matched-Anchor | 228/300 start with “To…”; 38/300 start directly with `\boxed{}` |

Bare 输出更接近自由文本续写：开场形式不统一，较容易先给答案、重复正式 marker，或在答案后继续生成。MATH Bare 的 baseline 中还有 45/300 个输出出现无关客套长尾，而两种 Chat 设置中均为 0/300。

Native Chat 的回答形式最稳定，通常先说明目标，再展开计算，最后提交答案。Chat Matched-Anchor 仍保留这一整体风格，但更容易直接进入数字、公式或正式答案。

### 4.2 Output Ordering and Generation Health

综合完整剂量结果，可以区分三个相互关联但并不等价的变化：

1. **Output ordering**：candidate 是否提前出现、candidate 前是否存在可见推理文本。
2. **Generation compression**：输出是否缩短、第一行是否更快出现数值。
3. **Generation health**：输出能否正常停止、是否循环、截断或缺少可解析答案。

正向高剂量通常同时推动 candidate 前移和输出压缩，但两者与准确率并非一一对应。例如，Native Chat GSM8K 在 `+6` 已出现明显重排，准确率却仍接近 baseline；Matched-Anchor MATH 则同时出现重排、格式覆盖率下降和明显性能损失。

因此，candidate 前移可以描述生成策略的变化，但不能单独解释准确率变化。长度下降也可能来自更简洁的回答、提前提交或输出异常，必须结合 `valid_sub_rate`、`loop_rate`、`natural_eos_rate` 与 `truncation_rate` 判断。

### 4.3 Interpretation Boundaries

本节的行为指标均来自干预后的可见输出，只能用于描述模型如何组织答案：

- `early_cand_rate` 表示答案候选是否很早出现。
- `reason_first_rate` 表示 candidate 之前是否出现表面推理文本。
- `cand_posN_med` 表示第一个 candidate 在全文中的相对位置。
- `posN_med` 表示正式答案 marker 的相对位置。
- `first_line_has_num` 与 `first_line_le60` 描述开场是否快速进入数字或简短回答。

candidate 与正式 marker 不是同一个事件；表面 reasoning text 也不等于模型已经完成内部推理。因此，这些指标不能证明内部 commitment 的时间，也不能作为准确率变化的因果中介。

## 5. Conclusions

Chat interface 明显改变了 Llama 的基础输出策略：相比 Bare，模型更稳定地先解释、后提交答案，并显著减少循环、重复和无关长尾。

在这一接口下，Bare 中的负向工作点没有稳定重现。相反，最一致的剂量效应出现在正向高剂量：`+6/+8` 会推动 candidate 前移并压缩生成，在更强条件下伴随准确率下降。

Matched-Anchor 保留了相同方向，但使正向 transition 更早、更强，尤其是在 MATH 上。这说明 steering 效果不仅依赖任务和剂量，也依赖 Chat template、assistant-side anchor 与 injection geometry。

**Conclusion.** Chat template 可以稳定模型的 reasoning-first 输出形式，但不能消除 steering 对回答顺序的影响。负向 workpoint 不具有跨接口稳定性，而正向高剂量的 candidate 前移和生成压缩相对稳定；这些输出变化具有明显的 interface dependence，且不能单独解释性能变化。


---

## Summary Answers

### GSM8K

- alpha=-8: first_acc 63.33% -> 47.33% (Δ=-16.00pp, p_holm=1.101e-11, significant); early_candidate_rate 50.00% -> 68.67%; reason_first_rate 49.49% -> 30.30%; candidate_posN_median 0.2179 -> 0.0167; gen_chars_median 734 -> 638; loop_rate 0.00% -> 0.00%; truncation_rate 0.33% -> 0.00%
- alpha=+6: first_acc 63.33% -> 94.67% (Δ=+31.34pp, p_holm=7.346e-27, significant); early_candidate_rate 50.00% -> 1.00%; reason_first_rate 49.49% -> 100.00%; candidate_posN_median 0.2179 -> 0.4294; gen_chars_median 734 -> 771; loop_rate 0.00% -> 0.00%; truncation_rate 0.33% -> 0.33%
- alpha=+8: first_acc 63.33% -> 93.00% (Δ=+29.67pp, p_holm=9.131e-23, significant); early_candidate_rate 50.00% -> 2.00%; reason_first_rate 49.49% -> 99.67%; candidate_posN_median 0.2179 -> 0.3929; gen_chars_median 734 -> 585; loop_rate 0.00% -> 0.00%; truncation_rate 0.33% -> 0.00%

### MATH

- alpha=-8: first_acc 70.67% -> 71.33% (Δ=+0.66pp, p_holm=1, not significant); early_candidate_rate 0.00% -> 2.00%; reason_first_rate 98.63% -> 97.95%; candidate_posN_median 0.3205 -> 0.2775; gen_chars_median 1278 -> 1246; loop_rate 0.67% -> 0.33%; truncation_rate 1.33% -> 0.33%
- alpha=+6: first_acc 70.67% -> 72.67% (Δ=+2.00pp, p_holm=0.9224, not significant); early_candidate_rate 0.00% -> 0.33%; reason_first_rate 98.63% -> 98.97%; candidate_posN_median 0.3205 -> 0.3198; gen_chars_median 1278 -> 1245; loop_rate 0.67% -> 0.33%; truncation_rate 1.33% -> 0.67%
- alpha=+8: first_acc 70.67% -> 71.33% (Δ=+0.66pp, p_holm=1, not significant); early_candidate_rate 0.00% -> 0.33%; reason_first_rate 98.63% -> 98.98%; candidate_posN_median 0.3205 -> 0.3252; gen_chars_median 1278 -> 1281; loop_rate 0.67% -> 0.00%; truncation_rate 1.33% -> 0.00%

### GSM_HARD

- alpha=-8: first_acc 40.00% -> 25.67% (Δ=-14.33pp, p_holm=1.271e-08, significant); early_candidate_rate 31.00% -> 56.33%; reason_first_rate 69.05% -> 43.29%; candidate_posN_median 0.2591 -> 0.0321; gen_chars_median 914 -> 851; loop_rate 0.33% -> 0.33%; truncation_rate 1.00% -> 1.00%
- alpha=+6: first_acc 40.00% -> 54.33% (Δ=+14.33pp, p_holm=3.276e-06, significant); early_candidate_rate 31.00% -> 0.33%; reason_first_rate 69.05% -> 99.67%; candidate_posN_median 0.2591 -> 0.4088; gen_chars_median 914 -> 899; loop_rate 0.33% -> 0.00%; truncation_rate 1.00% -> 0.00%
- alpha=+8: first_acc 40.00% -> 54.67% (Δ=+14.67pp, p_holm=1.242e-07, significant); early_candidate_rate 31.00% -> 5.33%; reason_first_rate 69.05% -> 99.00%; candidate_posN_median 0.2591 -> 0.3784; gen_chars_median 914 -> 743; loop_rate 0.33% -> 0.33%; truncation_rate 1.00% -> 0.33%

**Cross-task direction check.** See the full data table below for exact values; the summary answers above state per-task direction and significance without pooling.

## Full Data Table (offline metrics, n=300/cell)

| task | alpha | first_acc | last_acc | inline_acc_pct | valid_submission_rate | conditional_accuracy | no_parseable_marker_rate | multiple_marker_rate | early_candidate_rate | candidate_coverage | reason_first_rate | pre_candidate_chars_median | post_candidate_chars_median | candidate_posN_median | posN_median | loop_rate | natural_eos_rate | truncation_rate | gen_chars_median | gen_tokens_median | first_line_le60_rate | first_line_has_number_rate | steering_fires |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | -8 | 47.33 | 61.00 | 47.33 | 97.00 | 46.05 | 3.00 | 23.33 | 68.67 | 99.00 | 30.30 | 5 | 506 | 0.0167 | 0.0000 | 0.00 | 100.00 | 0.00 | 638 | 201.50 | 100.00 | 68.67 | 1800 |
| gsm8k | 0 | 63.33 | 71.00 | 63.33 | 91.33 | 60.95 | 8.67 | 15.67 | 50.00 | 99.00 | 49.49 | 5 | 534 | 0.2179 | 0.0000 | 0.00 | 99.67 | 0.33 | 734 | 224.50 | 93.67 | 51.33 | 0 |
| gsm8k | 6 | 94.67 | 94.67 | 94.67 | 96.33 | 94.81 | 3.67 | 0.00 | 1.00 | 99.67 | 100.00 | 320 | 389 | 0.4294 | 0.9900 | 0.00 | 99.67 | 0.33 | 771 | 234.00 | 18.67 | 28.33 | 1800 |
| gsm8k | 8 | 93.00 | 93.00 | 93.00 | 98.67 | 93.58 | 1.33 | 0.00 | 2.00 | 99.67 | 99.67 | 224 | 338 | 0.3929 | 0.9874 | 0.00 | 100.00 | 0.00 | 585 | 188.50 | 19.33 | 45.67 | 1800 |
| math | -8 | 71.33 | 71.33 | 71.67 | 99.67 | 71.57 | 0.33 | 0.00 | 2.00 | 97.33 | 97.95 | 296.00 | 801.50 | 0.2775 | 0.9872 | 0.33 | 99.67 | 0.33 | 1246 | 474.50 | 27.67 | 51.00 | 1800 |
| math | 0 | 70.67 | 70.67 | 71.33 | 98.67 | 71.62 | 1.33 | 0.33 | 0.00 | 97.33 | 98.63 | 355.50 | 782.00 | 0.3205 | 0.9874 | 0.67 | 98.67 | 1.33 | 1278 | 471.00 | 4.00 | 67.67 | 0 |
| math | 6 | 72.67 | 72.67 | 73.33 | 99.33 | 73.15 | 0.67 | 0.00 | 0.33 | 97.00 | 98.97 | 351 | 735 | 0.3198 | 0.9869 | 0.33 | 99.33 | 0.67 | 1245 | 461.50 | 3.00 | 70.00 | 1800 |
| math | 8 | 71.33 | 71.33 | 72.00 | 100.00 | 71.33 | 0.00 | 0.33 | 0.33 | 97.67 | 98.98 | 350 | 754 | 0.3252 | 0.9873 | 0.00 | 100.00 | 0.00 | 1281 | 460.50 | 3.67 | 69.67 | 1800 |
| gsm_hard | -8 | 25.67 | 34.67 | None | 94.67 | 25.35 | 5.33 | 27.33 | 56.33 | 99.33 | 43.29 | 5.00 | 661.50 | 0.0321 | 0.0000 | 0.33 | 99.00 | 1.00 | 851 | 289.50 | 100.00 | 56.33 | 1800 |
| gsm_hard | 0 | 40.00 | 44.67 | None | 86.00 | 40.70 | 14.00 | 14.67 | 31.00 | 98.00 | 69.05 | 237.00 | 632.50 | 0.2591 | 0.9845 | 0.33 | 99.00 | 1.00 | 914 | 299.50 | 91.33 | 33.00 | 0 |
| gsm_hard | 6 | 54.33 | 54.33 | None | 93.33 | 55.36 | 6.67 | 0.33 | 0.33 | 100.00 | 99.67 | 349.00 | 519.00 | 0.4088 | 0.9877 | 0.00 | 100.00 | 0.00 | 899 | 297.50 | 18.33 | 22.67 | 1800 |
| gsm_hard | 8 | 54.67 | 54.67 | None | 94.33 | 55.12 | 5.67 | 0.00 | 5.33 | 100.00 | 99.00 | 270.00 | 438.00 | 0.3784 | 0.9855 | 0.33 | 99.67 | 0.33 | 743 | 255.00 | 19.33 | 46.67 | 1800 |

## Holm-Corrected Comparisons (per task, m=3, vs that task's own chat alpha=0)

### GSM8K

| alpha | n | base first_acc | cell first_acc | Δpp | discordant (base-only/cell-only) | p_raw | p_holm | significant |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -8 | 300 | 63.33% | 47.33% | -16.00 | 52 / 4 | 1.101e-11 | 1.101e-11 | **Yes** |
| +6 | 300 | 63.33% | 94.67% | +31.34 | 1 / 95 | 2.449e-27 | 7.346e-27 | **Yes** |
| +8 | 300 | 63.33% | 93.00% | +29.67 | 4 / 93 | 4.565e-23 | 9.131e-23 | **Yes** |

### MATH

| alpha | n | base first_acc | cell first_acc | Δpp | discordant (base-only/cell-only) | p_raw | p_holm | significant |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -8 | 300 | 70.67% | 71.33% | +0.66 | 18 / 20 | 0.8714 | 1 | No |
| +6 | 300 | 70.67% | 72.67% | +2.00 | 9 / 15 | 0.3075 | 0.9224 | No |
| +8 | 300 | 70.67% | 71.33% | +0.66 | 15 / 17 | 0.8601 | 1 | No |

### GSM_HARD

| alpha | n | base first_acc | cell first_acc | Δpp | discordant (base-only/cell-only) | p_raw | p_holm | significant |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -8 | 300 | 40.00% | 25.67% | -14.33 | 50 / 7 | 4.237e-09 | 1.271e-08 | **Yes** |
| +6 | 300 | 40.00% | 54.33% | +14.33 | 21 / 64 | 3.276e-06 | 3.276e-06 | **Yes** |
| +8 | 300 | 40.00% | 54.67% | +14.67 | 12 / 56 | 6.209e-08 | 1.242e-07 | **Yes** |

## Inline vs Offline Accuracy (diagnostic only)

Inline `correct`/`correct_neutral` fields are process-state only and are never used as the headline metric. Reported here purely as a consistency check against offline `first_acc`. GSM-Hard has no inline field (label-free) and is not shown.

| task | alpha | inline_acc_pct | offline_first_acc | abs_gap_pp |
|---|---:|---:|---:|---:|
| gsm8k | -8 | 47.33% | 47.33% | 0.00 |
| gsm8k | +0 | 63.33% | 63.33% | 0.00 |
| gsm8k | +6 | 94.67% | 94.67% | 0.00 |
| gsm8k | +8 | 93.00% | 93.00% | 0.00 |
| math | -8 | 71.67% | 71.33% | 0.34 |
| math | +0 | 71.33% | 70.67% | 0.66 |
| math | +6 | 73.33% | 72.67% | 0.66 |
| math | +8 | 72.00% | 71.33% | 0.67 |


| Task | Condition | first_acc | early_candidate | reason_first |
|---|---|---:|---:|---:|
| GSM8K | Bare α=0 | 68.00% | 96.33% | 0.00% |
| GSM8K | Chat α=0 | 63.33% | 50.00% | 49.49% |
| GSM8K | Bare +6 | 78.00% | 45.33% | 51.67% |
| GSM8K | Chat +6 | **94.67%** | **1.00%** | **100.00%** |
| GSM8K | Bare +8 | 86.00% | 5.00% | 98.00% |
| GSM8K | Chat +8 | **93.00%** | **2.00%** | **99.67%** |
| MATH | Bare α=0 | 60.67% | 67.33% | 11.11% |
| MATH | Chat α=0 | **70.67%** | **0.00%** | **98.63%** |
| MATH | Bare +6 | 68.33% | 19.33% | 77.59% |
| MATH | Chat +6 | 72.67% | 0.33% | 98.97% |
| GSM-Hard | Bare α=0 | 34.00% | 93.67% | 0.00% |
| GSM-Hard | Chat α=0 | **40.00%** | **31.00%** | **69.05%** |
| GSM-Hard | Bare +8 | 50.33% | 6.00% | 98.00% |
| GSM-Hard | Chat +8 | **54.67%** | 5.33% | 99.00% |