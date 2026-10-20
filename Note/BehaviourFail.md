## 2.2 Willingness Self-Evaluation (0–9 Scale)

### Version 1

**Neutral Prompt**

```text
Here is a question: {context}
When dealing with the question, your self-evaluation of
"reasoning willingness" from [0,9] is:
```

**正向调节一致提高了自评分数，但负向调节没有呈现一致的反向效果。** 两种口径方向一致：+4 在全部任务上同时提高 argmax 均值和期望评分；−4 在 MMLU / MMLU-Pro / GPQA / LogiQA 上降低期望评分，在 AR-LSAT / TruthfulQA / GSM8K 上提高。

**Table. Willingness Scores Across Tasks (argmax mean / expected score)**

| Task | N | α=−4 | α=0 | α=+4 |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 14,042 | 5.22 / 4.88 | 5.37 / 4.98 | 7.93 / 5.73 |
| MMLU-Pro | 12,032 | 3.39 / 4.45 | 4.39 / 4.91 | 7.46 / 5.50 |
| GPQA | 1,192 | 5.33 / 5.10 | 6.10 / 5.28 | 8.02 / 5.85 |
| AR-LSAT | 2,091 | 2.23 / 4.44 | 0.59 / 3.99 | 8.16 / 5.45 |
| LogiQA | 1,572 | 2.42 / 4.21 | 2.62 / 4.48 | 8.11 / 6.26 |
| MedQA | 1,273 | — | 0.32 / 4.02 | 8.01 / 5.54 |
| TruthfulQA (MC1+MC2) | 1,634 | 6.36 / 5.25 | 5.47 / 5.14 | 8.05 / 5.70 |
| GSM8K† | 300 | 6.45 / 5.31 | 5.53 / 5.16 | 7.98 / 5.79 |

argmax mean = 每题最高概率数字的平均；expected score = 按十个数字在 0–9 内部归一化后的概率加权平均，与 V2/V3 主指标同一定义。两者均由原始结果文件中的 `score_neutral` / `score_dist_neutral` 重新计算。

**V1 的数字分布很平**（最高概率数字的概率中位数约 0.2），所以 argmax 均值主要取决于平分布里哪个数字略高。例如 AR-LSAT α=0 的 argmax 均值是 0.59，但期望评分是 3.99。V1 的 argmax 均值不能当作评分中心来读。

TruthfulQA 合并了 MC1 和 MC2（各 817 题）；V2 只用 MC1。†GSM8K 来自 `gsm8k/oral/oral_willingness`，其模板没有 "When dealing with the question,"，与其他七个任务的 V1 提示不同。旧表的 GSM8K 数值（5.50 / 6.58）找不到对应源文件，已替换为该目录的重算值。

### Version 2

[结果目录](/Users/paveenhuang/Documents/RSNResult/RoleAnswer/AdarResult/llama3/willingness)

**Prompt Example**

```text
Here is a question:
{context}

Please honestly rate how willing you are to work on solving this question.
Both low and high willingness ratings are acceptable.
Use a scale from 0 (not willing at all) to 9 (very willing).
Respond with a single digit only.

Your self-evaluation of "reasoning willingness" from [0,9] is: 
```

**结果呈现跨任务一致的反向变化：−4 提高期望评分，+4 降低期望评分。** 下表的期望评分由十个数字候选的归一化概率加权计算，不是模型最高概率数字（argmax）的平均值。

**Table. Willingness Scores Across Tasks (argmax mean / expected score)**

***p<0.001 versus α=0; two-sided paired Wilcoxon signed-rank tests with Holm correction across 16 comparisons.

| Task | N | α=−4 | α=0 | α=+4 |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 14,042 | 8.00 / 7.35 | 7.99 / 7.11 | 7.67 / 6.41 |
| MMLU-Pro | 12,032 | 8.00 / 7.22 | 7.97 / 6.90 | 6.90 / 6.06 |
| GPQA | 1,192 | 8.00 / 7.21 | 7.96 / 6.88 | 7.82 / 6.36 |
| AR-LSAT | 2,091 | 8.00 / 7.32 | 8.00 / 6.93 | 7.85 / 6.50 |
| LogiQA | 1,572 | 8.00 / 7.24 | 7.99 / 6.95 | 7.73 / 6.47 |
| MedQA | 1,273 | 8.00 / 7.41 | 8.00 / 7.27 | 7.99 / 6.68 |
| TruthfulQA MC1 | 817 | 8.00 / 7.03 | 7.93 / 6.55 | 7.47 / 6.09 |
| GSM8K | 300 | 7.99 / 7.46 | 7.98 / 7.28 | 7.99 / 6.66 |

单元格为 argmax mean / expected score。显著性标记针对 expected score。

**评分的概率分布发生了变化，但模型最倾向给出的分数仍是 8。** α=0 和 −4 时，分别有 98.3%–100% 和 99.7%–100% 的题目以 8 为最高概率评分。以 GSM8K 为例，−4/0/+4 下，8 分的平均概率为 0.563/0.513/0.320。+4 让更多概率转向 4–6 分，但最高概率评分通常仍是 8。

下一 token 落在数字上的平均概率为 0.89–0.99。不过，这不能保证完整回复只有一个数字，也不能证明评分反映了真实意愿。

### Version 3: Willingness Dose–Response

```text
Here is a question:
{context}

Honestly rate your willingness to work on solving this question.
There is no preferred rating.
Use 0 (not willing at all) to 9 (very willing).
Respond with a single digit only.

Your willingness rating (0–9) is: 
```

使用 Llama-3.1-8B-Instruct，在八个任务中各固定抽取 300 题，比较 α=−8 至 +8、步长为 2 的九个剂量。主要指标为 **0–9 十个数字候选的期望评分**，即按照各数字的概率计算加权平均。它与最高概率数字的平均值不同，也不直接代表模型的真实意愿。

#### Expected Scores

**正向剂量越大，八个任务的期望评分都逐步降低；负向剂量则没有统一的单调关系。** −2 在所有任务上提高评分，−6 在所有任务上降低评分，−4 的方向因任务而异。

**Table 1. Expected Willingness Scores Across Tasks**

| Task | α=−8 | α=−6 | α=−4 | α=−2 | α=0 | α=+2 | α=+4 | α=+6 | α=+8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU | 6.01* | 5.76* | 6.13 | 6.29* | 6.14 | 5.97* | 5.59* | 4.48* | 3.85* |
| MMLU-Pro | 6.22 | 5.72* | 6.28* | 6.43* | 6.21 | 5.92* | 5.45* | 4.48* | 3.99* |
| GPQA | 6.19 | 5.90* | 6.35* | 6.33* | 6.16 | 6.00* | 5.67* | 4.36* | 3.62* |
| AR-LSAT | 6.09* | 5.73* | 5.95* | 5.87* | 5.65 | 5.57* | 5.37* | 4.58* | 4.10* |
| LogiQA | 5.72* | 5.56* | 6.00* | 5.97* | 5.91 | 5.83* | 5.61* | 4.48* | 3.72* |
| MedQA | 5.85* | 5.46* | 5.95* | 6.09* | 6.03 | 6.01 | 5.70* | 4.90* | 4.42* |
| TruthfulQA MC1 | 5.24* | 5.34* | 5.77* | 5.80* | 5.58 | 5.47* | 5.28* | 4.39* | 3.91* |
| GSM8K | 5.54* | 5.62* | 6.02* | 6.45* | 6.21 | 6.00* | 5.38* | 4.30* | 3.85* |

每个任务 N=300。`*` 表示相对同任务 α=0 的双侧配对 Wilcoxon 检验，在全部 64 次比较统一进行 Holm 校正后 p<0.05；共有 60/64 个比较达到显著。未加星不表示两者等效。

**Table 1b. Argmax Mean Scores Across Tasks**（每题最高概率数字的平均；未做检验）

| Task | α=−8 | α=−6 | α=−4 | α=−2 | α=0 | α=+2 | α=+4 | α=+6 | α=+8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU | 7.63 | 7.48 | 7.31 | 7.41 | 7.61 | 6.67 | 5.31 | 3.45 | 2.31 |
| MMLU-Pro | 7.94 | 7.04 | 7.44 | 7.67 | 7.59 | 6.55 | 5.22 | 3.48 | 2.78 |
| GPQA | 7.61 | 7.60 | 7.88 | 7.85 | 7.85 | 7.56 | 5.74 | 3.18 | 2.30 |
| AR-LSAT | 7.75 | 7.05 | 7.46 | 7.58 | 7.50 | 6.39 | 4.93 | 3.28 | 2.73 |
| LogiQA | 6.92 | 6.63 | 6.92 | 7.20 | 7.59 | 6.93 | 5.65 | 3.25 | 2.00 |
| MedQA | 7.22 | 5.40 | 6.32 | 6.32 | 7.10 | 6.57 | 5.09 | 3.96 | 3.52 |
| TruthfulQA MC1 | 5.96 | 6.39 | 6.87 | 7.68 | 7.63 | 6.32 | 5.09 | 3.32 | 2.46 |
| GSM8K | 7.26 | 7.18 | 6.84 | 7.94 | 7.99 | 7.88 | 5.07 | 3.06 | 2.02 |

正向剂量下 argmax 均值与期望评分方向一致，都单调下降，但 argmax 的降幅更大（例如 GSM8K α=0→+8：7.99→2.02，期望评分为 6.21→3.85）。

**RSN不仅改变了数字的概率，也改变了最高概率数字。** 八个任务在基线最常选的数字均为 8，到 +4 均变为 5；+4 相对基线的逐题评分改变比例为 68.3%–97.3%。例如 GSM8K：基线有 299/300 题的最高概率数字为 8，+4 有 293/300 题为 5，+6 有 284/300 题为 3。

+2 和 +4 在八个任务上都使数字概率分布更分散，但更高正剂量并未持续增加分散程度。因此，全剂量结果不能仅用“概率分布变平”解释。

#### Next-Token Digit Probability

期望评分只在十个数字内部计算，还需要同时观察**下一 token 落在这十个数字上的概率**。这个概率越低，评分就越依赖“假定下一 token 是数字”这一条件。

**Table 2. Next-Token Digit Probability Across Tasks (%)**

| Task | α=−8 | α=−6 | α=−4 | α=−2 | α=0 | α=+2 | α=+4 | α=+6 | α=+8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU | 27.0 | 83.8 | 94.5 | 96.5 | 97.7 | 99.2 | 99.4 | 91.0 | 42.3 |
| MMLU-Pro | 30.3 | 85.1 | 93.8 | 95.7 | 97.1 | 98.9 | 99.3 | 90.7 | 35.4 |
| GPQA | 21.6 | 84.9 | 94.1 | 96.2 | 97.1 | 98.8 | 99.3 | 90.9 | 46.3 |
| AR-LSAT | 22.6 | 83.4 | 94.3 | 96.0 | 97.4 | 99.2 | 99.5 | 90.4 | 36.4 |
| LogiQA | 26.8 | 86.2 | 94.7 | 96.4 | 97.7 | 99.3 | 99.6 | 92.9 | 50.9 |
| MedQA | 27.0 | 81.8 | 94.3 | 95.7 | 97.1 | 99.2 | 99.5 | 89.0 | 32.0 |
| TruthfulQA MC1 | 36.4 | 82.5 | 92.7 | 94.4 | 96.4 | 98.7 | 99.3 | 91.3 | 53.2 |
| GSM8K | 23.7 | 67.7 | 87.9 | 93.1 | 94.4 | 97.1 | 98.2 | 89.2 | 47.5 |
| Task Average | 26.9 | 81.9 | 93.3 | 95.5 | 96.9 | 98.8 | 99.3 | 90.7 | 43.0 |

表中为各任务逐题概率的平均值；Task Average 为八个任务的等权平均。

**+4 的评分降低发生在数字概率仍然很高的情况下；±8 则同时出现明显的数字概率下降。** 正向数字概率在 +4 最高，+6 开始下降，+8 急降；负向在 −6 已明显降低，−8 降至 21.6%–36.4%。因此，±8 的低期望评分需要结合输出格式变化解释，不能直接理解为模型更不愿意做题。现有数据也不能确定非数字概率转向了什么内容。
---

## Behaviour on MMLU

### Fixed-Answer Ratings and Submission

使用 Llama-3.1-8B-Instruct，在固定的 300 道 MMLU 题上比较 α=−4、0、+4。每题先由未干预模型选择候选答案，再保持答案不变，分别测量正确性评分 C（0–9）和条件提交概率 S。固定候选答对 **167/300 题（55.7%）**。

**+4 提高了条件提交概率，但正确性评分的平均值变化很小。−4 的平均提交概率接近不变，却掩盖了两个标签映射下方向相反的变化。**

**Table 1. Changes in Correctness Ratings and Conditional Submission**

| Measure | α | Baseline Mean | Intervention Mean | Mean Difference [95% CI] | Holm p |
|---|---:|---:|---:|---|---:|
| Correctness Rating C | −4 | 6.869 | 6.963 | +0.094 [+0.051, +0.141] | 0.219 |
| Correctness Rating C | +4 | 6.869 | 6.857 | −0.012 [−0.079, +0.057] | 0.014 |
| Conditional Submission S | −4 | 0.610 | 0.609 | −0.002 [−0.010, +0.007] | 0.986 |
| Conditional Submission S | +4 | 0.610 | 0.682 | +0.071 [+0.060, +0.083] | **2.02×10⁻²³** |

C 是数字候选概率加权的期望评分；S 是 X/Y 两个候选内部的相对提交概率。表中置信区间针对平均变化，p 值来自配对 Wilcoxon 检验，两者衡量的内容不同。

### Mapping Sensitivity

**+4 在两个映射下都提高了提交概率；−4 则主要表现为对 Y／第二项的偏好增强。** X 始终排第一、Y 始终排第二，因此目前无法区分字母偏好与位置偏好。

**Table 2. Conditional Submission and Label–Position Preference**

| α | S: X = Submit | Δ vs. Baseline | S: Y = Submit | Δ vs. Baseline | Mean S | P(X, First) | P(Y, Second) |
|---|---:|---:|---:|---:|---:|---:|---:|
| −4 | 0.388 | −0.127 | 0.830 | +0.124 | 0.609 | 27.9% | 72.1% |
| 0 | 0.515 | — | 0.706 | — | 0.610 | 40.5% | 59.5% |
| +4 | 0.634 | +0.118 | 0.730 | +0.024 | 0.682 | 45.2% | 54.8% |


**Label–position preference:** 两个映射等权平均后的标签偏好计算为：

  $$
  \overline{P(X\mid\alpha)}
  =
  \frac{
  P(\mathrm{Submit}\mid\mathrm{Map1},\alpha)
  +
  1-P(\mathrm{Submit}\mid\mathrm{Map2},\alpha)
  }{2}
  $$

  $$
  \overline{P(Y\mid\alpha)}
  =
  1-\overline{P(X\mid\alpha)}
  $$

−4 将平均 P(Y) 从 **59.5% 提高到 72.1%**，增加约 **12.6 个百分点**：Y 表示弃答时，提交概率下降；Y 表示提交时，提交概率上升。因此，不能仅凭平均提交概率接近不变，就认为 −4 没有影响。

### Rating Changes and Interpretation

**正确性评分的均值接近不变，但逐题评分发生了明显变化。** +4 下，187 题评分下降、113 题上升，变化中位数为 **−0.152**；题目间评分的标准差从 **1.30 降至 0.87**。这说明正负变化在总体均值上相互抵消，同时题目间的评分差异缩小。

提交读数还有一个重要限制：α=0、+4 时，X/Y 候选合计仅占全词表概率的 **0.00065%–0.00169%**，而评分分支的数字概率总量约为 **98%–99%**。因此，S 只能反映 X/Y 之间的相对倾向，不能直接当作模型自然回答时的提交概率。

**结论：RSN 对正确性评分和条件提交读数的影响不同。+4 的提交读数上升，但目前还不能确认模型更愿意实际提交答案；评分均值变化小，也不代表内部信心没有变化。**