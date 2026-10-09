## 2.2 Willingness Self-Evaluation (0–9 Scale)

### Version 1

**Neutral Prompt**

```text
Here is a question: {context}
When dealing with the question, your self-evaluation of
"reasoning willingness" from [0,9] is:
```

**正向调节一致提高了自评分数，但负向调节没有呈现一致的反向效果。**
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

**Table. Expected Willingness Scores Across Tasks**

***p<0.001 versus α=0; two-sided paired Wilcoxon signed-rank tests with Holm correction across 16 comparisons.

| Task | N | α=−4 | α=0 | α=+4 |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 14,042 | 7.35 | 7.11 | 6.41 |
| MMLU-Pro | 12,032 | 7.22 | 6.90 | 6.06 |
| GPQA | 1,192 | 7.21 | 6.88 | 6.36 |
| AR-LSAT | 2,091 | 7.32 | 6.93 | 6.50 |
| LogiQA | 1,572 | 7.24 | 6.95 | 6.47 |
| MedQA | 1,273 | 7.41 | 7.27 | 6.68 |
| TruthfulQA MC1 | 817 | 7.03 | 6.55 | 6.09 |
| GSM8K | 300 | 7.46 | 7.28 | 6.66 |

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
