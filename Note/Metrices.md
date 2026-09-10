 
Note：
CRUXEval-O Llama3无结果
ProofWriter-OWA 只整理了Chat版本

## Llama3.1-8B-Instruct — GSM8K

| 指标                    | 含义                                 |
| --------------------- | ---------------------------------- |
| `α`                   | RSN 干预强度；不同模型间不能直接比较数值大小。          |
| `n`                   | 该条件下的样本数。                          |
| `first_acc`           | 按第一次正式答案判断的准确率，主要性能指标。             |
| `last_acc`            | 按最后一次正式答案判断的准确率，用于观察答案修订。          |
| `fixed_n`             | 第一次答错、最后一次改对的样本数。                  |
| `broke_n`             | 第一次答对、最后一次改错的样本数。                  |
| `valid_sub_rate`      | 能解析出正式答案标记的比例。                     |
| `cond_acc`            | 只在存在有效正式答案的样本中计算的准确率。              |
| `early_cand_rate`     | 输出开头很早就出现答案候选的比例。                  |
| `cand_coverage`       | 能定位到第一个答案候选的样本比例。指标可计算性的覆盖率/质量检查指标 |
| `reason_first_rate` | 首次答案候选前出现 `=`、`Step 1` 或编号步骤等推理文本的比例；属于表面文本代理指标。 |
| `pre_cand_chars_med`  | 首个答案候选之前的字符数中位数。                   |
| `post_cand_chars_med` | 首个答案候选之后继续生成的字符数中位数。               |
| `posN_med`            | 正式答案标记在全文中的归一化位置中位数，范围为 0–1。       |
| `multi_marker_rate`   | 同一输出中出现多个正式答案标记的比例。                |

> **备注：**“答案候选（candidate）”与“正式答案标记（marker）”不是同一事件。candidate 是输出中最早出现的答案形态数值或表达式，例如开头的裸数字、等号右侧结果或 “the answer is …” 后的值；它不要求与正确答案匹配，也可能是中间结果。marker 则是 `####`（GSM8K）或 `\boxed{}`（MATH）等正式提交格式。因此，candidate 通常用于分析答案形成顺序，`posN_med` 用于分析正式提交位置。


## Llama3.1-8B-Instruct — MATH

### No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -8 | 300 | 39.33% | 41.00% | 90.67% | 42.65% | 15.33% | 79.86% | 344 | 4308 | 0.2776 | 69.00% |
| -6 | 300 | 43.33% | 44.00% | 92.67% | 46.40% | 6.33% | 79.44% | 360 | 3449 | 0.3142 | 68.67% |
| -4 | 300 | 40.00% | 39.67% | 87.33% | 45.80% | 10.33% | 53.79% | 68 | 4646 | 0.2140 | 66.00% |
| +0 | 300 | 36.67% | 36.00% | 86.00% | 42.25% | 26.67% | 47.18% | 7 | 5306 | 0.1467 | 63.00% |
| +4 | 300 | 33.00% | 34.00% | 85.33% | 37.89% | 58.33% | 30.11% | 0 | 5596 | 0.1561 | 68.67% |

### CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -8 | 300 | 45.33% | 44.67% | 94.33% | 48.06% | 3.33% | 94.96% | 462 | 3061 | 0.4477 | 63.33% |
| -6 | 300 | 49.00% | 48.00% | 92.33% | 53.07% | 1.00% | 97.85% | 493 | 3306 | 0.3413 | 62.00% |
| -4 | 300 | 45.00% | 44.00% | 92.33% | 48.74% | 2.33% | 92.47% | 443 | 3486 | 0.3257 | 63.33% |
| +0 | 300 | 42.00% | 41.00% | 92.67% | 45.32% | 12.00% | 79.36% | 290 | 4815 | 0.2330 | 66.33% |
| +4 | 300 | 38.67% | 38.00% | 87.00% | 44.06% | 45.67% | 44.04% | 0 | 3733 | 0.2791 | 66.00% |


## Llama3.1-8B-Instruct — GSM-Hard

### No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -8 | 300 | 11.00% | 11.33% | 67.67% | 6.40% | 66.00% | 27.10% | 5 | 2162 | 0.0000 | 10.33% | prospective_blind_selection |
| -6 | 300 | 24.33% | 23.67% | 54.67% | 20.73% | 28.67% | 61.82% | 204 | 1877 | 0.2740 | 13.67% | prospective_blind_selection |
| -4 | 300 | 24.00% | 23.00% | 53.00% | 28.30% | 28.00% | 38.19% | 0 | 1920 | 0.2351 | 15.67% | prospective_blind_selection |
| +0 | 300 | 18.00% | 17.33% | 54.33% | 20.25% | 45.67% | 26.64% | 0 | 1952 | 0.2161 | 13.33% | prospective_blind_selection |
| +4 | 300 | 17.00% | 17.33% | 44.67% | 19.40% | 60.00% | 15.79% | 0 | 1990 | 0.1274 | 11.33% | prospective_blind_selection |

### CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 26.00% | 27.00% | 52.00% | 17.95% | 30.33% | 63.74% | 234 | 1928 | 0.0000 | 17.00% | condition_transfer_supplement |
| -4 | 300 | 30.00% | 29.67% | 45.67% | 27.01% | 19.00% | 74.63% | 292 | 1864 | 0.3294 | 11.67% | post_hoc_local_stability |
| +0 | 300 | 20.00% | 20.67% | 41.33% | 19.35% | 43.67% | 41.29% | 0 | 1930 | 0.2810 | 11.00% | condition_transfer_supplement |

## Qwen2.5-7B-Instruct — GSM-Hard

### No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -4 | 300 | 34.33% | 36.67% | 79.33% | 34.03% | 94.67% | 0.00% | 0 | 1321 | 0.8062 | 35.00% | prospective_blind_selection |
| +0 | 300 | 34.00% | 34.67% | 72.67% | 36.24% | 93.67% | 0.00% | 0 | 1266 | 0.7680 | 29.00% | prospective_blind_selection |
| +4 | 300 | 34.67% | 36.00% | 78.00% | 35.90% | 92.00% | 0.67% | 0 | 1264 | 0.6791 | 39.33% | prospective_blind_selection |
| +6 | 300 | 40.33% | 40.33% | 86.33% | 42.47% | 58.67% | 33.67% | 0 | 1057 | 0.5969 | 39.33% | prospective_blind_selection |
| +8 | 300 | 50.33% | 48.33% | 98.33% | 50.51% | 6.00% | 98.00% | 204 | 614 | 0.7765 | 34.00% | prospective_blind_selection |
| +10 | 300 | 50.33% | 46.33% | 98.33% | 51.19% | 4.67% | 98.00% | 238 | 698 | 0.7987 | 32.33% | post_hoc_local_stability |

### CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | evidence_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +0 | 300 | 38.00% | 36.67% | 72.00% | 37.96% | 96.00% | 0.33% | 0 | 1456 | 0.8611 | 30.67% | condition_transfer_supplement |
| +6 | 300 | 49.00% | 47.33% | 91.00% | 50.92% | 48.00% | 59.33% | 108 | 812 | 0.8359 | 34.33% | post_hoc_local_stability |
| +8 | 300 | 51.33% | 50.33% | 98.67% | 52.03% | 8.33% | 97.33% | 226 | 582 | 0.8286 | 31.00% | condition_transfer_supplement |
| +10 | 300 | 50.33% | 49.33% | 98.67% | 51.01% | 2.67% | 98.33% | 228 | 580 | 0.8251 | 30.00% | post_hoc_local_stability |


## Llama3.1-8B-Instruct — GSM-Symbolic No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 900 | 57.56% | 55.56% | 50.89% | 53.98% | 17.89% | 60.07% | 167 | 1906 | 0.0751 | 0.3429 | 15.33% |
| -4 | 900 | 52.34% | 50.33% | 55.44% | 52.81% | 19.78% | 27.92% | 0 | 1908 | 0.0000 | 0.3042 | 18.00% |
| +0 | 900 | 48.11% | 45.78% | 61.67% | 54.04% | 25.22% | 31.18% | 0 | 1891 | 0.0000 | 0.2988 | 18.66% |
| +4 | 900 | 38.89% | 38.00% | 47.00% | 44.25% | 49.67% | 13.72% | 0 | 2012 | 0.0000 | 0.2088 | 14.78% |

## Llama3.1-8B-Instruct — GSM-Symbolic CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 900 | 56.67% | 58.89% | 35.11% | 43.76% | 16.45% | 79.29% | 291 | 1895 | 0.1356 | 0.3204 | 13.11% |
| -4 | 900 | 58.56% | 59.33% | 31.00% | 50.05% | 14.00% | 70.35% | 272 | 1899 | 0.1267 | 0.3756 | 10.67% |
| +0 | 900 | 55.56% | 55.56% | 34.78% | 51.84% | 28.33% | 59.26% | 209 | 1917 | 0.0951 | 0.3835 | 12.67% |
| +4 | 900 | 40.22% | 39.11% | 28.67% | 45.52% | 71.66% | 11.38% | 0 | 2186 | 0.0000 | 0.2092 | 7.56% |

## Qwen2.5-7B-Instruct — GSM-Symbolic No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 900 | 50.22% | 54.22% | 73.78% | 46.43% | 97.22% | 0.00% | 0 | 1593 | 0.0000 | 0.8368 | 34.33% |
| +0 | 900 | 53.33% | 56.11% | 69.11% | 51.23% | 96.22% | 0.00% | 0 | 1599 | 0.0000 | 0.8624 | 28.67% |
| +6 | 900 | 60.67% | 60.89% | 83.11% | 63.06% | 53.44% | 46.22% | 0 | 1036 | 0.0000 | 0.8226 | 25.11% |
| +8 | 900 | 66.89% | 65.33% | 98.33% | 67.49% | 6.56% | 98.44% | 170 | 706 | 0.1753 | 0.8318 | 27.22% |

## Qwen2.5-7B-Instruct — GSM-Symbolic CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 900 | 56.44% | 57.78% | 79.89% | 55.37% | 99.33% | 0.00% | 0 | 1626 | 0.0000 | 0.9007 | 38.67% |
| +0 | 900 | 52.89% | 53.11% | 85.22% | 52.52% | 99.11% | 0.00% | 0 | 1606 | 0.0000 | 0.9113 | 38.56% |
| +6 | 900 | 65.11% | 64.22% | 92.56% | 66.57% | 43.78% | 63.89% | 114 | 918 | 0.1216 | 0.8793 | 31.45% |
| +8 | 900 | 65.00% | 64.56% | 98.89% | 64.99% | 6.44% | 96.33% | 112 | 556 | 0.1859 | 0.8534 | 28.11% |

## Per-config accuracy audit (main/p1/p2)

Pooled `first_acc` here is the SAME equal-weighted mean used in the main tables; shown per-config to make cross-perturbation spread visible.

| model | condition | α | main first_acc | p1 first_acc | p2 first_acc | pooled first_acc (equal-weight) |
|---|---|---|---|---|---|---|
| llama3 | CoT | -6 | 64.00% | 59.00% | 47.00% | 56.67% |
| llama3 | CoT | -4 | 70.67% | 63.67% | 41.33% | 58.56% |
| llama3 | CoT | +0 | 62.67% | 58.67% | 45.33% | 55.56% |
| llama3 | CoT | +4 | 54.00% | 41.33% | 25.33% | 40.22% |
| llama3 | No-CoT | -6 | 71.00% | 60.67% | 41.00% | 57.56% |
| llama3 | No-CoT | -4 | 62.67% | 56.67% | 37.67% | 52.34% |
| llama3 | No-CoT | +0 | 57.33% | 53.33% | 33.67% | 48.11% |
| llama3 | No-CoT | +4 | 50.67% | 39.33% | 26.67% | 38.89% |
| qwen2.5 | CoT | -6 | 66.00% | 64.33% | 39.00% | 56.44% |
| qwen2.5 | CoT | +0 | 63.33% | 56.33% | 39.00% | 52.89% |
| qwen2.5 | CoT | +6 | 79.33% | 68.33% | 47.67% | 65.11% |
| qwen2.5 | CoT | +8 | 81.67% | 66.67% | 46.67% | 65.00% |
| qwen2.5 | No-CoT | -6 | 59.33% | 54.67% | 36.67% | 50.22% |
| qwen2.5 | No-CoT | +0 | 65.33% | 56.33% | 38.33% | 53.33% |
| qwen2.5 | No-CoT | +6 | 73.00% | 64.00% | 45.00% | 60.67% |
| qwen2.5 | No-CoT | +8 | 78.00% | 71.67% | 51.00% | 66.89% |

## Llama3.1-8B-Instruct — BBH object_counting

### No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | no_marker_rate | marker_unparsed_rate | degenerate_tail_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 250 | 40.80% | 40.80% | 82.00% | 47.80% | 84.40% | 9.20% | 0 | 1922 | 0.0875 | 51.60% | 10.40% | 7.60% | 6.00% |
| +0 | 250 | 41.60% | 41.20% | 86.80% | 46.54% | 95.20% | 2.00% | 0 | 1983 | 0.0633 | 63.60% | 9.20% | 4.00% | 3.60% |
| +4 | 250 | 32.80% | 32.80% | 83.60% | 35.89% | 99.20% | 0.00% | 0 | 2002 | 0.0963 | 58.00% | 10.80% | 5.60% | 4.00% |

### CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | no_marker_rate | marker_unparsed_rate | degenerate_tail_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 250 | 56.80% | 56.80% | 92.00% | 54.35% | 63.20% | 39.92% | 5 | 1790 | 0.0011 | 52.00% | 0.40% | 7.60% | 4.00% |
| -4 | 250 | 44.80% | 44.80% | 97.20% | 43.62% | 88.00% | 8.80% | 0 | 1790 | 0.0011 | 77.20% | 0.40% | 2.40% | 1.20% |
| +0 | 250 | 40.80% | 40.80% | 95.60% | 39.33% | 97.20% | 1.60% | 0 | 1790 | 0.0011 | 80.80% | 0.00% | 4.40% | 2.80% |
| +4 | 250 | 32.00% | 32.00% | 90.00% | 30.67% | 99.20% | 0.00% | 0 | 1790 | 0.0011 | 78.00% | 0.00% | 10.00% | 4.80% |

## Qwen2.5-7B-Instruct — BBH object_counting

### No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | no_marker_rate | marker_unparsed_rate | degenerate_tail_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 250 | 56.80% | 57.60% | 86.40% | 53.24% | 100.00% | 0.00% | 0 | 418 | 0.7428 | 20.00% | 0.00% | 13.60% | 2.80% |
| +0 | 250 | 55.20% | 56.00% | 86.80% | 52.53% | 100.00% | 0.00% | 0 | 422 | 0.7568 | 15.60% | 0.00% | 13.20% | 1.20% |
| +8 | 250 | 57.60% | 56.40% | 97.20% | 56.38% | 44.40% | 37.40% | 50 | 150 | 0.6986 | 30.80% | 0.00% | 2.80% | 0.40% |

### CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | posN_med | multi_marker_rate | no_marker_rate | marker_unparsed_rate | degenerate_tail_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 250 | 45.60% | 46.00% | 43.20% | 62.04% | 100.00% | 0.00% | 0 | 20 | 0.6497 | 8.80% | 0.00% | 56.80% | 0.00% |
| +0 | 250 | 52.80% | 54.00% | 38.40% | 62.50% | 100.00% | 0.00% | 0 | 18 | 0.3227 | 8.40% | 0.00% | 61.60% | 0.00% |
| +6 | 250 | 43.20% | 44.00% | 52.40% | 53.44% | 96.80% | 3.20% | 0 | 89 | 0.6673 | 23.60% | 0.00% | 47.60% | 0.00% |
| +8 | 250 | 66.80% | 66.40% | 97.60% | 68.44% | 7.20% | 80.32% | 211 | 143 | 0.6868 | 23.60% | 0.00% | 2.40% | 0.40% |

## Llama3.1-8B-Instruct — CRUXEval-O Bare No-CoT

需要额外解释的主要有四个：

- `answer_first_rate`：输出去除空白后是否直接以 `####` 开头。它比 `early_cand_rate` 更窄，只表示“正式标记先出现”，不能识别先输出 literal、后补 `####` 的情况。

- `marker_present_rate`：是否出现过 `####`，不要求后面的内容能成功解析。它与 `valid_sub_rate` 不同。

- `nonliteral_rate`：出现了 `####`，但第一次提交的内容无法被 `ast.literal_eval` 解析为合法 Python literal。反映格式失败，不代表这些样本按官方执行式评测一定正确。

- `degenerate_tail_rate`：输出末尾是否发生重复循环。这个指标尤其需要说明，而且当前脚本实现有误；应直接使用原始数据中的 `degenerate_tail`，修正前不要解读结果。

另外：

- `pre_marker_chars_med` 测量的是正式 `####` 之前的字符数，不是答案候选之前的字符数。
- `truncated_rate` 只在 Chat 数据中有记录；Bare 的 `n/a` 表示缺少元数据，不代表没有截断。


| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | answer_first_rate | marker_present_rate | pre_marker_chars_med | posN_med | multi_marker_rate | nonliteral_rate | degenerate_tail_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 31.00% | 28.67% | 96.33% | 32.18% | 30.33% | 99.33% | 7 | 0.0029 | 59.00% | 3.00% | 88.67% | n/a | 2504 |
| -4 | 300 | 33.33% | 30.33% | 96.67% | 34.48% | 5.33% | 98.00% | 10 | 0.0048 | 46.33% | 1.33% | 88.33% | n/a | 2481 |
| +0 | 300 | 34.67% | 31.00% | 96.67% | 35.86% | 0.00% | 97.67% | 10 | 0.0048 | 38.00% | 1.00% | 86.00% | n/a | 2488 |
| +4 | 300 | 33.67% | 30.67% | 97.67% | 34.47% | 0.00% | 98.33% | 11 | 0.0051 | 45.00% | 0.67% | 86.33% | n/a | 2499 |

## Llama3.1-8B-Instruct — CRUXEval-O Bare CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | answer_first_rate | marker_present_rate | pre_marker_chars_med | posN_med | multi_marker_rate | nonliteral_rate | degenerate_tail_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 34.00% | 32.33% | 97.00% | 35.05% | 14.33% | 98.33% | 43 | 0.0228 | 42.33% | 1.33% | 85.33% | n/a | 2553 |
| -4 | 300 | 35.67% | 34.33% | 95.67% | 37.28% | 0.00% | 97.00% | 120 | 0.0467 | 34.00% | 1.33% | 86.67% | n/a | 2548 |
| +0 | 300 | 34.67% | 37.00% | 97.00% | 35.74% | 0.00% | 98.33% | 11 | 0.0047 | 33.00% | 1.33% | 92.33% | n/a | 2532 |
| +4 | 300 | 32.33% | 36.00% | 97.33% | 33.22% | 0.00% | 98.33% | 10 | 0.0043 | 34.00% | 1.00% | 93.00% | n/a | 2536 |

## Qwen2.5-7B-Instruct — CRUXEval-O Bare No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | answer_first_rate | marker_present_rate | pre_marker_chars_med | posN_med | multi_marker_rate | nonliteral_rate | degenerate_tail_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 24.00% | 23.67% | 77.67% | 30.90% | 14.00% | 99.00% | 2 | 0.1333 | 34.33% | 21.33% | 8.00% | n/a | 29 |
| +0 | 300 | 29.33% | 28.33% | 87.33% | 33.59% | 16.67% | 99.67% | 8 | 0.3571 | 22.67% | 12.33% | 3.33% | n/a | 30 |
| +6 | 300 | 30.33% | 25.00% | 85.33% | 35.55% | 4.00% | 98.67% | 12 | 0.4167 | 24.33% | 13.33% | 3.00% | n/a | 34 |
| +8 | 300 | 37.67% | 22.33% | 86.67% | 43.46% | 13.67% | 98.33% | 31 | 0.4697 | 55.67% | 11.67% | 5.00% | n/a | 96 |

## Qwen2.5-7B-Instruct — CRUXEval-O Bare CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | answer_first_rate | marker_present_rate | pre_marker_chars_med | posN_med | multi_marker_rate | nonliteral_rate | degenerate_tail_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 34.67% | 31.00% | 85.67% | 40.47% | 4.33% | 98.67% | 47 | 0.1866 | 69.33% | 13.00% | 21.67% | n/a | 670 |
| +0 | 300 | 34.67% | 28.67% | 83.67% | 41.43% | 7.33% | 99.33% | 46 | 0.3237 | 69.67% | 15.67% | 15.67% | n/a | 552 |
| +6 | 300 | 43.00% | 24.67% | 85.00% | 50.59% | 0.00% | 99.67% | 474 | 0.9035 | 67.67% | 14.67% | 15.33% | n/a | 589 |
| +8 | 300 | 54.00% | 29.00% | 95.67% | 56.45% | 0.00% | 99.00% | 686 | 0.9490 | 69.67% | 3.33% | 14.67% | n/a | 804 |

## Llama3.1-8B-Instruct — CRUXEval-O Chat No-CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | answer_first_rate | marker_present_rate | pre_marker_chars_med | posN_med | multi_marker_rate | nonliteral_rate | degenerate_tail_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 46.67% | 46.67% | 97.67% | 47.78% | 0.00% | 99.00% | 449 | 0.9700 | 0.00% | 1.33% | 0.67% | 1.33% | 466 |
| -4 | 300 | 43.33% | 43.33% | 97.00% | 44.67% | 0.00% | 98.00% | 460 | 0.9689 | 0.00% | 1.00% | 1.67% | 2.00% | 483 |
| +0 | 300 | 45.67% | 45.67% | 97.00% | 47.08% | 0.00% | 98.33% | 435 | 0.9684 | 0.00% | 1.33% | 1.67% | 2.00% | 458 |
| +4 | 300 | 41.00% | 41.00% | 97.00% | 42.27% | 0.00% | 97.33% | 186 | 0.9214 | 0.00% | 0.33% | 2.33% | 2.67% | 206 |

## Llama3.1-8B-Instruct — CRUXEval-O Chat CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | answer_first_rate | marker_present_rate | pre_marker_chars_med | posN_med | multi_marker_rate | nonliteral_rate | degenerate_tail_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 51.33% | 51.33% | 96.33% | 53.29% | 0.00% | 97.00% | 970 | 0.9853 | 0.00% | 0.67% | 3.00% | 3.00% | 1000 |
| -4 | 300 | 49.33% | 49.33% | 97.33% | 50.68% | 0.00% | 97.33% | 994 | 0.9855 | 0.00% | 0.00% | 1.67% | 2.67% | 1023 |
| +0 | 300 | 52.00% | 52.00% | 97.67% | 53.24% | 0.00% | 97.67% | 978 | 0.9861 | 0.00% | 0.00% | 1.33% | 2.33% | 1002 |
| +4 | 300 | 50.33% | 50.33% | 97.00% | 51.89% | 0.00% | 97.33% | 682 | 0.9769 | 0.00% | 0.33% | 1.33% | 3.00% | 710 |

## Llama3.1-8B-Instruct — FinQA CoT

- `loop_rate`：结尾 40 个字符在全文重复至少 4 次的比例，表示严重循环生成，不是“推理更多”。
- `truncated_rate`：生成达到最大 token 上限的比例，反映模型没有自然停止。
- `gen_chars_med`：全文字符数中位数；若 loop/truncation 很高，这列会被重复文本抬高。
- `candidate_posN_med`：首个答案候选的归一化位置；使用 GSM8K 检测器，在 FinQA 上属于探索性指标。
- `posN_med`：第一个合法 `#### <value>` 标记的位置，与 `candidate_posN_med` 检测的对象和统计子集不同，不能直接等同。
- `multi_marker_rate`：出现至少两个“可合法解析”的答案标记的比例，用来看重复提交或答案修订。

其中最需要突出的是：Llama 的 `post_cand_chars_med` 和 `gen_chars_med` 很大，主要可能来自循环与截断，不能解释成更长、更充分的推理。

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 8.67% | 8.00% | 99.67% | 8.70% | 34.67% | 3.97% | 5 | 828 | 0.0053 | 0.0000 | 10.33% | 93.33% | 100.00% | 868 |
| -4 | 300 | 9.00% | 7.67% | 99.33% | 9.06% | 75.33% | 2.53% | 0 | 804 | 0.0000 | 0.0057 | 19.67% | 95.33% | 100.00% | 868 |
| +0 | 300 | 14.33% | 12.67% | 99.67% | 14.38% | 89.67% | 3.46% | 0 | 826 | 0.0000 | 0.0077 | 21.67% | 96.67% | 100.00% | 847 |
| +4 | 300 | 15.33% | 13.67% | 94.67% | 16.20% | 81.33% | 10.49% | 0 | 872 | 0.0000 | 0.0113 | 30.67% | 93.33% | 100.00% | 891 |

## Qwen2.5-7B-Instruct — FinQA CoT

| α | n | first_acc | last_acc | valid_sub_rate | cond_acc | early_cand_rate | reason_first_rate | pre_cand_chars_med | post_cand_chars_med | candidate_posN_med | posN_med | multi_marker_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 10.00% | 9.67% | 91.67% | 10.91% | 100.00% | 0.00% | 0 | 742 | 0.0000 | 0.5400 | 42.67% | 19.00% | 28.67% | 746 |
| +0 | 300 | 20.67% | 20.67% | 92.67% | 22.30% | 99.33% | 0.00% | 0 | 512 | 0.0000 | 0.3773 | 36.33% | 15.33% | 20.67% | 516 |
| +6 | 300 | 20.67% | 20.33% | 96.67% | 21.38% | 54.00% | 37.37% | 5 | 288 | 0.0122 | 0.6611 | 34.33% | 14.00% | 18.33% | 516 |
| +8 | 300 | 25.67% | 24.33% | 97.67% | 26.28% | 43.33% | 49.15% | 150 | 259 | 0.4138 | 0.2812 | 48.67% | 12.33% | 16.67% | 554 |


## Llama3.1-8B-Instruct — ProofWriter-OWA Bare CoT

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | first_last_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 14.33% | 10.33% | 26.00% | 74.00% | 55.13% | 0.00% | 373 | 0.1013 | 21.67% | 61.54% | 93.33% | 100.00% | 4070 |
| -4 | 300 | 12.00% | 8.67% | 22.33% | 77.67% | 53.73% | 0.00% | 505 | 0.1335 | 19.00% | 50.88% | 90.00% | 100.00% | 4164 |
| +0 | 300 | 10.33% | 6.00% | 20.67% | 79.33% | 50.00% | 0.00% | 697 | 0.1872 | 17.67% | 39.62% | 93.33% | 100.00% | 4165 |
| +4 | 300 | 21.67% | 18.67% | 39.67% | 60.33% | 54.62% | 0.00% | 661 | 0.1752 | 35.33% | 40.57% | 95.00% | 100.00% | 4310 |

## Llama3.1-8B-Instruct — ProofWriter-OWA Chat CoT

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | first_last_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 39.33% | 39.33% | 73.67% | 26.33% | 53.39% | 0.00% | 1368 | 0.9915 | 0.00% | n/a | 17.00% | 21.00% | 1605 |
| -4 | 300 | 34.00% | 34.00% | 59.33% | 40.67% | 57.30% | 0.00% | 1202 | 0.9902 | 0.00% | n/a | 26.67% | 29.67% | 1527 |
| +0 | 300 | 33.00% | 33.00% | 60.33% | 39.67% | 54.70% | 0.00% | 1078 | 0.9897 | 0.00% | n/a | 25.67% | 29.67% | 1358 |
| +4 | 300 | 31.67% | 31.67% | 55.33% | 44.67% | 57.23% | 0.00% | 929 | 0.9877 | 0.00% | n/a | 26.67% | 29.00% | 1203 |

## Qwen2.5-7B-Instruct — ProofWriter-OWA Bare CoT

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | first_last_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 49.33% | 49.00% | 90.33% | 9.67% | 54.61% | 0.00% | 737 | 0.9456 | 19.67% | 1.69% | 14.33% | 14.67% | 850 |
| +0 | 300 | 46.33% | 46.33% | 99.67% | 0.33% | 46.49% | 0.00% | 462 | 0.9299 | 34.67% | 0.00% | 34.33% | 34.33% | 558 |
| +6 | 300 | 49.67% | 49.67% | 100.00% | 0.00% | 49.67% | 0.00% | 484 | 0.9347 | 32.00% | 0.00% | 32.00% | 32.00% | 611 |
| +8 | 300 | 52.00% | 52.00% | 100.00% | 0.00% | 52.00% | 0.00% | 565 | 0.6764 | 48.67% | 2.74% | 47.33% | 47.33% | 894 |

## Qwen2.5-7B-Instruct — ProofWriter-OWA Chat CoT

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | first_last_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 0.67% | 0.67% | 1.00% | 99.00% | 66.67% | 0.00% | 500 | 0.9785 | 0.00% | n/a | 0.00% | 0.00% | 5 |
| +0 | 300 | 41.00% | 41.00% | 100.00% | 0.00% | 41.00% | 50.33% | 0 | 0.0000 | 0.67% | 0.00% | 0.00% | 0.00% | 231 |
| +6 | 300 | 39.00% | 39.00% | 99.67% | 0.33% | 39.13% | 96.32% | 0 | 0.0000 | 0.00% | n/a | 0.00% | 0.00% | 12 |
| +8 | 300 | 47.67% | 47.67% | 99.67% | 0.33% | 47.83% | 4.01% | 549 | 0.9778 | 0.00% | n/a | 0.00% | 0.00% | 563 |


## Llama3.1-8B-Instruct — ProofWriter-OWA Chat CoT

- `valid_sub_rate`：至少出现一次严格格式 `#### True/False/Unknown` 的比例。
- `no_answer_rate`：没有出现严格答案标记的比例，等于 `100% - valid_sub_rate`。
- `cond_acc`：仅在有严格答案标记的样本中，按第一次标记计算的准确率。
- `answer_first_rate`：在有效提交样本中，首个非空行就是严格答案标记的比例。
- `pre_marker_chars_med`：首个严格答案标记之前的字符数中位数，仅在有效提交样本中统计。
- `marker_posN_med`：首个严格答案标记在全文中的归一化位置中位数，仅在有效提交样本中统计；这是输出位置，不是内部 commitment。
- `multi_marker_rate`：全文出现至少两个严格答案标记的比例，分母为全部样本。
- `first_last_disagreement_rate`：在至少有两个严格标记的样本中，第一次与最后一次答案标签不同的比例；最好同时报告该子集数量 `revision_eligible_n`。

此外要注明：`loop_rate`、`truncated_rate` 和 `gen_chars_med` 会影响长度及位置指标，尤其是 Llama，因此属于必要的解释性指标。

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | first_last_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 39.33% | 39.33% | 73.67% | 26.33% | 53.39% | 0.00% | 1368 | 0.9915 | 0.00% | n/a | 17.00% | 21.00% | 1605 |
| -4 | 300 | 34.00% | 34.00% | 59.33% | 40.67% | 57.30% | 0.00% | 1202 | 0.9902 | 0.00% | n/a | 26.67% | 29.67% | 1527 |
| +0 | 300 | 33.00% | 33.00% | 60.33% | 39.67% | 54.70% | 0.00% | 1078 | 0.9897 | 0.00% | n/a | 25.67% | 29.67% | 1358 |
| +4 | 300 | 31.67% | 31.67% | 55.33% | 44.67% | 57.23% | 0.00% | 929 | 0.9877 | 0.00% | n/a | 26.67% | 29.00% | 1203 |

## Qwen2.5-7B-Instruct — ProofWriter-OWA Chat CoT

| α | n | first_acc | last_acc | valid_sub_rate | no_answer_rate | cond_acc | answer_first_rate | pre_marker_chars_med | marker_posN_med | multi_marker_rate | first_last_disagreement_rate | loop_rate | truncated_rate | gen_chars_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 300 | 0.67% | 0.67% | 1.00% | 99.00% | 66.67% | 0.00% | 500 | 0.9785 | 0.00% | n/a | 0.00% | 0.00% | 5 |
| +0 | 300 | 41.00% | 41.00% | 100.00% | 0.00% | 41.00% | 50.33% | 0 | 0.0000 | 0.67% | 0.00% | 0.00% | 0.00% | 231 |
| +6 | 300 | 39.00% | 39.00% | 99.67% | 0.33% | 39.13% | 96.32% | 0 | 0.0000 | 0.00% | n/a | 0.00% | 0.00% | 12 |
| +8 | 300 | 47.67% | 47.67% | 99.67% | 0.33% | 47.83% | 4.01% | 549 | 0.9778 | 0.00% | n/a | 0.00% | 0.00% | 563 |

## Llama3.1-8B-Instruct — ZebraLogic-Easy

### Behavioural metrics (gold-free, computed by this script)

| α | n | valid_sub_rate | no_answer_rate | full_grid_rate | reason_before_solution_rate | pre_solution_chars_med | solution_posN_med | first_last_grid_agreement_med | first_last_grid_disagreement_rate | loop_rate | truncated_rate | gen_chars_med | gen_tokens_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 280 | 75.71% | 24.29% | 71.07% | 75.71% | 544 | 0.1295 | 1.0000 | 9.47% | 90.71% | 100.00% | 4192 | 1024 |
| -4 | 280 | 75.00% | 25.00% | 69.64% | 75.00% | 515 | 0.1236 | 1.0000 | 7.94% | 93.21% | 100.00% | 4162 | 1024 |
| +0 | 280 | 74.64% | 25.36% | 67.86% | 74.64% | 520 | 0.1270 | 1.0000 | 10.19% | 95.00% | 100.00% | 4118 | 1024 |
| +4 | 280 | 75.00% | 25.00% | 70.71% | 75.00% | 522 | 0.1234 | 1.0000 | 9.34% | 90.00% | 100.00% | 4168 | 1024 |

### Accuracy (background context, carried over from the frozen formal evaluation — not rederived here)

| α | first_puzzle_acc | last_puzzle_acc | first_cell_acc | last_cell_acc |
|---|---|---|---|---|
| -6 | 35.71% | 33.93% | 48.55% | 41.91% |
| -4 | 33.93% | 32.86% | 47.36% | 42.09% |
| +0 | 36.79% | 37.14% | 48.73% | 48.68% |
| +4 | 32.14% | 30.00% | 46.27% | 38.64% |

## Qwen2.5-7B-Instruct — ZebraLogic-Easy

### Behavioural metrics (gold-free, computed by this script)

| α | n | valid_sub_rate | no_answer_rate | full_grid_rate | reason_before_solution_rate | pre_solution_chars_med | solution_posN_med | first_last_grid_agreement_med | first_last_grid_disagreement_rate | loop_rate | truncated_rate | gen_chars_med | gen_tokens_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -6 | 280 | 100.00% | 0.00% | 92.50% | 100.00% | 508 | 0.1669 | 0.1111 | 67.27% | 6.07% | 60.36% | 3906 | 1024 |
| +0 | 280 | 100.00% | 0.00% | 93.57% | 100.00% | 506 | 0.1492 | 0.0714 | 76.34% | 7.50% | 72.86% | 3967 | 1024 |
| +6 | 280 | 99.64% | 0.36% | 96.79% | 99.64% | 546 | 0.1614 | 0.2000 | 60.65% | 9.29% | 68.57% | 3956 | 1024 |
| +8 | 280 | 39.29% | 60.71% | 38.93% | 39.29% | 479 | 0.3411 | 1.0000 | 2.75% | 62.50% | 63.57% | 3070 | 1024 |

### Accuracy (background context, carried over from the frozen formal evaluation — not rederived here)

| α | first_puzzle_acc | last_puzzle_acc | first_cell_acc | last_cell_acc |
|---|---|---|---|---|
| -6 | 30.00% | 16.07% | 58.32% | 35.14% |
| +0 | 34.64% | 15.36% | 64.05% | 28.45% |
| +6 | 36.07% | 12.14% | 65.64% | 35.95% |
| +8 | 23.93% | 24.64% | 23.73% | 24.18% |
