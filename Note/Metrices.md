 
Note：
CRUXEval-O Llama3无结果
ProofWriter-OWA 只整理了Chat版本

## Llama3.1-8B-Instruct — GSM8K



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
