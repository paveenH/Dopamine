
confidence_ratio = assertive_count / (hedging_count + self_correction_count + 1)

## LLama3-8B-IT

| Group | **W/o ref Acc** |
| --- | --- |
| **Neutral** | 61.67 |
| **4 [11,20)** | 50.67 |
| **-4 [11,20)** | 74.33 |

```bash
--- Analysis Configuration ---
Target Task:  gsm8k
Target Model: llama3
Using Cleaned Files: True
Files found:
  original: gsm8k/gsm8k_8B_answers_llama3_original_clean.json
  positive: gsm8k/gsm8k_8B_answers_llama3_positive_clean.json
  negative: gsm8k/gsm8k_8B_answers_llama3_negative_clean.json

================================================================================
LINGUISTIC MARKER ANALYSIS - CONFIDENCE COMPARISON
================================================================================
Metric                                   original      positive      negative
--------------------------------------------------------------------------------
Samples                                       300           300           300
Avg word count                             113.60        118.80        108.70

Hedging - total count                          13            22             7
Hedging - mean per sample                    0.04          0.07          0.02
Hedging - per 100 words                      0.06          0.08          0.02
Hedging - % samples with any                 3.70          5.30          2.00

Self-correction - total                         6             3             3
Self-correction - mean/sample                0.02          0.01          0.01
Self-correction - per 100 words              0.01          0.01          0.01
Self-correction - % with any                 1.30          1.00          0.70

Assertive - total count                       334           427           279
Assertive - mean per sample                  1.11          1.42          0.93
Assertive - per 100 words                    1.07          1.22          1.02
Assertive - % samples with any              61.70         64.30         58.70

Confidence ratio (assert/hedge)              1.08          1.37          0.91
================================================================================

  Top markers in [original]:
           assertive | \bthe answer is\b                        |  148
           assertive | \btherefore\b                            |  135
           assertive | \bwe (can )?(see|know|conclude)\b        |   45
             hedging | \bi hope it is correct\b                 |   12
     self_correction | \bactually\b                             |    4
           assertive | \bso the answer\b                        |    3
           assertive | \bsimpl[ey]\b                            |    3
     self_correction | \bi made (a )?mistake\b                  |    2
             hedging | \bmight\b                                |    1

  Top markers in [positive]:
           assertive | \bthe answer is\b                        |  200
           assertive | \btherefore\b                            |  165
           assertive | \bwe (can )?(see|know|conclude)\b        |   43
             hedging | \bi hope it is correct\b                 |   18
           assertive | \bso the answer\b                        |   11
           assertive | \bsimpl[ey]\b                            |    4
     self_correction | \bi made (a )?mistake\b                  |    3
           assertive | \bjust\b                                 |    2
           assertive | \bexact(ly)?\b                           |    2
             hedging | \bapproximately\b                        |    1

  Top markers in [negative]:
           assertive | \btherefore\b                            |  121
           assertive | \bthe answer is\b                        |  106
           assertive | \bwe (can )?(see|know|conclude)\b        |   45
             hedging | \bi hope it is correct\b                 |    6
           assertive | \bsimpl[ey]\b                            |    4
     self_correction | \bi made (a )?mistake\b                  |    3
           assertive | \bso the answer\b                        |    2
             hedging | \bcould be\b                             |    1
           assertive | \bhence\b                                |    1

================================================================================
BREAKDOWN BY CORRECTNESS
================================================================================

--- CORRECT answers ---

================================================================================
LINGUISTIC MARKER ANALYSIS - CONFIDENCE COMPARISON
================================================================================
Metric                                   original      positive      negative
--------------------------------------------------------------------------------
Samples                                       185           152           223
Avg word count                              99.50         96.90        102.30

Hedging - total count                           4            10             4
Hedging - mean per sample                    0.02          0.07          0.02
Hedging - per 100 words                      0.02          0.07          0.02
Hedging - % samples with any                 2.20          4.60          1.30

Self-correction - total                         0             1             2
Self-correction - mean/sample                0.00          0.01          0.01
Self-correction - per 100 words              0.00          0.00          0.01
Self-correction - % with any                 0.00          0.70          0.40

Assertive - total count                       177           187           189
Assertive - mean per sample                  0.96          1.23          0.85
Assertive - per 100 words                    1.06          1.20          1.00
Assertive - % samples with any              60.00         61.80         57.00

Confidence ratio (assert/hedge)              0.94          1.18          0.84
================================================================================

--- INCORRECT answers ---

================================================================================
LINGUISTIC MARKER ANALYSIS - CONFIDENCE COMPARISON
================================================================================
Metric                                   original      positive      negative
--------------------------------------------------------------------------------
Samples                                       115           148            77
Avg word count                             136.20        141.20        127.20

Hedging - total count                           9            12             3
Hedging - mean per sample                    0.08          0.08          0.04
Hedging - per 100 words                      0.11          0.09          0.02
Hedging - % samples with any                 6.10          6.10          3.90

Self-correction - total                         6             2             1
Self-correction - mean/sample                0.05          0.01          0.01
Self-correction - per 100 words              0.04          0.01          0.01
Self-correction - % with any                 3.50          1.40          1.30

Assertive - total count                       157           240            90
Assertive - mean per sample                  1.37          1.62          1.17
Assertive - per 100 words                    1.10          1.24          1.10
Assertive - % samples with any              64.30         66.90         63.60

Confidence ratio (assert/hedge)              1.29          1.56          1.11
================================================================================
```

## Qwen3-8B-IT

| Group | **W/o ref Acc** |
| --- | --- |
| **Neutral** | 41.67 |
| **4 [11,20)** | 44.33 |
| **-4 [11,20)** | 34.33 |

```bash
--- Analysis Configuration ---
Target Task:  gsm8k
Target Model: qwen3
Using Cleaned Files: True
Files found:
  original: gsm8k/gsm8k_8B_answers_qwen3_original_clean.json
  positive: gsm8k/gsm8k_8B_answers_qwen3_positive_clean.json
  negative: gsm8k/gsm8k_8B_answers_qwen3_negative_clean.json

================================================================================
LINGUISTIC MARKER ANALYSIS - CONFIDENCE COMPARISON
================================================================================
Metric                                   original      positive      negative
--------------------------------------------------------------------------------
Samples                                       300           300           300
Avg word count                             283.70        275.60        298.10

Hedging - total count                         596           539           722
Hedging - mean per sample                    1.99          1.80          2.41
Hedging - per 100 words                      0.59          0.53          0.71
Hedging - % samples with any                65.30         60.70         71.30

Self-correction - total                      1086          1042          1245
Self-correction - mean/sample                3.62          3.47          4.15
Self-correction - per 100 words              1.12          1.06          1.26
Self-correction - % with any                81.30         75.30         85.00

Assertive - total count                       297           301           318
Assertive - mean per sample                  0.99          1.00          1.06
Assertive - per 100 words                    0.38          0.37          0.37
Assertive - % samples with any              54.70         50.30         51.70

Confidence ratio (assert/hedge)              0.31          0.33          0.27
================================================================================

  Top markers in [original]:
     self_correction | \bwait\b                                 |  827
             hedging | \bmaybe\b                                |  507
     self_correction | \bhmm\b                                  |  178
           assertive | \btherefore\b                            |   94
           assertive | \bthe answer is\b                        |   69
           assertive | \bso the answer\b                        |   61
     self_correction | \bi made (a )?mistake\b                  |   61
             hedging | \bi think\b                              |   52
           assertive | \bjust\b                                 |   44
             hedging | \bmight\b                                |   15

  Top markers in [positive]:
     self_correction | \bwait\b                                 |  810
             hedging | \bmaybe\b                                |  453
     self_correction | \bhmm\b                                  |  158
           assertive | \bthe answer is\b                        |   92
           assertive | \btherefore\b                            |   73
           assertive | \bso the answer\b                        |   69
     self_correction | \bi made (a )?mistake\b                  |   51
             hedging | \bi think\b                              |   47
           assertive | \bjust\b                                 |   32
             hedging | \bmight\b                                |   18

  Top markers in [negative]:
     self_correction | \bwait\b                                 |  959
             hedging | \bmaybe\b                                |  634
     self_correction | \bhmm\b                                  |  194
           assertive | \bthe answer is\b                        |   95
           assertive | \btherefore\b                            |   77
           assertive | \bso the answer\b                        |   69
     self_correction | \bi made (a )?mistake\b                  |   68
             hedging | \bi think\b                              |   51
           assertive | \bjust\b                                 |   45
     self_correction | \bactually\b                             |   15

================================================================================
BREAKDOWN BY CORRECTNESS
================================================================================

--- CORRECT answers ---

================================================================================
LINGUISTIC MARKER ANALYSIS - CONFIDENCE COMPARISON
================================================================================
Metric                                   original      positive      negative
--------------------------------------------------------------------------------
Samples                                       125           133           103
Avg word count                             221.90        212.50        241.50

Hedging - total count                         105            99           121
Hedging - mean per sample                    0.84          0.74          1.17
Hedging - per 100 words                      0.28          0.24          0.38
Hedging - % samples with any                43.20         38.30         51.50

Self-correction - total                       194           190           212
Self-correction - mean/sample                1.55          1.43          2.06
Self-correction - per 100 words              0.55          0.47          0.70
Self-correction - % with any                60.00         51.90         67.00

Assertive - total count                       145           161           141
Assertive - mean per sample                  1.16          1.21          1.37
Assertive - per 100 words                    0.48          0.50          0.53
Assertive - % samples with any              59.20         54.10         53.40

Confidence ratio (assert/hedge)              0.52          0.55          0.49
================================================================================

--- INCORRECT answers ---

================================================================================
LINGUISTIC MARKER ANALYSIS - CONFIDENCE COMPARISON
================================================================================
Metric                                   original      positive      negative
--------------------------------------------------------------------------------
Samples                                       175           167           197
Avg word count                             327.90        325.90        327.70

Hedging - total count                         491           440           601
Hedging - mean per sample                    2.81          2.63          3.05
Hedging - per 100 words                      0.80          0.76          0.88
Hedging - % samples with any                81.10         78.40         81.70

Self-correction - total                       892           852          1033
Self-correction - mean/sample                5.10          5.10          5.24
Self-correction - per 100 words              1.52          1.53          1.55
Self-correction - % with any                96.60         94.00         94.40

Assertive - total count                       152           140           177
Assertive - mean per sample                  0.87          0.84          0.90
Assertive - per 100 words                    0.30          0.28          0.28
Assertive - % samples with any              51.40         47.30         50.80

Confidence ratio (assert/hedge)              0.16          0.16          0.16
================================================================================
```

## Conclusion

### 四组 Confidence Ratio 总览

| Task + Model | original | positive | negative | 趋势 |
| --- | --- | --- | --- | --- |
| **GSM8K + Llama3** | 1.08 | **1.37** | 0.91 | pos > orig > neg ✓ |
| **GSM8K + Qwen3** | 0.31 | **0.33** | 0.27 | pos > orig > neg ✓ |
| **TriviaQA + Llama3** | 0.04 | 0.03 | 0.05 | 几乎无差异 |
| **TriviaQA + Qwen3** | 0.60 | 0.55 | 0.56 | 微弱，无清晰趋势 |

### 关键发现

**GSM8K 上趋势一致且明确：positive > original > negative**

- **Llama3** 差异最显著：positive 的 "the answer is" 出现 200 次 vs negative 106 次，"therefore" 165 vs 121
- **Qwen3** 趋势相同但幅度较小，因为 Qwen3 本身就大量使用 "wait"（800-960次）、"maybe"（450-630次）等思考链词汇，self-correction 远高于 Llama3

**TriviaQA 上基本没有差异 (所以略过了trivialqa的具体结果)**

- 回答太短（avg ~45 words），marker 出现次数极低（个位数），无法区分
- 这符合预期——TriviaQA 是简短知识问答，不像数学题需要推理过程

**Qwen3 vs Llama3 的风格差异很大（重点）**

- Qwen3 天然更"犹豫"：self_correction 是 Llama3 的 100 倍+（1000+ vs 3-6），因为 Qwen3 使用思维链（"Wait", "Hmm", "Maybe"）
- 所以 Qwen3 的 confidence ratio 整体远低于 Llama3（0.27-0.33 vs 0.91-1.37）
- 因此对于llama3来说，在多部推理中，negative steering 降低confidence反而提升了performance；而qwen3则是通过positive steering提升confidence，来提升performance