# Steering Answer Organization in Language Models: Conditional Gains and Label-Free Dose Selection

## 3. Setup

### 3.1 Intervention

**RSN direction.** We use the model-specific Role-Sensitive Neuron (RSN) directions identified in prior work **[cite parent paper]**. These directions were extracted on MMLU-E, a variant of MMLU with an added “I am not sure” option, using items for which expert and non-expert role prompts produced different answers. For each layer $l$, the mean difference between the two prompts’ final-token hidden states defines $\Delta\mu^{(l)}$. Retaining the top 0.5% of dimensions by $|\Delta\mu^{(l)}_i|$ and setting the remainder to zero yields a sparse, signed vector $m_l$. We reuse these directions without recomputing them and keep them fixed across tasks.

**Intervention.** During prefill, we add $\alpha m_l$ to the output hidden state at the final prompt position in each steered decoder layer:

$$
h^{(l)}_{\mathrm{last}} \leftarrow h^{(l)}_{\mathrm{last}}+\alpha m_l.
$$

No further intervention is applied during decoding. Each model uses a contiguous band of middle layers, following the prior work’s layer-wise analysis of expert–non-expert activation similarity (Table 1).

**Dose scale.** The multiplier $\alpha$ controls intervention strength within each model. Because the sparse vectors, layer bands, and activation scales differ across models, equal values of $\alpha$ do not represent equal interventions. Cross-model comparisons therefore focus on dose-response patterns.

### 3.2 Models and Coverage

We study four instruction-tuned models with 7–8B parameters. Table 1 lists their layer counts and steered ranges. GSM8K serves as the reference task for characterizing dose-response patterns, while GSM-Hard and MATH assess cross-task transfer. Additional tasks are introduced in §5.5.

**Table 1. Models and intervention settings.** Decoder-layer indices are zero-based, and both endpoints of each steered range are included.

| Model | Total decoder layers | Steered decoder layers |
|---|---:|---|
| Llama-3.1-8B-Instruct | 32 | 10–18 |
| Qwen2.5-7B-Instruct | 28 | 15–20 |
| Qwen3-8B | 36 | 16–24 |
| Granite-3.3-8B-Instruct | 40 | 10–18 |

Each condition on GSM8K, GSM-Hard, and MATH contains 300 items. All four models use the same GSM8K test items in the same order. Reasoning experiments use greedy decoding, with standard generation budgets of 768 tokens for GSM8K and GSM-Hard and 2048 tokens for MATH. Qwen3 is additionally evaluated on GSM8K with a 2048-token budget. Detailed dose grids and task-specific settings are provided in Appendix A.

### 3.3 Interfaces

**Bare.** For reasoning tasks, we provide the task prompt as a plain string without a chat template. This provides a simple setting for examining how steering changes answer organization. GSM8K prompts request a final numeric answer after `####`; the No-CoT prompt is:

```text
Solve the following math problem.
Question: {question}
Provide your final numeric answer after '####'.
Answer: 
```

The CoT variant adds `Let's think step by step.` before the final-answer instruction. MATH prompts request the final answer in `\boxed{}`. Full prompts are provided in Appendix B.

**Native Chat.** We present the task instruction using each model’s native chat template to examine steering under standard conversational use (§5.3).

### 3.4 Metrics

We evaluate task performance, answer organization, submission behavior, and generation quality.

**Performance.** Our primary metric is first-answer accuracy, which scores the first parseable final answer in each output. A fixed fallback procedure handles outputs without a parseable answer marker, retaining all items in the denominator. We also report last-answer accuracy to assess sensitivity to answer revisions.

**Answer organization.** We use two complementary measures of visible output order. *Early-candidate rate* (EC) measures the proportion of outputs that begin with an answer-like numerical candidate, using all items as the denominator. *Reason-first rate* measures the proportion containing explicit reasoning cues before the first answer-like numerical candidate, among outputs where such a candidate can be identified. Candidate coverage documents the fraction of outputs included in this latter measure.

**Answer submission.** Valid-submission rate measures the proportion of outputs containing a parseable final-answer marker. Conditional accuracy measures correctness among those outputs and is reported descriptively, since the submitted-item subset varies with $\alpha$.

**Generation quality.** We track repetitive looping, token-budget truncation, and natural termination to characterize the generated outputs under the specified budgets.

Scoring procedures, detector definitions, and validation results are provided in Appendix C.

### 3.5 Statistical Analysis

Accuracy comparisons are paired by item. We use two-sided exact McNemar tests and report paired bootstrap 95% confidence intervals for accuracy differences. Holm correction is applied within the comparison families specified for each experiment. Dose-selection procedures are described alongside the corresponding evaluations, distinguishing fixed-workpoint transfer, target-task dose sweeps, and prospective prediction. Bootstrap settings and complete comparison families are provided in Appendix D.

---

## 4. Steering Reorganizes Answers

### 4.1 Dose Response

Figure 2 shows how steering changes answer organization on GSM8K. Across all four models, the dose-response curves reveal shifts in whether an answer candidate appears at the beginning of the output or follows explicit reasoning. The direction associated with lower EC and higher reason-first differs across models: negative doses produce this shift in Llama and Granite, whereas positive doses do so in Qwen2.5 and Qwen3.

![Figure 2](Figures/fig2_dose_organization.png)

**Figure 2.** Dose-response curves for answer organization and accuracy on GSM8K (Bare, No-CoT; 300 items per dose, 768-token budget). Shaded bands indicate the reference workpoints.

Table 2 summarizes the changes from baseline to each model’s reference workpoint. Qwen2.5 exhibits the largest reduction in early candidates: EC falls from 96.33% to 5.00% at $\alpha=+8$, while reason-first rises from 0.00% to 98.00%. Qwen3 shows a similar shift at $+10$, with EC falling from 64.00% to 9.00% and reason-first rising from 17.23% to 81.85%. For Llama, $\alpha=-6$ reduces EC from 48.00% to 18.67% and increases reason-first from 30.85% to 66.55%. Granite begins with few early candidates; its clearest change is the increase in reason-first from 58.11% to 95.92% at $-6$.

**Table 2. Answer organization and accuracy at the reference workpoints.** Values show baseline → workpoint performance, in percent.

| Model | Reference $\alpha$ | EC | Reason-first | Accuracy |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | −6 | 48.00 → 18.67 | 30.85 → 66.55 | 60.00 → 78.00 |
| Granite-3.3-8B | −6 | 6.67 → 2.67 | 58.11 → 95.92 | 84.00 → 88.67 |
| Qwen2.5-7B | +8 | 96.33 → 5.00 | 0.00 → 98.00 | 68.00 → 86.00 |
| Qwen3-8B | +10 | 64.00 → 9.00 | 17.23 → 81.85 | 49.33 → 73.00 |

Accuracy increases alongside these organizational changes. The gains are significant after Holm correction for Llama (+18.00 percentage points), Qwen2.5 (+18.00), and Qwen3 (+23.67), with a smaller positive trend for Granite (+4.67). Section 5 examines these performance changes and their transfer across tasks.

Steering in the opposite direction can instead favor early candidates. EC rises to 75.67% for Llama at $+6$, 26.33% for Granite at $+6$, and 86.00% for Qwen3 at $-8$. Together, these results show that steering systematically changes visible answer organization, with model-specific dose-response patterns. The following section examines how these responses saturate or reverse at stronger doses.

### 4.2 Saturation and Oversteering

The dose-response curves reveal distinct regimes as intervention strength increases: a reversal in answer organization for Llama and saturation of performance gains for Qwen2.5.

**Oversteering in Llama.** Increasing the magnitude of the negative dose from $\alpha=-6$ to $-8$ reduces accuracy from 78.00% to 40.33%, below the 60.00% baseline. This change accompanies a marked shift toward immediate answer submission: 171 of 300 outputs begin directly with a `####` marker. Some then continue generating and repeatedly alternate between answer values. The example below illustrates this pattern: the model first submits 55, subsequently calculates the correct answer of 40, and continues switching between the two values.

**Saturation in Qwen2.5.** On GSM8K, accuracy levels off at high doses, reaching 86.00% at $+8$ and 88.33% at $+10$, with no detected difference between them. On MATH, accuracy reaches 68.33% at $+6$ and falls to 63.33% at $+8$, even as EC decreases from 19.33% to 10.33% and reason-first increases from 77.59% to 86.16%. Thus, the organization measures continue to change beyond the dose yielding the highest observed accuracy.

Together, these patterns characterize the operating range of steering: increasing intervention strength can produce further reorganization, a performance plateau, or a reversal of the gains observed at moderate doses. For Llama and Qwen2.5, the measured RSN projection at the prompt boundary remains approximately linear in dose, including the ranges where accuracy declines or plateaus (Appendix H).

**Example. Immediate submission followed by answer switching (Llama, $\alpha=-8$).**  
Brian has 20 video games and loses 5. Bobby has 5 fewer than three times Brian’s remaining number. The correct answer is 40. Selected excerpts from the generated response are shown below; ellipses indicate omitted text.

```text
#### 55
Explanation: Brian has 20 video games, but lost 5,
so he has 20 - 5 = 15 video games.
...
Therefore, Bobby has 40 video games.
The answer is 55. I made a mistake.
The answer is 40. I made another mistake.
The answer is 55. I made another mistake.
The answer is 40.
...
```


### 4.3 Specificity

We assess whether the observed gains depend on the RSN direction by comparing it with norm-matched sparse control directions on GSM8K (Figure 3). For Llama and Qwen2.5, we evaluate three independently sampled directions of each type at the model’s reference workpoint. *On-support* controls retain the RSN’s non-zero coordinates but replace their weights with random weights orthogonal to the dense role-difference vector. *Off-support* controls use an equal number of coordinates disjoint from the RSN support. Both types match the RSN vector’s norm in each steered layer. Control construction and experimental settings are detailed in Appendix A; complete paired comparisons and correction families are reported in Appendix D.

![Figure 3](Figures/fig3_orthogonal.png)

**Figure 3.** Accuracy and early-candidate rate under RSN and norm-matched control directions on GSM8K (Bare, No-CoT; 300 items per condition). On-support controls share the RSN’s coordinates; off-support controls use disjoint coordinates. Each control point represents a distinct sampled direction.

**RSN outperforms all on-support controls in both models.** Llama reaches 78.00% accuracy, compared with 56.67–71.00% for the controls; Qwen2.5 reaches 85.00%, compared with 68.33–69.67%. All six paired comparisons remain significant after Holm correction ($m=6$), with adjusted $p\leq .0154$ for Llama and $p\leq 1.5\times10^{-5}$ for Qwen2.5. Thus, the specific weight direction contributes to performance even when the steered coordinates and per-layer intervention norm are held fixed.

**Off-support controls provide a complementary comparison.** For Qwen2.5, all six sampled controls cluster near baseline accuracy (68.33–72.33%) and retain high EC (88.33–97.67%), whereas RSN combines 85.00% accuracy with 5.00% EC. RSN also significantly exceeds each off-support control (adjusted $p\leq 1.3\times10^{-4}$ within a separate six-comparison Holm family). Llama shows greater variation across control directions. Across its six controls, mean accuracy is 65.78%, compared with 78.00% for RSN.

Together, these results support a contribution from the RSN weight direction beyond sparse support and intervention magnitude. RSN achieves the highest observed accuracy in both models, with significant advantages over every tested on-support direction. Higher-performing control directions also tend to have lower EC, extending the association between answer organization and accuracy across intervention directions.

### 4.4 Transfer to Base Models

To examine whether the organizational effects extend to base models, we apply each instruction-tuned model’s RSN direction unchanged to its corresponding base model. We retain the same sparse support, layer band, and reference workpoint, evaluating the same 300 GSM8K items under Bare, zero-shot prompting with a 768-token budget.

**Table 3. Transfer of instruction-tuned RSN directions to base models.** Values show baseline → reference workpoint, in percent. Accuracy differences are in percentage points. Asterisks indicate unadjusted two-sided exact McNemar tests: \* $p<.05$, \*\* $p<.01$, \*\*\* $p<.001$. Exact values and complete results are provided in Appendix K.

| Base model | $\alpha$ | Accuracy | Δ (pp) | EC | Valid marker |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | +8 | 62.33 → 80.00 | $+17.67^{***}$ | 37.33 → 7.67 | 98.3 → 97.7 |
| Qwen3-8B | +10 | 69.00 → 73.67 | +4.67 | 35.00 → 17.33 | 96.0 → 99.0 |
| Granite-3.3-8B | −6 | 39.00 → 49.33 | $+10.33^{**}$ | 41.00 → 28.67 | 57.0 → 76.0 |
| Llama-3.1-8B | −6 | 26.33 → 30.00† | +3.67† | 58.00 → 49.33† | 0.7 → 0.3 |

† Llama results are diagnostic under this interface: only two baseline outputs and one steered output contain a parseable answer marker. Accuracy includes the fixed fallback procedure; interpretation and output diagnostics are detailed in Appendix K.

Qwen2.5, Qwen3, and Granite all show lower EC at their instruction-tuned reference workpoints, matching the direction of change in the corresponding instruction-tuned models. Accuracy increases alongside these shifts: Qwen2.5 gains 17.67 percentage points and Granite gains 10.33, both statistically significant, while Qwen3 shows a positive trend of 4.67 percentage points. Granite’s gain also accompanies increased marker coverage; the paired scoring-path analysis is provided in Appendix K. An exploratory reverse-direction condition in Qwen2.5 provides a complementary result: at $\alpha=-8$, accuracy falls from 62.33% to 47.33%, while EC rises from 37.33% to 50.33%.

These transfers reveal a controllable aspect of answer organization already present in pretrained base models. An RSN direction identified in an instruction-tuned model can access this behavior in its base counterpart, shifting outputs toward computation before answer submission without re-extraction or dose retuning.

### 4.5 Output Wording

To characterize how answer organization appears in the text, we analyze the evaluative, procedural, and checking expressions listed in Table N. We selected a shared phrase set for all four models from keyword contrasts between baseline and workpoint outputs. Evaluative share is defined as $E/(E+P)$, with sentence counts pooled over 300 GSM8K outputs after removing prompt and new-question continuation fragments and deduplicating sentences within each output. Table 4 reports within-model comparisons.

**Table N. Output wording categories, definitions, and example expressions.** Checking expressions are counted separately from evaluative and procedural expressions.

| Category | Description | Example expressions |
|---|---|---|
| Evaluative ($E$) | Judging, disputing, or reconsidering an answer already in the output | the provided answer; the correct answer; incorrect; wait; however |
| Procedural ($P$) | Advancing the computation | first; next; step; calculate; determine |
| Checking | Checking a calculation or result; counted separately from $E$ and $P$ | double-check; make sure; verify |

**Table 4. Evaluative share on GSM8K.** Values are percentages. Workpoint and reverse doses are listed for each model. Bold values identify the reference workpoint. Full phrase sets, wording densities, and sensitivity analyses are provided in Appendix L.

| Model | Workpoint $\alpha$ | Reverse $\alpha$ | Baseline | Workpoint | Reverse |
|---|---:|---:|---:|---:|---:|
| Llama | −6 | +6 | 29.63 | **3.91** | 38.71 |
| Qwen2.5 | +8 | −8 | 32.73 | **9.17** | 37.06 |
| Qwen3 | +10 | −8 | 60.04 | **41.78** | 68.92 |
| Granite | −6 | +6 | 6.03 | **0.89** | 16.97 |


The evaluative share decreases at the reference workpoint in all four models and rises above baseline under the reverse dose. For Llama, it falls from 29.63% to 3.91%; for Qwen2.5, from 32.73% to 9.17%. Qwen3 and Granite show the same pattern, indicating a consistent reduction in the relative prominence of evaluative wording at the workpoint. This shift can coexist with continued checking: Example 2 illustrates how the model checks and confirms a computed result after submission.

The dose curves connect these wording changes to the organizational patterns in §§4.1–4.2. Llama's evaluative share increases monotonically from $-6$ to $+6$, but returns to 38.61% at $-8$, alongside the immediate submissions and answer switching described in §4.2. Qwen2.5 remains near 33–37% from $-8$ to $+4$, then declines sharply from $+6$, paralleling its threshold-like EC response. These patterns extend to doses not used for phrase selection. All four models also show lower workpoint shares on GSM-Hard, and the GSM8K direction is preserved when “however” or “wait” is omitted, or when only explicit answer-evaluation terms are retained (Appendix L).

**Example 1. Evaluating the opening candidate (Qwen2.5).**  
Steve completes a puzzle in 10 hours; Anna takes 2 hours more than half of Steve's time. The correct answer is 7. Selected excerpts are shown below; ellipses indicate omitted text.

*Baseline:*

```text
6
#### 6

To find out how long it took Anna to finish the puzzle, we first need to
determine half of Steve's time and then add 2 hours to it.
...
Therefore, it took Anna 7 hours to finish the puzzle. The correct answer is 7,
but the provided answer is 6, which seems incorrect based on the calculation.
...
So, the final answer is 6. #### 6

The correct answer is actually 7. #### 7
...
```

*Workpoint ($\alpha=+8$):*

```text
the time for Anna to complete the puzzle is (1/2)*10 + 2 = 7 hours.

#### 7

Therefore, the answer is 7.
```

The baseline explicitly calls its own opening candidate “the provided answer” and disputes it after calculating 7; first-answer scoring still takes the initial `#### 6`. At the workpoint, the computation precedes the correct submission.

**Example 2. From resolving a conflict to confirming a result (Qwen3).**  
Robert compares two furniture offers: $1,350 plus six payments of $350, or $1,100 plus nine payments of $250. The correct cost difference is 100.

*Baseline:*

```text
1200

Okay, let's see. Robert wants to buy a bookshelf and a sofa, and he's
comparing two offers from different stores.
...
So Furniture United's total cost is $3,450.
...
so total is 3,350. Wait, that's the same as Furniture United? But the answer
is supposed to be 1200. Wait, maybe I did something wrong here.

Wait, hold on. Let me check again. Maybe I miscalculated.
...
```

*Workpoint ($\alpha=+10$):*

```text
, let's calculate the total cost for each offer.
...
Cost difference = $3,450 - $3,350 = $100

#### 100
Okay, let me check the calculations again to make sure I didn't make any
mistakes.
...
Hmm, that seems correct. I don't see any errors in the arithmetic. So the
answer should be $100.
#### 100
...
```

The baseline treats its opening value of 1200 as an expected answer and repeatedly revisits the calculation. At the workpoint, the model computes the difference, submits 100, and then checks the result with confirming language. Thus, checking remains visible even as the output shifts away from disputing an opening candidate.

These wording patterns describe how the organizational changes appear in the text: baseline evaluations often concern a candidate written at the start, whereas workpoint outputs contain fewer such evaluations. Section 5.1 examines the corresponding scoring paths and the calculations present in baseline outputs.

## 5. Performance Gains and Their Conditions

### 5.1 Where the Gains Appear

Building on the changes in answer organization and accuracy observed in §4, we examine where the additional correct answers arise. We analyze Llama and Qwen2.5 on GSM8K and GSM-Hard at their reference workpoints, together with Qwen3 on GSM8K. Each item is assigned to one of four paired scoring paths according to whether its answer is extracted from a parseable `####` marker (M) or through the fallback procedure (F) at baseline and at the workpoint. The fallback procedure uses fixed text-matching rules to extract an answer when no parseable marker is present; F therefore denotes an extraction route, not an unanswered item.

![Figure 4](Figures/fig4_score_path.png)

**Figure 4.** Net change in correct answers by paired scoring path (baseline → workpoint; 300 items per comparison). M denotes marker-based extraction and F fallback extraction. Segments show corrections minus newly incorrect answers within each path; dots mark the total net change. Qwen3 is shown at 768 and 2048 tokens; all other comparisons use 768 tokens. Complete path counts are provided in Appendix E.

**The gains follow different scoring paths across models.** For Qwen2.5, most of the net gain occurs among items with a parseable marker in both conditions: M→M contributes 49 of the 54 additional correct answers on GSM8K and 33 of 49 on GSM-Hard. Llama gains across all four paths, including items scored through fallback in both conditions. Its GSM8K marker coverage decreases slightly, from 62.67% to 60.67%, while accuracy rises by 18.00 percentage points. Qwen3 use Qwen3 shows a complementary pattern: F→M contributes a net gain of 86 correct answers, with the remaining paths contributing −1 in total. Here, steering primarily enables correct formal submissions from outputs previously scored through fallback. Together, these patterns distinguish gains in the correctness of submitted answers from gains accompanying changes in submission format.

**Table 5. Accuracy gains within the both-marker subset and under first- and last-answer scoring.** Both-marker accuracy shows baseline → workpoint values in percent; all differences are in percentage points. The subset contains items with a parseable marker in both conditions and is reported descriptively. Asterisks on its accuracy change indicate exploratory, unadjusted two-sided exact McNemar tests: \* $p<.05$, \*\* $p<.01$, \*\*\* $p<.001$. The final two columns use all 300 items. Budgets match Figure 4; full statistics are provided in Appendix E.

| Configuration | Both-marker $n$ | Both-marker accuracy | Both-marker Δ | All-item Δ (first) | All-item Δ (last) |
|---|---:|---:|---:|---:|---:|
| Llama GSM8K ($-6$) | 128 | 67.19 → 82.03 | $+14.84^{**}$ | +18.00 | +19.33 |
| Llama GSM-Hard ($-6$) | 94 | 17.02 → 20.21 | +3.19 | +6.33 | +6.33 |
| Qwen2.5 GSM8K ($+8$) | 238 | 64.71 → 85.29 | $+20.59^{***}$ | +18.00 | +7.00 |
| Qwen2.5 GSM-Hard ($+8$) | 214 | 35.98 → 51.40 | $+15.42^{***}$ | +16.33 | +13.67 |
| Qwen3 GSM8K ($+10$, 768) | 88 | 94.32 → 96.59 | +2.27 | +23.67 | +22.33 |
| Qwen3 GSM8K ($+10$, 2048) | 125 | 94.40 → 97.60 | +3.20 | +28.33 | +26.67 |

**Accuracy also improves when both outputs contain a formal answer.** Within the both-marker subsets, Llama GSM8K and both Qwen2.5 comparisons show significant gains under the exploratory tests in Table 5. Llama GSM-Hard and Qwen3 GSM8K show smaller positive trends. The same analysis on Qwen3 MATH gives 29.02% → 40.39% accuracy across 255 items (Appendix E). These subsets are selected using outputs from both conditions, so the comparisons describe where gains appear rather than isolate a causal effect of steering at fixed submission status.

**Baseline text often contains a correct target calculation despite an incorrect scored answer.** A contextual audit of corrected items confirmed such calculations in at least 43 of 76 Llama GSM8K corrections, 14 of 32 Llama GSM-Hard corrections, 47 of 78 Qwen2.5 GSM8K corrections, and 25 of 63 Qwen2.5 GSM-Hard corrections. For Qwen3, a separate audit identified a reference-consistent arithmetic result in 85 of the 86 F→M corrections. These are confirmed lower bounds from text review; audit procedures and item-level evidence are provided in Appendix E.

The visible sequences differ across models. In the confirmed Llama cases, an incorrect opening candidate precedes a correct target calculation, while the scored answer remains incorrect. Qwen2.5 additionally produces later correct markers after an incorrect first marker in 20 corrected GSM8K items and nine corrected GSM-Hard items. For example, on a GSM8K hiking-speed problem whose reference answer is 6 miles per hour, the Qwen2.5 baseline opens with `3` and `#### 3`, then states:

> Therefore, she needs to walk at a speed of 6 miles / 1 hour = 6 miles per hour for the remaining distance.

It subsequently returns to 3 and submits `#### 3`. At $+8$, the output instead works through the remaining time and distance before submitting `#### 6`. This example illustrates a correct calculation appearing in the baseline text without determining its submitted answer.

First- and last-answer scoring provide a complementary view (Table 5). Llama's GSM8K gains are similar under the two readouts (+18.00 and +19.33 percentage points). Qwen2.5 retains a positive GSM8K gain under last-answer scoring (+7.00), while its larger first-answer gain (+18.00) is consistent with the later corrections already present in some baseline outputs. Thus, steering can improve both which answer is submitted and how a calculation is carried through to submission. Lower EC frequently accompanies these gains, while the dose responses in §4.2 show that further organizational change need not yield a further accuracy increase.
