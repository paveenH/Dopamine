# Steering Answer Organization in Language Models: Conditional Gains and Label-Free Dose Selection

*Draft — Sections 3 and 4. Outline: [TMLR1028.md](TMLR1028.md).*

---

## 3. Setup

### 3.1 Intervention

**Direction.** We reuse the Role-Sensitive Neuron (RSN) direction of prior work [cite parent paper]. For each model, the direction is extracted once on MMLU-E, a multiple-choice format with an added "I am not sure" option. The extraction keeps the items on which an expert and a non-expert role prompt lead to different answers. For each layer $l$ it averages the difference between the two prompts' final-token hidden states, giving $\Delta\mu^{(l)}$. It then keeps the top 0.5% of dimensions by $|\Delta\mu^{(l)}_i|$ and zeroes the rest. The result is a sparse, signed vector $m_l$ per layer. We reuse each model’s existing RSN direction without recomputing it and keep it fixed across tasks.

**Injection.** At inference we add $\alpha\, m_l$ to the output hidden state of each steered decoder layer $l$. The addition is made at a single position: the last prompt token, during prefill only. Decoding is unmodified, so every generated token is produced by the unsteered model, conditioned on a perturbed prompt state. Every run logs the number of injection events and checks it against layers × items. At $\alpha=0$ the same code path runs with an all-zero vector.

**Steered layers.** For each model the steered band is a contiguous block of middle layers. Following the prior work, it is placed where the layer-wise similarity between expert and non-expert hidden states begins to fall (Table 1).

**Dose scale.** $\alpha$ scales a model-specific steering vector. Masks, band widths, and activation scales differ across models, so equal values of $\alpha$ do not represent equal interventions. We therefore compare dose-response patterns across models, since the numerical values of $\alpha$ are not directly comparable.

### 3.2 Models and Coverage

We study four instruction-tuned models with 7–8B parameters. Table 1 lists the models and their steered layer ranges. GSM8K serves as the reference task for characterizing dose-response patterns, while GSM-Hard and MATH assess cross-task transfer. Additional tasks are introduced in §5.5.

**Table 1. Models and intervention settings.** Decoder layers are indexed from zero, and both endpoints of each steered range are included.

| Model | Total decoder layers | Steered decoder layers |
|---|---:|---|
| Llama-3.1-8B-Instruct | 32 | 10–18 |
| Qwen2.5-7B-Instruct | 28 | 15–20 |
| Qwen3-8B | 36 | 16–24 |
| Granite-3.3-8B-Instruct | 40 | 10–18 |

For GSM8K, GSM-Hard, and MATH, each experimental cell contains 300 items. All four models use the same 300 GSM8K test items in the same order. The standard generation budgets are 768 tokens for GSM8K and GSM-Hard and 2048 tokens for MATH; Qwen3 is additionally evaluated on GSM8K with a 2048-token budget. Detailed dose grids and task-specific settings are provided in Appendix A.

### 3.3 Interfaces

**Bare.** For reasoning tasks, we provide the task prompt as a plain string without a chat template. GSM8K prompts request a final numeric answer after `####`, with the CoT variant adding `Let's think step by step.` before the final-answer instruction. MATH prompts request the final answer in `\boxed{}`. We use Bare as a simple setting for examining how steering changes the model’s answer organization.

```
Solve the following math problem.
Question: {question}
Provide your final numeric answer after '####'.
Answer: 
```

**Native Chat** presents the task instruction using each model’s native chat template, allowing us to examine steering under standard conversational use (§5.3).

All runs use greedy decoding with a fixed token budget: 768 tokens for the GSM8K family and 2048 for MATH. All conclusions are conditional on these budgets.

### 3.4 Metrics

We separate how well the model answers from how it organizes its answer. 

**Performance.** Our primary metric is first-answer accuracy, which evaluates the first parseable final answer in each output. A fixed fallback procedure handles outputs without a parseable answer marker, and all items remain in the denominator. We also report last-answer accuracy to assess sensitivity to answer revisions. Scoring details are provided in Appendix C.

**Answer organization.** The two organization metrics describe the visible order of the generated text.
- Early-candidate rate (EC) measures the proportion of outputs that open with an answer-like numerical candidate before presenting a derivation. We compute this rate over all items using a fixed text-based detector, with detection criteria and validation results provided in Appendix C.
- **Reason-first rate** measures the proportion of outputs that contain explicit reasoning cues before the first answer-like numerical candidate, among outputs where such a candidate can be identified. Detection criteria are provided in Appendix C.

**Answer submission.** We report the valid-submission rate, the proportion of outputs containing a parseable final-answer marker, and conditional accuracy among those outputs. Conditional accuracy is descriptive because the submitted-item subset varies with $\alpha$.

**Generation quality.** We track repetitive looping, token-budget truncation, and natural termination. Operational definitions are provided in Appendix C.

### 3.5 Statistics

All comparisons are paired by item. We report the exact two-sided McNemar test on discordant items and a paired bootstrap 95% CI for the accuracy difference (B = 10,000, seed 0, percentile). Holm correction is applied within each family of comparisons, and the family is declared with each result. We distinguish three kinds of evaluation:

1. *Dose sweeps on GSM8K*, where the workpoint is chosen. Workpoint-versus-baseline contrasts on these items are in-sample.
2. *Fixed-workpoint transfer*, where the GSM8K workpoint is applied unchanged to another task. These are the main tests of whether a gain holds up.
3. *A prospective test*, where dose predictions were frozen before GSM-Hard gold answers were unsealed (§6.2).

---

## 4. Steering Reorganizes Answers

### 4.1 Dose Response

Figure 2 shows, for each model, how EC, reason-first and accuracy change with dose on GSM8K. In all four models, $\alpha$ moves the position of the first answer candidate. The sign of the effective direction depends on the model. For Llama and Granite, negative $\alpha$ lowers EC and raises reason-first. For Qwen2.5 and Qwen3, positive $\alpha$ does.

![Figure 2](Figures/fig2_dose_organization.png)

**Figure 2.** GSM8K dose response (Bare, No-CoT, 300 items per point, 768 tokens). Accuracy (`first_acc`) and early-candidate rate use all items. Reason-first uses the candidate-covered items. The shaded band marks each model's workpoint. The $\alpha$ axes are model-specific and are not comparable across panels.

The change is largest where the baseline opens with an answer most often (Table 2). Qwen2.5 opens with an answer on 96.33% of items at $\alpha=0$; at +8 this falls to 5.00%, and reason-first rises from 0% to 98.00%. Qwen3 moves from 64.00% to 9.00% EC at +10. Llama starts at 48.00% EC and falls to 18.67% at −6. Granite already reasons first on most items at baseline (EC 6.67%). Its workpoint lowers EC only slightly but raises reason-first from 58.11% to 95.92%.

At each model's workpoint, accuracy rises along with this reorganization: +18.00 pp for Llama, +18.00 pp for Qwen2.5 and +23.67 pp for Qwen3. Granite rises +4.67 pp, which is not significant. These are in-sample contrasts, because each workpoint was chosen on these items. Section 5 tests the same workpoints on other tasks.

**Table 2. Answer organization at each model's GSM8K workpoint** (Bare, No-CoT, n=300). Accuracy changes are paired against $\alpha=0$; "Fixed / broke" counts items that went from wrong to right / from right to wrong.

| Model | $\alpha$ | `first_acc` | EC | Reason first | Pre-candidate chars (median) | Valid submission | Fixed / broke | Exact McNemar $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0 | 60.00% | 48.00% | 30.85% | 0 | 62.67% | — | — |
| | **−6** | **78.00%** | **18.67%** | **66.55%** | 175 | 60.67% | 76 / 22 | 3.9×10⁻⁸ |
| Granite-3.3-8B | 0 | 84.00% | 6.67% | 58.11% | 125 | 88.00% | — | — |
| | **−6** | **88.67%** | **2.67%** | **95.92%** | 226 | 90.00% | 32 / 18 | .065 |
| Qwen2.5-7B | 0 | 68.00% | 96.33% | 0.00% | 0 | 81.00% | — | — |
| | **+8** | **86.00%** | **5.00%** | **98.00%** | 140 | 97.00% | 78 / 24 | 7.7×10⁻⁸ |
| Qwen3-8B | 0 | 49.33% | 64.00% | 17.23% | 0 | 42.00% | — | — |
| | **+10** | **73.00%** | **9.00%** | **81.85%** | 513 | 62.67% | 103 / 32 | 6.8×10⁻¹⁰ |

The Holm-adjusted $p$ within each model's GSM8K sweep is $<10^{-4}$ for Qwen2.5 (+8), $4.8\times10^{-9}$ for Qwen3 (+10, m = 7) and .195 for Granite (−6, m = 6). Under a 2048-token budget, Qwen3 reaches 58.33% → 86.67% at the same workpoint, with EC 64.00% → 8.67%.

Two features recur across models. First, the same visible quantity moves in all four, namely whether the generation opens with an answer-shaped value. What changes is which sign of $\alpha$ moves it. Second, valid submission does not track accuracy consistently. It falls slightly for Llama (62.67% → 60.67%) while accuracy rises 18 pp, and it rises for Qwen2.5 and Qwen3. Whether more items carry a formal marker is therefore a separate question from whether they are answered correctly. Section 5.1 decomposes it.

The dose curves are not symmetric around the workpoint. Moving $\alpha$ the other way raises EC: Llama rises to 75.67% at +6, Granite to 26.33% at +6, and Qwen3 to 86.00% at −8. Accuracy falls on that side for Llama, Granite and Qwen3. Granite's +6 decline (−8.67 pp, Holm $p$ = .0078) is its only significant contrast. In Granite, reason-first also rises at positive doses (82.35% at +10) even while EC rises, so on this model the two organization metrics do not always move together.

### 4.2 Non-Monotone Responses

Organization and accuracy decouple in two ways: when the dose overshoots, and when the reorganization saturates.

**Overshoot (Llama, −8).** Going one step past Llama's workpoint collapses accuracy to 40.33%, below the 60.00% baseline. The failure is not a return to the baseline pattern. At −8, 171 of 300 generations begin directly with a formal `#### N` submission. The median position of the first parseable marker is 0.000 of the generation (0.177 at baseline). Accuracy among items with a marker is 23.63%. The model submits at once and then keeps generating: 41 items switch between two or more different answer values, and the median length after the first candidate is 2,259 characters. EC is high here (76.67%), as on the positive side, but the visible pattern differs. Positive doses produce an early candidate followed by continued checking; −8 produces an early formal submission followed by unstable answer values.

**Saturation (Qwen2.5).** On GSM8K, accuracy reaches a plateau beyond the workpoint: 86.00%, 88.33% and 88.67% at +8, +10 and +12, with no detected difference between any two of them. EC is already at 2.67–5.00%. On MATH, the same model's accuracy peaks at +6 (68.33%, +7.67 pp over baseline, Holm $p$ = .0087) and falls at +8 (63.33%; exploratory $p$ = .040 against +6), while EC keeps falling (19.33% → 10.33%) and reason-first keeps rising (77.59% → 86.16%). Further delaying the first candidate does not keep improving accuracy. The MATH decline is concentrated in Level-5 problems (47.2% → 36.0%).

Both cases show that the organization metrics describe an output pattern, not a quantity to be pushed to its extreme. Low EC accompanies the best-performing doses in every model, but it is not sufficient for them.

### 4.3 Specificity

Is the gain specific to the RSN direction, or would any sparse perturbation of the same strength do? We injected norm-matched control directions at each model's workpoint and regenerated all 300 GSM8K items (Figure 3). We tested two kinds of control:
- *On-support* controls use exactly the RSN's non-zero coordinates, matched per layer in norm, with Gaussian weights orthogonal to the dense role-difference vector.
- *Off-support* controls use the same number of coordinates per layer, disjoint from the RSN's, also matched in norm.

Each kind has three seeds, which are three different directions. The baseline and RSN cells were regenerated in the same round as the controls. Llama's reproduce the dose-sweep cells exactly; Qwen2.5's read 69.33% and 85.00% (68.00% and 86.00% in the sweep), and the comparisons below use the matched round.

![Figure 3](Figures/fig3_orthogonal.png)

**Figure 3.** RSN versus norm-matched control directions on GSM8K at each model's workpoint (300 items per point). On-support controls share the RSN's coordinates; off-support controls use disjoint coordinates. Each control point is one direction.

**On support, RSN beats all three controls in both models.** All six direct comparisons survive Holm correction (m = 6). Llama: RSN 78.00% against 71.00%, 66.00% and 56.67% (RSN − control +7.00 to +21.33 pp, Holm $p$ ≤ .0154). Qwen2.5: RSN 85.00% against 69.67%, 68.33% and 69.67% (+15.33 to +16.67 pp, Holm $p$ ≤ 1.5×10⁻⁵). Holding the coordinates and per-layer norm fixed, the weights of the direction matter.

**Off support, the two models differ.** For Qwen2.5, no control reproduces either the gain or the reorganization. All six control directions stay near baseline accuracy (68.33–72.33%) and baseline EC (88.33–97.67%), and RSN exceeds each off-support control (+12.67 to +16.67 pp, Holm $p$ ≤ 1.3×10⁻⁴, a separate m = 6 family). For Llama, some random directions also help. Off-support seed 1 reaches 75.33% with EC 23.67%, and RSN does not detectably exceed it (+2.67 pp, $p$ = .42). Averaged over all six control directions, Llama's controls sit 5.78 pp above baseline (95% CI [+1.61, +9.94]), and RSN sits 12.22 pp above that average (CI [+7.72, +16.61]).

The supported claim is therefore that RSN outperforms the six norm-matched directions tested. It is not that only the RSN direction can produce a gain. In Llama, the control directions that raise accuracy also lower EC (Figure 3, left), so accuracy and organization move together across directions as well as across doses.

### 4.4 Transfer to Base Checkpoints

The RSN direction is extracted from instruction-tuned models. We applied each model's instruction-tuned mask, unchanged, to its pretrained base checkpoint on the same 300 GSM8K items (Bare, zero-shot, 768 tokens).

On Qwen2.5-7B Base, the effect is two-sided. Relative to $\alpha=0$ (62.33%, EC 37.33%):
- At +8, accuracy rises to 80.00% (+17.67 pp; 67 fixed / 14 broke; $p$ = 1.9×10⁻⁹) and EC falls to 7.67%.
- At −8, accuracy falls to 47.33% (−15.00 pp; $p$ = 7.4×10⁻⁷) and EC rises to 50.33%.

Marker coverage stays at 293–295 of 300 in all three cells, with at most four truncated generations. The organization effect and the accuracy gain both survive the move from the instruction-tuned to the base checkpoint, and the direction of each matches the instruction-tuned model.

Llama-3.1-8B Base does not give an interpretable test under this interface. Only 1–6 of 300 generations carry a parseable marker, and about 27–32% exhaust the budget. Opening answers often conflict with later conclusions. A one-shot follow-up raised marker coverage to 94–97% but left every generation at the budget limit, with no detected accuracy change (Appendix K). We treat Llama Base as an interface limitation, not evidence for or against transfer.

### 4.5 Entry State and Generated Behavior

Because the intervention is a single additive change at one position, its immediate effect on the prompt state is simple. In a separate collection on the same 300 items (batch size 1), we projected the last-token prefill state of the steered layers onto the RSN direction. This projection is linear in $\alpha$: $R^2$ = .999 across nine doses for Llama and .9998 across eleven for Qwen2.5. The behavior that follows is not linear. Llama's accuracy peaks at −6 and collapses at −8, while the entry projection keeps moving at a constant rate. Across Llama's nine doses, the entry projection describes the accuracy curve poorly ($R^2$ = .136), whereas EC describes it closely ($R^2$ = .945). Both fits are curve-level descriptions over nine points (Appendix H).

The size of the prompt-state perturbation therefore does not stand in for its behavioral effect. What the effect looks like depends on how the model's generation unfolds from that state, which the organization metrics capture. Section 6.1 returns to this point: entry gain alone predicts item-level correctness weakly, and adds nothing once output features are included.

---

## Notes for authors (remove before submission)

1. **§4.2 Llama −8 用了 182 批次的数字。** 提纲原写"首个 marker 位置 0.298→0.043、约 96% 文本在提交之后"，那组数来自 signal 批次（bs=1，取均值），而 40.33% 准确率来自 182 批次（bs=24）。为了同一段内批次一致，草稿改用 182 批次：marker 开头 171/300、首个可解析 marker 位置中位数 0.000（基线 0.177）、`cond_acc` 23.63%、≥2 次换答 41 题。
2. **Granite 写法。** Granite 的工作点在 GSM8K 上不显著（+4.67 pp，Holm p=.195），EC 基线只有 6.67%（地板），主要变化体现在 reason-first（58.11→95.92%）。正向一侧 EC 和 reason-first 同时上升，两项组织指标不同步，已在 §4.1 如实写出。C1 里"Granite 为负向"目前主要靠 reason-first 和正向侧的 EC 上升支撑。
3. **新算的数。** Qwen3、Granite 的 reason-first 和候选前字符数是这次用冻结实现新算的（`scripts/compute_qwen3_granite_org.py` → `data/gsm8k_org_qwen3_granite.csv`）；`first_acc` 和 EC 与 ReasoningBare 表 9.1、9.2、9.9 完全一致。提纲"Before Drafting"第 1 项已完成。
4. **Table 1 待核对。** Llama/Qwen2.5 的 Native Chat 和 Chat-CoT 预算、GSM-Symbolic 的 token 预算、ProofWriter/FinQA/ZebraLogic 的 n，写 §5 时再对一遍。Qwen3 和 Granite 的总层数（36、40）按公开配置填写，未从本地文件核对。
5. **Qwen2.5 正交对照的批次。** 正交那轮的基线和 RSN（69.33/85.00）与剂量扫描（68.00/86.00）不同，§4.3 已注明比较用的是同一轮的配对数据。
6. **§4.5 的 R² 来自 signal 批次**，与 182 批次的准确率曲线是两次采集；正文写成同一批 300 题上的另一次采集。如果想更保守，可以只留线性 R²，把 .136/.945 移到附录 H。
