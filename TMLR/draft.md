# Steering Answer Organization in Language Models: Conditional Gains and Label-Free Dose Selection

## 3. Setup

### 3.1 Intervention

**RSN direction.** We use the model-specific Role-Sensitive Neuron (RSN) directions identified in prior work **[cite parent paper]**. These directions were extracted on MMLU-E, a multiple-choice benchmark with an additional “I am not sure” option, using items for which expert and non-expert role prompts produced different answers. For each layer $l$, the mean difference between the two prompts’ final-token hidden states defines $\Delta\mu^{(l)}$. Retaining the top 0.5% of dimensions by $|\Delta\mu^{(l)}_i|$ and setting the remainder to zero yields a sparse, signed vector $m_l$. We reuse these directions without recomputing them and keep them fixed across tasks.

**Intervention.** During prefill, we add $\alpha m_l$ to the output hidden state at the final prompt position in each steered decoder layer:

$$
h^{(l)}_{\mathrm{last}} \leftarrow h^{(l)}_{\mathrm{last}}+\alpha m_l.
$$

No further intervention is applied during decoding. Each model uses a contiguous band of middle layers, following the prior work’s layer-wise analysis of expert–non-expert activation similarity (Table 1).

**Dose scale.** The multiplier $\alpha$ controls intervention strength within each model. Because the sparse vectors, layer bands, and activation scales differ across models, equal values of $\alpha$ do not represent equal interventions. Cross-model comparisons therefore focus on dose-response patterns.

### 3.2 Models and Coverage

We study four instruction-tuned models with 7–8B parameters. Table 1 lists their architectures and steered layer ranges. GSM8K serves as the reference task for characterizing dose-response patterns, while GSM-Hard and MATH assess cross-task transfer. Additional tasks are introduced in §5.5.

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

**Table 2. Answer organization and accuracy at the reference workpoints.** Values show baseline ($\alpha=0$) → workpoint performance, in percent.

| Model | Reference $\alpha$ | EC | Reason-first | Accuracy |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | −6 | 48.00 → 18.67 | 30.85 → 66.55 | 60.00 → 78.00 |
| Granite-3.3-8B | −6 | 6.67 → 2.67 | 58.11 → 95.92 | 84.00 → 88.67 |
| Qwen2.5-7B | +8 | 96.33 → 5.00 | 0.00 → 98.00 | 68.00 → 86.00 |
| Qwen3-8B | +10 | 64.00 → 9.00 | 17.23 → 81.85 | 49.33 → 73.00 |

Accuracy increases alongside these organizational changes. The gains are significant after Holm correction for Llama (+18.00 percentage points), Qwen2.5 (+18.00), and Qwen3 (+23.67), with a smaller positive trend for Granite (+4.67). Section 5 examines these performance changes and their transfer across tasks.

Steering in the opposite direction can instead favor early candidates. EC rises to 75.67% for Llama at $+6$, 26.33% for Granite at $+6$, and 86.00% for Qwen3 at $-8$. Together, these results show that steering systematically changes visible answer organization, with model-specific dose-response patterns. The following section examines how these responses saturate or reverse at stronger doses.

### 4.2 Non-Monotone Responses

Organization and accuracy decouple in two ways: when the dose overshoots, and when the reorganization saturates.

**Overshoot (Llama, −8).** Going one step past Llama's workpoint collapses accuracy to 40.33%, below the 60.00% baseline. The failure is not a return to the baseline pattern. At −8, 171 of 300 generations begin directly with a formal `#### N` submission. The median position of the first parseable marker is 0.000 of the generation (0.177 at baseline). Accuracy among items with a marker is 23.63%. The model submits at once and then keeps generating: 41 items switch between two or more different answer values, and the median length after the first candidate is 2,259 characters. EC is high here (76.67%), as on the positive side, but the visible pattern differs. Positive doses produce an early candidate followed by continued checking; −8 produces an early formal submission followed by unstable answer values.

**Saturation (Qwen2.5).** On GSM8K, accuracy levels off beyond the workpoint: 86.00% at +8 and 88.33% at +10, with no detected difference between them. EC is already at 2.67–5.00%. On MATH, the same model's accuracy peaks at +6 (68.33%, +7.67 pp over baseline, Holm $p$ = .0087) and falls at +8 (63.33%; exploratory $p$ = .040 against +6), while EC keeps falling (19.33% → 10.33%) and reason-first keeps rising (77.59% → 86.16%). Further delaying the first candidate does not keep improving accuracy. The MATH decline is concentrated in Level-5 problems (47.2% → 36.0%).

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

Because the intervention is a single additive change at one position, its immediate effect on the prompt state is simple. Projecting the last-token prefill state of the steered layers onto the RSN direction gives a quantity that is linear in $\alpha$ ($R^2$ = .999 across nine doses for Llama, .9998 across eleven for Qwen2.5; Appendix H). The behavior that follows is not linear: Llama's accuracy peaks at −6 and collapses at −8, while the entry projection keeps moving at a constant rate. Across doses, EC tracks the accuracy curve far more closely than the entry projection does (Appendix H).

The size of the prompt-state perturbation therefore does not stand in for its behavioral effect. What the effect looks like depends on how the model's generation unfolds from that state, which the organization metrics capture. Section 6.1 returns to this point: entry gain alone predicts item-level correctness weakly, and adds nothing once output features are included.
