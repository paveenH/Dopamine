# P3 Supplement: CoT Robustness and Commitment Dynamics

**Protocol `p3-supp-v1`. Frozen BEFORE any CoT cell is generated.**

This is a **post-P3 supplement**. It does NOT modify, reinterpret or re-open the
closed P3 blind validation (`docs/p3_result_20260830.json`, tag
`p3-result-unsealed`). P3's verdict, wording and boundaries stand exactly as
frozen.

---

## 0. Two parts with DIFFERENT evidential status

This distinction is the reason the supplement is split, and it must survive into
every citation.

| Part | Status | Why |
|---|---|---|
| **1. CoT condition transfer** | **locked prospective condition test** | The four CoT cells do not exist yet. Predictions are frozen here, before generation, and scored only afterwards. |
| **2. Commitment pattern analysis (No-CoT)** | **EXPLORATORY** | The ten No-CoT cells were unsealed on 2026-08-30. Anything computed on them now is post-hoc, however principled. |

Part 2's CoT half inherits Part 1's locked status, because those cells are
generated under this frozen protocol. Part 2's No-CoT half never can.

**This is NOT a new blind dataset validation.** The questions and their gold are
the same 300 already unsealed. What is prospective is the CONDITION (CoT), not
the data.

---

## 1. CoT condition transfer

### 1.1 Cells (four, no more)

| Model | CoT baseline | CoT + transferred α | Band | Mask |
|---|---|---|---|---|
| Llama3-8B | α = 0 | **α = −6** | 11–20 (L=9) | `nmd_0.5_11_20_8B.npy` |
| Qwen2.5-7B | α = 0 | **α = +8** | 16–22 (L=6) | `nmd_0.5_16_22_7B.npy` |

α is the workpoint already established on GSM8K and already used in P3. **It is
NOT re-searched, and no CoT dose curve is run.** Four cells answer whether CoT
and a fixed steering workpoint are complementary, redundant or conflicting;
a full curve would answer a different question at several times the cost.

### 1.2 Held constant (identical to the P3 No-CoT cells)

- the same 300 questions, same order, same `sample_id`, digest `48cc763545d2ee23…`
- `max_new_tokens=768`, `temperature=0.0`, `batch_size=24`, greedy
- prefill-only steering, `tail_len=1`, injection at the `Answer: ` anchor
- one model's two cells on ONE card (bf16 greedy is not byte-reproducible across GPUs)
- **the ONLY change is inserting `Let's think step by step.`** — verified to be the
  single line separating the two frozen templates

### 1.3 Primary question

> Does the GSM8K workpoint still beat α=0 **under CoT**?

**Primary metric** `ΔAcc = Acc(CoT+α) − Acc(CoT)`, on `first_acc` via the frozen
offline extractor, paired per question.

**Statistics.** Exact two-sided McNemar per model with discordant counts; paired
difference in percentage points with a bootstrap 95% CI (question is the unit,
B=10000, seed 0). **The two models form ONE Holm family (m=2)** for the primary
metric.

### 1.4 Interaction

    Δ_interaction = [Acc(CoT+α) − Acc(CoT)] − [Acc(NoCoT+α) − Acc(NoCoT)]

The No-CoT half is READ FROM THE ALREADY-UNSEALED P3 CELLS. Consequently:

- **the interaction is NOT a locked prediction** — one of its two halves is
  already known. It is reported as a descriptive contrast with a CI, and is
  excluded from the Holm family.
- CI computed by paired question-level bootstrap on the per-question difference
  of differences.

Pre-registered readings, declared now so the result cannot pick its own frame:

| Δ_interaction | Reading |
|---|---|
| ≈ 0 | steering's effect does not depend on CoT |
| significantly < 0 | CoT and steering partly overlap or saturate |
| significantly > 0 | CoT and steering are complementary |
| steering WORSE under CoT | the workpoint carries prompt-condition specificity |

### 1.5 Frozen before generation

`p3_supp_predictions.json` records, before any CoT cell exists:

- the four cells and their α, band and mask
- the predicted direction of `ΔAcc` for each model, and its justification
- the frozen predictor's mean predicted score per cell (the same frozen artifacts,
  **not refitted, not recalibrated**)

The evaluator refuses to run unless that file exists — the same ordering guard as
P3.

### 1.6 Success and failure

Success is **not** required for the supplement to be reportable. A null, or
steering being worse under CoT, is a result about condition specificity and is
reported with equal prominence. Nothing in the frozen predictor, features,
marker adapter or workpoints may be changed in light of the outcome.

---

## 2. Commitment pattern analysis

Fourteen cells: the ten unsealed P3 No-CoT cells + the four new CoT cells.

### 2.1 Readouts (all from the frozen extractors, none refitted)

`early_candidate` rate · commit position `posN` · the four-value commit state
(committed / marker_unparsed_nonloop / loop / no_marker) · reasoning length ·
`n_markers` · post-commit character proportion · frozen predictor score ·
`first_acc`.

### 2.2 Questions

1. **Does the GSM-Hard dose effect still act through commitment timing?**
   Llama `−8 → −6`: does commitment move from a dysregulated to a healthy range?
   Qwen `0 → +8`: does answer formation and submission become more stable?
2. **Does CoT change HOW steering acts?**
   Does CoT alone already improve commitment? Does α improve it further? Does an
   accuracy gain travel with a commitment change in the same direction?

### 2.3 Per-question AUROC of the frozen predictor on GSM-Hard

Reported **separately for No-CoT and CoT**, never pooled. No refitting, no
recalibration. Ranking (AUROC) and calibration (predicted vs observed rate) are
reported and interpreted separately — P3 already established that ordering
transferred while absolute calibration did not.

### 2.4 Status labelling is mandatory

Every No-CoT number in Part 2 is labelled **exploratory**. Every CoT number
inherits Part 1's locked status. A table mixing both must label its rows.

### 2.5 What Part 2 cannot do

It is post-hoc on the No-CoT side, so it **describes** a mechanism consistent
with the P3 result; it does not test one. Commitment features are themselves
outcomes of α, so stratifying accuracy on them is post-treatment stratification —
consistent-with evidence, never mediation.

---

## 3. Storage

Generation text and commitment metrics only. **No hidden states** — nothing here
needs them, and collecting them would cost 24–30 GB for no readout.

## 4. What this supplement may never be called

- a blind validation (the data is already unsealed)
- a replication of P3 (same questions, same gold)
- evidence that changes P3's verdict, wording or boundaries
