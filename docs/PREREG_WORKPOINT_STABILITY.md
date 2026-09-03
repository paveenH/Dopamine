# Pre-registration: workpoint-stability supplement (`wps-v0`)

Frozen 2026-09-03, BEFORE any of the seven cells below was generated.

## 1. Question

The fixed workpoints (llama `−6`, qwen `+8`) were read from the frozen GSM8K
No-CoT record. Two observations since then suggest they may not be the local
optimum under every condition:

- Llama GSM8K CoT: `−4` (85.0%) beats `−6` (75.3%). The four-cell paired DiD
  `[(CoT−4)−(CoT−6)] − [(NoCoT−4)−(NoCoT−6)]` is **+14.67 pp**, bootstrap 95%
  CI **[+7.67, +21.67]**, sign-flip permutation **p < 1e−4**.
- Qwen GSM8K CoT: the peak sits at `+6`, but `+6 vs +8` is not separated
  (p=.371).

This supplement asks a NARROW question:

> **Is each reported workpoint a local optimum, or the edge of an untested
> region?**

It does **not** re-open any closed result and does **not** search for a new
workpoint to report as a headline. Every existing statistical family and every
frozen conclusion stands unchanged.

## 2. The seven cells

Each cell adds a NEIGHBOUR to an existing cell so that the reported point has
data on both sides.

| # | Model | Benchmark | Condition | New α | Purpose |
|---|---|---|---|---:|---|
| 1 | Llama | GSM8K | CoT | `−2` | is `−4` a local peak or a `−4/−2` near-optimal plateau |
| 2 | Llama | GSM-Hard | CoT | `−4` | does the GSM8K CoT shift `−6→−4` appear across datasets |
| 3 | Qwen | GSM-Hard | CoT | `+6` | compare `+6/+8` under CoT |
| 4 | Qwen | GSM-Hard | CoT | `+10` | right neighbour of `+8`: peak, plateau, or still rising |
| 5 | Qwen | GSM-Hard | No-CoT | `+10` | `+8` is at the tested boundary; add its right neighbour |
| 6 | Llama | MATH | No-CoT | `−8` | left neighbour of the boundary best `−6` |
| 7 | Llama | MATH | CoT | `−8` | is `−6` still in the peak region under CoT |

## 3. Statistics, frozen before the run

- **One family, Holm `m=7`.** Each cell contributes exactly ONE pre-declared
  contrast: the new α versus the SAME condition's stored `α=0`, paired per
  question, exact two-sided McNemar.
- **Never pooled** with the P3 (m=2), P3-supplement (m=2), P4 (m=2), P4b (m=2),
  MATH dose (m=3) or Llama CoT dose (m=3) families. Those are closed.
- **Neighbour comparisons** (`−4 vs −2`, `+8 vs +10`, `−6 vs −8`, …) are the
  actual object of interest but are **EXPLORATORY and outside Holm**, reported
  with unadjusted p. Reason: they are the comparison the supplement was built
  after seeing, so they cannot also carry a corrected claim.
- **Primary readout is `first_acc`** through the frozen offline extractor;
  `last_acc` is sensitivity only.
- **Report a near-optimal REGION, not an argmax.** For each curve, report every
  dose whose paired difference from the empirical best is not distinguishable,
  together with the top-vs-neighbour paired difference and its CI. A single
  argmax over noisy neighbours is what this supplement exists to avoid.

## 4. What a result may and may not say

- **May**: "the reported workpoint is / is not a local optimum under this
  condition"; "the near-optimal region is {…}".
- **May NOT**: redefine any frozen workpoint retroactively; convert a better
  neighbour into a new headline accuracy; describe two-point comparison as a
  dose curve; pool `|α|` across models (different mask, band L=9 vs L=6,
  different activation scale — raw α is not a common dose scale).
- **The "CoT needs smaller |α|" hypothesis stays CONDITIONAL.** Current
  evidence is mixed: Llama GSM8K supports it, Llama MATH does not (both
  conditions peak at `−6`), Qwen GSM8K is weakly consistent but not separated,
  Qwen MATH does not move. Frozen wording: *CoT may compress the effective dose
  range or move the best workpoint toward a smaller absolute dose, but this
  depends on model and task and is not a uniform rule.*

## 5. Run constraints

Every new cell uses the SAME physical GPU, prompt, batch size and generation
budget as the stored cells it is compared against. Budgets are inherited, not
chosen: GSM8K/GSM-Hard `768`/bs `24`; MATH `2048`/bs `8`.

One model's cells within one curve stay on one card (bf16 greedy is not
byte-reproducible across GPUs). Stored cells carry no device field in their
summary CSV, so every new-vs-stored contrast is a **cross-run pairing** and must
be cited with that caveat.

## 6. No-GPU analyses accompanying this supplement

1. GSM8K `−4 vs −6` **last-answer** DiD, bootstrap CI and permutation p (the
   `first_acc` version is already computed: +14.67 pp, [+7.67, +21.67], p<1e−4).
2. A **top-vs-neighbour paired difference** table across all conditions, ending
   in a reported near-optimal region rather than a bare argmax.
