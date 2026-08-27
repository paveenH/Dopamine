# Manifold Geometry Analysis
## 0. Executive Summary

**Question.** Is RSN steering a scalar gain along one direction, a retiming
within the natural manifold, or a departure from it — and can the same
geometric account explain why Llama peaks at α=−6 while Qwen plateaus on the
positive arm?

**Core Llama finding (last-prefill, decoder 18).** Steering is *piecewise*, not
one global scalar gain:

- The **negative arm is approximately a one-dimensional scalar family**:
  cos(−8, −6) = 0.989, scalar-fit residual 2.1%, 300/300 questions same-signed,
  and the least-squares `k = 1.379` matches the independent magnitude ratio
  24.63/17.67 = 1.394. So −8 is −6 travelled further along the same axis.
- The **positive arm is partially anti-aligned with a substantial orthogonal
  residual**: cos(−6, +6) = −0.662, so the shared axis carries `cos² ≈ 44%` of
  the energy and ~56% is orthogonal. +6 is neither a mirror nor a smaller −6.

This yields the most valuable interpretive result: **α=−6 reaches a working
region and α=−8 overshoots along the same axis**, so collapse need not mean
arriving at a wholly different state.

**Evidence boundary.** All of this is at **last-prefill**, the injection site
and the only strictly α-matched position. **No stable incremental behavioural
predictive value was detected** (frozen three-dose criterion), and
**last-prefill geometry did not stably extend to commit-aligned decode**.

**Current standing.** Llama analysis is **complete**; the manifold is
positioned as **last-prefill explanatory geometry**, not the mechanism line and
not a predictive line.

**Next step.** Exactly one: the frozen Qwen last-prefill analysis
(`PREREG_qwen_prefill.md`). Whether the manifold enters the paper beyond a
supplement is decided by its outcome.

---

## 1. Research Questions

Three competing accounts of what α does to the hidden state:

1. **Scalar gain** — steering changes magnitude along a fixed direction.
2. **On-manifold retiming** — trajectories stay within the natural manifold but
   change speed, phase occupancy or commitment timing.
3. **Directional reorganisation / off-subspace deviation** — extreme doses turn
   the trajectory, coinciding with degraded accuracy, loops or format failure.

**Cross-model question.** Llama shows an asymmetric peaked response (optimum
α=−6, collapse at −8) while Qwen shows a high-dose plateau. If both receive
gain-like steering, the candidate account is that **Llama overshoots along its
axis while Qwen's displacement saturates** — which would explain peak-vs-plateau
from geometry rather than from behaviour alone.

**On prediction.** The plan carried a binding constraint: the manifold must add
information beyond `s_t`/`Z_t` + commitment behaviour, or it is demoted to
auxiliary visualization. That check was run (§3.5, §3.6) and **not passed**.
But incremental prediction was always a *value check*, never the purpose:
distinguishing the three accounts above does not require predicting which
question is answered correctly. **A prediction failure does not erase the
descriptive geometry.**

---

## 2. Methods

### 2.1 Data and conditions

Llama-3.1-8B, GSM8K, 300 questions, No-CoT unless stated. Stored hidden states
only — no model output was regenerated for this analysis.

| condition | role |
|---|---|
| α=0 | natural baseline; the basis is fit here |
| α=−6 | best working point |
| α=−8 | negative-arm collapse |
| α=+6 | positive-arm damage representative |
| CoT α=0 / −4 | conditional validation set (§3.6) |

Band `[11,20)` → decoder layers 10–18 (L=9). **Primary layer: decoder 18**
(export slot `8`); sensitivity: decoder 10 (slot `0`); other layers supplement.
Pairing is always by `question_idx`, never row order.

### 2.2 Frozen question split

60/20/20 **by question**, realised **185/55/60**, `manifold-split-v1`.

- **Train** — fits the α=0 PCA basis. Fitting only; no k selection.
- **Validation** — `k = 5/10/20` robustness.
- **Test** — final dose comparison and prediction checks.

Rules: one split shared by every cell (per-cell splits would confound dose with
question difficulty); split by question, never by token (tokens within a
question are correlated, so a token split leaks); counts are **not** rebalanced
to 180/60/60 (thresholds sit on hash values, and sorting-and-slicing would make
each assignment depend on the whole set, breaking extension stability).

**k rule (frozen).** Primary `k = 20`; `k = 5/10/20` reported as sensitivity,
with agreement across all three required for a claim of robustness; the top-30
spectrum is diagnostic tail only and never enters the basis. **k is not chosen
from dose effects** — validation NRE falls monotonically in k and cannot
produce an extremum, so "select k on val" is not a coherent procedure.

### 2.3 Alpha=0 PCA reference geometry

Fit on **α=0 TRAIN questions only** — never on val (which checks k), never on
test (which carries the numbers), never on a steered cell (the thing under
test). Four phases each get their **own independent basis** (9 layers × 4
phases = 36), sharing nothing.

Two settings that are load-bearing:

- **Per-question equal weighting** (rows scaled by `1/√n_i`). Without it a
  20-token trajectory outvotes a 3-token one 7:1 and the basis becomes the
  manifold *of the slow samples* — and length correlates with α.
- **Every cell centred by the same α=0 `mu`.** Centring a steered cell on its
  own mean would subtract the very displacement under test.

### 2.4 Token phases and pairing rules

| phase | span | rows/question | role |
|---|---|---|---|
| `prefill` | last prefill token only | 1 | **the only strictly α-matched position** (same prompt, same token) — the only phase licensing a displacement claim |
| `pre_commit` | `[c−20, c)` | ≤20 | event-aligned distribution comparison |
| `post_commit` | `[c, c+20)` | ≤20 | same; commit itself is the first row |
| `decode_all` | whole decode, per-question row mean | 1 | level sensitivity; the only phase a no-commit sample can enter |

- **A sample committing before token 20 is kept, on its actual short window.**
  Dropping `c < 20` would systematically delete fast commitment — precisely the
  behaviour α moves (~23% of Llama α=0 samples commit before token 20, and that
  fraction is itself α-dependent). Truncate the window, never the sample.
- **No-commit samples are excluded only from aligned phases**, still entering
  `decode_all`; coverage is reported per cell and is never a gate.
- **Decode phases support event-aligned distribution comparison only.** α
  changed the generated text, so tokens differ across cells: no per-token
  pairing, no per-state displacement claim there.

### 2.5 Geometry metrics and matched nulls

**Normalized reconstruction error (NRE).**
`NRE(α) = mean(RE_α) / mean(RE_{α=0, held-out})`, per layer × phase — a **ratio
of cohort means**, never a mean of per-question ratios, which explodes on
questions whose α=0 residual is tiny. Hidden-state-norm normalization is
sensitivity only: it would divide out the scalar-gain effect under test.

**PCA-subspace alignment.** (Renamed from "local tangent alignment": this is a
*global per-phase* basis, so `local tangent` must not be used unless α=0
kNN/local PCA is actually implemented.) For prefill, `d = h(α) − h(0)` decomposes
into energy inside the α=0 top-k subspace and the remainder; primary is the
**energy-pooled ratio** `Σ‖W_k d‖² / Σ‖d‖²`, not the mean of per-question
ratios, which would weight a near-zero displacement as heavily as a large one.

**Cross-dose scalar fit.** Least squares `d_a ≈ k·d_b`, reporting cos, `k` and
residual. `residual ≡ 1 − cos²` exactly at the least-squares `k` (verified to
1.1e-16), so **`k` is the only independent number of the three** and is always
reported beside the residual. A per-question sign fraction accompanies the
pooled cosine, since a pooled value can hide a mixture of aligned and
anti-aligned questions.

**Commitment-centroid distance.** Distance to the α=0 TRAIN `post_commit`
centroid, defined on train and evaluated on test.

**Matched isotropic null.** Mandatory beside every spectrum: the null must match
each phase's `m`, `nq`, `dim` **and** the per-question weighting, through the
same Gram path. Otherwise low-rank structure cannot be distinguished from the
sampling necessity of `m ≪ dim`. The random reference for displacement is
`k/dim` — an isotropic displacement puts `20/4096 = 0.488%` of its energy in any
20-D subspace.

Retired null: **shuffled-question**, which is generally meaningless for pooled
PCA. Valid geometric negative controls are the matched isotropic spectrum,
a random orthonormal subspace at the same k, and trajectory-order shuffle (for
speed/curvature only).

### 2.6 Statistical rules and claim boundaries

- Bootstrap/cluster unit is the **question**, throughout.
- Dose contrasts frozen in advance: `−8 vs 0`, `−6 vs 0`, `+6 vs 0`.
- Holm within a metric family; families are not pooled.
- A dose counts as improved only if **both** members of its metric pair move
  the right way; "stable" requires **all three doses** to agree in direction.
  Anything less is reported as mixed / not detected, never as a positive.

Frozen claim boundaries:

- PCA shows **linear low-rank**, not a nonlinear manifold.
- `k = 20` is an analysis cap, **never an intrinsic dimension**.
- Energy **outside** the top-k subspace is **not** "off-manifold" — k=20 spans
  only ~50% of α=0 variance, so the complement is largely ordinary variation.
- Null ratios compare **within a phase only**.
- Results describe the **computational geometry of RSN steering**; they are not
  biological dopamine evidence, and **no causal claim** is made.

---

## 3. Results

### 3.1 Data integrity, accuracy and coverage

Four primary cells accepted at **full probe** (not sampled): n=300 each,
`stored_layer_indices = [10..18, 31]`, band `[11,20)`.

- Projection reproduction reads **exactly 0.00e+00** in all four cells.
- Per-question agreement with the lightweight batch measured **1.000** on all
  three fields in all four cells — an observed property of this batch.
- Accuracy **79.67 / 60.00 / 51.67 / 40.67** (α = −6 / 0 / +6 / −8) reproduces
  the −6 peak **same-batch**. This is the 184 bs=1 batch and may never be mixed
  per-question with the 182 dose table.

**Commit coverage** (reported as a behavioural result, never a gate):

| cell | α | coverage |
|---|---|---|
| `nocot` | 0 | 297/300 = .990 |
| `nocot_aneg6` | −6 | 298/300 = .993 |
| `nocot_aneg8` | −8 | 294/300 = .980 |
| `nocot_a6` | +6 | **281/300 = .937** |
| `cot` | 0 | 297/300 = .990 |
| `cot_aneg4` | −4 | 296/300 = .987 |

The +6 drop of ~5pp is α-dependent and matches the direction already recorded
for +α format degradation. Consequence: any commit-aligned number at +6 rests
on 16 fewer questions, selected by the manipulation's own outcome.

Fit-phase row counts: prefill / `decode_all` m=185; `pre_commit` m=2940
(nq=**150**); `post_commit` m=3576 (nq=183). `pre_commit`'s nq is lower because
a sample committing at decode step 0 has a post window but no pre window.

### 3.2 Alpha=0 spectral concentration

Cumulative explained variance, observed vs matched isotropic null
(20 draws, median [2.5, 97.5]):

| phase | m | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| prefill | 185 | .257 vs .039 | .365 vs .076 | **.499 vs .148** [.1477, .1489] |
| pre_commit | 2940 | .243 vs .006 | .344 vs .011 | **.446 vs .022** [.0222, .0223] |
| post_commit | 3576 | .296 vs .005 | .397 vs .010 | **.484 vs .020** [.0200, .0201] |
| decode_all | 185 | .641 vs .039 | .730 vs .076 | .812 vs .148 |

Null intervals are narrower than 1% relative; 20 draws have converged. All
three k values agree in direction.

**Frozen wording: low-rank spectral concentration relative to a matched
isotropic null.** Ratios are within-phase only — prefill's 3.4× and the commit
windows' ~20× are not commensurable, because prefill has `m = nq = 185` so the
null alone already reaches 14.8%. The spectrum is a slow tail with no elbow,
which is exactly why k is fixed at 20 with sensitivity reported.

`decode_all` must be read per phase: reduced to one row per question, it is fit
on ~185 points, so a flat spectrum there is the reduction removing within-question
variation, **not** an unstable manifold.

### 3.3 Exact last-prefill PCA-subspace analysis

Ambient-space decomposition, decoder 18, 300 questions paired strictly by
`question_idx`, same α=0 train basis, `f_k = ‖W_k d‖²/‖d‖²`, primary =
energy-pooled ratio.

**Primary (TEST split, k=20):**

| α | mean‖d‖ | inside [95% CI] | outside |
|---|---|---|---|
| −8 | 24.63 | **21.4%** [20.9, 21.8] | 78.6% |
| −6 | 17.67 | **21.2%** [20.8, 21.7] | 78.8% |
| +6 | 13.23 | **9.8%** [9.2, 10.5] | 90.2% |

**k sensitivity (TEST):** −8 goes 9.6 → 17.0 → 21.4 and −6 goes 9.5 → 16.8 →
21.2 (both **+11.8pp** from k=5 to k=20, near-identical per-dimension profiles),
while +6 goes 5.8 → 8.1 → 9.8 (**+4.0pp**) — extra dimensions do not recover
+6's energy.

**Split agreement at k=20** (train / val / test): −8 21.8 / 21.1 / 21.4;
−6 21.8 / 21.0 / 21.2; +6 10.7 / 9.5 / 9.8. Not overfitting to the basis's own
train questions. Pooled and per-question means differ by <0.1pp, so no
large-displacement question dominates.

**Random reference, reported alongside:** an isotropic displacement puts
`20/4096 = 0.488%` of its energy in any 20-D subspace. So +6's 9.8% is **20×**
random and −6/−8's 21.2% is **43×** — **all three doses are strongly aligned**
with the α=0 principal structure, differing by a factor of two. Without this
reference 9.8% misreads as "barely aligned".

Magnitude is monotone in |α| (24.63 / 17.67 / 13.23) while the inside ratio is
not. **Direction conclusions do not follow from the inside ratio** — two
displacements can fill the same top-k subspace equally and still point
different ways. Direction is settled in §3.4.

### 3.4 Cross-dose direction and scalar fit

Least-squares `d_a ≈ k·d_b`, TEST split:

| pair | cos [95% CI] | k | residual | same-signed |
|---|---|---|---|---|
| −8 \| −6 | **0.989** [0.989, 0.990] | 1.379 | 0.021 | 100.0% |
| −8 \| +6 | −0.657 [−0.667, −0.647] | −1.222 | 0.569 | 0.0% |
| −6 \| +6 | −0.662 [−0.674, −0.650] | −0.884 | 0.562 | 0.0% |

All four splits agree to three decimals.

- **The negative arm is approximately a one-dimensional scalar family**:
  cos 0.989, residual 2.1%, 300/300 same-signed, and `k = 1.379` matches the
  independently computed magnitude ratio 24.63/17.67 = 1.394.
- **The positive arm is partially anti-aligned with a substantial orthogonal
  residual**: cos ≈ −0.66, shared axis `cos² ≈ 44%`, orthogonal ~56%. 100%
  same-signed means a *consistent partial* anti-alignment, not a mixture of
  sub-populations.
- Therefore steering does **not** share one line across all doses.

The inside ratio (§3.3) and this cosine are **two agreeing observations, not
the same fact**. Establishing that they are one would require testing whether
+6's orthogonal component sits outside top-k.

### 3.5 Incremental prediction

**Verdict: no stable incremental behavioural predictive value was detected.**

Provenance: round 1 (correctness, Z-only baseline) spent the TEST split for its
intended purpose and exhausted it. Round 2 was designed after seeing round 1
and is therefore **post-hoc**, not confirmatory. Using a test set once is not
the error; re-tuning against it afterwards would be.

**Commit position** — pre-generation-only baseline `[Z_prefill, prefill
confidence]`; commitment behaviour is inadmissible because it *is* the outcome:

| α | baseline R² / MAE / ρ | +geometry R² / MAE / ρ |
|---|---|---|
| −8 | .003 / 73.1 / .129 | .019 / 72.3 / .192 |
| −6 | .074 / 74.1 / .276 | .104 / 72.5 / .359 |
| +6 | .004 / 67.8 / .033 | −.004 / 68.3 / −.010 |

Both negative doses improve on all three metrics; +6 worsens on all three.
Under the frozen "all three doses must agree" rule this is **mixed**.

**Correctness** — commitment *is* admissible here (predictor, not outcome):
AUC .700 / .386 / .547 → .688 / .497 / .443; log-loss .6294 / .4990 / .6917 →
.6429 / .4882 / .7247. **Results are inconsistent across metrics and doses** —
−6's AUC improves (.386 → .497) while −8's and +6's fall, so this must not be
written as "all three worsened".

Round 1, for the record (correctness, Z-only baseline): AUC .485 / .479 / .570
→ .503 / .617 / .458, one dose up and two down.

Caveats that travel with every number: TEST is exhausted; n=60, so sampling
error exceeds the observed differences; and the wording is **"not detected"**,
never "disproved" — a near-chance baseline does not invalidate the test, since
geometry could have improved on it independently and did not.

### 3.6 CoT negative-arm conditional confirmation

H1, frozen before any CoT projection: *adding prefill geometry improves
commit-position prediction for negative α and does not for positive α.*
Frozen model transferred untouched; CoT projected onto the **existing No-CoT
α=0 basis** (refitting would make it a new model rather than a confirmation).

**CoT α=−4, commit position:**

| | R² | MAE | ρ |
|---|---|---|---|
| baseline (Z, conf) | −0.101 | 56.5 | −0.091 |
| +geometry | **−0.056** | **53.9** | **0.121** |

All three move in the predicted direction, so **H1's negative half passes its
pre-set directional criterion**.

**But the absolute predictive power is weak**: R² remains *negative* after
adding geometry, meaning the model still does worse than the training mean; ρ
remains small. The honest reading is **a reproducible weak directional signal,
not strong predictive evidence** — it improved from worse-than-the-mean to
less-bad.

Correctness on the same cell got clearly worse: AUC .531 → .462, log-loss
.3298 → .4242.

Scope limits, frozen in advance: only the **negative half** of H1 is testable
(CoT has no positive dose, and the positive half must not be reported as
confirmed nor quietly dropped); **α=−4 is not among the doses H1 was derived
from** (−8/−6), so this is a generalisation to an unmeasured dose; and the
numbers describe where CoT states sit relative to the **No-CoT** natural
manifold.

### 3.7 Minimal pre/post-commit decode analysis

Decoder 18, k=20, TEST split, event-aligned distributions only.

| phase | α | n | NRE | speed | centroid dist. |
|---|---|---|---|---|---|
| pre_commit | −8 | 28 | 1.087 | 6.791 | 2.708 |
| | −6 | 55 | 0.988 | 6.846 | 2.464 |
| | 0 | 47 | 1.000 | 7.096 | 2.617 |
| | +6 | 45 | 1.044 | 6.801 | 2.801 |
| post_commit | −8 | 60 | 1.409 | 5.319 | 4.883 |
| | −6 | 59 | 0.836 | 6.903 | 4.392 |
| | 0 | 60 | 1.000 | 6.112 | 4.252 |
| | +6 | 57 | 1.050 | 6.323 | 4.770 |

Against the pre-set rule — (a) −6 and −8 consistent with each other, (b) +6
stably separated, (c) visible in **both** phases — none of the three holds:

- In `post_commit`, −8 sits well above the α=0 reference (NRE 1.409) and −6 well
  below (0.836); the two negative doses do not group together.
- +6 sits at 1.044 / 1.050, close to the α=0 reference, while −8 is the outlying
  cell — the reverse of the prefill picture, where +6 was the distinct one.
- `pre_commit` NRE spans only 0.988–1.087 across all four doses, so the
  structure appears in one phase only.

Additionally `pre_commit` n is strongly imbalanced (−8: 28 vs −6: 55), because α
moves commit timing and hence the fraction of samples with a pre-commit window.
That is a selection effect on top of everything else, so even descriptive
comparison there is discounted.

**Conclusion: last-prefill geometry did not stably extend to commit-aligned
decode.** This does not falsify the manifold; it bounds where the clean
structure lives.

---

## 4. Interpretation

### 4.1 Supported findings

1. **α=0 states carry low-rank spectral concentration relative to a matched
   isotropic null** — 20–24× the null at k=20 in the commit windows.
2. **The negative arm is approximately a one-dimensional scalar family**, with
   magnitude growing from −6 to −8 along a shared axis.
3. **The positive arm is partially anti-aligned with a substantial orthogonal
   residual** — not a mirror, not merely a smaller displacement.
4. **Steering is piecewise, not one global scalar gain.**
5. Consequently, **α=−6 reaches a working region and α=−8 overshoots along the
   same axis** — collapse need not mean arriving at a wholly different state.
   This is the single most valuable interpretive result of the line.
6. **Positive/negative behavioural asymmetry has a geometric counterpart**: the
   asymmetry is present in the hidden-state geometry itself, not only in the
   accuracy numbers.

All six are **last-prefill** statements.

### 4.2 Relation to Dopamine and the Thinking Curve

These results can support a **computational-level** dopamine analogy only:

- The negative arm's shared direction behaves like a stable **gain-control
  axis**.
- −6 → −8 is movement along that axis from an effective dose into an excess
  region — structurally the Yerkes–Dodson optimum → overdose shape.
- +6's directional reorganisation suggests that over- or reverse-regulation can
  enter a **different computational regime**, not merely a smaller or reversed
  one.
- The candidate cross-model account: **Llama's effective displacement keeps
  growing and eventually overshoots; Qwen's may be compressed or saturating**,
  which would explain a peak versus a plateau. This is precisely what the frozen
  Qwen analysis tests.

**The sign of α is not an increase or decrease of biological dopamine.** The
manifold describes the computational geometry of RSN steering, not a
neurotransmitter.

### 4.3 Unsupported claims and limitations

Not supported by anything here:

- Any **nonlinear manifold** claim — PCA establishes linear low-rank only.
- `k = 20` as an **intrinsic dimension** — it is an analysis cap.
- **"Off-manifold"** for energy outside the top-k subspace — k=20 spans ~50% of
  α=0 variance, so the complement is largely ordinary variation.
- **Causal** claims of any kind. This is an offline re-projection of stored
  states; a causal test needs random/orthogonal *injection* and re-collection.
- Cross-model comparison of **raw α, PC axes or hidden-state values** — masks,
  layer counts (L=9 vs L=6) and activation scales differ, so an equal α is not
  an equal intervention.
- **General behavioural predictive value** — §3.5 and §3.6 found none, and the
  one passing criterion sits on a model whose R² is negative.

Structural limitations:

- **This batch has no independent second α=0 cell**, so α=0 manifold stability
  can only be estimated by train/val subsampling or bootstrap, never by
  cross-validating two independent α=0 batches. This weakens the "the manifold
  is stable" premise and is a pre-registered stop condition.
- **TEST is exhausted** — every post-round-1 number is supplementary.
- Commit-aligned cohorts are **selected by the manipulation's own outcome**
  (coverage and pre-commit availability both move with α).
- **Prediction failure does not erase the descriptive geometry**, and equally,
  the descriptive geometry does not license a predictive or causal claim.

---

## 5. Status and Next Step

**Llama analysis: COMPLETE.** Positioned as **last-prefill explanatory
geometry** — a mechanistic-explanatory supplement, not the mechanism line and
not a predictive line.

**Not being extended** (deliberate, not pending): TLE, UMAP/t-SNE, full
per-layer sweeps, additional doses, and any further correctness-prediction
work. The decode check was the last extension and it stopped the line.

**Only remaining step: the frozen Qwen last-prefill analysis.** Scope is fixed
in `PREREG_qwen_prefill.md` — last-prefill only, Qwen's own α=0 basis, own band
`[16,22)`, own commit locator, three questions (does the positive arm share one
direction; does displacement magnitude saturate; inside ratio against each
model's own subspace), with failure conditions written in advance.

**Whether the manifold enters the paper beyond a supplement is decided by that
outcome.** If Llama grows along one axis into overshoot while Qwen's magnitude
flattens along one axis, the geometry offers a candidate account of
peak-versus-plateau. If Qwen's positive arm does not share one direction, or its
magnitude keeps growing at +12, or the two models simply look alike, that is
reported as such and the line closes as a Llama-only supplement.

Remaining doses (No-CoT ±2/±4/+8) are available but are **continuity checks
only** — same questions, same basis — never an independent validation set.

---

## Appendix. Artifact and provenance index

Implementation details, exact commands, guards, tests and failure provenance
live in `CLAUDE.md` § *Manifold pilot*. This index lists what exists and where.

**Scripts** (in the Dopamine repo)

| file | role |
|---|---|
| `check_hs_llama.py` | §3.1 H5 acceptance (server, read-only) |
| `manifold/split_manifest.py` + `.json` | §2.2 frozen split |
| `manifold_fit.py` | §2.3–2.4 basis fit + projection |
| `manifold_prefill_exact.py` | §3.3 ambient displacement decomposition |
| `manifold_prefill_direction.py` | §3.4 cross-dose cosine and scalar fit |
| `run_manifold_pilot.sh` | launcher |
| `test_check_hs_llama.py`, `manifold/test_split_manifest.py`, `test_manifold_fit.py` | guard suites |

**Offline analysis** (`RoleAnswer/manifold/`, not in git)

`incremental.py` (§3.5 round 1) · `incremental2.py` (§3.5 round 2) ·
`confirm_cot.py` (§3.6) · `decode_minimal.py` (§3.7)

**Pre-registrations** (`RoleAnswer/manifold/`)

`PREREG_incremental.md` · `PREREG_negative_arm_confirm.md` ·
`PREREG_decode_minimal.md` · `PREREG_qwen_prefill.md`

**Data artifacts**

- Server: `components/llama3/manifold/phase1b_eot/` (basis + four No-CoT cells),
  `components/llama3/manifold/phase1b_eot_cot/` (CoT cells, same basis reused)
- Local: `RoleAnswer/llama3/dopamine/manifold/` — `basis.npz`,
  `basis_meta.json`, `manifold_*.json`, `prefill_exact.json`,
  `prefill_direction.json`
- Source H5: `components/hidden_states/gsm8k/phase1b_eot/`
