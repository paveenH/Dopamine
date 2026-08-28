# P2 Protocol — Commitment-Based Prediction and Cross-Task Workpoint Selection

**Protocol version:** `p2-v1.1`
**Frozen:** 2026-08-28 (`p2-v1`); amended 2026-08-28 (`p2-v1.1`, see Amendments)
**Repo git hash at freeze:** `75d738c26cf36ac4c3e3e8015eb9e94dec922744` (clean tree)
**Status at freeze:** written BEFORE any P2A model was fitted and BEFORE any
MATH accuracy was read in this line of work.

Any deviation from this document after the freeze MUST be labelled
**exploratory** in the results and MUST NOT be presented as a
protocol-conforming result.

---

## 0. Scope and standing constraints

- Uses ONLY already-generated outputs. **No new model inference, no GPU.**
- **P1 is CLOSED.** This protocol does not modify P1 data, metric definitions,
  figures or conclusions. Nothing in P2 may be used to revise P1.
- Frozen extractors are **imported, never reimplemented**:
  `all_hash`, `all_boxed`, `norm_gsm8k`, `norm_math`, `fallback_gsm8k`,
  `fallback_math` (from `analyze_first_last_acc`), and `has_early_candidate`
  (from `early_candidate_detector`, `MAX_LINE_CHARS=60`).
  A private copy is how the inline-vs-offline gap opened on MATH and how a
  private commit locator silently redefined the event on Qwen.

### 0.1 Audit facts established in P2 Phase 0 (these motivated the decisions below)

| Fact | Consequence |
|---|---|
| MATH JSONs carry only `question/generated/correct/gold_answer/gold_solution/level/type/pred_answer` — **no `x_prefill`** | entry gain `z` CANNOT transfer to MATH → decision A1 |
| Llama MATH No-CoT has only `−4 / 0 / +4`; Qwen MATH has all 9 α (`−8…+8`) | Llama cannot test peak/overshoot → decision A2 |
| Manifold split is `sha256(salt:question_text)`, accuracy-independent | a hash split is legitimate; P2 builds its own → decision A3 |
| GSM8K ∩ MATH question sets = 0 | MATH is a genuinely distinct task; no leakage by construction |
| All 9 Qwen MATH cells hold the same 300 questions in the same order | order-based pairing is sound, and is re-asserted at load |
| `all_boxed` is a frozen balanced-brace extractor | the MATH marker adapter reuses frozen code, it is not a new rule |

---

## 1. Analysis hierarchy (frozen)

- **P2A primary** — text-only commitment predictor. Directly transferable to MATH.
- **P2A supplementary** — GSM8K-internal `entry-only` and `entry + commitment`,
  used ONLY to state whether entry gain adds predictive information.
  These do **not** enter the MATH transfer chain.
- **P2B primary** — apply the frozen text-only predictor to each MATH α cell,
  take the mean predicted correctness, and from that select steering direction
  and workpoint.

**Final scope statement (frozen wording):**

> Qwen tests the complete direction, ordering, workpoint and regret.
> Llama tests only the local direction within the existing three points.

---

## 2. Data

### 2.1 GSM8K (P2A)

Lightweight signal JSONs — these carry projections AND generated text in one
object, so a question's features cannot desynchronise.

- Llama: `llama3/dopamine/signal/dopamine_signal_gsm8k_8B_nocot{a}_ema0.95_L11-20.json`
  α ∈ {−8, −6, −4, −2, 0, +2, +4, +6, +8} → 9 cells × 300 = 2700 rows
- Qwen: `qwen2.5/dopamine/signal/dopamine_signal_gsm8k_7B_nocot{a}_ema0.95_L16-22.json`
  α ∈ {−8, −6, −4, −2, 0, +2, +4, +6, +8, +10, +12} → 11 cells × 300 = 3300 rows

CoT cells are **excluded** from P2A primary (No-CoT is the main condition and
CoT is a contrast control; mixing regimes would confound the commitment
features with the generation regime).

### 2.2 MATH (P2B)

- Qwen: `qwen2.5/math/mdf_{0,±2,±4,±6,±8}/math_7B_16_22.json` — 9 cells × 300
- Llama: `llama3/math/math_eot/mdf_{0,4,neg4}/math_8B_11_20.json` — 3 cells × 300

CoT MATH cells are excluded, matching P2A.

---

## 3. Feature definitions (frozen)

### 3.1 Primary features — text-only, transferable (A1(a))

All six are computable from `generated` alone on BOTH tasks.

| Feature | Type | Definition |
|---|---|---|
| `early_candidate` | binary | `has_early_candidate(text)`, frozen, identical definition on both tasks |
| `cs_marker_unparsed_nonloop` | dummy | commit-state, see 3.2 |
| `cs_loop` | dummy | commit-state, see 3.2 |
| `cs_no_marker` | dummy | commit-state, see 3.2 |
| `posN` | continuous | normalized char position of first parseable marker; missing → imputed, see 3.3 |
| `posN_observed` | binary | 1 iff a parseable first marker exists |

Baseline of the commit-state dummies is `committed`.

### 3.2 Commit-state — FOUR mutually exclusive values (A6)

**P1's three-value description is unchanged. The four-value encoding is used
for predictor encoding ONLY** and must never be reported as a revision of P1.

```
committed                 a parseable marker exists
marker_unparsed_nonloop   marker present, not parseable, n_bare < 4
loop                      marker present, not parseable, n_bare >= 4
no_marker                 no marker anywhere
```

Rationale for splitting P1's `marker_unparsed`: `is_loop` is a strict subset of
`marker_unparsed`, so a three-way one-hot plus a separate `loop` column is
structurally collinear. The four-value partition is exhaustive and mutually
exclusive, and carries the same information without the collinearity.

**Acceptance:** the four counts must sum to 300 in every cell, and
`marker_unparsed_nonloop + loop` must equal P1's `marker_unparsed` count.

### 3.3 `posN` denominator and missingness (A5)

- `posN = commit_char / len(text)`, denominator = **total generated chars**
  (this is P1's frozen denominator; it is NOT the pre-commit span).
- `posN` is defined ONLY on `committed` samples.
- **No complete-case deletion.**
- **Imputation:** the median of observed `posN` **within the training folds**.
  For the final frozen predictor, the median over ALL GSM8K training data,
  computed once and stored in the predictor artifact.
  A fixed constant (e.g. 0.5) is NOT used.
- `posN_observed` carries the missingness information.

### 3.4 Supplementary features (GSM8K-internal only)

`z` = entry gain `x_prefill`, standardized **within model** using the α=0
mean/SD. `gen_chars`, `n_markers`, `n_decode`. Confidence metrics
(`entropy/top1/margin`) are supplementary and, per the standing repo rule,
do **not** enter any cross-model primary model.

### 3.5 Forbidden inputs

Feature extraction MUST NOT read `gold_answer`, `gold_solution`, `correct`,
`pred_answer`, `level`, `type`, or any offline accuracy. Asserted in code.
`level` is forbidden as a feature because it is a MATH-only difficulty label
with no GSM8K counterpart, and using it would make the predictor untransferable.

---

## 4. MATH marker adapter (A7 — format adaptation, frozen before reading accuracy)

GSM8K's `####` and MATH's `\boxed{}` implement the **same semantic definition**:
a final-answer marker, a first parseable commit position, and repeated
submission. The adapter changes the marker only; the feature semantics,
the four-value partition, the `posN` denominator and the imputation rule are
byte-identical across tasks.

| Semantic role | GSM8K | MATH |
|---|---|---|
| all parseable markers | `all_hash(text)` | `[b for b in all_boxed(text) if b != ""]` |
| bare marker count | `text.count("####")` | `text.count(r"\boxed{")` |
| first-marker char position | `re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text).start()` | char index of the first `\boxed{` whose content is non-empty |
| early candidate | `has_early_candidate` | `has_early_candidate` (**identical**, already frozen for both tasks) |

**Empty-`\boxed{}` handling.** CLAUDE.md records that an empty `\boxed{}` can
occupy the first-marker slot (a generation discussing the format). P2 Phase 0 measured
this: 3/2700 Qwen and 1/900 Llama MATH samples. The adapter treats an empty
`\boxed{}` as **not a parseable marker** (it contributes to the bare count but
not to `all_boxed`-parseable), which is the same treatment `all_hash` gives an
unparseable `####`. This rule is frozen here, before any accuracy is read, and
is NOT a patch of the underlying extractor gap — that gap stays unpatched per
CLAUDE.md and would require its own re-freeze.

**This adapter must be frozen and tested before MATH accuracy is read, and must
not be adjusted on the basis of results.**

### 4.1 Three label-free acceptance checks (run before any accuracy is read)

1. Every sample falls into **exactly one** commit-state; the four counts sum to
   the cell's n in every cell of both tasks.
2. `posN` denominator is total chars and missingness is reported explicitly
   per cell (count and rate, with the denominator stated).
3. The extraction code reads none of the forbidden fields in 3.5 — asserted by
   loading only the whitelisted keys.

---

## 5. Cross-validation design (A3(b), A4)

### 5.1 P2 fold manifest — deterministic 5-fold

- New, P2-specific, frozen manifest: `p2/p2_fold_manifest.json`.
- Method: `fold(q) = int(sha256(f"{salt}:{question_text}")[:8], 16) % 5`,
  `salt = "rsn-p2-fold-v1"`.
- Keyed on **question text**, so it is accuracy-independent by construction and
  stable under reordering.
- Built separately for GSM8K (the P2A task). Folds are **not** needed for MATH,
  which is evaluated only by the frozen predictor.
- Both models and **all doses** use the same fold assignment: the GSM8K question
  set and order are identical across Llama and Qwen (verified in P1), so the two
  models share folds item-for-item.

### 5.2 Grouping and leakage rules

- A question's **entire dose group** (all 9 or 11 rows) lies in ONE fold.
- **Raw α is not a feature**, and dose identity is not used by the primary
  predictor. A predictor that saw α would learn the dose rather than the state
  and could not transfer.
- Standardization / imputation parameters are estimated **from training folds
  only** and applied to the held-out fold.
- **No question fixed effects.**

---

## 6. Models

Simple, interpretable **logistic regression** predicting per-sample correctness.
Any hyperparameter tuning uses **nested grouped CV** inside the training folds.
Llama and Qwen are modelled **separately** throughout; no shared raw α and no
shared absolute thresholds.

| Model | Features | Role |
|---|---|---|
| `commitment-only` | the six primary features (3.1) | **P2A primary**, transfers to MATH |
| `entry-only` | `z` | P2A supplementary |
| `entry+commitment` | `z` + the six | P2A supplementary |

Supplementary models exist to answer one question: **does entry gain add
predictive information beyond commitment?** They are not transferred.

---

## 7. Metrics and inference

Primary metrics, all on held-out folds: **AUROC**, **Brier score**, **log loss**,
**calibration slope and intercept**.

All confidence intervals and all model-vs-model differences use **cluster
bootstrap with the question as the cluster unit** (n = 300 questions), never
row-level resampling — the dose rows of one question are strongly dependent.

The headline is the **paired difference** between models (primarily
`commitment-only` vs `entry-only` on GSM8K), not any single model's absolute
value.

---

## 8. Frozen predictor artifact (Phase 3)

After P2A, freeze per model into `p2/p2_predictor_{llama,qwen}.json`:

- feature schema and column order
- preprocessing (standardization means/SDs; the `posN` imputation median)
- fitted coefficients and intercept
- missing-value rule
- output score definition
- tie-breaking rule

**Output score** = per-sample predicted probability of correctness.
**A dose's commitment score** = the mean predicted correctness over that cell's
300 questions.

No shared raw α, no shared absolute thresholds between the two models.

---

## 9. P2B — retrospective locked transfer (Phase 4)

Order of operations is binding:

1. Extract MATH features with the frozen adapter (§4).
2. Run the three label-free acceptance checks (§4.1).
3. Apply the frozen predictor (§8).
4. Compute mean predicted correctness per α.
5. **Write and freeze** `p2/p2b_predictions.json` containing:
   - steering direction from α=0 (positive / negative)
   - predicted best α
   - predicted near-optimal region
   - predicted overshoot or plateau onset
6. **Only then** read MATH correctness via the frozen `first_acc` extractor.

### 9.1 Evaluation

- steering direction correct (yes/no)
- Spearman correlation between predicted score curve and true accuracy curve
- performance regret: `max_α Acc(α) − Acc(α̂)`
- whether α̂ lies in the near-optimal set (doses statistically indistinguishable
  from the best dose)

### 9.2 Per-model scope (A2(a)) — conclusions frozen SEPARATELY, never merged

- **Qwen (9 α):** direction, ordering, workpoint, regret — the full test.
- **Llama (3 α: −4/0/+4):** **local direction only.** With three points the
  protocol **cannot** test peak, overshoot, plateau onset, or full-curve regret,
  and no such claim may be made. This limitation is declared here, in advance.

### 9.3 Accuracy 口径 (A8)

- **`first_acc` is the sole MAIN readout**; `last_acc` is a sensitivity readout.
- The offline total must reproduce the frozen published cell counts.
- The two known extractor gaps (leading-zero normalization; empty first
  `\boxed{}`) and any inline-vs-offline difference of 1–2 items are **recorded,
  not patched** in P2.

### 9.4 Required wording

This stage MUST always be written as a

> **retrospective locked transfer test**

It must NOT be described as blind, preregistered, or fully held-out validation,
because MATH accuracy has already been observed by the researcher in prior work.

---

## 10. Success and failure conditions (frozen in advance)

- **P2A holds, P2B fails** → commitment features monitor error but do not select
  a cross-task workpoint.
- **P2B gets direction right but not the point** → supports label-free steering
  **direction** selection only.
- **P2B selects a near-optimal region with low regret** → supports retrospective
  cross-task workpoint selection.
- **A transferable inference-control principle may be claimed only after blind,
  preregistered validation on a dataset whose accuracy has never been viewed.**

### 10.1 Explicit failure conditions

- P2A gate (amended in `p2-v1.1`): a model passes only if its
  `commitment-only` **AUROC 95% CI lower bound > 0.5** on held-out folds.
  "CI does not contain 0.5" is NOT the criterion — a CI lying entirely below
  0.5 does not contain 0.5 yet must fail.
  If a model fails, its P2B is **not run**: a predictor with no signal cannot
  select a workpoint, and running it anyway would be fitting to MATH.
- **The two models are gated INDEPENDENTLY; one failing does not block the
  other.** Qwen passing enables the full Qwen P2B; Llama passing enables the
  Llama three-point local direction test only (§9.2).
- P2B: if the frozen predictor's curve has no variation across α
  (all cells within bootstrap noise), the result is reported as
  **"no transferable signal detected"**, never as a wrong direction.
- Any acceptance check in §4.1 failing is a **hard stop**, not a warning.

---

## 11. Required outputs

- this frozen protocol
- `p2/p2_fold_manifest.json` + its `--check` reproduction
- feature exhaustiveness audit (the four-state partition, per cell, both tasks)
- P2A held-out metrics table (both models, three model specs)
- calibration figure
- frozen predictor artifacts with parameters
- `p2/p2b_predictions.json`, saved BEFORE MATH accuracy is read
- predicted-score vs actual-accuracy figure
- all negative results and any triggered failure conditions

Code and acceptance detail → `CLAUDE.md`.
`AdaptiveThinking.md` records only 口径, main results, evidence level and
conclusion boundaries. **P1 is not modified on the basis of P2 results.**

---

## 12. Amendments

### `p2-v1.1` — pre-analysis clarification (2026-08-28)

**Status: made BEFORE any P2A model was fitted and before any MATH accuracy was
read.** This is a pre-analysis clarification, not a post-hoc revision, so the
preregistration remains intact.

**Repo git hash at amendment:** `75d738c26cf36ac4c3e3e8015eb9e94dec922744`
(unchanged — the amendment touches only this protocol document, which lives
outside the git repo, in `RoleAnswer/`.)

Two changes:

1. **§10.1 gate tightened.** The `p2-v1` wording "AUROC CI contains 0.5" had a
   hole: a CI lying entirely **below** 0.5 does not contain 0.5 and would have
   passed, despite being worse than chance. The criterion is now
   **AUROC 95% CI lower bound > 0.5**.

2. **§10.1 gates made explicitly per-model and independent.** Qwen and Llama are
   judged separately; one model failing does not block the other. Qwen passing
   enables the full Qwen P2B (direction / ordering / workpoint / regret); Llama
   passing enables only the three-point local direction test of §9.2.

Naming note: this protocol's audit stage is called **P2 Phase 0** throughout, to
avoid collision with the project's earlier P0 / Manifold stages.

No other section is changed. Feature definitions, fold manifest, MATH adapter,
metrics and success/failure conditions are as frozen in `p2-v1`.

---

## 13. Pre-modelling observations (recorded before any model was fitted)

Recorded at the completion of the §4.1 audit, **before** any P2A model was
fitted and before any MATH accuracy was read. These are observations about the
feature distribution, not protocol changes. No definition is altered.

### 13.1 P1 reproduction (acceptance)

The four-value encoding re-sums to P1's published three-value counts exactly.
Llama GSM8K α=0: `committed` 177, `marker_unparsed_nonloop + loop` = 14 + 52 =
66, `no_marker` 57 — identical to P1, including the loop sub-flag's 52.

### 13.2 Two primary features are near-degenerate ON MATH

The marker adapter is semantically faithful, but `\boxed{}` does not produce the
degenerate repeated tail that `####` does, so on MATH:

| feature | Llama MATH (n=900) | Qwen MATH (n=2700) |
|---|---|---|
| `cs_marker_unparsed_nonloop` | 0 | 1 |
| `cs_loop` | 0 | 0 |
| `cs_no_marker` | 124 | 45 |
| `committed` | 776 | 2654 |

**Consequence, stated in advance:** two of the six primary features carry
essentially no variance on the transfer target, so their fitted coefficients are
inert there. The transferable signal rests effectively on `early_candidate`,
`posN`, `posN_observed` and `cs_no_marker`.

This narrows the transfer channel and **must be reported with any P2B result**,
whether that result is positive or negative. It is recorded here so it cannot
later be presented as a post-hoc explanation of a P2B outcome.

The features are NOT redefined or dropped in response: they are legitimately
defined on both tasks, the partition is exhaustive, and altering the feature set
after seeing the distribution would be exactly the post-hoc freedom this
protocol exists to prevent.

### 13.3 Commit-state moves with α in opposite ways on the two models (GSM8K)

Llama's `no_marker` rises with α (51 → 57 → 90 at α=−8 / 0 / +6) while Qwen's is
**0 in all 11 cells** — Qwen always emits `####`, matching the frozen repo rule.
Qwen's variation is instead `committed` rising at high dose (239 → 294 at
α=0 → +8). The two models therefore populate the four states differently, which
is why they are modelled separately with no shared thresholds.

---

## 14. Provenance freeze (pre-modelling)

`PREREG_P2.md` lives in `RoleAnswer/`, which is **not** in the Dopamine git
repo, so the repo hash `75d738c` does NOT pin this file's content. Both
remedies are applied, before any model was fitted:

1. **Hash manifest** — `p2/p2_freeze_manifest.json` records the SHA256 of every
   P2 artifact. Its `PREREG_P2.md` entry is the hash **before** this section was
   appended (a file cannot contain its own hash).
2. **Git copy** — a verbatim copy of this protocol, with this section included,
   is committed to the Dopamine repo at `docs/PREREG_P2.md`. That commit is the
   authoritative content pin; its hash is recorded in the commit message.

Artifact hashes at freeze:

| file | sha256 (first 16) |
|---|---|
| `PREREG_P2.md` (pre-§14) | `26e98c1955979 1aa` |
| `p2_fold_manifest.json` | `13e141101c20e37d` |
| `build_p2_folds.py` | `89b0d5e25f61d97e` |
| `p2_features.py` | `210d4eb5936b6b28` |
| `run_p2_audit.py` | `49b8385ea92ca8d0` |
| `p2_feature_audit.json` | `224bff52d9053825` |

No P2A model had been fitted and no MATH accuracy had been read at this point.

---

## 15. P2B execution clarifications (recorded before MATH predictions were computed)

Recorded **before** the frozen predictors were applied to MATH and before any
MATH accuracy was read. Clarifications of existing sections; no definition is
changed.

### 15.1 The primary readout is ORDERING, not calibration

Absolute predicted probabilities may lose calibration under dataset shift: the
predictors were fitted on GSM8K, and MATH differs in base rate, marker
convention and generation length. Therefore the P2B primary readouts are
**dose ordering, steering direction and dose selection** — NOT whether the
predicted probability equals the true accuracy.

A calibration gap on MATH is therefore **expected and is not itself a failure**.
Reporting the predicted curve as if it estimated MATH accuracy would be a
misreading of what was frozen.

### 15.2 Nothing is refitted or recalibrated

- The frozen `commitment-only` predictors are applied as-is: no refitting, no
  recalibration, no per-task intercept adjustment.
- `posN` is imputed with the **frozen GSM8K median** (Llama 0.1478, Qwen 0.7652).
  Using a MATH-derived median would let the transfer target's own distribution
  enter the predictor.
- All six frozen features are retained, **including the two that are
  near-degenerate on MATH** (§13.2). Dropping them after seeing the distribution
  is exactly the post-hoc freedom this protocol prevents.
- Predictor, features and marker adapter are **not adjusted regardless of
  outcome**, success or failure.

### 15.3 The within-α AUROC check is exploratory

The per-dose AUROC breakdown run after P2A (Llama 0.59–0.71, Qwen 0.50–0.80;
α=0 alone 0.677 / 0.748) was **not preregistered**. It is reported as an
**exploratory robustness check** against the artifact that the predictor merely
separates doses. It does not alter the P2A gate, which was decided on the
preregistered criterion alone.

### 15.4 Evidence level of the P2A result

P2A establishes that commitment timing **predicts** correctness on unseen GSM8K
questions and outperforms entry gain, with no detectable added value from entry
gain. This is **predictive evidence, not causal evidence** — the causal question
belongs to the direction-injection control, which is a separate experiment.
