# P3 Protocol — Blind Cross-Task Validation on GSM-Hard

**Protocol version:** `p3-v1`
**Frozen:** 2026-08-28
**Repo git hash at freeze:** `2ef25fe6a8ca1ced130f3291993e6926146c857c`
**Content pin:** committed to the Dopamine repo at `docs/PREREG_P3.md`; the
commit hash is the authoritative pin (this file lives in `RoleAnswer/`, which is
not in git, so a repo hash alone does NOT pin its content — the lesson from P2).

**Status at freeze: NO MODEL HAS BEEN RUN. The GSM-Hard dataset has not been
downloaded. No GSM-Hard accuracy has been computed or viewed by anyone.**

Any deviation after this freeze MUST be labelled **exploratory** and MUST NOT be
presented as a protocol-conforming result.

---

## 0. What makes P3 different from P2

P2 was a **retrospective locked transfer test**: MATH accuracy had already been
observed in prior work, so however cleanly the predictions were locked, the test
could never be blind.

P3 is a **genuine blind validation**, and that rests entirely on the eligibility
audit in §1. If GSM-Hard accuracy is viewed by anyone before the prediction file
is frozen, **that property is destroyed permanently and cannot be restored by
any later re-freeze.** This is the single most fragile asset in the protocol.

---

## 1. Eligibility audit (performed 2026-08-28, before any download)

| Check | Result |
|---|---|
| GSM-Hard data / loader / runner in the Dopamine repo | **none** — appears only as a *plan* in `TODO.md` |
| GSM-Hard artifacts in `RoleAnswer/` | **none** |
| GSM-Hard files anywhere on the local machine | **none** (not downloaded) |
| Any GSM-Hard accuracy ever computed | **impossible** — no data, no loader, no results tree |

**Verdict: ELIGIBLE for blind validation.**

Commands that established this (re-runnable):

```bash
grep -rin "gsm.hard\|gsm8k.hard\|gsmhard" --include="*.py" --include="*.md" --include="*.sh" .
find /Users/paveenhuang -maxdepth 6 -iname "*gsm*hard*"
```

---

## 2. Frozen decisions

### 2.1 Extractor — reuse the GSM8K version unchanged (decision 1)

GSM-Hard preserves GSM8K's `####` format and prompt structure, so
`all_hash` / `norm_gsm8k` / `fallback_gsm8k` / `has_early_candidate` and the
`commit_state` / `posN` definitions apply **with zero adaptation** — a cleaner
transfer than MATH, which needed a `\boxed{}` adapter.

After download, a **schema/format acceptance check only**. If the format is
incompatible, this is a **HARD STOP**. Adapting the extractor after seeing
results is forbidden.

### 2.2 Candidate doses — five per model, fixed in advance (decision 2)

- **Llama:** `−8 / −6 / −4 / 0 / +4`
- **Qwen:** `−4 / 0 / +4 / +6 / +8`

**Qwen retains a negative probe by design.** With all candidates on the positive
side, "correctly predicted the positive direction" would not be a real test —
the prediction could not have been wrong. `+8` suffices to probe high-dose
fall-back; `+12` is deliberately excluded from this round.

Ten cells total. Doses are **not re-searched** and are not expanded after seeing
any result.

### 2.3 Sample — 300 questions by frozen hash (decision 3)

- 300 questions, matching P2's 口径.
- Selected by **question hash, NOT the first 300 in dataset order**.
- Rule: `sha256(f"{salt}:{question_text}")`, `salt = "rsn-p3-sample-v1"`, take
  the 300 smallest digests. Deduplicate on exact question text first.
- The selection rule, salt, dedup rule and the resulting question-set digest are
  **frozen before generation** in `p3/p3_sample_manifest.json`.
- **Whether to extend to the full 1319 is NOT decided after seeing results.**

### 2.4 Large integers — do not keep scoring in a way known to be wrong (decision 4)

Audit finding: `norm_gsm8k` routes through `float`, so integers above `2^53`
corrupt silently (`12345678901234567890` → `12345678901234567168`). GSM-Hard
deliberately substitutes large values, so this boundary may genuinely be hit.

Procedure, after download but **before running any model**:

- If **no** gold answer exceeds `2^53`: use the original extractor unchanged.
- If any does: add an **exact-integer normalization** to the central utility,
  used for this validation only, and **verify it reproduces the original scoring
  EXACTLY on the existing GSM8K data**; then freeze its version and test result.
- **Large-integer questions are neither deleted nor separately excluded.**

The frozen predictor and commitment features are **not modified** by this — it
concerns scoring only.

### 2.5 Success metrics — two distinct near-optimal notions (decision 5)

- **Primary:** direction; selected workpoint; performance regret.
- **Secondary:** Spearman ρ; predicted overshoot.

Two near-optimal sets, which must never be conflated:

- **Predicted near-optimal region** — from the frozen predicted-score rule
  (within 10% of the predicted-score spread, as in P2).
- **Observed near-optimal set** — doses whose **paired per-question accuracy
  difference from the empirical best** is statistically indistinguishable
  (question-clustered bootstrap CI contains 0).

**The verdict is whether the predicted workpoint falls inside the OBSERVED
near-optimal set.** Closeness in predicted score is NOT sufficient.

### 2.6 Unreadable curve — no α=0 accuracy threshold (decision 6)

A low baseline does not mean the steering curve is unreadable; a large
improvement may still exist. So **no accuracy threshold is used.** Instead:

- Per model, a **paired omnibus test** over per-question correctness across that
  model's five doses.
- **Holm correction across the two models' omnibus tests.**
- If no dose difference is detected → **"no readable dose curve"**.
- In that case direction and workpoint are **not evaluable**: recorded as
  **neither a prediction success nor a prediction failure.**

---

## 3. Locked execution order (binding)

1. Freeze this protocol, the sample manifest, the scorer test and the label
   firewall. Commit them.
2. Download GSM-Hard. Run the **schema/format acceptance check** and the
   **`2^53` gold-answer audit**. Hard stop on format incompatibility.
3. End-to-end format test on a few samples: confirm the generated files can be
   read by both the predictor and the accuracy evaluator. **Format only — no
   accuracy.**
4. Generate all ten cells. **Do not compute accuracy.**
5. Read **commitment features only**; apply the frozen P2 predictors; predict
   direction and workpoint; **freeze `p3_predictions.json` and its SHA256.**
6. **Only then** unlock accuracy, once, and evaluate.

Steps 4–5 must not read `answer`. The evaluator must refuse to run unless the
prediction file already exists — the same fail-closed guard as P2.

---

## 4. Predictor reuse (unchanged from P2)

The frozen P2 `commitment-only` predictors are applied **as-is**: no refitting,
no recalibration, no per-task intercept. `posN` is imputed with the **frozen
GSM8K median** (Llama 0.1478, Qwen 0.7652). All six features are retained.

- `p2_predictor_llama.json` sha256 `b9ee07e4…`
- `p2_predictor_qwen.json` sha256 `9aad950a…`

Per P2 §15.1, the **primary readout is ordering, not calibration**. Absolute
predicted probabilities may drift under dataset shift; that is expected and is
not a failure.

---

## 5. Result interpretation (frozen in advance)

- **Direction correct + workpoint in the observed near-optimal set** → blind
  cross-task validation succeeds; this is the evidence level P2 could not reach,
  and the point at which a transferable reasoning-control principle may be
  claimed for these two models on these tasks.
- **Direction correct, workpoint outside** → supports label-free direction
  selection only.
- **Direction wrong** → the P2 transfer does not generalize to a harder
  arithmetic task; report as a negative result, do not retune.
- **No readable dose curve** (§2.6) → not evaluable; neither success nor failure.

A negative or non-evaluable outcome is a legitimate, preregistered result and
**must be reported**. Nothing in the predictor, features, doses, sample or
extractor is adjusted in response to any outcome.

---

## 6. Pre-download audit results (recorded before any download, any model run)

### 6.1 `norm_exact` is verified against existing GSM8K — 0 verdict changes

The §2.4 normalizer was written and tested before download.

**The acceptance criterion is that no scoring VERDICT changes, NOT that no
string differs.** A normalizer repairing a float64 corruption differs on the
corrupted string by construction; demanding byte-identity would reject the fix
for doing its job. What must never change is whether an answer counts as
correct — that would silently redefine every stored GSM8K and MATH number.

Result over 6600 predictions from 16 GSM8K result files:

| | |
|---|---|
| scoring verdicts changed | **0** |
| float64 corruptions repaired (verdict unchanged) | **1** |

The single repair is already present in existing data — a model predicted
`977777777777777777777`, which the frozen `norm_gsm8k` corrupts to
`977777777777777836032`. Gold for that question is `8000`, so the prediction is
wrong under both normalizers and **no stored number changes.** This confirms the
`2^53` hazard is real and reachable by actual model output, not hypothetical.

`norm_exact` is therefore cleared for use **if and only if** the §2.4 audit finds
GSM-Hard gold answers exceeding `2^53`. It is additive; the frozen
`norm_gsm8k` is NOT patched, per the standing rule that patching it requires its
own re-freeze of all MATH and GSM8K numbers.

### 6.2 A vacuous-pass bug was found and fixed in the audit itself

The first version of the equivalence test globbed a wrong filename pattern,
compared **0 values, and reported OK** — the fail-open pattern this project has
repeatedly been bitten by. It now exits non-zero when it matches no files or
compares no values: *a guard never exercised is not a guard.*
