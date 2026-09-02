# P4b — Fixed-workpoint transfer to BBH numeric reasoning

**Protocol id** `bbh-p4b-v0`
**Status** frozen 2026-09-02, before any α=0 cell was generated
**Relation to P4** This is not a new phase. P4 asks one question — *does a
workpoint established on GSM8K still help when carried unchanged to another
reasoning task* — and LogiQA 2.0 (`logiqa2-p4-v0`) was its first task. BBH
numeric is the **second task of that same question**, hence `P4b`, not `P5`.

---

## 1. Why this task, and what LogiQA could not answer

P4's LogiQA result was a double null. It cannot say *why*, because LogiQA
changed **two** things at once relative to GSM8K:

| | answer space | reasoning content | submission |
|---|---|---|---|
| GSM8K | model constructs an integer | arithmetic word problems | `####` |
| LogiQA 2.0 | choose among 4 **given** candidates | textual logic | `Final answer: X` |
| **BBH numeric** | model constructs an integer | counting / nested arithmetic | `####` |

So a LogiQA null is consistent with two incompatible readings: the workpoint
does not transfer to *non-arithmetic reasoning*, or it does not transfer to a
*classification answer space*. These two BBH tasks hold the answer space and
the submission interface **fixed** and vary only the reasoning content, which
is the minimal-variable control needed to separate them.

Reading of the outcome, fixed in advance:

- **BBH transfers** → the answer space is the binding constraint; LogiQA's null
  is attributable to choosing among given candidates rather than constructing
  an answer.
- **BBH does not transfer** → the reasoning content matters too, and the
  workpoint's transfer range is narrower than the GSM8K→MATH→GSM-Hard chain
  suggested. This is a negative result and is reported with equal prominence.

Neither reading is available from one task alone, which is why the task order
in §2 is fixed here rather than chosen later.

## 2. Task tier, fixed in advance

1. **`object_counting`** — counting/enumeration, small-integer answer.
2. **`multistep_arithmetic_two`** — nested integer arithmetic. Run **only if**
   `object_counting` fails the stage-0 gate for both models.
3. **`dyck_languages`** — *not implemented and not authorised here.* It changes
   the answer space to a symbol string and needs its own frozen parser and
   prompt, i.e. its own protocol. It is named only so that the tier is on
   record, not left to be chosen after seeing results.

The order is frozen so that task selection is not a post-hoc narrative. A task
is reached only by its predecessor failing the gate, never by a result being
uninteresting.

## 3. Data

Source `lukaemon/bbh`, revision pinned to the full 40-hex SHA
`982bb89fd79532a8ac676a61fc42eb1aeec63f99`, `test` split, **all 250 items, no
sampling**. Verified pure Parquet with columns `['input','target']`.

Measured at freeze (descriptive; see §5 for why these do **not** move the gate):

| task | n | unique gold | gold range | majority-class rate | q chars (med) | digest |
|---|---|---|---|---|---|---|
| `object_counting` | 250 | 17 | 2 … 18 | **.104** | 112 | `4cfbf739e1fe7870` |
| `multistep_arithmetic_two` | 250 | 185 | −39960 … 250992 | **.016** | 41 | `e69d300b94274ce3` |

`data_bbh_numeric.py` hard-stops on: schema drift, row count ≠ 250, duplicate
questions, non-integer gold, `####` occurring inside a question, or any change
to the recorded gold distribution or content digest.

**This is not a blind validation.** BBH gold is public and reachable; P3
(GSM-Hard) was the blind test and is closed. What carries over from P3/P4 is
the *other* half of the discipline: α is read from the frozen GSM8K record and
never re-searched, the predictor / features / marker adapter are untouched,
generation and scoring are separate scripts, and the sample is frozen before
any steered cell runs. The sample is still emitted as a blind copy plus a
gold-bearing copy, so generation cannot reach a label even by mistake.

## 4. Held constant with the frozen GSM8K path

Prompt `select_templates_gsm8k(suite="default", cot=False, wording="plain")`,
`templates["neutral"]` — the neutral `#### <integer>` directive.
Budget `max_new_tokens=768`, `temperature=0.0`, `batch_size=24`.
Steering prefill-only, `tail=1`; llama band `11-20` (L=9) mask
`nmd_0.5_11_20_8B.npy`; qwen band `16-22` (L=6) mask `nmd_0.5_16_22_7B.npy`.
Parser `utils.extract_gsm8k_answer` + shared fallback + `normalize_gsm8k`,
**imported, never reimplemented**.

**No few-shot and no CoT.** The BBH release ships 3-shot CoT prompts;
inheriting them would add exemplars and explicit reasoning, re-mixing the
variables this design exists to separate. `cot=False` and `wording="plain"` are
hardcoded in the runner rather than exposed as flags — the "pushy" wording is a
known early-`####` inducer, and neither variant has a cell in this protocol.

α is expressible **only** as 0 or the model's frozen GSM8K workpoint
(llama `−6`, qwen `+8`). The launcher cannot name another dose and the scorer
rejects one.

One model per card, and a model's two cells share it: they are a paired
per-item contrast and bf16 greedy is not byte-reproducible across GPUs. The two
*models* may run on two cards; they are never compared per item.

## 5. Stage 0 — the headroom gate

**Purpose.** Stage 0 does not test whether α works. It confirms the task is
readable at all, so that a later null is interpretable rather than a
baseline-ceiling artifact. This repo has been bitten by that twice (Qwen's MMLU
betting cell, 97.9% of questions already betting 5; pv6 Easy-bare at
`late_opt_frac` .739).

**Procedure.** Generate α=0 only, on all 250 items, for both models. Score with
the independent scorer.

**Gate, frozen:** `α=0 first_acc ∈ [0.30, 0.85]`, judged **per model**.

- `> .85` → ceiling risk; do **not** run that model's workpoint cell.
- `< .30` → capability/format floor; do **not** run it.
- inside → the model is eligible.

The interval is fixed. The majority-class rate is **recorded and reported but
does not move it**: it is the trivial constant-guess baseline, not a
random-guess rate, and letting it adjust the interval after the data were
inspected would make the gate itself adjustable. At `.104` and `.016` neither
task's trivial baseline is high enough to matter, which is a fact about the
data, not a licence to revisit the rule.

**Diagnostics recorded, and explicitly not a gate:** no-marker rate, parseable
marker rate, multi-marker rate, generation length. Their only purpose is to
confirm the 768 budget and the shared extractor did not fail *technically*.
Seeing the model's output shape must not lead to a changed prompt, a redefined
parser, or a re-tuned budget.

**Eligibility is per model, and the panel is not.** Both models eligible → the
full two-model panel (§6). Exactly one eligible → its workpoint cell is still
run, but is pre-specified as **single-model exploratory transfer**: raw McNemar
and the paired CI are reported, Holm is **withheld**, and the raw p is labelled
unadjusted. Running Holm at m=1 would report an m=1 adjustment under an m=2
label. Neither eligible → no workpoint cell; move to the next task in §2.

## 6. Transfer test

Primary `ΔAcc = Acc(α) − Acc(0)` on `first_acc`, paired per item.
Exact two-sided McNemar with discordant counts reported.
**Item-level** paired bootstrap 95% CI, B=10000, seed 0. Each row is an
independent question here; there is no clustering structure, so the bootstrap
unit is the item.
**Holm over the two models, m=2**, judged only when both are complete.

`first_acc` (first `####`) is MAIN, matching GSM8K/GSM-Hard production.
`last_acc` is a tail-pollution **sensitivity** readout and is never the
headline. Both are reported.

## 7. `earlycand-v1` validation

Frozen separately in `docs/p4b_earlycand_audit_<task>.json` by
`freeze_p4b_earlycand_audit.py`, **before** stage-0 generation.

`earlycand-v1` was frozen on GSM8K, where a short first line containing a
number is answer-shaped. On `object_counting` the question is itself a list of
objects and the reasoning is a running count, so a leading number may merely
restate the question. Its GSM8K blind audit (precision 1.000, recall .976) was
on arithmetic text and does not transfer by assumption.

- **30 items**, ranked by a salted `sha256` of the question text — not by
  `sample_id` order, and never `hash()` (process-salted in Python 3).
  Selection digest `04508a19ce30b0b6` for `object_counting`.
- Labelled by the rubric in that file **without seeing the detector flag**;
  only afterwards are the two compared.
- **Pass** → `early_candidate` may be reported as an *exploratory* timing
  readout for this task, outside Holm.
- **Fail** → withdrawn as a timing metric for this task; only marker/format
  description survives.
- **Either way the accuracy main test is unaffected** — it never uses the
  detector — and **the detector is not re-tuned**, which would fork the
  definition against every stored GSM8K and MATH number.

`early_candidate` is an **outcome of α**, so stratifying accuracy on it is
post-treatment stratification: consistent-with evidence, never mediation.

## 8. Run order

1. Freeze both tasks' samples (`data_bbh_numeric.py`).
2. Freeze the earlycand audit list (`freeze_p4b_earlycand_audit.py`).
3. Stage-0 α=0, both models, one card each.
4. Score the gate (`eval_bbh_numeric.py` — the only script that reads gold).
5. Manual earlycand audit on the 30 frozen items of the α=0 cell.
6. Workpoint cells, for eligible models only.
7. Score the transfer.

Steps 1–2 precede any generation. Step 5 precedes step 6 only so that the
timing readout's status is settled before the steered data exist; its outcome
does not gate step 6.

## 9. What this protocol does not permit

- Re-searching α, or running any dose other than 0 and the frozen workpoint.
- Changing the predictor, features, or marker adapter.
- Changing the prompt, parser, or budget in response to observed output.
- Moving the `[0.30, 0.85]` interval, for any reason including the
  majority-class rate.
- Re-tuning `earlycand-v1`.
- Extending `--task` to a third config; `dyck_languages` needs its own protocol.
- Reporting a single-model result as the two-model panel, or applying Holm at
  m=1 under an m=2 label.
- Rewriting the closed P3 blind result or the frozen P4 LogiQA result.

Corrections are recorded as **additive amendments** (`docs/p4b_amendment_NN.json`),
never as silent edits followed by continued running.
