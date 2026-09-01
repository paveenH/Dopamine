# P4: Fixed-Workpoint Transfer to Generative Logical Reasoning (LogiQA 2.0)

**Protocol `logiqa2-p4-v0`. Two-stage freeze. This is stage 0, frozen BEFORE any
loader, preflight or generation exists.**

Stage 0 (this document) freezes the scientific question, the doses, the 300-item
sampling algorithm, the parsing rule, the statistics, and the token-budget
upgrade rule. Stage 1 (`logiqa2-p4-v1`, a separate additive amendment) freezes
only the token budget selected by the blind preflight, plus the manifest hash.
Nothing else may change between the two stages.

---

## 0. What this is, and what it is not

**Is:** a test of whether a steering workpoint established on GSM8K still helps
on *generative logical reasoning*, without re-searching alpha on the new task.

**Is not:**

- **Not commitment-based workpoint SELECTION.** That is P2B, frozen and closed.
  Selection asks "can commitment features pick the best dose on a new task?";
  this asks "does an already-established workpoint still help?". The two answer
  different questions and are never pooled. On Qwen they already diverge: its
  GSM8K workpoint is `+8` while its MATH optimum is `+6`.
- **Not a blind dataset validation.** P3 (GSM-Hard) was the blind test and is
  CLOSED. LogiQA 2.0 gold is public and reachable; no sealing is claimed.
- **Not evidence that workpoints transfer generally.** A null on both models is
  a task-boundary result and is reported with equal prominence.

Frozen framing sentence, to be used verbatim:

> LogiQA 2.0 tests fixed-workpoint transfer to generative logical reasoning.

---

## 1. Data

**Source.** `csitfun/LogiQA2.0`, English MRC test split, raw path
`logiqa/DATA/LOGIQA/test.txt` (JSON Lines). This is the same file the existing
`data_logiqa.py` already downloads; that script is NOT reused (see 1.4) and is
NOT modified.

**Measured at freeze time (2026-09-01), and asserted by the loader:**

| Property | Value |
|---|---|
| rows | 1572 |
| fields | `id`, `answer`, `text` (passage), `question`, `options`, `type` |
| options per row | 4 in all 1572 rows |
| label distribution (0/1/2/3) | 347 / 384 / 417 / 424 |
| `id` uniqueness | **1568 of 1572 — NOT unique** |
| `(passage, question)` uniqueness | **1557 of 1572 — NOT unique** |
| composite-key uniqueness | 1572 of 1572 |

A revision pin is recorded in the manifest at stage 1 (the raw URL follows
`main`, so the commit SHA is captured at download time and stored; a later run
that resolves to a different SHA is a hard stop, not a silent re-download).

### 1.1 The item key is a composite, and this is load-bearing

Neither the official `id` nor the question text is a key: both carry duplicates.
The frozen key is

    sha256("logiqa2-p4-v0" + ":" + id + US + passage + US + question + US + options.join(RS))

with `US` = `\x1f` and `RS` = `\x1e`. Measured unique across all 1572 rows.

**Never `hash()`** — it is process-salted in Python 3 and yields a different
sample every run. Both the official `id` and the content hash are stored per
item, so an upstream edit is detectable rather than silently absorbed.

### 1.2 The 300-item sample

Frozen algorithm, in this exact order:

1. Compute the composite key for every row.
2. Deduplicate on that key, keeping first occurrence in file order.
3. Group by gold label.
4. Within each label, sort by key (hex, ascending) and take the first **75**.
5. Concatenate the four groups and sort the result by key.

**Verified at freeze time: this yields exactly 75/75/75/75 = 300**, and the
manifest digest reproduces (`sha256` over the sorted keys, first 16 hex
`4d4b25e071a2a6dd`). Every label pool (347–424) exceeds 75, so no stratum is
starved.

If a future re-download changes any pool below 75, the response is to **record
achieved marginals in the manifest and report them**, never to silently relax
balance or substitute items.

### 1.3 What is stratified, and what is only provenance

Stratified: **gold label only** (75 each). This controls the one axis a
position/label prior could exploit.

`type` is a **multi-label dict** — a single row commonly carries 3–4 reasoning
types (e.g. Categorical + Conjunctive + Sufficient Conditional). It is stored
per item as provenance and may be described post-hoc, but it is **not** a
stratification axis and no per-type claim is pre-registered. Splitting 300 items
across an unbalanced multi-label space is the sampling problem that shelved
MMLU-Pro; it is avoided here rather than re-imported.

Options are **not shuffled**. The stored gold is the official `answer` index.

### 1.4 Why a new loader

`data_logiqa.py` downloads the correct file but is unusable here because it
writes to the stale `/data2/.../RolePlaying` tree, drops `id`, the raw
passage/question/options and `type` provenance, freezes no sampling manifest,
and has no generative runner. `data_logiqa2.py` is new; `data_logiqa.py` is left
byte-unchanged.

---

## 2. Cells

Four cells. No dose curve, no re-search.

| Model | Baseline | Transferred workpoint | Band | Mask |
|---|---|---|---|---|
| Llama3.1-8B | alpha = 0 | **alpha = -6** | 11-20 (L=9) | `nmd_0.5_11_20_8B.npy` |
| Qwen2.5-7B | alpha = 0 | **alpha = +8** | 16-22 (L=6) | `nmd_0.5_16_22_7B.npy` |

alpha is the workpoint already established on GSM8K and already used in P3 and
the P3 CoT supplement. **It is not re-searched here.** Adding a secondary
diagnostic dose (e.g. the parent paper's `+4`) is explicitly declined: the parent
used Llama3 / Qwen3 with different masks, bands and injection configs, so a
cross-paper dose contrast cannot isolate readout from model version.

**Held constant across a model's two cells:** the same 300 items in the same
order, `temperature=0.0`, greedy, prefill-only steering with `tail_len=1` at the
frozen anchor, and **one card for both cells of one model** (bf16 greedy is not
byte-reproducible across GPUs, so splitting a paired contrast across devices
mixes the device difference into the alpha effect). The two models may run on two
cards, since they are never compared per-question.

---

## 3. Protocol

Free generation. The model reasons, then emits a final line

    Final answer: X

with `X` in `A`-`D`. Reasoning is retained as a **behavioral readout only** —
generated rationales are never treated as evidence about internal mechanism.

### 3.1 Parsing (frozen)

| Role | Rule |
|---|---|
| **MAIN** | the **LAST** valid `Final answer: [A-D]` match |
| sensitivity | the **FIRST** valid match |
| no match | `no_marker`, scored incorrect |

Denominator is fixed at **300**. **No rescue generation**, no retry, no
second-pass prompt.

**Last-as-MAIN is task-specific and its reason differs from MATH's.** MATH uses
FIRST because tail loops pollute the last `\boxed{}`. LogiQA uses LAST because
the generative protocol invites the model to revise an initial judgement during
reasoning, and taking the first match would lock onto a pre-reasoning guess —
the answer-first pattern already documented on Qwen GSM8K and on Llama GSM-Hard
CoT. The two conventions are opposite on purpose; both are frozen here so
neither can be re-read as the other.

The first/last disagreement rate is reported as a descriptive revision measure.

### 3.2 Truncation is measured at generation time

The runner stores, per sample: `generated_token_count`, the stop reason
(natural EOS vs budget exhausted), and the raw text.

Offline re-tokenization is a **conservative cross-check only** and uses
**`>= budget - 1`**, never `== budget`. P3 established this: offline
re-tokenization is approximate at the boundary and an exact test under-reported
Llama's cap rate by ~12 pp.

---

## 4. Token budget: pre-declared upgrade rule

Budget is **not** chosen after seeing accuracy. It is chosen by a blind,
format-only preflight, under the rule frozen here.

**Preflight sample.** 5 items per gold label = **20 items**, drawn from the rows
that did **not** enter the formal 300, by the same key ordering (i.e. the next
5 by key within each label after the first 75). This guarantees zero overlap
with the formal set.

**Preflight is run at `max_new_tokens = 512`, on all four cells.**

**What is inspected:** output format (`Final answer: [A-D]` present and unique),
`generated_token_count`, stop reason, generation length, truncation rate.

**What is NOT inspected, computed, printed or stored: accuracy.** The preflight
loader does not receive gold for these 20 items.

**Upgrade rule (mechanical, no judgement):**

> If ANY of the 20 preflight outputs, in ANY of the four cells, has offline
> generated length `>= 511`, the formal budget becomes **1024**. Otherwise it
> stays **512**.

The selected budget, the preflight measurements, and the manifest hash are
written into `logiqa2-p4-v1` as an additive amendment. **Only the budget may be
set by the preflight.** If the format itself violates this protocol, the
response is a **HARD STOP** — the prompt and the parser are not redesigned in
response to output.

---

## 5. Statistics

**Primary metric.** `dAcc = Acc(alpha) - Acc(0)` per model, on MAIN parsing,
paired per item.

**Tests.** Exact two-sided McNemar with discordant counts reported; paired
difference in percentage points with a question-level bootstrap 95% CI
(B = 10000, seed 0).

**Holm family: the two models, m = 2.** The family is judged only when both
models' cells are complete. If one model is missing, the adjusted column is
**withheld entirely** and raw p is labelled unadjusted — running Holm over a
partial family reports an `m=1` adjustment under an `m=2` label, which is
anti-conservative.

**Secondary, excluded from Holm and labelled descriptive:** sensitivity accuracy
(FIRST parsing), `no_marker` rate, first/last disagreement rate, generation
length, truncation rate, and the frozen predictor's commitment score. Commitment
features are themselves outcomes of alpha, so stratifying accuracy on them is
post-treatment stratification — consistent-with evidence, never mediation.

---

## 6. Pre-registered readings

Declared now so the result cannot pick its own frame.

| Outcome | Reading |
|---|---|
| both models improve | the GSM8K workpoint transfers to generative logical reasoning |
| one improves, one does not | transfer is model-specific; report both, do not generalise from the positive one |
| neither improves | **task boundary**: the workpoint does not carry to this task-protocol combination |
| steering worse | the workpoint carries task specificity; report as such |

None of these outcomes licenses a claim about *why*. In particular, a Llama
result in the opposite direction to the parent paper's MCQ `+4` may **not** be
attributed to readout (option logits vs free generation): model version, mask,
band and injection config all differ between the two settings, so that
comparison cannot isolate readout. Isolating it would require the same model,
questions, prompt and steering config, comparing immediate option logits against
a free-generation final answer.

---

## 7. Fixed before anything is generated

Nothing in the frozen predictor, its features, the marker adapter, or the
workpoints may be changed in light of any outcome here. Corrections are recorded
as **additive amendments** (`docs/p4_amendment_NN.json`); no file frozen at an
earlier stage is silently edited.

## 8. Execution order (locked)

1. Freeze this document (stage 0).
2. Implement `data_logiqa2.py` (loader + manifest) and the format-only preflight.
3. Run the blind preflight. No accuracy.
4. Apply section 4's rule, write `logiqa2-p4-v1` with budget + manifest hash.
5. Only then implement and run the formal runner.
