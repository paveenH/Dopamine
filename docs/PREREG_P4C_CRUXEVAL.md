# P4c — Fixed-workpoint transfer to open-ended program-execution reasoning (CRUXEval-O)

**Protocol id** `cruxeval-p4c-v0`
**Status** frozen 2026-09-03, before any cell was generated
**Relation to P4** Not a new phase. P4 asks one question — *does a workpoint
established on GSM8K still help when carried unchanged to another reasoning
task* — and this is its **third task**. LogiQA 2.0 (`logiqa2-p4-v0`) was the
first, BBH numeric (`bbh-p4b-v0`) the second. Hence `P4c`, not `P5`.

---

## 1. Scientific question, and what it can and cannot isolate

Does the GSM8K workpoint (llama `−6`, qwen `+8`) still help on a task that is:

- **option-free** — no candidate set to choose among;
- **open-ended in answer space** — the model constructs an arbitrary Python
  literal, not an integer;
- **program-state tracking** — the reasoning is execution simulation, not
  arithmetic word-problem solving;
- **deterministically scored** — semantic equality of parsed Python objects,
  not fuzzy text matching.

### What P4c removes, and what it adds

|  | answer space | reasoning content | submission |
|---|---|---|---|
| GSM8K | model constructs an integer | arithmetic word problems | `####` |
| LogiQA 2.0 | choose among 4 **given** candidates | textual logic | `Final answer: X` |
| BBH numeric | model constructs an integer | counting / nested arithmetic | `####` |
| **CRUXEval-O** | model constructs an **arbitrary Python literal** | program-state tracking | `####` |

P4c **removes LogiQA's option-comparison interface** and keeps the `####`
submission marker, but it **changes reasoning type AND answer structure at the
same time**. It therefore **cannot by itself isolate** which of the two decides
the transfer outcome.

**This limit is stated here, in advance, precisely because it is the same
double-move that made LogiQA's null ambiguous.** P4c's value is as a **third,
harder point on the transfer boundary**, not as a controlled single-factor
manipulation. Separating a reasoning-type effect from an answer-space effect
would need a within-item contrast (the same items with the answer space varied),
which is a different experiment and is **not authorised here**.

### Frozen outcome wording

- **Transfers** → "the workpoint transferred to open-ended program-execution
  reasoning." **Never** "the workpoint transfers to code reasoning in general",
  and never a claim that the option interface was the LogiQA cause.
- **Does not transfer** → "the workpoint's transfer range does not extend to
  open-ended program-execution reasoning." This does **not** establish which of
  reasoning type or answer space is the binding constraint. A null is reported
  with equal prominence.

**In neither case may P4c claim to have identified a causal role for reasoning
type, answer space, or submission timing.**

Final positioning, verbatim:

> **Fixed-workpoint transfer to open-ended program-execution reasoning.**

---

## 2. Data

- Source `cruxeval-org/cruxeval`, config `default`, split `test`, **800 rows**
- Revision pinned to the full 40-hex commit SHA
  `b96af0450242eb4da433032b90998f25588a5d0f`
  (a branch name is not a pin: it follows upstream exactly as an unset
  revision would)
- Columns `code / input / output / id`; `id` = `sample_0 … sample_799`
- **Formal sample: 300 items**, frozen before any generation

### Sampling rule

Rank all 800 by `sha256(salt:id:code:input)` with salt `cruxeval-p4c-v0`, take
the first 300. **Never Python `hash()`**, which is process-salted and would
silently produce a different sample every run.

The same 300 items, in the same order, serve **all eight cells**.

Stored per item: `sample_id` (0..299, the position in the frozen order), the
original upstream `id`, and a `content_sha256`, so an upstream edit is
detectable rather than silently absorbed.

### Measured at freeze (2026-09-03), asserted by the loader

| property | value |
|---|---|
| `questions_sha256[:16]` | `4580b7a9a9ef6054` |
| `gold_sha256[:16]` | `a214d1fc7d84a2d9` |
| gold parse | **800/800 `ast.literal_eval`-able**; 300/300 in the sample |
| gold containing a newline | **0 / 800** |
| `####` in any code/input/output | **absent** |
| unique gold in the sample | 235 / 300 |
| majority-class gold | `[]`, 13 items, rate **.0433** |
| output type distribution (300) | str 138, list 74, int 38, dict 26, bool 17, tuple 6, float 1 |
| code chars | median 133, max 278 |
| output chars | median 9, max 50 |

The output-type distribution is **recorded as provenance only**. It does **not**
re-stratify the sample, does not adjust the selection, and is not a
stratification axis for any analysis.

The two structural facts — no newline in any gold, `####` absent from the source
— are what make a single-line `#### <literal>` marker safe here. They are
asserted by the loader, not assumed.

### Label firewall

The loader emits the sample **twice from one selection**:

```
cruxeval_p4c_formal_blind.json   300 items, gold ABSENT   -> the RUNNER reads this
cruxeval_p4c_formal.json         300 items, gold PRESENT  -> the SCORER reads this
cruxeval_p4c_manifest.json       provenance + digests
```

The blind copy is built from an explicit field **whitelist** and then asserted
to carry no forbidden key. "The code does not access gold" is much weaker than
"gold is not reachable" — the P2 lesson.

**This is not a blind validation.** CRUXEval gold is public; P3 (GSM-Hard) was
the blind test and is CLOSED. What carries over is the other half of the
discipline: α is read from the frozen GSM8K record and never re-searched, and
the sample is frozen before any cell runs.

---

## 3. Experimental matrix

| model | α | role |
|---|---:|---|
| Llama3.1-8B | 0 | baseline |
| Llama3.1-8B | **−6** | GSM8K fixed workpoint — **MAIN** |
| Llama3.1-8B | −4 | neighbour dose / stability diagnostic |
| Llama3.1-8B | +4 | reverse dose diagnostic |
| Qwen2.5-7B | 0 | baseline |
| Qwen2.5-7B | **+8** | GSM8K fixed workpoint — **MAIN** |
| Qwen2.5-7B | +6 | neighbour dose / stability diagnostic |
| Qwen2.5-7B | −6 | reverse dose diagnostic |

### Primary statistical family — and only this

- llama `−6 vs 0`
- qwen `+8 vs 0`
- paired exact two-sided McNemar with discordant counts
- item-level paired bootstrap 95% CI (B=10000, seed 0)
- **Holm m=2**, judged only when both models are complete

If only one model completes, its contrast is reported as **single-model
exploratory transfer**: raw McNemar and the paired CI, Holm **WITHHELD**, the
raw p labelled unadjusted. Running Holm at m=1 under an m=2 label is the error
this rule prevents.

### Diagnostics — frozen in advance, reported always, outside Holm

`−4` / `+6` read whether the fixed workpoint sits in a locally stable region.
`+4` / `−6` read whether the direction ordering continues.

Binding constraints on all four:

- **outside the primary Holm family**; their p values are **unadjusted**;
- **they MUST NOT redefine the workpoint.** α stays read from the frozen GSM8K
  record. A diagnostic cell that happened to score higher would **not** become
  the workpoint — this protocol never searches doses on CRUXEval;
- **they describe the ordering of the sampled doses only.** Four points are not
  a dose-response curve: they cannot locate a peak, establish an inverted-U, or
  license calling any dose an overshoot point.

They are reported whether or not they are significant. Significance limits
claim strength; it does not license hiding a cell.

---

## 4. Prompt — frozen verbatim

Neutral, **No-CoT**, **no few-shot**. `Think step by step` is deliberately
absent: whether reasoning is externalised must remain the model's own behaviour
rather than an instruction, which is what makes any timing readout attributable.

```
[PYTHON]
{code}

assert f({input}) == ??
[/PYTHON]

Complete the assertion by predicting the output of executing the function.
End your response with exactly one line in the following format:
#### <Python literal>

Response:
```

The anchor is `Response: ` (with the trailing space), where prefill-only
steering lands. The template is sha256-pinned and fatal to edit.

**No stop string.** The prompt literally contains `####`, and HF `stop_strings`
matches anywhere in the output — the CGT failure, where invalid_rate jumped
0.02 → 0.11. Generation ends on natural EOS or the budget.

---

## 5. Parsing and scoring

- **Marker** `#### <Python literal>`; the parser takes the remainder of the
  marker's line, then removes two **decoding artifacts** (`p4c-amend-01`,
  added after the format-only preflight and before any formal cell): a
  trailing EOS token text, and a trailing `####` occurring **outside** a
  string literal. Neither changes an answer. A `####` **inside** a string
  literal is preserved — 5 of the 300 gold values contain `#`, and all five
  were verified to round-trip. Prose after the payload is **not** rescued:
  that is the model failing the frozen format, a result to report rather than
  a parser to widen.
- **FIRST match is MAIN** (matching GSM8K/GSM-Hard/BBH production, where
  `extract_gsm8k_answer` takes the first `####`).
- **LAST match is SENSITIVITY** — a tail-revision readout, never the headline.
- **No marker scores incorrect.** Denominator stays 300. **No rescue
  generation.**
- The extracted text is parsed with `ast.literal_eval`; a parse failure scores
  incorrect.
- Correctness is **equality of the parsed Python objects**, so semantically
  equal spellings both pass:

```python
[1,2] == [1, 2]        # both correct
```

Before scoring, the scorer verifies that **every gold in the sample parses**;
a gold that does not is a hard stop, not a silently skipped item.

### Why the official executor is NOT used

Official CRUXEval-O scoring is `exec(f"{code}\nassert {gold} == {candidate}")`.
That interpolates **model-generated text** into an executed expression. This
protocol does **not** do that on the server or on a normal local machine.

`ast.literal_eval` cannot execute code by construction, and — verified above —
**800/800 CRUXEval gold values are pure literals**, so on this benchmark it
agrees with the official comparison wherever the candidate is itself a literal.

Two consequences recorded in advance rather than discovered later:

1. **This is not the official execution-based `pass@1`.** A candidate that is a
   non-literal *expression* evaluating to the right value (`[1]*3`,
   `list(range(3))`) is scored **incorrect** here and would be scored correct
   officially. The rate of such candidates is reported per cell as a
   diagnostic, so the size of the gap is visible rather than assumed.
2. Reproducing the official executor would require a genuinely isolated,
   network-free, resource-limited container. **`reliability_guard` is not a
   security sandbox** and must not be described as one.

---

## 6. Generation configuration

| | |
|---|---|
| temperature | `0.0` (greedy) |
| max_new_tokens | `768` |
| batch_size | `24` |
| steering | prefill-only, `tail=1`, at the `Response: ` anchor |
| Llama band / mask | `11–20`, `nmd_0.5_11_20_8B.npy` |
| Qwen band / mask | `16–22`, `nmd_0.5_16_22_7B.npy` |
| CoT | **False**, hardcoded — no flag exists |
| few-shot | **False** — no exemplar path exists |

α=0 passes a **real all-zero matrix**, exactly as the GSM8K driver does. Hooks
register and the zero add executes; `steering_fires` reads 0 only because a
zero row is not counted as steered. This is not the pv6/PV10 "no hook at all"
path and must not be optimised into one, or α=0 would run through a different
execution path than the steered cells and stop being their baseline.

**Cells are NOT required to share a GPU.** `host`, `CUDA_VISIBLE_DEVICES` and
whether the model was sharded are recorded as provenance. Because bf16 greedy
output may vary across hardware, any contrast between cells that ran on
different devices is reported as a **cross-run pairing**, and pairing means
alignment by the frozen item order — never hardware identity.

### Preflight — format only

A small preflight may check, **before** the formal cells:

- whether the marker is produced at all;
- whether the literal parser works;
- whether the token budget is obviously being hit;
- whether `steering_fires` reads the expected `L × n × 1`.

**Preflight accuracy must not be viewed, and α must not be adjusted from it.**
The preflight runner computes and stores no correctness field.

---

## 7. No accuracy gate; hard stops are technical only

**There is no `[.20, .85]`-style accuracy gate.** Even a low baseline leaves the
300-item paired contrast reportable; a capability floor or ceiling is recorded
as a limitation on the reading, not used to cancel the test. This deliberately
removes the failure mode where a gate interval becomes adjustable after seeing
the data.

The following are hard stops, and each is checked in code:

1. sample content or order disagrees with the frozen digests;
2. the blind file carries a gold field;
3. `steering_fires` differs from `0` (α=0) or `L × n × 1` (α≠0);
4. the parser fails **systematically** — a cell producing no valid marker in
   any of its 300 items;
5. a cell's stored configuration (model, α, band, budget, batch size, protocol)
   disagrees with the frozen matrix;
6. a gold value in the sample does not `ast.literal_eval`.

A hard stop is a stop. It is not a licence to redesign the prompt, redefine the
parser, or re-tune the budget in response to what the output looked like.

---

## 8. Order of operations

1. freeze this protocol
2. `data_cruxeval.py` — download, assert digests, write blind + gold + manifest
3. format-only preflight (optional; **no accuracy**)
4. generate all eight cells with **no accuracy in the loop**
5. `eval_cruxeval.py` — the only script that reads gold; run once

Generation and scoring are separate scripts so that a generation run cannot
quietly become an accuracy run.

## 9. Post-result rule

Once scored, the P4c main analysis caliber is **CLOSED**: the prompt, parser,
marker, workpoints, matrix and statistical family may not be changed in light
of the result. Any further analysis is **exploratory** and must be labelled so.
