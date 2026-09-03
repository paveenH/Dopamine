# Explicit-CoT follow-up to P4 fixed-workpoint transfer

**Protocol id** `cot-transfer-followup-v0`
**Status** frozen before any CoT cell is generated for any of the three tasks
**Relation to P4 / P4b / P4c** This is **not** a fourth task and **not** a new
phase. It is a **post-hoc, exploratory follow-up** to the three already-run
No-CoT transfer tests (LogiQA 2.0 `logiqa2-p4-v0`, BBH `object_counting`
`bbh-p4b-v0`, CRUXEval-O `cruxeval-p4c-v0`), motivated by having seen those
results. **It does not replace, correct, rescale, or supersede any of the three
frozen No-CoT results.** Those stay CLOSED exactly as written in
`AdaDopamine_gsm8k.md` §5.7–5.9 (or wherever they are indexed) and in
`docs/p4_logiqa2_evaluation.json` / `docs/bbh_p4b_object_counting_result.json` /
`docs/p4c_cruxeval_evaluation.json`. Nothing in those files or in
`docs/PREREG_P4_LOGIQA2.md`, `docs/PREREG_P4B_BBH.md`, or
`docs/PREREG_P4C_CRUXEVAL.md` is edited by this document or by this follow-up.

---

## 0. What this is, and what it is not

**Is:** a same-everything-else check of whether adding one explicit CoT
instruction changes the outcome of fixed-workpoint transfer, on the same three
tasks, same items, same models, same steering, same budget, same batch size,
same parser, same marker, same scoring.

**Is not:**

- **Not a confirmatory replication of the No-CoT result.** It was designed
  *after* the No-CoT results were known, specifically because those results
  raised the question of whether an implicit-vs-explicit reasoning trajectory
  changes the transfer outcome. Calling it confirmatory would misstate its
  evidential status.
- **Not a re-run, correction, or reinterpretation of the No-CoT cells.** Every
  No-CoT number stands. This follow-up adds a disjoint set of cells (CoT
  cells) and compares them to their own CoT baseline (`α=0, CoT`), never to
  the No-CoT baseline as if it were the same condition.
- **Not evidence about whether the model "reasons internally."** Explicit CoT
  changes the *visible* generation trajectory. It does not license any claim
  about hidden/implicit computation in the No-CoT condition. A CoT cell
  outperforming its own baseline says the externalized trajectory helped
  (or the steering interacted with it differently); it says nothing about
  what happened without externalization.
- **Not a single-mechanism attribution.** CoT simultaneously (a) inserts an
  explicit self-generated reasoning trajectory into the context the model
  conditions on for every subsequent token, (b) lengthens the generation
  before the final marker, which changes how much decode-time the model has
  to work with under the same token budget, and (c) changes how far the
    prefill-only steering perturbation has propagated by the time the model
  reaches its answer (steering is injected once, at the prompt's last token,
  before any of the added CoT text is generated). **A difference between the
  CoT and No-CoT transfer effect can be produced by any of these, in any
  combination, and this design cannot separate them.** Do not write "CoT
  improved transfer because it let the model reason more" — write "the CoT
  condition showed a different transfer effect than the No-CoT condition",
  and stop there unless a dedicated follow-up isolates the mechanism.

Frozen framing sentence, to be used verbatim wherever this follow-up is cited:

> Holding data, model, steering, budget, batch size, parser, and scoring fixed,
> does an explicit CoT instruction change GSM8K-derived fixed-workpoint
> transfer? This is a post-hoc exploratory follow-up to the No-CoT result, not
> a replication of it.

---

## 1. Cells

Eight cells per task (24 cells total across the three tasks), all **CoT**.

### Llama3.1-8B (band `11-20`, L=9, mask `nmd_0.5_11_20_8B.npy`)

| α | role |
|---:|---|
| 0 | CoT baseline |
| **−6** | frozen GSM8K workpoint — **MAIN comparison** |
| −4 | neighbour diagnostic |
| +4 | reverse diagnostic |

### Qwen2.5-7B (band `16-22`, L=6, mask `nmd_0.5_16_22_7B.npy`)

| α | role |
|---:|---|
| 0 | CoT baseline |
| **+8** | frozen GSM8K workpoint — **MAIN comparison** |
| +6 | neighbour diagnostic |
| −6 | reverse diagnostic |

**α is read from the already-frozen GSM8K record, byte-identical to the
doses used in the No-CoT P4/P4b/P4c cells.** It is **not** re-selected,
re-searched, or re-derived from anything observed in this follow-up or in the
No-CoT CoT-free results. The neighbour and reverse doses are likewise carried
over unchanged from the corresponding No-CoT protocol (P4c already has all
four per model; P4/P4b are extended here to the same four-point matrix so all
three tasks share one CoT matrix shape).

---

## 2. Only the prompt condition changes

Every other axis of configuration is **inherited verbatim** from the task's
own No-CoT protocol:

- same model checkpoint / revision;
- same formal 300 (LogiQA, CRUXEval-O) / 250 (BBH `object_counting`) items,
  same order, same `sample_id`s, same content digest;
- same blind/gold firewall (generation reads the label-free file; only the
  eval script reads gold);
- same steering mask, band, and injection mechanism (prefill-only,
  `tail_len=1`, at the same anchor token);
- same `temperature=0.0` (greedy);
- same `max_new_tokens` **per task**: LogiQA 2.0 `1024` (the P4 stage-1 frozen
  budget), BBH `object_counting` `768`, CRUXEval-O `768`;
- same `batch_size` **per task**: LogiQA 2.0 `8`, BBH `object_counting` `24`,
  CRUXEval-O `24`;
- same parser (`ANSWER_RE` / `extract_gsm8k_answer` / the `####`+literal
  parser), imported or copied verbatim, never rewritten;
- same MAIN/sensitivity convention per task: LogiQA 2.0 MAIN=LAST,
  sensitivity=FIRST; BBH and CRUXEval-O MAIN=FIRST, sensitivity=LAST — **the
  conventions themselves are not revisited by this follow-up**, only inherited;
- same final marker (`Final answer: X` / `#### <integer>` / `#### <Python
  literal>`) and same anchor (`Response: ` / `Answer: ` / `Response: `);
- same normalization and correctness rule (`normalize_gsm8k`,
  `ast.literal_eval` + Python object equality, exact-letter match);
- same no-marker / non-literal handling (scored incorrect, denominator stays
  fixed, **no rescue generation**);
- **no stop strings** added on any task (CRUXEval-O's prompt still literally
  contains `####`, so a stop string would still risk the CGT-style truncation
  failure regardless of CoT);
- **the budget, parser, and prompt are not changed in response to any
  preflight or formal output** — the same discipline as every P4 sub-protocol.

**The only planned change is inserting one line, the project's existing GSM8K
CoT cue, into each task's prompt**, in the position specified in §3.

Cells are **not** required to share a GPU. `host` and `CUDA_VISIBLE_DEVICES`
are recorded as provenance on every generation file, exactly as in P4c. A
model's own cells for one task still share a card (bf16 greedy is not
byte-reproducible across GPUs, and the primary/diagnostic contrasts are paired
per item); the three tasks and the two models may run on any number of cards.
Statistical pairing is by `sample_id`, never by hardware.

---

## 3. CoT prompt

The cue is the project's existing GSM8K/MATH CoT cue, reused **verbatim**:

```
Let's think step by step.
```

(one line, trailing newline, exactly as it appears in every `cot=True` branch
of `template.py`, e.g. `build_gsm8k_default_suite`).

**One cue, reused across all three tasks — no task gets its own wording.**
The cue is inserted immediately **before** the task's existing final-format
instruction line(s), and the original final marker instruction, anchor, and
everything before the insertion point are otherwise byte-identical to the
No-CoT template.

### 3.1 LogiQA 2.0 (`logiqa2-p4-v0` prompt + CoT cue)

```
{passage}

{question}
A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}

Let's think step by step.
End your response with 'Final answer: X' where X is A, B, C, or D.

Response: 
```

sha256(template, first 16 hex) = `275ea7768b2f94ec`

### 3.2 BBH `object_counting` (GSM8K default-suite CoT template, reused
unmodified — this is not a new string, it is
`select_templates_gsm8k(suite="default", cot=True, wording="plain")["neutral"]`,
already implemented in `template.py` and used by the frozen GSM8K/MATH CoT
lines)

```
Solve the following math problem.
Question: {context}
Let's think step by step.
Provide your final numeric answer after '####'.
Answer: 
```

sha256(template, first 16 hex) = `1ccbb24cc6938f09`

### 3.3 CRUXEval-O (`cruxeval-p4c-v0` prompt + CoT cue)

```
[PYTHON]
{code}

assert f({input}) == ??
[/PYTHON]

Let's think step by step.
Complete the assertion by predicting the output of executing the function.
End your response with exactly one line in the following format:
#### <Python literal>

Response: 
```

sha256(template, first 16 hex) = `d3ec2bd57b927b4a`

### 3.4 What is unchanged, verified by construction

- **No few-shot.** No exemplar path exists in any of the three runners; this
  follow-up does not add one.
- **No worked reasoning example.** The cue is a bare instruction, not an
  illustrated example.
- **No role or persona.** All three prompts stay in the neutral/no-persona
  voice already used by the No-CoT protocols.
- **Question text and answer format are byte-identical** to the No-CoT
  templates except for the one inserted line. The final marker line, the
  anchor line, and the anchor's trailing content (`Response: ` / `Answer: `)
  are copied character-for-character from the frozen No-CoT template.
- **The injection token is therefore unchanged by construction**: prefill-only
  steering with `tail_len=1` injects into the last prompt token, and the last
  prompt token is the tail of the anchor string, which is byte-identical
  between the CoT and No-CoT templates. Inserting a line earlier in the prompt
  shifts token *positions* but not the *identity* of the final token, so the
  same token id at the same relative position (the anchor's trailing token) is
  the injection site in both conditions.
- This must still be **verified, not assumed**, on the server before any
  formal generation, exactly as `check_signal_qwen.py` / `check_cgt_seq_qwen.py`
  verify injection tokens for other protocols. Each of the three CoT runners
  below prints the last-token id (and its decoded string) for both the CoT and
  the corresponding No-CoT template on the actual tokenizer at `--model_dir`,
  and **asserts they are equal** before generating anything. A mismatch is a
  hard stop.

---

## 4. Statistical analysis

### 4.1 Primary exploratory family (6 comparisons, Holm m=6)

For each of the three tasks and each of the two models, compare that task's
**frozen workpoint** against **that same task's own CoT `α=0`** (never against
the No-CoT `α=0`):

| # | task | model | comparison |
|---|---|---|---|
| 1 | LogiQA 2.0 | llama3 | CoT `−6` vs CoT `0` |
| 2 | LogiQA 2.0 | qwen2.5 | CoT `+8` vs CoT `0` |
| 3 | BBH object_counting | llama3 | CoT `−6` vs CoT `0` |
| 4 | BBH object_counting | qwen2.5 | CoT `+8` vs CoT `0` |
| 5 | CRUXEval-O | llama3 | CoT `−6` vs CoT `0` |
| 6 | CRUXEval-O | qwen2.5 | CoT `+8` vs CoT `0` |

Each comparison: paired accuracy difference (MAIN parsing, per task's own
convention), exact two-sided paired McNemar with discordant counts, item-level
paired bootstrap 95% CI (B=10000, seed 0). **Holm correction over all six,
m=6.** This is a **new, independent statistical family** and is **never
pooled** with any P3/P4/P4b/P4c Holm family (each of those stays its own
m=2 family, judged on its own cells).

Sensitivity accuracy (task's own non-MAIN parsing) and all morphological
diagnostics (no-marker rate, degenerate/loop rate, answer-first rate,
multi-marker rate, generation length, truncation rate) are reported for every
CoT cell, always outside Holm, and are exploratory: they are outcomes of α, so
stratifying accuracy on them is post-treatment stratification —
consistent-with evidence, never mediation. The neighbour (`−4`/`+6`) and
reverse (`+4`/`−6`) diagnostics are reported per task exactly as in the
matching No-CoT protocol — outside Holm, unadjusted p, and they **must not**
redefine the workpoint.

If only one model has both cells for a given task, that task's row is reported
as **single-model exploratory**, Holm withheld for that row's contribution
(the remaining rows still form their own m≤6 family over whatever is
complete — report the realized m explicitly), matching the P4/P4b/P4c
convention for a partial family.

### 4.2 CoT × steering interaction (descriptive, outside Holm)

For each task and model where both the CoT and the No-CoT results exist:

```
DiD =
  [Acc(CoT, workpoint) − Acc(CoT, 0)]
  −
  [Acc(No-CoT, workpoint) − Acc(No-CoT, 0)]
```

computed as a **question-level paired bootstrap** (B=10000, seed 0; each
bootstrap draw resamples a `sample_id` and reads all four accuracies —
CoT-workpoint, CoT-0, No-CoT-workpoint, No-CoT-0 — for that same item, exactly
mirroring the four-cell joint-paired procedure already used for the GSM8K
No-CoT-vs-CoT interaction in `AdaDopamine_gsm8k.md` §3.5.1). Reported as a
point estimate with a 95% CI, **never as a significance test** and **never
pooled with §4.1's Holm family** — it answers a different question (does the
steering effect depend on the CoT condition) than §4.1 does (does the
workpoint beat its own baseline).

**A CI containing 0 is reported as "not detected", never as an equivalence
claim** (the same 口径 rule used for the GSM8K CoT interaction and the P3-supp
interaction). Given the three confounds named in §0, a nonzero DiD is reported
as "the steering effect differs between the CoT and No-CoT conditions",
**never** attributed to reasoning length, propagation depth, or
self-conditioning individually.

---

## 5. What this follow-up does not permit

- Rewriting, rescaling, or reinterpreting any frozen No-CoT P4/P4b/P4c number.
- Re-selecting, re-deriving, or tuning α from anything observed here.
- Changing the budget, batch size, parser, marker, or scoring convention in
  response to CoT preflight or formal output.
- Adding a fourth task, a second CoT cue, few-shot exemplars, worked-example
  reasoning, or role/persona framing.
- Pooling this follow-up's Holm family with any other protocol's Holm family.
- Claiming this design isolates trajectory length, self-conditioning, or
  steering-propagation depth as the mechanism behind any observed CoT-vs-No-CoT
  difference.
- Calling this a blind validation (LogiQA 2.0 and BBH gold are public;
  CRUXEval-O gold is public; the only blind test in this project's P-line is
  P3/GSM-Hard, which stays closed and untouched).

---

## 6. Order of operations (locked)

1. Freeze this document.
2. Implement the three CoT runners (forks of `get_answer_logiqa2.py` /
   `get_answer_bbh_numeric.py` / `get_answer_cruxeval.py` with the CoT prompt
   substituted and the injection-token equality assertion added) and the
   corresponding launchers. Generation and scoring stay separate scripts, as
   in every P4 sub-protocol.
3. Run each task's CoT `α=0` cell first per model; verify the injection-token
   assertion output and marker/scorability diagnostics before running the
   remaining three doses.
4. Generate all remaining CoT cells. No accuracy is computed during
   generation.
5. Score with a dedicated eval script per task (or a shared one covering all
   three — implementation detail, not a protocol constraint) that reads
   **only** the CoT generation files plus the task's existing gold file, and
   computes §4.1 and §4.2.
6. Report CoT and No-CoT results side by side, never merged into one number,
   with the framing sentence in §0 attached.
