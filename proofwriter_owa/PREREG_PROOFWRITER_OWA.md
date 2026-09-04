# ProofWriter OWA multi-hop — task-specific workpoint exploration

**Protocol id** `proofwriter-owa-v0`
**Status** **SUSPENDED 2026-09-04 at the alpha=0 feasibility gate.** Design and code
are complete and an alpha=0 preflight was run; NO non-zero alpha cell has ever been
run, and none may be until the gate below is cleared. See section 0.1.

## 0.1 SUSPENSION (2026-09-04, human decision)

**Do not run the canary, the pilot, or any alpha sweep.** The alpha=0 preflight shows
the evaluation interface is not usable in this configuration:

| model | parse_fail | trunc | overall acc |
|---|---|---|---|
| llama3 | 90.0% | 100% | 1/30 = 3.33% |
| qwen2.5 | 73.3% | -- | 1/30 = 3.33% |

At this parse-failure rate a dose search would compare output-format failure and
loop degeneration against each other, not steering.

**Two things this does NOT establish, and both are easy to get wrong:**

1. It does NOT show that steering is ineffective on ProofWriter. No non-zero alpha
   has been run. The only supportable statement is that **the explicit-CoT
   configuration did not pass feasibility**.
2. The preflight's `No effective workpoint` verdict is **VACUOUS and must not be
   cited**. A workpoint verdict computed over a single dose has nothing to compare
   against; the string is an artifact of running the evaluator on one cell.

**Resumption condition.** Results are sealed as-is. If this line is revived, it starts
with a SMALL balanced **3-shot** prompt feasibility test -- not a sweep. If parse
failure is still high there, the task is formally TERMINATED rather than re-tuned
again. (The v1 prompt revision was already one such response to a preflight; a second
round of prompt tuning against the same symptom would be fitting the interface to the
observation.)
**Relation to the GSM8K/MATH/BBH/CRUXEval/LogiQA fixed-workpoint transfer line**
NONE. This is not a transfer test and does not read the frozen GSM8K workpoint
(llama `-6`, qwen `+8`). It is ProofWriter's **own** 4-point dose exploration
(llama `{-6,-4,0,+4}`, qwen `{-6,0,+6,+8}`), asking whether a task-specific
workpoint exists on ProofWriter OWA multi-hop deduction at all.

Everything below is implemented, under `proofwriter_owa/`, fully isolated from
the existing GSM8K/MATH/BBH/CRUXEval/LogiQA/ZebraLogic pipelines. No existing
runner, doc, or result file was modified.

---

## 1. Task and data

**ProofWriter, Open-World Assumption (OWA), Task 1 (answer prediction / proof
generation), official release V2020.12.3.** Two depth buckets only:

- **OWA D3** — theories requiring up to depth-3 reasoning
- **OWA D5** — theories requiring up to depth-5 reasoning

Labels are **True / False / Unknown only** (OWA's three-valued semantics). No
CWA, no D0/D1/D2, no abduction, no implication-enumeration variant, no
Birds-Electricity, no ParaRules. The loader's dataset-shape assertions
(`proofwriter_owa/data_proofwriter_owa.py`) hard-stop on anything else.

**OWA semantics, stated explicitly because it is easy to get backwards:**
query is provable → **True**; the EXPLICIT NEGATION of the query is provable →
**False**; NEITHER is provable → **Unknown**. Failing to prove the query does
**not** by itself mean False — that is the CWA rule and does not apply here.

### 1.1 Source and verification policy

Primary source: the **official AI2 release archive**,
`proofwriter-dataset-V2020.12.3.zip`, published at the canonical AI2/Aristo
public-data location documented on the ProofWriter project page
(`allenai.org/data/proofwriter`; archive host
`aristo-data-public.s3.amazonaws.com`). The loader downloads this archive
directly by default. **No third-party HF mirror is used unless it has first
been verified byte-for-byte equivalent to the official files** (same
`sample_id`/`id`, `theory`, `question`, `answer`, `proof` text; verification
script and hashes recorded before any mirror is trusted). None of the HF
mirrors surveyed while designing this (`D3xter1922/proofwriter-dataset`,
`theoxo/proofwriter-deduction-balanced`, `tasksource/proofwriter`,
`hitachi-nlp/proofwriter_processed_OWA`, `renma/ProofWriter`, `smoorsmith/*`)
is used as a silent substitute — `theoxo`'s is a relabeled/rebalanced 300-row
subset (`Unknown`→`Uncertain`, not the raw D3/D5 files) and would silently
violate "labels are True/False/Unknown only" and the "no selecting samples by
model result" rule if used unexamined; `tasksource/proofwriter`'s schema
(`id, maxD, NFact, NRule, theory, question, answer, QDep, QLen, allProofs,
config`) looks like a faithful re-export of the official fields but has not
been hash-verified against the original zip and MUST NOT be treated as
equivalent until `--verify_against_official` (documented in §1.3) confirms it
on the actual server.

**Archive layout, as documented by the official release** (verified against
the schema, not assumed — the loader asserts every one of these at parse
time): a folder per rule-set (`birds-electricity`, `AttNegNoRule`, `AttRulesNoNeg`
etc.) is **excluded** entirely; the two used are `depth-3` (`OWA` subfolder)
and `depth-5` (`OWA` subfolder — release naming is `depth-3ext` / `depth-5`
etc. per depth family; the loader's schema check reports the ACTUAL directory
names found and refuses to silently reinterpret them). Each depth folder
contains `meta-{train,dev,test}.jsonl` — one JSON object per **theory**
(`id`, `triples`, `rules`, `questions`), where `questions` is a dict keyed by
`Q<n>` with each question object carrying `question` (query text), `answer`
(bool or `"Unknown"`), `QDep` (int, the proof/query depth used for stratifying
this dataset — see §1.2 for what this field actually means), and `proofs`
(text, when `answer` is not Unknown). This is the schema the loader parses;
**if the actual on-disk schema differs, the loader reports the real field
names and refuses to guess** (see the "fail closed on schema" rule in §1.4).

### 1.2 What "depth" means here — reported, not assumed

The user's design brief calls for D3 True/False items with `QDep == 3` and D5
items with `QDep == 5`, and flags this explicitly as an assumption to verify
rather than silently reinterpret. `QDep` in the official ProofWriter release
is the **depth of the shortest proof for that specific question** (an
integer, present on every question object, including some Unknown ones where
a closed-world variant would have found a proof at that depth — but under OWA
an "Unknown" item generally carries no closed proof and its `QDep` semantics
differ; the loader reports the empirical distribution of `QDep` split by label
before any manifest is built, exactly as requested in §2). The loader does
**not** assume `QDep == 3` on every D3-file question, or `QDep == 5` on every
D5-file question — the D3/D5 split is a property of the FILE (which theories
were generated for), not a per-question guarantee that every question in that
file has query depth exactly 3 or 5. Both readings are reported side by side:
"depth of the file's theory family" (D3 vs D5, from directory of origin) and
"depth of this specific question's shortest proof" (`QDep` field, empirical
distribution reported in §2's output). Manifest selection (§3) uses the
**per-question `QDep`** field for the True/False depth-3 / depth-5
requirement, not the file-of-origin alone, and states this in the loader
output.

**If the archive's actual field for a question's proof depth is not named
`QDep`, or does not exist at the object level the loader expects, the loader
prints the real schema and hard-stops rather than inventing a mapping.**

### 1.3 Revision / integrity pinning

- Archive is fetched once and its **SHA256 is recorded** in the manifest and
  compared on every subsequent invocation (`--archive_sha256`, set after the
  first successful download; a mismatch on a later run is a hard stop, not a
  silent re-download).
- If `--hf_mirror <repo_id>` is passed (opt-in, off by default),
  `--verify_against_official` must ALSO be passed and must succeed (exact
  string match on `theory`/`question`/`answer`/`proof` for a sampled or full
  set of IDs against the official archive already on disk) before the mirror
  rows are used for anything beyond that verification. There is no code path
  that silently falls back to an unverified mirror.

### 1.4 Fail-closed schema handling

The loader asserts, and hard-stops (never silently reinterprets) on:
- expected top-level archive structure (folder names for `depth-3`/`depth-5`
  OWA test splits actually exist)
- every parsed question object carries `question`, `answer` ∈
  {`True`, `False`, `Unknown`} (after AI2's own string/bool normalization —
  reported, not assumed: the loader prints the raw label token distribution
  before any normalization it applies)
- `QDep` is present and integer-valued on every question object it will use
  for depth stratification; if this is false for a nontrivial fraction, the
  loader reports the fraction and the run stops rather than dropping items
  silently
- no duplicate `(theory_id, question_key)` pair
- theory text (`triples` + `rules`, rendered as English sentences per the
  release's own sentence templates already baked into `question`/`theory`
  fields — the loader uses the plain-English rendering the release ships, not
  a re-templated one, and reports if plain-English rendering is not present)

## 2. Required first-run report (before any manifest is built)

For D3 and D5 separately: theory count, question count, True/False/Unknown
counts, gold `QDep` distribution (a histogram, not just a mean), missing-ID
count, duplicate-ID count, and unparseable-gold count. **No sample is chosen
based on model output** — this report exists purely to characterize the raw
data before selection.

## 3. The 300-item manifest

- 300 items total: **D3 = 150, D5 = 150**.
- Within each dataset, **True/False/Unknown as close to 50/50/50 as the data
  allows.**
  - True and False items are preferentially the dataset's genuine multi-hop
    items: **D3 → `QDep == 3`, D5 → `QDep == 5`** (per §1.2, verified against
    the actual field semantics before this rule is applied as written; if the
    field means something else the manifest-building rule is reported and
    adjusted, not silently kept).
  - Unknown items typically have no single gold proof depth under OWA. They
    are matched to the True/False pool on **dataset, theory size (#triples +
    #rules), and, where available, a QDep-like closure-depth field** — never
    described as "have a 3-step / 5-step gold proof". The manifest's Unknown
    rows carry an explicit `depth_match_note` field documenting how each was
    matched, so this limitation is visible per-row, not just in a README.
  - If any bucket cannot reach 50 exactly, the loader reports the shortfall
    and fills from the nearest available depth/label pool, recording the
    deviation in the manifest rather than silently forcing 50/50/50.
- Selection is by **frozen salted hash** over the item's official ID (analog
  to every other loader in this repo — `sha256(SALT:official_id:theory:
  question)`), never `hash()` (process-salted) and never dataset order.
- **Fixed**: random seed (`0`, recorded), sample IDs, order, dataset
  archive SHA256, and the manifest's own SHA256. All models and all α use the
  identical 300-item manifest — there is exactly one manifest file, read by
  every generation cell.
- The manifest is emitted twice from one selection, matching the P3/P4/P4b/P4c
  label-firewall pattern: `manifest_blind.json` (no `answer`/`proof` field,
  whitelist-built, asserted label-free — read by the generator) and
  `manifest_gold.json` (with gold — read only by the scorer/commitment audit).

## 4. Prompt

Explicit CoT, **not claimed to be an official ProofWriter LLM prompt** — it is
this project's own construction, following the same "own construction,
labelled as such" convention as every other CoT runner in this repo
(`get_answer_bbh_numeric_cot.py`, GSM8K's `Let's think step by step.`).

- Theory and question text are the **official strings, byte-unchanged** (no
  paraphrase, no re-templating).
- The model is asked to (1) reason step by step about whether the query or its
  explicit negation follows from the facts and rules, then (2) end with
  exactly one line of the form `Answer: True`, `Answer: False`, or
  `Answer: Unknown`.
- The OWA definition (query provable → True; negation provable → False;
  neither → Unknown) is stated in the prompt in those terms.
- Optional few-shot exemplars, if used, are frozen **train-split only**, never
  from test. Default is zero-shot; `--n_shot` is available but defaults to 0,
  and the exemplar pool (if `--n_shot > 0` is ever used) is drawn from the
  official train split by the same frozen-hash selection, cached once so
  every α cell of every model sees byte-identical exemplars.
- Every α cell of one model uses byte-identical prompts (same template
  string, same exemplars if any).

## 5. Generation configuration

| | Llama-3.1-8B-Instruct | Qwen2.5-7B-Instruct |
|---|---|---|
| layer band | `[11,20)` | `[16,22)` |
| α | `{-6,-4,0,+4}` | `{-6,0,+6,+8}` |

Common: prefill-only, `tail=1`, neutral (no persona), bare-string (no chat
template), greedy (`temperature=0`), `batch_size=8`, `max_new_tokens=1024`
default (see §6 — this was raised from an original default of 768, and is now
FROZEN at 1024 for both models). All α of one model use the **same batched
`vc.regenerate()` call path** — α=0 is not silently routed through
`vc.generate()` or any other path. On OOM the run **stops and reports**; it
does not silently fall back to `batch_size=1`. One model's whole α curve runs
on one machine; a cross-GPU comparison must first pass the outcome-level
canary (§7).

## 6. Token budget escalation rule (RESOLVED 2026-09-04)

Original default was `max_new_tokens=768`. The α=0 preflight (§7, 30 items)
measured **100% truncation on llama3** (30/30) and **16.7% on qwen2.5**
(5/30). Manual inspection of five llama3 samples found the truncation is
**not uniform in cause**: some are genuine multi-hop reasoning that simply
needed more room, at least one violated the "last line only" format
instruction mid-generation (wrote `Answer: True` mid-text, then continued
into unrelated Python code), and at least one was a stable **degenerate
repetition loop** (`# Corrected output format.` repeated dozens of times) —
a known, stable llama3 behavior on this task, not a bug this budget change is
meant to cure.

**Decision (human, 2026-09-04): raise `max_new_tokens` to `1024` for BOTH
models, uniformly, ONE TIME, and do not chase it further upward.**
Consequences, all explicit:

- Loop and truncation rates continue to be measured and reported in every
  result (`eval_proofwriter_owa.py`'s `loop_rate` / `truncation_rate` /
  `is_true_last_line` diagnostics), but they are **not a preflight gate** —
  llama3's degenerate-loop tendency on this task is treated as a known,
  stable phenomenon to characterize, not an error condition that blocks the
  sweep.
- Scoring is unaffected: the frozen parser (`answer_parser.py`, last strict
  `Answer:` marker, fail-closed) is applied exactly as designed. A sample
  with a valid marker is scored normally regardless of truncation or looping
  elsewhere in the text; a sample with no valid marker (including one
  truncated before ever emitting one) is a parse failure / no-answer, exactly
  per the existing rules in §8 — no special-casing for llama3.
  **`is_true_last_line=false` does not disqualify a scored answer** — a
  marker followed by trailing text (a format violation, e.g. sample_0's
  `Answer: True` followed by unrelated Python code) is still scored on that
  marker per the existing "prefer the last strict marker" rule; the
  diagnostic only records that the model did not follow the literal "last
  line" instruction, and this is reported for audit, not used to drop the
  sample.
- **This is a ONE-TIME raise, enforced in code, not just by convention.**
  `get_answer_proofwriter_owa.py` hard-rejects any `--max_new_tokens` value
  other than the frozen `MAX_NEW_TOKENS_FROZEN = 1024` constant — the CLI
  flag exists but is not a tunable knob. If material truncation is observed
  again at `1024`, the run stops and is reported; raising the budget further
  requires deliberately editing `MAX_NEW_TOKENS_FROZEN` in the generator
  source AND this section, not passing a different CLI value.
- Any report of results **must state explicitly**: llama3 shows a high rate
  of degenerate repetition loops on this task, so its generation-length and
  commitment-timing diagnostics (§9, e.g. `pre_answer_reasoning_tokens`,
  `first_answer_marker_pos`) carry weaker interpretive weight on this model;
  its parseable-answer accuracy remains usable and is scored identically to
  qwen2.5.

## 7. Preflight, pilot, canary — order and gates

1. **α=0 preflight** (per model): 15 D3 + 15 D5 = 30 items, all three labels
   covered. Checks: loader, prompt render, gold lookup, final-label parser,
   injection position, padding, fire count, truncation, result write-out.
   α=0 steering fire count **must be exactly 0**.
2. **Canary** (per model, non-zero α): the same 30-item preflight set at one
   representative non-zero α. `steering_fires` must equal exactly
   `L × n_samples × tail(=1)`, matching every other runner in this repo.
   If two GPUs are ever used across a model's own doses (discouraged by §5
   but the harness cannot forbid it outright), an **outcome-level canary**
   (documented alongside — same 30 items, same α, both cards) must be run and
   compared before treating results as comparable.
3. **α=0 pilot**: 150 items per model (D3=75, D5=75), same manifest subset
   (first 75 of each dataset in manifest order — frozen, not resampled).
   Reports overall / D3 / D5 / per-label accuracy and flags ceiling risk. The
   pilot **only reports** — it must not be used to revise prompt, sample, α,
   or the scoring metric. It is a separate shell stage that a human reviews
   before the formal sweep is launched.
4. **Formal sweep** (llama, then qwen — two separate stages): the full 300
   items × 4 α, one shell stage per model, launched manually after the pilot
   is reviewed.
5. **Analyze**: scoring + commitment-extractor + statistics, a separate stage.

## 8. Scoring

**Primary metric: Exact Label Accuracy** against official True/False/Unknown
gold. Also reported: D3 accuracy, D5 accuracy, per-label accuracy, parse
failure rate, no-answer rate, invalid/multiple-final-answer rate, loop rate,
truncation rate, generation token length.

**Parser** (`proofwriter_owa/answer_parser.py`):
- Parses only the generated continuation, never the input context.
- Prefers the **last** valid `Answer:` marker.
- Does not treat an ordinary "true"/"false"/"unknown" word inside the
  reasoning body as a final answer.
- **Fail-closed**: if there is no unique legal final answer, it is a parse
  failure (scored as incorrect for accuracy, and reported separately).
- No LLM judge anywhere in this pipeline.

Gold proofs are available for stratification/manual audit only. **No
string-exact match against a generated CoT is used as a proof-correctness
metric**, and no claim of "the generated proof is logically correct" is made
anywhere — there is no reliable logic verifier in this pipeline.

## 9. Commitment extractor (frozen BEFORE any non-zero-α result is examined)

`proofwriter_owa/commitment.py`:
- `answer_first_rate`
- `first_answer_marker_pos` (normalized position of the first valid `Answer:`
  marker in the continuation)
- `pre_answer_reasoning_chars`
- `pre_answer_reasoning_tokens` (tokenizer-based; NaN/omitted if a tokenizer
  is unavailable — never estimated by a chars/4 heuristic)
- `reason_before_answer_rate`
- `first_final_label_agreement` (does the FIRST valid marker's label match the
  LAST valid marker's label)
- `label_revision_rate` (1 − `first_final_label_agreement`, reported
  separately for clarity in tables)
- `multiple_answer_marker_rate`

Rules, matching the CLAUDE.md-wide convention for this whole project:
- Metrics analyze only the generated continuation.
- They are keyed on the explicit `Answer:` marker, not on informal
  true/false/unknown words in the reasoning body.
- At least 30 α=0 outputs are manually inspected; detector version and known
  error modes are recorded in `proofwriter_owa/results/`.
- These are **descriptive co-occurrence statistics**. They are never reported
  as causal mediation evidence for an accuracy effect.

## 10. Statistics

- Per model: each non-zero α vs that model's own α=0. Exact two-sided
  McNemar on label-accuracy agreement, Holm correction with **m=3 per
  model** (three non-zero α).
- Report: accuracy difference (pp), discordant counts, raw p, Holm-adjusted p.
- D3/D5 and per-label breakdowns are reported in full but are **exploratory
  subgroups by default**, not pooled into the primary Holm family.
- If confidence intervals are reported, they are **question-paired
  bootstrap** (same convention as every fixed-workpoint-transfer script in
  this repo).
- No dropping of unfavorable depths or labels.

## 11. Workpoint reporting

- Discrete argmax over the sampled α (ties broken toward smaller `|α|`).
- Near-optimal region defined **only over the actually sampled α points**,
  reported separately from Holm significance.
- **If no non-zero α passes Holm against that model's own α=0, no effective
  workpoint is declared.** Frozen sentence, verbatim:
  > No effective ProofWriter OWA workpoint was detected in the sampled dose set.
- If a commitment metric moves but accuracy does not, the only licensed
  conclusion is: submission behavior changed, but it did not convert into a
  task benefit. Never phrased as steering "improving" accuracy in that case.

## 12. File layout

```
proofwriter_owa/
  PREREG_PROOFWRITER_OWA.md          this file
  data_proofwriter_owa.py            loader: download, parse, report, manifest
  prompt.py                          CoT prompt builder (own construction)
  answer_parser.py                   final-label parser (fail-closed)
  commitment.py                      frozen commitment extractor
  scoring.py                         accuracy + McNemar + Holm + bootstrap
  get_answer_proofwriter_owa.py      generation (label-free input)
  eval_proofwriter_owa.py            scoring + commitment + stats (reads gold)
  results/                           local JSON outputs land here
  tests/                             local, no-GPU unit tests
run_proofwriter_owa.sh               7-stage launcher
```

No existing GSM8K/MATH/BBH/CRUXEval/LogiQA/ZebraLogic runner, no CLAUDE.md, no
AdaDopamine_gsm8k.md, and no existing result file was modified for this line.
