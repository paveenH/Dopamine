# Pre-registration: ZebraLogic-Easy targeted four-point workpoint exploration

Protocol tag: `zebralogic-easy-v0`. Frozen at commit `4f8e7ef1da4a16f7d4b44ffe84222b9d7ae7d94b`
(2026-09-04), before any generation cell exists.

## 0. What this is, and what it is not

This is a **task-specific four-point workpoint exploration on ZebraLogic-Easy**,
under explicit CoT. It is:

- **NOT** a GSM8K fixed-workpoint transfer test (P3/P3-supp/P4/P4b/P4c). No α is
  read from the frozen GSM8K record here.
- **NOT** a full dose-response curve. Four points per model, sampled in advance,
  is all that exists. The only claims this line can support are: the observed
  argmax among the four sampled points, and an **observed near-optimal region**
  (points statistically indistinguishable from that argmax). It can never claim
  a continuous optimum, a peak, or an inverted-U — those need many more doses.
- **NOT** a blind validation. ZebraLogic-Easy's public split has withheld gold
  (see §2); the **private** gold split (`allenai/ZebraLogicBench-private`) is
  reachable only with per-account HF consent, which this environment does not
  currently have. Nothing here is "sealed" in the P3 sense — it is ordinary
  gated access, not a designed blind protocol.

This work is entirely isolated: new code lives under `zebralogic/`, new results
under a ZebraLogic-only output tree, and touches no existing runner, mask,
dataset loader, or document (`AdaDopamine_gsm8k.md`, `CLAUDE.md` untouched).

## 1. Data

- Dataset: `allenai/ZebraLogicBench`, config `grid_mode`, split `test`.
- Revision pinned: `2f94a445d7079f20146f5443e2606049de8543e0` (verified HEAD of
  `main` as of 2026-09-03; a full 40-hex commit SHA, never a branch name).
- The public `test` split's `solution` field is **entirely blank** (every cell
  is the literal string `"___"`) for all 1000 rows, verified by exhaustive scan.
  This is the official leaderboard submission split; it is not itself scorable.
- Real gold lives in `allenai/ZebraLogicBench-private`, config `grid_mode`,
  split `test`, same 1000 `id`s, **gated: "auto"** on the HF hub — requires a
  human to click through HF's per-repo access-request flow before any
  `HF_TOKEN` can download it. This is the exact mechanism the official
  `WildEval/ZeroEval` scorer (`src/evaluation/zebra_grid_eval.py`) uses:
  `load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")`,
  keyed by `id`.
- **Sample: exactly the "easy" tier as defined VERBATIM by the official
  `WildEval/ZeroEval` scorer**, `zebra_grid_eval.py`:
  `easy_sizes = ['2*2','2*3','2*4','2*5','2*6','3*2','3*3']`. All 40 rows per
  size in the test split (40 x 7 = 280), no further sampling — the "easy" tier
  is already exactly this set of `id`s in the public split, so there is no
  selection hash to freeze; the *set of ids* is frozen instead (§7, digest
  method).
- No hard puzzles, no other size buckets. `zebralogic/data_zebralogic.py`
  structurally cannot emit any size outside this list (`EASY_SIZES` is a
  closed tuple, not a CLI flag).

## 2. Prompt and scoring: aligned to the official ZeroEval implementation

- **Prompt**: the official `ZEBRA_GRID` template verbatim, copied byte-for-byte
  from `WildEval/ZeroEval` `src/templates/ZEBRA_GRID.py` (fetched 2026-09-03,
  see `zebralogic/official_zebra_grid_template.py` for the frozen copy + its
  source URL and fetch date). One-shot CoT: one worked 3-house example with
  `reasoning` + `solution` JSON, then the target puzzle, then an instruction to
  answer in the identical JSON schema (a `json_template` built per-item from
  the puzzle's own `solution.header`/`rows` shape, matching
  `apply_lgp_grid_template`'s construction exactly — this is the ONLY per-item
  templated content, everything else is fixed prompt text).
- **Parser**: the official `extract_last_complete_json` (brace-stack scan for
  the LAST complete top-level `{...}` block in the generated text), copied
  verbatim from `WildEval/ZeroEval` `src/evaluation/eval_utils.py`. Frozen copy
  in `zebralogic/official_zebra_grid_scorer.py` alongside its source URL.
- **Scorer**: the official single-prediction cell/puzzle scoring logic from
  `zebra_grid_eval.py`'s `eval_model` (the `n_size == 1` / "single" branch —
  this project always generates exactly one completion per item, never
  best-of-N, so the `best_of_n`/`majority_of_n`/reward-model branches never
  fire and are not ported). Per-cell comparison is `str.lower().strip()`
  equality against `private_solutions[id]`; puzzle-level `solved` is
  `correct_cells == total_cells` for that puzzle (every cell must match, not a
  threshold). Ported verbatim into `zebralogic/official_zebra_grid_scorer.py`,
  not reimplemented from a paraphrase.
- **Gold source in the scorer is ALWAYS the private dataset** — the scorer
  refuses to run without a working `HF_TOKEN` that has been granted access to
  `allenai/ZebraLogicBench-private`. It never falls back to a substitute
  dataset, a hand-solved puzzle, or a locally-built solver. This is a
  structural refusal (`sys.exit`), not a soft warning.
- No LLM judge anywhere in this line.

## 3. Run configuration (both models)

- Role: neutral. Bare-string prompt — **no chat template** (matches the
  project-wide default for the GSM8K/BBH/CRUXEval/LogiQA family; ZeroEval's
  own `unified_infer.py` for local HF models formats the SAME way, a plain
  instruction string, not a chat-templated one, for base completion-style
  local runs — consistent with using bare string here).
- Greedy decoding, `temperature=0.0`.
- `batch_size=8` for the formal run. **All α, including α=0, go through the
  same batched `vc.regenerate()` path — never a per-cell `bs=1` fallback.** If
  a batch OOMs, that is a stop-and-report condition, not a silent bs change for
  one cell (mixing bs across α would violate "same generation config for every
  α of one model").
- `max_new_tokens=2048` by default. **Escalation to `3072` is the ONLY
  allowed step, and only if α=0 preflight truncation exceeds ~1-2%; the new
  budget applies uniformly to every α of that model, never per-cell.** If
  `3072` still shows material truncation, STOP and report — `4096` is
  explicitly out of scope and may not be enabled unilaterally. **This is
  enforced structurally, not by discipline alone: `get_answer_zebralogic.py`
  hard-refuses any `--max_new_tokens` value other than 2048 or 3072.**
- Prefill-only steering, `prefill_tail_len=1` (last prompt token only).
- Tokenizer: `padding_side="left"` (already the `VicundaModel.__init__` global
  default — no override needed, verified in `llms.py:92`).
- Every α of one model shares identical batch size and generation config.

### Doses

| Model | Band | L | α set |
|---|---|---|---|
| Llama3.1-8B-Instruct | `[11,20)` | 9 | `{-6, -4, 0, +4}` |
| Qwen2.5-7B-Instruct | `[16,22)` | 6 | `{-6, 0, +6, +8}` |

Raw α, mask, and layer band are **not comparable across models** — same
convention as every other line in this repo (P3/P4/P4b/P4c). Doses are fixed
by this pre-registration and are never searched or added to.

## 4. Machine / GPU

- Same machine, same GPU model family, same software environment for all
  cells of both models. Different physical GPUs are permitted; device
  (`CUDA_VISIBLE_DEVICES`, `nvidia-smi` GPU name) is recorded per cell.
- **A model's own four α cells must share ONE physical card** — this is a
  paired per-item contrast (McNemar/bootstrap against that model's own α=0),
  and bf16 greedy is not byte-reproducible across GPUs (measured elsewhere in
  this repo: a cross-GPU re-run differed on 205/300 GSM8K samples). The two
  MODELS may use two different cards; they are never compared per-item against
  each other.
- If multiple physical GPUs are used anywhere in this line, a **fixed canary
  subset** (a small, deterministic slice of the 280-item set, α=0 only) must be
  run on each card in use and compared before any cross-GPU pooling is trusted.
  Canary agreement criterion (frozen, matches the user's correction of
  2026-09-03): **text need not match verbatim**, but Puzzle Accuracy, Cell
  Accuracy, parse status, and truncation status must show **no systematic
  divergence** between cards. If they do diverge systematically, that model's
  cells may NOT be pooled into one paired α-curve across those cards — the
  model's whole four-point sweep must be re-run on a single card instead.

## 5. α=0 preflight

- **5 items PER SIZE** (7 sizes x 5 = 35 items), fixed by index within each
  size's block of 40 in the frozen 280-item order (the first 5 items of each
  size, i.e. `sample_id`s `{0..4, 40..44, 80..84, 120..124, 160..164, 200..204,
  240..244}`) — not 5 items total, and not all drawn from one size. Both
  models share this SAME 35-item set (one frozen item order), run at α=0 only.
  This corrects an earlier draft of this section that said "5 items... 35
  items total across both models" while describing a single-size 5-item
  subset — those two statements were contradictory; 35 items covering all
  seven sizes is the frozen definition.
- Preflight checks (all must pass before any formal cell runs):
  - official prompt renders correctly (worked example + target puzzle +
    per-item JSON template);
  - the private-gold scorer resolves those 5 `id`s and scores correctly
    against a byte-identical local port of `eval_model`'s single-prediction
    branch;
  - `extract_last_complete_json` parses the generated JSON reliably;
  - tokenizer is confirmed `padding_side="left"`;
  - the last prompt token (the actual injection site under prefill-only
    steering) is read out and logged, not assumed;
  - α=0 `steering_fires == 0`;
  - a non-zero smoke config (α≠0, not part of the frozen four points, used
    ONLY for this firing-count assertion) confirms `steering_fires == L * N`;
  - no-answer rate, parse-failure rate, and truncation rate at the preflight
    budget;
  - generation-length median / P95 / P99.
- **Preflight governs implementation and token-budget only.** It may not be
  used to change the prompt, the item set, the doses, or the scoring method
  based on observed accuracy. If `max_new_tokens=2048` preflight truncation is
  ≤ ~1-2%, the formal budget freezes at 2048. Only explicit truncation above
  that triggers the single allowed escalation to 3072 (§3).

## 6. Commitment metrics (frozen before any non-zero-α formal result is read)

ZebraLogic-native, defined on the parsed JSON output plus the raw generated
text — **not** the digit-based `earlycand-v1` detector (that detector assumes
a short first line containing a bare number, which does not describe this
task's JSON-grid output format and would silently ceiling/floor here the way
it does on LogiQA's letter-choice format).

- `solution_first_rate` — the FIRST complete top-level JSON object appearing
  in the text (via a forward-scanning variant of the brace-stack parser,
  distinct from `extract_last_complete_json`'s backward "last complete" scan)
  contains a non-empty `solution` key with no unfilled (`"___"` or empty)
  cells, i.e. the model emitted a fully-specified grid before any later
  revision. Distinguishes "committed a full grid early" from "still reasoning
  when the only JSON block appears".
- `first_solution_pos` — normalized character offset of the `"solution"` KEY
  inside that first complete JSON object, within the full generated text (0 =
  start, 1 = end) — **not** the object's opening brace. This matters because
  the official output format is one JSON object,
  `{"reasoning": "...", "solution": {...}}`: CoT reasoning normally lives
  INSIDE the object, before the "solution" key, so anchoring on the opening
  `{` would read near-zero regardless of reasoning length. Falls back to the
  opening-brace offset only if the "solution" key cannot be located textually
  within the object's span (defensive; should not occur when a "solution"
  value was successfully parsed). Undefined (reported as coverage, not
  imputed) for texts with no complete JSON object.
- `pre_solution_chars` / `pre_solution_tokens` — characters / whitespace-token
  count preceding that `"solution"` key (i.e. any free text before the JSON's
  opening brace PLUS the "reasoning" value and surrounding JSON syntax up to
  the "solution" key).
- `reason_before_solution` — boolean: does non-trivial free text (more than a
  few characters) precede the first complete JSON object. Complements
  `solution_first_rate` for texts where the first JSON is partial/malformed.
- `first_final_grid_agreement` — cell-level agreement between the FIRST
  complete JSON's `solution` grid and the LAST complete JSON's `solution` grid
  (the one the official scorer actually grades) — 1.0 means the model never
  revised its grid after first writing it.
- `revision_wrong_to_right` / `revision_right_to_wrong` — among cells where
  first-grid and last-grid disagree, the count (using the resolved gold) that
  moved from incorrect to correct, and correct to incorrect, respectively.
  **Requires gold and is therefore only computable at analysis time, on the
  private-gold side, never during label-free generation.**
- Standard health diagnostics, also frozen: no-answer rate (no parseable JSON
  at all), invalid-JSON rate, missing/extra row or column count, the project's
  strict loop/repetition detector (final 40-char block recurring >= 4 times —
  same convention as the rest of this repo, not a permissive n-gram proxy),
  truncation rate (`>= max_new_tokens - 1`), and generation length
  (median/P95/P99, in characters).

**These are OUTPUT DIAGNOSTICS, computed from steered generations. They can
only ever show co-occurrence with accuracy changes, never causal mediation —
same standing rule as every other commitment-metric line in this repo (P2,
P3-supp, the CoT-followup line). Any subgroup breakdown by these metrics is
post-treatment stratification.**

## 7. Formal run and per-cell validation

Both models, all 280 items, all four α each. Every cell is auto-validated on
write:

- `N == 280`;
- `sample_id`s complete, unique, and covering the frozen easy-tier id set
  (digest-checked against `zebralogic/EASY_IDS_SHA256` written by the loader);
- sample order identical across every α of that model (and, since both models
  share one item order, across models too);
- prompt hash (`sha256` of the rendered per-item prompt list, concatenated in
  order) identical across every α of one model — confirms only α/mask
  differs, never the prompts;
- α, mask path + sha256, layer band recorded in cell metadata and cross-checked
  against the config that produced the file;
- generation config (`max_new_tokens`, `temperature`, `top_p`, `batch_size`)
  identical across every α of one model;
- `steering_fires == 0` at α=0, `== L * N` at α != 0 (hard stop otherwise —
  intervention unverified);
- the scorer reproduces its own summary numbers from the raw per-item file on
  a second pass (idempotency check);
- raw generations are saved in full (no truncation/summarization on write);
- output paths are entirely under the isolated ZebraLogic tree; the writer
  refuses to overwrite an existing cell file.

## 8. Statistics

Per model, independently:

- Each non-zero α paired against that SAME model's own α=0, item-by-item
  (`sample_id`-aligned, never by row position).
- **Puzzle Accuracy**: exact two-sided paired McNemar (matching this repo's
  `mcnemar_exact` convention in `eval_bbh_numeric.py` / `eval_p3.py` family —
  exact binomial tail on discordant pairs, not the chi-square approximation).
- **Holm correction, m=3, separately per model** (Llama's three non-zero α
  form one Holm family; Qwen's three non-zero α form an independent Holm
  family — never pooled across models, matching the standing rule that Holm
  families are task/model-scoped and never mixed).
- Report per α: accuracy, ΔAcc (pp), discordant counts (`0->1`, `1->0`), raw p,
  Holm-adjusted p, and item-level paired bootstrap 95% CI (B=10000, seed=0,
  resampling puzzle-level per-item differences — matching `boot_ci` in
  `eval_bbh_numeric.py`).
- **Cell Accuracy**: paired bootstrap at the PUZZLE level (resample puzzles,
  not individual cells) — a puzzle's cells are not independent observations of
  the same phenomenon, so cell-level bootstrap would understate variance. No
  McNemar for Cell Accuracy (it is not binary per unit).
- All seven easy sizes reported separately (40 items each) as descriptive
  breakdowns; **no single size may be cited as an independently significant
  workpoint** — 40 items is far too few for its own inferential claim.

### Workpoint rule (frozen, verbatim from the specification)

1. Among the four sampled points (including α=0), the highest Puzzle Accuracy
   is the discrete argmax.
2. Exact ties: prefer the smaller `|α|`.
3. Any point whose paired difference from the argmax is NOT detected (Holm-
   adjusted p > .05, or CI includes 0) may be listed in the **observed
   near-optimal region**.
4. Near-optimal-region comparisons (argmax vs. non-argmax non-zero points) are
   EXPLORATORY and excluded from the `α vs. 0` Holm family in §8 — that family
   covers only the three `α vs. 0` contrasts.
5. **If no non-zero α clears Holm correction against α=0, the conclusion is
   "no effective workpoint detected among the four sampled points" — the
   highest raw number may NOT be reported as an established workpoint.**
6. If only Cell Accuracy improves while Puzzle Accuracy does not, the
   permitted conclusion is limited to "localized grid-filling quality
   improved" — never a claim about solving more complete puzzles.

## 8b. Easy-tier confirmation gate (per the user's 2026-09-04 instruction)

Before the formal sweep, the launcher must print the official
`WildEval/ZeroEval` `zebra_difficulty.py` distribution (`easy_sizes` /
`hard_sizes`, and the finer `small/medium/large/xl` split) alongside this
protocol's frozen 7-size list, and then **STOP and wait for explicit
confirmation** before running the formal 280-item sweep for either model. This
gate exists independent of the fact that the two lists happen to already match
exactly (§1) — the instruction was to show the distribution and wait, not to
skip the wait because the sets match.

## 9. Access blocker (recorded, not solved by this pre-registration)

Real gold requires per-account HF consent on
`https://huggingface.co/datasets/allenai/ZebraLogicBench-private`
plus a valid `HF_TOKEN` for that account, available as an environment variable
on the machine that runs the scorer. **The scorer hard-stops with an
explanatory message if this is not satisfied — it never substitutes another
dataset, a hand-built solver, or a relaxed parser.** This blocks the FORMAL
run and the private-gold-dependent commitment metrics
(`revision_wrong_to_right` / `revision_right_to_wrong`), not the label-free
generation, preflight (format-only parts), or code preparation.

**Private-gold integrity check (`load_private_gold`, hardened 2026-09-04):**
beyond confirming requested ids resolve, it also hard-stops on (a) the private
split not having exactly 1000 rows, (b) duplicate ids in the private split,
(c) a malformed `solution` (missing header/rows, `header[0] != "House"`, or a
row whose length disagrees with its header), and (d), when the caller supplies
`expected_shapes` (every call site in `eval_zebralogic.py` does, built from
each generation cell's own `solution_shape` field), a per-id shape mismatch
between the public and private datasets. It also prints the resolved private
split's row count, sorted-id digest, and resolved revision for provenance,
since this repo has no revision pin for the private dataset (§1).
