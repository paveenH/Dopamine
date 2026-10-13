# P-phase archive — result tables, artifact hashes, runbooks

Split out of `CLAUDE.md` on 2026-09-02 to shrink the always-loaded context. P2,
P3 and the P3 supplement are all **COMPLETE + FROZEN**; their frozen wording,
口径 traps and boundaries stay in `CLAUDE.md`, and their narrative results live
in `ReasoningBare.md` §5.1–5.6. What moved here is the material a future
session does not need in context to avoid a mistake: the numeric result tables
already published in the results document, the per-artifact SHA256 lists (also
in the `docs/*_manifest.json` / `docs/*_result_*.json` files), and the
reproduction runbooks.

Nothing was deleted; every block below is byte-identical to what stood in
`CLAUDE.md`.

---

## P2 — commitment prediction + cross-task workpoint transfer

Protocol `docs/PREREG_P2.md` (commit `37713c8`, protocol sha256 `8fddcf12…`).
Offline in `RoleAnswer/p2/`, `python3.10`, no server, no GPU.

### P2A frozen result details

- **Held-out performance.** Question-grouped five-fold OOF AUROC is `.687
  [.656, .719]` for Llama and `.749 [.710, .787]` for Qwen with the
  commitment-only predictor. Entry-only AUROC is `.548 [.526, .571]` and `.628
  [.601, .654]`, respectively. The paired question-cluster bootstrap difference
  `commitment − entry` is `+.139 [+.104, +.172]` for Llama and `+.121
  [+.084, +.156]` for Qwen.

- **Incremental entry-gain comparison.** The paired `combined − commitment`
  difference is `−.001 [−.004, +.002]` for Llama and `+.002 [−.002, +.007]`
  for Qwen. Both intervals include zero. Frozen wording is **"no detectable
  incremental predictive gain"**, never "entry gain is useless" or "entry gain has
  no information".

- **Gate and calibration.** The preregistered gate is commitment-only AUROC 95% CI
  lower bound `> .5`; both models pass (Llama `.6561`, Qwen `.7098`). GSM8K OOF
  calibration slopes are `.95` and `.98`, respectively, and are shown in
  `fig_p2a_calibration.png`. These results establish out-of-sample prediction on the
  training task, not causal mediation or intervention specificity.


### P2B frozen result details

- **Qwen full-curve transfer.** Across 9 MATH doses, the frozen predictor selects the
  positive direction and `alpha=+6`; observed `first_acc` also peaks at `+6`. Spearman
  `rho=+.962` (`n=9`), regret is `.000`, and the selection falls in the near-optimal set
  `[+4,+6]`. The predicted rollover at `+8` is also the observed accuracy decline.

- **Llama local-direction transfer.** Only three MATH cells exist (`−4/0/+4`), so the
  frozen evaluation is local direction only. It selects the negative direction and
  `alpha=−4`; observed `first_acc` is locally best at `−4`, regret is `.000`, and the
  selection falls in `[−4,0]`. The resulting `rho=+1.000` at `n=3` is mechanically weak
  evidence and must not be presented beside Qwen's nine-point correlation as an
  equivalent full-curve result.

- **Freeze and figure interpretation.** `p2b_predictions.json` was frozen at SHA256
  `4e52b079…` before MATH accuracy was read. In `fig_p2b_transfer.png`, panel A retains
  the uncalibrated raw gap (Qwen predicted score `.83–.88` versus observed accuracy
  `.54–.68`, roughly `.25` high), while panel B plots change relative to `alpha=0` and
  shows the matched positive direction, `+6` peak, and `+8` decline. No prediction was
  rescaled or recalibrated using MATH labels. The supported claim is retrospective
  locked workpoint selection, not blind validation and not transferable absolute
  accuracy calibration.


### Artifacts and reproduction

- Artifacts + SHA256 in `p2/p2_freeze_manifest.json`: `PREREG_P2.md` `929004f1…`,
  `p2b_predictions.json` `4e52b079…`, `p2b_evaluation.json` `0e11cb0d…`,
  `p2_predictor_llama.json` `b9ee07e4…`, `p2_predictor_qwen.json` `9aad950a…`,
  `fig_p2a_calibration.png` `7f135253…`, `fig_p2b_transfer.png` `f380461e…`.

```bash
# Reproduce, in order. From RoleAnswer/, python3.10, no GPU.
python3.10 p2/build_p2_folds.py --check   # fold manifest reproduces exactly
python3.10 p2/run_p2_audit.py             # label firewall + exhaustive partition,
                                          # reproduces P1's 177/66/57 + loop 52
python3.10 p2/run_p2a.py                  # 5-fold OOF, cluster bootstrap, gates
python3.10 p2/freeze_p2_predictor.py      # refit on all GSM8K, freeze artifacts
python3.10 p2/run_p2b_predict.py          # MATH predictions; REFUSES to overwrite
python3.10 p2/run_p2b_eval.py             # unlocks accuracy; REFUSES without predictions
python3.10 p2/plot_p2.py                  # both required figures
```

Both `run_p2b_predict.py` and `build_p2_folds.py` refuse to overwrite an existing
frozen file: re-running the pipeline end-to-end requires deliberately deleting
them, which is the point — a silent regeneration would let a later fit
masquerade as the locked prediction.

---

## P3 — blind cross-task validation on GSM-Hard

Protocol `docs/PREREG_P3.md` (`p3-v1`, tag `p3-prereg-v1`, commit `1c0b865`).
Result record `docs/p3_result_20260830.json`. Gold was unsealed once, on
2026-08-30; **the main analysis 口径 is CLOSED** and anything further is
exploratory.

```bash
# Server, from /data1/paveen/Dopamine. Download writes to components/benchmark/;
# a non-existent --out_dir is refused rather than created, so a wrong working
# directory fails before the download instead of writing to the wrong tree.
python data_gsm_hard.py --revision 960448f73503112d4226baeb8eb41d3fb5ae2506

# Format preflight, then the ten cells. One model per card.
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_hard_llama3.sh --preflight > p3_pre_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_hard_qwen25.sh --preflight > p3_pre_qwen.log 2>&1 &
# cat the log immediately -- a wrong PY exits 127 before anything runs and the
# nohup log looks empty.
```

---

## P3 supplement — CoT condition transfer

Protocol `docs/PREREG_P3_SUPPLEMENT.md` (`p3-supp-v1`, tag `p3-supp-frozen`).
Result record `docs/p3_supp_result_20260830.json`.

```bash
# Server, from /data1/paveen/Dopamine. One model per card; both cells of a model
# stay together. cat the log immediately -- a wrong PY exits 127 before anything
# runs and the nohup log looks empty.
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_hard_cot_llama3.sh --preflight > p3_cot_pre_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_hard_cot_qwen25.sh  --preflight > p3_cot_pre_qwen.log  2>&1 &
# then, only after the format check passes:
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_hard_cot_llama3.sh --full > p3_cot_full_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_hard_cot_qwen25.sh  --full > p3_cot_full_qwen.log  2>&1 &
```

```bash
# Offline (RoleAnswer/, python3.10, no GPU). Refuses to overwrite a frozen file,
# so re-running the chain requires deliberately deleting it.
python3.10 p3/freeze_p3_supp_predictions.py   # stage 1; already run, fcf8b9c9b8fa8b70
python3.10 p3/freeze_p3_supp_commit.py        # stage 2; already run, 6a16d4d862edbdaa
                                              # applies the frozen P2 predictor to the
                                              # four CoT cells and seals the commitment
                                              # side BEFORE any accuracy is read
python3.10 p3/run_p3_supp_eval.py             # unlocks CoT accuracy; REFUSES to start
                                              # without the stage-2 file
python3.10 p3/commit_panel_gsm_hard.py        # commitment + cap% panel (EXPLORATORY)
```

## 2026-10-13 cut — protocol / implementation / runbook blocks of the CLOSED P3, P3-supp, P4, P4b and P4c sections

Moved byte-identical from `CLAUDE.md` on 2026-10-13. CLAUDE.md keeps each line's RESULT block, frozen wording and the rules a citation still depends on; everything here is pre-result design, provenance, hashes and runbooks. All five lines carry a POST-RESULT rule: their main-analysis 口径 is CLOSED.

### P3 protocol / implementation bullets (former CLAUDE.md lines 1201–1285)

<!-- Why: the blind property is the only thing separating P3 from P2, it cannot be
restored once lost, and nothing in the code can enforce "a human did not look".
Evidence: docs/PREREG_P3.md section 0; the eligibility audit in section 1.
Scope: all of P3. Read before touching anything under gsm_hard_p3. -->
- **P3's entire value is that GSM-Hard accuracy has never been viewed. Reading
  `components/benchmark/gsm_hard_p3_gold.SEALED.json` before `p3_predictions.json`
  is frozen destroys that PERMANENTLY** — no later re-freeze repairs it, and the
  result degrades to another retrospective test like P2. That includes an editor
  preview or a `head`. Eligibility was audited before download (no data, no loader,
  no artifacts anywhere on the machine) and recorded in the protocol.

- **The loader writes TWO files and that split IS the firewall.**
  `data_gsm_hard.py` → `gsm_hard_p3_questions.json` (question + `sample_id`, meta
  declares `contains_labels: false`) and `gsm_hard_p3_gold.SEALED.json`. Generation
  reads only the former. A single file would leave the label one attribute access
  from every script; P2 established that "unused" is far weaker than "unreachable".
  Both live in **`components/benchmark/`** (the server's flat benchmark tree), NOT
  `benchmark/`.

- **`get_answer_gsm_hard_blind.py` is a FORK of `get_answer_regenerate_gsm8k.py`, not
  a reuse.** That script calls `is_correct_gsm8k` per sample and prints accuracy
  (lines 83–96), so running it would compute GSM-Hard accuracy during generation.
  Prompt construction and every `vc.regenerate` argument are identical to the frozen
  GSM8K path — so commitment features keep the meaning they had in P2 — while
  correctness/gold/`pred_answer` are removed entirely.

- **Generation params match the frozen GSM8K main line, deliberately: `max_new_tokens
  =768`, `temperature=0.0`, `bs=24`.** The P2 predictor was fitted on GSM8K output
  produced this way and `first_parseable_marker_posN` / `early_candidate` are sensitive to length and
  sampling. **Do NOT borrow MATH's 2048/bs=8** — GSM-Hard keeps GSM8K's structure and
  only enlarges the numbers.

- **Doses are fixed in advance and are NOT re-searched:** Llama `−8/−6/−4/0/+4`
  (band `11-20`), Qwen `−4/0/+4/+6/+8` (band `16-22`). **Qwen keeps a negative probe
  by design** — with every candidate on the positive side, "correctly predicted the
  positive direction" could not have been wrong and would not be a test.

- **One MODEL's five doses must share one card; the two models may run on two.**
  bf16 greedy is not byte-reproducible across GPUs (a measured re-run differed on
  205/300 samples), so a split dose curve mixes the device difference into the α
  effect. The models are never compared per-question, so they parallelise freely.

- **Sample = 300 by frozen hash, never dataset order** (`sha256(salt:question_text)`,
  dedup on exact text first). Digest `48cc7635…`, revision pinned to
  `960448f73503112d4226baeb8eb41d3fb5ae2506`. `--revision` is type-checked to a full
  40-hex SHA: `required=True` alone is not a pin, since `--revision main` still
  follows upstream.

<!-- Why: auditing a float64 precision failure BY CALLING float() is self-defeating,
and the string-form test passed while the real input shape would have been missed.
Evidence: docs/p3_amendment_02.json; test_p3_label_firewall.py section 4b1.
Scope: the 2^53 audit only. Measured 0 of 300 at the pinned revision. -->
- **The `2^53` audit must never route through `float`, and a float64 gold in the
  unsafe range is a HARD STOP.** `float('9007199254740993') == 2**53`, so a
  float-based audit misses the first value that matters; `Decimal` parses the literal
  exactly. Separately, HF stores `target` as float64, so `datasets` hands back a
  python float whose `+1` is already destroyed upstream — unrecoverable, hence
  `assert_float_safe` refuses rather than under-reporting. **Measured at the pinned
  revision: 0 of 300 exceed `2^53`, so the ORIGINAL extractor is used and
  `norm_exact` is NOT enabled.** `p3/p3_bigint_audit.py` verified `norm_exact` against
  6600 stored GSM8K predictions with **0 verdict changes**.

- **Two near-optimal sets exist and must never be conflated.** The *predicted*
  region comes from the frozen predicted-score rule; the *observed* set is the doses
  whose paired per-question accuracy difference from the empirical best is
  indistinguishable. **The verdict is whether the predicted workpoint lands in the
  OBSERVED set** — closeness in predicted score is not sufficient.

- **No α=0 accuracy threshold gates readability.** A low baseline does not mean the
  curve is unreadable. Instead: a paired omnibus over each model's five doses, Holm
  across the two models; if no dose difference is detected the result is **"no
  readable dose curve"** and direction/workpoint are **not evaluable** — neither a
  success nor a failure.

- **Locked run order:** freeze protocol → download + schema/`2^53` audit → format
  preflight (`--preflight`, 5 samples, format ONLY) → generate all ten cells with NO
  accuracy → extract commitment features, apply the frozen P2 predictors, **freeze
  `p3_predictions.json` + SHA256** → only then unseal gold, once. **If the preflight
  format violates the frozen protocol the response is a HARD STOP** — not a redesigned
  prompt or extractor.

**Reproduction runbook (download, preflight, the ten cells) moved to
`docs/P_PHASE_ARCHIVE.md` (2026-09-02).** The trap it carries still applies to any
server launch: `cat` the log immediately — a wrong `PY` exits 127 before anything
runs and the `nohup` log looks empty.

### P3 RESULT provenance bullets (former CLAUDE.md lines 1339–1370)

- **Order of operations, each step tagged or hashed.** protocol `p3-prereg-v1` (`1c0b865`)
  → download + sealed gold (`2^53` audit **0/300**, so the original extractor was used and
  `norm_exact` stayed OFF) → ten cells with no accuracy code path → **predictions frozen**
  `9ad21ee`, sha256 `8f0c514ae4776441` → **evaluator frozen** `p3-evaluator-frozen`
  (`850ef29`) → gold unsealed ONCE (`p3-result-unsealed`, `ed55753`). The evaluator that
  ran was verified **byte-identical** to the tagged one before the result was accepted.

- **Artifacts.** gold `a464d591abe659dc`, predictions `8f0c514ae4776441`, evaluation
  `0e3511c74ba89115`, evaluator `2ca6de427d2a8ceb`, unseal log `0ba5510226ef30e8`,
  questions `48cc763545d2ee23`, GSM-Hard revision `960448f73503112d4226baeb8eb41d3fb5ae2506`.
  Full record: `docs/p3_result_20260830.json`; the ten generation SHA256 are in
  `docs/p3_amendment_05.json`.

- **`run_p3_predict.py` copies `apply_frozen`/`near_optimal` VERBATIM from
  `p2/run_p2b_predict.py` and re-verifies them character-for-character at every run.**
  That P2 script cannot be imported (its module body refuses to run once
  `p2b_predictions.json` exists), and a first attempt that sliced the functions out of the
  source text at runtime silently extracted the WRONG region — `inspect.getsource` showed
  the module docstring instead. Mutation-tested: perturbing the sigmoid makes the guard
  fail closed.

- **`run_p3_eval.py` fails closed** on a missing `p3_predictions.json` (the ordering
  guard), a cell whose `questions_sha256` differs from gold, a missing/extra/duplicate
  `sample_id`, a missing cell file, or an existing output. Alignment is **by `sample_id`,
  never by row position** (asserted by reversing a cell's row order). `first_acc` is MAIN
  via the frozen offline extractor; `last_acc` is sensitivity only.

- **`test_p3_eval.py`: 35/35 on synthetic fixtures, no real gold.** Statistics are checked
  against hand-computed values (`McNemar(10,0) = 2/1024`), every guard is mutation-tested,
  and — the load-bearing one — a deliberately WRONG prediction fixture confirms the
  evaluator reports `direction_correct=False`, large regret and negative ρ rather than
  passing a bad result silently.

### P3-supp protocol bullets (former CLAUDE.md lines 1428–1489)

<!-- Why: the supplement's whole claim to being "locked" rests on the CoT cells not
existing when the direction was frozen; a reader who treats it as another blind
validation will over-claim, and one who treats it as post-hoc will under-claim.
Evidence: p3_supp_predictions.json cot_cells_exist_at_freeze_time=false.
Scope: every supplement citation. -->
- **TWO PARTS, DIFFERENT EVIDENTIAL STATUS, and the split is the point.** Part 1 (the four
  CoT cells) is a **locked prospective condition test** — they did not exist when
  `p3_supp_predictions.json` was frozen. Part 2's No-CoT half (`Note/ReasoningBare.md` §5.5's commitment
  panel) is **EXPLORATORY and permanently so** — those ten cells were unsealed first.
  **This is NOT a new blind dataset validation**: same 300 questions, same gold. What is
  prospective is the CONDITION.

- **FOUR CELLS ONLY: llama `0`/`−6`, qwen `0`/`+8`.** The α is the workpoint already
  established on GSM8K and already used in P3 — **not re-searched, and no CoT dose curve
  is run**. Everything else is held identical to the P3 No-CoT cells (same questions and
  order, `768`/`0.0`/bs `24`, prefill-only `tail=1` at the `Answer: ` anchor). **Both cells
  of one model share a card** — the primary metric is a per-question paired contrast
  between them.

- **`--cot` inserts EXACTLY one line, `Let's think step by step.`** Verified by diffing the
  two frozen templates: the `####` directive and the `Answer: ` anchor are untouched, so
  the injection site (token 220) and the commitment features keep the meaning they had
  when the predictor was fitted. The flag defaults to False, so every P3 caller is
  byte-identical, and it is recorded in meta — `prompt_template` alone would leave a
  consumer that reads only `alpha` unable to tell the two batches apart.

- **`meta.protocol` is `p3-supp-v1` when `--cot`, else `p3-v1`** (amend-01). A CoT cell
  carrying `p3-v1` would claim provenance from the CLOSED blind validation, and `cot=true`
  does not repair that — a consumer filtering on `protocol` would still pick it up as P3.

- **The freeze is TWO-STAGE, because the one-stage version was impossible** (amend-01).
  §1.5 originally listed "the frozen predictor's mean predicted score per cell" among the
  items frozen BEFORE generation — not computable, since the cells have no generations and
  therefore no features. The artifact was already correct (it contains no such field); only
  the protocol text overstated it. Stage 1 = freeze the **direction** of `ΔAcc` before
  generation (done, `fcf8b9c9b8fa8b70`); stage 2 = freeze the per-cell commitment score
  after generation but **before** accuracy.

- **Statistics.** Primary `ΔAcc = Acc(CoT+α) − Acc(CoT)` on `first_acc`, paired per
  question; exact two-sided McNemar with discordant counts; **the two models are ONE Holm
  family (m=2)**. The interaction `[CoT+α − CoT] − [NoCoT+α − NoCoT]` is **descriptive and
  EXCLUDED from Holm** — its No-CoT half is already unsealed, so it is not a locked
  prediction. Its four readings are pre-registered so the result cannot pick its own frame.

- **BUDGET POLICY IS STAGED AND CONDITIONAL (amend-03).** `768` stays primary. **A single
  extra cell at 1024 is explicitly rejected** — adding only No-CoT α=0 cannot separate a
  budget increase from a steering effect, because there is no α contrast at the new budget;
  a complete 1024 comparison needs a same-budget **2×2 per model**, `{No-CoT, CoT} ×
  {α=0, α=workpoint}`. Trigger order: run the 768 CoT cells → add llama CoT `0`/`−6` at
  1024 **only if** llama's CoT cap-hit makes truncation a live alternative explanation →
  add llama No-CoT `0`/`−6` at 1024 **only if** the interaction then needs a common budget.
  **Qwen is NOT auto-included** (its No-CoT cap-hit is 13–23% against llama's 91–96%); it
  gets a 1024 arm on its own evidence, never for symmetry. **Predictor scores on 1024
  output are EXPLORATORY** — the coefficients were fitted on 768-token generations, and
  `first_parseable_marker_posN` is a normalized position while `early_candidate` is a first-line rule.

- **PREFLIGHT PASSED 2026-08-30** (5 samples, format only): `protocol=p3-supp-v1`,
  `cot=true`, the template carries the CoT line, `####` parses, and
  `steering_fires` read **0 / 45** (llama, `9×5×1`) and **0 / 30** (qwen, `6×5×1`).
  The four formal cells then read exactly **2700 / 0** (llama) and **1800 / 0** (qwen),
  n=300 each, one shared digest `48cc7635…`, no label fields.


### P3-supp artifacts + runbook pointer (former CLAUDE.md lines 1536–1547)

- **Artifacts:** stage 1 `fcf8b9c9b8fa8b70`, stage 2 `6a16d4d862edbdaa`, evaluation
  `7e39f9e36edfd0c4`; full record `docs/p3_supp_result_20260830.json`.
- **`run_p3_supp_eval.py` REFUSES to start without `p3_supp_commit.json`** — the same
  ordering guard `run_p3_eval.py` has. Both new scripts were mutation-tested before
  use: ten guards on the stage-2 freeze and six on the evaluator each verified to
  fire with the right message, against a passing unmutated control.

**Both runbooks (server generation, offline freeze/eval chain) moved to
`docs/P_PHASE_ARCHIVE.md` (2026-09-02).** The offline scripts refuse to overwrite a
frozen file, so re-running the chain requires deliberately deleting it.



### P4 protocol / implementation / runbook (former CLAUDE.md lines 1622–1756)

- **Two-stage freeze, and the split is the point.** Stage 0 fixed the question,
  doses, sampling, parsing, statistics and the token-budget upgrade rule BEFORE
  any loader existed. Stage 1 (`p4-amend-04`) added ONLY the budget the blind
  preflight selected. Anything else changing between them would be an
  adaptive edit.
- **`data_logiqa.py` (LogiQA 1.0-era) downloads the same file and is left
  byte-unchanged.** It is unusable here for four reasons: it writes to the
  stale `/data2/.../RolePlaying` tree, drops `id` / raw fields / `type`
  provenance, freezes no manifest, and has no generative runner. `data_logiqa2.py`
  is new.
- **The item key is a COMPOSITE, and this is load-bearing.** Measured on the
  1572-row test split: `id` is unique only 1568/1572 and `(passage, question)`
  only 1557/1572 — **neither is a key**. The frozen key is
  `sha256(salt:id US passage US question US options.join(RS))`, unique 1572/1572.
  Never `hash()` (process-salted). Both the official `id` and a salt-free
  `content_sha256` are stored, so an upstream edit is detectable.
- **Sample: 300 items, exactly 75 per gold label**, taken as ranks 1–75 by key
  within each label; the 20 preflight items are ranks 76–80, so **disjointness is
  structural, not checked after the fact**. Frozen digest `4d4b25e071a2a6dd`.
  Label pools are 347/384/417/424, so no stratum is starved.
- **`type` is a MULTI-LABEL dict (3–4 reasoning types per row) and is provenance
  only, never a stratification axis.** Splitting 300 items across an unbalanced
  multi-label space is the sampling problem that shelved MMLU-Pro; it is avoided
  here rather than re-imported.
- **The prompt is NEUTRAL-GENERATION and frozen verbatim** (`p4-amend-02`,
  sha256 `c42dc9c81f117a6c`, anchor `Response: `). `p4-amend-01`'s original
  prompt said "Think step by step." with a `Reasoning: ` anchor — an explicit
  CoT elicitation, so a reasoning trajectory could not be attributed to the model
  rather than the instruction. Deleting the sentence alone was insufficient:
  `Reasoning: ` itself names the expected output as reasoning. **Measured, not
  assumed:** the tail is token **220 `' '`** on BOTH tokenizers (llama
  `382/2647/25/220`, qwen `382/2582/25/220`), and `rstrip()` moves it to `':'`
  (25) on both — the trailing space is an assertion. **Do not write that the two
  models share "the same decision-bottleneck token"**: an identical integer in
  two tokenizers is a numbering coincidence.
- **Parsing: MAIN = LAST match, sensitivity = FIRST, and the reason differs from
  MATH's on purpose.** MATH uses FIRST because tail loops pollute the last
  `\boxed{}`; LogiQA uses LAST because the generative protocol invites revision
  and FIRST would lock a pre-reasoning guess. Zero markers scores incorrect,
  denominator stays 300, **no rescue generation**. The instruction still names
  `Final answer: X` verbatim, which is also why **no stop string may be used** —
  HF `stop_strings` matches anywhere in the output and the model restates the
  instruction (the CGT failure, invalid_rate 0.02→0.11).
- **The label firewall is STRUCTURAL on both blind paths, and the distinction is
  one I got wrong once.** `data_logiqa2.py` emits the formal sample twice from
  one selection: `logiqa2_p4_formal_blind.json` (gold absent, whitelist-built,
  payload-scanned) for generation and `logiqa2_p4_formal.json` (gold present)
  for `eval_logiqa2.py` only. An earlier runner read the gold-bearing file and
  merely declined to touch `answer_letter` — **"the code does not access gold"
  is much weaker than "gold is not reachable"**, which is the whole point of the
  P2 firewall. `get_answer_logiqa2.py` refuses any input whose meta does not say
  `contains_labels: false`.
- **Generation and scoring are separate scripts**, mirroring P3: a generation run
  cannot quietly become an accuracy run.
- **PREFLIGHT RESULT (20 held-out items × 4 cells, 512 tokens, gold-blind):
  Llama is SCORABLE BUT ITS TERMINATION IS DEGRADED; Qwen is normal.** llama
  α=0 20/20 cap-hit (min length 512, **no output stopped naturally**),
  `answer_first` 13/20, degenerate 17/20, first marker at position **0.000**;
  α=−6 `answer_first` 4/20, degenerate 14/20. qwen 0/20 `answer_first`,
  degenerate 0–1/20, first marker at 0.94–0.98, cap-hit 2–4/20. FIRST and LAST
  agree in **39/40** scorable outputs.
- **Degeneration does NOT make it unreadable — this repo's GSM8K precedent is
  strictly more extreme** (97% cap-hit, raw loop rate 74–88% with no clean α
  trend) and remains a main result source. Accuracy is readable because the
  answer is complete BEFORE the degeneration. Adopted with GSM8K's convention:
  **the degeneration rate is descriptive and must NOT be read as perseveration
  absent a clean α trend**, and it uses the same strict `is_loop` detector
  (final 40-char block recurring ≥4×) so the two tasks stay comparable — a
  permissive n-gram proxy read 80–86% on GSM8K and was all false positives.
  One real difference is recorded rather than smoothed over: GSM8K's `####` is
  terminal so its loop is always post-submission, while llama α=0 here is 13/20
  `answer_first`, a different shape.
- **The format gate was RELAXED and the budget rule was EXECUTED AS WRITTEN —
  opposite directions, deliberately** (`p4-amend-05`). The gate was written
  against "cannot produce the format", which is not what happened, so one valid
  marker now makes an output scorable and only a cell with NO marker anywhere is
  a hard stop. The budget rule fires because of repetition rather than long
  reasoning, but it states a MECHANICAL condition; keeping 512 would have been a
  SECOND adaptive change made in response to preflight output. **Budget = 1024**,
  with longer llama tails as the accepted, recorded cost.
- **`answer_first` 13/20 → 4/20 at α=−6 is a PILOT-GENERATED TIMING HYPOTHESIS,
  not effect evidence**: n=20, held-out items, a sample authorised only for
  fixing the budget, and degeneration stayed high (17/20 → 14/20) rather than
  falling with it. Direction is consistent with GSM8K's `early_candidate` minimum
  at the same workpoint.
- **Do NOT write "externalized reasoning emergence increased."** Text before the
  marker is not necessarily reasoning, and no frozen content-judgement rule
  exists. Report morphological fields only: `answer_first`, `pre_marker_chars`,
  `first_marker_pos`, `multi_marker` / `degenerate` rate. **`first_marker_pos` is
  AUXILIARY** — a longer repetition tail lowers it independently of anything
  preceding the marker, so it cannot carry a "more reasoning" reading alone.
  Qwen's 0.94–0.98 means the marker is LATE, **not** that the preceding text is
  complete or correct reasoning.
- **Descriptive fields are PREFLIGHT-INFORMED SECONDARY**: outside the Holm
  family, and themselves outcomes of α, so stratifying accuracy on them is
  post-treatment stratification — consistent-with evidence, never mediation.
- **Statistics**: primary `ΔAcc` on MAIN parsing paired per item; exact two-sided
  McNemar with discordant counts; question-level paired bootstrap 95% CI
  (B=10000, seed 0); **Holm over the TWO MODELS (m=2), judged only when both are
  complete** — a partial family reports an `m=1` adjustment under an `m=2` label.
- **`p4_amendment_04.json` (the stage-1 budget freeze) is REFERENCED BY `p4-amend-02` and `p4-amend-05` BUT IS NOT ON DISK (verified 2026-09-03).** It was produced by the earlier warning-then-write decider build, which warned on llama α=−6's 19/20 AND wrote the artifact anyway; re-running the current gate-corrected code reproduced it with NO differing keys, which is why it was kept rather than renumbered. That reproduction is the only surviving evidence — **the file itself is absent, so the numbering is preserved by the two amendments that name it, not by an artifact.** Do not renumber `04` and do not silently regenerate it; if it is ever needed, regenerate it deliberately and record that it is a reconstruction. The decider is **`decide_p4_budget.py`** (named here 2026-09-04; the rule was described but the script was not). Its rule is MECHANICAL — if any of the 20 outputs in any of the 4 preflight cells has generated length ≥511 the formal budget becomes 1024, else it stays 512 — with no per-model judgement and no accuracy anywhere; it fails closed on a missing model, a wrong α set, a cell that is not exactly 20 items, a prompt-hash or preflight-digest mismatch between the two files, disagreeing budget parameters, an unverified-steering cell, or an existing output. The decider defect is fixed at `407e6ab`. Provenance in `p4-amend-06`.
- **Two frozen-artifact rules this line established the hard way.** A generator of
  frozen artifacts must ship **no `--allow_overwrite` escape hatch** — "refuses to
  overwrite" is not true while a flag bypasses it. And a **format violation is a
  hard stop BEFORE any write**: warning-then-write both downgrades the rule and
  leaves a stage-1 freeze derived from non-conforming output on disk.
- **A launcher bug worth not repeating:** `${1:?usage: ... {llama3|qwen2.5}}`
  ends the parameter expansion at the FIRST `}`, so `MODEL` became the literal
  `llama3}`. `bash -n` is clean on this — the syntax is valid, the semantics are
  not. Always invoke a launcher with real arguments, not just a syntax check.
- **`MODEL_DIR` is an HF REPO ID and the mask dir is `${MODEL}_non_logits`** (not
  `_nmd_logits`), matching every other launcher; there is no filesystem model
  tree. A nonexistent path is handed to transformers, which parses it as a repo
  id and raises an `HFValidationError` naming neither the launcher nor the
  missing directory. All cheap checks now run BEFORE the model loads.

```bash
# Server, from /data1/paveen/Dopamine. One model per card; both cells of a
# model stay together (bf16 greedy is not byte-reproducible across GPUs and the
# two cells are a paired contrast). The two MODELS may run on two cards.
python data_logiqa2.py --out_dir components/benchmark   # digest must read 4d4b25e071a2a6dd

CUDA_VISIBLE_DEVICES=0 nohup bash run_logiqa2_formal.sh llama3  > p4_formal_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_logiqa2_formal.sh qwen2.5 > p4_formal_qwen.log  2>&1 &
cat p4_formal_llama.log   # immediately -- a wrong PY exits 127 and the log looks empty

# Scoring is a SEPARATE script and the only one that reads gold. Run it once,
# after BOTH models finish (a single model is scored with Holm WITHHELD).
python eval_logiqa2.py \
  --generations components/logiqa2/formal/formal_llama3.json \
                components/logiqa2/formal/formal_qwen2.5.json \
  --formal_file components/benchmark/logiqa2_p4_formal.json \
  --out docs/p4_logiqa2_evaluation.json
```


### P4b stage-0, audit, amend-01 (former CLAUDE.md lines 1767–1851)

### STAGE-0 RESULT (`object_counting`, 2026-09-02, `docs/bbh_p4b_object_counting_stage0.json`)

**Both models PASS the frozen `[0.30, 0.85]` headroom gate on α=0 `first_acc`:
llama3 `.4160` (last `.4120`), qwen2.5 `.5520` (last `.5600`).** Both sit ~4–5×
the `.104` majority-class rate and 30–43 pp below the ceiling, so a later null on
either model **cannot** be a baseline-ceiling artifact — the only thing this gate
exists to establish. The majority-class rate stayed descriptive and did not move
the interval.

**The public ≈`.90` figure is NOT comparable to `.416`** — it uses BBH's 3-shot
CoT format, which this protocol deliberately does not inherit. It explains why a
headroom gate was set at all, nothing more.

**Diagnostics — recorded, NOT a gate, and not grounds to change prompt, parser or
budget.** Llama shows a looping/truncation **PHENOTYPE SIMILAR TO GSM-Hard**;
similar phenotype, **not** an established shared mechanism. cap-hit `.956` (decode
median exactly 768), no-marker `.092`, multi-marker `.840`, chars med 1988. Qwen
does not loop (cap-hit `.036`, chars med 423). **Consequence:** Llama's accuracy
stays valid as a **fixed-768-budget estimand**, but its length-derived readouts
press against that cap and are **not natural lengths**; if α moves truncation or
marker-absence, both travel with the main result. `last_acc` sensitivity is
retained even though first/last currently agree to ≤0.8 pp.

**Gold provenance:** the server gold is byte-identical to a locally regenerated
copy (`questions_sha256 4cfbf739e1fe7870`, matching both generation cells), so
identity is proved by digest rather than assumed.

### EARLYCAND AUDIT: PASS, with a ceiling limitation that constrains its use

Order preserved: 30 items frozen from the blind file → blind CSV with **no**
detector flag / gold / accuracy → manual labels written → **only then** was
`earlycand-v1` run. **Precision 1.000, recall 1.000, agreement 30/30, detector
positives 29** — clearing the frozen minimum of 10, so the `INCONCLUSIVE` branch
does not fire. The single FALSE (`id 39`) is the one generation opening with
working rather than a bare number; the other 29 open with a standalone digit and
justify afterwards.

<!-- Why: the audit validates the detector but its baseline is saturated, so the
readout can only fall; reading a flat rate as "α does not move timing" is the
baseline-ceiling error the accuracy gate exists to exclude, relocated to timing.
Evidence: docs/p4b_earlycand_audit_result_object_counting.json.
Scope: every P4b early_candidate number on object_counting. -->
**CRITICAL: the α=0 base rate is at CEILING (llama `.952`, qwen `1.000`).** The
audit shows the detector reads this task correctly, but a saturated baseline
leaves headroom only for a **decrease**. **A flat rate under α must NOT be read as
"α does not move answer-formation timing."** Scope: **EXPLORATORY, outside Holm**;
`early_candidate` is an OUTCOME of α, so stratifying accuracy on it is
post-treatment stratification — consistent-with evidence, never mediation. The
detector was not re-tuned. Labels come from **one annotator**; the 29/30 split is
morphologically stark but no inter-rater agreement was measured.

### `p4b-amend-01`: one opposite-signed DIAGNOSTIC cell per model

Additive. The primary test is unchanged — fixed-workpoint transfer, α read from
the frozen GSM8K record (llama `−6`, qwen `+8`), **Holm m=2 over the two workpoint
contrasts only**. Adds ONE reverse dose per model (**llama `+4`, qwen `−6`**) to
read whether the direction ordering continues (`−6 > 0 > +4`; `+8 > 0 > −6`).

**Binding:** the reverse cell is **outside the Holm family**, its p is
**unadjusted**, and it **MUST NOT redefine the workpoint** — a reverse cell that
scored higher would still not become the workpoint, because this protocol never
searches doses on BBH. `run_bbh_numeric.sh REVERSE` refuses until **both** the α=0
and the workpoint cell exist, so it cannot be mistaken for the search that
produced the workpoint. **One point is not a curve:** it can show an ordering
continues or breaks, never a peak, an inverted-U, or an "overshoot point".

<!-- Why: with two steered cells `[x for x in byalpha if x != 0][0]` returns the
SMALLEST alpha, which for qwen is the DIAGNOSTIC -6 -- it would have been reported
as the primary transfer result while the real workpoint went unreported.
Evidence: eval_bbh_numeric.py workpoint selection; verified on a fixture where the
reverse dose beats the workpoint and +8 still occupies the Holm table.
Scope: any scorer holding more than one steered cell per model. -->
**A latent hazard the reverse cell exposed:** the scorer selected the steered α as
`[x for x in byalpha if x != 0][0]`, fine with one steered cell but wrong with
two. It now selects `WORKPOINT[model]` **explicitly**. Verified on synthetic
fixtures including the case that matters — a qwen fixture where the reverse dose
(`.696`) BEATS the workpoint (`.532`) still puts `+8` in the Holm table, reports
`−6` separately as `BREAKS` with an unadjusted p, and does not promote it. An
unfrozen α (`−4`) is refused; Holm stays m=2 with reverse cells present.

**All THREE cells of one model stay on that model's original physical GPU**
(llama GPU 0, qwen GPU 1) — paired per-item contrasts, and bf16 greedy is not
byte-reproducible across GPUs. `WORKPOINT` and `REVERSE` of one model run
**sequentially** (same card, and `REVERSE` checks the workpoint cell exists).


### P4b protocol / runbook (former CLAUDE.md lines 1980–2132)

**P4b IS NOT A NEW PHASE — it is the SECOND TASK of P4's question.** It was
briefly numbered `P5`; that was renamed before any artifact existed, because
`P5` implied a stage that does not exist. Do not renumber it back.

**Why it exists.** P4's LogiQA null cannot say *why*, because LogiQA changed two
things at once relative to GSM8K:

| | answer space | reasoning content | submission |
|---|---|---|---|
| GSM8K | model constructs an integer | arithmetic word problems | `####` |
| LogiQA 2.0 | choose among 4 **given** candidates | textual logic | `Final answer: X` |
| **BBH numeric** | model constructs an integer | counting / nested arithmetic | `####` |

The two BBH tasks restore the answer space and the submission interface to
GSM8K's. **That table compares three CHOSEN dimensions and is not exhaustive** —
BBH also differs in item format, domain, text distribution and generation shape,
so this is a between-task comparison, not a controlled single-factor
manipulation.

**FROZEN OUTCOME WORDING, both weaker than a causal claim.** Transfer → "the
workpoint transferred to an option-free numeric reasoning task, **consistent
with** the choice interface being part of the LogiQA boundary" — **never**
"LogiQA failed because of the options". Null → "removing the option interface
was **not sufficient** to restore transfer", which does **not** establish that
reasoning content is the binding constraint. **Separating a choice-interface
effect from a reasoning-type effect needs a within-item contrast (the same items
with and without options); that is a different experiment and is not authorised
here.**

- **Task tier is fixed in advance so task choice is not a post-hoc narrative**:
  `object_counting` → `multistep_arithmetic_two` (only if the first fails the
  gate for BOTH models) → `dyck_languages`, the last **unimplemented and
  unauthorised** (it changes the answer space to a symbol string and needs its
  own parser, prompt and gate). `--task` is an allowlist of the two numeric
  configs; a third cannot be smuggled through it.
- **`lukaemon/bbh` pinned to `982bb89fd79532a8ac676a61fc42eb1aeec63f99`**, `test`
  split, **all 250 items, no sampling** (so no salt and no selection step).
  `maveriq/bigbenchhard` is script-based and would hit current `datasets`'
  rejection of loading scripts — the same trap SIQA hit.
- **Measured at freeze:** `object_counting` 17 gold values 2…18, majority-class
  **.104**, digest `4cfbf739e1fe7870`; `multistep_arithmetic_two` 185 values
  −39960…250992, majority-class **.016**, digest `e69d300b94274ce3`. Both
  all-integer gold, so `normalize_gsm8k` applies unchanged.
- **Stage-0 gate is `α=0 first_acc ∈ [0.30, 0.85]`, judged PER MODEL, and the
  interval is FROZEN.** Its purpose is not to test α but to confirm a later null
  is interpretable rather than a baseline-ceiling artifact (the Qwen-MMLU-betting
  and pv6-Easy-bare failure). **The majority-class rate is recorded but does NOT
  move the interval** — it is the trivial constant-guess baseline, not a
  random-guess rate, and letting it adjust the gate after inspecting the data
  would make the gate itself adjustable. At .104/.016 it is moot, which is a fact
  about the data, not a licence to revisit the rule.
- **Format/truncation diagnostics are NOT a gate.** They exist only to confirm
  the 768 budget and the shared extractor did not fail *technically*. Seeing the
  output shape must not lead to a changed prompt, a redefined parser, or a
  re-tuned budget.
- **Per-model eligibility, but the two-model panel is the design.** Both eligible
  → Holm m=2. Exactly one → its workpoint cell still runs, pre-specified as
  **single-model exploratory transfer** with Holm **WITHHELD** and the raw p
  labelled unadjusted; running Holm at m=1 under an m=2 label is the error this
  prevents. Neither → no workpoint cell, move to the next task in the tier.
- **α is expressible ONLY as 0 or the frozen GSM8K workpoint** (llama `−6`, qwen
  `+8`). The launcher cannot name another dose and the scorer rejects one.
  `WORKPOINT` also refuses to run before the α=0 cell exists — the gate is judged
  on α=0, so running the workpoint first would make the gate unfalsifiable.
- **`cot=False` and `wording="plain"` are HARDCODED in the runner, not flags.**
  BBH ships 3-shot CoT prompts; inheriting them would add exemplars and explicit
  reasoning, re-mixing the variables this design exists to separate. The "pushy"
  wording is a known early-`####` inducer. Neither has a cell here.
- **Not a blind validation, and the files say so.** BBH gold is public; P3 was
  the blind test and is closed. What carries over is the OTHER half of the
  discipline: α read from the frozen record and never re-searched, predictor /
  features / marker adapter untouched, generation and scoring as separate
  scripts, sample frozen before any steered cell. The sample is still emitted as
  a blind copy (field whitelist, then asserted label-free) plus a gold copy, so
  generation cannot reach a label even by mistake.
- **`earlycand-v1` must be re-validated on this task BEFORE it may be called a
  commitment metric.** It was frozen on GSM8K, where a short first line with a
  number is answer-shaped; on `object_counting` the question IS a list of objects
  and the reasoning is a running count, so a leading number may merely restate
  the question. Its GSM8K blind audit (precision 1.000, recall .976) was on
  arithmetic text and does not transfer by assumption.
  `freeze_p4b_earlycand_audit.py` fixes **30 items from the frozen question
  digests before stage-0 generation** (ranked by salted `sha256` of the question
  text — never `hash()`, which is process-salted; `object_counting` selection
  digest `04508a19ce30b0b6`), because choosing them after seeing detector output
  would let the sample flatter the detector. The rubric is applied **without
  seeing the detector flag**; comparison happens only afterwards.
  **Three outcomes, all frozen**: pass → exploratory timing readout, outside
  Holm; fail → withdrawn as a timing metric, only marker/format description
  survives; **fewer than 10 detector positives among the 30 → INCONCLUSIVE**,
  report the raw counts (detector positives / manual positives / agreements) and
  claim no precision in either direction. In every case the accuracy main test is
  unaffected (it never uses the detector), the sample is not re-drawn or enlarged
  to chase positives, and **the detector is NOT re-tuned** — that would fork the
  definition against every stored GSM8K and MATH number.
- **`early_candidate` is an OUTCOME of α**, so stratifying accuracy on it is
  post-treatment stratification: consistent-with evidence, never mediation.

```bash
# Server, from /data1/paveen/Dopamine. Steps 1-2 precede ANY generation.
python data_bbh_numeric.py --task object_counting --out_dir components/benchmark
python data_bbh_numeric.py --task multistep_arithmetic_two --out_dir components/benchmark
#   digests must read 4cfbf739e1fe7870 / e69d300b94274ce3
python freeze_p4b_earlycand_audit.py --task object_counting --bench components/benchmark
#   selection_digest must read 04508a19ce30b0b6

# Stage-0: the two MODELS may share nothing but the task -- they are never
# compared per item, so they parallelise. A model's own two cells must stay on
# ONE card (paired per-item contrast; bf16 greedy is not byte-reproducible).
CUDA_VISIBLE_DEVICES=0 nohup bash run_bbh_numeric.sh llama3  object_counting STAGE0 > p4b_oc_l_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_bbh_numeric.sh qwen2.5 object_counting STAGE0 > p4b_oc_q_s0.log 2>&1 &
cat p4b_oc_l_s0.log      # immediately -- a wrong PY exits 127 and the log looks empty

# The gate, and the ONLY script that reads gold:
python eval_bbh_numeric.py \
  --generations components/llama3/bbh/object_counting/mdf_0/bbh_object_counting_8B_11_20.json \
                components/qwen2.5/bbh/object_counting/mdf_0/bbh_object_counting_7B_16_22.json \
  --gold_file components/benchmark/bbh_p4b_object_counting.json \
  --out docs/bbh_p4b_object_counting_stage0.json

# WORKPOINT only for models that printed PASS, and only after the manual audit.
# Then REVERSE (p4b-amend-01), which REFUSES until alpha=0 AND the workpoint
# cell exist. Both cells of one model are SEQUENTIAL on that model's own card.
CUDA_VISIBLE_DEVICES=0 nohup bash run_bbh_numeric.sh llama3  object_counting WORKPOINT > p4b_oc_l_wp.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_bbh_numeric.sh qwen2.5 object_counting WORKPOINT > p4b_oc_q_wp.log 2>&1 &
# after each finishes, on the SAME card:
CUDA_VISIBLE_DEVICES=0 nohup bash run_bbh_numeric.sh llama3  object_counting REVERSE > p4b_oc_l_rv.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_bbh_numeric.sh qwen2.5 object_counting REVERSE > p4b_oc_q_rv.log 2>&1 &

# steering_fires must read L*250 at alpha!=0: llama 2250 (L=9), qwen 1500 (L=6).
# The runner asserts this itself; 0 means the hook never fired.

# Score all SIX cells at once. The scorer accepts only three alphas per model
# (0, the frozen workpoint, the frozen reverse dose) and refuses anything else.
python eval_bbh_numeric.py \
  --generations components/llama3/bbh/object_counting/mdf_{0,neg6,4}/bbh_object_counting_8B_11_20.json \
                components/qwen2.5/bbh/object_counting/mdf_{0,8,neg6}/bbh_object_counting_7B_16_22.json \
  --gold_file components/benchmark/bbh_p4b_object_counting.json \
  --out docs/bbh_p4b_object_counting_result.json
```

**Runtime, measured off the stage-0 length distributions:** Llama ≈25–40 min per
cell (250 items, 95.6% running to the 768 cap), Qwen ≈5–10 min (median 131
tokens). Sequential per model: Llama ≈1–1.5 h for its two cells, Qwen ≈15–20 min;
with the two models on two cards the wall clock is Llama's. α can move length in
either direction, so treat these as estimates.

**Prior worth knowing before reading the gate**: public reports put
Llama-3.1-8B near **.90** on `object_counting` under *their* prompt, budget and
extractor — not comparable to ours, but enough that a `> .85` ceiling failure is
a live outcome, not a formality. If it fails, `multistep_arithmetic_two`'s sample
is already frozen; switch `--task`, no code change.


### P4c protocol / runbook (former CLAUDE.md lines 2153–2268)

**Data.** `cruxeval-org/cruxeval`, revision pinned `b96af0450242eb4da433032b90998f25588a5d0f`,
800 rows, columns `code/input/output/id`. **300 selected** by `sha256(salt:id:code:input)`
— never `hash()`, which is process-salted. Digests `4580b7a9a9ef6054` (questions) /
`a214d1fc7d84a2d9` (gold). Measured and asserted at freeze: **800/800 gold `literal_eval`,
0 gold contain a newline, `####` appears nowhere in the source** — those three facts are
what make a single-line `#### <literal>` marker safe, and the loader checks them rather
than assuming them.

**Matrix: 8 cells.** llama `0/−6/−4/+4` (band 11–20), qwen `0/+8/+6/−6` (band 16–22).
**Holm m=2 covers ONLY the two workpoint contrasts** (llama −6, qwen +8). The neighbour
(`−4`/`+6`) and reverse (`+4`/`−6`) cells are reported always, sit **outside Holm with
unadjusted p**, and **MUST NOT redefine the workpoint** — α is read from the frozen GSM8K
record and never re-searched. Four sampled doses are not a dose-response curve.

<!-- Why: the official CRUXEval scorer execs model output; a later reader may "restore
fidelity" by porting it. The gap it would close is already reported as nonliteral_rate.
Evidence: docs/PREREG_P4C_CRUXEVAL.md §5; test_cruxeval_p4c.py exec-payload guards.
Scope: P4c scoring only. -->
- **SCORING IS `ast.literal_eval` + PYTHON OBJECT EQUALITY, NOT the official
  execution-based `pass@1`.** Official CRUXEval runs
  `exec(f"{code}\nassert {gold} == {candidate}")`, interpolating **model-generated text**
  into an executed expression. This protocol never does that, and **`reliability_guard`
  is not a security sandbox** — do not describe it as one. The recorded gap: a candidate
  that is a non-literal *expression* evaluating correctly (`[1]*3`) scores **incorrect**
  here and would score correct officially; its size is reported per cell as
  `nonliteral_rate` rather than assumed. The test suite pins the security property by
  feeding `__import__('os').system(...)` and asserting it neither parses nor executes.

<!-- Why: the parser was right BY COINCIDENCE before amend-01 -- '#' opens a Python
comment, so literal_eval silently truncated a trailing '####'. 5 of the 300 gold values
contain '#', so the coincidence was unsafe.
Evidence: docs/p4c_amendment_01.json; the REGRESSION checks in test_cruxeval_p4c.py.
Scope: P4c parsing. -->
- **`p4c-amend-01` strips TWO decoding artifacts, and the second one is subtle.** (a) a
  trailing EOS token text — Qwen returned `[1, 1, 1, 1]<|endoftext|>`, a correct answer
  failing to parse purely because the decoder surfaced EOS (5 of 8 preflight payloads per
  Qwen cell). (b) a trailing `####` occurring **outside** a string literal — Llama writes
  `#### <literal> ####` then loops. **Without (b) the payload parsed anyway**, because `#`
  opens a Python comment and `literal_eval` truncates there — correct by coincidence, and
  unsafe because **5 of the 300 gold values contain `#`** (`sample_755` is
  `'ph>t#A#BiEcDefW#ON#iiNCU'`). The cut is quote-parity-guarded, so a `####` **inside** a
  string literal is preserved; all five `#`-bearing gold values were verified to round-trip.

<!-- Why: the tempting move is to widen the parser or switch FIRST->LAST, and both
would hide a steering effect: qwen format obedience is 8/8 at a=0 but 6/8 at +8.
Evidence: docs/p4c_amendment_02.json (no code change); REGRESSION amend-02 checks.
Scope: P4c parsing; the same logic applies to any task where format obedience is dosed. -->
- **`p4c-amend-02` is a decision NOT to change code, and it is the load-bearing one.**
  Two Qwen `+8` preflight payloads still fail: prose after the payload
  (`#### 'ohesteo' The function f removes…`), and a **bare `####`** followed by a second
  marker carrying the answer. Both are the **MODEL failing the frozen "exactly one line"
  format**, not decoding artifacts. **Widening the parser would systematically hide a
  steering effect** — Qwen reads 8/8 at α=0 and 6/8 at α=+8, so the failures are
  dose-specific. **Both parse under LAST and fail under FIRST**, which is exactly the shape
  that tempts a caliber switch: **FIRST stays MAIN** (frozen, matching GSM8K/GSM-Hard/BBH
  production), and the difference is already visible as `nonliteral_rate` + `last_acc`.
  The rule generalises: *a decoding artifact is a parser bug; a model failing the format
  is a result.*

- **PREFLIGHT is FORMAT-ONLY and structurally cannot become an accuracy read.**
  `--preflight` forces n=8, writes to a separate `_preflight/` tree with a `preflight`
  filename prefix and meta flag, and **the scorer refuses any preflight cell**. It checks
  the frozen digest and the 300-item count on the FULL file *before* truncating — the
  reverse order would let a preflight run on anything.

- **PREFLIGHT MEASURED (2026-09-03, after amend-01):** marker 8/8 in all four cells;
  literal 8/8 / 8/8 / 8/8 / **6/8** (llama 0, llama −6, qwen 0, qwen +8);
  `steering_fires` 0 / 72 / 0 / 48, matching `L×n`. **Llama is truncation-bound**:
  capped 8/8 and 7/8, `chars` median ≈2450, tails degenerating to `[/PYTHON] [/PYTHON]…`.
  Qwen is not (1/8, median 50/241).

<!-- Why: GSM-Hard froze this wording after a ceiling effect was nearly read as an
observation; llama P4c sits against the same 768 cap.
Evidence: the P3 GSM-Hard truncation bullet; preflight capped 8/8 and 7/8.
Scope: every llama P4c length or accuracy claim. -->
- **Llama accuracy is a FIXED-768-BUDGET ESTIMAND.** The paired comparison stays valid
  (every cell has the same budget), but it may **not** be read as unconstrained reasoning
  capability, and llama's length-derived readouts are **not natural lengths**. The answer
  completes *before* the degenerate tail, so accuracy is readable — this is **not** a
  Mistral-style extraction floor. **Do not raise the budget now**: that changes the frozen
  primary question. Per the P3-supp precedent, finish 768 first and build a larger-budget
  sensitivity separately if needed, never substituting it for the main result.

- **NO ACCURACY GATE, deliberately** — unlike P4b's `[0.30, 0.85]`. A low baseline is a
  limitation on the reading, not a cancelled test; this removes the failure mode where a
  gate interval becomes adjustable after seeing data. **Hard stops are technical only**
  (six, listed in §7), and a hard stop is a stop — not a licence to redesign the prompt,
  parser or budget.

- **Cells are NOT required to share a GPU** (per the repo-wide rule); `host` and
  `CUDA_VISIBLE_DEVICES` are recorded as provenance, and a cross-device contrast is
  reported as a **cross-run pairing**. Pairing means alignment by the frozen item order.

```bash
# Server, from /data1/paveen/Dopamine. Loader first -- digests must match.
python data_cruxeval.py --out_dir components/benchmark

# Format-only preflight (no accuracy exists to read), then inspect:
CUDA_VISIBLE_DEVICES=2 bash run_cruxeval.sh llama3  PREFLIGHT
CUDA_VISIBLE_DEVICES=3 bash run_cruxeval.sh qwen2.5 PREFLIGHT
python inspect_p4c_preflight.py --root components
python dump_p4c_nonliteral.py --root components --preflight   # classify failures

# Formal cells. The two MODELS may run on two cards.
CUDA_VISIBLE_DEVICES=2 nohup bash run_cruxeval.sh llama3  ALL > p4c_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash run_cruxeval.sh qwen2.5 ALL > p4c_qwen.log  2>&1 &
cat p4c_llama.log      # immediately -- a wrong PY exits 127 and the log looks empty

# Score ONCE, after BOTH models finish (a single model withholds Holm).
python eval_cruxeval.py \
  --generations components/llama3/cruxeval/mdf_{0,neg6,neg4,4}/cruxeval_o_8B_11_20.json \
                components/qwen2.5/cruxeval/mdf_{0,8,6,neg6}/cruxeval_o_7B_16_22.json \
  --gold_file components/benchmark/cruxeval_p4c_formal.json \
  --out docs/p4c_cruxeval_evaluation.json
```


