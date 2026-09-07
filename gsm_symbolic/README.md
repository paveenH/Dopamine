# GSM-Symbolic task-specific steering workpoint exploration

Numeric/template robustness check of the frozen GSM8K workpoints
(llama3 `-6`, qwen2.5 `+8`) under `apple/GSM-Symbolic`
(https://huggingface.co/datasets/apple/GSM-Symbolic). **This is a robustness
check of the GSM8K workpoint under numeric perturbation, not an independent
cross-domain transfer test** — GSM-Symbolic is generated from GSM8K's own
reasoning templates, re-instantiated with different symbolic values (`main`)
plus one (`p1`) or two (`p2`) extra clauses.

**Scale: this runs the FULL official test split per config, NOT a fixed
300-item sample.** Measured (Hub dataset-viewer, 2026-09): `main` ~1319,
`p1` ~5000, `p2` ~2500 rows → ~8819 items × 4 alphas = ~35,276 generations per
model, ~70,552 for both models combined. Budget GPU time accordingly.

Self-contained under `gsm_symbolic/`; no existing GSM8K/MATH/other-task
runner, loader, or launcher is modified.

## Files

- `data_gsm_symbolic.py` — loader. Downloads all 3 official test-split configs
  (`main`, `p1`, `p2`, FULL split each) from `apple/GSM-Symbolic`, writes one
  JSON per config plus a combined 30-item preflight subset (10/config,
  deterministic stride sample by `(original_id, instance, id)`). Preserves the
  **official HF row `id`** verbatim (not a re-enumerated local index) and adds
  `sample_id = "{config}:{id}"` (unique across all 3 configs combined, since
  raw `id` repeats across configs). Gold is extracted via the same
  `#### <number>` convention `utils.extract_gsm8k_answer` uses (kept local so
  the loader has no torch/heavy import). Records both `revision_requested`
  (unpinned, `None`, since no verified commit SHA was available to hand-pin)
  and `revision_resolved` (best-effort read-back of what was actually
  fetched) in every output file's meta. `--check` validates schema and writes
  nothing; a load failure (e.g. if `main` turns out not to load) is a HARD
  STOP with the real traceback — it is never caught and silently skipped.
- `get_answer_gsm_symbolic.py` — generation + scoring driver. Fork of
  `get_answer_regenerate_gsm8k.py`; imports `utils.extract_gsm8k_answer`,
  `utils.is_correct_gsm8k`, `utils.parse_configs`, `utils.decoder_layer_range`,
  `llms.VicundaModel`, and
  `template.select_templates_gsm8k(suite="default", cot=True, wording="plain")`
  — the existing, frozen explicit-CoT GSM8K prompt, byte-identical to the main
  GSM8K line. Role fixed to `neutral`. One `--gsm_config` per call. **Hard
  checks (raise, not warn):** `len(batch_out) == len(batch_prompts)` after
  every batch and `len(outs) == len(samples)` after all batches; and
  `steering_fires == 0` at α=0 / `== n_layers_band × n_samples × tail_len(=1)`
  at α≠0, computed from `utils.decoder_layer_range` and compared against the
  value `VicundaModel.steering_fire_count()` actually recorded. Output meta
  also carries `mask_sha256`, `prompt_template_sha256`, `n_layers_band`,
  `steering_fires_expected`, and the full `sample_ids` list, so the evaluator
  can verify cross-cell consistency before computing any statistics.
- `eval_gsm_symbolic.py` — scoring/statistics. Re-derives correctness via the
  frozen extractor (never trusts the inline `correct` field). Runs a
  **cross-cell consistency check FIRST, before any statistics**: every cell
  for a model must agree on model/model_dir/size/prompt(+hash)/cot/role/
  generation config; every non-zero-α cell's `mask_sha256` must match every
  other non-zero-α cell of that model; every cell's recorded
  `steering_fires` must equal its own `steering_fires_expected`; and every
  config's 4 α cells must cover the identical `sample_id` set. Any mismatch
  is a hard stop.
  **Statistics design (revised — clustered, not row-level):** `p1`/`p2` each
  re-instantiate the SAME `original_id` multiple times with different
  symbolic values (p1 ~5000 rows over far fewer distinct `original_id`, p2
  ~2500 similarly), so rows sharing an `original_id` are correlated
  re-samples of one underlying template, not independent trials. Treating
  all ~8819 pooled rows as independent Bernoulli draws (what exact McNemar
  assumes) manufactures pseudo-replication and inflates significance, and a
  plain row-weighted pool lets `p1` (the largest config by row count)
  dominate a result meant to summarize three configs equally. Fixed as
  follows:
  - **PRIMARY**: a paired **cluster bootstrap** resampling `original_id`
    clusters *within* each config (main/p1/p2 never mixed during
    resampling), then combining the three configs' deltas with **equal
    weight per replicate** — so `p1`'s row count cannot dominate. Reports a
    95% percentile CI (the primary evidence) plus a two-sided bootstrap
    p-value, Holm `m=3` over the model's three non-zero doses. Only this
    family (`pooled_main_p1_p2_PRIMARY_cluster_bootstrap`) gets an
    `established_workpoint` verdict.
  - **SENSITIVITY, not primary evidence**: the naive per-row exact McNemar
    (stdlib binomial CDF, no scipy dependency) on the pooled rows, reported
    explicitly labeled `SENSITIVITY_ONLY`, to see whether the cluster-aware
    and naive-row verdicts diverge — never cited as the significance result
    on its own.
  - **DESCRIPTIVE per (model, config)**: accuracy at each α plus the same
    row-level McNemar (own block, no Holm, no significance verdict) and
    per-`original_id` accuracy mean/std, so a config's headline number isn't
    read off one instantiation.
  Also reports diagnostics (`no_marker`, `marker_unparsed`, `no_answer`,
  multi-marker, first/last disagreement, loop, truncation, generation
  length — see the no_answer vs no_marker note below).
- `run_gsm_symbolic_preflight.sh` — α=0 only, 30-item preflight, one model per
  invocation. `BATCH_SIZE` (env-overridable, default 24, same value as the
  formal sweep) is ONE shared knob for the whole script.
- `run_gsm_symbolic_formal.sh` — full 4-point sweep over the complete official
  test split of all 3 configs, one model, one GPU, sequential. Same
  `BATCH_SIZE` convention: if a cell OOMs, lower it for the WHOLE model and
  re-run that model's full sweep — never drop it for one config/α only.
- `sanity_check_eval.py` — offline sanity check (exact McNemar edge cases,
  Holm monotonicity, `score_cell`/`per_instance_breakdown` against hand-built
  generations run through the real frozen extractor, and
  `check_cell_consistency` verified to both pass on matching meta and reject
  prompt drift / fires mismatch / sample_id drift). No model, no server, no
  network. Run with `python3.10 sanity_check_eval.py`.

## `no_answer` vs `no_marker` vs `marker_unparsed`

`extract_gsm8k_answer` has a fallback chain (`####` marker → "the answer is
X" → `\boxed{}` → last number in the text), so a generation with **no**
`####` marker at all can still score correct via the fallback. Three
distinct fields are recorded to avoid conflating "no marker" with "no
answer":
- `no_marker` — the text contains no `####` substring at all (pure format
  diagnostic; the fallback chain may still recover a number).
- `marker_unparsed` — `####` is present but the digits after it never parsed.
- `no_answer` — the FULL fallback chain found nothing at all. Accuracy always
  uses the frozen scorer unmodified; these fields describe generation
  behavior, they never change scoring.

## Design decisions carried over from the existing GSM8K/GSM-Hard/P3-supp lines

- **Same-GPU-per-model-curve rule**: bf16 greedy is not byte-reproducible
  across GPUs, so each model's full 4-alpha × 3-config sweep runs on one
  pinned card, sequentially.
- **Prefill-only, tail=1, bare-string, greedy** (`vc.regenerate` defaults;
  `temperature=0.0`).
- **Model IDs are HF repo IDs** (`meta-llama/Llama-3.1-8B-Instruct`,
  `Qwen/Qwen2.5-7B-Instruct`), matching every other GSM8K/GSM-Hard launcher.
- **Mask paths** follow the existing convention:
  `{base_dir}/mask/{hs}_{type}_logits/{mask_type}_{percentage}_{start}_{end}_{size}.npy`
  — i.e. `mask/llama3_non_logits/nmd_0.5_11_20_8B.npy` and
  `mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy`. Not guessed; copied from
  `run_gsm8k.sh` / `run_gsm8k_qwen25.sh`.
- **`max_new_tokens=768`, `batch_size=24`** — same as the frozen GSM8K main
  line for both models (this project's `max_new_tokens` convention is
  per-*task*, not per-*model*; GSM-Symbolic keeps GSM8K's own budget since it
  is a numeric perturbation of the same task, unlike MATH's 2048/bs=8).
- **`#### <numeric answer>` final format**, explicit CoT (`Let's think step by
  step.` line, from the existing `cot=True` GSM8K template) — no new prompt
  protocol.
- **Scoring reuses `utils.extract_gsm8k_answer` / `utils.is_correct_gsm8k`
  verbatim** via `sys.path` insertion of the repo root; no parser is
  reimplemented anywhere in this directory.
- **No LLM judge anywhere.**

## Output layout (server, `${WORK_DIR}/components`)

```
benchmark/gsm_symbolic/
  gsm_symbolic_main_test.json    # FULL test split, main config
  gsm_symbolic_p1_test.json      # FULL test split, p1 config
  gsm_symbolic_p2_test.json      # FULL test split, p2 config
  gsm_symbolic_preflight_30.json # 10 items/config, deterministic subset

{model}/answer_gsm_symbolic_preflight/{main,p1,p2}/mdf_0/
  gsm_symbolic_{cfg}_{size}_{ls}_{le}.json      # preflight, alpha=0 only

{model}/answer_gsm_symbolic/{main,p1,p2}/mdf_{alpha}/
  gsm_symbolic_{cfg}_{size}_{ls}_{le}.json      # formal sweep
  summary_gsm_symbolic_{model}_{size}_{ls}_{le}.csv
```

## Two-step workflow

1. **Verify the data first** (in particular `main`, which the Hub
   dataset-viewer's config table did not list at inspection time, though the
   dataset card documents `load_dataset("apple/GSM-Symbolic", "main")` and a
   `main/test` parquet export exists):
   ```bash
   python data_gsm_symbolic.py --check
   ```
   If any config fails to load, this stops with a real traceback — it is
   never silently dropped from the experiment matrix.

2. **Preflight (alpha=0 only, 30 items, both models on the same file)**:
   ```bash
   python data_gsm_symbolic.py --out_dir components/benchmark/gsm_symbolic --preflight_n 10
   CUDA_VISIBLE_DEVICES=0 bash run_gsm_symbolic_preflight.sh llama3
   CUDA_VISIBLE_DEVICES=1 bash run_gsm_symbolic_preflight.sh qwen2.5
   ```
   Stops here. Inspect `meta.steering_fires` (the driver itself now hard-fails
   if it disagrees with `steering_fires_expected`), `no_marker`/`no_answer`,
   `truncated`, and a few raw `generated` strings per config before running
   the formal sweep.

3. **Formal sweep (after approval)**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_symbolic_formal.sh llama3  > gsms_llama.log 2>&1 &
   CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_symbolic_formal.sh qwen2.5 > gsms_qwen.log  2>&1 &
   # after both finish:
   python eval_gsm_symbolic.py --model llama3  --base_dir components --out gsms_llama3_eval.json
   python eval_gsm_symbolic.py --model qwen2.5 --base_dir components --out gsms_qwen25_eval.json
   ```
   `eval_gsm_symbolic.py` checks cross-cell consistency before computing
   anything; a mismatch (model/prompt/mask/fires/sample-id drift) is a hard
   stop, not a warning.
