# GSM-Symbolic task-specific steering workpoint exploration

Numeric/template robustness check of the frozen GSM8K workpoints
(llama3 `-6`, qwen2.5 `+8`) under `apple/GSM-Symbolic`
(https://huggingface.co/datasets/apple/GSM-Symbolic). **This is a robustness
check of the GSM8K workpoint under numeric perturbation, not an independent
cross-domain transfer test** — GSM-Symbolic is generated from GSM8K's own
reasoning templates, re-instantiated with different symbolic values (`main`)
plus one (`p1`) or two (`p2`) extra clauses.

Self-contained under `gsm_symbolic/`; no existing GSM8K/MATH/other-task
runner, loader, or launcher is modified.

## Files

- `data_gsm_symbolic.py` — loader. Downloads all 3 official test-split configs
  (`main`, `p1`, `p2`) from `apple/GSM-Symbolic`, writes one JSON per config
  plus a combined 30-item preflight subset (10/config, deterministic stride
  sample by `(original_id, instance, id)`). Gold is extracted via the same
  `#### <number>` convention `utils.extract_gsm8k_answer` uses (kept local so
  the loader has no torch/heavy import). `--check` validates schema and
  writes nothing.
- `get_answer_gsm_symbolic.py` — generation + scoring driver. Fork of
  `get_answer_regenerate_gsm8k.py`; imports `utils.extract_gsm8k_answer`,
  `utils.is_correct_gsm8k`, `utils.parse_configs`, `llms.VicundaModel`, and
  `template.select_templates_gsm8k(suite="default", cot=True, wording="plain")`
  — the existing, frozen explicit-CoT GSM8K prompt, byte-identical to the main
  GSM8K line. Role fixed to `neutral`. One `--gsm_config` per call.
- `eval_gsm_symbolic.py` — scoring/statistics. Re-derives correctness via the
  frozen extractor (never trusts the inline `correct` field), runs paired
  exact McNemar (stdlib binomial CDF, no scipy dependency) per non-zero alpha
  vs. that model's own alpha=0, Holm correction with `m=3` (the model's three
  non-zero doses) **per config and pooled — never pooled across models or
  merged into a larger family**. Also reports per-`original_id` accuracy
  mean/std (so a config's headline number isn't read off one instantiation)
  and diagnostics (no-answer, multi-marker, first/last disagreement, loop,
  truncation, generation length).
- `run_gsm_symbolic_preflight.sh` — alpha=0 only, 30-item preflight, one model
  per invocation.
- `run_gsm_symbolic_formal.sh` — full 4-point sweep over the complete official
  test split of all 3 configs, one model, one GPU, sequential.
- `sanity_check_eval.py` — offline sanity check (exact McNemar edge cases,
  Holm monotonicity, `score_cell`/`per_instance_breakdown` against hand-built
  generations run through the real frozen extractor). No model, no server, no
  network. Run with `python3.10 sanity_check_eval.py`.

## Design decisions carried over from the existing GSM8K/GSM-Hard/P3-supp lines

- **Same-GPU-per-model-curve rule**: bf16 greedy is not byte-reproducible
  across GPUs, so each model's full 4-alpha x 3-config sweep runs on one
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
  gsm_symbolic_main_test.json
  gsm_symbolic_p1_test.json
  gsm_symbolic_p2_test.json
  gsm_symbolic_preflight_30.json

{model}/answer_gsm_symbolic_preflight/{main,p1,p2}/mdf_0/
  gsm_symbolic_{cfg}_{size}_{ls}_{le}.json      # preflight, alpha=0 only

{model}/answer_gsm_symbolic/{main,p1,p2}/mdf_{alpha}/
  gsm_symbolic_{cfg}_{size}_{ls}_{le}.json      # formal sweep
  summary_gsm_symbolic_{model}_{size}_{ls}_{le}.csv
```

## Two-step workflow

1. **Preflight (alpha=0 only, 30 items, both models on the same file)**:
   ```bash
   python data_gsm_symbolic.py --out_dir components/benchmark/gsm_symbolic --preflight_n 10
   CUDA_VISIBLE_DEVICES=0 bash run_gsm_symbolic_preflight.sh llama3
   CUDA_VISIBLE_DEVICES=1 bash run_gsm_symbolic_preflight.sh qwen2.5
   ```
   Stops here. Inspect `meta.steering_fires` (must be 0 at alpha=0),
   `no_answer`, `truncated`, and a few raw `generated` strings per config
   before running the formal sweep.

2. **Formal sweep (after approval)**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_symbolic_formal.sh llama3  > gsms_llama.log 2>&1 &
   CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_symbolic_formal.sh qwen2.5 > gsms_qwen.log  2>&1 &
   # after both finish:
   python eval_gsm_symbolic.py --model llama3  --base_dir components --out gsms_llama3_eval.json
   python eval_gsm_symbolic.py --model qwen2.5 --base_dir components --out gsms_qwen25_eval.json
   ```

## Open items / things I did not decide unilaterally

See the numbered questions sent alongside this implementation.
