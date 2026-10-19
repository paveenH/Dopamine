# PREREG — Chat-CoT Saturation Check (`chat-cot-sat-v1`)

Frozen 2026-10-19, before any cell exists. **Post-hoc by design:** added after the Native Chat No-CoT results were seen.
"Saturation" is a hypothesis; it is never written as a finding. Code: `chat_cot_sat_lib.py`, `get_answer_chat_cot_sat.py`,
`run_chat_cot_sat.sh`; offline `RoleAnswer/analyze_chat_cot_sat_v1.py`.

## Question
Under the existing Native Chat interface, does adding explicit CoT leave room for RSN to change output organisation and
first-answer accuracy? Focus: Llama/GSM-Hard (early-candidate already ≈0, accuracy has room); Qwen/GSM8K (is the CoT gain
descriptively smaller than the No-CoT gain).

## Eight cells (2,400 replies), no other dose
| Task | Model | α | Band |
|---|---|---|---|
| GSM8K, GSM-Hard | `meta-llama/Llama-3.1-8B-Instruct` | 0, −6 | `[11,20)` (decoder 10–18, L=9) |
| GSM8K, GSM-Hard | `Qwen/Qwen2.5-7B-Instruct` | 0, +8 | `[16,22)` (decoder 15–20, L=6) |

Native Chat (`apply_chat_template`, `add_generation_prompt=True`, duplicated BOS stripped, model's default system content kept,
no added system message), neutral, zero-shot; same 300 items/order as the Native Chat cells (`benchmark/gsm8k_test_sample.json`;
`benchmark/gsm_hard_p3_questions.json`, label-free generation); greedy, 768 tokens, bs=24; original RSN masks; prefill-only,
`tail_len=1` = last token of the assistant header (read from the real tokenizer for ALL prompts; not Bare's token 220).
α=0 is re-generated under the CoT prompt. Both α of a (model, task) run in ONE model load.

## The single prompt change
Exactly one line, `Let's think step by step.`, supplied by the existing `cot=True` template before the final-answer instruction.
Verified at run time, fail-closed: (1) template level, (2) every rendered prompt, (3) every **wrapped chat prompt** — removing the
line reproduces the wrapped No-CoT prompt byte for byte. No `Answer: ` anchor, role, few-shot or extra text. `prompt_diff.json`
stores the template pair and examples; the offline analysis also requires `nocot_chat_prompt_sha256` to equal the old Native Chat cell's `prompt_sha256`.

## Integrity (a cell failing any is unusable)
300 replies per cell, each batch returns its input count; mask non-zero rows == configured decoder band; `steering_fires` = 2700
(Llama α≠0) / 1800 (Qwen α≠0) / 0 (α=0); stop reason ∈ {natural_eos, budget_exhausted}. An existing cell is skipped only if
protocol, α, mask sha, prompt sha, settings, fires and row completeness verify; otherwise the run dies. Hardware is recorded
(host, `CUDA_VISIBLE_DEVICES`, GPU names, sharding, library versions, git HEAD) and never enforced.

## Metrics
`first_acc` (primary performance; frozen offline extractor; denominator 300). `early_candidate_rate` (main output-organisation
readout, `earlycand-v1`, denominator 300). `reason_first_rate` (auxiliary; frozen definition; denominator = candidate-covered subset,
count reported; never `1 − early_candidate`). Diagnostics: candidate coverage, loop, truncation, empty/no-stop-reason, natural EOS,
no-parseable-marker. No row is dropped.

## Inference (the only tests)
Within Chat-CoT, workpoint vs α=0, per model and task: Δ`first_acc`, paired question bootstrap (B=10000, seed 0, percentile 95% CI),
two-sided exact McNemar with discordant counts. **Two independent Holm families, m=2 each:** GSM8K {Llama, Qwen}; GSM-Hard {Llama, Qwen}.
Output-organisation and health metrics are descriptive, no new family.

## Comparison with No-CoT (descriptive only)
Existing Native Chat No-CoT α=0 and workpoint cells are checked for question/gold order, mask sha, chat-template hash, wrapped-prompt hash,
injection token and generation settings (mismatch ⇒ analysis aborts); host/GPU/sharding differences are listed, not failures.
Reported: both gains and their difference in pp. **No interaction test.** "Not significant under CoT vs significant under No-CoT" is
not evidence the gains differ, and nothing here shows CoT replaces RSN.

## Interpretation limit
Fixed 768-token budget only. `early_candidate` ≈ 0 means that readout has little room to move; it does not by itself show internal
reasoning is saturated. EC / reason-first describe visible output order, not internal decisions or mediation.
