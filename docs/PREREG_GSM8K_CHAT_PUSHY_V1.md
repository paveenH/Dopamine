# GSM8K Native Chat Pushy (`gsm8k-chat-pushy-v1`)

Protocol written before any cell exists. Exploratory, post-hoc relative to the Native Chat No-CoT / Chat-CoT results.

## Question
On the Native Chat No-CoT interface, does replacing only the answer-format line with the Bare "pushy" wording
(1) induce early answer candidates, (2) does the fixed RSN workpoint change that behaviour, and (3) does frozen
`first_acc` improve (corrections vs damage)?

## Prompt (single change)
| | format line |
|---|---|
| Chat No-CoT (plain) | `Provide your final numeric answer after '####'.` |
| **Pushy (this protocol)** | `Give your final answer as a single number after '####'.` |

Template = `build_gsm8k_default_suite(cot=False, wording="pushy")["neutral"]`. No CoT line, no "answer first"/"as soon as
possible", no stop instruction, no example/role. Trailing `Answer: ` kept. One user message through
`apply_chat_template(add_generation_prompt=True)`; no custom system message; nothing appended after the assistant
header. Direct-answer output is allowed; absence of explanation is not an anomaly. The runner asserts the wrapped
pushy prompt equals the wrapped plain prompt after swapping the format line, for all 300 items.

## Cells (fixed; no dose search, no gating on α=0 EC or accuracy)
| model | mask | band | α |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | `mask/llama3_non_logits/nmd_0.5_11_20_8B.npy` | [11,20) L=9 | 0, −6 |
| `Qwen/Qwen2.5-7B-Instruct` | `mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy` | [16,22) L=6 | 0, +8 |

Same 300 items (`benchmark/gsm8k_test_sample.json`), neutral, zero-shot, greedy (temperature 0, top_p 1),
768 new tokens, batch 24, prefill-only, tail 1. Injection token = last token of the assistant header, read from the
real tokenizer (not Bare's id 220). Expected fires: α=0 → 0; Llama −6 → 2700; Qwen +8 → 1800.

## Execution
`run_gsm8k_chat_pushy_4cell.sh`: two parallel processes (`LLAMA_GPU` ≠ `QWEN_GPU`), one model load each, α in order.
Output `components/<model>/gsm8k_chat_pushy_v1/mdf_<α>/gsm8k_chat_pushy_<size>_answers_<ls>_<le>.json` + `prompt_diff.json`.
In-run fail-closed guards: item identity/order, mask rows vs band, fires, 300 replies, stop reasons, atomic writes;
an existing cell is reused only if protocol/α/mask/prompt/settings/data identity all match, else the run aborts.
Hardware is recorded, not enforced.

## Offline analysis (`RoleAnswer/reasoningTaskAnalysis/chat_pushy_v1/`)
Frozen `analyze_first_last_acc.py` (`first_acc` MAIN, `last_acc` sensitivity, fallback chain kept, denominator 300)
and `early_candidate_detector.py` (`earlycand-v1`, denominator 300); inline `correct` is not used. Per cell: valid-marker
rate, no-marker, empty, loop, truncation, natural EOS, generated-token mean/median. Contrasts: Llama −6 vs 0, Qwen +8 vs 0;
paired bootstrap (B=10000, seed 0, 95% percentile CI), exact two-sided McNemar, Holm m=2 as a NEW family; wrong→right,
right→wrong, net. EC and health are descriptive. Historical Chat plain / CoT / Bare pushy are descriptive references
only (different prompt/interface/run); no interaction claim from one-significant-one-not.

## Interpretation limits
A positive result supports at most: "under a pushy Chat prompt, RSN changed visible answer organisation, accompanied by
improved first-answer accuracy." Not the same internal mechanism, not causal mediation, not a general headroom rule.
If the pushy wording does not induce early candidates (or EC ≈ 100%), report as is; no prompt/dose changes follow.
