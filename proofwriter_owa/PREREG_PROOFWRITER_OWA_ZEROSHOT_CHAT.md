# ProofWriter-OWA Chat Zero-Shot (`proofwriter-owa-zs-chat-v1`)

Frozen 2026-10-21 before any zero-shot cell existed. Independent of the closed
bare-string line (`proofwriter-owa-v0`) and the 1-shot chat line
(`proofwriter-owa-chat-v1`); neither is modified or superseded.

**Question.** Under a single-user-message chat interface with NO exemplar, do the
two GSM8K workpoints (llama3 -6, qwen2.5 +8; read from the frozen GSM8K record,
not searched here) change ProofWriter-OWA accuracy and the visible order in which
a label is stated?

## Prompt (single user message, `apply_chat_template(..., add_generation_prompt=True)`, nothing appended)

```text
{SYSTEM_RULES}            # prompt.SYSTEM_RULES, imported unchanged

Theory: {theory}
Question: {question}

Do not write code or restate the theory.
Include exactly one answer line using one of these three forms:
#### True
#### False
#### Unknown
Stop immediately after that line.
```

CoT variant = the same string + one final line `Let's think step by step.`
Template ids: `proofwriter-owa-zs-nocot-v1`, `proofwriter-owa-zs-cot-v1` (both
registered in `answer_parser.MARKER_FAMILIES` -> v2 `####` family). `n_shot=0`.

## Fixed settings

| item | value |
|---|---|
| data | the existing 300 items, same order (`manifest_blind.json`) |
| llama3 | Llama-3.1-8B-Instruct, alpha {0, -6}, band [11,20), L=9 |
| qwen2.5 | Qwen2.5-7B-Instruct, alpha {0, +8}, band [16,22), L=6 |
| generation | greedy, temperature 0, top_p 1, 1024 new tokens, batch 8 |
| injection | last token of the wrapped prompt, prefill-only, tail 1, full-length diff matrix; alpha=0 uses the same path with an all-zero matrix |
| fires (n=300) | alpha=0: 0; llama -6: 2700; qwen +8: 1800 (mismatch = cell not written / not analysed) |

The final prompt token id, its text and the 8-token tail are read from each real
tokenizer and stored in `meta.injection_site`; they are not assumed equal across models.

## Stages

1. `check` (tokenizer only) -> 2. No-CoT alpha=0 for both models (**format gate**) ->
3. each model that passes: its own workpoint cell -> 4. `eval`.
CoT (4 cells) is a later stage with its own gate/eval; nothing starts it automatically.

**Format gate (frozen; decided on submission format only, never on accuracy or early-candidate):**
valid_submission_rate >= 0.90 (>=1 strict line-anchored `#### True|False|Unknown`),
truncation_rate <= 0.10, loop_rate <= 0.10 (strict final-40-char-block test). On
failure the result is kept; the prompt, token budget and scoring are not changed
automatically, and that model's workpoint cell is refused.

## Scoring and readouts

First strict marker is the answer; no marker = incorrect; denominator always 300;
no bare-label fallback. Reported per cell: `first_acc`; valid submission; conditional
accuracy; label early candidate (`pw-label-early-v1`, all 300; 40/80 as sensitivity);
`answer_first` (valid subset); post-marker fate (`clean_stop` / `trailing_text` /
`additional_marker`, valid subset); loop; truncation; natural EOS; token length; label
distributions. Marker-first alone does not show the stop instruction was obeyed.

## Statistics

Per model, workpoint vs the same model's alpha=0, paired by sample identity: exact
McNemar + question-paired bootstrap CI (B=10000, seed 0). The two models'
workpoint contrasts are ONE Holm family, m=2, per condition; separate from the m=3
families of the older lines. Early-candidate contrasts are exploratory (unadjusted).
The evaluator accepts only this experiment's own four cells and aborts on any
config / provenance / identity / fire-count mismatch.

## Interpretation limits

Order change with an accuracy rise = co-variation, not causal mediation. Order
change without a detected gain does not show errors are not due to premature
answering. A near-zero alpha=0 early-candidate rate means little room to move, not
that steering cannot help. Early-candidate / answer-first are visible output order,
not internal commitment. No-CoT vs CoT differ by one line (a CoT-instruction
contrast); comparison with the old 1-shot CoT line also differs in other wording and
cannot be attributed to the exemplar alone.
