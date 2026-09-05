#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explicit-CoT prompt builder for ProofWriter OWA.

THIS IS OUR OWN CONSTRUCTION, NOT AN OFFICIAL PROOFWRITER LLM PROMPT. It
follows the same "own construction, labelled as such" convention used
elsewhere in this repo (e.g. GSM8K's "Let's think step by step.",
`get_answer_bbh_numeric_cot.py`'s CoT variant).

Design constraints (PREREG_PROOFWRITER_OWA.md S4):
  - theory and question text are the OFFICIAL strings, byte-unchanged; no
    paraphrase, no re-templating.
  - the model is asked to reason step by step about whether the query or its
    EXPLICIT NEGATION follows from the facts/rules, then end with exactly one
    line: "Answer: True" / "Answer: False" / "Answer: Unknown".
  - the OWA definition is stated explicitly in those terms, so the model is
    not left to guess CWA-style "can't prove it -> False".
  - every alpha cell of one model uses the byte-identical template string.
  - optional few-shot exemplars are frozen TRAIN-split only, never test.

v1 REVISION (2026-09-04, PREREG_PROOFWRITER_OWA.md S4/S6, human decision):
v0's TASK_INSTRUCTIONS only asked the model to reason then end with a final
line, with NO constraint on reasoning length or on stopping after the answer
line. The llama3 alpha=0 preflight (1024-token budget) measured 100%
truncation and 96.7% parse_failure_rate on 30 items -- manual inspection of
eight generations found NONE reached a strict "Answer: <Label>" line within
budget: some kept elaborating multi-hop reasoning indefinitely, one wrote
Python code, at least two entered stable degenerate repetition loops
("# Corrected output format." x dozens; "...use the rules to answer the
question, and then use the rules to answer the negation..." cycling). This
is a format/length-control gap, not a task-definition problem, so v1 changes
ONLY TASK_INSTRUCTIONS: adds an explicit step budget ("at most 5 steps"), an
explicit prohibition on code/restating-the-theory/repeating-previous-steps,
and an explicit stop instruction after the answer line. SYSTEM_RULES (the
OWA semantics) and the theory/question substitution are BYTE-UNCHANGED.

v2 REVISION (2026-09-05, human decision, CLAUDE.md ProofWriter-OWA row): v1
did NOT fix the gate -- llama3 acc .0333 parse_fail .900 loop .800 trunc
1.000; qwen2.5 acc .0333 parse_fail .733 (WORSE than its own v0 cell's .400).
v2 makes two changes, both narrowly scoped to the SUSPENSION's authorized
"one small ... feasibility probe":
  (1) ONE fixed Unknown-labeled train-split exemplar is added (1-shot, not
      the originally-planned True/False/Unknown-balanced probe -- this is a
      recorded, not silent, scope narrowing; see CLAUDE.md's
      "label_bias_note" wording and exemplar_unknown_v2.json's own
      "label_bias_note" field for why this may bias toward Unknown). The
      exemplar's theory/question/answer are the official train-split strings
      verbatim; its one-sentence "reasoning" is THIS PROJECT'S OWN
      CONSTRUCTION under OWA semantics, explicitly NOT the official proof
      (see exemplar_unknown_v2.json's "reasoning_provenance" field for why
      the official proof_text could not be used directly -- it is CWA-style
      and would misstate this task's OWA semantics).
  (2) The final-answer marker changes from a free-standing "Answer: <Label>"
      line to a "#### <Label>" line (matching this repo's GSM8K/MATH/BBH/
      LogiQA/CRUXEval-O "####"-marker convention), on the theory that a
      terser, more familiar-to-the-model marker token may reduce the
      elaboration/looping v1 still showed. TASK_INSTRUCTIONS is updated to
      match; SYSTEM_RULES (the OWA semantics) is BYTE-UNCHANGED from v0/v1.
The exemplar's own worked answer uses THIS SAME "#### <Label>" marker (not
v1's "Answer:" form) -- render_exemplar's marker style now follows the
active prompt version's ANSWER_MARKER_PREFIX, so the model never sees two
different marker conventions inside one prompt.
"""

from __future__ import annotations

SYSTEM_RULES = (
    "You are given a theory (a list of facts and rules) and a question. "
    "Determine the truth value of the question using ONLY the given facts "
    "and rules, under the following open-world assumption:\n"
    "  - If the question can be proven true from the facts and rules, the "
    "answer is True.\n"
    "  - If the EXPLICIT NEGATION of the question can be proven true from "
    "the facts and rules, the answer is False.\n"
    "  - If NEITHER the question NOR its explicit negation can be proven, "
    "the answer is Unknown. Simply failing to prove the question does NOT "
    "by itself make the answer False -- only a proof of the negation does.\n"
)

TASK_INSTRUCTIONS_V1 = (
    "Reason briefly in at most 5 steps.\n"
    "Do not write code, restate the theory, or repeat previous steps.\n"
    "After reasoning, output exactly one final line using one of these "
    "three forms:\n"
    "Answer: True\n"
    "Answer: False\n"
    "Answer: Unknown\n"
    "Stop immediately after that line.\n"
)

# v2 (2026-09-05): the marker changes from "Answer: <Label>" to
# "#### <Label>", matching this repo's GSM8K/MATH/BBH/LogiQA/CRUXEval-O
# convention. Everything else in TASK_INSTRUCTIONS is unchanged from v1.
TASK_INSTRUCTIONS_V2 = (
    "Reason briefly in at most 5 steps.\n"
    "Do not write code, restate the theory, or repeat previous steps.\n"
    "After reasoning, output exactly one final line using one of these "
    "three forms:\n"
    "#### True\n"
    "#### False\n"
    "#### Unknown\n"
    "Stop immediately after that line.\n"
)

# Kept as the un-suffixed name for backward compatibility with any existing
# caller that imports TASK_INSTRUCTIONS directly; PROMPT_TEMPLATE_ID below is
# v2's, so this alias also now points at v2's instructions. v1's exact text
# remains available (byte-unchanged) as TASK_INSTRUCTIONS_V1.
TASK_INSTRUCTIONS = TASK_INSTRUCTIONS_V2

# {theory} and {question} are filled with the OFFICIAL, byte-unchanged
# strings from the release. No exemplar block by default (n_shot=0).
BASE_TEMPLATE = (
    "{system_rules}\n"
    "Theory: {theory}\n"
    "Question: {question}\n\n"
    "{task_instructions}"
    "{exemplar_block}"
    "Let's think step by step.\n"
)

# The marker prefix used both in TASK_INSTRUCTIONS and in the worked
# exemplar's own answer line -- kept as ONE constant so the two can never
# silently drift apart (the v1->v2 change is exactly this kind of drift risk:
# TASK_INSTRUCTIONS says "####" while an exemplar still built with the old
# render_exemplar default would have said "Answer:").
ANSWER_MARKER_PREFIX_V1 = "Answer:"
ANSWER_MARKER_PREFIX_V2 = "####"
ANSWER_MARKER_PREFIX = ANSWER_MARKER_PREFIX_V2


def render_exemplar(theory: str, question: str, answer: str, reasoning: str,
                     marker_prefix: str = ANSWER_MARKER_PREFIX) -> str:
    """One frozen few-shot exemplar block. `reasoning` is a short, fixed,
    hand-written justification -- never sampled from a model and never taken
    from test-split proofs. `marker_prefix` must match whichever
    TASK_INSTRUCTIONS variant is in use ("Answer:" for v1, "####" for v2),
    so the exemplar's own worked answer uses the SAME marker convention the
    model is being asked to produce -- passing a mismatched value here would
    silently show the model two different final-answer conventions in one
    prompt."""
    return (
        f"Theory: {theory}\n"
        f"Question: {question}\n"
        f"{reasoning}\n"
        f"{marker_prefix} {answer}\n\n"
    )


def build_prompt(theory: str, question: str, exemplars: list[dict] | None = None,
                  task_instructions: str = TASK_INSTRUCTIONS_V2,
                  marker_prefix: str = ANSWER_MARKER_PREFIX) -> str:
    """exemplars: list of {"theory","question","answer","reasoning"} dicts,
    frozen train-split items only. Default None/[] = zero-shot.
    `task_instructions` / `marker_prefix` default to v2; a caller can pass
    TASK_INSTRUCTIONS_V1 / ANSWER_MARKER_PREFIX_V1 to reproduce the v1
    template byte-for-byte (e.g. for a comparison run), but PROMPT_TEMPLATE_ID
    below always reflects THIS MODULE's active default (v2) -- a caller
    overriding these two arguments is responsible for also overriding what it
    records as its own prompt_template_id, exactly as get_answer_
    proofwriter_owa.py already does by importing PROMPT_TEMPLATE_ID directly
    rather than hardcoding a version string."""
    exemplar_block = ""
    if exemplars:
        blocks = [render_exemplar(e["theory"], e["question"], e["answer"],
                                  e["reasoning"], marker_prefix=marker_prefix)
                  for e in exemplars]
        exemplar_block = (
            "Here are worked examples using the same rules (these use "
            "DIFFERENT theories from the one above):\n\n"
            + "".join(blocks)
            + "Now answer the theory and question above.\n\n"
        )
    return BASE_TEMPLATE.format(
        system_rules=SYSTEM_RULES,
        theory=theory,
        question=question,
        task_instructions=task_instructions,
        exemplar_block=exemplar_block,
    )


PROMPT_TEMPLATE_ID = "proofwriter-owa-cot-v2"
