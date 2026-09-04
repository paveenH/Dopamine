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

TASK_INSTRUCTIONS = (
    "Reason briefly in at most 5 steps.\n"
    "Do not write code, restate the theory, or repeat previous steps.\n"
    "After reasoning, output exactly one final line using one of these "
    "three forms:\n"
    "Answer: True\n"
    "Answer: False\n"
    "Answer: Unknown\n"
    "Stop immediately after that line.\n"
)

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


def render_exemplar(theory: str, question: str, answer: str, reasoning: str) -> str:
    """One frozen few-shot exemplar block. `reasoning` is a short, fixed,
    hand-written justification -- never sampled from a model and never taken
    from test-split proofs."""
    return (
        f"Theory: {theory}\n"
        f"Question: {question}\n"
        f"{reasoning}\n"
        f"Answer: {answer}\n\n"
    )


def build_prompt(theory: str, question: str, exemplars: list[dict] | None = None) -> str:
    """exemplars: list of {"theory","question","answer","reasoning"} dicts,
    frozen train-split items only. Default None/[] = zero-shot."""
    exemplar_block = ""
    if exemplars:
        blocks = [render_exemplar(e["theory"], e["question"], e["answer"], e["reasoning"])
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
        task_instructions=TASK_INSTRUCTIONS,
        exemplar_block=exemplar_block,
    )


PROMPT_TEMPLATE_ID = "proofwriter-owa-cot-v1"
