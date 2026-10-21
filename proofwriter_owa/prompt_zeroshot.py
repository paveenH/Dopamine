#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot prompt builder for the ProofWriter-OWA CHAT zero-shot experiment
(protocol `proofwriter-owa-zs-chat-v1`).

INDEPENDENT of prompt.py's v2 1-shot prompt (`proofwriter-owa-cot-v2`), which
stays byte-unchanged for the closed bare/chat lines. Only `SYSTEM_RULES` is
reused (imported, never retyped). Theory / Question are the official strings,
unchanged. No exemplar is ever loaded.

Two variants, differing by EXACTLY ONE inserted line:
  No-CoT   : SYSTEM_RULES + Theory/Question + INSTRUCTIONS
  CoT      : the same string with "Let's think step by step.\n" inserted
             immediately BEFORE the "Include exactly one answer line..."
             instruction (no other difference)

There is NO "do not write code / restate the theory" line and NO "stop
immediately" line (removed 2026-10-21): nothing in the prompt asks the model
to stop, so post-marker text is observed behaviour, not instruction breach.

The old v1/v2 wording ("Reason briefly in at most 5 steps.", "After
reasoning, ...", the worked-example lead-in) must not appear in either
variant; tests/test_zs_prompt.py enforces that.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import SYSTEM_RULES  # noqa: E402

NOCOT_TEMPLATE_ID = "proofwriter-owa-zs-nocot-v1"
COT_TEMPLATE_ID = "proofwriter-owa-zs-cot-v1"
TEMPLATE_IDS = {"nocot": NOCOT_TEMPLATE_ID, "cot": COT_TEMPLATE_ID}

INSTRUCTIONS = (
    "Include exactly one answer line using one of these three forms:\n"
    "#### True\n"
    "#### False\n"
    "#### Unknown\n"
)
COT_LINE = "Let's think step by step.\n"


def build_user_prompt(theory: str, question: str, cot: bool) -> str:
    """The single user-message string (before chat-template wrapping)."""
    return (
        f"{SYSTEM_RULES}\n"
        f"Theory: {theory}\n"
        f"Question: {question}\n\n"
        f"{COT_LINE if cot else ''}"
        f"{INSTRUCTIONS}"
    )


def condition_to_cot(condition: str) -> bool:
    if condition not in TEMPLATE_IDS:
        raise ValueError(f"condition must be one of {sorted(TEMPLATE_IDS)}, "
                         f"got {condition!r}")
    return condition == "cot"
