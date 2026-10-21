#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot prompt builder for the ProofWriter-OWA CHAT zero-shot experiment
(protocol `proofwriter-owa-zs-chat-v1`).

INDEPENDENT of prompt.py's v2 1-shot prompt (`proofwriter-owa-cot-v2`), which
stays byte-unchanged for the closed bare/chat lines. Only `SYSTEM_RULES` is
reused (imported, never retyped). Theory / Question are the official strings,
unchanged. No exemplar is ever loaded.

Two variants, differing by EXACTLY ONE trailing line:
  No-CoT   : SYSTEM_RULES + Theory/Question + INSTRUCTIONS
  CoT      : the No-CoT string + "Let's think step by step.\n"

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
    "Do not write code or restate the theory.\n"
    "Include exactly one answer line using one of these three forms:\n"
    "#### True\n"
    "#### False\n"
    "#### Unknown\n"
    "Stop immediately after that line.\n"
)
COT_LINE = "Let's think step by step.\n"


def build_user_prompt(theory: str, question: str, cot: bool) -> str:
    """The single user-message string (before chat-template wrapping)."""
    s = (
        f"{SYSTEM_RULES}\n"
        f"Theory: {theory}\n"
        f"Question: {question}\n\n"
        f"{INSTRUCTIONS}"
    )
    return s + COT_LINE if cot else s


def condition_to_cot(condition: str) -> bool:
    if condition not in TEMPLATE_IDS:
        raise ValueError(f"condition must be one of {sorted(TEMPLATE_IDS)}, "
                         f"got {condition!r}")
    return condition == "cot"
