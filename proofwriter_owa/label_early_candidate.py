#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`pw-label-early-v1`: label-only early-candidate detector for ProofWriter-OWA,
extracted as a PURE module from
RoleAnswer/proofwriter_chat_early_candidate_v1/analyze_proofwriter_chat_early_candidate.py
(rules and regexes unchanged; not re-tuned).

Rule: the first non-empty stripped line is <= `cap` characters (default 60;
40/80 are sensitivity checks), contains exactly ONE distinct label word, and
is a bare label, `#### label`, `Answer: label`, a label followed by
punctuation / "because", or a leading "(the|my) answer/label/truth value is
<label>" assertion. It does NOT infer labels from propositions, negation,
uncertainty or proof text. Not the numeric `earlycand-v1`. A hit describes
VISIBLE output order only -- never internal commitment.

tests/test_zs_label_early.py checks tag-for-tag equality with the original
script on its 2400 stored replies.
"""

from __future__ import annotations

import re

VERSION = "pw-label-early-v1"
MAX_LINE_CHARS = 60

_LABEL = r"(true|false|unknown)"
_WORDS = re.compile(r"\b" + _LABEL + r"\b", re.I)
_EXACT = re.compile(r"^(?:(?:####|answer\s*:)\s*)?" + _LABEL + r"\s*\.?$", re.I)
_LEADING = re.compile(r"^" + _LABEL + r"(?:\s*[,.:;!—-]\s*|\s+because\b)", re.I)
_ASSERTION = re.compile(
    r"^(?:(?:the|my)\s+)?(?:answer|label|truth value)\s*(?:is\s+|[:=]\s*)"
    + _LABEL + r"\b", re.I)


def first_line(text: str) -> str:
    return next((line.strip() for line in text.split("\n") if line.strip()), "")


def candidate(text: str, cap: int = MAX_LINE_CHARS):
    """Canonical label ("True"/"False"/"Unknown") of the early candidate on the
    first non-empty line, or None."""
    line = first_line(text)
    labels = {w.lower() for w in _WORDS.findall(line)}
    if not line or len(line) > cap or len(labels) != 1:
        return None
    for pattern in (_EXACT, _LEADING, _ASSERTION):
        m = pattern.match(line)
        if m:
            return m.group(1).capitalize()
    return None
