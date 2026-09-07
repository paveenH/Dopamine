#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinQA numeric answer extraction + gold normalization (protocol `finqa-v0`).

Shared by the generator (diagnostics only, no accuracy) and the evaluator
(the only script that reads gold) so both sides use exactly one definition.

MAIN = first legal '#### <value>'; a value is "legal" if it normalizes to a
finite float under `normalize_finqa_answer` (money/percent/comma/unit forms
all accepted, matching the same normalizer the loader used to build the gold
pool). No legal marker => scored incorrect. LAST is a sensitivity readout
using the last legal marker instead of the first.

Reuses `normalize_finqa_answer` from data_finqa.py rather than reimplementing
it, so the loader's "numeric answer" definition and the scorer's are provably
the same function.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from data_finqa import normalize_finqa_answer  # noqa: E402

# One marker candidate: '####' then a value token. The value token is
# deliberately permissive (captures $, commas, %, unit words) -- legality is
# decided by normalize_finqa_answer, not by this regex.
_MARKER_RE = re.compile(
    r"####\s*(\$?\s*-?[\d,]+(?:\.\d+)?\s*%?\s*(?:thousand|million|billion|trillion)?)",
    re.IGNORECASE,
)


def find_all_markers(text: str):
    """Return list of (raw_value_string, normalized_float_or_None) for every
    '####' occurrence in generation order, whether or not it parses.
    """
    out = []
    for m in re.finditer(r"####", text):
        # look at up to ~40 chars after the marker for a value token
        window = text[m.end(): m.end() + 40]
        vm = _MARKER_RE.match("####" + window)
        raw = vm.group(1).strip() if vm else ""
        val = normalize_finqa_answer(raw) if raw else None
        out.append((raw, val))
    return out


def first_legal_answer(text: str):
    """Return the first legal (raw, value) marker, or (None, None)."""
    for raw, val in find_all_markers(text):
        if val is not None:
            return raw, val
    return None, None


def last_legal_answer(text: str):
    """Return the last legal (raw, value) marker, or (None, None)."""
    result = (None, None)
    for raw, val in find_all_markers(text):
        if val is not None:
            result = (raw, val)
    return result


def is_correct(pred_val, gold_val, rel_tol=1e-3, abs_tol=1e-4) -> bool:
    """Numeric match with a small tolerance for floating formatting drift
    (e.g. a model reporting 94.0 vs gold 94, or 12.50 vs 12.5). FinQA gold is
    itself sometimes rounded, so an exact string match would be too strict;
    a relative tolerance of 0.1% (plus a small absolute floor for values near
    zero) matches how FinQA-family evaluators commonly score numeric answers.
    """
    if pred_val is None or gold_val is None:
        return False
    diff = abs(pred_val - gold_val)
    tol = max(abs_tol, rel_tol * max(abs(gold_val), 1e-9))
    return diff <= tol


def is_loop(text: str) -> bool:
    """Strict degenerate-tail detector, same convention as the frozen GSM8K
    detector: the final 40-char block recurs >=4x in the text. A permissive
    n-gram proxy over-fires on ordinary restatement (see CLAUDE.md), so this
    stays strict.
    """
    t = text.strip()
    if len(t) < 160:
        return False
    tail = t[-40:]
    return t.count(tail) >= 4


def diagnostics_for_text(text: str, max_new_tokens: int, generated_token_count=None):
    """One-generation diagnostic bundle: no_answer, multi_answer, loop,
    truncated (by token count if available, else by trailing-no-EOS proxy on
    length), gen_chars.
    """
    markers = find_all_markers(text)
    legal = [v for _, v in markers if v is not None]
    return {
        "n_markers": len(markers),
        "n_legal_markers": len(legal),
        "no_answer": len(legal) == 0,
        "multi_answer": len(legal) >= 2,
        "loop": is_loop(text),
        "gen_chars": len(text),
        "truncated": (generated_token_count is not None
                      and generated_token_count >= max_new_tokens),
    }
