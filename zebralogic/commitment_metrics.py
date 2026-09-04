#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZebraLogic-native commitment metrics. FROZEN per docs/PREREG_ZEBRALOGIC_EASY.md
section 6, before any non-zero-alpha formal result is read.

Deliberately NOT `earlycand-v1` (the digit-based first-line detector used
elsewhere in this repo): that detector assumes a short first line containing a
bare number, which does not describe this task's multi-line JSON-grid output
and would silently ceiling/floor here exactly the way it does on LogiQA's
letter-choice format (see CLAUDE.md's cot-transfer-followup-v0 section).

All metrics here are OUTPUT DIAGNOSTICS computed from already-generated,
steered text. They can only ever show co-occurrence with accuracy changes,
never causal mediation -- the same standing rule as every other
commitment-metric line in this repo (P2, P3-supp, the CoT-followup line). Any
subgroup breakdown by these metrics is post-treatment stratification.

Two of the metrics (`revision_wrong_to_right` / `revision_right_to_wrong`)
require gold and are therefore computed only in the analysis/scoring script,
never during label-free generation -- see zebralogic/eval_zebralogic.py.
Everything else here is gold-free and can run on raw generations alone.

@author: paveenhuang
"""

import json
import re


def find_first_complete_json(s: str):
    """Forward-scanning counterpart to official_zebra_grid_scorer's
    extract_last_complete_json: returns the FIRST complete top-level {...}
    block (as a parsed dict), or None. Deliberately a separate function
    (not "just take the first element of a list of all JSON blocks") so its
    scanning direction is explicit and auditable against the official
    backward-scanning sibling.
    """
    stack = []
    first_start = None
    for i, char in enumerate(s):
        if char == "{":
            if not stack:
                first_start = i
            stack.append(i)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and first_start is not None:
                    candidate = s[first_start:i + 1]
                    try:
                        return json.loads(candidate.replace("\n", "")), first_start, i + 1
                    except json.JSONDecodeError:
                        # not valid JSON -- keep scanning for a later,
                        # well-formed complete block rather than giving up
                        first_start = None
                        continue
    return None, None, None


_STRICT_LOOP_BLOCK = 40
_STRICT_LOOP_MIN_REPEATS = 4


def is_strict_loop(text: str) -> bool:
    """Project-standard strict loop/repetition detector: the final
    _STRICT_LOOP_BLOCK-char block of `text` recurs at least
    _STRICT_LOOP_MIN_REPEATS times in the whole text. Matches the convention
    used throughout this repo (analyze_loop_anxiety.py's `--mode loop`,
    P4b's degenerate-tail detector) -- deliberately NOT a permissive n-gram
    proxy, which this repo's own history shows reads 74-88% false positives.
    """
    text = text or ""
    if len(text) < _STRICT_LOOP_BLOCK:
        return False
    tail = text[-_STRICT_LOOP_BLOCK:]
    return text.count(tail) >= _STRICT_LOOP_MIN_REPEATS


def grid_is_fully_specified(solution_obj) -> bool:
    """True iff `solution_obj` (the "solution" value of a parsed JSON block)
    is a non-empty dict whose every house's every cell is a present,
    non-placeholder, non-empty string/list value. "___" and "" count as
    unfilled -- matching the placeholder convention the official prompt
    template itself uses for the answer scaffold.
    """
    if not isinstance(solution_obj, dict) or not solution_obj:
        return False
    for house, cols in solution_obj.items():
        if not isinstance(cols, dict) or not cols:
            return False
        for col, val in cols.items():
            if val is None:
                return False
            if isinstance(val, list):
                if not val or val[0] is None or str(val[0]).strip() in ("", "___"):
                    return False
            elif isinstance(val, str):
                if val.strip() in ("", "___"):
                    return False
            # any other type (dict, int, ...) counts as present/filled --
            # matches the official scorer's permissive `.lower().strip()`
            # comparison, which coerces on read rather than rejecting types
            # up front.
    return True


def _cell_value(cell):
    """Normalize one prediction cell to a comparable lowercase-stripped
    string, or None if absent/empty -- mirrors official_zebra_grid_scorer's
    cell-handling exactly (None passthrough, list -> first element)."""
    if cell is None:
        return None
    if isinstance(cell, list):
        if not cell:
            return None
        return str(cell[0]).lower().strip()
    if isinstance(cell, str):
        return cell.lower().strip()
    return str(cell).lower().strip()


def first_final_grid_agreement(first_solution, last_solution):
    """Cell-level agreement between the FIRST complete JSON's `solution` grid
    and the LAST complete JSON's `solution` grid (the one the official scorer
    actually grades). 1.0 means the model never revised any cell after first
    writing a complete grid. Returns None if either grid is missing/malformed
    (no comparison is possible) rather than a misleading 0.0 or 1.0.

    Denominator is the UNION of (house, column) keys appearing in either
    grid, so a cell added or dropped between first and last counts as a
    disagreement rather than being silently ignored.
    """
    if not isinstance(first_solution, dict) or not isinstance(last_solution, dict):
        return None
    keys = set()
    for grid in (first_solution, last_solution):
        for house, cols in grid.items():
            if isinstance(cols, dict):
                for col in cols:
                    keys.add((house, col))
    if not keys:
        return None
    agree = 0
    for house, col in keys:
        a = _cell_value(first_solution.get(house, {}).get(col) if isinstance(first_solution.get(house), dict) else None)
        b = _cell_value(last_solution.get(house, {}).get(col) if isinstance(last_solution.get(house), dict) else None)
        if a == b:
            agree += 1
    return agree / len(keys)


_WS_TOKEN_RE = re.compile(r"\S+")


def compute_commitment_metrics(generated_text: str) -> dict:
    """Compute the gold-free subset of the frozen ZebraLogic commitment
    metrics for one generation. Returns a flat dict; every field is always
    present (None where undefined, e.g. no complete JSON found at all) so
    downstream aggregation never has to special-case missing keys.
    """
    text = generated_text or ""
    n_chars = len(text)

    first_obj, first_start, first_end = find_first_complete_json(text)
    has_first = first_obj is not None and isinstance(first_obj, dict)
    first_solution = first_obj.get("solution") if has_first else None

    solution_first = bool(has_first and grid_is_fully_specified(first_solution))

    if has_first and first_start is not None:
        pre_chars = first_start
        pre_tokens = len(_WS_TOKEN_RE.findall(text[:first_start]))
        first_pos_norm = (first_start / n_chars) if n_chars > 0 else None
    else:
        pre_chars = None
        pre_tokens = None
        first_pos_norm = None

    # reason_before_solution: non-trivial free text precedes the first
    # complete JSON object. Threshold of 20 chars separates "wrote the JSON
    # object immediately" from "wrote at least a clause of reasoning first";
    # frozen alongside the other thresholds in this module rather than left
    # implicit.
    reason_before_solution = bool(has_first and pre_chars is not None and pre_chars >= 20)

    return {
        "n_chars": n_chars,
        "has_first_json": has_first,
        "first_json_has_solution_key": bool(has_first and "solution" in first_obj),
        "solution_first_rate": solution_first,  # per-item boolean; caller averages
        "first_solution_pos": first_pos_norm,
        "pre_solution_chars": pre_chars,
        "pre_solution_tokens": pre_tokens,
        "reason_before_solution": reason_before_solution,
        "is_strict_loop": is_strict_loop(text),
        "_first_solution_grid": first_solution,  # kept for agreement calc; not a metric itself
    }
