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

zebralogic-easy-amend-01 (docs/zebralogic_easy_amendment_01.json, fixed
2026-09-05): `find_first_answer_json` is the correct "first answer" scan --
it skips any syntactically-complete JSON block that lacks a non-empty dict
"solution" and returns the first one that has it. `find_first_complete_json`
(kept, unchanged) returns the first syntactically complete JSON block
regardless of content, which is NOT the same thing whenever a model emits
some other JSON object before its actual answer. `compute_commitment_metrics`
and the first-answer scorer (zebralogic/first_answer_scorer.py) both use
`find_first_answer_json`.

@author: paveenhuang
"""

import json
import re


def _iter_complete_json_blocks(s: str):
    """Yield (parsed_dict, start, end) for every complete top-level {...}
    block in `s`, in the order they appear (left to right), skipping blocks
    that fail to parse as JSON (matching upstream's `.replace("\\n", "")`
    before `json.loads`, same as extract_last_complete_json /
    find_first_complete_json). A syntactically-complete-but-invalid block is
    skipped, not treated as a scan-terminating failure -- scanning continues
    for the next brace-balanced span.
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
                    start = first_start
                    candidate = s[start:i + 1]
                    first_start = None
                    try:
                        yield json.loads(candidate.replace("\n", "")), start, i + 1
                    except json.JSONDecodeError:
                        # not valid JSON -- keep scanning for a later,
                        # well-formed complete block rather than giving up
                        continue


def find_first_complete_json(s: str):
    """Forward-scanning counterpart to official_zebra_grid_scorer's
    extract_last_complete_json: returns the FIRST complete top-level {...}
    block (as a parsed dict), or None. Deliberately a separate function
    (not "just take the first element of a list of all JSON blocks") so its
    scanning direction is explicit and auditable against the official
    backward-scanning sibling.

    NOTE: this returns the first SYNTACTICALLY complete JSON object, with no
    check on its contents. It does NOT know whether that object is an
    "answer" -- a model that emits some other JSON object before its answer
    (e.g. restating part of the puzzle, or an intermediate structure) would
    have THAT object returned here. Use find_first_answer_json (below) when
    "first answer" specifically means "first JSON carrying a non-empty dict
    solution", which is what commitment metrics and first-answer scoring
    actually need.
    """
    for obj, start, end in _iter_complete_json_blocks(s):
        return obj, start, end
    return None, None, None


def find_first_answer_json(s: str):
    """Returns the FIRST complete top-level {...} block that carries a
    non-empty, dict-valued "solution" key (parsed_dict, start, end), or
    (None, None, None) if no such block exists. This is "first answer", not
    merely "first complete JSON" -- a syntactically complete JSON object with
    no "solution" key, a None "solution", or a non-dict "solution" (matching
    grid_is_fully_specified's own type check) is skipped and scanning
    continues to the next complete block, rather than being reported as "no
    answer" while a real answer JSON follows later in the text.

    Distinguishes: (a) the model's FIRST JSON object IS its answer (the
    common case) vs (b) the model emits some other JSON first (partial
    restatement, an intermediate structure) and only later emits the JSON
    that actually carries "solution" -- (b) must not be scored as no-answer
    just because it is not the syntactically-first JSON block.
    """
    for obj, start, end in _iter_complete_json_blocks(s):
        sol = obj.get("solution") if isinstance(obj, dict) else None
        if isinstance(sol, dict) and sol:
            return obj, start, end
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
# Matches the `"solution"` key inside the FIRST complete JSON object, so its
# match START is the character offset of that key within the full text.
# `\s*` between the quote and colon tolerates the model's own whitespace;
# this is a KEY-LOCATION scan restricted to the already-located first-JSON
# span (never applied to raw arbitrary text), so it cannot mismatch a
# "solution" substring appearing inside a string value elsewhere.
_SOLUTION_KEY_RE = re.compile(r'"solution"\s*:')


def _find_solution_key_offset(text: str, first_start: int, first_end: int):
    """Offset (within the FULL text) of the `"solution"` key inside the first
    complete JSON object spanning [first_start, first_end). Returns None if
    the object has no textual `"solution"` key in that span (should not
    happen when the parsed object already has a "solution" entry, but a
    caller-side scan is safer than assuming the parser's newline-stripped
    view lines up 1:1 with the raw span)."""
    m = _SOLUTION_KEY_RE.search(text, first_start, first_end)
    return m.start() if m else None


def compute_commitment_metrics(generated_text: str) -> dict:
    """Compute the gold-free subset of the frozen ZebraLogic commitment
    metrics for one generation. Returns a flat dict; every field is always
    present (None where undefined, e.g. no complete JSON found at all) so
    downstream aggregation never has to special-case missing keys.

    LOAD-BEARING: the official output format is a SINGLE JSON object
    `{"reasoning": "...", "solution": {...}}` (official_zebra_grid_template),
    so a model's CoT reasoning normally lives INSIDE the same JSON object,
    before the "solution" key -- not as free text before the JSON's opening
    brace. Measuring `pre_*`/`first_solution_pos` from the object's opening
    `{` (the previous implementation) would therefore read near-0 in nearly
    every case regardless of how much reasoning the model actually wrote,
    since the object typically opens immediately. These metrics are anchored
    on the `"solution"` KEY's position instead, so a long `"reasoning"` value
    preceding it is correctly counted as pre-commitment content. Free text
    genuinely preceding the JSON's opening brace (e.g. an introductory
    sentence before the object starts) is also included, since the offset is
    measured from the start of `generated_text`, not from `first_start`.

    LOAD-BEARING (fixed 2026-09-05): anchored on find_first_answer_json, NOT
    find_first_complete_json. The first SYNTACTICALLY complete JSON block in
    the text is not necessarily the first ANSWER -- a model that emits some
    other complete JSON object before the one carrying "solution" (e.g. a
    partial restatement, or an intermediate structure) would previously have
    that earlier, answer-less block anchor these metrics, reading
    has_first_json=True with first_solution=None and no defined
    first_solution_pos, while a real answer JSON followed later in the same
    text. find_first_answer_json skips any complete JSON lacking a non-empty
    dict "solution" and returns the first one that has it, so "first answer"
    means what it says.
    """
    text = generated_text or ""
    n_chars = len(text)

    first_obj, first_start, first_end = find_first_answer_json(text)
    has_first = first_obj is not None and isinstance(first_obj, dict)
    first_solution = first_obj.get("solution") if has_first else None

    # grid_is_fully_specified re-checks fill status (find_first_answer_json
    # only guarantees a non-empty dict "solution", not that every cell is
    # filled) -- solution_first_rate is specifically "committed a FULLY
    # SPECIFIED grid", a stricter condition than "found an answer JSON".
    solution_first = bool(has_first and grid_is_fully_specified(first_solution))

    if has_first and first_start is not None:
        sol_key_offset = _find_solution_key_offset(text, first_start, first_end)
        # Fall back to the object's opening brace only if the "solution" key
        # cannot be located textually (defensive; the parsed object already
        # has a "solution" entry whenever first_solution is not None) --
        # never silently report None while has_first is True with a real
        # solution present.
        anchor = sol_key_offset if sol_key_offset is not None else first_start
        pre_chars = anchor
        pre_tokens = len(_WS_TOKEN_RE.findall(text[:anchor]))
        first_pos_norm = (anchor / n_chars) if n_chars > 0 else None
    else:
        pre_chars = None
        pre_tokens = None
        first_pos_norm = None

    # reason_before_solution: non-trivial content precedes the "solution" key
    # (free text before the JSON, and/or a "reasoning" value inside it).
    # Threshold of 20 chars separates "wrote the solution immediately" from
    # "wrote at least a clause of reasoning first"; frozen alongside the
    # other thresholds in this module rather than left implicit.
    reason_before_solution = bool(has_first and pre_chars is not None and pre_chars >= 20)

    return {
        "n_chars": n_chars,
        # NOTE: with find_first_answer_json anchoring (fixed 2026-09-05),
        # has_first_json is now "found a JSON carrying a non-empty dict
        # solution" (i.e. "found an answer"), not merely "found any complete
        # JSON block" -- the field name is kept for schema stability with
        # already-collected data, but its meaning narrowed.
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
