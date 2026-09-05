#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The official ZebraLogic JSON extractor and per-item scorer, ported VERBATIM
from WildEval/ZeroEval (single-prediction path only).

Sources (fetched 2026-09-03):
  extract_last_complete_json  <- src/evaluation/eval_utils.py
  score_one_item               <- the `n_size == 1` ("single") branch inside
                                   eval_model(), src/evaluation/zebra_grid_eval.py

This project ALWAYS generates exactly one completion per item (never
best-of-N), so only that branch is ported -- the `best_of_n` /
`majority_of_n` / `longest_of_n` / reward-model branches in the upstream
`eval_model` never fire for a single prediction and are deliberately not
reproduced here. Cell/puzzle scoring semantics (case-insensitive stripped
string equality per cell; puzzle solved iff every cell of that puzzle
matches) are reproduced exactly, not paraphrased.

This module contains NO gold and NO network access. It is a pure scoring
function of (prediction_text, solution_table). The caller
(zebralogic/eval_zebralogic.py) is responsible for sourcing `solution_table`
from allenai/ZebraLogicBench-private and for refusing to run without it.

zebralogic-easy-amend-01 (docs/zebralogic_easy_amendment_01.json): the MAIN
score is now `score_one_item_first` (FIRST complete JSON, via
commitment_metrics.find_first_complete_json), not `score_one_item`
(official LAST-complete-JSON scoring). `score_one_item` /
`extract_last_complete_json` are UNCHANGED and kept as the LAST-JSON
sensitivity readout -- see eval_zebralogic.py's score_rows(). This module
never decides which one is MAIN; that is eval_zebralogic.py's job.

@author: paveenhuang
"""

import json


def score_prediction_table(prediction_table, solution_table):
    """Cell/puzzle comparison logic shared by score_one_item (LAST, official)
    and score_one_item_first (FIRST, amend-01 main). Ported from the same
    upstream cell-comparison block inside eval_model()'s single-prediction
    branch -- factored out so the two callers cannot silently diverge in how
    a cell is compared (None/list/str handling, case-insensitive strip)."""
    total_cells = sum(len(cols) for cols in solution_table.values())
    correct_cells = 0
    for house in solution_table:
        for column in solution_table[house]:
            if house in prediction_table and column in prediction_table[house]:
                truth_cell = solution_table[house][column].lower().strip()
                cell = prediction_table[house][column]
                if cell is None:
                    continue
                if isinstance(cell, list):
                    if not cell:
                        continue
                    predicted_cell = str(cell[0]).lower().strip()
                elif isinstance(cell, str):
                    predicted_cell = cell.lower().strip()
                else:
                    raise ValueError(f"Unknown cell value type: {type(cell)}")
                if truth_cell == predicted_cell:
                    correct_cells += 1
    return correct_cells, total_cells


def extract_last_complete_json(s: str):
    """Verbatim port of ZeroEval's extract_last_complete_json.

    Stack-based scan for the LAST complete top-level {...} block in `s`.
    Returns the parsed dict, or None if no complete JSON object is found or
    it fails to parse (after stripping newlines, matching upstream exactly --
    upstream calls `.replace("\\n", "")` before `json.loads`).
    """
    stack = []
    last_json_start = None
    last_json_str = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    last_json_str = s[last_json_start:i + 1]
                    last_json_start = None

    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None


def build_solution_table(solution: dict) -> dict:
    """Reshape the private gold `solution` (header/rows) into the
    House -> {column: value} table the scorer compares against, matching
    upstream's inline reshaping in eval_model() exactly:

        solution_table[f'House {i+1}'] = {columns[j]: rows[i][j]
                                           for j in range(1, len(columns))}
    """
    columns = solution["header"]
    assert columns[0] == "House", f"expected header[0] == 'House', got {columns[0]!r}"
    rows = solution["rows"]
    table = {}
    for i in range(len(rows)):
        table[f"House {i + 1}"] = {
            columns[j]: rows[i][j] for j in range(1, len(columns))
        }
    return table


def score_one_item(generated_text: str, solution_table: dict):
    """Verbatim port of the upstream single-prediction ("n_size == 1") scoring
    branch inside eval_model(). Returns a dict with:

        parsed            bool  -- a complete JSON object with a non-None
                                    "solution" key was found
        correct_cells     int
        total_cells       int
        solved            bool  -- correct_cells == total_cells (only
                                    meaningful when parsed is True; total_cells
                                    is still reported for the no-answer case so
                                    a caller can compute Cell Accuracy
                                    denominators without re-deriving it)
        prediction_table  dict or None
        reasoning         str   -- the model's own "reasoning" field, "" if
                                    absent or unparsed (upstream: `.get("reasoning", "")`)

    Matches upstream's own no-answer handling: if extraction yields None, or
    the parsed object has no "solution" key, or "solution" is None, the item
    counts as NOT parsed / no answer (upstream: predictions filtered to
    `p is not None and "solution" in p and p["solution"] is not None`, and an
    empty prediction list increments `no_answer`).
    """
    total_cells = sum(len(cols) for cols in solution_table.values())

    parsed = extract_last_complete_json(generated_text)
    if parsed is None or "solution" not in parsed or parsed["solution"] is None:
        return {
            "parsed": False,
            "correct_cells": 0,
            "total_cells": total_cells,
            "solved": False,
            "prediction_table": None,
            "reasoning": "",
        }

    prediction_table = parsed["solution"]
    reasoning = parsed.get("reasoning", "")

    # upstream handles None and list-valued cells defensively, and raises
    # ValueError on an unknown cell type; score_prediction_table reproduces
    # that behavior verbatim (factored out so score_one_item_first cannot
    # silently diverge in how a cell is compared).
    correct_cells, _ = score_prediction_table(prediction_table, solution_table)

    return {
        "parsed": True,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "solved": correct_cells == total_cells,
        "prediction_table": prediction_table,
        "reasoning": reasoning,
    }


def score_one_item_first(generated_text: str, solution_table: dict, find_first_complete_json):
    """zebralogic-easy-amend-01 MAIN scoring: identical to score_one_item
    (same no-answer handling, same cell-comparison rules via
    score_prediction_table) except the JSON extractor is the FORWARD-scanning
    `find_first_complete_json` (commitment_metrics.py) instead of the
    official backward-scanning `extract_last_complete_json`.

    `find_first_complete_json` is passed in rather than imported at module
    level, to keep this module's only import a stdlib one and to make the
    caller's choice of extractor explicit at the call site (this function is
    the ONLY place in the codebase that scores FIRST-JSON puzzle/cell
    accuracy -- it must never silently import a different extractor than the
    one commitment_metrics.py itself uses to define `solution_first_rate` /
    `first_solution_pos`, or the "first JSON" the accuracy number describes
    and the "first JSON" the commitment metrics describe could drift apart).

    Returns the same shape as score_one_item, i.e. a dict with parsed/
    correct_cells/total_cells/solved/prediction_table/reasoning. No-answer
    handling matches score_one_item exactly: extraction failure, a missing
    "solution" key, or a None "solution" value all count as NOT parsed.
    """
    total_cells = sum(len(cols) for cols in solution_table.values())

    parsed, _first_start, _first_end = find_first_complete_json(generated_text)
    if parsed is None or "solution" not in parsed or parsed["solution"] is None:
        return {
            "parsed": False,
            "correct_cells": 0,
            "total_cells": total_cells,
            "solved": False,
            "prediction_table": None,
            "reasoning": "",
        }

    prediction_table = parsed["solution"]
    reasoning = parsed.get("reasoning", "")
    correct_cells, _ = score_prediction_table(prediction_table, solution_table)

    return {
        "parsed": True,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "solved": correct_cells == total_cells,
        "prediction_table": prediction_table,
        "reasoning": reasoning,
    }
