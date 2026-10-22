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

@author: paveenhuang
"""

import json


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

    correct_cells = 0
    for house in solution_table:
        for column in solution_table[house]:
            if house in prediction_table and column in prediction_table[house]:
                truth_cell = solution_table[house][column].lower().strip()
                cell = prediction_table[house][column]
                # upstream handles None and list-valued cells defensively;
                # ported verbatim rather than assuming a clean string.
                if cell is None:
                    continue
                if isinstance(cell, list):
                    if not cell:
                        continue
                    predicted_cell = str(cell[0]).lower().strip()
                elif isinstance(cell, str):
                    predicted_cell = cell.lower().strip()
                else:
                    # upstream raises ValueError on an unknown type; we do the
                    # same rather than silently coercing an unexpected shape.
                    raise ValueError(f"Unknown cell value type: {type(cell)}")
                if truth_cell == predicted_cell:
                    correct_cells += 1

    return {
        "parsed": True,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "solved": correct_cells == total_cells,
        "prediction_table": prediction_table,
        "reasoning": reasoning,
    }
