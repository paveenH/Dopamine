#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First-answer puzzle/cell scorer for zebralogic-easy-amend-01
(docs/zebralogic_easy_amendment_01.json).

Deliberately kept OUT of official_zebra_grid_scorer.py, which is a byte-for-
byte VERBATIM port of WildEval/ZeroEval's own scoring code
(`extract_last_complete_json` / `score_one_item`) and must stay auditable
against its upstream source with nothing else mixed in. This module is this
project's OWN amendment logic, not an upstream port, so it lives separately.

score_one_item_first is the MAIN puzzle/cell scorer as of amend-01: it grades
the FIRST answer JSON (via commitment_metrics.find_first_answer_json), not
the official LAST-complete-JSON scoring (score_one_item /
extract_last_complete_json, both UNCHANGED in official_zebra_grid_scorer.py
and kept there as a sensitivity readout -- see eval_zebralogic.py's
score_rows()).

"First answer" means "the first complete top-level JSON object carrying a
non-empty, dict-valued 'solution' key" -- NOT "the first syntactically
complete JSON object" (find_first_complete_json). A model can emit some
other complete JSON object (a partial restatement, an intermediate
structure) before the one that actually carries its answer; scoring the
merely-first-complete-JSON would then misjudge that earlier, answer-less
block as "no answer" while a real answer followed later in the same text.
find_first_answer_json (commitment_metrics.py) is what actually implements
this "first answer" scan; this module only consumes it for scoring.

Cell/puzzle comparison rules (case-insensitive stripped string equality per
cell; None/list-valued cell handling; puzzle solved iff every cell of that
puzzle matches) are identical to score_one_item's, reproduced here rather
than imported, to keep official_zebra_grid_scorer.py free of any import of
project-specific code (it imports only stdlib `json`).

@author: paveenhuang
"""

from commitment_metrics import find_first_answer_json


def score_one_item_first(generated_text: str, solution_table: dict):
    """zebralogic-easy-amend-01 MAIN scoring: same no-answer handling and
    cell-comparison rules as official_zebra_grid_scorer.score_one_item,
    except the JSON extractor is find_first_answer_json (first complete JSON
    carrying a non-empty dict "solution") instead of the official backward-
    scanning extract_last_complete_json.

    Returns the same shape as score_one_item: a dict with parsed/
    correct_cells/total_cells/solved/prediction_table/reasoning. No-answer
    handling matches score_one_item exactly: extraction failure, a missing
    "solution" key, or a None "solution" value all count as NOT parsed --
    find_first_answer_json already guarantees a non-empty dict "solution"
    whenever it returns a non-None object, so the "solution" key/None checks
    below are defensive rather than expected to ever fire, kept for exact
    parity with score_one_item's own guard structure.
    """
    total_cells = sum(len(cols) for cols in solution_table.values())

    parsed, _first_start, _first_end = find_first_answer_json(generated_text)
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
                # matches score_one_item's own defensive None/list/str
                # handling verbatim, so the two scorers cannot silently
                # diverge in how a cell is compared.
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

    return {
        "parsed": True,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "solved": correct_cells == total_cells,
        "prediction_table": prediction_table,
        "reasoning": reasoning,
    }
