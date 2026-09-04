#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The official ZebraLogic one-shot CoT prompt template, ported VERBATIM.

Source: WildEval/ZeroEval, `src/templates/ZEBRA_GRID.py`, fetched 2026-09-03
from https://raw.githubusercontent.com/WildEval/ZeroEval/main/src/templates/ZEBRA_GRID.py

`ZEBRA_GRID` below is a byte-for-byte copy of that file's `ZEBRA_GRID` string
(including its leading/trailing whitespace) -- do NOT edit its wording. If the
upstream template ever changes, re-fetch and re-freeze deliberately (bump the
protocol tag in docs/PREREG_ZEBRALOGIC_EASY.md), never hand-edit in place.

`apply_lgp_grid_template` is likewise ported from `src/_TEMPLATES.py`'s
function of the same name, adapted only to accept this project's item schema
(a dict with `puzzle` and a `solution` scaffold carrying `header`/`rows`,
matching the shape the official grid_mode dataset ships -- see
zebralogic/data_zebralogic.py). The per-item JSON template it builds is
label-free: it copies only the SHAPE of `solution` (house count, column
names), never any cell value, so it cannot leak gold into the prompt even
though the upstream `item["solution"]["rows"]` it reads from is itself blank
in the public split.

@author: paveenhuang
"""

import json

ZEBRA_GRID = """
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def apply_lgp_grid_template(item: dict) -> str:
    """Render the official one-shot CoT prompt for one ZebraLogic-grid item.

    `item` must carry `puzzle` (str) and `solution` (dict with `header`
    (list[str], first entry "House") and `rows` (list[list[str]], any
    placeholder content -- only len(rows) and header are read, never cell
    values), matching the shape of the public grid_mode dataset. This mirrors
    `apply_lgp_grid_template` in ZeroEval's `src/_TEMPLATES.py` exactly.
    """
    prompt_str = ZEBRA_GRID[:]
    prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House", f"expected header[0] == 'House', got {columns[0]!r}"
    json_template = {"reasoning": "___", "solution": {}}
    for i in range(num_houses):
        json_template["solution"][f"House {i + 1}"] = {
            columns[j]: "___" for j in range(1, len(columns))
        }
    json_str = json.dumps(json_template, indent=4)
    prompt_str = prompt_str.replace("{json_template}", json_str)
    return prompt_str
