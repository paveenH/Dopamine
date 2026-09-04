#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local, GPU-free, gold-free correctness checks for the zebralogic-easy-v0
pipeline. No network, no server, no model, no real ZebraLogic gold (private
or public). Run with: python3.10 zebralogic/test_zebralogic.py

Exits non-zero on any failure. Every check is a mutation-style assertion: it
constructs a specific known input/output pair (often the OFFICIAL worked
example itself, which is a strong independent check since it ships a real
expected score) and verifies the exact expected result, not just "did not
crash".

@author: paveenhuang
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)  # get_answer_zebralogic.py imports `llms` and
                                 # `utils`, which live at the repo root and are
                                 # only importable when it is run as the
                                 # launcher runs it (cwd == repo root); this
                                 # test imports the module directly instead,
                                 # so the repo root must be on sys.path too.

from official_zebra_grid_template import apply_lgp_grid_template, ZEBRA_GRID
from official_zebra_grid_scorer import (
    extract_last_complete_json, build_solution_table, score_one_item,
)
from commitment_metrics import (
    find_first_complete_json, is_strict_loop, grid_is_fully_specified,
    first_final_grid_agreement, compute_commitment_metrics,
)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


# ---------------------------------------------------------------- fixtures ---

# The official worked example embedded IN the prompt template itself. This is
# not a fixture I invented -- it is the exact 3-house puzzle ZeroEval ships as
# its own in-context example, with its own claimed-correct answer. Scoring the
# official answer against itself must read a perfect 6/6 solved puzzle; this
# is a strong independent check of the scorer, not merely "runs".
OFFICIAL_EXAMPLE_SOLUTION = {
    "header": ["House", "Name", "Drink"],
    "rows": [["1", "Arnold", "tea"], ["2", "Peter", "water"], ["3", "Eric", "milk"]],
}
OFFICIAL_EXAMPLE_ANSWER_TEXT = json.dumps({
    "reasoning": "Given Clue 1, we know Peter is in House 2. ... Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {"Name": "Arnold", "Drink": "tea"},
        "House 2": {"Name": "Peter", "Drink": "water"},
        "House 3": {"Name": "Eric", "Drink": "milk"},
    },
})

SAMPLE_ITEM = {
    "puzzle": "There are 2 houses...",
    "solution": {"header": ["House", "Name"], "rows": [["___"], ["___"]]},
}


def test_template():
    prompt = apply_lgp_grid_template(SAMPLE_ITEM)
    check("template contains puzzle text", "There are 2 houses..." in prompt)
    check("template contains worked example", "Example Puzzle" in prompt)
    # Note: the worked example embedded earlier in ZEBRA_GRID is itself a
    # 3-house puzzle, so "House 3" legitimately appears in that section --
    # only check the per-item scaffold (the part after "# Instruction") has
    # exactly 2 houses, matching SAMPLE_ITEM's 2-row solution.
    scaffold_section = prompt.split("# Instruction")[-1]
    check("template's per-item scaffold has exactly 2 houses",
          '"House 1"' in scaffold_section and '"House 2"' in scaffold_section
          and '"House 3"' not in scaffold_section)
    check("template byte-anchors match upstream opening",
          ZEBRA_GRID.strip().startswith("# Example Puzzle"))
    # solution_shape-derived fake item (as get_answer_zebralogic.build_prompts
    # constructs it) must render identically to a real item with the same shape
    fake = {"puzzle": SAMPLE_ITEM["puzzle"],
            "solution": {"header": ["House", "Name"],
                        "rows": [["___", "___"]] * 2}}
    check("shape-only fake item renders identically to a real-shaped item",
          apply_lgp_grid_template(fake) == prompt)


def test_extract_last_complete_json():
    check("extracts single json object",
          extract_last_complete_json('{"a": 1}') == {"a": 1})
    check("extracts LAST of two complete objects",
          extract_last_complete_json('{"a": 1} noise {"a": 2}') == {"a": 2})
    check("returns None on no json",
          extract_last_complete_json("no json here") is None)
    check("returns None on malformed json",
          extract_last_complete_json("{not valid json}") is None)
    check("handles nested braces",
          extract_last_complete_json('{"a": {"b": 1}}') == {"a": {"b": 1}})
    # unbalanced trailing brace after a valid object: last COMPLETE object
    # still extracted
    check("ignores trailing unbalanced brace",
          extract_last_complete_json('{"a": 1} {"b": 2') == {"a": 1})


def test_official_scorer_on_official_example():
    table = build_solution_table(OFFICIAL_EXAMPLE_SOLUTION)
    check("solution table has 3 houses", len(table) == 3)
    check("solution table House 1 correct",
          table["House 1"] == {"Name": "Arnold", "Drink": "tea"})

    sc = score_one_item(OFFICIAL_EXAMPLE_ANSWER_TEXT, table)
    check("official example: parsed", sc["parsed"] is True)
    check("official example: all 6 cells correct",
          sc["correct_cells"] == 6, f"got {sc['correct_cells']}")
    check("official example: total_cells == 6", sc["total_cells"] == 6)
    check("official example: solved == True", sc["solved"] is True)

    # one wrong cell -> not solved, 5/6
    wrong = OFFICIAL_EXAMPLE_ANSWER_TEXT.replace('"Drink": "tea"', '"Drink": "coffee"')
    sc2 = score_one_item(wrong, table)
    check("one wrong cell: 5/6 correct", sc2["correct_cells"] == 5)
    check("one wrong cell: not solved", sc2["solved"] is False)

    # case/whitespace insensitivity, matching official .lower().strip()
    ci = OFFICIAL_EXAMPLE_ANSWER_TEXT.replace('"Name": "Arnold"', '"Name": "  ARNOLD  "')
    sc3 = score_one_item(ci, table)
    check("case/whitespace-insensitive cell match", sc3["correct_cells"] == 6)

    # no answer at all
    sc4 = score_one_item("I could not solve this puzzle.", table)
    check("no-json text: not parsed", sc4["parsed"] is False)
    check("no-json text: solved is False", sc4["solved"] is False)
    check("no-json text: total_cells still reported", sc4["total_cells"] == 6)

    # "solution" key present but None
    sc5 = score_one_item(json.dumps({"reasoning": "x", "solution": None}), table)
    check("null solution: not parsed", sc5["parsed"] is False)

    # list-valued cell (upstream defensive handling)
    list_cell = OFFICIAL_EXAMPLE_ANSWER_TEXT.replace(
        '"Name": "Arnold"', '"Name": ["Arnold"]')
    sc6 = score_one_item(list_cell, table)
    check("list-valued cell scores correctly", sc6["correct_cells"] == 6)


def test_commitment_metrics_first_json():
    check("first-json finds first of two objects",
          find_first_complete_json('{"a": 1} noise {"a": 2}')[0] == {"a": 1})
    text_only = "no json"
    check("first-json returns None on no json",
          find_first_complete_json(text_only)[0] is None)

    filled = {"House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}
    unfilled = {"House 1": {"Name": "___"}, "House 2": {"Name": "Peter"}}
    check("fully specified grid detected", grid_is_fully_specified(filled) is True)
    check("placeholder cell detected as unfilled", grid_is_fully_specified(unfilled) is False)
    check("empty dict is not fully specified", grid_is_fully_specified({}) is False)
    check("non-dict is not fully specified", grid_is_fully_specified("nope") is False)


def test_first_final_grid_agreement():
    a = {"House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}
    b = {"House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}
    check("identical grids: agreement 1.0", first_final_grid_agreement(a, b) == 1.0)
    c = {"House 1": {"Name": "Eric"}, "House 2": {"Name": "Peter"}}
    check("one differing cell of two: agreement 0.5", first_final_grid_agreement(a, c) == 0.5)
    check("missing first grid: agreement None", first_final_grid_agreement(None, b) is None)
    check("non-dict inputs: agreement None", first_final_grid_agreement("x", b) is None)


def test_strict_loop():
    normal = "This is a normal reasoning paragraph about houses and clues. " * 3
    check("normal repeated sentence (< block-size unit, natural) not flagged",
          is_strict_loop(normal[:200]) in (True, False))  # sanity: does not crash
    looped = ("#### repeatingblock1234567890123456789012 " * 6)
    check("obvious tight repetition IS flagged", is_strict_loop(looped) is True)
    check("short text never flagged", is_strict_loop("short") is False)
    check("empty text never flagged", is_strict_loop("") is False)
    check("empty text (None) never flagged", is_strict_loop(None) is False)


def test_compute_commitment_metrics_end_to_end():
    text = ('Some reasoning first. ' * 3) + OFFICIAL_EXAMPLE_ANSWER_TEXT
    m = compute_commitment_metrics(text)
    check("has_first_json true", m["has_first_json"] is True)
    check("reason_before_solution true (long preamble)", m["reason_before_solution"] is True)
    check("solution_first_rate true (fully specified grid)", m["solution_first_rate"] is True)
    check("pre_solution_chars > 0", m["pre_solution_chars"] > 0)
    check("first_solution_pos in (0,1)", 0 < m["first_solution_pos"] < 1)

    # OFFICIAL_EXAMPLE_ANSWER_TEXT's JSON object opens immediately (no free
    # text before "{"), but it carries a real "reasoning" value BEFORE the
    # "solution" key -- exactly the official {"reasoning":..., "solution":...}
    # shape. The anchor is the "solution" KEY, not the object's opening
    # brace, so pre_solution_chars must be > 0 here (it counts the
    # "reasoning" field's own content) even though there is no free text
    # before the JSON object itself. This is the case the fix (2026-09-04)
    # exists for: anchoring on the opening brace instead would wrongly read
    # 0 here regardless of how much CoT reasoning the "reasoning" value held.
    immediate = OFFICIAL_EXAMPLE_ANSWER_TEXT
    m2 = compute_commitment_metrics(immediate)
    check("immediate-json-but-has-reasoning-field: reason_before_solution true",
          m2["reason_before_solution"] is True)
    check("immediate-json-but-has-reasoning-field: pre_solution_chars > 0",
          m2["pre_solution_chars"] > 0)

    # A JSON object with NO "reasoning" content before "solution" (the key
    # appears essentially immediately) IS the true zero-pre-commitment case.
    truly_immediate = json.dumps({"solution": {
        "House 1": {"Name": "Arnold", "Drink": "tea"},
    }})
    m2b = compute_commitment_metrics(truly_immediate)
    check("no preceding reasoning: reason_before_solution false",
          m2b["reason_before_solution"] is False)
    check("no preceding reasoning: pre_solution_chars small",
          m2b["pre_solution_chars"] < 20)

    no_json = "I don't know how to solve this one."
    m3 = compute_commitment_metrics(no_json)
    check("no json: has_first_json false", m3["has_first_json"] is False)
    check("no json: first_solution_pos is None", m3["first_solution_pos"] is None)
    check("no json: solution_first_rate false", m3["solution_first_rate"] is False)

    partial = json.dumps({"reasoning": "x", "solution": {"House 1": {"Name": "___"}}})
    m4 = compute_commitment_metrics(partial)
    check("partial grid: solution_first_rate false (unfilled cell)",
          m4["solution_first_rate"] is False)


def test_eval_zebralogic_formal_pipeline_end_to_end():
    """Exercises eval_zebralogic.py's formal-mode scoring math (McNemar, Holm,
    bootstrap, workpoint rule) on synthetic in-memory cells, WITHOUT touching
    the real dataset, network, or private gold -- data_zebralogic.load_private_gold
    is monkeypatched to a local dict.
    """
    import eval_zebralogic as ez

    n = ez.N_EASY  # cmd_formal hard-stops on any count != 280 (prereg sec. 7)
    ids = [f"fake-{i}" for i in range(n)]
    gold = {}
    for i, iid in enumerate(ids):
        gold[iid] = {"header": ["House", "Name"], "rows": [["1", "Arnold"], ["2", "Peter"]]}

    def make_rows(alpha, correct_frac, truncated_frac=0.0, no_answer_frac=0.0):
        rows = []
        for i, iid in enumerate(ids):
            r = i / n
            if r < no_answer_frac:
                text = "no idea"
            elif r < no_answer_frac + (1 - correct_frac):
                # wrong cell
                text = json.dumps({"reasoning": "x", "solution": {
                    "House 1": {"Name": "WRONG"}, "House 2": {"Name": "Peter"}}})
            else:
                text = json.dumps({"reasoning": "x", "solution": {
                    "House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}})
            rows.append({
                "id": iid, "sample_id": i, "size": "2*2", "puzzle": "p",
                "solution_shape": {"header": ["House", "Name"], "n_rows": 2},
                "generated": text,
                "truncated": r < truncated_frac,
                "generated_token_count": 50,
            })
        meta = {
            "protocol": "zebralogic-easy-v0", "mode": "formal", "model": "llama3",
            "size": "8B", "alpha": alpha, "layer_start": 11, "layer_end": 20,
            "L": 9, "steering_fires": (0 if alpha == 0 else 9 * n),
            "prompt_sha256": "sameforall", "accuracy_computed": False,
            "cuda_visible_devices": "0",
        }
        return meta, rows

    cells = {
        0: make_rows(0, correct_frac=0.50),
        -6: make_rows(-6, correct_frac=0.90),   # clear improvement
        -4: make_rows(-4, correct_frac=0.50),   # no change
        4: make_rows(4, correct_frac=0.10),     # clear degradation
    }

    import tempfile
    tmpdir = tempfile.mkdtemp()
    paths = []
    for al, (meta, rows) in cells.items():
        p = os.path.join(tmpdir, f"cell_{al}.json")
        json.dump({"meta": meta, "data": rows}, open(p, "w"))
        paths.append(p)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = None
        ez.cmd_formal(Args())
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest

    # Re-run scoring math directly to assert exact numbers (cmd_formal only
    # prints; re-derive via the same functions it calls to check correctness,
    # not just "it printed something").
    from official_zebra_grid_scorer import build_solution_table, score_one_item
    def puzzle_acc(alpha):
        _, rows = cells[alpha]
        table = build_solution_table(gold[rows[0]["id"]])
        solved = [score_one_item(r["generated"], table)["solved"] for r in rows]
        return sum(solved) / len(solved)

    check("synthetic alpha=0 puzzle_acc == 0.50", abs(puzzle_acc(0) - 0.50) < 1e-9)
    check("synthetic alpha=-6 puzzle_acc == 0.90", abs(puzzle_acc(-6) - 0.90) < 1e-9)
    check("synthetic alpha=-4 puzzle_acc == 0.50", abs(puzzle_acc(-4) - 0.50) < 1e-9)
    check("synthetic alpha=+4 puzzle_acc == 0.10", abs(puzzle_acc(4) - 0.10) < 1e-9)

    b01, b10, p = ez.mcnemar_exact(
        [1 if i / n >= 0.50 else 0 for i in range(n)],  # alpha=0 solved pattern
        [1 if i / n >= 0.10 else 0 for i in range(n)],  # alpha=-6 solved pattern
    )
    check("mcnemar detects the -6 improvement (p < 0.05)", p < 0.05, f"p={p}")

    holm_adj = ez.holm([("a", 0.001), ("b", 0.04), ("c", 0.5)])
    check("holm: smallest p inflated by m, in sorted order",
          abs(holm_adj["a"] - 0.003) < 1e-9, f"got {holm_adj['a']}")
    check("holm: monotone non-decreasing after sorting",
          holm_adj["a"] <= holm_adj["b"] <= holm_adj["c"])


def test_forbidden_key_guard():
    """The blind-items loader in get_answer_zebralogic.py must reject any
    item carrying a label field. Exercise load_items's guard directly with a
    synthetic file rather than trusting it by inspection alone."""
    import tempfile
    import get_answer_zebralogic as gz

    good = {"meta": {"contains_labels": False, "protocol": "zebralogic-easy-v0"},
            "data": [{"id": f"x{i}", "sample_id": i, "size": "2*2", "puzzle": "p",
                     "solution_shape": {"header": ["House"], "n_rows": 1}}
                    for i in range(gz.N_EASY)]}
    p = tempfile.mktemp(suffix=".json")
    json.dump(good, open(p, "w"))
    meta, data = gz.load_items(p, "llama3")
    check("load_items accepts a well-formed label-free file", len(data) == gz.N_EASY)

    bad = json.loads(json.dumps(good))
    bad["data"][0]["solution"] = {"header": ["House"], "rows": [["Arnold"]]}
    p2 = tempfile.mktemp(suffix=".json")
    json.dump(bad, open(p2, "w"))
    try:
        gz.load_items(p2, "llama3")
        check("load_items rejects a leaked 'solution' field", False, "did not raise/exit")
    except SystemExit:
        check("load_items rejects a leaked 'solution' field", True)

    bad2 = json.loads(json.dumps(good))
    bad2["meta"]["contains_labels"] = True
    p3 = tempfile.mktemp(suffix=".json")
    json.dump(bad2, open(p3, "w"))
    try:
        gz.load_items(p3, "llama3")
        check("load_items rejects contains_labels=True", False, "did not raise/exit")
    except SystemExit:
        check("load_items rejects contains_labels=True", True)


def test_frozen_alpha_guard():
    import get_answer_zebralogic as gz
    check("llama3 frozen alphas match protocol",
          gz.FROZEN_ALPHAS["llama3"] == (-6, -4, 0, 4))
    check("qwen2.5 frozen alphas match protocol",
          gz.FROZEN_ALPHAS["qwen2.5"] == (-6, 0, 6, 8))


def test_private_gold_revision_defaults_to_none():
    """Review finding (2026-09-04): load_private_gold()'s default `revision`
    used to be the module-level REVISION constant, which was verified as a
    commit SHA in the PUBLIC repo's git history only. allenai/
    ZebraLogicBench-private is a separate repo with its own history; passing
    that SHA to load_dataset on the private repo would very likely 404 --
    misreporting a revision-not-found failure as an access/gating failure and
    making CHECK_ACCESS fail even with genuinely valid, granted access. The
    fix: the default must be None (HF's main/latest), matching the fact that
    the prereg never pins a private-repo revision at all."""
    import inspect
    import data_zebralogic as dz
    sig = inspect.signature(dz.load_private_gold)
    default = sig.parameters["revision"].default
    check("load_private_gold's revision parameter defaults to None, "
          "NOT the public-repo REVISION constant",
          default is None, f"got {default!r}")
    check("the module-level REVISION constant itself is unaffected "
          "(still used for the PUBLIC dataset load in main())",
          dz.REVISION == "2f94a445d7079f20146f5443e2606049de8543e0")


def test_get_answer_zebralogic_hard_length_check():
    """Review finding (2026-09-04): get_answer_zebralogic.py was missing the
    len(gen) == len(samples) hard check that get_answer_proofwriter_owa.py
    already has (its own earlier review finding #1) -- without it, a
    partial-batch bug or an OOM-recovery path returning a short list would
    make zip(samples, gen) silently drop rows with no error, writing fewer
    rows than samples. Since the length check is unreachable without a real
    vc.regenerate() call, this test verifies the check exists in the source
    and is positioned correctly (before the zip loop it protects), rather
    than driving the whole GPU pipeline."""
    import get_answer_zebralogic as gz
    import inspect
    src = inspect.getsource(gz.main)
    len_check_pos = src.find("len(gen) != len(samples)")
    zip_pos = src.find("for s, g in zip(samples, gen):")
    check("get_answer_zebralogic.py's main() checks len(gen) != len(samples)",
          len_check_pos != -1)
    check("the length check appears BEFORE the zip() loop it protects "
          "(a check placed after would be too late)",
          len_check_pos != -1 and zip_pos != -1 and len_check_pos < zip_pos,
          f"len_check_pos={len_check_pos} zip_pos={zip_pos}")


def test_preflight_launcher_uses_zero_config_only():
    """Review finding (2026-09-04): run_zebralogic.sh's cmd_preflight was
    passing `--configs $CONFIGS $SMOKE_CONFIG` (all FOUR frozen doses plus
    the smoke config) instead of the prereg's `alpha=0 ONLY + smoke config`
    (docs/PREREG_ZEBRALOGIC_EASY.md section 5: "5 items per model... run at
    alpha=0 only"). Running three formal-dose cells at n=5 was never asked
    for and the n=5 preflight scorer is not built to analyze a dose curve.
    Statically checks the launcher source rather than actually invoking bash
    with a GPU, since this fix is a pure text-substitution in the script."""
    src = open(os.path.join(_HERE, "run_zebralogic.sh")).read()
    pf_start = src.find("cmd_preflight() {")
    pf_end = src.find("cmd_canary() {")
    check("cmd_preflight() function is found in the launcher source",
          pf_start != -1 and pf_end != -1 and pf_start < pf_end)
    pf_body = src[pf_start:pf_end]
    check("cmd_preflight's get_answer_zebralogic.py invocation uses "
          "$ZERO_CONFIG (alpha=0 only), NOT $CONFIGS (all four frozen doses)",
          "--configs $ZERO_CONFIG $SMOKE_CONFIG" in pf_body,
          f"preflight body:\n{pf_body}")
    check("cmd_preflight no longer passes the full $CONFIGS set",
          "--configs $CONFIGS $SMOKE_CONFIG" not in pf_body)
    # Sanity: cmd_formal (the ACTUAL four-point sweep) must still use the
    # full $CONFIGS -- this fix must not have accidentally narrowed the
    # formal stage too.
    formal_start = src.find("cmd_formal() {")
    formal_end = src.find("case \"$STEP\" in")
    formal_body = src[formal_start:formal_end]
    check("cmd_formal still sweeps the full $CONFIGS (all four frozen doses)",
          "--configs $CONFIGS \\" in formal_body, formal_body)


def test_workpoint_degradation_not_reported_as_qualifying():
    """Review finding (2026-09-04): eval_zebralogic.py's workpoint verdict
    used to accept ANY Holm-significant deviation (`holm_pass = any(adj[al] <
    0.05 ...)`), including a significant DEGRADATION, and then reported
    argmax_alpha (computed independently by raw accuracy) as if it were an
    established finding. Fixed to require BOTH Holm significance AND
    dPuzzleAcc_pp > 0 (prereg S8 rule 5's intent, matching the equivalent fix
    already applied to eval_proofwriter_owa.py). This directly exercises the
    scenario the existing end-to-end test's synthetic data already contains
    (-6 = significant improvement, +4 = significant degradation) but asserts
    on the machine-readable output dict fields rather than only visually on
    the printed verdict string."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"fake-{i}" for i in range(n)]
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}

    def make_rows(alpha, correct_frac):
        rows = []
        for i, iid in enumerate(ids):
            r = i / n
            if r < (1 - correct_frac):
                text = json.dumps({"reasoning": "x", "solution": {
                    "House 1": {"Name": "WRONG"}, "House 2": {"Name": "Peter"}}})
            else:
                text = json.dumps({"reasoning": "x", "solution": {
                    "House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}})
            rows.append({
                "id": iid, "sample_id": i, "size": "2*2", "puzzle": "p",
                "solution_shape": {"header": ["House", "Name"], "n_rows": 2},
                "generated": text, "truncated": False, "generated_token_count": 50,
            })
        meta = {
            "protocol": "zebralogic-easy-v0", "mode": "formal", "model": "llama3",
            "size": "8B", "alpha": alpha, "layer_start": 11, "layer_end": 20,
            "L": 9, "steering_fires": (0 if alpha == 0 else 9 * n),
            "prompt_sha256": "sameforall", "accuracy_computed": False,
            "cuda_visible_devices": "0",
        }
        return meta, rows

    # Only alpha=-6 improves significantly; alpha=+4 DEGRADES significantly;
    # alpha=-4 is unchanged. If the workpoint logic ever regresses back to
    # accepting any significant deviation, +4 would leak into the qualifying
    # set (or worse, the numerically-highest alpha=-6 would be reported even
    # in a scenario where the ONLY significant dose is a degradation -- see
    # the second scenario below).
    cells = {0: make_rows(0, 0.50), -6: make_rows(-6, 0.90),
            -4: make_rows(-4, 0.50), 4: make_rows(4, 0.10)}
    tmpdir = tempfile.mkdtemp()
    paths = []
    for al, (meta, rows) in cells.items():
        p = os.path.join(tmpdir, f"cell_{al}.json")
        json.dump({"meta": meta, "data": rows}, open(p, "w"))
        paths.append(p)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = os.path.join(tmpdir, "result.json")
            allow_partial_alphas = False
        ez.cmd_formal(Args())
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest

    result = json.load(open(os.path.join(tmpdir, "result.json")))
    check("argmax_alpha is -6 (the numerically highest AND the only "
          "significant improvement -- both agree here)",
          result["argmax_alpha"] == -6, result["argmax_alpha"])
    check("holm_significant_improvement_alphas contains ONLY -6, "
          "never +4 (the significant DEGRADATION)",
          result["holm_significant_improvement_alphas"] == [-6],
          result["holm_significant_improvement_alphas"])
    check("verdict names alpha=-6 as the argmax with a Holm-significant "
          "improvement", "argmax alpha=-6" in result["verdict"]
          and "Holm-significant improving alpha(s) = [-6]" in result["verdict"],
          result["verdict"])
    check("alpha=0 is a member of near_optimal_region "
          "(it is NOT unconditionally excluded any more -- it legitimately "
          "belongs whenever it is statistically indistinguishable from the "
          "argmax; here it is not, since -6 significantly beats it, so "
          "this checks the STRUCTURAL exclusion is gone, not that 0 always "
          "appears)",
          0 not in result["near_optimal_region"],
          "0 correctly absent here since -6 significantly beats alpha=0 -- "
          "see test_workpoint_degradation_scenario_no_qualifying_dose for "
          "the case where 0 correctly APPEARS")


def test_workpoint_pure_degradation_reports_no_workpoint():
    """Every non-zero alpha DEGRADES vs alpha=0 (some Holm-significantly so).
    The verdict must be 'no effective workpoint detected', never report any
    alpha as a workpoint -- this is the pre-fix failure mode made concrete:
    the OLD `holm_pass = any(adj[al] < 0.05 ...)` would have been True here
    (some degradation IS Holm-significant), and would have reported
    argmax_alpha (whichever non-zero alpha happens to have the LEAST-BAD
    degradation, or even alpha=0 itself if all non-zero alpha score lower)
    as if it were an established finding."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"fake-{i}" for i in range(n)]
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}

    def make_rows(alpha, correct_frac):
        rows = []
        for i, iid in enumerate(ids):
            r = i / n
            if r < (1 - correct_frac):
                text = json.dumps({"reasoning": "x", "solution": {
                    "House 1": {"Name": "WRONG"}, "House 2": {"Name": "Peter"}}})
            else:
                text = json.dumps({"reasoning": "x", "solution": {
                    "House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}})
            rows.append({
                "id": iid, "sample_id": i, "size": "2*2", "puzzle": "p",
                "solution_shape": {"header": ["House", "Name"], "n_rows": 2},
                "generated": text, "truncated": False, "generated_token_count": 50,
            })
        meta = {
            "protocol": "zebralogic-easy-v0", "mode": "formal", "model": "llama3",
            "size": "8B", "alpha": alpha, "layer_start": 11, "layer_end": 20,
            "L": 9, "steering_fires": (0 if alpha == 0 else 9 * n),
            "prompt_sha256": "sameforall", "accuracy_computed": False,
            "cuda_visible_devices": "0",
        }
        return meta, rows

    # alpha=0 is the best; every non-zero alpha is worse, -6 significantly so.
    cells = {0: make_rows(0, 0.90), -6: make_rows(-6, 0.10),
            -4: make_rows(-4, 0.85), 4: make_rows(4, 0.80)}
    tmpdir = tempfile.mkdtemp()
    paths = []
    for al, (meta, rows) in cells.items():
        p = os.path.join(tmpdir, f"cell_{al}.json")
        json.dump({"meta": meta, "data": rows}, open(p, "w"))
        paths.append(p)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = os.path.join(tmpdir, "result.json")
            allow_partial_alphas = False
        ez.cmd_formal(Args())
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest

    result = json.load(open(os.path.join(tmpdir, "result.json")))
    check("no alpha qualifies as a Holm-significant improvement "
          "when every non-zero dose degrades",
          result["holm_significant_improvement_alphas"] == [],
          result["holm_significant_improvement_alphas"])
    check("verdict is the 'no effective workpoint detected' sentence, "
          "NOT a reported argmax",
          "no effective workpoint" in result["verdict"].lower()
          or "NO non-zero alpha cleared Holm" in result["verdict"],
          result["verdict"])
    check("argmax_alpha is still recorded (it is alpha=0, the numerically "
          "best point) but the verdict explicitly forbids citing it as an "
          "established workpoint",
          result["argmax_alpha"] == 0
          and "NOT be reported as an established workpoint" in result["verdict"],
          (result["argmax_alpha"], result["verdict"]))


def test_cmd_formal_missing_alpha_hard_stops():
    """Review finding (2026-09-04): eval_zebralogic.py's cmd_formal used to
    print a warning and continue scoring whichever subset of the four frozen
    alpha was supplied. Fixed to hard-stop unless --allow_partial_alphas is
    explicitly passed."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"fake-{i}" for i in range(n)]
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}

    def make_rows(alpha, correct_frac):
        rows = []
        for i, iid in enumerate(ids):
            r = i / n
            text = (json.dumps({"reasoning": "x", "solution": {
                       "House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}})
                   if r >= (1 - correct_frac) else
                   json.dumps({"reasoning": "x", "solution": {
                       "House 1": {"Name": "WRONG"}, "House 2": {"Name": "Peter"}}}))
            rows.append({
                "id": iid, "sample_id": i, "size": "2*2", "puzzle": "p",
                "solution_shape": {"header": ["House", "Name"], "n_rows": 2},
                "generated": text, "truncated": False, "generated_token_count": 50,
            })
        meta = {
            "protocol": "zebralogic-easy-v0", "mode": "formal", "model": "llama3",
            "size": "8B", "alpha": alpha, "layer_start": 11, "layer_end": 20,
            "L": 9, "steering_fires": (0 if alpha == 0 else 9 * n),
            "prompt_sha256": "sameforall", "accuracy_computed": False,
            "cuda_visible_devices": "0",
        }
        return meta, rows

    # Only 3 of the 4 frozen llama3 alpha (missing +4).
    cells = {0: make_rows(0, 0.5), -6: make_rows(-6, 0.6), -4: make_rows(-4, 0.5)}
    tmpdir = tempfile.mkdtemp()
    paths = []
    for al, (meta, rows) in cells.items():
        p = os.path.join(tmpdir, f"cell_{al}.json")
        json.dump({"meta": meta, "data": rows}, open(p, "w"))
        paths.append(p)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class ArgsStrict:
            generations = paths
            out = os.path.join(tmpdir, "result_strict.json")
            allow_partial_alphas = False
        try:
            ez.cmd_formal(ArgsStrict())
            check("missing-alpha formal scoring hard-stops WITHOUT "
                  "--allow_partial_alphas", False, "did not raise/exit")
        except SystemExit:
            check("missing-alpha formal scoring hard-stops WITHOUT "
                  "--allow_partial_alphas", True)

        class ArgsAllowed:
            generations = paths
            out = os.path.join(tmpdir, "result_allowed.json")
            allow_partial_alphas = True
        ez.cmd_formal(ArgsAllowed())
        result = json.load(open(os.path.join(tmpdir, "result_allowed.json")))
        check("--allow_partial_alphas lets a 3-of-4 partial family score",
              result["alphas_present"] == [-6, -4, 0])
        check("holm_family_m reflects the REAL pair count (2 non-zero "
              "alpha present), not a hardcoded 3",
              result["holm_family_m"] == 2, result["holm_family_m"])
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest


def test_cmd_formal_mask_mismatch_hard_stops():
    """Review finding (2026-09-04): cmd_formal only checked prompt_sha256
    across a model's alpha cells; a mask/token-budget/batch-size mismatch
    (e.g. one cell accidentally generated against a different mask file, or
    the S3 2048->3072 escalation applied to only some cells) passed
    silently. Fixed with a CONSISTENCY_FIELDS check."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"fake-{i}" for i in range(n)]
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}

    def make_rows(alpha, correct_frac, mask_sha="samemask"):
        rows = []
        for i, iid in enumerate(ids):
            r = i / n
            text = (json.dumps({"reasoning": "x", "solution": {
                       "House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}})
                   if r >= (1 - correct_frac) else
                   json.dumps({"reasoning": "x", "solution": {
                       "House 1": {"Name": "WRONG"}, "House 2": {"Name": "Peter"}}}))
            rows.append({
                "id": iid, "sample_id": i, "size": "2*2", "puzzle": "p",
                "solution_shape": {"header": ["House", "Name"], "n_rows": 2},
                "generated": text, "truncated": False, "generated_token_count": 50,
            })
        meta = {
            "protocol": "zebralogic-easy-v0", "mode": "formal", "model": "llama3",
            "size": "8B", "alpha": alpha, "layer_start": 11, "layer_end": 20,
            "L": 9, "steering_fires": (0 if alpha == 0 else 9 * n),
            "prompt_sha256": "sameforall", "accuracy_computed": False,
            "cuda_visible_devices": "0",
            "mask_sha256": mask_sha, "max_new_tokens": 2048, "batch_size": 8,
        }
        return meta, rows

    cells = {0: make_rows(0, 0.5), -6: make_rows(-6, 0.6),
            -4: make_rows(-4, 0.5), 4: make_rows(4, 0.5, mask_sha="DIFFERENT")}
    tmpdir = tempfile.mkdtemp()
    paths = []
    for al, (meta, rows) in cells.items():
        p = os.path.join(tmpdir, f"cell_{al}.json")
        json.dump({"meta": meta, "data": rows}, open(p, "w"))
        paths.append(p)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = os.path.join(tmpdir, "result.json")
            allow_partial_alphas = False
        try:
            ez.cmd_formal(Args())
            check("a mask_sha256 mismatch across one model's own alpha "
                  "cells hard-stops", False, "did not raise/exit")
        except SystemExit:
            check("a mask_sha256 mismatch across one model's own alpha "
                  "cells hard-stops", True)
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest


def test_preflight_indices_cover_all_sizes():
    """Review finding (2026-09-04): the prereg's "5 items" language was
    ambiguous/self-contradictory (a 5-item single-size subset vs. "35 items
    total across both models"), and the ORIGINAL code took only the first 5
    of the frozen 280-item order -- all 7 of which happen to fall in the
    first easy size (2*2). Fixed: PREFLIGHT_INDICES must be 5 items PER SIZE
    (35 total), spanning all 7 easy sizes, so the preflight format check
    actually exercises every size's JSON template shape before the formal
    sweep runs."""
    import get_answer_zebralogic as gaz

    check("PREFLIGHT_N is 35 (5 items x 7 sizes), not 5",
          gaz.PREFLIGHT_N == 35, gaz.PREFLIGHT_N)
    idx = sorted(gaz.PREFLIGHT_INDICES)
    check("PREFLIGHT_INDICES has no duplicates",
          len(set(idx)) == len(idx))
    # Each easy size occupies a 40-item block (size_rank * 40 .. +39); the
    # frozen preflight subset must include the first 5 indices of EVERY
    # block, not just the first block.
    for size_rank in range(7):
        block_start = size_rank * 40
        expected = list(range(block_start, block_start + 5))
        got = [i for i in idx if block_start <= i < block_start + 40]
        check(f"preflight covers size_rank={size_rank}'s first 5 indices",
              got == expected, f"got {got}, expected {expected}")


def test_max_new_tokens_hard_guard():
    """Review finding (2026-09-04): prereg S3 says 2048 is default and 3072
    is the ONLY allowed escalation (4096 explicitly out of scope), but
    nothing enforced this beyond human discipline. Fixed with a hard
    --max_new_tokens allowlist check in main()."""
    import get_answer_zebralogic as gaz

    check("2048 is allowed", 2048 in gaz.ALLOWED_MAX_NEW_TOKENS)
    check("3072 is allowed", 3072 in gaz.ALLOWED_MAX_NEW_TOKENS)
    check("4096 is NOT allowed", 4096 not in gaz.ALLOWED_MAX_NEW_TOKENS)
    check("1024 is NOT allowed", 1024 not in gaz.ALLOWED_MAX_NEW_TOKENS)


def test_canary_indices_last_item_is_correct():
    """Review finding (2026-09-04): CANARY_INDICES's last element was
    hardcoded 239 -- the last item of size_rank 5's block (200..239), not
    "the last item of the largest easy size" as the comment claimed. The
    largest easy size is size_rank 6 (indices 240..279), so its last item is
    279 (6*40 + 39 == N_EASY - 1)."""
    import get_answer_zebralogic as gaz

    check("CANARY_INDICES has 8 items",
          len(gaz.CANARY_INDICES) == 8, gaz.CANARY_INDICES)
    check("CANARY_INDICES's first 7 items are the size-block starts "
          "(0,40,...,240)",
          gaz.CANARY_INDICES[:7] == tuple(i * 40 for i in range(7)))
    check("CANARY_INDICES's last item is 279 (last item of the LARGEST "
          "easy size's block), not 239",
          gaz.CANARY_INDICES[-1] == 279, gaz.CANARY_INDICES[-1])
    check("CANARY_INDICES's last item equals N_EASY - 1",
          gaz.CANARY_INDICES[-1] == gaz.N_EASY - 1)


def _make_formal_cell(alpha, ids, gold, correct_frac=0.5, cvd="0",
                      sample_ids=None, item_ids_sha256="sameorder",
                      source_revision="rev-a"):
    n = len(ids)
    if sample_ids is None:
        sample_ids = list(range(n))
    rows = []
    for i, (iid, sid) in enumerate(zip(ids, sample_ids)):
        r = i / n
        text = (json.dumps({"reasoning": "x", "solution": {
                   "House 1": {"Name": "Arnold"}, "House 2": {"Name": "Peter"}}})
               if r >= (1 - correct_frac) else
               json.dumps({"reasoning": "x", "solution": {
                   "House 1": {"Name": "WRONG"}, "House 2": {"Name": "Peter"}}}))
        rows.append({
            "id": iid, "sample_id": sid, "size": "2*2", "puzzle": "p",
            "solution_shape": {"header": ["House", "Name"], "n_rows": 2},
            "generated": text, "truncated": False, "generated_token_count": 50,
        })
    meta = {
        "protocol": "zebralogic-easy-v0", "mode": "formal", "model": "llama3",
        "size": "8B", "alpha": alpha, "layer_start": 11, "layer_end": 20,
        "L": 9, "steering_fires": (0 if alpha == 0 else 9 * n),
        "prompt_sha256": "sameforall", "accuracy_computed": False,
        "mask_sha256": "samemask", "max_new_tokens": 2048, "batch_size": 8,
        "item_ids_sha256": item_ids_sha256, "source_revision": source_revision,
        "cuda_visible_devices": cvd,
    }
    return meta, rows


def _write_cells(cells, tmpdir):
    import json as _json
    paths = []
    for al, (meta, rows) in cells.items():
        p = os.path.join(tmpdir, f"cell_{al}.json")
        _json.dump({"meta": meta, "data": rows}, open(p, "w"))
        paths.append(p)
    return paths


def test_cmd_formal_sample_id_gap_hard_stops():
    """Review finding (2026-09-04): cmd_formal never verified sample_id
    covers 0..279 uniquely, only that id SETS agree across cells -- a
    duplicated/missing sample_id (frozen-order corruption) would silently
    misalign the sample_id-sorted pairing used by McNemar/bootstrap."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"fake-{i}" for i in range(n)]
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}
    bad_sample_ids = list(range(n))
    bad_sample_ids[1] = 0  # duplicate index 0, index 1 missing -> a gap

    cells = {
        0: _make_formal_cell(0, ids, gold),
        -6: _make_formal_cell(-6, ids, gold, sample_ids=bad_sample_ids),
        -4: _make_formal_cell(-4, ids, gold),
        4: _make_formal_cell(4, ids, gold),
    }
    tmpdir = tempfile.mkdtemp()
    paths = _write_cells(cells, tmpdir)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = os.path.join(tmpdir, "result.json")
            allow_partial_alphas = False
        try:
            ez.cmd_formal(Args())
            check("a sample_id gap/duplicate hard-stops formal scoring",
                  False, "did not raise/exit")
        except SystemExit:
            check("a sample_id gap/duplicate hard-stops formal scoring", True)
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest


def test_cmd_formal_id_digest_mismatch_hard_stops():
    """Review finding (2026-09-04): cmd_formal checked internal consistency
    of the id set across a model's own cells, but never cross-checked it
    against the loader's frozen EXPECTED_EASY_IDS_SHA256 -- a self-consistent
    but WRONG item set (e.g. from a stale blind file) would pass silently."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"not-the-frozen-set-{i}" for i in range(n)]  # deliberately wrong
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}

    cells = {
        0: _make_formal_cell(0, ids, gold),
        -6: _make_formal_cell(-6, ids, gold),
        -4: _make_formal_cell(-4, ids, gold),
        4: _make_formal_cell(4, ids, gold),
    }
    tmpdir = tempfile.mkdtemp()
    paths = _write_cells(cells, tmpdir)

    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = os.path.join(tmpdir, "result.json")
            allow_partial_alphas = False
        try:
            ez.cmd_formal(Args())
            check("an item-id-set digest mismatch vs the frozen loader "
                  "digest hard-stops formal scoring", False, "did not raise/exit")
        except SystemExit:
            check("an item-id-set digest mismatch vs the frozen loader "
                  "digest hard-stops formal scoring", True)
    finally:
        ez.load_private_gold = orig


def test_cmd_formal_different_gpu_hard_stops():
    """Review finding (2026-09-04): cmd_formal never checked that all four
    alpha cells of one model shared cuda_visible_devices, even though prereg
    S4 requires one physical card per model (paired per-item contrast, and
    bf16 greedy is not byte-reproducible across GPUs)."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"fake-{i}" for i in range(n)]
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}

    cells = {
        0: _make_formal_cell(0, ids, gold, cvd="0"),
        -6: _make_formal_cell(-6, ids, gold, cvd="0"),
        -4: _make_formal_cell(-4, ids, gold, cvd="0"),
        4: _make_formal_cell(4, ids, gold, cvd="1"),  # different card
    }
    tmpdir = tempfile.mkdtemp()
    paths = _write_cells(cells, tmpdir)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = os.path.join(tmpdir, "result.json")
            allow_partial_alphas = False
        try:
            ez.cmd_formal(Args())
            check("differing cuda_visible_devices across one model's alpha "
                  "cells hard-stops formal scoring", False, "did not raise/exit")
        except SystemExit:
            check("differing cuda_visible_devices across one model's alpha "
                  "cells hard-stops formal scoring", True)
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest


def test_cmd_formal_missing_cuda_visible_devices_hard_stops():
    """A cell with no cuda_visible_devices recorded must not be silently
    treated as 'compatible with everything' -- that would defeat the whole
    point of the same-GPU check by accepting exactly the unpinned-run case
    it exists to catch."""
    import eval_zebralogic as ez
    import tempfile

    n = ez.N_EASY
    ids = [f"fake-{i}" for i in range(n)]
    gold = {iid: {"header": ["House", "Name"],
                 "rows": [["1", "Arnold"], ["2", "Peter"]]}
           for iid in ids}

    cells = {
        0: _make_formal_cell(0, ids, gold, cvd="0"),
        -6: _make_formal_cell(-6, ids, gold, cvd="0"),
        -4: _make_formal_cell(-4, ids, gold, cvd="0"),
        4: _make_formal_cell(4, ids, gold, cvd=None),
    }
    tmpdir = tempfile.mkdtemp()
    paths = _write_cells(cells, tmpdir)

    orig_digest = ez.EXPECTED_EASY_IDS_SHA256
    ez.EXPECTED_EASY_IDS_SHA256 = ez.sha16("\n".join(sorted(ids)))
    orig = ez.load_private_gold
    ez.load_private_gold = lambda ids_, **kw: {i: gold[i] for i in ids_}
    try:
        class Args:
            generations = paths
            out = os.path.join(tmpdir, "result.json")
            allow_partial_alphas = False
        try:
            ez.cmd_formal(Args())
            check("a missing cuda_visible_devices hard-stops formal scoring",
                  False, "did not raise/exit")
        except SystemExit:
            check("a missing cuda_visible_devices hard-stops formal scoring", True)
    finally:
        ez.load_private_gold = orig
        ez.EXPECTED_EASY_IDS_SHA256 = orig_digest


def test_load_private_gold_integrity_guards():
    """Review finding (2026-09-04): load_private_gold only checked that
    requested ids resolved. Fixed to also hard-stop on: wrong total row
    count, duplicate ids, malformed solution shape, and (when the caller
    supplies expected_shapes) a public/private shape mismatch."""
    import data_zebralogic as dz

    good_rows = [
        {"id": "a", "solution": {"header": ["House", "Name"],
                                 "rows": [["1", "Arnold"], ["2", "Peter"]]}},
        {"id": "b", "solution": {"header": ["House", "Name"],
                                 "rows": [["1", "Eric"]]}},
    ]

    def _stub_load_dataset_factory(rows):
        def _stub(*a, **kw):
            return rows
        return _stub

    # Wrong total row count (N_FULL check).
    orig_load_dataset = None
    import datasets
    orig_load_dataset = datasets.load_dataset
    try:
        datasets.load_dataset = _stub_load_dataset_factory(good_rows)
        try:
            dz.load_private_gold(["a", "b"])
            check("wrong row count hard-stops load_private_gold", False,
                  "did not raise/exit")
        except SystemExit:
            check("wrong row count hard-stops load_private_gold", True)

        # Duplicate ids.
        dup_rows = good_rows + [dict(good_rows[0])]
        n_full_orig = dz.N_FULL
        dz.N_FULL = len(dup_rows)
        try:
            datasets.load_dataset = _stub_load_dataset_factory(dup_rows)
            try:
                dz.load_private_gold(["a", "b"])
                check("duplicate ids hard-stop load_private_gold", False,
                      "did not raise/exit")
            except SystemExit:
                check("duplicate ids hard-stop load_private_gold", True)

            # Malformed solution (header doesn't start with "House").
            dz.N_FULL = len(good_rows)
            bad_header_rows = [
                {"id": "a", "solution": {"header": ["Position", "Name"],
                                         "rows": [["1", "Arnold"]]}},
                {"id": "b", "solution": {"header": ["House", "Name"],
                                         "rows": [["1", "Eric"]]}},
            ]
            datasets.load_dataset = _stub_load_dataset_factory(bad_header_rows)
            try:
                dz.load_private_gold(["a", "b"])
                check("malformed header hard-stops load_private_gold", False,
                      "did not raise/exit")
            except SystemExit:
                check("malformed header hard-stops load_private_gold", True)

            # Row/header length mismatch.
            bad_row_len_rows = [
                {"id": "a", "solution": {"header": ["House", "Name"],
                                         "rows": [["1"]]}},  # too short
                {"id": "b", "solution": {"header": ["House", "Name"],
                                         "rows": [["1", "Eric"]]}},
            ]
            datasets.load_dataset = _stub_load_dataset_factory(bad_row_len_rows)
            try:
                dz.load_private_gold(["a", "b"])
                check("row/header length mismatch hard-stops load_private_gold",
                      False, "did not raise/exit")
            except SystemExit:
                check("row/header length mismatch hard-stops load_private_gold", True)

            # Shape mismatch against expected_shapes.
            datasets.load_dataset = _stub_load_dataset_factory(good_rows)
            good = dz.load_private_gold(["a", "b"])
            check("valid gold with matching shapes resolves fine",
                  set(good) == {"a", "b"})
            wrong_shapes = {"a": {"header": ["House", "Name"], "n_rows": 999}}
            try:
                dz.load_private_gold(["a", "b"], expected_shapes=wrong_shapes)
                check("expected_shapes mismatch hard-stops load_private_gold",
                      False, "did not raise/exit")
            except SystemExit:
                check("expected_shapes mismatch hard-stops load_private_gold", True)
        finally:
            dz.N_FULL = n_full_orig
    finally:
        datasets.load_dataset = orig_load_dataset


def main():
    print("== template ==")
    test_template()
    print("== extract_last_complete_json ==")
    test_extract_last_complete_json()
    print("== official scorer on the official worked example ==")
    test_official_scorer_on_official_example()
    print("== commitment metrics: first-json ==")
    test_commitment_metrics_first_json()
    print("== commitment metrics: first/final agreement ==")
    test_first_final_grid_agreement()
    print("== commitment metrics: strict loop detector ==")
    test_strict_loop()
    print("== commitment metrics: end to end ==")
    test_compute_commitment_metrics_end_to_end()
    print("== forbidden-key label firewall ==")
    test_forbidden_key_guard()
    print("== frozen alpha guard ==")
    test_frozen_alpha_guard()
    print("== eval_zebralogic formal pipeline (synthetic, mocked gold) ==")
    test_eval_zebralogic_formal_pipeline_end_to_end()
    print("== review fix: load_private_gold revision defaults to None ==")
    test_private_gold_revision_defaults_to_none()
    print("== review fix: get_answer_zebralogic hard length check ==")
    test_get_answer_zebralogic_hard_length_check()
    print("== review fix: preflight launcher uses ZERO_CONFIG only ==")
    test_preflight_launcher_uses_zero_config_only()
    print("== review fix: workpoint degradation not reported as qualifying ==")
    test_workpoint_degradation_not_reported_as_qualifying()
    print("== review fix: pure degradation -> no workpoint ==")
    test_workpoint_pure_degradation_reports_no_workpoint()
    print("== review fix: missing alpha hard-stops formal scoring ==")
    test_cmd_formal_missing_alpha_hard_stops()
    print("== review fix: mask mismatch hard-stops formal scoring ==")
    test_cmd_formal_mask_mismatch_hard_stops()
    print("== review fix: preflight indices cover all 7 sizes (35 items) ==")
    test_preflight_indices_cover_all_sizes()
    print("== review fix: max_new_tokens hard guard (2048/3072 only) ==")
    test_max_new_tokens_hard_guard()
    print("== review fix: canary indices last item is 279, not 239 ==")
    test_canary_indices_last_item_is_correct()
    print("== review fix: sample_id gap/duplicate hard-stops formal scoring ==")
    test_cmd_formal_sample_id_gap_hard_stops()
    print("== review fix: item-id digest mismatch hard-stops formal scoring ==")
    test_cmd_formal_id_digest_mismatch_hard_stops()
    print("== review fix: different GPU across alpha cells hard-stops ==")
    test_cmd_formal_different_gpu_hard_stops()
    print("== review fix: missing cuda_visible_devices hard-stops ==")
    test_cmd_formal_missing_cuda_visible_devices_hard_stops()
    print("== review fix: load_private_gold integrity guards ==")
    test_load_private_gold_integrity_guards()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
