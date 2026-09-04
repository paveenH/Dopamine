#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test for build_subset.py's label-balancing (review finding #1,
2026-09-04): both select_preflight and select_pilot must select a genuine
per-label spread under the REAL manifest sort order
(dataset, label True<False<Unknown, hash), not a positional row[:N] slice
that would grab (almost) all-True under that ordering.

Also checks the label firewall on the blind output written by main().

No network, no GPU, no real ProofWriter data. Run with:
    python3.10 proofwriter_owa/tests/test_build_subset.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_subset as bs  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def make_gold_rows():
    """Mimic the REAL frozen manifest sort order exactly: within one
    dataset, every True row's sample_id is lower than every False row's,
    which is lower than every Unknown row's -- matching
    data_proofwriter_owa.py's build_manifest sort key
    (dataset, label_order["True"|"False"|"Unknown"], hash)."""
    rows = []
    sid = 0
    for ds in ("D3", "D5"):
        for lab in ("True", "False", "Unknown"):
            for _ in range(50):
                rows.append({"sample_id": sid, "dataset": ds, "answer": lab})
                sid += 1
    return rows


def test_select_preflight_is_balanced():
    rows = make_gold_rows()
    ids = bs.select_preflight(rows, per_label=5)
    by_id = {r["sample_id"]: r for r in rows}
    picked = [by_id[i] for i in ids]

    check("preflight selects exactly 30 items", len(ids) == 30, f"got {len(ids)}")
    for ds in ("D3", "D5"):
        for lab in ("True", "False", "Unknown"):
            n = sum(1 for r in picked if r["dataset"] == ds and r["answer"] == lab)
            check(f"preflight {ds}/{lab} count == 5 (pre-fix positional slice "
                  "would give 5/0/0 for D3 under this sort order)",
                  n == 5, f"got {n}")


def test_select_pilot_is_balanced():
    """THE load-bearing check: select_pilot must NOT reproduce the
    rows[:75] bug. Under make_gold_rows()'s ordering (50 True, 50 False,
    50 Unknown per dataset), rows[:75] would give 50 True + 25 False + 0
    Unknown -- exactly the failure mode found in review."""
    rows = make_gold_rows()
    ids = bs.select_pilot(rows, per_label=25)
    by_id = {r["sample_id"]: r for r in rows}
    picked = [by_id[i] for i in ids]

    check("pilot selects exactly 150 items", len(ids) == 150, f"got {len(ids)}")
    for ds in ("D3", "D5"):
        for lab in ("True", "False", "Unknown"):
            n = sum(1 for r in picked if r["dataset"] == ds and r["answer"] == lab)
            check(f"pilot {ds}/{lab} count == 25 "
                  "(pre-fix rows[:75] bug: True=50 False=25 Unknown=0)",
                  n == 25, f"got {n}")

    # Reproduce the exact pre-fix bug on the SAME data to prove it would have
    # been visibly wrong, not just theoretically.
    buggy_ids = []
    for ds in ("D3", "D5"):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        buggy_ids.extend(r["sample_id"] for r in ds_rows[:75])
    buggy_picked = [by_id[i] for i in buggy_ids]
    n_buggy_unknown = sum(1 for r in buggy_picked if r["answer"] == "Unknown")
    n_buggy_true = sum(1 for r in buggy_picked if r["answer"] == "True")
    check("sanity: the OLD rows[:75] approach really would give 0 Unknown "
          "(confirms the bug scenario is not a strawman)",
          n_buggy_unknown == 0, f"got {n_buggy_unknown}")
    check("sanity: the OLD rows[:75] approach really would give 100 True "
          "(50/dataset x 2 datasets) out of 150",
          n_buggy_true == 100, f"got {n_buggy_true}")


def test_shortfall_warning_and_fill():
    """A dataset with too few of one label should print a warning (not
    silently under-select) and fill from leftovers."""
    rows = []
    sid = 0
    for _ in range(2):  # only 2 True (< per_label=5)
        rows.append({"sample_id": sid, "dataset": "D3", "answer": "True"}); sid += 1
    for _ in range(50):
        rows.append({"sample_id": sid, "dataset": "D3", "answer": "False"}); sid += 1
    for _ in range(50):
        rows.append({"sample_id": sid, "dataset": "D3", "answer": "Unknown"}); sid += 1
    for lab in ("True", "False", "Unknown"):
        for _ in range(50):
            rows.append({"sample_id": sid, "dataset": "D5", "answer": lab}); sid += 1

    ids = bs.select_preflight(rows, per_label=5)
    by_id = {r["sample_id"]: r for r in rows}
    picked = [by_id[i] for i in ids]
    d3_picked = [r for r in picked if r["dataset"] == "D3"]
    check("D3 still reaches 15 items total despite True shortfall (filled "
          "from leftovers, not silently short)",
          len(d3_picked) == 15, f"got {len(d3_picked)}")
    n_true = sum(1 for r in d3_picked if r["answer"] == "True")
    check("D3 True count reflects the real shortfall (only 2 available)",
          n_true == 2, f"got {n_true}")


def test_main_label_firewall():
    """End-to-end: main() must emit a blind file with NO 'answer' field
    anywhere, and both output files must have the same sample_id set."""
    rows = make_gold_rows()
    gold_data = [{**r, "extra_gold_field": "x"} for r in rows]
    blind_data = [{"sample_id": r["sample_id"], "dataset": r["dataset"],
                  "some_blind_field": "y"} for r in rows]

    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    blind_path = os.path.join(tmpdir, "blind.json")
    out_gold = os.path.join(tmpdir, "out_gold.json")
    out_blind = os.path.join(tmpdir, "out_blind.json")
    json.dump({"meta": {"contains_labels": True}, "data": gold_data}, open(gold_path, "w"))
    json.dump({"meta": {"contains_labels": False}, "data": blind_data}, open(blind_path, "w"))

    argv_backup = sys.argv
    sys.argv = ["build_subset.py", "--gold", gold_path, "--blind", blind_path,
               "--mode", "pilot", "--out_blind", out_blind, "--out_gold", out_gold]
    try:
        bs.main()
    finally:
        sys.argv = argv_backup

    blind_out = json.load(open(out_blind))
    gold_out = json.load(open(out_gold))
    check("blind output has no 'answer' field on any row",
          all("answer" not in r for r in blind_out["data"]))
    check("blind output has exactly 150 rows", len(blind_out["data"]) == 150,
          f"got {len(blind_out['data'])}")
    check("gold and blind outputs share the same sample_id set",
          {r["sample_id"] for r in gold_out["data"]} ==
          {r["sample_id"] for r in blind_out["data"]})
    check("gold output is genuinely label-balanced (25/label/dataset)",
          sum(1 for r in gold_out["data"]
              if r["dataset"] == "D3" and r["answer"] == "Unknown") == 25)


def main():
    print("== select_preflight balance ==")
    test_select_preflight_is_balanced()
    print("== select_pilot balance (load-bearing) ==")
    test_select_pilot_is_balanced()
    print("== shortfall warning + fill ==")
    test_shortfall_warning_and_fill()
    print("== main() end-to-end label firewall ==")
    test_main_label_firewall()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
