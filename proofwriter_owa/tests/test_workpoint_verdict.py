#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test for eval_proofwriter_owa.py's workpoint-verdict logic
(review finding #6, 2026-09-04): a Holm-significant DEGRADATION must never
be reported as a workpoint candidate, and the reported alpha must actually
BE the one that is both Holm-significant AND an improvement -- not an
unrelated argmax computed independently.

Also covers finding #5 (the tautological row-count check) end to end: a
cell with the wrong row count, or duplicate sample_ids, must hard-stop
rather than being silently accepted.

Builds synthetic cell files and drives eval_proofwriter_owa.main() exactly
as the CLI would, then inspects the written --out JSON. No network, no GPU,
no real ProofWriter data.

Run with: python3.10 proofwriter_owa/tests/test_workpoint_verdict.py
"""

import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PW_DIR = os.path.dirname(_HERE)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


N = 60  # small synthetic gold set, 20 per dataset x D3/D5... use 30/30 for simplicity
GOLD_IDS = list(range(N))


def make_gold(path):
    rows = []
    for i in GOLD_IDS:
        ds = "D3" if i < N // 2 else "D5"
        lab = ("True", "False", "Unknown")[i % 3]
        rows.append({"sample_id": i, "dataset": ds, "answer": lab})
    json.dump({"meta": {"contains_labels": True, "manifest_sha256_16": "test",
                        "owa_semantics": "test"}, "data": rows},
              open(path, "w"))


def make_cell(path, model, alpha, layer_start, layer_end, L, correct_ids,
             n_rows=None, duplicate_one=False):
    """correct_ids: sample_ids this cell should answer CORRECTLY (matching
    gold's label exactly); everything else gets a wrong-but-valid answer."""
    rows_ids = GOLD_IDS if n_rows is None else GOLD_IDS[:n_rows]
    gold_lookup = {i: ("True", "False", "Unknown")[i % 3] for i in GOLD_IDS}
    rows = []
    for i in rows_ids:
        gold_lab = gold_lookup[i]
        if i in correct_ids:
            pred_lab = gold_lab
        else:
            pred_lab = next(l for l in ("True", "False", "Unknown") if l != gold_lab)
        rows.append({"sample_id": i, "generated": f"reasoning...\nAnswer: {pred_lab}",
                    "truncated": False, "generated_token_count": 20,
                    "pre_answer_reasoning_tokens": 10})
    if duplicate_one and rows:
        rows.append(dict(rows[0]))  # inject a duplicate sample_id
    fires = 0 if alpha == 0 else L * len(rows)
    # prompt_template_id/marker_family required since load_cell now derives
    # and cross-checks marker_family from prompt_template_id (2026-09-05
    # fix) -- this fixture's rows use v1's "Answer:" marker (see the
    # f-string above), so meta must declare the matching v1 family.
    meta = {"protocol": "proofwriter-owa-v0", "model": model, "alpha": alpha,
           "layer_start": layer_start, "layer_end": layer_end, "L": L,
           "steering_fires": fires, "accuracy_computed": False,
           "prompt_template_id": "proofwriter-owa-cot-v1",
           "marker_family": "v1"}
    json.dump({"meta": meta, "data": rows}, open(path, "w"))


def run_eval(gold_path, cell_paths, out_path, holm_m=3):
    r = subprocess.run(
        [sys.executable, os.path.join(_PW_DIR, "eval_proofwriter_owa.py"),
         "--gold", gold_path, "--generations", *cell_paths,
         "--out", out_path, "--holm_m", str(holm_m)],
        capture_output=True, text=True)
    return r


def test_degradation_not_reported_as_workpoint():
    """alpha=0: 30/60 correct (50%). alpha=-6: 5/60 correct (severe,
    Holm-significant DEGRADATION). alpha=-4: 32/60 (small, non-significant).
    alpha=+4: 55/60 correct (large, Holm-significant IMPROVEMENT).
    Expected: workpoint_alpha == 4, NEVER -6, and the verdict string must
    name alpha=4."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)

    # baseline: correct on the first 30 of 60 (varied, deterministic)
    base_correct = set(range(0, 30))
    degraded_correct = set(range(0, 5))          # much worse
    flat_correct = set(range(0, 32))               # barely different
    improved_correct = set(range(0, 55))            # much better

    p0 = os.path.join(tmpdir, "c0.json")
    pn6 = os.path.join(tmpdir, "cn6.json")
    pn4 = os.path.join(tmpdir, "cn4.json")
    p4 = os.path.join(tmpdir, "c4.json")
    make_cell(p0, "llama3", 0, 11, 20, 9, base_correct)
    make_cell(pn6, "llama3", -6, 11, 20, 9, degraded_correct)
    make_cell(pn4, "llama3", -4, 11, 20, 9, flat_correct)
    make_cell(p4, "llama3", 4, 11, 20, 9, improved_correct)

    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, [p0, pn6, pn4, p4], out_path)
    check("eval_proofwriter_owa.py exits 0 on a well-formed run",
          r.returncode == 0, f"stderr={r.stderr}")

    result = json.load(open(out_path))
    wp = result["results"]["llama3"]["workpoint"]

    check("degradation alpha=-6 IS Holm-significant (sanity: the scenario "
          "actually exercises the bug condition)",
          -6 not in wp["holm_significant_improvement_alphas"] and
          any(v["p_holm_adj"] is not None and v["p_holm_adj"] < 0.05
              for k, v in result["results"]["llama3"]["mcnemar_vs_alpha0"].items()
              if k == "-6"),
          wp)
    check("workpoint_alpha is +4 (the genuine improvement), NEVER -6 "
          "(pre-fix bug: any_holm_pass didn't check direction, so a "
          "significant DEGRADATION at -6 could satisfy it)",
          wp["workpoint_alpha"] == 4, f"got {wp['workpoint_alpha']}")
    check("verdict string names alpha=4",
          "alpha=4" in wp["verdict"], wp["verdict"])
    check("holm_significant_improvement_alphas contains only +4, not -6",
          wp["holm_significant_improvement_alphas"] == [4],
          wp["holm_significant_improvement_alphas"])


def test_pure_degradation_reports_no_workpoint():
    """Every non-zero alpha is WORSE than alpha=0, some Holm-significantly
    so. The verdict must be the frozen NO_WORKPOINT sentence, not any
    alpha -- this is the "even alpha=0 could be reported as a workpoint"
    failure mode from the review, made concrete: if every non-zero alpha
    degrades, nothing should be declared, not alpha=0 and not the least-bad
    degraded alpha."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)

    base_correct = set(range(0, 40))         # high baseline
    degraded_a = set(range(0, 5))              # severe degradation
    degraded_b = set(range(0, 10))             # severe degradation
    degraded_c = set(range(0, 35))             # mild, maybe non-significant

    p0 = os.path.join(tmpdir, "c0.json")
    pn6 = os.path.join(tmpdir, "cn6.json")
    pn4 = os.path.join(tmpdir, "cn4.json")
    p4 = os.path.join(tmpdir, "c4.json")
    make_cell(p0, "llama3", 0, 11, 20, 9, base_correct)
    make_cell(pn6, "llama3", -6, 11, 20, 9, degraded_a)
    make_cell(pn4, "llama3", -4, 11, 20, 9, degraded_b)
    make_cell(p4, "llama3", 4, 11, 20, 9, degraded_c)

    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, [p0, pn6, pn4, p4], out_path)
    check("eval exits 0", r.returncode == 0, r.stderr)

    result = json.load(open(out_path))
    wp = result["results"]["llama3"]["workpoint"]
    check("workpoint_alpha is None when every non-zero dose degrades",
          wp["workpoint_alpha"] is None, wp)
    check("verdict is the frozen NO_WORKPOINT sentence",
          "No effective ProofWriter OWA workpoint was detected" in wp["verdict"],
          wp["verdict"])


def test_row_count_guard():
    """A cell with the wrong row count (not matching len(gold)) must
    hard-stop -- the pre-fix `len(rows) not in (N, len([1 for _ in rows]))`
    was a tautology that could never be false."""
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)

    p0 = os.path.join(tmpdir, "c0.json")
    make_cell(p0, "llama3", 0, 11, 20, 9, set(range(30)), n_rows=45)  # short!

    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, [p0], out_path)
    check("a cell with fewer rows than gold hard-stops (exit != 0)",
          r.returncode != 0, f"returncode={r.returncode}")
    check("error message mentions the row-count mismatch",
          "rows" in r.stderr.lower() and "45" in r.stderr, r.stderr)


def test_duplicate_sample_id_guard():
    tmpdir = tempfile.mkdtemp()
    gold_path = os.path.join(tmpdir, "gold.json")
    make_gold(gold_path)

    p0 = os.path.join(tmpdir, "c0.json")
    # n_rows=N-1 + one duplicate of row 0 -> row count matches N, but
    # sample_id 0 appears twice and some other id is silently missing;
    # duplicate detection must catch this even though the COUNT is right.
    make_cell(p0, "llama3", 0, 11, 20, 9, set(range(30)),
             n_rows=N - 1, duplicate_one=True)

    out_path = os.path.join(tmpdir, "result.json")
    r = run_eval(gold_path, [p0], out_path)
    check("a cell with a duplicate sample_id hard-stops (exit != 0)",
          r.returncode != 0, f"returncode={r.returncode}")
    check("error message mentions duplicate sample_id",
          "duplicate" in r.stderr.lower(), r.stderr)


def main():
    print("== workpoint verdict: degradation must not be reported ==")
    test_degradation_not_reported_as_workpoint()
    print("== workpoint verdict: pure degradation -> no workpoint ==")
    test_pure_degradation_reports_no_workpoint()
    print("== row-count guard (finding #5) ==")
    test_row_count_guard()
    print("== duplicate sample_id guard ==")
    test_duplicate_sample_id_guard()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
