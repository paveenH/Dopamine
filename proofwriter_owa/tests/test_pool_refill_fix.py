#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted regression test for the build_manifest() pool-refill bug found in
review (2026-09-04): the refill loop reassigned the FOR-LOOP VARIABLE `pool`
instead of writing back into `true_pool`/`false_pool`, so a genuine
exact-depth shortfall silently produced a manifest short of N_PER_LABEL rows
for that label while `shortfalls` still claimed the gap was filled.

This constructs a synthetic D3 record set where "True" has FEWER than
N_PER_LABEL items at the exact target depth (forcing the refill path) but
MORE than N_PER_LABEL items at a nearby depth (so a correct fix has enough
material to actually reach N_PER_LABEL after refilling). Asserts the final
manifest has exactly N_PER_LABEL True rows for D3, not the pre-fix short
count.

No network, no GPU, no real ProofWriter data. Run with:
    python3.10 proofwriter_owa/tests/test_pool_refill_fix.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_proofwriter_owa as dpo  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def make_record(dataset, tid, qid, answer, qdep, n_facts=3, n_rules=3):
    return {
        "dataset": dataset, "split": "test",
        "official_theory_id": tid, "official_qid": qid,
        "theory_text": f"theory {tid}", "question_text": f"question {qid}",
        "answer": answer, "qdep": qdep, "proof_text": "",
        "n_facts": n_facts, "n_rules": n_rules,
    }


def build_synthetic_records():
    """D3 target depth is 3. Build:
      - True:  only 10 items at qdep==3 (< N_PER_LABEL=50), but 80 more at
               qdep==2 or qdep==4 (nearby depths) -- enough for the refill
               path to reach 50 if it actually writes back correctly.
      - False: a clean 60 items at qdep==3 (>= N_PER_LABEL, no refill needed)
               -- this label's pool must NOT be affected by True's refill,
               proving the fix is per-label, not a global side effect.
      - Unknown: 60 items (qdep=None, matched by size instead).
    D5's records are a trivial, fully-sufficient pool (60 True / 60 False /
    60 Unknown at qdep==5) so D5 never exercises the refill path -- this
    isolates the assertion to D3's forced shortfall.
    """
    recs = {"D3": [], "D5": []}
    i = 0
    for _ in range(10):
        recs["D3"].append(make_record("D3", f"t{i}", "Q1", "True", 3)); i += 1
    for _ in range(80):
        recs["D3"].append(make_record("D3", f"t{i}", "Q1", "True",
                                      2 if i % 2 == 0 else 4)); i += 1
    for _ in range(60):
        recs["D3"].append(make_record("D3", f"t{i}", "Q1", "False", 3)); i += 1
    for _ in range(60):
        recs["D3"].append(make_record("D3", f"t{i}", "Q1", "Unknown", None)); i += 1

    for _ in range(60):
        recs["D5"].append(make_record("D5", f"t{i}", "Q1", "True", 5)); i += 1
    for _ in range(60):
        recs["D5"].append(make_record("D5", f"t{i}", "Q1", "False", 5)); i += 1
    for _ in range(60):
        recs["D5"].append(make_record("D5", f"t{i}", "Q1", "Unknown", None)); i += 1
    return recs


def main():
    print("== pool-refill fix (build_manifest) ==")
    recs = build_synthetic_records()
    rows, shortfalls = dpo.build_manifest(recs, seed=0)

    d3_true = [r for r in rows if r["dataset"] == "D3" and r["answer"] == "True"]
    d3_false = [r for r in rows if r["dataset"] == "D3" and r["answer"] == "False"]
    d3_unknown = [r for r in rows if r["dataset"] == "D3" and r["answer"] == "Unknown"]
    d5_true = [r for r in rows if r["dataset"] == "D5" and r["answer"] == "True"]

    check("shortfall was recorded for D3/True",
          any("D3/True" in s for s in shortfalls), shortfalls)
    check(f"D3 True rows reach N_PER_LABEL={dpo.N_PER_LABEL} after refill "
          "(pre-fix bug: stuck at 10, the exact-depth-only count)",
          len(d3_true) == dpo.N_PER_LABEL,
          f"got {len(d3_true)}")
    check("D3 False rows are UNAFFECTED by True's refill (per-label isolation)",
          len(d3_false) == dpo.N_PER_LABEL, f"got {len(d3_false)}")
    check("D3 Unknown rows reach N_PER_LABEL (matched against the REFILLED "
          "tf_pool, not the pre-fix under-sized one -- a stale tf_pool would "
          "still often reach 50 here since Unknown has 60 available, so this "
          "check is weaker than the True-count check above, but still worth "
          "asserting)",
          len(d3_unknown) == dpo.N_PER_LABEL, f"got {len(d3_unknown)}")
    check("D5 True rows unaffected (no shortfall triggered there)",
          len(d5_true) == dpo.N_PER_LABEL, f"got {len(d5_true)}")

    total_expected = dpo.N_PER_LABEL * 3 * 2  # 3 labels x 2 datasets
    check(f"total manifest size == {total_expected} (pre-fix: short by "
          "however many True rows the refill failed to add)",
          len(rows) == total_expected, f"got {len(rows)}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
