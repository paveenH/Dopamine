#!/usr/bin/env python3.10
"""End-to-end: the evaluator must read a file in the DRIVER'S REAL SCHEMA.

This test exists because it did not. The capability manifest froze the
evaluator's hash, but nothing ever checked that the evaluator could read what
run_bandit_pv10_episodes.py actually writes. It could not: the driver stores
episodes under "runs", the evaluator read "episodes" and fell back to [data],
so a 20-episode cell was silently judged as n=1. See
pv10_capability_amendment_01.json.

The schema here is copied from the driver's payload construction, not from the
evaluator's expectations -- that direction is the whole point.
"""
import json, subprocess, sys, tempfile
from pathlib import Path

import evaluate_pv10_capability as cap

fails = []
def check(ok, msg):
    if not ok: fails.append(msg)

def episode(seed, arm, correct, term="autonomous_commit"):
    """One episode in the driver's real record shape."""
    return {
        "seed": seed, "termination_reason": term,
        "committed_arm": arm, "true_best": "A",
        "commit_correct": correct, "tau": 20, "n_at_invalid": None,
        "invalid_kind": None, "model_calls": 17, "steering_fires": 0,
        "display_order": ["A", "B", "C", "D"],
        "final_counts": {"A": [5, 10], "B": [1, 4], "C": [1, 4], "D": [1, 4]},
        "commit_evidence": {"committed_is_empirical_leader": True},
    }

with tempfile.TemporaryDirectory() as td:
    d = Path(td)

    # ---- the driver's real payload shape --------------------------------
    payload = {
        "resume_key": "pv10_...",
        "config": {"alpha": 0.0, "seed_bank": list(range(6))},
        "runs": [episode(i, "ABCD"[i % 4], i % 2 == 0) for i in range(6)],
    }
    (d / "bandit_pv10_pv10_bai_candidate_8B_11_20.json").write_text(
        json.dumps(payload))

    eps = cap.load_cell(d)
    check(len(eps) == 6,
          f"driver-schema file must yield 6 episodes, got {len(eps)}")
    check(all("seed" in e for e in eps),
          "loaded objects must be episodes, not the wrapping payload")
    # the exact regression: payload-as-one-episode
    check(not any("runs" in e for e in eps),
          "REGRESSION: whole payload was wrapped as a pseudo-episode")

    r = cap.evaluate(eps)
    check(r["descriptive"]["n_episodes"] == 6,
          f"n_episodes must be 6, got {r['descriptive']['n_episodes']}")

# ---- fail-closed on an unknown schema -----------------------------------
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    (d / "bandit_pv10_x.json").write_text(json.dumps({"episodes": []}))
    try:
        cap.load_cell(d)
        fails.append("a payload without 'runs' must fail closed, not be guessed")
    except SystemExit:
        pass

# ---- a bare list is still accepted --------------------------------------
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    (d / "bandit_pv10_x.json").write_text(json.dumps([episode(0, "A", True)]))
    check(len(cap.load_cell(d)) == 1, "a bare JSON list must still load")

if fails:
    print("FAIL"); [print("  -", f) for f in fails]; sys.exit(1)
print("test_pv10_gate_end_to_end: all checks passed")
