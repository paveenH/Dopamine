#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixture test for proofwriter_owa/compare_canary.py: builds two small synthetic
canary cell files (mimicking two GPUs) plus a matching gold subset, and drives
compare_canary.main() end-to-end via a real subprocess (so --out actually
gets exercised and its JSON verified), plus a couple of guard checks
(protocol mismatch, mismatched sample_id sets) driven the same way.

No GPU, no network, no real ProofWriter data.

    python3 test_compare_canary.py
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PW_DIR = os.path.dirname(HERE)
SCRIPT = os.path.join(PW_DIR, "compare_canary.py")

FAILS = []
N = 0


def check(cond, msg):
    global N
    N += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


def write_json(path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def make_gold(path, n=6):
    data = []
    for i in range(n):
        ans = ["True", "False", "Unknown"][i % 3]
        data.append({"sample_id": i, "dataset": "D3" if i < n // 2 else "D5",
                    "answer": ans})
    write_json(path, {"meta": {"contains_labels": True}, "data": data})


def make_cell(path, n, alpha, host, cuda, agree=True):
    rows = []
    for i in range(n):
        gold_ans = ["True", "False", "Unknown"][i % 3]
        text = f"reasoning...\nAnswer: {gold_ans}\n" if agree or i != 0 else \
               "reasoning...\nAnswer: Unknown\n"  # force a disagreement at i=0 when agree=False
        rows.append({"sample_id": i, "generated": text, "truncated": False})
    meta = {
        "protocol": "proofwriter-owa-v0", "alpha": alpha, "model": "llama3",
        "provenance": {"host": host, "cuda_visible_devices": cuda},
    }
    write_json(path, {"meta": meta, "data": rows})


def run(args):
    return subprocess.run([sys.executable, SCRIPT] + args,
                          capture_output=True, text=True)


def test_agreeing_cells_report_zero_divergence():
    tmp = tempfile.mkdtemp()
    gold_p = os.path.join(tmp, "gold.json")
    cell0_p = os.path.join(tmp, "cell0.json")
    cell1_p = os.path.join(tmp, "cell1.json")
    out_p = os.path.join(tmp, "out.json")
    make_gold(gold_p, n=6)
    make_cell(cell0_p, n=6, alpha=-6, host="gpu0host", cuda="0", agree=True)
    make_cell(cell1_p, n=6, alpha=-6, host="gpu1host", cuda="1", agree=True)

    r = run(["--gold", gold_p, "--cell", cell0_p, cell1_p, "--out", out_p])
    check(r.returncode == 0, f"expected exit 0, got {r.returncode}: {r.stderr}")
    check(os.path.exists(out_p), "should have written --out")
    out = json.load(open(out_p))
    check(out["diverging_item_ids"] == [],
          f"identical-text cells should show zero disagreement, got "
          f"{out['diverging_item_ids']}")
    check(len(out["summaries"]) == 2, f"expected 2 device summaries, got {len(out['summaries'])}")
    check(out["summaries"][0]["accuracy"] == out["summaries"][1]["accuracy"],
          "identical cells must report identical accuracy")


def test_disagreeing_cell_is_flagged():
    tmp = tempfile.mkdtemp()
    gold_p = os.path.join(tmp, "gold.json")
    cell0_p = os.path.join(tmp, "cell0.json")
    cell1_p = os.path.join(tmp, "cell1.json")
    make_gold(gold_p, n=6)
    make_cell(cell0_p, n=6, alpha=-6, host="gpu0host", cuda="0", agree=True)
    make_cell(cell1_p, n=6, alpha=-6, host="gpu1host", cuda="1", agree=False)

    r = run(["--gold", gold_p, "--cell", cell0_p, cell1_p])
    check(r.returncode == 0, f"expected exit 0 (report, not crash), got {r.returncode}: {r.stderr}")
    check("sample_id=0" in r.stdout or "[0]" in r.stdout,
          f"item 0 should be flagged as diverging in stdout, got:\n{r.stdout}")


def test_mismatched_sample_ids_hard_stops():
    tmp = tempfile.mkdtemp()
    gold_p = os.path.join(tmp, "gold.json")
    cell0_p = os.path.join(tmp, "cell0.json")
    cell1_p = os.path.join(tmp, "cell1.json")
    make_gold(gold_p, n=6)
    make_cell(cell0_p, n=6, alpha=-6, host="a", cuda="0")
    make_cell(cell1_p, n=5, alpha=-6, host="b", cuda="1")   # missing one item

    r = run(["--gold", gold_p, "--cell", cell0_p, cell1_p])
    check(r.returncode != 0, "mismatched sample_id sets must hard-stop, not silently compare a subset")


def test_protocol_mismatch_hard_stops():
    tmp = tempfile.mkdtemp()
    gold_p = os.path.join(tmp, "gold.json")
    cell0_p = os.path.join(tmp, "cell0.json")
    cell1_p = os.path.join(tmp, "cell1.json")
    make_gold(gold_p, n=4)
    make_cell(cell0_p, n=4, alpha=-6, host="a", cuda="0")
    # corrupt cell1's protocol tag
    d = json.load(open(cell0_p))
    d["meta"]["protocol"] = "some-other-protocol"
    write_json(cell1_p, d)

    r = run(["--gold", gold_p, "--cell", cell0_p, cell1_p])
    check(r.returncode != 0, "a cell with a different protocol tag must hard-stop")


def test_single_cell_refused():
    tmp = tempfile.mkdtemp()
    gold_p = os.path.join(tmp, "gold.json")
    cell0_p = os.path.join(tmp, "cell0.json")
    make_gold(gold_p, n=4)
    make_cell(cell0_p, n=4, alpha=-6, host="a", cuda="0")

    r = run(["--gold", gold_p, "--cell", cell0_p])
    check(r.returncode != 0, "a single --cell has nothing to compare against and must be refused")


def test_refuses_to_overwrite_existing_out():
    tmp = tempfile.mkdtemp()
    gold_p = os.path.join(tmp, "gold.json")
    cell0_p = os.path.join(tmp, "cell0.json")
    cell1_p = os.path.join(tmp, "cell1.json")
    out_p = os.path.join(tmp, "out.json")
    make_gold(gold_p, n=4)
    make_cell(cell0_p, n=4, alpha=-6, host="a", cuda="0")
    make_cell(cell1_p, n=4, alpha=-6, host="b", cuda="1")
    write_json(out_p, {"pre_existing": True})

    r = run(["--gold", gold_p, "--cell", cell0_p, cell1_p, "--out", out_p])
    check(r.returncode != 0, "must refuse to overwrite an existing --out file")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{N - len(FAILS)}/{N} checks passed")
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
