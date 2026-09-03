#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump the P4c payloads that do NOT parse. READ-ONLY. NO GOLD. NO ACCURACY.

Reads generation files only. They carry no gold and no correctness field, so
accuracy is unreachable here, not merely unreported.

PURPOSE: to CLASSIFY a non-parsing payload, not to rescue it. The frozen rule
(p4c-amend-01) is:

    a DECODING/FORMAT ARTIFACT   (a stray EOS token, a marker the model
                                  appended after its answer)
        -> a parser defect; fix the parser.

    the MODEL FAILING THE FROZEN FORMAT  (prose after the payload, an empty
                                  marker line, no answer at all)
        -> a RESULT to report. Never widen the parser: the format-obedience
           difference between cells is part of what P4c measures, and
           rescuing it would systematically hide a steering effect.

It prints FIRST and LAST side by side because a payload that fails under FIRST
and parses under LAST is exactly the shape that tempts a caliber switch. FIRST
stays MAIN (frozen, matching GSM8K/GSM-Hard/BBH production); the difference is
already reported per cell as `nonliteral_rate` and `last_acc`, so it is visible
without changing anything.

    python dump_p4c_nonliteral.py --root components
    python dump_p4c_nonliteral.py --root components --preflight
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cruxeval import extract, as_literal          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="components")
    ap.add_argument("--preflight", action="store_true",
                    help="read the _preflight tree instead of the formal cells")
    ap.add_argument("--max_chars", type=int, default=600,
                    help="raw text printed per sample")
    a = ap.parse_args()

    if a.preflight:
        pat = os.path.join(a.root, "*", "cruxeval", "_preflight", "*",
                           "cruxeval_o_preflight_*.json")
    else:
        pat = os.path.join(a.root, "*", "cruxeval", "mdf_*",
                           "cruxeval_o_[0-9]*.json")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no cells under {pat}")

    total = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        m, rows = d["meta"], d["data"]
        if m.get("contains_labels") or m.get("accuracy_computed") is not False:
            sys.exit(f"{f}: carries labels or accuracy; refusing to read")

        hits = []
        for r in rows:
            p = extract(r["generated"], "first")
            if p is not None and not as_literal(p)[0]:
                hits.append(r)
        n_mark = sum(1 for r in rows
                     if extract(r["generated"], "first") is not None)
        print(f"\n{'=' * 72}")
        print(f"{m['model']}  alpha={m['alpha']}  n={len(rows)}  "
              f"marker={n_mark}  non-literal(FIRST)={len(hits)}")
        total += len(hits)

        for r in hits:
            g = r["generated"]
            pf = extract(g, "first")
            pl = extract(g, "last")
            okf, _ = as_literal(pf)
            okl, _ = as_literal(pl)
            # the classification hint -- a hint, never an automatic decision
            if pf == "":
                kind = "EMPTY marker line (model gave no payload on that line)"
            elif okl and not okf:
                kind = ("FIRST fails, LAST parses -> prose/extra text after the "
                        "first payload = MODEL failed the frozen format")
            else:
                kind = "neither FIRST nor LAST parses"
            print(f"\n  --- sample_id {r['sample_id']}  chars={len(g)}  "
                  f"markers={r['n_markers']}  answer_first={r['answer_first']}")
            print(f"      classify: {kind}")
            print(f"      FIRST {repr(pf)[:90]}  parses={okf}")
            print(f"      LAST  {repr(pl)[:90]}  parses={okl}")
            print(f"      RAW   {repr(g[:a.max_chars])}")

    print(f"\n{'=' * 72}")
    print(f"{total} non-parsing payload(s) under the MAIN (FIRST) caliber.")
    print("Classify each. Fix the parser ONLY for a decoding/format artifact;")
    print("a model that failed the frozen format is a RESULT, not a bug.")
    print("NO gold and NO accuracy were read.")


if __name__ == "__main__":
    main()
