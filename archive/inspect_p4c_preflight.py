#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4c preflight FORMAT inspection. READ-ONLY. NO GOLD. NO ACCURACY.

Reads only the preflight generation files, which carry no gold and no
correctness field, so accuracy is not merely unreported here -- it is
unreachable. Per the pre-registration, preflight accuracy must not be viewed
and alpha must not be adjusted from it. This script therefore reports FOUR
things and nothing else:

    1. does the '#### <literal>' marker appear
    2. does the payload ast.literal_eval  (the parser the scorer will use)
    3. is the token budget obviously being hit
    4. does steering_fires read the expected L * n * 1

Seeing this output must NOT lead to a changed prompt, a redefined parser, a
re-tuned budget, or a different alpha. Only a HARD STOP (pre-registration
section 7) may stop the run, and a hard stop is a stop -- not a redesign.

    python inspect_p4c_preflight.py --root components
"""
import argparse, ast, glob, json, os, re, sys

# Import the SCORER's parser rather than reimplementing it. A second copy is
# exactly how the inline/offline caliber gap opened on MATH: the inspector
# would report a format the scorer does not actually accept.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cruxeval import extract, as_literal          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="components")
    ap.add_argument("--show", type=int, default=2,
                    help="raw generations to print per cell (format reading)")
    a = ap.parse_args()

    pat = os.path.join(a.root, "*", "cruxeval", "_preflight", "*",
                       "cruxeval_o_preflight_*.json")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no preflight cells under {pat}")

    print(f"{'model':9s} {'a':>3} {'n':>3} {'fires':>6} {'exp':>6} "
          f"{'marker':>7} {'literal':>8} {'capped':>7} {'chars':>7}  verdict")
    problems = []
    cells = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        m, rows = d["meta"], d["data"]
        if not m.get("preflight"):
            sys.exit(f"{f}: not a preflight cell; this script reads only those")
        if m.get("accuracy_computed") is not False or m.get("contains_labels"):
            sys.exit(f"{f}: carries labels or accuracy; refusing to read")

        n = len(rows)
        exp = 0 if m["alpha"] == 0 else m["L"] * n
        texts = [r["generated"] for r in rows]

        payloads = [extract(t) for t in texts]
        n_mark = sum(1 for p in payloads if p is not None)
        n_lit = 0
        bad_payloads = []
        for p in payloads:
            if p is None:
                continue
            if as_literal(p)[0]:
                n_lit += 1
            else:
                bad_payloads.append(p[:60])

        # budget: a CLUSTER at the cap is the signal, never one long sample
        chars = sorted(len(t) for t in texts)
        capped = sum(1 for r in rows if r.get("degenerate_tail"))
        med = chars[n // 2]

        v = []
        # HARD STOPS (pre-registration section 7). Only these stop the run.
        if m.get("steering_fires") != exp:
            v.append("FIRES-MISMATCH"); problems.append(f)
        if n_mark == 0:
            v.append("NO-MARKER"); problems.append(f)
        if n_mark and n_lit == 0:
            v.append("NO-LITERAL"); problems.append(f)
        # ATTENTION, not a hard stop. A low-but-non-zero literal rate is a
        # RESULT to report (the scorer reports it per cell as
        # nonliteral_rate), never a licence to widen the parser. It is
        # surfaced because an earlier version of this script printed
        # "format ok" while 5 of 8 payloads failed to parse -- a verdict that
        # only fired at exactly zero is a verdict that does not work.
        if n_mark and n_lit < n_mark:
            v.append(f"ATTN-{n_mark - n_lit}-NONLITERAL")
        verdict = " ".join(v) if v else "ok"

        print(f"{m['model']:9s} {m['alpha']:>3} {n:>3} "
              f"{m.get('steering_fires'):>6} {exp:>6} "
              f"{n_mark}/{n:<5} {n_lit}/{n:<6} {capped:>7} {med:>7}  {verdict}")
        cells.append((f, m, rows, bad_payloads))

    print("\n  marker  = a '####' appears        literal = its payload ast.literal_eval's")
    print("  capped  = degenerate repeating tail (a CLUSTER, not one long sample, is the signal)")
    print("  chars   = median generated characters")

    for f, m, rows, bad in cells:
        if bad:
            print(f"\n[{m['model']} a={m['alpha']}] payloads that did NOT parse "
                  f"({len(bad)}):")
            for b in bad:
                print(f"    {b!r}")

    if a.show:
        print("\n" + "=" * 72)
        print("RAW GENERATIONS (format reading only -- no gold is available here)")
        for f, m, rows, _ in cells:
            print(f"\n--- {m['model']} alpha={m['alpha']} " + "-" * 40)
            for r in rows[:a.show]:
                g = r["generated"]
                print(f"  [sample_id {r['sample_id']}] chars={len(g)} "
                      f"markers={r['n_markers']} answer_first={r['answer_first']}")
                print("  " + repr(g[:400]))

    n_attn = sum(1 for f, m, rows, bad in cells if bad)
    print("\n" + "=" * 72)
    if n_attn:
        print(f"ATTENTION: {n_attn} cell(s) have payloads that do not parse. "
              "This is NOT a hard stop.")
        print("  Judge each one: a DECODING/FORMAT ARTIFACT (a stray EOS token, "
              "a trailing marker)")
        print("  is a parser defect and must be fixed. Prose after the payload "
              "is the MODEL failing")
        print("  the frozen format -- a result to report, never a parser to "
              "widen.")
    if problems:
        print("HARD-STOP CANDIDATES -- see pre-registration section 7:")
        for p in sorted(set(problems)):
            print(f"  {p}")
        print("A hard stop is a STOP. It is not a licence to redesign the "
              "prompt, redefine the parser, or re-tune the budget.")
        sys.exit(1)
    print("format ok on all preflight cells. NO accuracy was read.")
    print("Proceed to the formal cells; alpha stays as frozen.")


if __name__ == "__main__":
    main()
