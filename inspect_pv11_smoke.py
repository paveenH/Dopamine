#!/usr/bin/env python3.10
"""Print PV11 smoke text for human reading. Diagnostic only, never a gate.

The summary line the driver prints (`first=sample arm=A n_samples=5`) is not
enough to judge whether a prompt works. The first smoke run looked clean by
that line -- no invalid, correct terminations, fires=0 -- while the generated
text showed the model asserting "its sample size is larger than the others"
about a block where every arm sits at 20 trials, and trailing into
```python import numpy as np. Neither is visible in a summary.

So this prints the text, plus the three degeneracies that would make a gate
result unreadable rather than merely negative:

  * a CONSTANT first action (nothing for either rule to move)
  * a CONSTANT first arm (label preference confounded with evidence reading)
  * drift markers (code fences, task restatement) that the stop marker missed

Usage:  python3.10 inspect_pv11_smoke.py <dir-or-json> [--rounds N] [--full]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DRIFT = {
    "code_fence": re.compile(r"```"),
    "import_stmt": re.compile(r"\bimport\s+\w+"),
    "def_stmt": re.compile(r"\bdef\s+\w+\("),
    "restates_options": re.compile(r"^\s*-\s*Button\s+[A-E]:", re.M),
    "restates_policy_fmt": re.compile(r"Policy:\s*(SAMPLE|COMMIT)\s+Button\s+X"),
}

# Claims that are checkable against the state's own counts. Only mechanically
# verifiable ones -- whether "close" is a fair description of a .20 gap is a
# judgement and stays out.
LARGER_N = re.compile(
    r"sample size is larger|more (?:trials|samples) than|largest sample",
    re.I)


def load(path: Path) -> list[dict]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    runs = []
    for f in files:
        payload = json.loads(f.read_text())
        runs.extend(payload.get("runs", []))
    if not runs:
        raise SystemExit(f"no episodes found in {path}")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rounds", type=int, default=2,
                    help="rounds of text to print per episode")
    ap.add_argument("--full", action="store_true",
                    help="print raw_generation instead of the stopped text")
    args = ap.parse_args()

    runs = load(Path(args.path))

    for r in runs:
        sec = r["secondary_trajectory"]
        fa = r["first_action"]
        counts = r["opening_counts"]
        trials = {a: c[1] for a, c in counts.items()}
        uniform_n = len(set(trials.values())) == 1
        print("=" * 72)
        print(f"{r['state_uid']}  cell={r['cell']}  H={r['horizon']}  "
              f"probe={r.get('probe_label') or '-'}  "
              f"true_best={r['true_best']}  shown_best={r['displayed_best']}")
        print(f"  counts: " + "  ".join(
            f"{a}={c[0]}/{c[1]}({c[0]/c[1]:.2f})" for a, c in counts.items())
            + ("   [all arms at equal n]" if uniform_n else ""))
        print(f"  FIRST: {fa['kind']} {fa['arm'] or ''}  "
              f"probe={fa.get('is_probe')}  "
              f"shown_best={fa.get('is_displayed_best')}")
        for x in sec["rounds"][:args.rounds]:
            text = x["raw_generation"] if args.full else x["generation_stopped"]
            flags = [k for k, rx in DRIFT.items() if rx.search(text)]
            # A claim of unequal sample size is FALSE whenever every arm has
            # the same trial count -- checkable, unlike "close".
            if uniform_n and LARGER_N.search(text):
                flags.append("FALSE_larger_n")
            mark = ("  <" + ",".join(flags) + ">") if flags else ""
            print(f"  [{x['call_index']}] {x['action']} {x['arm']} "
                  f"reward={x['reward']}{mark}")
            print(f"      {text.strip()[:300]!r}")
        print(f"  -> {sec['termination_reason']}  "
              f"committed={sec['committed_arm']}  "
              f"correct={sec['commit_correct']}  "
              f"n_samples={sec['n_samples']}  "
              f"fires={r['attestation']['steering_fires']}")

    # ── degeneracy summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DEGENERACY CHECK  (a constant readout cannot be moved by anything)")
    print("=" * 72)
    kinds = Counter(r["first_action"]["kind"] for r in runs)
    arms = Counter(r["first_action"]["arm"] for r in runs)
    print(f"  first action : {dict(kinds)}")
    print(f"  first arm    : {dict(arms)}")
    if len(kinds) == 1:
        k = next(iter(kinds))
        print(f"  ! CONSTANT first action ({k}). Both gate rules are"
              f" differences in first-action rates, so a constant readout"
              f" yields a 0.000 difference regardless of the manipulation."
              f" That is an interface failure, not evidence about the model.")
    if len(arms) == 1 and None not in arms:
        print(f"  ! CONSTANT first arm ({next(iter(arms))}). Evidence reading,"
              f" label preference and row preference are collinear here;"
              f" widen the smoke across state_ids before reading anything.")

    n_drift = sum(
        1 for r in runs
        for x in r["secondary_trajectory"]["rounds"][:1]
        if any(rx.search(x["generation_stopped"]) for rx in DRIFT.values()))
    print(f"  drift in first round: {n_drift}/{len(runs)}")

    n_false = sum(
        1 for r in runs
        if len({c[1] for c in r["opening_counts"].values()}) == 1
        for x in r["secondary_trajectory"]["rounds"][:1]
        if LARGER_N.search(x["generation_stopped"]))
    if n_false:
        print(f"  ! {n_false}/{len(runs)} first rounds assert an unequal"
              f" sample size where every arm is at equal n. The Reason text"
              f" is then DESCRIPTIVE only -- do not read it as the model's"
              f" evidence state.")

    # Commitment-specific: the M1 precondition.
    comm = [r for r in runs if r["block"] == "commitment"]
    if comm:
        by_ev = {}
        for r in comm:
            ev = r["cell"].split("/")[0]
            by_ev.setdefault(ev, []).append(
                r["first_action"]["kind"] == "commit")
        print("\n  M1 precondition (first-step COMMIT rate by evidence):")
        for ev, vals in sorted(by_ev.items()):
            print(f"    {ev:7s} {sum(vals)}/{len(vals)}")
        if all(not any(v) for v in by_ev.values()):
            print("    ! never commits at the first step in any cell:"
                  " M1's difference is 0.000 by construction")

    acq = [r for r in runs if r["block"] == "acquisition"]
    if acq:
        by_cell = {}
        for r in acq:
            by_cell.setdefault(r["cell"], []).append(
                bool(r["first_action"].get("is_probe")))
        print("\n  M2 precondition (first-step probe SAMPLE rate by cell):")
        for cell, vals in sorted(by_cell.items()):
            print(f"    {cell:20s} {sum(vals)}/{len(vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
