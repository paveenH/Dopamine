#!/usr/bin/env python3
"""pv7 competence-gate wrapper. LOADER ONLY -- the rules are not re-implemented.

`evaluate_competence_gate.py` was frozen before Track A ran and must not be
edited now that data exists. But its loader hardcodes `bandit_pv6_*.json` under
`mdf_*/`, which pv7 deliberately does not produce: disguising a pv7 file as pv6
would make two non-poolable protocols silently poolable.

So this file supplies a pv7 reader and calls the SAME frozen `evaluate()`. The
four rules, their windows, their tie handling and their seed-integrity errors
are imported, never copied -- a copy would drift, and a drifted gate is worse
than no gate.

The environment is genuinely unchanged between pv6 and pv7, which is what makes
this legitimate: the frozen seed banks and algorithmic baselines in
`bandit_pv6_baseline_manifest.json` remain the correct comparison basis.

WHAT A PASS MEANS, AND WHAT IT DOES NOT  (frozen wording -- do not soften)
-------------------------------------------------------------------------
A pv7 Easy-bare pass shows Bandit competence UNDER A STRUCTURED,
PARSER-ASSISTED rationale interface with Policy-following constrained action.
It is NOT native free-generation competence: P1b's native termination still
fails (119/120 frozen-state rationales hit the token cap) and the Policy that
Stage 2 reads is recovered by an extractor. Any writeup carries that qualifier.

pv7 additionally separates two things pv6 could not:
  * DECISION QUALITY   -- what Stage 1's Policy proposed
  * EXECUTION CONSISTENCY -- whether Stage 2 carried it out
A gate failure now has an address. Both are reported alongside the verdict,
but neither is a gate rule: the four rules are frozen exactly as they were.

Usage
-----
    python3.10 evaluate_competence_gate_pv7.py --result .../pv7_easy_bare
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

import bandit_reference as br
import evaluate_competence_gate as gate


def load_pv7_dir(path: str):
    """Read pv7 result JSONs. alpha=0 only -- the gate is an alpha=0 statement.

    Rejects a pv6 file rather than reading it: this wrapper exists to keep the
    protocols apart, so quietly accepting one would defeat its purpose.
    """
    files = sorted(glob.glob(os.path.join(path, "**", "bandit_pv7_*.json"),
                             recursive=True))
    if not files:
        stray = glob.glob(os.path.join(path, "**", "bandit_pv6_*.json"),
                          recursive=True)
        raise SystemExit(
            f"no pv7 result JSON under {path}"
            + (f"\n  found {len(stray)} pv6 file(s) instead -- evaluate those "
               "with evaluate_competence_gate.py, not this wrapper." if stray
               else ""))
    out = []
    for f in files:
        d = json.load(open(f))
        if d.get("protocol") != "pv7":
            raise SystemExit(f"{f} declares protocol {d.get('protocol')!r}, "
                             "not 'pv7'")
        ra = float(d.get("rationale_alpha", 0.0))
        aa = float(d.get("action_alpha", 0.0))
        if ra != 0.0 or aa != 0.0:
            continue
        env = br.get_environment(
            {"reference_easy": "easy", "reference_hard": "hard",
             "native_floor": "native_floor"}[d["environment"]["name"]])
        out.append((d, env, f))
    if not out:
        raise SystemExit(f"no alpha=0 cell under {path} (the gate is an "
                         "alpha=0 statement)")
    return out


def interface_report(runs: list[dict]) -> dict:
    """Decision quality vs execution consistency. DIAGNOSTIC, not a gate rule.

    Reported so a verdict has an address: a pv7 failure can now be attributed
    to Stage 1 proposing badly or Stage 2 failing to execute. Adding a
    threshold here would be exactly the post-hoc freedom the frozen gate
    exists to prevent, so there is none.
    """
    parse = [r["policy_parse_rate"] for r in runs if "policy_parse_rate" in r]
    follow = [r["action_follows_policy_rate"] for r in runs
              if "action_follows_policy_rate" in r
              and r["action_follows_policy_rate"] == r["action_follows_policy_rate"]]
    fires = [r.get("steering_fires") for r in runs]
    # A missing counter FAILS this check. Filtering None out would make an
    # unattested cell -- the one case where steering cannot be verified at all
    # -- report True vacuously, which is backwards for a fail-closed check.
    unsteered = bool(runs) and all(f == {"rationale": 0, "action": 0}
                                   for f in fires)
    return {
        "policy_parse_rate": st.mean(parse) if parse else float("nan"),
        "action_follows_policy_rate": st.mean(follow) if follow else float("nan"),
        "n_runs_with_parse": len(parse),
        "all_unsteered": unsteered,
        "n_runs_without_fire_count": sum(1 for f in fires if f is None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="append", default=[],
                    help="pv7 result dir (repeatable)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.result:
        ap.error("--result is required")

    verdicts = []
    for path in args.result:
        for d, env, f in load_pv7_dir(path):
            runs = d["runs"]
            bank = br.build_seed_bank(env)
            label = f"pv7 {os.path.basename(path)} [{os.path.basename(f)}]"
            # The FROZEN rules, imported not reimplemented.
            res = gate.evaluate(runs, env, bank, label)
            gate.report(res)

            iface = interface_report(runs)
            print("\nINTERFACE DIAGNOSTIC (not a gate rule)")
            print(f"  policy parse rate        {iface['policy_parse_rate']:.3f}")
            print(f"  action-follows-policy    "
                  f"{iface['action_follows_policy_rate']:.3f}")
            if iface["n_runs_without_fire_count"]:
                print(f"  WARNING: {iface['n_runs_without_fire_count']} run(s) "
                      "carry NO steering_fires -- the injection is unattested")
            elif not iface["all_unsteered"]:
                print("  WARNING: a run carries non-zero steering_fires in an "
                      "alpha=0 gate cell")
            print("\n  A pass means competence under a STRUCTURED, "
                  "PARSER-ASSISTED interface")
            print("  with Policy-following constrained action -- NOT native "
                  "free generation.")
            res["interface_diagnostic"] = iface
            verdicts.append(res)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(verdicts, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
