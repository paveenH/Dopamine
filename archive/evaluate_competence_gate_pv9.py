#!/usr/bin/env python3
"""PV9 competence-gate wrapper. LOADER ONLY -- the rules are not re-implemented.

`evaluate_competence_gate.py` was frozen before Track A ran and must not be
edited now that data exists. Its loader hardcodes `bandit_pv6_*.json`, which
PV9 deliberately does not produce: disguising a PV9 file as pv6 would make two
non-poolable protocols silently poolable. So this file supplies a PV9 reader
and calls the SAME frozen `evaluate()`. The four rules, their windows, their
tie handling and their seed-integrity errors are imported, never copied.

TWO ENVIRONMENTS, AND ONLY ONE OF THEM CAN ANCHOR COMPETENCE
------------------------------------------------------------
`easy` is competence-eligible; the frozen banks and algorithmic baselines
remain the correct basis because the environment itself did not change.

`neartie` is NOT (`competence_eligible=False`). With a true gap of 0.05 and
~25 pulls per arm, the empirical-rate standard error (~0.10) is DOUBLE the
gap, so the best arm is statistically unidentifiable for much of the horizon
and a high SuffFail measures the environment rather than the policy. The
rules still RUN on it -- the numbers are useful diagnostics and are directly
comparable to neartie's own Random/Greedy baselines -- but the wrapper prints
a not-a-competence-anchor header and refuses to call the result a verdict.
This mirrors how the pv6 evaluator treats a chat directory.

BASELINE MANIFEST -- REBOUND, NOT INHERITED
-------------------------------------------
`evaluate_competence_gate.MANIFEST` is hardcoded to the pv6 file, which has no
`neartie` block, so calling the frozen `evaluate()` directly fails with
"reference_neartie not in the frozen manifest". This wrapper therefore rebinds
that module constant to `bandit_pv9_baseline_manifest.json` around the call
and restores it in a `finally`.

Rebinding is safe precisely because the two manifests AGREE on `easy`
(asserted by `freeze_pv9_baseline.py --check`, and again here before any
evaluation runs): PV9-Easy and pv8-Easy are judged against one basis, and the
only thing the rebind adds is the neartie block. It is scoped and restored so
a caller importing both wrappers in one process cannot leak the PV9 basis into
a pv6/pv7/pv8 evaluation.

WHAT A PASS MEANS, AND WHAT IT DOES NOT  (frozen wording -- do not soften)
-------------------------------------------------------------------------
A PV9 Easy-bare pass shows Bandit competence UNDER A STRUCTURED,
PARSER-ASSISTED rationale interface with Policy-following constrained action,
a CHOICE HISTORY block, a self-relevant score, AND AN UNTRIED-ARM EXPLORATION
CUE. The cue is a SCAFFOLD: it states a benefit direction, so any exploration
observed under PV9 is scaffolded discovery, never autonomous exploration. All
of those qualifiers travel with the verdict.

PV9, like pv7/pv8, separates two things pv6 could not:
  * DECISION QUALITY      -- what Stage 1's Policy proposed
  * EXECUTION CONSISTENCY -- whether Stage 2 carried it out
Both are reported alongside the verdict; neither is a gate rule.

Usage
-----
    python3.10 evaluate_competence_gate_pv9.py --result .../pv9_easy_bare
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

PV9_MANIFEST = "bandit_pv9_baseline_manifest.json"


class pv9_manifest:
    """Point the frozen evaluator at the PV9 basis, then put it back.

    `gate.MANIFEST` is a module constant read at call time. Mutating it
    permanently would make any later pv6/pv7/pv8 evaluation in the same
    process silently use the PV9 basis, so the swap is scoped.
    """

    def __enter__(self):
        self.original = gate.MANIFEST
        gate.MANIFEST = PV9_MANIFEST
        return self

    def __exit__(self, *exc):
        gate.MANIFEST = self.original
        return False


def assert_easy_basis_agrees() -> None:
    """The rebind is only legitimate while easy is identical across manifests.

    Checked before evaluating anything: if the two ever diverge, PV9-Easy and
    pv8-Easy would be judged against different bases while appearing
    comparable, which is worse than a hard stop.
    """
    if not os.path.exists(PV9_MANIFEST):
        raise SystemExit(f"missing {PV9_MANIFEST}; run freeze_pv9_baseline.py")
    with open(PV9_MANIFEST) as f:
        pv9 = json.load(f)
    if not os.path.exists(gate.MANIFEST):
        return
    with open(gate.MANIFEST) as f:
        pv6 = json.load(f)
    if pv9["environments"]["easy"] != pv6["environments"]["easy"]:
        raise SystemExit(
            "the easy basis differs between the pv6 and PV9 manifests; "
            "PV9-Easy would not be comparable to pv8-Easy")


def load_pv9_dir(path: str):
    """Read PV9 result JSONs. alpha=0 only -- the gate is an alpha=0 statement.

    Rejects a pv6 file rather than reading it: this wrapper exists to keep the
    protocols apart, so quietly accepting one would defeat its purpose.
    """
    files = sorted(glob.glob(os.path.join(path, "**", "bandit_pv9_*.json"),
                             recursive=True))
    if not files:
        stray = glob.glob(os.path.join(path, "**", "bandit_pv6_*.json"),
                          recursive=True)
        raise SystemExit(
            f"no PV9 result JSON under {path}"
            + (f"\n  found {len(stray)} pv6 file(s) instead -- evaluate those "
               "with evaluate_competence_gate.py, not this wrapper." if stray
               else ""))
    out = []
    for f in files:
        d = json.load(open(f))
        if d.get("protocol") != "pv9":
            raise SystemExit(f"{f} declares protocol {d.get('protocol')!r}, "
                             "not 'pv9'")
        ra = float(d.get("rationale_alpha", 0.0))
        aa = float(d.get("action_alpha", 0.0))
        if ra != 0.0 or aa != 0.0:
            continue
        env = br.get_environment(
            {"reference_easy": "easy", "reference_neartie": "neartie", "reference_hard": "hard",
             "native_floor": "native_floor"}[d["environment"]["name"]])
        out.append((d, env, f))
    if not out:
        raise SystemExit(f"no alpha=0 cell under {path} (the gate is an "
                         "alpha=0 statement)")
    return out


def interface_report(runs: list[dict]) -> dict:
    """Decision quality vs execution consistency. DIAGNOSTIC, not a gate rule.

    Reported so a verdict has an address: a PV9 failure can now be attributed
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
                    help="PV9 result dir (repeatable)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.result:
        ap.error("--result is required")

    assert_easy_basis_agrees()

    verdicts = []
    for path in args.result:
        for d, env, f in load_pv9_dir(path):
            runs = d["runs"]
            bank = br.build_seed_bank(env)
            label = f"pv9 {os.path.basename(path)} [{os.path.basename(f)}]"
            if not env.competence_eligible:
                # Run the rules, refuse the verdict. The numbers are a useful
                # diagnostic against neartie's OWN baselines; they are not a
                # competence statement, because at gap .05 the empirical SE is
                # double the gap and SuffFail reflects the environment.
                print("\n" + "!" * 68)
                print(f"!! {env.name} IS NOT A COMPETENCE ANCHOR "
                      "(competence_eligible=False)")
                print("!! Rules below are DIAGNOSTIC ONLY. Do not report a "
                      "PASS/FAIL verdict,")
                print("!! and do not use the words capability-effect / rescue "
                      "/ improvement.")
                print("!" * 68)
            # The FROZEN rules, imported not reimplemented. The manifest is
            # rebound only for the duration of the call.
            with pv9_manifest():
                res = gate.evaluate(runs, env, bank, label)
            gate.report(res)
            res["baseline_manifest"] = PV9_MANIFEST
            res["competence_eligible"] = env.competence_eligible
            res["is_verdict"] = env.competence_eligible

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
            if env.competence_eligible:
                print("\n  A pass means competence under a STRUCTURED, "
                      "PARSER-ASSISTED interface with")
                print("  Policy-following constrained action, a history "
                      "block, a self-relevant score,")
                print("  and a SCAFFOLDED untried-arm cue -- NOT native free "
                      "generation, and NOT")
                print("  autonomous exploration.")
            else:
                print("\n  DIAGNOSTIC ONLY -- see the banner above.")
            res["interface_diagnostic"] = iface
            verdicts.append(res)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(verdicts, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
