#!/usr/bin/env python3.10
"""PV10 minimum capability check. FROZEN BEFORE ANY alpha=0 DATA EXISTS.

This is NOT a competence gate. It does not ask whether the model does BAI well,
whether it beats Uniform/SH/TTTS/LUCB, or whether it stops at a sensible time.
It excludes exactly one thing:

    the task is completely inexecutable on this interface, so an alpha
    difference could not be read as a behavioural change in a BAI context.

Anything short of full degeneration proceeds to the alpha cells. A limited but
identifiable BAI ability is enough; conclusions are then bounded by the model's
actual demonstrated range, and alpha effects are never written up as BAI
capability improvement.

WHAT IS DELIBERATELY *NOT* IN HERE
  * Uniform / SH / TTTS / LUCB comparisons. The offline prescreen positions the
    environment; it never gates the model. P2 already failed there and was
    demoted for exactly this reason -- whether an adaptive ALGORITHM beats equal
    allocation is a different question from whether alpha moves the MODEL.
  * Any stopping-time requirement. Sampling to T_max in every episode is a
    conservative subjective threshold, not an inability to do BAI. Timing is
    reported as DESCRIPTION and gates nothing.
  * Any accuracy floor. Identification accuracy is an outcome the alpha
    contrast is about; making it an entry condition would select the cell on
    the very quantity under study.

THE FOUR CRITERIA (frozen; see FROZEN_CRITERIA for exact thresholds)

  C1 interface validity     -- enough episodes produce parseable decisions
  C2 label non-degeneration -- behaviour is not locked to one label
  C3 row non-degeneration   -- behaviour is not locked to one display row
  C4 evidence sensitivity   -- choices track observed action-reward evidence
                               at all, above what a label/row lock would give

A FAIL is not "the model is bad". It means this INTERFACE did not elicit
executable behaviour, and any prompt change is a new protocol version whose
alpha=0 must be re-run.

Usage:
    python3.10 evaluate_pv10_capability.py --freeze     # write the manifest
    python3.10 evaluate_pv10_capability.py --check      # verify the manifest
    python3.10 evaluate_pv10_capability.py --selftest   # rules vs fixtures
    python3.10 evaluate_pv10_capability.py --result <dir>   # judge a cell
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import bandit_pv10 as p10

MANIFEST_PATH = Path(__file__).with_name("pv10_capability_manifest.json")
PRESCREEN_MANIFEST = Path(__file__).with_name("pv10_prescreen_manifest.json")

CAPABILITY_VERSION = "pv10-capability-v1"

# ─────────────────────────── frozen criteria ────────────────────────────────
#
# Thresholds are deliberately permissive. Each excludes a DEGENERATE regime that
# was actually observed somewhere in the pv1-pv9 history, not a hypothetical:
#
#   C1  pv5 saw 2-6% invalid at K=5 and pv1 Llama saw 32-38% multi-match. A
#       cell where most episodes cannot be parsed has no behaviour to read.
#       0.50 is far below any usable rate and only catches collapse.
#   C2  pv7 measured a Button-A label prior at ~6x the row effect, and pv1
#       Qwen locked the first-listed arm in 47/50 rounds. Requiring merely that
#       the committed label is not ALWAYS the same catches that lock without
#       demanding balance.
#   C3  the pv1 position-leakage bug made OptFrac indistinguishable from a
#       first-option bias. Same shape of test, on display row.
#   C4  pv7's one-shot-zero lock-in produced trajectories that ignored evidence
#       entirely. This asks only that commitment is not independent of the
#       observed rates -- the weakest possible form of "the numbers matter".
FROZEN_CRITERIA = {
    "C1_interface_validity": {
        "statistic": "fraction of episodes whose termination_reason is not "
                     "invalid_policy",
        "threshold": 0.50,
        "direction": ">=",
        "excludes": "most episodes unparseable -- no behaviour to read",
    },
    "C2_label_non_degeneration": {
        "statistic": "fraction of terminating episodes whose committed arm "
                     "carries the MODAL committed label",
        "threshold": 1.00,
        "direction": "<",
        "excludes": "every commit goes to the same label (label lock)",
    },
    "C3_row_non_degeneration": {
        "statistic": "fraction of terminating episodes whose committed arm sits "
                     "in the MODAL display row",
        "threshold": 1.00,
        "direction": "<",
        "excludes": "every commit goes to the same display row (position lock)",
    },
    "C4_evidence_sensitivity": {
        "statistic": "fraction of commits landing on an arm in the empirical-"
                     "best set at commit time (tie-tolerant)",
        "threshold": 0.35,
        "direction": ">",
        "excludes": "commitment independent of observed evidence; 0.35 sits "
                    "just above the 1/K=0.25 chance floor and well below any "
                    "competence claim",
    },
}

CHANCE_FLOOR_K4 = 0.25


# ───────────────────────────── loading ──────────────────────────────────────

def load_cell(result_dir: Path) -> list[dict]:
    """Load PV10 episodes from a cell directory. Read-only.

    Globs pv10_*.json only. The pv6/pv7/pv8/pv9 loaders are deliberately blind
    to each other's filenames so a file cannot be judged by the wrong rules.
    """
    files = sorted(result_dir.glob("bandit_pv10_*.json"))
    if not files:
        raise SystemExit(f"no bandit_pv10_*.json under {result_dir}")
    episodes: list[dict] = []
    for f in files:
        data = json.loads(f.read_text())
        if isinstance(data, list):
            episodes.extend(data)
            continue
        # The driver writes episodes under "runs". An earlier build read
        # "episodes" and fell back to [data], which wrapped the WHOLE payload
        # as a single pseudo-episode: the gate then read n=1 with every
        # criterion degenerate instead of erroring. Fail closed instead.
        if "runs" not in data:
            raise SystemExit(
                f"{f} has no 'runs' key. PV10 cells are written by "
                f"run_bandit_pv10_episodes.py, which stores episodes under "
                f"'runs'; refusing to guess at the schema.")
        episodes.extend(data["runs"])
    return episodes


def _empirical_best_set(counts: dict) -> set[str]:
    """Tie-TOLERANT empirical-best set.

    Tie-tolerant to match evaluate_competence_gate.py:115. A bare argmax
    silently returns the first-listed tied arm, reads low, and disagrees with
    the frozen gate. Beta(1,1)-style structural ties are common early, so this
    is not a rare edge case.
    """
    rates = {a: (s / t if t else 0.0) for a, (s, t) in counts.items()}
    top = max(rates.values())
    return {a for a, v in rates.items() if math.isclose(v, top, abs_tol=1e-12)}


# ───────────────────────────── evaluation ───────────────────────────────────

def evaluate(episodes: list[dict]) -> dict:
    """Apply the four frozen criteria. Read-only: never edits the criteria."""
    n = len(episodes)
    if n == 0:
        raise SystemExit("no episodes to evaluate")

    terminating = [e for e in episodes
                   if e.get("termination_reason") != "invalid_policy"
                   and e.get("committed_arm")]
    n_invalid = sum(1 for e in episodes
                    if e.get("termination_reason") == "invalid_policy")

    # C1
    c1_value = (n - n_invalid) / n

    # C2 / C3 -- degeneration is judged on the COMMITTED choice.
    if terminating:
        labels = [e["committed_arm"] for e in terminating]
        modal_label_frac = max(labels.count(x) for x in set(labels)) / len(labels)
        rows = [e["display_order"].index(e["committed_arm"]) for e in terminating]
        modal_row_frac = max(rows.count(x) for x in set(rows)) / len(rows)
    else:
        modal_label_frac = modal_row_frac = 1.0

    # C4 -- did the commit land on an arm the evidence favoured?
    hits = 0
    for e in terminating:
        counts = {a: tuple(v) for a, v in e["final_counts"].items()}
        if e["committed_arm"] in _empirical_best_set(counts):
            hits += 1
    c4_value = hits / len(terminating) if terminating else 0.0

    results = {
        "C1_interface_validity": (c1_value, c1_value >= 0.50),
        "C2_label_non_degeneration": (modal_label_frac, modal_label_frac < 1.00),
        "C3_row_non_degeneration": (modal_row_frac, modal_row_frac < 1.00),
        "C4_evidence_sensitivity": (c4_value, c4_value > 0.35),
    }

    # ---- DESCRIPTIVE ONLY: gates nothing ----------------------------------
    auton = [e for e in episodes if e.get("autonomous_commit")]
    forced = [e for e in episodes
              if e.get("termination_reason") == "forced_commit"]
    taus = [e["tau"] for e in episodes if e.get("tau") is not None]
    correct = [e["commit_correct"] for e in terminating
               if e.get("commit_correct") is not None]
    invalid_kinds: dict[str, int] = {}
    for e in episodes:
        if e.get("invalid_kind"):
            invalid_kinds[e["invalid_kind"]] = invalid_kinds.get(e["invalid_kind"], 0) + 1

    descriptive = {
        "n_episodes": n,
        "n_invalid": n_invalid,
        "invalid_kinds": invalid_kinds,
        "n_autonomous_commit": len(auton),
        "n_forced_commit": len(forced),
        "autonomous_rate": len(auton) / n,
        "tau_mean": (sum(taus) / len(taus)) if taus else None,
        "tau_min": min(taus) if taus else None,
        "tau_max": max(taus) if taus else None,
        "identification_accuracy": (sum(correct) / len(correct)) if correct else None,
        "note": ("Timing and accuracy are DESCRIPTIVE and gate nothing. All "
                 "episodes reaching T_max is a conservative subjective "
                 "threshold, not an inability to do BAI."),
    }

    verdict = "PASS" if all(ok for _, ok in results.values()) else "FAIL"
    return {
        "capability_version": CAPABILITY_VERSION,
        "criteria": {k: {"value": v, "passes": ok,
                         "threshold": FROZEN_CRITERIA[k]["threshold"],
                         "direction": FROZEN_CRITERIA[k]["direction"]}
                     for k, (v, ok) in results.items()},
        "verdict": verdict,
        "descriptive": descriptive,
    }


def report(res: dict) -> None:
    print()
    print("=" * 78)
    print(f"PV10 MINIMUM CAPABILITY CHECK  ({res['capability_version']})")
    print("=" * 78)
    print("Excludes only a fully inexecutable interface. NOT a competence gate:")
    print("no baseline comparison, no accuracy floor, no stopping requirement.")
    print()
    for name, d in res["criteria"].items():
        mark = "pass" if d["passes"] else "FAIL"
        print(f"  {name:<28} {d['value']:.4f} "
              f"({d['direction']} {d['threshold']})   {mark}")
        print(f"      excludes: {FROZEN_CRITERIA[name]['excludes']}")
    print()
    dsc = res["descriptive"]
    print("-" * 78)
    print("DESCRIPTIVE (gates nothing)")
    print("-" * 78)
    print(f"  episodes                 {dsc['n_episodes']}")
    print(f"  invalid                  {dsc['n_invalid']}  {dsc['invalid_kinds']}")
    print(f"  autonomous / forced      {dsc['n_autonomous_commit']} / "
          f"{dsc['n_forced_commit']}")
    if dsc["tau_mean"] is not None:
        print(f"  tau  mean/min/max        {dsc['tau_mean']:.1f} / "
              f"{dsc['tau_min']} / {dsc['tau_max']}")
    if dsc["identification_accuracy"] is not None:
        print(f"  identification accuracy  {dsc['identification_accuracy']:.4f}")
    print()
    print(f"VERDICT: {res['verdict']}")
    if res["verdict"] == "PASS":
        print("  -> run the alpha cells. Conclusions remain bounded by the")
        print("     model's demonstrated range; alpha effects are never written")
        print("     up as BAI capability improvement.")
    else:
        print("  -> this INTERFACE did not elicit executable behaviour. Any")
        print("     prompt change is a NEW protocol version whose alpha=0 must")
        print("     be re-run. Do not adjust these criteria.")
    print("=" * 78)


# ───────────────────────────── manifest ─────────────────────────────────────

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen_basis() -> dict:
    """Everything the verdict depends on, hashed.

    mtime is recorded but is NOT the freeze evidence: a file copy or a touch
    rewrites it. The hashes are what a result file is checked against.
    """
    from bandit_reference import build_seed_bank, get_environment

    seeds = build_seed_bank(get_environment("easy"), n=20)
    orders = p10.assign_orders(seeds, 4)
    order_payload = {str(s): [list(orders[s].display_order),
                              list(orders[s].initial_pull_order)]
                     for s in seeds}

    me = Path(__file__)
    basis = {
        "capability_version": CAPABILITY_VERSION,
        "criteria": FROZEN_CRITERIA,
        "chance_floor_k4": CHANCE_FLOOR_K4,
        "evaluator_sha256": _sha256_file(me),
        "evaluator_mtime_advisory": me.stat().st_mtime,
        "module_sha256": {
            "bandit_pv10.py": _sha256_file(me.with_name("bandit_pv10.py")),
            "bandit_pv10_episode.py": _sha256_file(
                me.with_name("bandit_pv10_episode.py")),
        },
        "versions": {
            "protocol": p10.PROTOCOL_VERSION,
            "stage1_instruction": p10.STAGE1_INSTRUCTION_VERSION,
            "policy_parser": p10.POLICY_PARSER_VERSION,
            "order": p10.ORDER_VERSION,
            "interface_tag": p10.interface_tag(4, seeds),
        },
        "seed_bank": seeds,
        "seed_bank_sha256": _sha256_obj(seeds),
        "order_bank_sha256": _sha256_obj(order_payload),
        "prescreen_manifest_sha256": (
            _sha256_file(PRESCREEN_MANIFEST) if PRESCREEN_MANIFEST.exists()
            else None),
    }
    return basis


def verify_basis(stored: dict, strict_evaluator: bool = True) -> list[str]:
    """Return the list of mismatched paths. Empty means the basis reproduces.

    `evaluator_sha256` cannot be self-consistent inside its own manifest run
    once the file is edited, so it is compared explicitly and reported like any
    other path -- a changed evaluator MUST invalidate stored verdicts.
    """
    fresh = frozen_basis()
    diffs = []
    for key in sorted(set(stored) | set(fresh)):
        if key == "evaluator_mtime_advisory":
            continue                      # advisory only, never a mismatch
        if stored.get(key) != fresh.get(key):
            diffs.append(key)
    if not strict_evaluator and "evaluator_sha256" in diffs:
        diffs.remove("evaluator_sha256")
    return diffs


# ───────────────────────────── selftest ─────────────────────────────────────

def _fixture(commits, *, invalid=0, display=("A", "B", "C", "D"),
             counts=None) -> list[dict]:
    eps = []
    for i, arm in enumerate(commits):
        eps.append({
            "termination_reason": "autonomous_commit",
            "autonomous_commit": True,
            "committed_arm": arm,
            "commit_correct": 1,
            "tau": 20,
            "display_order": list(display),
            "final_counts": counts or {a: (5, 10) if a == arm else (2, 10)
                                       for a in display},
        })
    for _ in range(invalid):
        eps.append({
            "termination_reason": "invalid_policy",
            "autonomous_commit": False,
            "committed_arm": None, "commit_correct": None, "tau": None,
            "invalid_kind": "no_policy",
            "display_order": list(display),
            "final_counts": {a: (2, 5) for a in display},
        })
    return eps


def selftest() -> None:
    fails = []

    def expect(name, eps, want, which=None):
        r = evaluate(eps)
        got = r["verdict"]
        ok = got == want
        if which is not None:
            ok = ok and not r["criteria"][which]["passes"]
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got}")
        if not ok:
            fails.append(name)

    # A healthy cell: varied commits, evidence-aligned, all parseable.
    expect("healthy cell passes", _fixture(["A", "B", "C", "D", "B"]), "PASS")

    # Degeneration cases, each failing exactly its own criterion.
    expect("all invalid fails C1", _fixture(["A"], invalid=9),
           "FAIL", "C1_interface_validity")
    expect("label lock fails C2", _fixture(["A"] * 8),
           "FAIL", "C2_label_non_degeneration")

    # Row lock: same display row every time, but different labels, so C2 passes
    # and only C3 catches it.
    row_eps = []
    for i, disp in enumerate([("A", "B", "C", "D"), ("B", "C", "D", "A"),
                              ("C", "D", "A", "B"), ("D", "A", "B", "C")] * 2):
        row_eps += _fixture([disp[0]], display=disp)
    r = evaluate(row_eps)
    ok = (not r["criteria"]["C3_row_non_degeneration"]["passes"]
          and r["criteria"]["C2_label_non_degeneration"]["passes"])
    print(f"  {'ok  ' if ok else 'FAIL'} row lock fails C3 but not C2")
    if not ok:
        fails.append("row lock")

    # Evidence-blind: always commit to the WORST arm.
    blind = []
    for _ in range(10):
        blind += _fixture(["D"], counts={"A": (8, 10), "B": (7, 10),
                                         "C": (6, 10), "D": (0, 10)})
    r = evaluate(blind)
    ok = not r["criteria"]["C4_evidence_sensitivity"]["passes"]
    print(f"  {'ok  ' if ok else 'FAIL'} evidence-blind commits fail C4")
    if not ok:
        fails.append("evidence blind")

    # Sampling to T_max in EVERY episode must still pass: a conservative
    # subjective threshold is not an inability to do BAI.
    forced = _fixture(["A", "B", "C", "D"])
    for e in forced:
        e["termination_reason"] = "forced_commit"
        e["autonomous_commit"] = False
        e["tau"] = 100
    r = evaluate(forced)
    ok = r["verdict"] == "PASS"
    print(f"  {'ok  ' if ok else 'FAIL'} all-forced-commit still PASSES "
          f"(timing gates nothing)")
    if not ok:
        fails.append("all forced")

    # Tie tolerance: a commit onto a TIED empirical best must count as a hit.
    tied = _fixture(["B"] * 4, counts={a: (5, 10) for a in "ABCD"})
    r = evaluate(tied)
    ok = r["criteria"]["C4_evidence_sensitivity"]["value"] == 1.0
    print(f"  {'ok  ' if ok else 'FAIL'} tie-tolerant empirical-best set")
    if not ok:
        fails.append("tie tolerance")

    if fails:
        raise SystemExit(f"selftest FAILURES: {fails}")
    print("selftest: capability rules discriminate as intended")


# ───────────────────────────── main ─────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", type=Path, help="cell directory to judge")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.freeze:
        if MANIFEST_PATH.exists():
            raise SystemExit(
                f"{MANIFEST_PATH.name} already exists. Re-freezing after data "
                f"exists is exactly what this file prevents. Delete it "
                f"deliberately if the protocol genuinely changed.")
        MANIFEST_PATH.write_text(json.dumps(frozen_basis(), indent=2,
                                            sort_keys=True))
        print(f"wrote {MANIFEST_PATH}")
        return

    if args.check:
        if not MANIFEST_PATH.exists():
            raise SystemExit(f"no manifest at {MANIFEST_PATH}; run --freeze")
        stored = json.loads(MANIFEST_PATH.read_text())
        diffs = verify_basis(stored)
        if diffs:
            raise SystemExit(
                f"BASIS MISMATCH in {diffs}: the frozen capability basis does "
                f"not reproduce, so no PV10 verdict is citable. If the "
                f"evaluator changed, stored verdicts are void.")
        print(f"BASIS CHECK: OK ({MANIFEST_PATH.name} reproduces)")
        return

    if args.result:
        if MANIFEST_PATH.exists():
            diffs = verify_basis(json.loads(MANIFEST_PATH.read_text()))
            if diffs:
                raise SystemExit(
                    f"BASIS MISMATCH in {diffs}; refusing to judge a cell "
                    f"against an unverified basis")
        else:
            print("WARNING: no capability manifest -- the basis is unverified")
        report(evaluate(load_cell(args.result)))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
