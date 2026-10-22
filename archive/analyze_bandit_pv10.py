#!/usr/bin/env python3.10
"""PV10 analysis: frozen metric definitions for PV10-B and the PV10-A control.

Every number reported for PV10 has exactly ONE definition here. Three
口径 are pinned because they are the ones that silently diverge:

TOP-TWO SHARE -- two different quantities, always reported as a PAIR:
  * true_top2_share       samples on the two arms with the highest TRUE probs.
                          Uses environment ground truth, so it is a DIAGNOSTIC
                          only -- the model cannot see it and no policy can be
                          credited or blamed for it.
  * empirical_top2_share  samples on the two arms leading by empirical mean
                          AT THE MOMENT OF EACH PULL. This can be inflated by
                          early wrong lock-in (concentrating on two arms you
                          wrongly believe are best scores high), so it is NOT
                          on its own evidence of effective exploration.

TIES AND ENTROPY:
  * empirical-best is the TIE-TOLERANT set (matches
    evaluate_competence_gate.py:115). A bare argmax returns the first-listed
    tied arm and disagrees with the frozen gate.
  * allocation entropy is NORMALISED by log(K) -> [0, 1]. 1.0 = uniform
    allocation, 0.0 = every sample on one arm.
  * final accuracy scores the ONE arm actually committed. A tie in the
    empirical-best set is NOT scored as correct; the model submitted a single
    letter and is graded on that letter.

A/B PREFIX COMPARISON:
  PV10-A removes COMMIT from decision one, so a prefix comparison is NOT a
  state-matched counterfactual: as soon as the two protocols choose
  differently the reward states diverge. Report it as a paired-seed prefix
  comparison describing whether an action-space change is accompanied by
  different allocation from early on. It does NOT identify the causal effect
  of extra budget alone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# ─────────────────────────── loading ────────────────────────────────────────

def load_cell(d: Path) -> list[dict]:
    """Episodes from a PV10-B or PV10-A cell. Fails closed on an unknown key."""
    files = sorted(d.glob("bandit_pv10*_*.json"))
    if not files:
        raise SystemExit(f"no bandit_pv10*.json under {d}")
    eps = []
    for f in files:
        data = json.loads(f.read_text())
        if isinstance(data, list):
            eps.extend(data); continue
        if "runs" not in data:
            raise SystemExit(f"{f} has no 'runs' key; refusing to guess schema")
        eps.extend(data["runs"])
    return eps


# ─────────────────────────── frozen primitives ──────────────────────────────

def empirical_best_set(counts: dict) -> set[str]:
    """TIE-TOLERANT empirical-best set (evaluate_competence_gate.py:115)."""
    rates = {a: (s / t if t else 0.0) for a, (s, t) in _pairs(counts)}
    top = max(rates.values())
    return {a for a, v in rates.items() if math.isclose(v, top, abs_tol=1e-12)}


def _pairs(counts: dict):
    for a, v in counts.items():
        yield a, (v[0], v[1])


def allocation_entropy(counts: dict, k: int) -> float:
    """Shannon entropy of the pull distribution, NORMALISED by log(K).

    1.0 = perfectly uniform, 0.0 = all samples on one arm. Normalisation makes
    it comparable across K and readable as a fraction of maximum spread.
    """
    tr = [v[1] for _, v in _pairs(counts)]
    n = sum(tr)
    if n == 0:
        return float("nan")
    h = -sum((t / n) * math.log(t / n) for t in tr if t > 0)
    return h / math.log(k) if k > 1 else float("nan")


def true_top2_share(ep: dict) -> float:
    """Share of samples on the two arms with the highest TRUE probabilities.

    DIAGNOSTIC ONLY: uses ground truth the model cannot observe.
    """
    probs = ep["arm_true_probs"]
    top2 = {a for a, _ in sorted(probs.items(), key=lambda kv: -kv[1])[:2]}
    tr = {a: v[1] for a, v in ep["final_counts"].items()}
    n = sum(tr.values())
    return sum(t for a, t in tr.items() if _letter(a) in {_letter(x) for x in top2}) / n


def empirical_top2_share(ep: dict) -> float:
    """Share of SAMPLE actions landing on the two empirically-leading arms
    AT THE TIME OF THE PULL.

    Can be inflated by early wrong lock-in; never report alone as exploration
    quality.
    """
    hit = tot = 0
    for r in ep.get("rounds", []):
        if r.get("action") != "SAMPLE" or not r.get("arm"):
            continue
        rates = {a: (v[0] / v[1] if v[1] else 0.0)
                 for a, v in r["pre_counts"].items()}
        top2 = {a for a, _ in sorted(rates.items(), key=lambda kv: -kv[1])[:2]}
        tot += 1
        hit += r["arm"] in top2
    return hit / tot if tot else float("nan")


def _letter(x: str) -> str:
    return x.split()[-1] if " " in x else x


# ─────────────────────────── the frozen report ──────────────────────────────

def episode_metrics(ep: dict, k: int) -> dict:
    fc = ep["final_counts"]
    tr = {a: v[1] for a, v in fc.items()}
    n = sum(tr.values())
    eb = empirical_best_set(fc)
    committed = ep.get("committed_arm")
    return {
        "seed": ep["seed"],
        "termination": ep["termination_reason"],
        "tau": ep.get("tau"),
        # final accuracy: the ONE submitted arm, ties not credited
        "correct": ep.get("commit_correct"),
        "max_arm_share": max(tr.values()) / n,
        "min_trials": min(tr.values()),
        "true_top2_share": true_top2_share(ep),
        "empirical_top2_share": empirical_top2_share(ep),
        "alloc_entropy_norm": allocation_entropy(fc, k),
        "true_best_in_empirical_best": ep["true_best"] in eb,
        "commit_follows_empirical_best": (
            committed in eb if committed is not None else None),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None and not (
        isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def report(cells: dict[str, list[dict]], k: int = 4) -> None:
    print("=" * 78)
    print("PV10 FROZEN REPORT")
    print("  top-two: BOTH true (diagnostic) and empirical (lock-in sensitive)")
    print("  entropy: normalised by log(K) -> 1.0 = uniform")
    print("  accuracy: the single committed arm; ties NOT credited")
    print("=" * 78)
    rows = {nm: [episode_metrics(e, k) for e in eps] for nm, eps in cells.items()}
    names = list(rows)
    fields = [
        ("identification accuracy", "correct"),
        ("max-arm share", "max_arm_share"),
        ("min trials across arms", "min_trials"),
        ("true top-2 share (diag)", "true_top2_share"),
        ("empirical top-2 share", "empirical_top2_share"),
        ("alloc entropy (norm)", "alloc_entropy_norm"),
        ("true best in emp-best", "true_best_in_empirical_best"),
        ("commit follows emp-best", "commit_follows_empirical_best"),
    ]
    print("%-28s" % "", "".join("%12s" % n for n in names))
    for label, key in fields:
        vals = []
        for n in names:
            v = [r[key] for r in rows[n]
                 if not (key in ("correct", "commit_follows_empirical_best")
                         and r[key] is None)]
            vals.append(_mean([float(x) if not isinstance(x, bool) else float(x)
                               for x in v]))
        print("%-28s" % label, "".join("%12.3f" % v for v in vals))
    print()
    print("%-28s" % "n episodes", "".join("%12d" % len(rows[n]) for n in names))


def prefix_compare(a_eps, b_eps, prefixes=(10, 20), k: int = 4) -> None:
    """Paired-seed PREFIX comparison of allocation concentration.

    NOT a state-matched counterfactual: PV10-A withdraws COMMIT from decision
    one, so trajectories diverge as soon as the protocols choose differently.
    This describes whether the action-space change is accompanied by different
    allocation from early on; it does NOT isolate the effect of extra budget.
    """
    print("=" * 78)
    print("PAIRED-SEED PREFIX COMPARISON  (descriptive, NOT causal)")
    print("  PV10-A removes COMMIT from round 1, so states diverge on the")
    print("  first differing choice. This does not identify 'extra budget'.")
    print("=" * 78)
    A = {e["seed"]: e for e in a_eps}
    B = {e["seed"]: e for e in b_eps}
    seeds = sorted(set(A) & set(B))
    print("%-8s %26s %26s" % ("prefix", "PV10-A max-share / ent", "PV10-B max-share / ent"))
    for p in prefixes:
        av, ae, bv, be = [], [], [], []
        for s in seeds:
            for src, mv, me in ((A[s], av, ae), (B[s], bv, be)):
                seq = [r["sampled_arm"] for r in src.get("rounds", [])
                       if r.get("sampled_arm")][:p]
                if not seq:
                    continue
                cnt = {}
                for x in seq:
                    cnt[x] = cnt.get(x, 0) + 1
                n = len(seq)
                mv.append(max(cnt.values()) / n)
                h = -sum((c / n) * math.log(c / n) for c in cnt.values())
                me.append(h / math.log(k))
        print("%-8s %13.3f %12.3f %13.3f %12.3f"
              % (f"<= {p}", _mean(av), _mean(ae), _mean(bv), _mean(be)))
    print(f"\n  paired on {len(seeds)} seeds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True,
                    help="name=path pairs, e.g. a0=/path/pv10_a0")
    ap.add_argument("--prefix_compare", nargs=2, metavar=("A_DIR", "B_DIR"),
                    help="PV10-A dir and PV10-B dir, paired by seed")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cells = {}
    for spec in args.cells:
        name, _, path = spec.partition("=")
        cells[name] = load_cell(Path(path))
    report(cells, k=args.k)

    if args.prefix_compare:
        a, b = args.prefix_compare
        print()
        prefix_compare(load_cell(Path(a)), load_cell(Path(b)), k=args.k)


if __name__ == "__main__":
    main()
