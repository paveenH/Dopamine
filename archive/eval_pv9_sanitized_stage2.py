#!/usr/bin/env python3
"""PV9 Stage-2 SANITIZED RESCORE: is the alpha->margin path epistemic wording?

WHAT THIS TESTS
---------------
PV9 Easy's headline mechanism claim is that alpha reaches Stage 2 only through
the Stage-1 TEXT (Stage 2 is never steered: `action: 0`, verified 20/20). But
`rationale_clean` carries template residue into Stage 2 -- ```python (~43% of
rounds), `Tweet`, repeated "Button X is the best choice" lines. So the stored
margin difference is consistent with two different stories:

    (a) alpha changed the EPISTEMIC WORDING, and that moved the action logits
    (b) alpha changed residue length / repetition, and THAT moved them

This re-scores every stored round with the rationale cut to exactly two lines
-- the Evidence sentence plus the first complete Policy sentence -- and nothing
else. If +4's margin advantage survives, (a) is supported. If it vanishes, the
mediation claim was carried by residue and must be withdrawn.

WHY A RESCORE AND NOT A RE-RUN
------------------------------
Stage 1 is NOT re-generated. The stored rationales are reused verbatim and only
the Stage-2 scoring pass is repeated, so this measures the sanitizer's effect
with the text held fixed. A re-run would confound the two.

WHAT IT CANNOT SHOW
-------------------
Trajectories are NOT re-simulated. Each round is re-scored against the state
that actually occurred, so a changed argmax here does not propagate. This
answers "would Stage 2 have decided differently given a sanitized rationale",
NOT "what would the episode have looked like". A flipped action is a
counterfactual about one round, not an alternative history.

OFFLINE PRE-CHECK (already run, no GPU): residue length is balanced across
cells (mean 31.9 / 30.1 / 24.4 chars, median 9 in all three) and correlates
NEGATIVELY with margin (rho=-0.029), so residue cannot manufacture +4's
advantage. This script is the direct test of the same question.

    python eval_pv9_sanitized_stage2.py --dry_run        # no GPU: sanitizer only
    python eval_pv9_sanitized_stage2.py --pv9_root <dir> # the real rescore
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import bandit_pv7 as p7
import bandit_pv7_episode as p7ep
import bandit_reference as br

CELLS = {"-4": "pv9_easy_bare_am4", "0": "pv9_easy_bare", "+4": "pv9_easy_bare_ap4"}
DETAIL = "bandit_pv9_easy_8B_11_20.json"

# The Policy sentence ends at the first period followed by whitespace or EOS.
# Anchored to the Policy line so a period inside the Evidence half cannot end
# the match early.
_POLICY_SENT = re.compile(r"Policy:[^\n]*?\.(?=\s|$)")


def sanitize_two_line(clean: str) -> str:
    """Evidence + the first complete Policy sentence. Nothing else.

    Everything after the Policy sentence is dropped: ```python, Tweet, hashtag
    spray, repeated "Button X is the best choice" lines. The Evidence half is
    kept verbatim -- it is the epistemic content under test, so editing it
    would defeat the purpose.
    """
    evidence = clean.split("Policy:")[0].strip()
    m = _POLICY_SENT.search(clean)
    if m:
        policy = m.group(0).strip()
    else:
        # No terminal period (truncated mid-sentence): keep the Policy line as
        # far as it goes rather than dropping the stance entirely.
        parts = clean.split("Policy:")
        policy = ("Policy:" + parts[1].split("\n")[0]).rstrip() if len(parts) > 1 else ""
    if not policy:
        return evidence
    return f"{evidence}\n\n{policy}" if evidence else policy


def load_cells(root: Path) -> dict:
    out = {}
    for alpha, sub in CELLS.items():
        path = root / sub / DETAIL
        if not path.exists():
            raise SystemExit(f"missing cell: {path}")
        doc = json.loads(path.read_text())
        if doc.get("protocol") != "pv9":
            raise SystemExit(f"{path} is protocol {doc.get('protocol')!r}, not pv9")
        # Stage 2 must have been unsteered in every stored cell, or the stored
        # margin is not a pure text-mediated quantity to begin with.
        for run in doc["runs"]:
            fires = run.get("steering_fires")
            if fires is None or fires.get("action", -1) != 0:
                raise SystemExit(
                    f"{path} seed {run['seed']}: steering_fires={fires}; this "
                    "analysis assumes Stage 2 was never steered")
        out[alpha] = {r["seed"]: r for r in doc["runs"]}
    return out


def rebuild_state(run, round_idx):
    """The (arm_order, history) the round actually saw."""
    history = list(zip(run["choices"][:round_idx], run["feedbacks"][:round_idx]))
    return run["arm_order"], history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv9_root", default=str(
        Path.home() / "Documents/RSNResult/RoleAnswer/llama3/bandit/pv9"))
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--environment", default="easy")
    ap.add_argument("--out", default="pv9_sanitized_stage2.json")
    ap.add_argument("--dry_run", action="store_true",
                    help="sanitizer + prompt construction only, no model")
    ap.add_argument("--limit_rounds", type=int, default=0,
                    help="debug: score only the first N rounds per episode")
    args = ap.parse_args()

    root = Path(args.pv9_root)
    cells = load_cells(root)
    env = br.get_environment(args.environment)
    seeds = sorted(cells["0"])

    # ---- sanitizer report (no GPU) -------------------------------------
    print("=" * 74)
    print("SANITIZER: rationale_clean -> Evidence + first Policy sentence")
    print("=" * 74)
    for alpha in ["-4", "0", "+4"]:
        o = n = 0
        resid = 0
        for s in seeds:
            for rd in cells[alpha][s]["rounds"]:
                c = rd["rationale_clean"]
                t = sanitize_two_line(c)
                o += len(c)
                n += len(t)
                resid += len(c) - len(t)
        tot = len(seeds) * len(cells[alpha][seeds[0]]["rounds"])
        print(f"  a={alpha:>3}  clean {o/tot:7.1f} -> sanitized {n/tot:7.1f} chars"
              f"   removed {100*resid/o:5.1f}%")

    if args.dry_run:
        ex = cells["0"][seeds[0]]["rounds"][20]
        print("\nEXAMPLE")
        print("  clean    :", repr(ex["rationale_clean"]))
        print("  sanitized:", repr(sanitize_two_line(ex["rationale_clean"])))
        arm_order, history = rebuild_state(cells["0"][seeds[0]], 20)
        prompt = p7ep.build_action_prompt_s1(
            arm_order, history, 20, env, sanitize_two_line(ex["rationale_clean"]))
        print("\nSTAGE-2 PROMPT TAIL:", repr(prompt[-80:]))
        print("(dry run: no model loaded)")
        return 0

    # ---- the rescore ----------------------------------------------------
    from llms import VicundaModel
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    rows = []
    for alpha in ["-4", "0", "+4"]:
        for s in seeds:
            run = cells[alpha][s]
            rounds = run["rounds"]
            if args.limit_rounds:
                rounds = rounds[:args.limit_rounds]
            for rd in rounds:
                i = rd["round"]
                arm_order, history = rebuild_state(run, i)
                clean = sanitize_two_line(rd["rationale_clean"])
                prompt = p7ep.build_action_prompt_s1(
                    arm_order, history, i, env, clean)
                # diff_mtx=None: Stage 2 is unsteered here exactly as it was in
                # the stored run, so any difference is the sanitizer's.
                scores, action = p7ep.score_candidates_pv7(vc, prompt, env, None)
                vals = sorted(scores.values(), reverse=True)
                rows.append({
                    "alpha": alpha, "seed": s, "round": i,
                    "action_old": rd["action"], "action_new": action,
                    "margin_old": rd["margin"],
                    "margin_new": vals[0] - vals[1],
                    "policy_target": rd["policy_target"],
                    "flipped": action != rd["action"],
                })
            print(f"  a={alpha:>3} seed={s} done ({len(rows)} rounds)", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}  ({len(rows)} rows)")

    # ---- verdict --------------------------------------------------------
    from scipy.stats import wilcoxon
    print("\n" + "=" * 74)
    print("SANITIZED Stage-2 margin, per-seed paired vs alpha=0")
    print("=" * 74)
    by = {a: {s: [] for s in seeds} for a in CELLS}
    for r in rows:
        by[r["alpha"]][r["seed"]].append(r["margin_new"])
    v = {a: np.array([np.mean(by[a][s]) for s in seeds]) for a in CELLS}
    for a in ["-4", "+4"]:
        p = wilcoxon(v[a], v["0"])[1]
        print(f"  a={a:>3}  {v[a].mean():.3f} vs {v['0'].mean():.3f}  "
              f"delta={v[a].mean()-v['0'].mean():+.3f}  p={p:.3f}")
    print(f"\n  flipped actions: "
          + "  ".join(f"a={a}: {sum(r['flipped'] for r in rows if r['alpha']==a)}"
                      for a in ["-4", "0", "+4"]))
    print("\nREAD: if +4's advantage survives here, the mediation is epistemic")
    print("wording. If it vanishes, it was residue and must be withdrawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
