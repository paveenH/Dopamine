#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Beta-calculator ablation on the frozen pv7 states. alpha=0, H1 base.

    H1   choice history + raw counts                    (the shared base)
    C1   H1 + Beta posterior MEAN
    C2   H1 + Beta posterior mean + 90% credible interval

All three carry the CHOICE HISTORY block: the history ablation showed it does
not change decisions but does stabilise the output format (hashtags .683 ->
.496, policy_parsed .837 -> .919), and it keeps the task a continuous
decision rather than a per-round table exercise. H1 is therefore the base,
not H0, and the calculator contrast is measured on top of it.

THE QUESTION
------------
pv7 localised the deficit precisely: an arm at 1 pull / 0 reward is treated as
settled bad news, while an untried arm is treated as uncertain. Three
interventions have now failed to move it -- alpha (.017 at every level),
history (.017 -> .017). The calculator attacks the representation directly:
Beta(1,1) rewrites `0/1 = 0.00` as `mean 0.33, wide interval`, putting it on
the same scale as an untried arm's `0.50`.

    C1 vs H1  the posterior-mean CALCULATOR INTERFACE, as a whole
    C2 vs C1  does an explicit credible interval add anything on top?

WHICH CONTRAST IS CLEAN
-----------------------
`C2 - C1` is the only single-variable comparison in this file: the two
prompts differ by exactly one line per arm, the `90% credible interval` row.

`C1 - H1` is NOT "the effect of Beta smoothing". C1 changes four things at
once -- the point estimate (empirical rate -> posterior mean), the field
names, the layout (one line per arm -> four), and the added calculator note.
Report it as *the total effect of the posterior-mean calculator interface*
and never attribute it to the smoothing alone. Isolating the smoothing would
need a fifth arm holding layout and wording fixed while swapping only the
number, which is not run here.

WHAT LICENSES THE STRONG CLAIM
------------------------------
"Explicit uncertainty drove targeted revisiting" requires ALL of:

  1. C2 > C1 on `policy_targets_one_shot_zero`
  2. the gain is not confined to states where a one-shot-zero arm already
     reaches the posterior maximum (see the split table)
  3. `policy_targets_untried` does not blow up indiscriminately
  4. persistence/churn and grounding do not degrade in step
  5. ideally the gain survives in the `does not reach posterior max` subgroup

With fewer than all five, the honest statement is only "the interval changed
the choices" -- not that it produced targeted, uncertainty-driven revisiting.

THE CONFOUND THIS FILE IS BUILT AROUND
--------------------------------------
Beta smoothing does not only express uncertainty; it MOVES THE POINT
ESTIMATES, and it moves them most for small n. A model that simply follows
the largest posterior mean -- doing no uncertainty reasoning at all -- would
therefore score above the current floor on the primary metric purely from
re-ranking.

Ties matter here and are structural: Beta(1,1) sends every `0/1` arm to
exactly (0+1)/(1+2) and every untried arm to exactly (0+1)/(0+2), so many
states have a tied posterior maximum and a bare argmax would silently pick
the first-listed arm.

The report therefore prints a re-ranking BAND per run -- a lower bound
counting only unique maxima and an upper bound counting tied ones -- computed
from the data, never hardcoded here. A C1/C2 value inside that band is
consistent with re-ranking alone. Read the band before the primary metric.

WHAT IS NOT PROVIDED
--------------------
No UCB score, no Thompson sample, no recommendation, no ranking. The prompt
states that the summaries describe belief and do not recommend a button. That
boundary is what keeps this "help the model represent uncertainty" rather than
"run the bandit algorithm for it" -- the latter cannot answer whether the
model can do it.

SCOPE LIMIT (same as every frozen-state result here)
----------------------------------------------------
These states were sampled from an alpha=0, no-history trajectory, and each
yields ONE next choice with no feedback. A null here means "did not change
the immediate choice in already-locked states", NOT "the scaffold does not
work". A scaffold's most likely action is preventing lock-in from forming,
which only a full episode can measure.

Usage
-----
    python3.10 eval_pv7_calculator.py --dry_run
    python3.10 eval_pv7_calculator.py --out .../pv7_calculator.json
    python3.10 eval_pv7_calculator.py --report .../pv7_calculator.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

import bandit_reference as br
import bandit_pv7 as p7
import bandit_pv7_episode as ep
import eval_pv7_frozen_states as fe
import eval_pv7_stage1_alpha as SA
import eval_pv7_history_ablation as HA


EVAL_VERSION = "pv7-calculator-v1"
CALC_VERSION = "beta11-mean-ci90-v1"
DEFAULT_BANK = Path(__file__).with_name("bandit_pv7_lockin_states.json")
DEFAULT_OUT = Path(__file__).with_name("pv7_calculator.json")
ARMS = ("H1", "C1", "C2")

_CALC_NOTE = ("The posterior summaries describe current belief and "
              "uncertainty; they do not recommend a button.")


# ------------------------------------------------------------------- stats

def _beta_ci(k: int, n: int, mass: float = 0.90) -> tuple[float, float]:
    """Equal-tailed credible interval for Beta(k+1, n-k+1).

    scipy is not guaranteed on the analysis box, so the quantile is found by
    bisection on the regularised incomplete beta, computed by a continued
    fraction. Accuracy is far beyond the two decimals that reach the prompt.
    """
    a, b = k + 1, n - k + 1
    lo_p, hi_p = (1 - mass) / 2, 1 - (1 - mass) / 2
    return _beta_ppf(lo_p, a, b), _beta_ppf(hi_p, a, b)


def _beta_ppf(p: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _betainc(a: float, b: float, x: float, _reflected: bool = False) -> float:
    """Regularised incomplete beta I_x(a, b), by continued fraction.

    The reflection I_x(a,b) = 1 - I_{1-x}(b,a) is applied BEFORE the expansion,
    not after: deciding afterwards lets both branches reflect each other
    forever. The continued fraction converges quickly only for
    x < (a+1)/(a+b+2), which is exactly the reflection condition.
    """
    import math
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # STRICT `>`, and never reflect twice: with a == b the threshold is
    # exactly 0.5, so `>=` lets x and 1-x each send the other back forever.
    # `_reflected` makes the trip one-way regardless of the boundary case.
    if x > (a + 1) / (a + b + 2) and not _reflected:
        return 1.0 - _betainc(b, a, 1 - x, _reflected=True)
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz's algorithm for the continued fraction.
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-14:
            break
    return front * (f - 1.0)


def arm_stats(state: dict) -> dict[str, dict]:
    trials, wins = Counter(), Counter()
    for e in state["history"]:
        trials[e["arm"]] += 1
        wins[e["arm"]] += e["reward"]
    out = {}
    for arm in state["arm_order"]:
        n, k = trials[arm], wins[arm]
        lo, hi = _beta_ci(k, n)
        out[arm] = {"n": n, "k": k,
                    "empirical": (k / n) if n else None,
                    "post_mean": (k + 1) / (n + 2),
                    "ci_lo": lo, "ci_hi": hi}
    return out


def posterior_top_arms(state: dict) -> list[str]:
    """ALL arms tied at the maximum posterior mean.

    Returned as a set rather than one arm because ties are STRUCTURAL here,
    not incidental: Beta(1,1) sends every `0/1` arm to exactly (0+1)/(1+2)
    and every untried arm to exactly (0+1)/(0+2), so a tied posterior maximum
    is common. Collapsing that with a bare argmax silently picks the
    first-listed arm; when this was first measured on the pv7 lock-in bank it
    turned a 0-state count into a 17-state one. The report prints both
    readings per run rather than trusting either alone.
    """
    s = arm_stats(state)
    top = max(v["post_mean"] for v in s.values())
    return [a for a in state["arm_order"] if s[a]["post_mean"] == top]


def posterior_greedy_arm(state: dict) -> str | None:
    """The unique posterior-greedy arm, or None when the maximum is tied."""
    tied = posterior_top_arms(state)
    return tied[0] if len(tied) == 1 else None


def empirical_greedy_arm(state: dict) -> str | None:
    return HA.empirical_best(state)


# ------------------------------------------------------------------ prompts

def _options_block(state: dict, arm: str) -> str:
    """OPTIONS rows for one condition. H1 keeps pv7's own one-line format."""
    s = arm_stats(state)
    lines = ["OPTIONS"]
    for a in state["arm_order"]:
        v = s[a]
        if arm == "H1":
            if v["n"] == 0:
                lines.append(f"- {a}: UNTRIED (unknown)")
            else:
                lines.append(
                    f"- {a}: {v['k']} reward{'' if v['k'] == 1 else 's'} / "
                    f"{v['n']} trial{'' if v['n'] == 1 else 's'}, "
                    f"empirical rate {v['empirical']:.2f}")
            continue
        head = ("UNTRIED" if v["n"] == 0
                else f"observed rewards {v['k']} / {v['n']}")
        lines.append(f"- {a}:")
        lines.append(f"  {head}")
        lines.append(f"  Beta posterior mean {v['post_mean']:.2f}")
        if arm == "C2":
            lines.append(f"  90% credible interval "
                         f"[{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")
    return "\n".join(lines)


def build_prompt(state: dict, env, arm: str) -> str:
    """Stage 1 prompt for one condition. All three end at token 220."""
    if arm not in ARMS:
        raise ValueError(arm)
    hist_pairs = [(e["arm"], e["reward"]) for e in state["history"]]
    base = p7.render_state(state["arm_order"], hist_pairs,
                           state["round_idx"], env,
                           prompt_variant=p7.PROMPT_P1B)
    head = base.split("\n\nOPTIONS\n", 1)[0]
    body = f"{head}\n\n{HA.history_block(state['history'])}\n\n" \
           f"{_options_block(state, arm)}"
    if arm != "H1":
        body += f"\n\n{_CALC_NOTE}"
    prompt = f"{body}\n\n{p7._P1B_INSTRUCTION}\n\n{p7.RATIONALE_ANCHOR}"
    p7._assert_single_trailing_space(prompt, p7.RATIONALE_ANCHOR)
    return prompt


def check_prompts(state: dict, env, prompts: dict[str, str]) -> None:
    """Hard invariants. A violation stops the run rather than being recorded."""
    s = arm_stats(state)
    for arm, p in prompts.items():
        if not p.endswith(p7.RATIONALE_ANCHOR):
            raise AssertionError(f"{state['state_id']} {arm}: bad anchor")
        for word in ("true probability", "best arm", "optimal", "recommend "):
            if word in p.lower() and word != "recommend ":
                raise AssertionError(
                    f"{state['state_id']} {arm}: prompt contains {word!r}")
        # Counts shown must equal the history, in every condition.
        for a in state["arm_order"]:
            v = s[a]
            if v["n"] == 0:
                if "UNTRIED" not in p.split(f"- {a}:")[1][:60]:
                    raise AssertionError(
                        f"{state['state_id']} {arm}: {a} has 0 trials but is "
                        "not shown as UNTRIED")
            elif arm != "H1" and f"observed rewards {v['k']} / {v['n']}" not in p:
                raise AssertionError(
                    f"{state['state_id']} {arm}: {a} counts not shown")
    # The three differ ONLY in the OPTIONS block and the calculator note.
    h1_head = prompts["H1"].split("\n\nOPTIONS\n", 1)[0]
    for arm in ("C1", "C2"):
        if prompts[arm].split("\n\nOPTIONS\n", 1)[0] != h1_head:
            raise AssertionError(
                f"{state['state_id']} {arm}: differs from H1 above OPTIONS")
    if prompts["C1"] == prompts["C2"]:
        raise AssertionError(f"{state['state_id']}: C1 and C2 are identical")
    if "credible interval" in prompts["C1"]:
        raise AssertionError(f"{state['state_id']}: C1 leaked the interval")


# ------------------------------------------------------------------- runner

def run_state(vc, state: dict, env, dry_run: bool) -> dict:
    prompts = {a: build_prompt(state, env, a) for a in ARMS}
    check_prompts(state, env, prompts)

    low1 = SA.low_sample_arms(state, 1)
    untried = SA.untried_arms(state)
    pg_top = posterior_top_arms(state)
    pg = posterior_greedy_arm(state)
    eg = empirical_greedy_arm(state)
    crit_arm = state["diagnostics"].get("critical_arm")

    out = {
        "state_id": state["state_id"],
        "state_type": state["state_type"],
        "state_fingerprint": state["state_fingerprint"],
        "seed": state["seed"],
        "round_idx": state["round_idx"],
        "is_critical": state["tags"]["is_critical_one_shot_zero"],
        "critical_arm": crit_arm,
        "low_sample_arms_n1": low1,
        "untried_arms": untried,
        # The re-ranking baseline. Stored per state so every result can be
        # scored against what posterior-greedy alone predicts. BOTH readings
        # are kept: `_unique` requires a single top arm, `_any_tied` counts a
        # state where a one-shot-zero arm merely reaches the maximum. The
        # truth for a real posterior-greedy policy lies between them, since
        # such a policy must break its own ties somehow.
        "posterior_greedy_arm": pg,
        "posterior_top_arms": pg_top,
        "empirical_greedy_arm": eg,
        "posterior_greedy_is_low_sample": bool(pg and pg in low1),
        "posterior_top_includes_low_sample": any(a in low1 for a in pg_top),
        "posterior_top_tied": len(pg_top) > 1,
        "posterior_flips_ranking": bool(pg and eg and pg != eg),
        "arm_stats": arm_stats(state),
        "prompts": prompts if dry_run else None,
        "cells": {},
    }
    if dry_run:
        return out

    for arm in ARMS:
        prompt = prompts[arm]
        aud_r = ep.audit_pv7_prompt(vc, prompt, "rationale")
        vc.steering_fire_count(reset=True)
        _torch().manual_seed(state["seed"] * 100_003 + state["round_idx"])
        gen = vc.generate(inputs=[prompt],
                          max_new_tokens=ep.RATIONALE_MAX_TOKENS,
                          temperature=0.0)
        fired_r = vc.steering_fire_count(reset=True)
        raw = gen[0] if isinstance(gen, list) else gen
        clean = p7.extract_evidence_policy_block(raw)

        # Stage 2 is the FROZEN S1 prompt: no history, no calculator, never
        # steered. Keeping it constant is what makes the contrast a Stage-1
        # contrast.
        a_prompt = ep.build_action_prompt_s1(
            state["arm_order"],
            [(e["arm"], e["reward"]) for e in state["history"]],
            state["round_idx"], env, clean)
        for leak in ("posterior", "credible", "CHOICE HISTORY"):
            if leak in a_prompt:
                raise AssertionError(f"{leak!r} leaked into Stage 2")
        aud_a = ep.audit_pv7_prompt(vc, a_prompt, "action")
        scores, act = ep.score_candidates_pv7(vc, a_prompt, env, None)
        fired_a = vc.steering_fire_count(reset=True)
        if (fired_r, fired_a) != (0, 0):
            raise SystemExit(
                f"{state['state_id']} {arm}: steering fired "
                f"({fired_r}, {fired_a}) in an alpha=0 ablation.")

        pol = ep._policy_record(clean, act)
        tgt = pol["policy_target"]
        gnd = fe.grounding_flags(clean, SA._snap(state))
        ordered = sorted(scores.values(), reverse=True)
        out["cells"][arm] = {
            "rationale_raw": raw,
            "rationale_clean": clean,
            **pol,
            "action": act,
            "tokens_rationale": aud_r,
            "tokens_action": aud_a,
            # 1 primary
            "policy_targets_one_shot_zero": tgt in low1,
            "low_sample_revisit_choice": act in low1,
            # 2 control channel
            "policy_targets_untried": tgt in untried,
            "chose_untried": act in untried,
            # 3 the re-ranking baseline
            "policy_targets_posterior_greedy": tgt == pg,
            "chose_posterior_greedy": act == pg,
            "policy_targets_empirical_greedy": tgt == eg,
            # 4 persistence
            "policy_targets_last_chosen": bool(
                state["history"] and tgt == state["history"][-1]["arm"]),
            "chose_last_chosen": bool(
                state["history"] and act == state["history"][-1]["arm"]),
            # 5 critical, oracle-selected
            "critical_arm_revisit_choice": bool(
                state["tags"]["is_critical_one_shot_zero"]
                and act == crit_arm),
            # 6 text / quality
            "uncertainty_recognition": bool(
                SA.UNCERTAINTY_RE.search(clean)),
            "mentions_posterior": ("posterior" in clean.lower()
                                   or "interval" in clean.lower()),
            "any_grounding_error_clean": gnd["any_grounding_error"],
            "hashtag_present": bool(SA.HASHTAG_RE.search(raw)),
            "empty_clean": not clean.strip(),
            "margin": (ordered[0] - ordered[1] if len(ordered) > 1
                       else float("nan")),
            "norm_entropy": ep._norm_entropy(list(scores.values())),
            "candidate_scores": dict(scores),
            "n_prompt_tokens": aud_r["n_tokens"],
        }
    return out


def _torch():
    import torch
    return torch


# -------------------------------------------------------------------- report

def _rate(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (sum(bool(x) for x in v) / len(v)) if v else float("nan")


def report(doc: dict) -> None:
    S = doc["states"]
    elig = [s for s in S if s["low_sample_arms_n1"]]
    pg_low = [s for s in elig if s["posterior_greedy_is_low_sample"]]

    print("=" * 76)
    print(f"pv7 BETA-CALCULATOR ABLATION   ({doc['eval_version']})")
    print(f"  {len(S)} frozen states, alpha=0, arms {doc['arms']}")
    print("  H1 base = choice history + counts; Stage 2 unchanged")
    print("=" * 76)
    print(f"eligible (>=1 arm at n=1/reward=0): {len(elig)}/{len(S)}\n")

    pg_any = [s for s in elig if s["posterior_top_includes_low_sample"]]
    n_tied = sum(1 for s in S if s["posterior_top_tied"])
    print("RE-RANKING BASELINE -- read this BEFORE the primary metric")
    print(f"  posterior top arm != empirical top arm (unique max):")
    print(f"    {sum(1 for s in S if s['posterior_flips_ranking'])}/{len(S)}")
    print(f"  states with a TIED posterior top: {n_tied}/{len(S)}")
    print("    Ties are STRUCTURAL: Beta(1,1) sends every 0/1 arm to exactly")
    print("    0.33 and every untried arm to exactly 0.50.")
    print("  a one-shot-zero arm reaches the posterior maximum:")
    print(f"    unique max only : {len(pg_low)}/{len(elig)} = "
          f"{len(pg_low) / len(elig):.1%}   (lower bound)")
    print(f"    incl. tied max  : {len(pg_any)}/{len(elig)} = "
          f"{len(pg_any) / len(elig):.1%}   (upper bound)")
    print("  A posterior-greedy model doing NO uncertainty reasoning would")
    print("  score somewhere in that band, depending on how it breaks ties.")
    print("  A C1/C2 value inside the band is consistent with re-ranking")
    print("  alone. The gap that matters is C2 - C1.\n")

    A = list(doc["arms"])
    print(f"{'metric':40s} " + "  ".join(f"{a:>9s}" for a in A))
    print("-" * 76)
    rows = [
        ("1 policy_targets_one_shot_zero PRIMARY",
         "policy_targets_one_shot_zero", "elig"),
        ("1b low_sample_revisit_choice", "low_sample_revisit_choice", "elig"),
        ("2 policy_targets_untried (control)",
         "policy_targets_untried", None),
        ("3 targets_posterior_greedy (baseline)",
         "policy_targets_posterior_greedy", None),
        ("3c chose_posterior_greedy", "chose_posterior_greedy", None),
        ("3b targets_empirical_greedy", "policy_targets_empirical_greedy",
         None),
        ("4 policy_targets_last_chosen", "policy_targets_last_chosen", None),
        ("4b chose_last_chosen", "chose_last_chosen", None),
        ("5 uncertainty_recognition", "uncertainty_recognition", None),
        ("5b mentions_posterior/interval", "mentions_posterior", None),
        ("6 any_grounding_error", "any_grounding_error_clean", None),
        ("6b policy_parsed", "policy_parsed", None),
        ("6c action_follows_policy", "action_follows_policy", None),
        ("6d hashtag_present", "hashtag_present", None),
        ("6e empty_clean", "empty_clean", None),
    ]
    for label, key, scope in rows:
        src = elig if scope == "elig" else S
        vals = [_rate([s["cells"][a] for s in src], key) for a in A]
        print(f"{label:40s} " + "  ".join(f"{v:9.3f}" for v in vals))
    for key in ("margin", "norm_entropy", "n_prompt_tokens"):
        vals = [st.mean(s["cells"][a][key] for s in S) for a in A]
        print(f"{('7 ' + key):40s} " + "  ".join(f"{v:9.3f}" for v in vals))

    print("\nINFERENCE: paired cluster bootstrap, unit = SEED (n=20)")
    for key, label, scope in [
        ("policy_targets_one_shot_zero", "1 targets_1shot0", "elig"),
        ("low_sample_revisit_choice", "1b revisit_choice", "elig"),
        ("policy_targets_untried", "2 targets_untried", None),
        ("policy_targets_posterior_greedy", "3 targets_post_greedy", None),
        ("any_grounding_error_clean", "6 grounding_err", None),
    ]:
        src = elig if scope == "elig" else S
        for a, b, tag in [("H1", "C1", "C1-H1"), ("C1", "C2", "C2-C1"),
                          ("H1", "C2", "C2-H1")]:
            r = SA.cluster_bootstrap_delta(src, key, a, b)
            flag = "" if (r["lo"] <= 0 <= r["hi"]) else "  *"
            print(f"  {label:24s} {tag:7s} {r['delta']:+7.3f}  "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]{flag}")
    print("  * = excludes 0. Straddling 0 = NOT DETECTED, never 'no effect'.")

    print("\nSPLIT BY WHETHER THE POSTERIOR RE-RANKS  (eligible states)")
    print("  If a C1 gain sits ONLY in the re-ranking group, it is the")
    print("  arithmetic, not the model reading uncertainty. Grouped by the")
    print("  INCLUSIVE reading (a one-shot-zero arm reaches the max, tied or")
    print("  not), which is the larger and therefore more conservative set.")
    for name, sub in [("one-shot-zero reaches post. max", pg_any),
                      ("it does not", [s for s in elig
                                       if not s["posterior_top_includes_low_sample"]])]:
        if not sub:
            continue
        vals = [_rate([s["cells"][a] for s in sub],
                      "policy_targets_one_shot_zero") for a in A]
        print(f"  {name:32s} n={len(sub):3d}  "
              + "  ".join(f"{v:9.3f}" for v in vals))

    crit = [s for s in S if s["is_critical"]]
    print(f"\nCRITICAL SUBSET (n={len(crit)}) -- ORACLE-SELECTED, per-state")
    for s in crit:
        pgm = s["arm_stats"][s["critical_arm"]]["post_mean"]
        print(f"  {s['state_id']:34s} abandoned {s['critical_arm']} "
              f"(posterior mean {pgm:.2f})")
        for a in A:
            c = s["cells"][a]
            hit = "  <-- re-chose" if c["critical_arm_revisit_choice"] else ""
            print(f"    {a}  act={c['action']:9s} "
                  f"tgt={str(c['policy_target']):9s}{hit}")
    n_rev = {a: sum(1 for s in crit
                    if s["cells"][a]["critical_arm_revisit_choice"])
             for a in A}
    print("  re-chose abandoned arm: "
          + "  ".join(f"{a}: {n_rev[a]}/{len(crit)}" for a in A))

    # ---- the five conditions, evaluated rather than left to the reader ----
    print("\n" + "=" * 76)
    print("CLAIM CHECK: 'explicit uncertainty drove TARGETED revisiting'")
    print("=" * 76)
    key = "policy_targets_one_shot_zero"
    d21 = SA.cluster_bootstrap_delta(elig, key, "C1", "C2")
    sub_no = [s for s in elig if not s["posterior_top_includes_low_sample"]]
    d_no = (SA.cluster_bootstrap_delta(sub_no, key, "C1", "C2")
            if sub_no else None)
    unt = SA.cluster_bootstrap_delta(S, "policy_targets_untried", "C1", "C2")
    gnd = SA.cluster_bootstrap_delta(S, "any_grounding_error_clean",
                                     "C1", "C2")
    chn = SA.cluster_bootstrap_delta(S, "chose_last_chosen", "C1", "C2")
    c1_rate = _rate([s["cells"]["C1"] for s in elig], key)
    c2_rate = _rate([s["cells"]["C2"] for s in elig], key)
    lo_band = len(pg_low) / len(elig)
    hi_band = len(pg_any) / len(elig)

    # Untargeted redirection: C2 moved the target away from C1's, and the new
    # target is NEITHER a one-shot-zero arm NOR an untried arm. Computed per
    # state so it can be bootstrapped like any other rate. Without this,
    # "persistence holds" is a label over a check that never ran.
    for s in S:
        c1c, c2c = s["cells"]["C1"], s["cells"]["C2"]
        moved = c1c["policy_target"] != c2c["policy_target"]
        s["cells"]["C1"]["_untargeted_change"] = False
        s["cells"]["C2"]["_untargeted_change"] = bool(
            moved
            and not c2c["policy_targets_one_shot_zero"]
            and not c2c["policy_targets_untried"])
    uch = SA.cluster_bootstrap_delta(S, "_untargeted_change", "C1", "C2")

    checks = [
        ("1 C2 > C1 on the primary metric",
         d21["delta"] > 0 and d21["lo"] > 0,
         f"delta {d21['delta']:+.3f} CI [{d21['lo']:+.3f}, {d21['hi']:+.3f}]"),
        # Same bar as condition 1: a strong claim cannot rest on a subgroup
        # point estimate that a single state could have produced.
        ("2 gain holds off the re-ranking states",
         bool(d_no) and d_no["delta"] > 0 and d_no["lo"] > 0,
         (f"n={len(sub_no)}: {d_no['delta']:+.3f} CI "
          f"[{d_no['lo']:+.3f}, {d_no['hi']:+.3f}]" if d_no
          else "subgroup empty")),
        # No ratio test: with a large primary delta, "untried < 2x primary"
        # would pass even as untried approached 1.0. Require simply that no
        # increase is detected.
        ("3 no detected rise in untried targeting",
         not (unt["delta"] > 0 and unt["lo"] > 0),
         f"untried {unt['delta']:+.3f} CI "
         f"[{unt['lo']:+.3f}, {unt['hi']:+.3f}]"),
        ("4 no detected rise in grounding error or",
         not (gnd["delta"] > 0 and gnd["lo"] > 0)
         and not (uch["delta"] > 0 and uch["lo"] > 0),
         f"grounding {gnd['delta']:+.3f}, untargeted "
         f"{uch['delta']:+.3f} CI [{uch['lo']:+.3f}, {uch['hi']:+.3f}]"),
        ("  untargeted target changes", None,
         f"(chose_last {chn['delta']:+.3f}, reported not gated)"),
        ("5 C2 above the re-ranking band",
         c2_rate > hi_band,
         f"C2 {c2_rate:.3f} vs band {lo_band:.3f}-{hi_band:.3f}"),
    ]
    for label, passed, detail in checks:
        mark = "      " if passed is None else \
            f"  [{'PASS' if passed else 'no  '}]"
        print(f"{mark} {label:44s} {detail}")
    gated = [p for _, p, _ in checks if p is not None]
    n_pass = sum(1 for p in gated if p)
    print(f"\n  {n_pass}/{len(gated)} conditions met.")
    if n_pass == 5:
        print("  => 'explicit uncertainty produced targeted, uncertainty-")
        print("     driven revisiting' is supported.")
    elif d21["delta"] > 0 and d21["lo"] > 0:
        print("  => the interval CHANGED the choices, but the targeted-")
        print("     revisiting claim is NOT licensed. Report the weaker one.")
    else:
        print("  => no C2-over-C1 effect detected; the strong claim is not")
        print("     available.")

    print("\n" + "=" * 76)
    print("FROZEN READING KEY (pre-registered)")
    print("=" * 76)
    print("  C2 > C1, all 5 conditions  -> explicit uncertainty is what acts")
    print("  C2 > C1, some conditions   -> the interval changed choices; the")
    print("                                mechanism is not established")
    print(f"  C1 inside {lo_band:.0%}-{hi_band:.0%} band, C2 flat "
          "-> the calculator mainly")
    print("                                strengthens posterior-greedy")
    print("  gain only in the re-rank   -> same, confirmed by the split table")
    print("    subgroup")
    print("  nothing moves              -> the deficit is not representational;")
    print("                                it is the uncertainty-to-action")
    print("                                policy itself")
    print("  everything up, margin down -> untargeted churn")
    print("\n  C1 - H1 is the TOTAL effect of the calculator interface")
    print("  (number, field names, layout, note), NOT of Beta smoothing.")
    print("  C2 - C1 is the only single-variable contrast here.")
    print("\n  Scope: these states came from an alpha=0, no-history")
    print("  trajectory and yield ONE choice with no feedback. A null means")
    print("  'did not change the immediate choice in already-locked states'.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--model_dir",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--history_ablation", type=Path,
                    help="stored H1 result; this run's H1 is asserted to match")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.report:
        report(json.loads(args.report.read_text()))
        return

    bank = json.loads(args.bank.read_text())
    env = br.get_environment("easy")
    states = bank["states"]

    fp = {
        "eval_version": EVAL_VERSION,
        "calculator_version": CALC_VERSION,
        "history_block_version": HA.HISTORY_BLOCK_VERSION,
        "bank_sha256": SA._sha256(args.bank),
        "bank_version": bank["state_bank_version"],
        "arms": list(ARMS),
        "model": args.model, "model_dir": args.model_dir, "size": args.size,
        "stage1_instruction_version": ep.STAGE1_INSTRUCTION_VERSION,
        "stage2_instruction_version": ep.STAGE2_INSTRUCTION_VERSION,
        "policy_parser_version": ep.POLICY_PARSER_VERSION,
        "rationale_max_tokens": ep.RATIONALE_MAX_TOKENS,
    }

    done: dict[str, dict] = {}
    if args.out.exists() and not args.overwrite and not args.dry_run:
        prev = json.loads(args.out.read_text())
        if prev.get("run_fingerprint") == fp:
            done = {s["state_id"]: s for s in prev.get("states", [])}
            print(f"[resume] {len(done)} states stored")
        else:
            raise SystemExit(f"{args.out} holds a different configuration; "
                             "use --overwrite or a new --out")

    vc = None
    if not args.dry_run:
        from llms import VicundaModel
        vc = VicundaModel(model_path=args.model_dir)
        vc.model.eval()
        if not hasattr(vc, "steering_fire_count"):
            raise SystemExit("VicundaModel exposes no steering_fire_count(); "
                             "sync llms.py -- do not bypass.")
        print(f"env={env.name} K={env.k}  arms {ARMS}  alpha=0 all")

    todo = [s for s in states if s["state_id"] not in done]
    print(f"{len(todo)} states to run, {len(done)} resumed")
    results = list(done.values())
    for i, sdef in enumerate(todo, 1):
        t0 = time.time()
        rec = run_state(vc, sdef, env, args.dry_run)
        results.append(rec)
        if args.dry_run:
            continue
        print(f"  [{i}/{len(todo)}] {sdef['state_id']:34s} "
              + " ".join(f"{a}:{rec['cells'][a]['action'][-1]}" for a in ARMS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        n_pg = sum(1 for r in results
                   if r["posterior_greedy_is_low_sample"])
        SA._atomic_write(args.out, json.dumps(
            {"eval_version": EVAL_VERSION, "run_fingerprint": fp,
             "arms": list(ARMS), "alpha": 0.0,
             "n_states": len(results),
             "n_posterior_greedy_is_low_sample": n_pg,
             "analysis_unit": (
                 f"Descriptive rates are state-level (n={len(results)}). "
                 f"Inference resamples SEEDS "
                 f"(n={len({r['seed'] for r in results})}). The critical "
                 "subset is oracle-selected, per-state only. Read the "
                 "re-ranking baseline before the primary metric."),
             "states": results}, indent=1))

    if args.dry_run:
        elig = [r for r in results if r["low_sample_arms_n1"]]
        pgl = [r for r in elig if r["posterior_greedy_is_low_sample"]]
        pga = [r for r in elig if r["posterior_top_includes_low_sample"]]
        flips = sum(1 for r in results if r["posterior_flips_ranking"])
        tied = sum(1 for r in results if r["posterior_top_tied"])
        print(f"\nDRY RUN: {len(results)} states, {len(elig)} eligible")
        print(f"  posterior re-ranks the top arm (unique max): "
              f"{flips}/{len(results)}")
        print(f"  tied posterior top (structural): {tied}/{len(results)}")
        print(f"  RE-RANKING BASELINE BAND -- a one-shot-zero arm reaches the")
        print(f"    posterior max: {len(pgl)}/{len(elig)} "
              f"({len(pgl) / len(elig):.1%}) unique .. {len(pga)}/{len(elig)} "
              f"({len(pga) / len(elig):.1%}) incl. ties")
        ex = next(r for r in results if r["round_idx"] == 50)
        for a in ARMS:
            print(f"\n--- {a} prompt ({ex['state_id']}) ---")
            print(ex["prompts"][a])
        print("--- end ---")
        print(f"\ngenerations if run: {len(results)} x {len(ARMS)} = "
              f"{len(results) * len(ARMS)} Stage 1 + same Stage 2 scorings")
        return

    if args.history_ablation is not None:
        if not args.history_ablation.exists():
            raise SystemExit(f"--history_ablation {args.history_ablation} "
                             "does not exist")
        prev = {s["state_id"]: s for s in
                json.loads(args.history_ablation.read_text())["states"]}
        missing = {r["state_id"] for r in results} - set(prev)
        if missing:
            raise SystemExit(f"{len(missing)} states absent from the history "
                             f"run, e.g. {sorted(missing)[:5]}")
        bad = [r["state_id"] for r in results
               if prev[r["state_id"]]["cells"]["H1"]["rationale_clean"]
               != r["cells"]["H1"]["rationale_clean"]]
        if bad:
            raise SystemExit(
                f"H1 differs from the stored history-ablation H1 on "
                f"{len(bad)}/{len(results)} states, e.g. {bad[:5]}. Same "
                "prompt => device or version drift; do not read C1/C2 "
                "against a base that did not reproduce.")
        print(f"\nH1 vs stored history ablation: {len(results)}/"
              f"{len(results)} identical")

    print(f"\nwrote {args.out}")
    report(json.loads(args.out.read_text()))


if __name__ == "__main__":
    main()
