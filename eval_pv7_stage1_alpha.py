#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Stage-1 alpha diagnostic on frozen pv7 lock-in states.

    (rationale_alpha, action_alpha) in {(-4, 0), (0, 0), (+4, 0)}

Stage 2 SCORES but is never steered. It is the executor (action_follows_policy
= .989 on the alpha=0 cell), so steering it too would mix "alpha changed the
policy" with "alpha changed the execution of that policy" into one number.
A (0, +4) executor-robustness ablation is a separate, later run.

WHAT IS AND IS NOT SHARED ACROSS THE THREE ALPHA CELLS
------------------------------------------------------
Byte-identical across cells, asserted at runtime:
  * the frozen state (history, arm_order, round_idx)
  * the Stage 1 prompt
Necessarily DIFFERENT across cells:
  * the Stage 2 prompt -- it embeds the rationale that this alpha produced.
    Requiring it to match would require alpha to have had no effect.

QUESTION SCOPE: IMMEDIATE REVISIT, NOT LOCK-BREAKING
-----------------------------------------------------
A frozen state yields ONE next choice. That can show whether alpha makes the
model re-sample a low-sample arm right now; it cannot show whether a long-run
lock is broken, which is a trajectory property. The metric names say so:

    low_sample_revisit_choice     chose ANY arm at n=1 with reward 0   PRIMARY
    critical_arm_revisit_choice   critical state: chose the abandoned arm
    oracle_best_revisit_choice    chose the true best arm     SECONDARY/oracle

`oracle_best_revisit_choice` reads `diagnostics.best_arm` and is reported
last, separately, and never as the headline: an arm's true identity is
invisible to the model, so making it primary would grade the policy on
information it does not have.

METRIC ORDER (mechanism first, outcome last)
--------------------------------------------
  1. uncertainty_recognition        does the text name low-sample uncertainty
  2. uncertainty_action_alignment   does the Policy then TARGET such an arm
  3. low_sample_revisit_choice      does Stage 2 actually choose one
  4. critical subset                per-state, n=5, proportions only
  5. grounding / hashtag / format   did quality degrade
  6. margin, entropy, oracle-best   outcome-flavoured, last

Layers 1-2 are Stage-1 text; layer 3 is the executed action. The alpha=0 cell
of the full episodes showed .996 recognition vs .031 alignment, so the gap
between layers 1 and 2 is the mechanism under test.

ACCEPTANCE (fail-closed, per state x alpha)
-------------------------------------------
  alpha=0   steering_fires {rationale: 0,       action: 0}
  alpha!=0  steering_fires {rationale: n_layers, action: 0}
Both stages' prompts must end at token 220, candidates must be 32-35 (bare
letters), and no double BOS. A violation raises rather than being recorded:
by the time a wrong injection site is read off a stored record, every
generation in the cell was steered at the wrong token.

Per state x alpha the fire counts are per-ROUND numbers -- one Stage 1 pass
(n_layers sites) and one Stage 2 pass of K rows (n_layers*K sites) -- not the
episode-level n_layers*horizon.

Usage
-----
    python3.10 eval_pv7_stage1_alpha.py --dry_run        # prompts, no GPU
    python3.10 eval_pv7_stage1_alpha.py --out ...json
    python3.10 eval_pv7_stage1_alpha.py --report ...json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

import bandit_reference as br
import bandit_pv7 as p7
import bandit_pv7_episode as ep
import eval_pv7_frozen_states as fe


EVAL_VERSION = "pv7-stage1-alpha-v1"
DEFAULT_BANK = Path(__file__).with_name("bandit_pv7_lockin_states.json")
DEFAULT_OUT = Path(__file__).with_name("pv7_stage1_alpha.json")

# Set once from --alphas before anything reads it. A module constant rather
# than a threaded parameter because eleven call sites read it, including
# report(); a missed site would silently mix alpha sets within one table.
#
# SPLITTING ACROSS GPUs: run `--alphas -4 0` and `--alphas 0 4` in separate
# processes with SEPARATE --out files, then merge with --merge. The alpha=0
# cell MUST appear in both: it is the paired baseline every contrast is
# computed against, and it is deterministic (temperature 0, same seed, no
# hook registered), so the two runs produce the same alpha=0 result. --merge
# verifies that rather than assuming it.
ALPHAS = (-4.0, 0.0, 4.0)

# Text-level uncertainty recognition. Deliberately broad: layer 1 asks whether
# the model NAMES low-sample uncertainty at all, and a narrow list would turn a
# recall question into a phrasing question. Layer 2 (alignment) is what carries
# the mechanism claim, and it is regex-free -- it reads the parsed target.
UNCERTAINTY_RE = re.compile(
    r"uncertain|unknown|untried|not (?:yet )?(?:been )?tried|weak|"
    r"few trials?|1 trial|one trial|single trial|low sample|small sample|"
    r"no (?:trials?|data|information|evidence)|limited",
    re.I)
HASHTAG_RE = re.compile(r"(?:^|\s)#\S+")


def low_sample_arms(state: dict, max_pulls: int = 1) -> list[str]:
    """Arms with <= max_pulls observations, all of them 0, in this state.

    max_pulls=1 is the strict reading used for the headline (`n=1, reward=0`,
    exactly the seed-3/26/31/46/50 signature). max_pulls=2 is reported beside
    it because a lock-in trajectory keeps the abandoned arm at n=1 for the rest
    of the episode, so the two nearly coincide -- if they diverge, the choice
    of threshold is doing work and must be stated.
    """
    trials, wins = Counter(), Counter()
    for e in state["history"]:
        trials[e["arm"]] += 1
        wins[e["arm"]] += e["reward"]
    return sorted(a for a in state["arm_order"]
                  if 0 < trials[a] <= max_pulls and wins[a] == 0)


def untried_arms(state: dict) -> list[str]:
    tried = {e["arm"] for e in state["history"]}
    return [a for a in state["arm_order"] if a not in tried]


def _snap(state: dict) -> dict:
    """Adapter for eval_pv7_frozen_states.grounding_flags.

    Its `_history` reads {"arm", "reward"} dicts -- the bank's own shape -- so
    the history passes through unchanged. Converting to pairs here (as the
    prompt builders need) would raise inside grounding_flags.
    """
    return {"arm_order": state["arm_order"],
            "history": state["history"],
            "round_idx": state["round_idx"]}


def run_state(vc, state: dict, env, diff_by_alpha: dict, n_layers: int,
              dry_run: bool) -> dict:
    """One frozen state under all three alphas. Stage 1 prompt shared."""
    r_prompt = p7.build_rationale_prompt(
        state["arm_order"], [(e["arm"], e["reward"]) for e in state["history"]],
        state["round_idx"], env, prompt_variant=p7.PROMPT_P1B)

    low1 = low_sample_arms(state, 1)
    low2 = low_sample_arms(state, 2)
    untried = untried_arms(state)
    best = state["diagnostics"]["best_arm"]        # oracle: secondary only
    crit = state["tags"]["is_critical_one_shot_zero"]
    # The abandoned arm, read from the bank. NEVER low1[0]: alphabetical order
    # carries no information about which arm was abandoned, and on this bank
    # that shortcut mislabels seeds 3 and 26.
    crit_arm = state["diagnostics"].get("critical_arm")
    if crit and not crit_arm:
        raise SystemExit(f"{state['state_id']} is flagged critical but the "
                         "bank stores no critical_arm; rebuild the bank")

    out = {
        "state_id": state["state_id"],
        "state_type": state["state_type"],
        "state_fingerprint": state["state_fingerprint"],
        "seed": state["seed"],
        "round_idx": state["round_idx"],
        "is_critical": crit,
        "critical_arm": crit_arm,
        "low_sample_arms_n1": low1,
        "low_sample_arms_n2": low2,
        "untried_arms": untried,
        "rationale_prompt": r_prompt if dry_run else None,
        "cells": {},
    }
    if dry_run:
        out["cells"] = {str(a): {"rationale_prompt_shared": True}
                        for a in ALPHAS}
        aud = None
        if vc is not None:
            aud = ep.audit_pv7_prompt(vc, r_prompt, "rationale")
        out["tokens_rationale"] = aud
        return out

    for alpha in ALPHAS:
        diff = diff_by_alpha[alpha]
        steer = diff is not None
        aud_r = ep.audit_pv7_prompt(vc, r_prompt, "rationale")

        vc.steering_fire_count(reset=True)
        # Deterministic and identical across cells: same seed, same prompt, so
        # any difference in the generation is attributable to the injection.
        _torch().manual_seed(state["seed"] * 100_003 + state["round_idx"])
        if steer:
            gen = vc.regenerate(inputs=[r_prompt], diff_matrices=diff,
                                prefill_only=True, prefill_tail_len=1,
                                max_new_tokens=ep.RATIONALE_MAX_TOKENS,
                                temperature=0.0)
        else:
            gen = vc.generate(inputs=[r_prompt],
                              max_new_tokens=ep.RATIONALE_MAX_TOKENS,
                              temperature=0.0)
        fired_r = vc.steering_fire_count(reset=True)
        raw = gen[0] if isinstance(gen, list) else gen
        clean = p7.extract_evidence_policy_block(raw)

        # Stage 2: scored, NEVER steered (diff_matrices=None by construction).
        a_prompt = ep.build_action_prompt_s1(
            state["arm_order"],
            [(e["arm"], e["reward"]) for e in state["history"]],
            state["round_idx"], env, clean)
        aud_a = ep.audit_pv7_prompt(vc, a_prompt, "action")
        scores, arm = ep.score_candidates_pv7(vc, a_prompt, env, None)
        fired_a = vc.steering_fire_count(reset=True)

        exp = {"rationale": n_layers if steer else 0, "action": 0}
        got = {"rationale": fired_r, "action": fired_a}
        if got != exp:
            raise SystemExit(
                f"{state['state_id']} alpha={alpha}: steering_fires {got} != "
                f"{exp}. Stage 1 must fire n_layers={n_layers} sites when "
                "steered and Stage 2 must never fire; the intervention is not "
                "what the config claims.")

        pol = ep._policy_record(clean, arm)
        fmt = p7.rationale_format_flags(raw, clean)
        gnd_clean = fe.grounding_flags(clean, _snap(state))
        gnd_raw = fe.grounding_flags(raw, _snap(state))
        ordered = sorted(scores.values(), reverse=True)
        tgt = pol["policy_target"]

        out["cells"][str(alpha)] = {
            "rationale_raw": raw,
            "rationale_clean": clean,
            **pol,
            "action": arm,
            "steering_fires": got,
            "tokens_rationale": aud_r,
            "tokens_action": aud_a,
            # 1. text recognises low-sample uncertainty
            "uncertainty_recognition": bool(UNCERTAINTY_RE.search(clean)),
            # 2. the Policy TARGETS a low-sample arm (regex-free: reads the
            #    parsed target). NOTE this does NOT require the text to have
            #    recognised uncertainty first -- hence the plain name. The
            #    recognition->action TRANSITION is the conjunction below, and
            #    its conditional form P(targets | recognition) is computed in
            #    the report from these two fields.
            "policy_targets_low_sample_n1": tgt in low1,
            "policy_targets_low_sample_n2": tgt in low2,
            "recognition_and_targets_low_sample": bool(
                UNCERTAINTY_RE.search(clean)) and tgt in low1,
            "policy_targets_untried": tgt in untried,
            # 3. the executed action
            "low_sample_revisit_choice_n1": arm in low1,
            "low_sample_revisit_choice_n2": arm in low2,
            "chose_untried": arm in untried,
            # Oracle-assisted secondary diagnostic: the subset itself was
            # selected using true-best identity.
            "critical_arm_revisit_choice": bool(crit and arm == crit_arm),
            # 5. quality
            "any_grounding_error_clean": gnd_clean["any_grounding_error"],
            "any_grounding_error_raw": gnd_raw["any_grounding_error"],
            "support_overclaim_clean": gnd_clean.get("support_overclaim"),
            "hashtag_present": bool(HASHTAG_RE.search(raw)),
            "n_hashtags": len(HASHTAG_RE.findall(raw)),
            "format_flags": fmt,
            "empty_clean": not clean.strip(),
            # 6. outcome-flavoured, reported last
            "candidate_scores": dict(scores),
            "margin": (ordered[0] - ordered[1] if len(ordered) > 1
                       else float("nan")),
            "norm_entropy": ep._norm_entropy(list(scores.values())),
            "oracle_best_revisit_choice": arm == best,
            "oracle_policy_targets_best": tgt == best,
        }

    # Stage 1 prompt is shared by construction; assert the invariant that
    # matters downstream -- the three cells differ only by the injection.
    heads = {c["tokens_rationale"]["n_tokens"] for c in out["cells"].values()}
    if len(heads) != 1:
        raise SystemExit(f"{state['state_id']}: Stage 1 prompt differs across "
                         "alpha cells")
    return out


def _torch():
    import torch
    return torch


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_masks(base_dir, hs, type_, mask_type, percentage, size, ls, le):
    """One scaled diff matrix per alpha. None at alpha=0 registers no hook."""
    path = (Path(base_dir) / "mask" / f"{hs}_{type_}_logits" /
            f"{mask_type}_{percentage}_{ls}_{le}_{size}.npy")
    raw = np.load(path)
    n_layers = int(sum(1 for row in raw if np.any(row != 0)))
    return ({a: (None if a == 0 else list(raw * a)) for a in ALPHAS},
            n_layers, _sha256(path))


def run_fingerprint(args, bank: dict, bank_path: Path,
                    mask_sha: str | None) -> dict:
    """Everything a stored result depends on. Resume compares ALL of it.

    Version strings alone are not enough: a bank rebuilt from a different
    source, a re-generated mask, a different model_dir or layer band all change
    the result while leaving every version string identical. Content hashes
    make that impossible to miss.
    """
    return {
        "eval_version": EVAL_VERSION,
        "bank_version": bank["state_bank_version"],
        "bank_sha256": _sha256(bank_path),
        "mask_sha256": mask_sha,
        "alphas": list(ALPHAS),
        "action_alpha": 0.0,
        "model": args.model,
        "model_dir": args.model_dir,
        "size": args.size,
        "hs": args.hs,
        "type": args.type,
        "mask_type": args.mask_type,
        "percentage": args.percentage,
        "layers": args.layers,
        "stage1_instruction_version": ep.STAGE1_INSTRUCTION_VERSION,
        "stage2_instruction_version": ep.STAGE2_INSTRUCTION_VERSION,
        "policy_parser_version": ep.POLICY_PARSER_VERSION,
        "rationale_max_tokens": ep.RATIONALE_MAX_TOKENS,
    }


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so an interrupt cannot truncate a JSON.

    A plain write_text() interrupted mid-flush leaves a partial file, and the
    next resume reads it as the authoritative checkpoint.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ---------------------------------------------------------------- reporting

def _rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(bool(v) for v in vals) / len(vals)) if vals else float("nan")


def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
            and r[key] == r[key]]
    return (sum(vals) / len(vals)) if vals else float("nan")


def cluster_bootstrap_delta(states, key, alpha_a, alpha_b, n_boot=10_000,
                            seed=12345):
    """Paired cluster bootstrap of rate(alpha_b) - rate(alpha_a).

    THE RESAMPLING UNIT IS THE SEED, NOT THE STATE. The 123 states come from
    20 trajectories, so treating them as 123 independent observations would
    understate the interval: states from one seed share a history, a reward
    tape and an arm order. Clusters are resampled with replacement and every
    state of a drawn seed comes along.

    Paired: both alpha cells of the same state are carried together, so the
    per-seed delta removes between-seed variance.

    n=20 clusters is small. This yields a bootstrap interval, not a p-value,
    and an interval that straddles 0 means NOT DETECTED -- never "no effect".
    """
    by_seed: dict[int, list[tuple[bool, bool]]] = {}
    for s in states:
        ca, cb = s["cells"].get(alpha_a), s["cells"].get(alpha_b)
        if not ca or not cb or ca.get(key) is None or cb.get(key) is None:
            continue
        by_seed.setdefault(s["seed"], []).append(
            (bool(ca[key]), bool(cb[key])))
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return {"delta": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n_clusters": len(seeds), "n_states": 0}

    def rate_delta(sample):
        a = [x for s in sample for x, _ in by_seed[s]]
        b = [y for s in sample for _, y in by_seed[s]]
        return sum(b) / len(b) - sum(a) / len(a)

    point = rate_delta(seeds)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = list(rng.choice(seeds, size=len(seeds), replace=True))
        boots.append(rate_delta(draw))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": point, "lo": float(lo), "hi": float(hi),
            "n_clusters": len(seeds),
            "n_states": sum(len(v) for v in by_seed.values())}


def report(doc: dict) -> None:
    states = doc["states"]
    A = [str(a) for a in ALPHAS]

    print("=" * 74)
    print(f"pv7 STAGE-1 ALPHA DIAGNOSTIC  ({doc['eval_version']})")
    print(f"  states {len(states)}  alphas {A}  action_alpha=0 (never steered)")
    print("=" * 74)
    print("Frozen states yield ONE next choice: these measure IMMEDIATE "
          "revisit,\nnot lock-breaking. Lock-breaking needs a full episode.\n")

    def cells(alpha, pred=None):
        return [s["cells"][alpha] for s in states
                if (pred is None or pred(s)) and alpha in s["cells"]]

    # --- eligibility: a metric is only meaningful where a target exists ----
    elig = [s for s in states if s["low_sample_arms_n1"]]
    print(f"PRIMARY SET: states with >=1 arm at n=1/reward=0: "
          f"{len(elig)}/{len(states)}")
    print("  (a revisit metric is undefined where there is nothing to "
          "revisit)\n")

    rows = [
        ("1 uncertainty_recognition", "uncertainty_recognition", None),
        ("2 policy_targets_low_sample (n=1)",
         "policy_targets_low_sample_n1", "elig"),
        ("2b recognition_AND_targets_low_sample",
         "recognition_and_targets_low_sample", "elig"),
        ("2c policy_targets_untried", "policy_targets_untried", None),
        ("3 low_sample_revisit_choice (n=1)",
         "low_sample_revisit_choice_n1", "elig"),
        ("3b chose_untried", "chose_untried", None),
        ("5 any_grounding_error (clean)", "any_grounding_error_clean", None),
        ("5b any_grounding_error (raw)", "any_grounding_error_raw", None),
        ("5c hashtag_present", "hashtag_present", None),
        ("5d empty_clean", "empty_clean", None),
        ("5e policy_parsed", "policy_parsed", None),
        ("5f action_follows_policy", "action_follows_policy", None),
    ]
    print(f"{'metric':38s} " + "  ".join(f"{('a=' + a):>9s}" for a in A))
    print("-" * 74)
    for label, key, scope in rows:
        src = elig if scope == "elig" else states
        vals = [_rate([s["cells"][a] for s in src if a in s["cells"]], key)
                for a in A]
        print(f"{label:38s} " + "  ".join(f"{v:9.3f}" for v in vals))

    # The TRANSITION: given that the text recognised uncertainty, did the
    # Policy then target a low-sample arm? This is the quantity the alpha=0
    # episodes put at .996 vs .031.
    print("\n2d TRANSITION  P(targets_low_sample | uncertainty_recognition)")
    cond = []
    for a in A:
        rec = [s["cells"][a] for s in elig
               if a in s["cells"] and s["cells"][a]["uncertainty_recognition"]]
        cond.append((_rate(rec, "policy_targets_low_sample_n1"), len(rec)))
    print(f"{'  conditional rate':38s} "
          + "  ".join(f"{v:9.3f}" for v, _ in cond))
    print(f"{'  denominator (recognised)':38s} "
          + "  ".join(f"{n:9d}" for _, n in cond))

    print("\n6 OUTCOME-FLAVOURED (reported last)")
    for label, key in [("margin", "margin"), ("norm_entropy", "norm_entropy"),
                       ("n_hashtags", "n_hashtags")]:
        vals = [_mean(cells(a), key) for a in A]
        print(f"{label:38s} " + "  ".join(f"{v:9.3f}" for v in vals))
    for label, key in [("oracle_best_revisit_choice",
                        "oracle_best_revisit_choice"),
                       ("oracle_policy_targets_best",
                        "oracle_policy_targets_best")]:
        vals = [_rate(cells(a), key) for a in A]
        print(f"{label:38s} " + "  ".join(f"{v:9.3f}" for v in vals))
    print("  ORACLE metrics read true-best identity, which the model cannot "
          "see.\n  Secondary diagnostics only -- never the headline.")

    # --- inference: seed-clustered, NOT state-level -----------------------
    print("\nINFERENCE: paired cluster bootstrap, RESAMPLING UNIT = SEED")
    print("  The 123 states come from 20 trajectories. Rates above are")
    print("  descriptive at state level (n=123); intervals below treat the")
    print("  seed as the unit (n=20), because states from one seed share a")
    print("  history, a reward tape and an arm order.")
    print(f"  {'metric':40s} {'contrast':11s} {'delta':>8s}  "
          f"{'95% CI':>18s}")
    for key, label in [
        ("uncertainty_recognition", "1 recognition"),
        ("policy_targets_low_sample_n1", "2 targets_low_sample"),
        ("recognition_and_targets_low_sample", "2b recog_AND_targets"),
        ("low_sample_revisit_choice_n1", "3 revisit_choice"),
        ("any_grounding_error_clean", "5 grounding_err_clean"),
        ("hashtag_present", "5c hashtag"),
    ]:
        src = elig if "low_sample" in key or "revisit" in key else states
        for a, tag in [("-4.0", "-4 vs 0"), ("4.0", "+4 vs 0")]:
            r = cluster_bootstrap_delta(src, key, "0.0", a)
            flag = "" if (r["lo"] <= 0 <= r["hi"]) else "  *"
            print(f"  {label:40s} {tag:11s} {r['delta']:+8.3f}  "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]{flag}")
    print("  * = interval excludes 0. An interval straddling 0 means NOT")
    print("    DETECTED at n=20 clusters -- never 'no effect'.")

    # --- per state type ---------------------------------------------------
    print("\nBY STATE TYPE  (low_sample_revisit_choice n=1, eligible only)")
    print("  Grid rows are 20 seeds paired within the state type (fewer if a")
    print("  state has no arm to revisit). The `critical_one_shot_zero` row is")
    print("  NOT the critical subset: it holds only the critical states whose")
    print("  round differs from a grid round. The critical subset is n=5 and")
    print("  is tabulated below by tag.")
    types = sorted({s["state_type"] for s in states})
    for t in types:
        sub = [s for s in elig if s["state_type"] == t]
        if not sub:
            continue
        vals = [_rate([s["cells"][a] for s in sub], "low_sample_revisit_choice_n1")
                for a in A]
        note = ("   <- not the critical subset; see below (n=5)"
                if t == "critical_one_shot_zero" else "")
        print(f"  {t:26s} n={len(sub):2d}  "
              + "  ".join(f"{v:9.3f}" for v in vals) + note)

    # --- critical subset: per-state, no statistics ------------------------
    crit = [s for s in states if s["is_critical"]]
    print(f"\nCRITICAL LOCK-IN SUBSET (n={len(crit)}) -- per-state only.")
    print("  ORACLE-SELECTED post-hoc subset: the criterion reads true-best")
    print("  identity. Every figure here is an oracle-assisted SECONDARY")
    print("  diagnostic. Proportions only; no significance testing and no")
    print("  generalization beyond these five trajectories.")
    print(f"  {'state':34s} {'abandoned':10s} "
          + " ".join(f"{('a=' + a):>11s}" for a in A))
    for s in crit:
        # From the bank, never low_sample_arms[0]: alphabetical order does not
        # encode which arm was abandoned.
        ab = s["critical_arm"] or "?"
        acts = []
        for a in A:
            c = s["cells"].get(a, {})
            mark = "*" if c.get("critical_arm_revisit_choice") else " "
            acts.append(f"{str(c.get('action', '?')).replace('Button ', ''):>10s}"
                        f"{mark}")
        print(f"  {s['state_id']:34s} {ab.replace('Button ', ''):10s} "
              + " ".join(acts))
    n_rev = {a: sum(1 for s in crit
                    if s["cells"].get(a, {}).get("critical_arm_revisit_choice"))
             for a in A}
    print(f"  {'re-chose abandoned arm':34s} {'':10s} "
          + " ".join(f"{n_rev[a]:>7d}/{len(crit)} " for a in A))
    print("  * = re-chose the abandoned arm")

    print("\nREAD ORDER: layers 1-2 are Stage-1 TEXT, layer 3 is the EXECUTED")
    print("action. A rise in 1 without 2 is recognition without action -- the")
    print("exact gap this experiment exists to test (.996 vs .031 at alpha=0).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, help="re-print from a stored JSON")
    ap.add_argument("--dry_run", action="store_true",
                    help="build prompts and eligibility only; no model")
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--hs", default="llama3")
    ap.add_argument("--type", default="non")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--layers", default="11-20")
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="alpha cells to run (default -4 0 4). To split "
                         "across GPUs use '-4 0' and '0 4' with separate "
                         "--out files, then --merge them. alpha=0 must be in "
                         "every shard: it is the paired baseline.")
    ap.add_argument("--merge", type=Path, nargs="+",
                    help="merge shard JSONs into --out and report")
    args = ap.parse_args()

    if args.report:
        report(json.loads(args.report.read_text()))
        return

    if args.merge:
        merge_shards(args.merge, args.out)
        return

    global ALPHAS
    if args.alphas is not None:
        ALPHAS = tuple(float(a) for a in args.alphas)
        if 0.0 not in ALPHAS:
            raise SystemExit(
                "alpha=0 must be included: every contrast is paired against "
                "it within the same state, and a shard without it cannot be "
                "merged or interpreted on its own.")
        if len(set(ALPHAS)) != len(ALPHAS):
            raise SystemExit(f"duplicate alphas: {ALPHAS}")

    bank = json.loads(args.bank.read_text())
    env = br.get_environment("easy")
    if bank["environment"]["name"] != env.name:
        raise SystemExit("bank environment mismatch")
    states = bank["states"]
    ls, le = (int(x) for x in args.layers.split("-"))

    vc = None
    diff_by_alpha = {a: None for a in ALPHAS}
    n_layers = 0
    mask_sha = None
    if not args.dry_run:
        diff_by_alpha, n_layers, mask_sha = load_masks(
            args.base_dir, args.hs, args.type, args.mask_type,
            args.percentage, args.size, ls, le)

    fp = run_fingerprint(args, bank, args.bank, mask_sha)
    done: dict[str, dict] = {}
    if args.out.exists() and not args.overwrite and not args.dry_run:
        prev = json.loads(args.out.read_text())
        old = prev.get("run_fingerprint", {})
        if old == fp:
            done = {s["state_id"]: s for s in prev.get("states", [])}
            print(f"[resume] {len(done)} states already stored")
        else:
            diffs = [f"    {k}: stored {old.get(k)!r} != current {v!r}"
                     for k, v in fp.items() if old.get(k) != v]
            raise SystemExit(
                f"{args.out} holds a DIFFERENT configuration:\n"
                + "\n".join(diffs or ["    (stored file predates fingerprinting)"])
                + "\nUse --overwrite or a new --out.")

    if not args.dry_run:
        from llms import VicundaModel
        vc = VicundaModel(model_path=args.model_dir)
        vc.model.eval()
        if not hasattr(vc, "steering_fire_count"):
            raise SystemExit(
                "this VicundaModel exposes no steering_fire_count(); the "
                "injection cannot be attested. Sync llms.py -- do not bypass.")
        print(f"env={env.name} K={env.k} layers={ls}-{le} (L={n_layers})")
        print(f"alphas {list(ALPHAS)}  action_alpha=0 (Stage 2 never steered)")
        print(f"expected fires per state: steered "
              f"{{rationale: {n_layers}, action: 0}}")

    todo = [s for s in states if s["state_id"] not in done]
    print(f"{len(todo)} states to run, {len(done)} resumed")

    results = list(done.values())
    t_start = time.time()
    for i, st in enumerate(todo, 1):
        t0 = time.time()
        rec = run_state(vc, st, env, diff_by_alpha, n_layers, args.dry_run)
        results.append(rec)
        if not args.dry_run:
            print(f"  [{i}/{len(todo)}] {st['state_id']:26s} "
                  + " ".join(f"a={a}:{rec['cells'][str(a)]['action'][-1]}"
                             for a in ALPHAS)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            doc = {
                "eval_version": EVAL_VERSION,
                "run_fingerprint": fp,
                "bank_version": bank["state_bank_version"],
                "bank_file": args.bank.name,
                "alphas": list(ALPHAS),
                "action_alpha": 0.0,
                "stage2_steered": False,
                "layers": [ls, le],
                "n_steered_layers": n_layers,
                "analysis_unit": (
                    "Descriptive rates are state-level (n=123). Inference "
                    "resamples SEEDS (n=20 clusters), paired within state: "
                    "states from one seed share a history, a reward tape and "
                    "an arm order. The critical subset is oracle-selected, "
                    "n=5, proportions only."),
                "states": results,
            }
            _atomic_write(args.out, json.dumps(doc, indent=1))   # checkpoint

    if args.dry_run:
        elig = [r for r in results if r["low_sample_arms_n1"]]
        print(f"\nDRY RUN: {len(results)} states, "
              f"{len(elig)} with an n=1/reward=0 arm")
        crit = [r for r in results if r["is_critical"]]
        print(f"critical subset {len(crit)}: "
              f"{[r['state_id'] for r in crit]}")
        ex = results[0]
        print(f"\n--- example Stage 1 prompt ({ex['state_id']}) ---")
        print(ex["rationale_prompt"])
        print("--- end (tail must be a single space = token 220) ---")
        print(f"generations if run: {len(results)} x {len(ALPHAS)} = "
              f"{len(results) * len(ALPHAS)} Stage 1, same count Stage 2 "
              f"scorings of K={env.k} rows")
        return

    print(f"\nwrote {args.out}  ({len(results)} states, "
          f"{time.time() - t_start:.0f}s)")
    report(json.loads(args.out.read_text()))


if __name__ == "__main__":
    main()
