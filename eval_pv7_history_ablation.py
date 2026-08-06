#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""H0/H1 choice-history ablation on the frozen pv7 states. alpha=0 only.

Asks ONE question: does showing the model its own choice sequence change the
Evidence/Policy it forms? No Calculator, no alpha, no new environment.

    H0   the current pv7 Stage-1 prompt, unchanged
    H1   H0 plus a CHOICE HISTORY block

pv7 files are not modified. This module imports `bandit_pv7` read-only and
builds H1 by inserting a block into the ALREADY-RENDERED H0 state string, so
H0 is byte-identical to pv7 by construction rather than by reimplementation.

WHY BOTH ARMS ARE RE-RUN
------------------------
H0 is re-generated here rather than read from `pv7_stage1_alpha.json`, even
though it is the same prompt. Comparing against the stored file would pair a
new H1 run against a cell produced on a different device and possibly a
different day; margin and norm_entropy are exactly the quantities that drift
across GPUs from bf16 accumulation order. Re-running both on one card makes
every contrast same-device and same-state. The stored pv7 cell is still used,
as an assertion: H0 must reproduce it.

HISTORY GOES IN STAGE 1 ONLY
----------------------------
Stage 2 keeps the frozen S1 prompt. If Stage 2 also saw `[A A A ...]`, the
repeated label would prime the candidate logits directly, and a change in the
chosen arm could not be attributed to reasoning rather than to token priming.
Putting history in Stage 1 alone keeps the readout on "did the model's own
Evidence/Policy change".

WHAT THIS FACTOR IS NOT
-----------------------
It is not neutral context. By r76 the block is ~60 identical letters, which is
simultaneously (a) evidence of lock-in a self-monitoring model should react to
and (b) a next-token continuation cue. In this state bank the last-chosen arm
IS the empirical best in 103/123 states, so for most states "continue the
sequence" and "exploit the best" predict the SAME button and a History effect
cannot be attributed between them. The 20 dissociating states are the
pre-registered subset for that question, reported per-state without testing.

Usage
-----
    python3.10 eval_pv7_history_ablation.py --dry_run
    python3.10 eval_pv7_history_ablation.py --out .../pv7_history.json
    python3.10 eval_pv7_history_ablation.py --report .../pv7_history.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import time
from collections import Counter
from pathlib import Path

import numpy as np

import bandit_reference as br
import bandit_pv7 as p7
import bandit_pv7_episode as ep
import eval_pv7_frozen_states as fe
import eval_pv7_stage1_alpha as SA


EVAL_VERSION = "pv7-history-ablation-v1"
HISTORY_BLOCK_VERSION = "hist-letters-v1"
DEFAULT_BANK = Path(__file__).with_name("bandit_pv7_lockin_states.json")
DEFAULT_OUT = Path(__file__).with_name("pv7_history_ablation.json")
ARMS = ("H0", "H1")

UNCERTAINTY_RE = SA.UNCERTAINTY_RE
HASHTAG_RE = SA.HASHTAG_RE


# ------------------------------------------------------------------ prompts

def _letters(history: list[dict]) -> list[str]:
    """`Button A` -> `A`. Raises on any label that is not `Button <X>`."""
    out = []
    for e in history:
        m = re.fullmatch(r"Button ([A-Z])", e["arm"])
        if not m:
            raise ValueError(f"unexpected arm label {e['arm']!r}")
        out.append(m.group(1))
    return out


def history_block(history: list[dict]) -> str:
    if not history:
        return "CHOICE HISTORY: none"
    return ("CHOICE HISTORY (oldest → newest):\n"
            f"[{' '.join(_letters(history))}]")


def build_h1_state(h0_state: str, history: list[dict]) -> str:
    """Insert the history block immediately before the OPTIONS table.

    String surgery on H0's own output, deliberately: re-implementing
    render_state would let H0 and H1 drift apart on some future pv7 edit, and
    the whole design rests on them differing ONLY by this block.
    """
    marker = "\n\nOPTIONS\n"
    if h0_state.count(marker) != 1:
        raise AssertionError("expected exactly one OPTIONS table in the state")
    head, tail = h0_state.split(marker, 1)
    # Keep H0's blank-line spacing on BOTH sides of the inserted block, so the
    # only textual difference is the block itself.
    return f"{head}\n\n{history_block(history)}{marker}{tail}"


def build_prompts(state: dict, env) -> tuple[str, str]:
    hist_pairs = [(e["arm"], e["reward"]) for e in state["history"]]
    h0_state = p7.render_state(state["arm_order"], hist_pairs,
                               state["round_idx"], env,
                               prompt_variant=p7.PROMPT_P1B)
    h1_state = build_h1_state(h0_state, state["history"])
    instr = p7._P1B_INSTRUCTION
    h0 = f"{h0_state}\n\n{instr}\n\n{p7.RATIONALE_ANCHOR}"
    h1 = f"{h1_state}\n\n{instr}\n\n{p7.RATIONALE_ANCHOR}"
    for p in (h0, h1):
        p7._assert_single_trailing_space(p, p7.RATIONALE_ANCHOR)
    return h0, h1


# ----------------------------------------------------------------- checking

def check_state(state: dict, env, h0: str, h1: str) -> dict:
    """Hard per-state validation. A violation raises; nothing is recorded.

    Every item here can silently corrupt the comparison rather than crash it,
    which is why each is asserted rather than reported.
    """
    hist = state["history"]
    order = state["arm_order"]
    letters = _letters(hist)

    if len(hist) != state["round_idx"]:
        raise AssertionError(f"{state['state_id']}: history length "
                             f"{len(hist)} != round_idx {state['round_idx']}")
    legal = {re.fullmatch(r"Button ([A-Z])", a).group(1) for a in order}
    bad = sorted(set(letters) - legal)
    if bad:
        raise AssertionError(f"{state['state_id']}: illegal labels {bad}")

    # Every arm's appearances must equal its OPTIONS trial count, so the two
    # renderings of the same history cannot disagree.
    shown = dict(Counter(letters))
    for arm in order:
        L = re.fullmatch(r"Button ([A-Z])", arm).group(1)
        n_hist = shown.get(L, 0)
        m = re.search(rf"- {re.escape(arm)}: (?:(\d+) rewards? / (\d+) "
                      r"trials?|UNTRIED)", h0)
        if not m:
            raise AssertionError(f"{state['state_id']}: no OPTIONS row for "
                                 f"{arm}")
        n_opt = int(m.group(2)) if m.group(2) else 0
        if n_hist != n_opt:
            raise AssertionError(
                f"{state['state_id']}: {arm} appears {n_hist}x in history "
                f"but OPTIONS says {n_opt} trials")

    if hist and letters[-1] != re.fullmatch(
            r"Button ([A-Z])", hist[-1]["arm"]).group(1):
        raise AssertionError(f"{state['state_id']}: tail label mismatch")
    if not hist and "CHOICE HISTORY: none" not in h1:
        raise AssertionError(f"{state['state_id']}: round 1 must show none")

    # No true-best leakage. The block is letters only and `diagnostics` never
    # reaches a renderer, but assert it rather than trust it: the block is
    # built from `history`, and a future edit that reached for `arm_map` or
    # `best_arm` would leak a probability into the prompt silently.
    for word in ("true", "best arm", "optimal", "probability of 0."):
        if word in h1.lower():
            raise AssertionError(
                f"{state['state_id']}: H1 contains {word!r}; the prompt must "
                "not reveal arm identities or true probabilities")

    # H0 and H1 must differ ONLY by the inserted block. This is the check that
    # actually catches a mis-rendered history: any wrong letter, extra line or
    # changed spacing makes the stripped H1 fail to equal H0.
    stripped = h1.replace("\n\n" + history_block(hist), "", 1)
    if stripped != h0:
        raise AssertionError(
            f"{state['state_id']}: H1 differs from H0 beyond the history "
            "block")
    return {
        "n_history": len(hist),
        "tail_run": _tail_run(letters),
        "distinct_last10": len(set(letters[-10:])) if letters else 0,
        "dominant_arm_fraction": (max(Counter(letters).values()) / len(letters)
                                  if letters else 0.0),
    }


def _tail_run(letters: list[str]) -> int:
    if not letters:
        return 0
    r = 1
    for x in reversed(letters[:-1]):
        if x == letters[-1]:
            r += 1
        else:
            break
    return r


def empirical_best(state: dict) -> str | None:
    """The empirical-best arm, or None when the maximum is TIED.

    Tie-tolerant on purpose, matching evaluate_competence_gate.py:115. Easy's
    three suboptimal arms share a probability, so ties are common (9/123 states
    here) and a bare argmax would silently return the first-listed tied arm.
    That is not a harmless simplification: it would classify 8 tied states as
    "dissociating" when there is no unique best to dissociate from, inflating
    the pre-registered subset from 12 to 20.
    """
    trials, wins = Counter(), Counter()
    for e in state["history"]:
        trials[e["arm"]] += 1
        wins[e["arm"]] += e["reward"]
    if not trials:
        return None
    rates = {a: wins[a] / trials[a] for a in trials}
    top = max(rates.values())
    tied = [a for a in rates if rates[a] == top]
    return tied[0] if len(tied) == 1 else None


def dissociates(state: dict) -> bool:
    """Last-chosen arm differs from a UNIQUE empirical best.

    Only in these states do 'continue the sequence' and 'exploit the best'
    predict different buttons, so only here can a History effect be attributed
    between inertia and self-monitoring. A tied best is excluded rather than
    broken arbitrarily: with a tie, "exploit the best" does not name one arm.
    """
    if not state["history"]:
        return False
    eb = empirical_best(state)
    return eb is not None and state["history"][-1]["arm"] != eb


# -------------------------------------------------------------------- runner

def run_state(vc, state: dict, env, dry_run: bool) -> dict:
    h0, h1 = build_prompts(state, env)
    diag = check_state(state, env, h0, h1)
    low1 = SA.low_sample_arms(state, 1)
    untried = SA.untried_arms(state)
    eb = empirical_best(state)
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
        "empirical_best": eb,          # None when the maximum is tied
        "empirical_best_tied": bool(state["history"]) and eb is None,
        "dissociating": dissociates(state),
        "history_diag": diag,
        "prompts": {"H0": h0, "H1": h1} if dry_run else None,
        "cells": {},
    }
    if dry_run:
        return out

    for arm, prompt in (("H0", h0), ("H1", h1)):
        aud_r = ep.audit_pv7_prompt(vc, prompt, "rationale")
        vc.steering_fire_count(reset=True)
        _torch().manual_seed(state["seed"] * 100_003 + state["round_idx"])
        gen = vc.generate(inputs=[prompt],
                          max_new_tokens=ep.RATIONALE_MAX_TOKENS,
                          temperature=0.0)
        fired_r = vc.steering_fire_count(reset=True)
        raw = gen[0] if isinstance(gen, list) else gen
        clean = p7.extract_evidence_policy_block(raw)

        # Stage 2: the FROZEN S1 prompt, no history, never steered.
        a_prompt = ep.build_action_prompt_s1(
            state["arm_order"],
            [(e["arm"], e["reward"]) for e in state["history"]],
            state["round_idx"], env, clean)
        if history_block(state["history"]) in a_prompt:
            raise AssertionError("history leaked into the Stage 2 prompt")
        aud_a = ep.audit_pv7_prompt(vc, a_prompt, "action")
        scores, act = ep.score_candidates_pv7(vc, a_prompt, env, None)
        fired_a = vc.steering_fire_count(reset=True)
        if (fired_r, fired_a) != (0, 0):
            raise SystemExit(
                f"{state['state_id']} {arm}: steering_fires "
                f"{{rationale: {fired_r}, action: {fired_a}}} != 0/0. This is "
                "an alpha=0 ablation; no hook may fire.")

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
            # 2 control channel
            "policy_targets_untried": tgt in untried,
            # 3 executed
            "low_sample_revisit_choice": act in low1,
            "chose_untried": act in untried,
            # 4 persistence
            "policy_targets_dominant": bool(
                state["history"]
                and tgt == Counter(e["arm"] for e in
                                   state["history"]).most_common(1)[0][0]),
            "policy_targets_last_chosen": bool(
                state["history"] and tgt == state["history"][-1]["arm"]),
            "policy_targets_empirical_best": tgt == eb,
            "chose_last_chosen": bool(
                state["history"] and act == state["history"][-1]["arm"]),
            "chose_empirical_best": act == eb,
            # 5 critical (oracle-selected)
            "critical_arm_revisit_choice": bool(
                state["tags"]["is_critical_one_shot_zero"]
                and act == crit_arm),
            # 6 text/quality
            "uncertainty_recognition": bool(UNCERTAINTY_RE.search(clean)),
            "any_grounding_error_clean": gnd["any_grounding_error"],
            "hashtag_present": bool(HASHTAG_RE.search(raw)),
            "n_hashtags": len(HASHTAG_RE.findall(raw)),
            "empty_clean": not clean.strip(),
            "margin": (ordered[0] - ordered[1] if len(ordered) > 1
                       else float("nan")),
            "norm_entropy": ep._norm_entropy(list(scores.values())),
            "candidate_scores": dict(scores),
        }
    return out


def _torch():
    import torch
    return torch


# ------------------------------------------------------------------ report

def _rate(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (sum(bool(x) for x in v) / len(v)) if v else float("nan")


def report(doc: dict) -> None:
    S = doc["states"]
    n = len(S)
    print("=" * 76)
    print(f"pv7 CHOICE-HISTORY ABLATION   ({doc['eval_version']})")
    print(f"  {n} frozen states, alpha=0 both arms, Stage 2 unchanged")
    print("=" * 76)
    print("Read order is mechanism-first. Reward and true-best outcome are")
    print("NOT readouts: the states came from alpha=0 pv7 trajectories.\n")

    elig = [s for s in S if s["low_sample_arms_n1"]]
    print(f"eligible (>=1 arm at n=1/reward=0): {len(elig)}/{n}\n")

    print(f"{'metric':40s} {'H0':>9s} {'H1':>9s} {'delta':>8s}")
    print("-" * 76)
    rows = [
        ("1 policy_targets_one_shot_zero  PRIMARY",
         "policy_targets_one_shot_zero", "elig"),
        ("2 policy_targets_untried (control)",
         "policy_targets_untried", None),
        ("3 low_sample_revisit_choice", "low_sample_revisit_choice", "elig"),
        ("3b chose_untried", "chose_untried", None),
        ("4 policy_targets_dominant", "policy_targets_dominant", None),
        ("4b policy_targets_last_chosen", "policy_targets_last_chosen", None),
        ("4c chose_last_chosen", "chose_last_chosen", None),
        ("4d chose_empirical_best", "chose_empirical_best", None),
        ("6 uncertainty_recognition", "uncertainty_recognition", None),
        ("6b any_grounding_error", "any_grounding_error_clean", None),
        ("6c policy_parsed", "policy_parsed", None),
        ("6d action_follows_policy", "action_follows_policy", None),
        ("6e hashtag_present", "hashtag_present", None),
        ("6f empty_clean", "empty_clean", None),
    ]
    for label, key, scope in rows:
        src = elig if scope == "elig" else S
        a = _rate([s["cells"]["H0"] for s in src], key)
        b = _rate([s["cells"]["H1"] for s in src], key)
        print(f"{label:40s} {a:9.3f} {b:9.3f} {b - a:+8.3f}")

    for key in ("margin", "norm_entropy"):
        a = st.mean(s["cells"]["H0"][key] for s in S)
        b = st.mean(s["cells"]["H1"][key] for s in S)
        print(f"{('7 ' + key):40s} {a:9.3f} {b:9.3f} {b - a:+8.3f}")

    print("\nINFERENCE: paired cluster bootstrap, unit = SEED (n=20)")
    for key, label in [("policy_targets_one_shot_zero", "1 targets_1shot0"),
                       ("policy_targets_untried", "2 targets_untried"),
                       ("low_sample_revisit_choice", "3 revisit_choice"),
                       ("policy_targets_last_chosen", "4b targets_last"),
                       ("chose_last_chosen", "4c chose_last"),
                       ("any_grounding_error_clean", "6b grounding_err")]:
        src = elig if "1shot0" in label or "revisit" in label else S
        r = SA.cluster_bootstrap_delta(src, key, "H0", "H1")
        flag = "" if (r["lo"] <= 0 <= r["hi"]) else "  *"
        print(f"  {label:26s} {r['delta']:+7.3f}  "
              f"[{r['lo']:+.3f}, {r['hi']:+.3f}]{flag}")
    print("  * = excludes 0. Straddling 0 = NOT DETECTED, never 'no effect'.")

    print("\nPERSISTENCE BY TAIL-RUN LENGTH  (the inertia dose axis)")
    print("  If History raises persistence monotonically with run length,")
    print("  that is the imitation reading, not self-monitoring.")
    print(f"  {'state_type':14s} {'tail_run':>9s} "
          f"{'H0 last':>9s} {'H1 last':>9s} {'delta':>8s}")
    for t in sorted({s["state_type"] for s in S}):
        sub = [s for s in S if s["state_type"] == t]
        tr = st.mean(s["history_diag"]["tail_run"] for s in sub)
        a = _rate([s["cells"]["H0"] for s in sub], "chose_last_chosen")
        b = _rate([s["cells"]["H1"] for s in sub], "chose_last_chosen")
        print(f"  {t:14s} {tr:9.1f} {a:9.3f} {b:9.3f} {b - a:+8.3f}")

    dis = [s for s in S if s["dissociating"]]
    n_tie = sum(1 for s in S if s.get("empirical_best_tied"))
    print(f"\nDISSOCIATING SUBSET (n={len(dis)}) -- pre-registered")
    print("  Last-chosen arm != a UNIQUE empirical best, so 'continue the")
    print("  sequence' and 'exploit the best' predict DIFFERENT buttons.")
    print("  Everywhere else a History effect cannot be attributed between")
    print(f"  them. {n_tie} further states have a TIED empirical best and are")
    print("  excluded: with a tie, 'the best arm' does not name one button.")
    print("  Proportions and per-state only; no significance testing.")
    for label, key in [("chose_last_chosen (inertia)", "chose_last_chosen"),
                       ("chose_empirical_best", "chose_empirical_best"),
                       ("targets_one_shot_zero",
                        "policy_targets_one_shot_zero")]:
        a = _rate([s["cells"]["H0"] for s in dis], key)
        b = _rate([s["cells"]["H1"] for s in dis], key)
        print(f"  {label:30s} {a:9.3f} {b:9.3f} {b - a:+8.3f}")

    crit = [s for s in S if s["is_critical"]]
    print(f"\nCRITICAL SUBSET (n={len(crit)}) -- ORACLE-SELECTED, per-state")
    print("  Secondary diagnostic. No testing, no generalization.")
    for s in crit:
        print(f"  {s['state_id']:34s} abandoned {s['critical_arm']}")
        for arm in ARMS:
            c = s["cells"][arm]
            hit = "  <-- re-chose" if c["critical_arm_revisit_choice"] else ""
            print(f"    {arm}  act={c['action']:9s} "
                  f"tgt={str(c['policy_target']):9s}{hit}")
    n_rev = {a: sum(1 for s in crit
                    if s["cells"][a]["critical_arm_revisit_choice"])
             for a in ARMS}
    print("  re-chose abandoned arm: "
          + "  ".join(f"{a}: {n_rev[a]}/{len(crit)}" for a in ARMS))

    print("\n" + "=" * 76)
    print("FROZEN READING KEY (pre-registered)")
    print("=" * 76)
    print("  1shot0 revisit UP, late exploit held -> History surfaces lock-in")
    print("  only untried UP                      -> does not fix one-shot-zero")
    print("  dominant/last-chosen UP              -> the list reinforces inertia")
    print("  everything up, margin far down       -> untargeted churn/flailing")
    print("  nothing moves                        -> counts suffice; go to")
    print("                                          the Beta Calculator")


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
    ap.add_argument("--pv7_alpha", type=Path,
                    help="stored pv7 alpha=0 result; H0 is asserted to match")
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
        "history_block_version": HISTORY_BLOCK_VERSION,
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
        print(f"env={env.name} K={env.k}  arms {ARMS}  alpha=0 both")

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
        SA._atomic_write(args.out, json.dumps(
            {"eval_version": EVAL_VERSION, "run_fingerprint": fp,
             "arms": list(ARMS), "alpha": 0.0,
             "analysis_unit": (
                 "Descriptive rates are state-level (n=123). Inference "
                 "resamples SEEDS (n=20). The dissociating subset (n=20) and "
                 "the critical subset (n=5) are per-state only."),
             "states": results}, indent=1))

    if args.dry_run:
        n_dis = sum(1 for r in results if r["dissociating"])
        print(f"\nDRY RUN: {len(results)} states, {n_dis} dissociating, "
              f"{sum(1 for r in results if r['is_critical'])} critical")
        ex = next(r for r in results if r["round_idx"] == 50)
        print(f"\n--- H1 prompt ({ex['state_id']}) ---")
        print(ex["prompts"]["H1"])
        print("--- end (tail must be a single space = token 220) ---")
        print(f"\ngenerations if run: {len(results)} x 2 = "
              f"{len(results) * 2} Stage 1 + same Stage 2 scorings")
        return

    if args.pv7_alpha and args.pv7_alpha.exists():
        prev = {s["state_id"]: s for s in
                json.loads(args.pv7_alpha.read_text())["states"]}
        bad = [r["state_id"] for r in results
               if r["state_id"] in prev
               and prev[r["state_id"]]["cells"]["0.0"]["rationale_clean"]
               != r["cells"]["H0"]["rationale_clean"]]
        print(f"\nH0 vs stored pv7 alpha=0: "
              f"{len(results) - len(bad)}/{len(results)} identical")
        if bad:
            print(f"  DIFFERS on {len(bad)}, e.g. {bad[:5]}")
            print("  H0 is the same prompt, so a difference is device or")
            print("  version drift -- investigate before reading H1-H0.")

    print(f"\nwrote {args.out}")
    report(json.loads(args.out.read_text()))


if __name__ == "__main__":
    main()
