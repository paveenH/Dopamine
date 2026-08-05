#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Phase 3: paired INTERFACE comparison on the frozen pv7 state bank.

What this measures
------------------
Three complete interfaces evaluated on the SAME frozen states:

    P0  pv6 prompt + pv6 sanitizer + "Choice: Button" anchor + ' A' candidates
    P1  pv7 structural prompt      + "Evidence: "/"Choose Button: " (tok 220)
                                   + bare A/B/C/D candidates
    P2  P1 + the light weak-evidence hint

P0 runs the REAL pv6 path (bandit_reference + bandit_pv6_episode.score_candidates)
and is never "aligned" to pv7's token 220 or bare candidates. Doing so would
measure a wording change instead of the interface change that actually happened,
and the interface (anchor position, candidate tokenization) is the thing under
test. Consequence: margin/entropy differences between P0 and P1/P2 include the
full interface delta and must NOT be read as a pure prompt-wording effect.

Held constant across all three: the frozen state, temperature=0, and
max_new_tokens, so generation budget never leaks into the prompt contrast.

Selection rule (frozen, and the reason this script does not compute reward):
prompts are chosen on validity, evidence grounding, rationale completion and
cost ONLY. The frozen states were sampled from alpha=0 pv6 trajectories, so
selecting on true-best outcome would fit the prompt to its own history.

Analysis units -- the bank has 120 slots but only 107 unique histories:
  * six per-type tables keep all 20 slots each (one history answers a different
    diagnostic question under each state type);
  * any POOLED statistic deduplicates on state_fingerprint and reports n=107.
Every comparison is PAIRED on state_fingerprint.

Usage
-----
    python3.10 eval_pv7_frozen_states.py --model_dir /path/to/llama3-8b \\
        --out pv7_frozen_eval.json                    # all three arms
    python3.10 eval_pv7_frozen_states.py --dry_run    # prompts only, no GPU
    python3.10 eval_pv7_frozen_states.py --report pv7_frozen_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as stats
import time
from collections import Counter, defaultdict
from pathlib import Path

import bandit_reference as br
import bandit_pv7 as p7

BANK = Path(__file__).with_name("bandit_pv7_frozen_states.json")

ARMS_P0 = "p0_pv6_interface"
ARMS_P1 = p7.PROMPT_P1
ARMS_P2 = p7.PROMPT_P2
ALL_ARMS = (ARMS_P0, ARMS_P1, ARMS_P2)

# Held constant across arms so generation budget is not confounded with prompt.
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0


# ───────────────────────────── prompt construction ──────────────────────────

def _history(snap: dict) -> list[tuple[str, int]]:
    """Bank stores history as {'arm','reward'} dicts; both protocols want pairs."""
    return [(h["arm"], h["reward"]) for h in snap["history"]]


def build_prompts(arm: str, snap: dict, env: br.Environment, clean: str | None):
    """Return (stage1_prompt, stage2_prompt_or_None) for one arm.

    Each arm builds through ITS OWN protocol module. P0 must not be routed
    through pv7 helpers -- that is the whole point of the comparison.
    """
    order = snap["arm_order"]
    history = _history(snap)
    ri = snap["round_idx"]

    if arm == ARMS_P0:
        # pv6 takes a mapping keyed by arm name; only the KEY ORDER is used for
        # display, so a value of 0.0 is safe and keeps true probs out of it.
        arm_map = {name: 0.0 for name in order}
        s1 = br.build_rationale_prompt(arm_map, history, ri, env)
        s2 = (None if clean is None else
              br.build_action_prompt(arm_map, history, ri, env, clean))
        return s1, s2

    s1 = p7.build_rationale_prompt(order, history, ri, env, prompt_variant=arm)
    s2 = (None if clean is None else
          p7.build_action_prompt(order, history, ri, env, clean,
                                 prompt_variant=arm))
    return s1, s2


def sanitize_for(arm: str, raw: str) -> str:
    return br.sanitize_rationale(raw) if arm == ARMS_P0 else p7.sanitize_rationale(raw)


def candidates_for(arm: str, env: br.Environment) -> tuple[list[str], list[str]]:
    """(suffixes_to_tokenize, arm_labels_they_denote) for this arm's protocol."""
    if arm == ARMS_P0:
        sfx = br.candidate_suffixes(env)          # ' A'  -> token 362
        return sfx, [f"Button{s}" for s in sfx]
    sfx = p7.candidate_suffixes(env)              # 'A'   -> token 32
    return sfx, [p7.candidate_arm(s) for s in sfx]


# ─────────────────────────── validity (per protocol) ────────────────────────

def check_validity(arm: str, tokenizer, s1: str, s2: str,
                   env: br.Environment, snap: dict) -> dict:
    """Prompt validity judged against THIS arm's own expectations.

    P0 is not required to end on token 220 -- pv6's anchor is "Choice: Button",
    whose tail is ' Button' (6739). Holding pv6 to pv7's invariant would report
    a spurious failure for a protocol that never claimed it.
    """
    ids1 = tokenizer.encode(s1, add_special_tokens=False)
    ids2 = tokenizer.encode(s2, add_special_tokens=False)
    sfx, labels = candidates_for(arm, env)
    cand_ids = [tokenizer.encode(s, add_special_tokens=False) for s in sfx]

    out = {
        "stage1_tail_id": ids1[-1],
        "stage2_tail_id": ids2[-1],
        "candidate_ids": [c[0] if len(c) == 1 else c for c in cand_ids],
        "candidates_single_token": all(len(c) == 1 for c in cand_ids),
        "n_tokens_stage1": len(ids1),
        "n_tokens_stage2": len(ids2),
    }
    if arm == ARMS_P0:
        tail = tokenizer.decode([ids2[-1]]).strip()
        out["anchor_ok"] = (s2.endswith(br.ACTION_ANCHOR) and tail == "Button")
        out["expected_tail"] = "' Button' (pv6 anchor)"
    else:
        out["anchor_ok"] = (
            ids1[-1] == p7.EXPECTED_WHITESPACE_TOKEN_ID
            and ids2[-1] == p7.EXPECTED_WHITESPACE_TOKEN_ID
            and s1.endswith(p7.RATIONALE_ANCHOR)
            and s2.endswith(p7.ACTION_ANCHOR))
        out["expected_tail"] = f"token {p7.EXPECTED_WHITESPACE_TOKEN_ID} (pv7)"

    # OPTIONS order + state arithmetic, recomputed independently of the renderer.
    order = snap["arm_order"]
    trials, succ = Counter(), Counter()
    for a, r in _history(snap):
        trials[a] += 1
        succ[a] += r
    listed = re.findall(r"^- (Button [A-E])", s1, flags=re.M)
    out["options_order_ok"] = listed == order
    bad = []
    for a in order:
        if trials[a] == 0:
            if re.search(rf"^- {a}: \d", s1, flags=re.M):
                bad.append(f"{a}: untried rendered numerically")
        else:
            want = f"{succ[a]}/{trials[a]}"
            m = re.search(rf"^- {a}: (\d+) reward\w* / (\d+) trial\w*, "
                          rf"empirical rate ([\d.]+)", s1, flags=re.M)
            if not m:
                bad.append(f"{a}: row missing")
            elif (int(m.group(1)), int(m.group(2))) != (succ[a], trials[a]):
                bad.append(f"{a}: counts {m.group(1)}/{m.group(2)} != {want}")
            elif abs(float(m.group(3)) - succ[a] / trials[a]) > 5e-3:
                bad.append(f"{a}: rate {m.group(3)} wrong")
    out["state_arithmetic_ok"] = not bad
    out["state_arithmetic_errors"] = bad
    return out


# ───────────────────── rationale completion / grounding ─────────────────────

_WORD = re.compile(r"\b[\w'-]+\b")
_POLICY = re.compile(r"policy\s*[:：]", re.I)
_EXPLORE = re.compile(r"\bexplor\w*", re.I)
_EXPLOIT = re.compile(r"\bexploit\w*", re.I)
_ARM_MENTION = re.compile(r"button\s*([A-E])\b", re.I)
_NUM_PAIR = re.compile(r"(\d+)\s*(?:reward|success)\w*\s*(?:/|out of|from)\s*(\d+)", re.I)
_RATE = re.compile(r"(?:rate|probability)\D{0,12}?([01]?\.\d+)", re.I)
# Bare observation claims: "1 reward", "gave a reward", "3 trials", "observed
# rewards". Used ONLY for the Round-1 test, where any such claim is false.
_BARE_OBSERVATION = re.compile(
    r"\b\d+\s*(?:reward|success|trial|pull)\w*|"
    r"\b(?:gave|yielded|returned|produced|observed|has|have|had)\b[^.;]{0,30}?"
    r"\b(?:reward|success)\w*", re.I)
_SUPPORT_CLAIM = re.compile(
    r"\b(strong|well[- ]support\w*|reliabl\w*|solid|robust|confident\w*|"
    r"clearly|proven|established)\b", re.I)


def completion_flags(arm: str, raw: str, clean: str, n_gen_tokens: int) -> dict:
    """Format/completion metrics. P0 is scored on BOTH raw and clean.

    pv6's sanitizer DELETES whole lines containing a `Choice:` marker, so a
    clean-only reading understates how often the model tried to commit early.
    pv7 does not delete, so its raw/clean differ only by trailing whitespace --
    reporting both keeps the two protocols on one scale.
    """
    f = p7.rationale_format_flags(raw, clean)
    f_raw = p7.rationale_format_flags(raw, raw.rstrip())
    body = clean.strip()
    words = _WORD.findall(body)
    out = {
        **f,
        # pre-sanitizer view (the generation tendency, before pv6 deletes)
        "raw_contains_action_anchor": f_raw["rationale_contains_action_anchor"],
        "raw_contains_evidence_anchor": f_raw["rationale_contains_evidence_anchor"],
        "sanitizer_removed_content": len(raw.rstrip()) != len(clean),
        # pv6-specific: its sanitizer drops `Choice:` lines
        "raw_contains_pv6_choice_marker": bool(
            re.search(r"choice\s*[:：]", raw, re.I)),
        "n_words": len(words),
        "within_35_words": len(words) <= 35,
        "is_single_line": "\n" not in body,
        "has_policy_marker": bool(_POLICY.search(body)),
        "ends_on_sentence_punct": bool(body) and body[-1] in ".!?",
        "hit_token_cap": n_gen_tokens >= MAX_NEW_TOKENS,
        "n_gen_tokens": n_gen_tokens,
    }
    return out


def grounding_flags(clean: str, snap: dict) -> dict:
    """Does the rationale misstate the state it was given?

    Conservative by construction: `any_grounding_error` flags ONLY mechanically
    checkable factual errors -- fabricated reward/trial counts, a wrong
    empirical rate, an UNTRIED arm described as observed, and (Round 1) any
    reward claim at all, since nothing has been observed yet.

    `support_overclaim` is DESCRIPTIVE and deliberately excluded from
    `any_grounding_error`: whether 1 trial may be called "well-supported" is an
    epistemic judgement, not a factual error, and folding it into a
    hallucination rate would conflate two different failure kinds.
    """
    trials, succ = Counter(), Counter()
    for a, r in _history(snap):
        trials[a] += 1
        succ[a] += r
    order = snap["arm_order"]
    untried = {a for a in order if trials[a] == 0}
    body = clean.strip()

    legit_pairs = {(succ[a], trials[a]) for a in order if trials[a]}
    legit_rates = {round(succ[a] / trials[a], 2) for a in order if trials[a]}
    claimed = [(int(m.group(1)), int(m.group(2))) for m in _NUM_PAIR.finditer(body)]
    fabricated_pairs = [c for c in claimed if c not in legit_pairs]
    claimed_rates = [round(float(m.group(1)), 2) for m in _RATE.finditer(body)]
    fabricated_rates = [r for r in claimed_rates if r not in legit_rates]

    # "Button X has N reward(s)" where X is untried
    untried_as_tried = []
    for m in re.finditer(r"button\s*([A-E])\b[^.;]{0,40}?(\d+)\s*reward", body, re.I):
        arm = f"Button {m.group(1).upper()}"
        if arm in untried:
            untried_as_tried.append(arm)

    # A Round-1 fabrication is usually a BARE claim ("Button A with 1 reward"),
    # not a well-formed "N/M" pair, so requiring _NUM_PAIR here would miss the
    # observed case. Any reward/trial/rate assertion counts at Round 1, where by
    # construction nothing has been observed.
    any_reward_claim = bool(claimed) or bool(claimed_rates) or bool(
        _BARE_OBSERVATION.search(body))

    # DESCRIPTIVE ONLY -- never part of any_grounding_error. Calling a 1-2 trial
    # arm "strong/well-supported/reliable" is an epistemic overclaim, not a
    # misstatement of the table.
    thin = {a for a in order if 0 < trials[a] <= 2}
    overclaim = []
    for m in re.finditer(r"button\s*([A-E])\b", body, re.I):
        arm = f"Button {m.group(1).upper()}"
        if arm in thin and _SUPPORT_CLAIM.search(body[max(0, m.start() - 60):
                                                     m.start() + 60]):
            overclaim.append(arm)

    return {
        "fabricated_count_pairs": fabricated_pairs,
        "fabricated_rates": fabricated_rates,
        "untried_described_as_tried": sorted(set(untried_as_tried)),
        "any_grounding_error": bool(fabricated_pairs or fabricated_rates
                                    or untried_as_tried),
        # Round 1: nothing has been observed, so ANY reward/rate claim is false.
        "round1_hallucinated_evidence": (snap["round_idx"] == 0
                                         and any_reward_claim),
        "support_overclaim": sorted(set(overclaim)),
    }


def policy_flags(clean: str, chosen: str | None) -> dict:
    """HEURISTIC, POST-HOC DESCRIPTIVE. Not a pre-registered outcome or gate.

    `policy_target_source` records how the target was obtained:
      * policy_segment    -- parsed after an explicit `Policy:` marker
      * full_text_fallback-- last button named anywhere (LOOSE: this can pick up
                             a button mentioned in the Evidence half, which is
                             not the policy target at all)
      * none              -- no button named

    Headline action-follows-policy therefore uses policy_segment WITH a clear
    (explore|exploit) stance; the fallback is a sensitivity analysis only. The
    parse rate must be reported alongside, or a high follow-rate computed over
    a small easy-to-parse subset will read as a general result.
    """
    body = clean.strip()
    explore, exploit = bool(_EXPLORE.search(body)), bool(_EXPLOIT.search(body))
    named = [f"Button {m.group(1).upper()}" for m in _ARM_MENTION.finditer(body)]
    m_pol = _POLICY.search(body)
    seg_named = ([f"Button {m.group(1).upper()}"
                  for m in _ARM_MENTION.finditer(body[m_pol.end():])]
                 if m_pol else [])
    if seg_named:
        target, source = seg_named[-1], "policy_segment"
    elif named:
        target, source = named[-1], "full_text_fallback"
    else:
        target, source = None, "none"
    stance = ("explore" if explore and not exploit else
              "exploit" if exploit and not explore else
              "both" if explore and exploit else "unclear")
    return {
        "policy_stance": stance,
        "stance_is_clear": stance in ("explore", "exploit"),
        "policy_names_button": bool(named),
        "policy_target": target,
        "policy_target_source": source,
        "action_follows_policy": (None if target is None or chosen is None
                                  else target == chosen),
    }


# ───────────────────────────── choice diagnostics ───────────────────────────

def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def choice_diag(scores: dict[str, float], order: list[str], chosen: str) -> dict:
    labels = list(scores)
    vals = [scores[l] for l in labels]
    p = _softmax(vals)
    srt = sorted(vals, reverse=True)
    ent = -sum(q * math.log(q) for q in p if q > 0)
    return {
        "margin": srt[0] - srt[1],
        "norm_entropy": ent / math.log(len(labels)),
        "top1_prob": max(p),
        "chosen_label": chosen,
        "chosen_letter": chosen.split()[-1],
        "chosen_position": order.index(chosen) + 1,
    }


# ─────────────────────────────── the eval loop ──────────────────────────────

def run_arm(arm, vc, bank, env, tokenizer, dry_run):
    import torch  # local: --dry_run must not need a GPU stack

    rows = []
    for i, snap in enumerate(bank["states"]):
        t0 = time.time()
        s1, _ = build_prompts(arm, snap, env, None)

        if dry_run:
            raw, n_gen, t_gen = "", 0, 0.0
        else:
            torch.manual_seed(0)
            out = vc.generate(inputs=[s1], max_new_tokens=MAX_NEW_TOKENS,
                              temperature=TEMPERATURE)
            raw = out[0] if isinstance(out, list) else out
            n_gen = len(tokenizer.encode(raw, add_special_tokens=False))
            t_gen = time.time() - t0

        clean = sanitize_for(arm, raw)
        _, s2 = build_prompts(arm, snap, env, clean)

        t1 = time.time()
        if dry_run:
            scores, chosen, t_score = {}, None, 0.0
        else:
            from bandit_pv6_episode import score_candidates
            if arm == ARMS_P0:
                scores, chosen = score_candidates(vc, s2, env, None)
            else:
                sfx, labels = candidates_for(arm, env)
                tok_ids = [tokenizer.encode(s, add_special_tokens=False)
                           for s in sfx]
                per_tok = vc.regenerate_logits_teacher_forcing(
                    prompts=[s2] * len(sfx), answer_token_ids=tok_ids,
                    diff_matrices=None)
                import numpy as np
                scores = {}
                for lab, ids, lg in zip(labels, tok_ids, per_tok):
                    lp = 0.0
                    for k, tid in enumerate(ids):
                        row = torch.from_numpy(np.asarray(lg[k], dtype=np.float32))
                        lp += float(torch.log_softmax(row, dim=-1)[tid])
                    scores[lab] = lp
                chosen = max(scores, key=scores.get)
            t_score = time.time() - t1

        row = {
            "state_id": snap["state_id"],
            "state_fingerprint": snap["state_fingerprint"],
            "state_type": snap["state_type"],
            "seed": snap["seed"],
            "round_idx": snap["round_idx"],
            "arm": arm,
            "rationale_raw": raw,
            "rationale_clean": clean,
            "validity": check_validity(arm, tokenizer, s1, s2, env, snap),
            "completion": completion_flags(arm, raw, clean, n_gen),
            # RAW is the primary text-generation readout (what the model wrote);
            # CLEAN explains the final choice (what Stage 2 actually saw). They
            # differ for P0, whose sanitizer deletes whole `Choice:` lines.
            "grounding_raw": grounding_flags(raw, snap),
            "grounding": grounding_flags(clean, snap),
            "cost": {"stage1_gen_s": round(t_gen, 4),
                     "stage2_score_s": round(t_score, 4),
                     "stage1_gen_tokens": n_gen},
        }
        if scores:
            row["scores"] = scores
            row["choice"] = choice_diag(scores, snap["arm_order"], chosen)
            row["policy"] = policy_flags(clean, chosen)
        else:
            row["policy"] = policy_flags(clean, None)
        rows.append(row)

        if not dry_run and (i + 1) % 20 == 0:
            print(f"  {arm}: {i + 1}/{len(bank['states'])}", flush=True)
    return rows


# ───────────────────────────────── reporting ────────────────────────────────

def _rate(rows, path, want=True):
    vals = []
    for r in rows:
        cur = r
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            vals.append(bool(cur) == want)
    return (100.0 * sum(vals) / len(vals)) if vals else float("nan")


def _dedup(rows):
    """Pooled statistics use the 107 unique histories, not the 120 slots."""
    seen, out = set(), []
    for r in rows:
        if r["state_fingerprint"] in seen:
            continue
        seen.add(r["state_fingerprint"])
        out.append(r)
    return out


def report(doc):
    arms = [a for a in ALL_ARMS if a in doc["rows"]]
    print("=" * 78)
    print("PHASE 3 — PAIRED INTERFACE COMPARISON on frozen states")
    print("=" * 78)
    n120 = len(doc["rows"][arms[0]])
    n107 = len(_dedup(doc["rows"][arms[0]]))
    print(f"slots={n120}  unique histories (pooled n)={n107}")
    print("ALL percentages below are DEDUPLICATED (n=%d) unless a row says" % n107)
    print("slot-level. Both denominators are printed for prompt validity, where")
    print("they differ; a bare percentage with no denominator is a reporting bug.")
    print("P0 runs the full pv6 interface; margin/entropy vs P1/P2 therefore")
    print("include anchor + candidate-tokenization differences, NOT wording alone.\n")

    # Prompt validity is the one block where slot-level and deduplicated numbers
    # visibly disagree (pv6's TRIED/UNTRIED split), so print BOTH -- otherwise
    # 80.0% and 81.3% look like an inconsistency rather than two valid units.
    print("── PROMPT VALIDITY (each arm vs its OWN protocol)")
    print("   Both denominators are shown: slot-level uses all 120 slots,")
    print("   dedup uses the 107 unique histories. They are two valid units,")
    print("   not an inconsistency -- always state which one a figure is.")
    for key in ("anchor_ok", "candidates_single_token", "options_order_ok",
                "state_arithmetic_ok"):
        print(f"   {key}")
        for a in arms:
            slots = doc["rows"][a]
            ded = _dedup(slots)
            so = sum(1 for r in slots if r["validity"][key])
            do = sum(1 for r in ded if r["validity"][key])
            print(f"     {a:<22} slot {so:>3}/{len(slots)} = {100*so/len(slots):5.1f}%"
                  f"   dedup {do:>3}/{len(ded)} = {100*do/len(ded):5.1f}%")
    print()

    blocks = [
        ("EVIDENCE GROUNDING on RAW (primary text-generation readout)", [
            ("any_grounding_error", ("grounding_raw", "any_grounding_error")),
            ("round1_hallucination", ("grounding_raw", "round1_hallucinated_evidence")),
        ]),
        ("RATIONALE COMPLETION", [
            ("empty_rationale", ("completion", "empty_rationale")),
            ("is_single_line", ("completion", "is_single_line")),
            ("within_35_words", ("completion", "within_35_words")),
            ("has_policy_marker", ("completion", "has_policy_marker")),
            ("ends_on_sentence_punct", ("completion", "ends_on_sentence_punct")),
            ("hit_token_cap", ("completion", "hit_token_cap")),
            ("starts_redundant_evidence", ("completion", "starts_with_redundant_evidence")),
        ]),
        ("ANCHOR COLLISION (clean, and RAW = pre-sanitizer)", [
            ("clean: action anchor", ("completion", "rationale_contains_action_anchor")),
            ("raw:   action anchor", ("completion", "raw_contains_action_anchor")),
            ("clean: evidence anchor", ("completion", "rationale_contains_evidence_anchor")),
            ("raw:   pv6 'Choice:' marker", ("completion", "raw_contains_pv6_choice_marker")),
            ("sanitizer removed content", ("completion", "sanitizer_removed_content")),
        ]),
        ("EVIDENCE GROUNDING on CLEAN (what Stage 2 actually saw)", [
            ("any_grounding_error", ("grounding", "any_grounding_error")),
            ("untried_as_tried", ("grounding", "untried_described_as_tried")),
            ("round1_hallucination", ("grounding", "round1_hallucinated_evidence")),
            ("support_overclaim [descriptive]", ("grounding", "support_overclaim")),
        ]),
    ]
    for title, items in blocks:
        print(f"── {title}")
        print(f"   {'metric':<30}" + "".join(f"{a:>18}" for a in arms))
        for label, path in items:
            cells = ""
            for a in arms:
                rows = _dedup(doc["rows"][a])
                if label.startswith("round1_"):
                    # Denominator is the 20 Round-1 states, not the whole bank.
                    sub = [r for r in rows if r["round_idx"] == 0]
                    v = (100.0 * sum(1 for r in sub if r[path[0]][path[1]])
                         / len(sub) if sub else float("nan"))
                else:
                    v = _rate(rows, path)
                cells += f"{v:>17.1f}%"
            print(f"   {label:<30}{cells}")
        print()

    print("── POLICY CLARITY  [HEURISTIC, POST-HOC DESCRIPTIVE —")
    print("   not a pre-registered outcome or gate metric]")
    print(f"   {'stance':<30}" + "".join(f"{a:>18}" for a in arms))
    for st in ("explore", "exploit", "both", "unclear"):
        cells = ""
        for a in arms:
            rows = _dedup(doc["rows"][a])
            v = 100.0 * sum(1 for r in rows
                            if r["policy"]["policy_stance"] == st) / len(rows)
            cells += f"{v:>17.1f}%"
        print(f"   {st:<30}{cells}")
    for label, key in [("names a button", "policy_names_button"),
                       ("stance is clear", "stance_is_clear")]:
        cells = "".join(f"{_rate(_dedup(doc['rows'][a]), ('policy', key)):>17.1f}%"
                        for a in arms)
        print(f"   {label:<30}{cells}")
    print(f"\n   {'policy_target_source':<30}" + "".join(f"{a:>18}" for a in arms))
    for src in ("policy_segment", "full_text_fallback", "none"):
        cells = ""
        for a in arms:
            rows = _dedup(doc["rows"][a])
            v = 100.0 * sum(1 for r in rows
                            if r["policy"]["policy_target_source"] == src) / len(rows)
            cells += f"{v:>17.1f}%"
        print(f"   {'  ' + src:<30}{cells}")

    print("\n── ACTION FOLLOWS POLICY  [HEURISTIC] — parse rate reported with each")
    print("   row, because a high rate over few easily-parsed rationales is not")
    print("   a general result.")
    print(f"   {'subset':<38}" + "".join(f"{a:>22}" for a in arms))
    subsets = [
        ("HEADLINE: policy_segment + clear stance",
         lambda r: (r["policy"]["policy_target_source"] == "policy_segment"
                    and r["policy"]["stance_is_clear"])),
        ("  ... and collision-free (raw)",
         lambda r: (r["policy"]["policy_target_source"] == "policy_segment"
                    and r["policy"]["stance_is_clear"]
                    and not r["completion"]["raw_contains_action_anchor"])),
        ("SENSITIVITY: incl. full_text_fallback", lambda r: True),
    ]
    for label, filt in subsets:
        cells = ""
        for a in arms:
            ded = _dedup(doc["rows"][a])
            rows = [r for r in ded if filt(r)]
            vals = [r["policy"]["action_follows_policy"] for r in rows
                    if r.get("policy", {}).get("action_follows_policy") is not None]
            v = 100.0 * sum(vals) / len(vals) if vals else float("nan")
            cells += f"{v:.1f}% (n={len(vals)}/{len(ded)})".rjust(22)
        print(f"   {label:<38}{cells}")

    if any("choice" in r for r in doc["rows"][arms[0]]):
        print("\n── CHOICE DIAGNOSTICS  (interface-inclusive; see header)")
        print(f"   {'metric':<30}" + "".join(f"{a:>18}" for a in arms))
        for label, key in [("margin (mean)", "margin"),
                           ("norm_entropy (mean)", "norm_entropy"),
                           ("top1_prob (mean)", "top1_prob")]:
            cells = ""
            for a in arms:
                v = stats.mean(r["choice"][key] for r in _dedup(doc["rows"][a])
                               if "choice" in r)
                cells += f"{v:>18.3f}"
            print(f"   {label:<30}{cells}")

        print(f"\n   {'chosen letter':<30}" + "".join(f"{a:>18}" for a in arms))
        letters = sorted({r["choice"]["chosen_letter"] for a in arms
                          for r in doc["rows"][a] if "choice" in r})
        for L in letters:
            cells = ""
            for a in arms:
                rows = [r for r in _dedup(doc["rows"][a]) if "choice" in r]
                v = 100.0 * sum(1 for r in rows
                                if r["choice"]["chosen_letter"] == L) / len(rows)
                cells += f"{v:>17.1f}%"
            print(f"   {'  Button ' + L:<30}{cells}")
        print(f"\n   {'chosen display position':<30}" + "".join(f"{a:>18}" for a in arms))
        for pos in (1, 2, 3, 4, 5):
            rows0 = [r for r in _dedup(doc["rows"][arms[0]]) if "choice" in r]
            if not any(r["choice"]["chosen_position"] == pos for r in rows0):
                continue
            cells = ""
            for a in arms:
                rows = [r for r in _dedup(doc["rows"][a]) if "choice" in r]
                v = 100.0 * sum(1 for r in rows
                                if r["choice"]["chosen_position"] == pos) / len(rows)
                cells += f"{v:>17.1f}%"
            print(f"   {'  pos ' + str(pos):<30}{cells}")

    print("\n── COST (mean per state; stages reported separately)")
    print(f"   {'metric':<30}" + "".join(f"{a:>18}" for a in arms))
    for label, key in [("stage1 gen tokens", "stage1_gen_tokens"),
                       ("stage1 gen seconds", "stage1_gen_s"),
                       ("stage2 score seconds", "stage2_score_s")]:
        cells = "".join(
            f"{stats.mean(r['cost'][key] for r in _dedup(doc['rows'][a])):>18.3f}"
            for a in arms)
        print(f"   {label:<30}{cells}")

    print("\n── BY STATE TYPE (all 20 slots each; NOT deduplicated)")
    for a in arms:
        print(f"   {a}")
        by = defaultdict(list)
        for r in doc["rows"][a]:
            by[r["state_type"]].append(r)
        for st in sorted(by):
            rows = by[st]
            print(f"     {st:<42} n={len(rows):<4} "
                  f"grounding_err={_rate(rows, ('grounding','any_grounding_error')):5.1f}%  "
                  f"cap={_rate(rows, ('completion','hit_token_cap')):5.1f}%  "
                  f"policy={_rate(rows, ('completion','has_policy_marker')):5.1f}%")
    print("\nSelection rule: choose on validity / grounding / completion / cost.")
    print("Never on true-best outcome — the states came from alpha=0 pv6 runs.")


def main():
    ap = argparse.ArgumentParser()
    # A HF repo id, matching run_bandit_reference.sh:119 -- VicundaModel passes
    # it straight to from_pretrained, so the local HF cache resolves it. There
    # is no filesystem model directory on the server.
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--arms", nargs="*", default=list(ALL_ARMS))
    ap.add_argument("--out", default="pv7_frozen_eval.json")
    ap.add_argument("--dry_run", action="store_true",
                    help="build/validate prompts only; no model, no GPU")
    ap.add_argument("--report", help="re-print tables from a stored result JSON")
    args = ap.parse_args()

    if args.report:
        report(json.load(open(args.report)))
        return

    bank = json.load(open(BANK))
    env = br.get_environment("easy")
    assert bank["environment"]["name"] == env.name

    if args.dry_run:
        # Tokenizer only, from the local cache: --dry_run must work on the
        # analysis box with no GPU and no download.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, local_files_only=True)
        vc = None
    else:
        from llms import VicundaModel
        vc = VicundaModel(model_path=args.model_dir)
        tokenizer = vc.tokenizer

    doc = {
        "protocol": "pv7-phase3-frozen-state",
        "state_bank_version": bank["state_bank_version"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "arms": args.arms,
        "note": ("P0 runs the complete pv6 interface (its own prompt, sanitizer, "
                 "anchor and ' A' candidates). Differences vs P1/P2 are INTERFACE "
                 "differences, not wording-only effects."),
        "rows": {},
    }
    for arm in args.arms:
        print(f"[{arm}]", flush=True)
        doc["rows"][arm] = run_arm(arm, vc, bank, env, tokenizer, args.dry_run)

    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {args.out}")
    report(doc)


if __name__ == "__main__":
    main()
