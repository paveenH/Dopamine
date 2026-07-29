#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Armed Bandit benchmark with RSN diff injection.
Closely follows EVOLvE/BanditBench design (ClothesShopping scenario):
  - K=5 arms with semantic multi-token names (shuffled per run)
  - Fixed Bernoulli reward probabilities: best=0.7, 0.5, 0.4, 0.3, worst=0.1
  - T=50 rounds per episode, full raw history shown each round
  - Generation mode (vc.regenerate), temperature=1.0
  - Parse output by case-insensitive string matching; invalid → random arm
  - RSN diff injection applied at prefill (via vc.regenerate)

Metrics:
  - opt_frac:        proportion of rounds choosing the best arm
  - explore_rate:    proportion of rounds choosing non-best arms
  - worst_frac:      proportion choosing the worst arm
  - cum_regret:      Σ(best_prob − chosen_prob) over all rounds
  - early_opt_frac:  opt_frac for rounds 1–20
  - late_opt_frac:   opt_frac for rounds 21–50
  - invalid_rate:    proportion of rounds with unparseable output

Usage:
  python get_answer_bandit.py \
    --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --hs llama3 --size 8B --type non \
    --percentage 0.5 --mask_type nmd \
    --configs 0-11-20 4-11-20 neg4-11-20 \
    --num_runs 30 --num_rounds 50 \
    --data data1
"""

import os
import gc
import re
import csv
import json
import shutil
import random
import argparse
import numpy as np
import torch

from llms import VicundaModel
import utils

# ── Arm names from EVOLvE ClothesShopping scenario (first 10 names) ──────────
CLOTHES_NAMES = [
    "Velvet Vogue Jacket",
    "Silk Serenity Dress",
    "Urban Mystique Jeans",
    "Celestial Symphony Scarf",
    "Retro Revival Sneakers",
    "Ethereal Elegance Blouse",
    "Midnight Mirage Trousers",
    "Vintage Vibe Coat",
    "Opulent Oasis Gown",
    "Mystic Mosaic Shirt",
]
K = 5
REWARD_PROBS_ORDERED = [0.7, 0.5, 0.4, 0.3, 0.1]  # assigned to shuffled names
ACTION_UNIT = "item"


def shuffle_arms(seed: int) -> dict[str, float]:
    """Return name→reward_prob mapping in PROMPT ORDER, counterbalancing position.

    The caller uses list(arm_map.keys()) as the arm order shown in the prompt, so
    the iteration order of this dict IS the display order.

    POSITION LEAKAGE (fixed 2026-07-28 — invalidates all earlier Bandit results):
    the previous version shuffled the NAMES and then zipped them against the
    descending probability list, so the dict was always
        {name_a: 0.7, name_b: 0.5, name_c: 0.4, name_d: 0.3, name_e: 0.1}
    i.e. the best arm sat at display position 1 and the worst at position 5 for
    EVERY seed (verified over seeds 0-29: best position was 1 in all 30). Only
    *which name* was best varied. So OptFrac could not be distinguished from a
    first-option bias, and combined with the old permissive parser (which
    returned the first arm name appearing anywhere in the output) a reply that
    merely echoed the option list scored as "chose the best arm, validly".

    The fix keeps the name→probability assignment shuffled AND shuffles the
    display order independently, so the best arm's position is uniform across
    seeds. `position_of_best(seed)` is exposed for the counterbalancing check.
    """
    rng = random.Random(seed)
    names = CLOTHES_NAMES[:K]
    # 1) which name gets which reward probability
    assign = names[:]
    rng.shuffle(assign)
    name_to_prob = {name: prob for name, prob in zip(assign, REWARD_PROBS_ORDERED)}
    # 2) independently, the order the names are DISPLAYED in
    display = names[:]
    rng.shuffle(display)
    return {name: name_to_prob[name] for name in display}


def position_of_best(seed: int) -> int:
    """1-based display position of the best arm for `seed` (counterbalance check)."""
    am = shuffle_arms(seed)
    probs = list(am.values())
    return probs.index(max(probs)) + 1


def best_arm(arm_map: dict[str, float]) -> str:
    return max(arm_map, key=arm_map.get)


def worst_arm(arm_map: dict[str, float]) -> str:
    return min(arm_map, key=arm_map.get)


def summarize_history(arm_names: list[str],
                      history: list[tuple[str, int]],
                      untried_semantics: bool = False) -> str:
    """EVOLvE SummaryContextLayerMAB: per-arm pull count + mean reward.

    Format copied verbatim from banditbench/agents/context.py:
        f"\\n{action_name} {action_unit}, {n} times, average reward {reward:.2f}"
    including the `n + 1e-6` denominator (so an unpulled arm reads 0.00, not a
    division error).

    NOTE it iterates `arm_names` in a FIXED order, so the summary block does not
    re-leak the display shuffle beyond what the option list already shows.

    `untried_semantics=True` fixes ONE encoding bug and nothing else: an arm with
    n=0 renders as "UNTRIED" instead of "0 times, average reward 0.00". The
    verbatim formula is KEPT for every n>0 arm, so tried arms stay numerically
    identical to EVOLvE and the only lost comparability is on the untried ones —
    which is precisely what is being fixed.

    WHY this is a bug and not a wording preference (measured on the 20-seed
    Llama3 α=0 run, `bandit_validity_C20_llama3`): an untried arm rendered 0.00
    reads as TIED-WORST or WORSE THAN EVERY TRIED ARM in 5 of 5 inspected seeds.
    6 of 20 seeds never tried the best arm at all and scored OptFrac exactly
    0.000, while the 14 that did try it averaged 0.434. The model was not failing
    to exploit — it exploits its own observations 70–100% of the time, and one
    seed did so 40/40 while locked onto an arm it had measured at 0.14, because
    the four arms it had never touched all displayed 0.00. Discovery was
    suppressed by the table, not by the model.
    """
    n_actions = {name: 0 for name in arm_names}
    action_rewards = {name: 0.0 for name in arm_names}
    for name, reward in history:
        n_actions[name] += 1
        action_rewards[name] += reward
    snippet = ""
    for name in arm_names:
        n = n_actions[name]
        if untried_semantics and n == 0:
            snippet += f"\n{name}: UNTRIED"
            continue
        reward = action_rewards[name] / (n + 1e-6)
        if untried_semantics:
            snippet += f"\n{name}: {n} trials, average reward {reward:.2f}"
        else:
            snippet += (f"\n{name} {ACTION_UNIT}, {n} times, "
                        f"average reward {reward:.2f}")
    return snippet


# Version of the PROMPT TEXT + PARSER contract, not of the flags.
#
# The resume key is built from the interface FLAGS, so it cannot see a change
# that leaves the flags identical: reword build_prompt(), loosen
# parse_choice_exact(), and a re-run with the same flags would find a matching
# key and SKIP the cell, returning results produced by the old protocol. Bump
# this whenever the prompt wording, the anchor, or the parser's accept/reject
# boundary changes, so those cells re-run instead of resuming.
#
# Do NOT bump for changes that cannot alter a stored result (comments,
# refactors, new opt-in flags that default off) — a bump invalidates every
# stored cell at that tag.
#
# The version is PER-PROTOCOL, not global: bumping it for everyone would
# invalidate the resume key of runs whose text did not change. `--untried_
# semantics` is the only flag that rewrites prompt text, so only it advances the
# version; every legacy combination stays on pv1 and resumes unchanged.
#
#   pv0 — implicit, pre-versioning. Legacy CSV rows only (see _legacy_iface_tag).
#   pv1 — 2026-07-29. summary_history / answer_anchor+prefill / strict
#         parse_choice_exact (whole-reply strictness), as run by
#         run_bandit_validity.sh.
#   pv2 — 2026-07-29, --untried_semantics ONLY. UNTRIED rendering + the
#         task-representation prompt (fixed-but-unknown reward probability,
#         explicit round/horizon, arbitrary names/positions, UNTRIED≠0, and a
#         concrete read-the-table instruction replacing EVOLvE's abstract
#         "Balance exploration—…" sentence).
PROTOCOL_VERSION_LEGACY = "pv1"
PROTOCOL_VERSION_UNTRIED = "pv2"


def build_prompt(
    arm_names: list[str],
    history: list[tuple[str, int]],
    use_role: bool = True,
    summary_history: bool = False,
    answer_anchor: bool = False,
    untried_semantics: bool = False,
    round_idx: int = 0,
    num_rounds: int = 50,
) -> str:
    """
    EVOLvE-style prompt:
      task instruction → history context → query

    use_role=True  → Assistant-Role variant (AI fashion-assistant persona;
                     the §4.7 "Assistant Role" condition).
    use_role=False → No-Role neutral variant: strips the boutique/recommend/
                     customer persona but keeps the EVOLvE explore/exploit
                     paragraph and the history+query structure verbatim, so the
                     only thing removed is the role framing (the §4.7 "No Role"
                     condition, baseline ~0.816). The explore/exploit wording,
                     verbalizer, and query line are identical across both arms.

    untried_semantics=True → the pv2 task-representation prompt. It repairs what
    the model is TOLD, never what it must DO: no "try every option once", no
    fixed explore/exploit round split, no UCB or confidence-interval hint, no
    forced initialization. Whether to explore stays the model's decision, which
    is the whole point — that decision is the behavioural channel α is meant to
    move, so scripting it would delete the dependent variable (the same trap as
    IGT v4, where externally supplying deliberation returned every value/risk
    readout to n.s.).

    Four repairs, each fixing a specific misstatement of the task:
      1. reward probability is FIXED but unknown — without it the model has no
         reason to believe past means predict future draws, i.e. no reason to
         think exploration pays off at all;
      2. explicit round/horizon — makes exploration's future value visible, and
         lets late rounds tilt toward exploitation on their own;
      3. names/positions are arbitrary — a control, not a redundancy: at n=20
         several Llama seeds DID degenerate toward a display-position lock
         (P(first)=1.00, 0.94, 0.68, 0.66), and Qwen locks position 1 at 0.988;
      4. UNTRIED≠0 (see summarize_history) plus a concrete instruction on how to
         read the table, replacing EVOLvE's abstract "Balance exploration—…"
         sentence, which states the concept without saying what in the table
         corresponds to it.

    ⚠ COMPARABILITY: repair 4 replaces text copied verbatim from EVOLvE, so a
    pv2 run is NO LONGER a faithful de-roled port of BanditBench and must not be
    described as one. That is deliberate — the verbatim text was measured to
    misrepresent the task — but any cross-paper claim has to be dropped or
    re-qualified accordingly.
    """
    names_str = "[" + ", ".join(arm_names) + "]"
    if untried_semantics:
        lines = [
            f"There are {K} options: {names_str}.",
            "",
            f"You will make {num_rounds} choices in total. "
            f"This is choice {round_idx + 1} of {num_rounds}.",
            "Each option has a fixed but unknown probability of giving reward 1.",
            "The option names and list positions are arbitrary and do not "
            "indicate reward.",
            "",
            "Observed results:",
        ]
        if history:
            lines.append(
                summarize_history(arm_names, history,
                                  untried_semantics=True).lstrip("\n")
            )
        else:
            # Round 1: every arm would render as UNTRIED. Five identical lines
            # read as a formatting artifact rather than information, and the
            # option list above already names the choice set.
            lines.append("No options have been tried yet.")
        lines += [
            "",
            "UNTRIED means that the option has no reward estimate yet; it does "
            "not mean that its average reward is zero.",
            "",
            "When choosing, consider both learning about UNTRIED options and "
            "using options that have produced high observed rewards. "
            f"Maximize your total reward across all {num_rounds} choices.",
            "",
            f"Choose exactly one option from {names_str}.",
            "Respond with only the exact option name.",
        ]
        return "\n".join(lines)
    if use_role:
        lines = [
            f"You are in an online boutique powered by a bandit algorithm "
            f"that offers a variety of clothing options from different brands.",
            f"There are {K} unique clothing items available, named {names_str}.",
            "You choose an item to recommend based on past choices and rewards.",
            "You aim to find the clothing item that customers are most likely to purchase and enjoy.",
            "Each time a customer buys a recommended item, you update your strategy "
            "to better predict and meet future customer preferences.",
            "",
        ]
    else:
        lines = [
            f"There are {K} options available, named {names_str}.",
            "You choose one option each turn based on past choices and rewards.",
            "You aim to find the option that gives the highest reward.",
            "Each turn, you observe the reward of the chosen option and update your strategy.",
            "",
        ]
    if answer_anchor:
        lines += [
            "Balance exploration—trying different options to estimate their rewards—with "
            "exploitation—choosing the option currently estimated to give the highest "
            "reward—to maximize total reward.",
            "",
        ]
    elif use_role:
        lines += [
            "A good strategy to optimize for reward in these situations requires balancing exploration "
            "and exploitation. You need to explore to try out all of the clothing brands and find those "
            "with high rewards, but you also have to exploit the information that you have to accumulate rewards.",
            "",
        ]
    else:
        lines += [
            "A good strategy to optimize for reward in these situations requires balancing exploration "
            "and exploitation. You need to explore to try out all of the options and find those "
            "with high rewards, but you also have to exploit the information that you have to accumulate rewards.",
            "",
        ]

    if history:
        n = len(history)
        if summary_history:
            # EVOLvE SummaryHistory: per-arm aggregate instead of the raw log.
            lines.append(
                f"So far you have interacted {n} times with the following "
                f"choices and rewards:"
            )
            lines.append(summarize_history(arm_names, history).lstrip("\n"))
            lines.append("")
        else:
            lines.append(f"So far you have interacted {n} times with the following choices and rewards:")
            for name, reward in history:
                lines.append(f"{name} {ACTION_UNIT}, reward {reward}")
            lines.append("")

    if answer_anchor:
        # `Choice: ` is prefilled immediately after this prompt, so the model
        # only needs to generate the exact item name.
        lines.append(
            f"Which {ACTION_UNIT} will you choose next?\n"
            f"Choose exactly one from {names_str}.\n"
            f"Respond with only the exact {ACTION_UNIT} name."
        )
    else:
        lines.append(
            f"Which {ACTION_UNIT} will you choose next? "
            f"PLEASE RESPOND ONLY WITH {names_str} AND NO TEXT EXPLANATION."
        )
    return "\n".join(lines)


def parse_choice(output: str, arm_names: list[str],
                 rng: random.Random = None) -> tuple[str, bool, int]:
    """Parse the model's arm choice. Returns (chosen_name, is_valid, n_matched).

    STRICT SINGLE-CHOICE (changed 2026-07-28 — invalidates earlier results).
    The old version returned the FIRST arm name found anywhere in the output,
    scanning in `arm_names` (= prompt display) order. Combined with the old
    position leakage — the best arm sat at display position 1 for every seed —
    that meant a reply which simply echoed the whole option list was recorded as
    a VALID choice of the BEST arm. So OptFrac and "restated the menu" were
    numerically indistinguishable, and invalid_rate was underestimated.

    Now a reply is valid only if it names EXACTLY ONE arm. Matching >1 distinct
    arm is a format failure (n_matched > 1), not a choice. `n_matched` is
    returned so the caller can record it and separate "said nothing parseable"
    (0) from "restated the menu" (>1) offline.

    `rng` MUST be passed by callers that need reproducibility — it used to fall
    back to the unseeded global `random`, which at Llama α=−4 (invalid 0.20)
    meant one in five recorded choices was an unreproducible coin flip that fed
    the reward draw and the next prompt's history.
    """
    output_lower = output.strip().lower()
    matched = [name for name in arm_names if name.lower() in output_lower]
    if len(matched) == 1:
        return matched[0], True, 1
    # 0 matches = nothing parseable; >1 = menu restatement / multiple names.
    # Both are format failures → uniform random (EVOLvE fallback).
    return (rng or random).choice(arm_names), False, len(matched)


CHOICE_LINE_RE = re.compile(r"^\s*choice\s*[:：]\s*(.+?)\s*$", re.I | re.M)


def parse_choice_exact(output: str, arm_names: list[str],
                       rng: random.Random = None) -> tuple[str, bool, int]:
    """STRICT `Choice: <arm name>` parser (opt-in via --answer_anchor).

    Valid ONLY if the committed line's payload, after stripping surrounding
    whitespace and punctuation, is EXACTLY one arm name (case-insensitive).
    Trailing prose makes the reply INVALID:

        "Choice: Velvet Vogue Jacket"                    → valid
        "Choice: Velvet Vogue Jacket because it's best"  → INVALID (n_matched=1)

    This is deliberate and is the whole point of the anchor. The pilot has to
    separate "committed a decision" from "produced text that happens to contain
    an arm name" — the substring parser could not, and accepting a name embedded
    in explanation would re-admit exactly that ambiguity. A model that cannot
    emit a bare name on the anchored line has not adopted the decision protocol,
    which is itself a task-validity signal worth measuring.

    Which line is "the committed line": the LAST `Choice:` line if the reply
    contains one (mirroring CGT `simple2`'s last-match convention — the model
    often restates before committing); otherwise, under the `Choice: ` PREFILL
    the generation starts mid-line, so the FIRST line of the continuation is the
    payload.

    Strictness extends BEYOND that line: every OTHER non-empty line must also be
    a `Choice:` line, so the reply as a whole contains nothing but the decision
    protocol. Without this, the prefill branch (which has no `Choice:` of its own
    to anchor on) would accept

        "Velvet Vogue Jacket\nBecause it seems best."
        "Velvet Vogue Jacket\nOptions: Silk Serenity Dress, Urban Mystique Jeans"

    since only the first line was inspected — re-admitting the "explained instead
    of deciding" and "restated the menu" cases the anchor exists to exclude.

    `n_matched` is kept compatible with parse_choice(): 0 = nothing recognisable,
    1 = exactly one arm name present, >1 = several. NOTE that with this parser
    n_matched=1 no longer implies valid — a name plus trailing prose scores
    (invalid, 1). That combination is precisely the "explained instead of
    deciding" case, so it is worth having its own signature. `n_matched` counts
    over the WHOLE reply, so a trailing menu restatement still scores >1.
    """
    lows = {name.lower(): name for name in arm_names}
    text = output or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    cands = CHOICE_LINE_RE.findall(text)
    if cands:
        payload = cands[-1]
        # Everything else must itself be a `Choice:` line — no prose, no menu.
        extra = [ln for ln in lines if not CHOICE_LINE_RE.match(ln)]
    else:
        # With the `Choice: ` PREFILL the anchor is already in the prompt, so the
        # generation starts mid-line and contains no `Choice:` of its own. Then
        # the first line of the continuation IS the payload, and there must be
        # no further text after it.
        if not lines:
            return (rng or random).choice(arm_names), False, 0
        payload = lines[0]
        extra = lines[1:]
    if extra:
        # Committed line may be fine, but the reply carries extra text.
        low_all = text.lower()
        inside = [name for low, name in lows.items() if low in low_all]
        return (rng or random).choice(arm_names), False, len(inside)
    payload = payload.strip().strip(".,;:!?\"'`*[]()").strip()
    pl = payload.lower()
    if pl in lows:
        return lows[pl], True, 1
    # Not an exact match. Count how many arm names appear so the caller can tell
    # "one name + prose" (1) from "restated the menu" (>1) from "nothing" (0),
    # but do NOT accept any of them as a choice.
    inside = [name for low, name in lows.items() if low in pl]
    return (rng or random).choice(arm_names), False, len(inside)


def get_feedback(arm: str, arm_map: dict[str, float], rng: random.Random) -> int:
    return 1 if rng.random() < arm_map[arm] else 0


def run_episode(
    vc: VicundaModel,
    diff_mtx,
    num_rounds: int,
    seed: int,
    use_role: bool = True,
    summary_history: bool = False,
    answer_anchor: bool = False,
    use_chat: bool = False,
    max_new_tokens: int = 20,
    untried_semantics: bool = False,
) -> dict:
    rng = random.Random(seed)
    # Separate stream for the invalid-parse fallback. Kept independent of `rng`
    # ON PURPOSE: if both drew from one stream, a condition with more invalid
    # parses (the −α cells, up to 20%) would consume different numbers of draws
    # and desynchronise the REWARD sequence, so α conditions would no longer see
    # comparable reward luck. Two streams keep reward draws aligned per round.
    fallback_rng = random.Random(1_000_000 + seed)
    arm_map = shuffle_arms(seed)
    arm_names = list(arm_map.keys())   # shuffled order for this run
    opt = best_arm(arm_map)
    worst = worst_arm(arm_map)

    history = []     # list of (name, reward)
    choices = []
    feedbacks = []
    invalids = []
    valid_flags = []   # per-round: was the reply a parseable single choice?
    n_matched = []     # per-round: how many distinct arm names appeared
    raws = []          # per-round: full generation (diagnostics)
    # Verbatim copies of the string actually handed to the model, AFTER the chat
    # wrapper and the `Choice: ` prefill. Round 0 (no history) and a mid-run
    # round (history present) are kept, because those are the two shapes that
    # differ. A rebuilt approximation is not attestation — this is the input.
    prompt_attest = {}

    for round_idx in range(num_rounds):
        prompt = build_prompt(arm_names, history, use_role=use_role,
                              summary_history=summary_history,
                              answer_anchor=answer_anchor,
                              untried_semantics=untried_semantics,
                              round_idx=round_idx, num_rounds=num_rounds)
        if use_chat:
            # Single user turn carrying the whole state (the task is fully
            # described by the summary/history block, so no dialogue is needed).
            # NOTE this shifts the activation distribution away from the bare
            # string the NMD mask was extracted in — acceptable for an α=0
            # task-validity pilot, but see CLAUDE.md before running a sweep.
            msgs = [{"role": "user", "content": prompt}]
            prompt = vc.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        if answer_anchor:
            # Prefill the anchor so the next token IS the arm name — same
            # 施力点 logic as betting's "Bet: " and CGT's "Answer: ". Without
            # this the anchor is only an instruction and the model can preface
            # it with reasoning that eats the token budget.
            prompt = prompt + ("Choice: " if use_chat else "\nChoice: ")
        if round_idx == 0 or round_idx == min(10, num_rounds - 1):
            prompt_attest[f"round_{round_idx}"] = prompt
        # temperature=1.0 means generation is sampled. Without re-seeding, the
        # torch RNG state at round t depends on every generation before it, so
        # two α conditions never face the same sampling noise and repeating a
        # run does not reproduce it. Seed per (seed, round) so sampling luck is
        # MATCHED ACROSS α — the α contrast is the whole point of the sweep, and
        # this removes one noise source from it. Deliberately independent of α:
        # the same (run, round) gets the same draw in every condition.
        torch.manual_seed(seed * 100_003 + round_idx)
        output = vc.regenerate(
            inputs=[prompt],
            diff_matrices=diff_mtx,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
        )
        raw = output[0] if isinstance(output, list) else output
        parser_fn = parse_choice_exact if answer_anchor else parse_choice
        arm, valid, nmatch = parser_fn(raw, arm_names, rng=fallback_rng)

        reward = get_feedback(arm, arm_map, rng)
        choices.append(arm)
        feedbacks.append(reward)
        invalids.append(0 if valid else 1)
        valid_flags.append(bool(valid))
        n_matched.append(nmatch)
        raws.append(raw)
        history.append((arm, reward))

    early_end = min(20, num_rounds)
    late_start = max(0, num_rounds - 30)

    # ── ITT (intention-to-treat): every round counts, including rounds whose arm
    # came from the invalid-parse fallback. This is the headline metric and the
    # one comparable to the pre-2026-07-28 numbers in shape, because a fallback
    # arm still drew a reward and still entered the next prompt's history.
    opt_frac        = sum(1 for c in choices if c == opt) / num_rounds
    explore_rate    = 1.0 - opt_frac
    worst_frac      = sum(1 for c in choices if c == worst) / num_rounds
    cum_regret      = float(sum(arm_map[opt] - arm_map[c] for c in choices))
    early_opt_frac  = sum(1 for c in choices[:early_end] if c == opt) / early_end
    late_opt_frac   = sum(1 for c in choices[late_start:] if c == opt) / (num_rounds - late_start)
    invalid_rate    = sum(invalids) / num_rounds

    # ── VALID-ONLY: restricted to rounds the model actually chose. Necessary
    # because at high |α| a large share of "choices" are the random fallback, so
    # ITT mixes model behaviour with a uniform-random agent. Read the two
    # together: valid-only is not automatically the truer number — dropping
    # invalid rounds positively selects the rounds the model could still format,
    # the same survivor bias seen in the betting +8 cell — so report
    # invalid_rate alongside as the format/engagement failure measure.
    vidx = [i for i, v in enumerate(valid_flags) if v]
    n_valid = len(vidx)
    if n_valid:
        vchoices = [choices[i] for i in vidx]
        valid_opt_frac = sum(1 for c in vchoices if c == opt) / n_valid
        valid_worst_frac = sum(1 for c in vchoices if c == worst) / n_valid
        valid_mean_regret = float(np.mean([arm_map[opt] - arm_map[c] for c in vchoices]))
    else:
        valid_opt_frac = valid_worst_frac = valid_mean_regret = float("nan")

    # Format-failure breakdown: 0 names found vs >1 (menu restatement). The old
    # permissive parser scored the latter as a valid pick of the first-listed arm.
    n_zero_match = sum(1 for m in n_matched if m == 0)
    n_multi_match = sum(1 for m in n_matched if m > 1)

    # ── DISCOVERY vs UTILIZATION ──────────────────────────────────────────
    # opt_frac silently multiplies two different abilities:
    #     did it ever try the best arm?   →  once tried, did it use it?
    #           DISCOVERY                          UTILIZATION
    # On the 20-seed α=0 run these separate almost perfectly: the 6 seeds that
    # never tried the best arm scored EXACTLY 0.000, the other 14 averaged
    # 0.434. A composite that is 0 for a third of runs cannot carry a dose
    # response — α would only be shifting the odds of a near-binary event — so
    # the two are recorded separately and opt_frac drops to secondary.
    # ⚠ ITT vs VALID-ONLY. `choices` includes rounds where the parse failed and
    # the arm came from the uniform-random fallback. At α=0 that is harmless
    # (invalid=0.000 on every C/C20 cell), but under an α sweep invalid rises,
    # and a random fallback arm would be scored as if the MODEL had chosen to
    # explore it or to exploit it — manufacturing coverage and discovery out of
    # noise, in the direction that flatters whichever α breaks format most.
    # So each metric is computed BOTH ways: `*_itt` over all rounds (comparable
    # in shape to the older numbers) and the unsuffixed name over valid rounds
    # only, which is what a behavioural claim must cite.
    vset = set(vidx)
    vchoices_seq = [choices[i] for i in sorted(vset)]

    coverage_itt = len(set(choices))
    coverage = len(set(vchoices_seq))
    first_best_trial_itt = next((i for i, c in enumerate(choices) if c == opt), None)
    first_best_trial = next((i for i, c in enumerate(choices)
                             if c == opt and i in vset), None)
    best_never_tried = first_best_trial is None

    # Utilization, primary: of the rounds where the model picked an ALREADY-TRIED
    # arm, how often was it one of the arms with the highest observed mean AT
    # THAT MOMENT? Computed from the model's own information state, so it never
    # credits or blames discovery. Ties all count as adherent (several arms can
    # legitimately share the top observed mean, especially early). Rounds that
    # pick an UNTRIED arm are exploration and are excluded — not failures.
    # Round 0 has no observed means at all and is undefined by construction.
    #
    # A fallback round is SKIPPED FOR SCORING but still UPDATES the information
    # state: the model really was shown that arm's reward in the next prompt, so
    # dropping it from the running means would score later rounds against a
    # history the model never saw.
    def _adherence(score_only_valid):
        seen_n, seen_sum = {}, {}
        hit = tot = expl = 0
        for i, (c, r) in enumerate(zip(choices, feedbacks)):
            scoreable = (not score_only_valid) or (i in vset)
            if seen_n and scoreable:
                means = {a: seen_sum[a] / seen_n[a] for a in seen_n}
                top = max(means.values())
                if c in means:
                    tot += 1
                    # float tolerance: means are k/n ratios, so exact ties are
                    # common and must not be lost to representation error.
                    hit += (means[c] >= top - 1e-9)
                else:
                    expl += 1
            seen_n[c] = seen_n.get(c, 0) + 1
            seen_sum[c] = seen_sum.get(c, 0.0) + r
        return (hit / tot if tot else float("nan")), tot, expl

    empirical_best_adherence, adhere_tot, explore_untried = _adherence(True)
    empirical_best_adherence_itt, _, _ = _adherence(False)

    # Utilization, secondary: best-arm rate in a FIXED 20-round window after
    # discovery. Fixed width because a variable window (first_best_trial..end)
    # has a denominator the intervention itself moves — finding the best arm
    # earlier lengthens the window and adds noisy early rounds, so the metric
    # would fall even with utilization unchanged. Undefined (NOT 0) when the arm
    # was never tried or was found too late to fill the window; filling those
    # with 0 would re-mix discovery back in.
    POST_WIN = 20
    if first_best_trial is not None and first_best_trial + 1 + POST_WIN <= num_rounds:
        seg = [(i, choices[i]) for i in
               range(first_best_trial + 1, first_best_trial + 1 + POST_WIN)]
        segv = [c for i, c in seg if i in vset]
        post_discovery_opt_frac = (sum(1 for c in segv if c == opt) / len(segv)
                                   if segv else float("nan"))
    else:
        post_discovery_opt_frac = float("nan")

    return {
        "seed":           seed,
        "arm_map":        {k: v for k, v in arm_map.items()},
        "best_arm":       opt,
        "worst_arm":      worst,
        # 1-based display position of the best arm — lets the analysis verify
        # position counterbalancing and test for residual first-option bias.
        "best_position":  arm_names.index(opt) + 1,
        "arm_order":      arm_names,
        # the literal model input (post chat-template, post prefill) — see above
        "prompt_attest":  prompt_attest,
        "choices":        choices,
        "feedbacks":      feedbacks,
        "valid_flags":    valid_flags,
        "n_matched":      n_matched,
        "raws":           raws,
        "invalid_rate":    float(invalid_rate),
        "zero_match_rate": float(n_zero_match / num_rounds),
        "multi_match_rate": float(n_multi_match / num_rounds),
        "opt_frac":        float(opt_frac),
        "explore_rate":    float(explore_rate),
        "worst_frac":      float(worst_frac),
        "cum_regret":      float(cum_regret),
        "early_opt_frac":  float(early_opt_frac),
        "late_opt_frac":   float(late_opt_frac),
        "n_valid":            n_valid,
        "valid_opt_frac":     float(valid_opt_frac),
        "valid_worst_frac":   float(valid_worst_frac),
        "valid_mean_regret":  float(valid_mean_regret),
        # ── discovery / utilization split (opt_frac is now a composite) ──
        # Unsuffixed = VALID-ONLY (cite these); *_itt = all rounds incl. the
        # random fallback (comparable in shape to older numbers). Identical
        # whenever invalid_rate == 0, which holds for every α=0 cell so far.
        "coverage":                 int(coverage),
        "coverage_itt":             int(coverage_itt),
        "best_never_tried":         bool(best_never_tried),
        # 0-BASED round index (0 = the first round), None = censored / never
        # tried. Do NOT impute a number for censored runs.
        "first_best_index":         (None if first_best_trial is None
                                     else int(first_best_trial)),
        "first_best_index_itt":     (None if first_best_trial_itt is None
                                     else int(first_best_trial_itt)),
        "empirical_best_adherence": float(empirical_best_adherence),
        "empirical_best_adherence_itt": float(empirical_best_adherence_itt),
        "n_adherence_rounds":       int(adhere_tot),
        "n_explore_untried":        int(explore_untried),
        # NaN = undefined (never tried, or found too late for a full window).
        "post_discovery_opt_frac":  float(post_discovery_opt_frac),
    }


def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)
    print(f"Reward probs (assigned after shuffle): {REWARD_PROBS_ORDERED}")
    print(f"Arm names pool (first {K}): {CLOTHES_NAMES[:K]}")

    FIELDNAMES = [
        "model", "size", "alpha", "start", "end", "TOP",
        "num_runs", "num_rounds", "iface",
        "mean_opt_frac", "mean_explore_rate", "mean_worst_frac",
        "mean_cum_regret", "mean_early_opt_frac", "mean_late_opt_frac",
        "mean_invalid_rate",
        "std_opt_frac", "std_explore_rate", "std_worst_frac",
        "std_cum_regret", "std_early_opt_frac", "std_late_opt_frac",
        "std_invalid_rate",
        # valid-only (fallback rounds excluded) + format-failure breakdown
        "mean_valid_opt_frac", "std_valid_opt_frac",
        "mean_valid_worst_frac", "mean_valid_mean_regret",
        "mean_zero_match_rate", "mean_multi_match_rate",
        "mean_best_position",
        # discovery / utilization split — opt_frac is now a composite secondary
        "mean_coverage", "best_never_tried_frac",
        "mean_empirical_best_adherence",
        "mean_post_discovery_opt_frac", "n_post_discovery_defined",
    ]

    # Seed list. Default = 0..num_runs-1 (legacy behaviour, seed == run index).
    # --seeds pins an explicit set; num_runs then follows the list length so the
    # resume key and the CSV stay consistent.
    if args.seeds:
        seed_list = [int(s) for s in args.seeds]
        args.num_runs = len(seed_list)
    else:
        seed_list = list(range(args.num_runs))
    bp_counts = {}
    for s in seed_list:
        bp_counts[position_of_best(s)] = bp_counts.get(position_of_best(s), 0) + 1
    print(f"[Seeds] {seed_list}  best-arm positions: {dict(sorted(bp_counts.items()))}")

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}.csv")

    # Resume key includes the settings that change what a row MEANS. Keying on
    # alpha alone silently skipped a cell when the layer range, run count or
    # round count changed within the same output dir — you would get the old
    # row back and never notice the new configuration had not run.
    # The interface fields are part of the key too: A/B/C write to separate
    # dirs so the launcher is safe either way, but reusing ONE ans_file while
    # changing --summary_history/--answer_anchor/--use_chat/--seeds would
    # otherwise return the old row and silently skip the new configuration.
    def _resume_key(alpha, st, en, nruns, nrounds, iface=None):
        return (float(alpha), int(st), int(en), int(nruns), int(nrounds),
                iface if iface is not None else _iface_tag())

    def _iface_tag():
        # Version is per-protocol: only --untried_semantics rewrites prompt
        # text, so only it advances to pv2. Every legacy combination stays on
        # pv1 and keeps resuming against rows written before this change.
        #
        # NOTE the pv1 branch emits NO `ut` segment. Adding `ut0` would have
        # been the natural symmetric thing to do and is WRONG: rows already on
        # disk read `pv1sh1aa1ch1nr1mt24sd…`, so an extra segment makes every
        # stored cell look unrun and silently re-runs it. The version prefix
        # already distinguishes the two protocols, so `ut` is redundant — a
        # new flag may only extend the key for the configurations it newly
        # creates, never for ones that already have stored rows.
        if args.untried_semantics:
            return (f"{PROTOCOL_VERSION_UNTRIED}"
                    f"sh{int(args.summary_history)}aa{int(args.answer_anchor)}"
                    f"ch{int(args.use_chat)}nr{int(args.no_role)}"
                    f"ut1"
                    f"mt{int(args.max_new_tokens)}"
                    f"sd{'-'.join(str(s) for s in seed_list)}")
        return (f"{PROTOCOL_VERSION_LEGACY}"
                f"sh{int(args.summary_history)}aa{int(args.answer_anchor)}"
                f"ch{int(args.use_chat)}nr{int(args.no_role)}"
                f"mt{int(args.max_new_tokens)}"
                f"sd{'-'.join(str(s) for s in seed_list)}")

    def _legacy_iface_tag(r):
        """Interface tag for a CSV row written before the flags existed.

        Such a row is by definition the legacy interface (raw history, substring
        parser, no chat) with seeds 0..num_runs-1 and the then-default
        max_new_tokens=20; --no_role is recoverable only from the dir, so it is
        taken from the current args (the launcher never mixes role settings
        inside one ans_file). Legacy rows are pinned to pv0 — the pre-versioning
        protocol — so bumping PROTOCOL_VERSION never silently revalidates them.
        """
        n = int(r["num_runs"])
        return (f"pv0sh0aa0ch0nr{int(args.no_role)}mt20"
                f"sd{'-'.join(str(s) for s in range(n))}")

    # A CSV written before the `iface` column existed has a header one field
    # short. Appending current-schema rows to it would put a value under no
    # column name, and the NEXT DictReader would silently drop it into the
    # None key — the resume logic would then re-run cells that were already
    # done. Migrate the file in place instead: re-write it with the current
    # header, filling the missing field from the legacy interface. The old
    # file is kept as .bak because this rewrites data the run depends on.
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            old_reader = csv.DictReader(f)
            old_header = old_reader.fieldnames or []
            missing = [c for c in FIELDNAMES if c not in old_header]
            old_rows = list(old_reader) if missing else []
        if missing:
            print(f"[Migrate] {csv_path} header lacks {missing}; rewriting "
                  f"({len(old_rows)} rows kept, .bak saved).")
            shutil.copy2(csv_path, csv_path + ".bak")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES,
                                   extrasaction="ignore")
                w.writeheader()
                for r in old_rows:
                    if not r.get("iface"):
                        try:
                            r["iface"] = _legacy_iface_tag(r)
                        except (KeyError, ValueError, TypeError):
                            r["iface"] = ""
                    w.writerow({k: r.get(k, "") for k in FIELDNAMES})

    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    # `iface` column exists only in rows written after
                    # 2026-07-29. A row without it predates the interface flags,
                    # so it can only have been the legacy interface.
                    iface = r.get("iface") or _legacy_iface_tag(r)
                    done_keys.add(_resume_key(r["alpha"], r["start"], r["end"],
                                              r["num_runs"], r["num_rounds"],
                                              iface))
                except (KeyError, ValueError):
                    # Row from before the key was widened at all: fall back to
                    # alpha-only so old dirs still resume rather than crash.
                    done_keys.add((float(r["alpha"]),))
        print(f"[Resume] {len(done_keys)} cells already done, skipping.")

    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        key = _resume_key(alpha, st, en, args.num_runs, args.num_rounds)
        if key in done_keys or (float(alpha),) in done_keys:
            print(f"[Skip] α={alpha} layers={st}-{en} "
                  f"runs={args.num_runs}x{args.num_rounds} already done.")
            continue

        # Seeds: default 0..num_runs-1 (legacy). --seeds overrides with an
        # explicit list, which is how the counterbalanced task-validity pilot
        # pins the best arm to positions 1-5 with 5 distinct best names.
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        raw_mask = np.load(mask_path)
        diff_mtx = list(raw_mask * alpha)
        TOP = max(1, int(args.percentage / 100 * raw_mask.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")

        run_results = []
        for run_idx, seed in enumerate(seed_list):
            print(f"  Run {run_idx + 1}/{len(seed_list)} (seed={seed})", end=" ... ", flush=True)
            with torch.no_grad():
                result = run_episode(
                    vc=vc,
                    diff_mtx=diff_mtx,
                    num_rounds=args.num_rounds,
                    seed=seed,
                    use_role=not args.no_role,
                    summary_history=args.summary_history,
                    answer_anchor=args.answer_anchor,
                    use_chat=args.use_chat,
                    max_new_tokens=args.max_new_tokens,
                    untried_semantics=args.untried_semantics,
                )
            run_results.append(result)
            print(
                f"opt={result['opt_frac']:.2f}  "
                f"validopt={result['valid_opt_frac']:.2f}  "
                f"worst={result['worst_frac']:.2f}  "
                f"regret={result['cum_regret']:.1f}  "
                f"invalid={result['invalid_rate']:.2f}"
                f"(0m={result['zero_match_rate']:.2f}/multi={result['multi_match_rate']:.2f})  "
                f"bestpos={result['best_position']}"
            )
            gc.collect()
            torch.cuda.empty_cache()

        opt   = [r["opt_frac"]       for r in run_results]
        expl  = [r["explore_rate"]   for r in run_results]
        wf    = [r["worst_frac"]     for r in run_results]
        regr  = [r["cum_regret"]     for r in run_results]
        eopt  = [r["early_opt_frac"] for r in run_results]
        lopt  = [r["late_opt_frac"]  for r in run_results]
        inv   = [r["invalid_rate"]   for r in run_results]
        vopt  = [r["valid_opt_frac"]    for r in run_results]
        vwf   = [r["valid_worst_frac"]  for r in run_results]
        vregr = [r["valid_mean_regret"] for r in run_results]
        zm    = [r["zero_match_rate"]   for r in run_results]
        mm    = [r["multi_match_rate"]  for r in run_results]
        bpos  = [r["best_position"]     for r in run_results]
        cov   = [r["coverage"]          for r in run_results]
        bnt   = [r["best_never_tried"]  for r in run_results]
        adh   = [r["empirical_best_adherence"] for r in run_results]
        # np.nanmean over an all-NaN slice warns and returns NaN; both these
        # metrics are legitimately undefined for some runs (censored discovery),
        # so guard rather than impute.
        pdo   = [r["post_discovery_opt_frac"]  for r in run_results]
        n_pdo = int(np.sum(~np.isnan(pdo)))

        row = {
            "model": args.model,
            "size":  args.size,
            "alpha": alpha,
            "start": st,
            "end":   en,
            "TOP":   TOP,
            "num_runs":   args.num_runs,
            "num_rounds": args.num_rounds,
            "iface":      _iface_tag(),
            "mean_opt_frac":        round(float(np.mean(opt)),  4),
            "mean_explore_rate":    round(float(np.mean(expl)), 4),
            "mean_worst_frac":      round(float(np.mean(wf)),   4),
            "mean_cum_regret":      round(float(np.mean(regr)), 4),
            "mean_early_opt_frac":  round(float(np.mean(eopt)), 4),
            "mean_late_opt_frac":   round(float(np.mean(lopt)), 4),
            "mean_invalid_rate":    round(float(np.mean(inv)),  4),
            "std_opt_frac":         round(float(np.std(opt)),   4),
            "std_explore_rate":     round(float(np.std(expl)),  4),
            "std_worst_frac":       round(float(np.std(wf)),    4),
            "std_cum_regret":       round(float(np.std(regr)),  4),
            "std_early_opt_frac":   round(float(np.std(eopt)),  4),
            "std_late_opt_frac":    round(float(np.std(lopt)),  4),
            "std_invalid_rate":     round(float(np.std(inv)),   4),
            # nanmean: a run with 0 valid rounds yields NaN valid-only metrics
            "mean_valid_opt_frac":    round(float(np.nanmean(vopt)),  4),
            "std_valid_opt_frac":     round(float(np.nanstd(vopt)),   4),
            "mean_valid_worst_frac":  round(float(np.nanmean(vwf)),   4),
            "mean_valid_mean_regret": round(float(np.nanmean(vregr)), 4),
            "mean_zero_match_rate":   round(float(np.mean(zm)),       4),
            "mean_multi_match_rate":  round(float(np.mean(mm)),       4),
            # sanity: should sit near (K+1)/2 = 3.0 once positions are
            # counterbalanced. A value pinned at 1.0 means position leakage.
            "mean_best_position":     round(float(np.mean(bpos)),     4),
            # ── discovery / utilization (see run_episode) ──
            "mean_coverage":            round(float(np.mean(cov)), 4),
            "best_never_tried_frac":    round(float(np.mean(bnt)), 4),
            "mean_empirical_best_adherence": round(float(np.nanmean(adh)), 4),
            # NaN-safe: undefined for censored runs, so the denominator is
            # reported alongside rather than silently shrinking.
            "mean_post_discovery_opt_frac": (round(float(np.nanmean(pdo)), 4)
                                             if n_pdo else ""),
            "n_post_discovery_defined":  n_pdo,
        }

        writer.writerow(row)
        csv_file.flush()

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        detail_path = os.path.join(out_dir, f"bandit_{args.size}_{TOP}_{st}_{en}.json")
        with open(detail_path, "w", encoding="utf-8") as fw:
            json.dump({
                "alpha": alpha,
                "reward_probs_ordered": REWARD_PROBS_ORDERED,
                "arm_names_pool": CLOTHES_NAMES[:K],
                # Which prompt/parser variant produced this file. Absent in
                # pre-2026-07-29 runs (= bare raw-history + substring parser).
                # The rendered prompts are NOT here: they are per-run, under
                # runs[i]["prompt_attest"], and are the literal model input
                # (chat template + `Choice: ` prefill applied). Do not rebuild
                # an approximation from these flags and call it the prompt.
                "config": {
                    "seeds":           seed_list,
                    "summary_history": bool(args.summary_history),
                    "answer_anchor":   bool(args.answer_anchor),
                    "use_chat":        bool(args.use_chat),
                    "no_role":         bool(args.no_role),
                    "max_new_tokens":  int(args.max_new_tokens),
                    "parser":          "strict_anchor" if args.answer_anchor else "substring",
                    "untried_semantics": bool(args.untried_semantics),
                    "protocol_version": (PROTOCOL_VERSION_UNTRIED
                                         if args.untried_semantics
                                         else PROTOCOL_VERSION_LEGACY),
                },
                "runs": run_results,
            }, fw, indent=2)
        print(f"  → {detail_path}")

        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    print("\n✅ Bandit Task run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Armed Bandit with RSN steering (EVOLvE design).")
    parser.add_argument("--model",       type=str, default="llama3")
    parser.add_argument("--model_dir",   type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs",          type=str, default="llama3")
    parser.add_argument("--size",        type=str, default="8B")
    parser.add_argument("--type",        type=str, default="non")
    parser.add_argument("--percentage",  type=float, default=0.5)
    parser.add_argument("--mask_type",   type=str, default="nmd")
    parser.add_argument("--abs",         action="store_true")
    parser.add_argument("--no_role",     action="store_true",
                        help="Use the No-Role neutral prompt (strips the AI-fashion-assistant persona; §4.7 No-Role condition).")
    parser.add_argument("--configs",     nargs="+", default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--num_runs",    type=int, default=30)
    parser.add_argument("--num_rounds",  type=int, default=50)
    parser.add_argument("--seeds",       nargs="+", default=None,
                        help="Explicit run seeds (overrides --num_runs). The "
                             "counterbalanced task-validity set is "
                             "'0 3 4 9 37': best arm at display positions "
                             "2,4,5,3,1 with 5 distinct best names. Default "
                             "0..num_runs-1 leaves position only randomly "
                             "balanced (seeds 0 and 1 collide on BOTH best "
                             "name and position, which is why the 2-run pilot "
                             "was not position-balanced).")
    parser.add_argument("--summary_history", action="store_true",
                        help="EVOLvE SummaryHistory: show per-arm pull count + "
                             "mean reward instead of the raw interaction log.")
    parser.add_argument("--answer_anchor", action="store_true",
                        help="Prefill 'Choice: ' and parse strictly (exactly "
                             "one arm name on the Choice line); lists/code/"
                             "multiple names count as invalid.")
    parser.add_argument("--untried_semantics", action="store_true",
                        help="pv2 task-representation prompt: render n=0 arms as "
                             "UNTRIED (not 'average reward 0.00'), state that "
                             "reward probabilities are fixed but unknown, show "
                             "round/horizon, and state that names/positions are "
                             "arbitrary. Repairs what the model is TOLD without "
                             "prescribing a strategy — no forced initialization, "
                             "no explore/exploit round split — so exploration "
                             "remains the model's own decision. Implies pv2 in "
                             "the resume key.")
    parser.add_argument("--use_chat",    action="store_true",
                        help="Wrap the prompt in the chat template. NOTE this "
                             "moves steering off the bare distribution the NMD "
                             "mask was extracted in — intended for the α=0 "
                             "task-validity pilot.")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="20 suits the bare 'name only' protocol; raise for "
                             "--answer_anchor if replies get truncated.")
    parser.add_argument("--ans_file",    type=str, default="answer_bandit")
    parser.add_argument("--data",        type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir",    type=str, default=None)

    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)
    print(f"Rounds: {args.num_rounds}, Runs: {args.num_runs}")

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/RolePlaying/components"

    MASK_DIR  = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
