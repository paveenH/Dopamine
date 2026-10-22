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

# pv6 F-reference (see --reference_environment). Imported unconditionally but
# used only on the pv6 branch; neither module imports this one, so the
# dependency stays one-way and pv1-pv5 behaviour is untouched.
import bandit_reference
import bandit_pv6_episode as pv6_episode

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

# K<5 probability tables for the capability-ladder step (K=2/K=3 in
# BanditExperiment_LiteratureReview.md §4.1). K=5 here is REWARD_PROBS_ORDERED
# itself (not a duplicate) so there is exactly one place either table can be
# edited from.
REWARD_PROBS_BY_K = {
    2: [0.7, 0.3],
    3: [0.7, 0.45, 0.2],
    5: REWARD_PROBS_ORDERED,
}

# Round index (0-based) at or after which a first best-arm pull counts as "too
# late to pay off" — see `late_best_discovery`. 30 of 50 leaves 20 rounds, the
# same horizon the fixed post-discovery window uses. Arbitrary but frozen: it
# is a descriptive threshold, not a test statistic, so it must not be tuned
# after seeing a result.
LATE_DISCOVERY_ROUND = 30


def shuffle_arms(seed: int, num_arms: int = 5) -> dict[str, float]:
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

    `num_arms` (2026-07-30, capability-ladder step): default 5 is BYTE-
    IDENTICAL to every stored C/D/D2/D3 result — `Random.shuffle` consumes RNG
    state based only on the shuffled list's length, so slicing CLOTHES_NAMES /
    REWARD_PROBS_BY_K to 5 elements before shuffling reproduces the exact
    original draw sequence. Do not reorder the two shuffle calls or slice
    differently; either would break that equivalence silently (no error, just
    different arm_maps for existing seeds).
    """
    rng = random.Random(seed)
    names = CLOTHES_NAMES[:num_arms]
    probs = REWARD_PROBS_BY_K[num_arms]
    # 1) which name gets which reward probability
    assign = names[:]
    rng.shuffle(assign)
    name_to_prob = {name: prob for name, prob in zip(assign, probs)}
    # 2) independently, the order the names are DISPLAYED in
    display = names[:]
    rng.shuffle(display)
    return {name: name_to_prob[name] for name in display}


def position_of_best(seed: int, num_arms: int = 5) -> int:
    """1-based display position of the best arm for `seed` (counterbalance check)."""
    am = shuffle_arms(seed, num_arms=num_arms)
    probs = list(am.values())
    return probs.index(max(probs)) + 1


def best_arm(arm_map: dict[str, float]) -> str:
    return max(arm_map, key=arm_map.get)


def worst_arm(arm_map: dict[str, float]) -> str:
    return min(arm_map, key=arm_map.get)


def summarize_history(arm_names: list[str],
                      history: list[tuple[str, int]],
                      untried_semantics: bool = False,
                      trial_counts: bool = False) -> str:
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

    `trial_counts=True` (pv3/D2) additionally renders a tried arm as
    "k rewards / n trials (observed rate r)" instead of "n trials, average
    reward r". Same numbers, different reading order: the count comes FIRST, so
    the evidence strength is visible before the point estimate. This targets the
    residual D failure — the model treats an n=1 mean as a settled property.
    On the 20-seed D run, 6 seeds pulled the best arm ≤1 time and 2 abandoned it
    permanently after a single reward=0; one held an arm measured at 0.16 for
    all 50 rounds. "0 rewards / 1 trial" states the sample size in the same
    breath as the outcome; "1 times, average reward 0.00" buries it.
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
        if trial_counts:
            k = int(round(action_rewards[name]))
            snippet += (f"\n{name}: {k} reward{'' if k == 1 else 's'} / "
                        f"{n} trial{'' if n == 1 else 's'} "
                        f"(observed rate {reward:.2f})")
        elif untried_semantics:
            snippet += f"\n{name}: {n} trials, average reward {reward:.2f}"
        else:
            snippet += (f"\n{name} {ACTION_UNIT}, {n} times, "
                        f"average reward {reward:.2f}")
    return snippet


def summarize_tried_untried(arm_names: list[str],
                            history: list[tuple[str, int]]) -> tuple[str, str]:
    """pv5 (E-direct/E-CoT) state rendering: two SEPARATE blocks instead of one
    table with an UNTRIED row mixed in among tried rows.

    Returns (tried_block, untried_block). Either string is "" if that category
    is empty — the caller decides whether to print the block/header at all
    (round 1 has no tried block; a run that has covered every arm has no
    untried block).

    WHY a structural split rather than a wording fix: D2 already stated
    "UNTRIED != 0" in a full sentence and STILL collapsed to greedy (D2 α=0:
    18/20 seeds picked the empirical-best arm every single valid round after
    the first few). The suspected mechanism is that a single merged table —
    even with correct UNTRIED semantics — visually reads as "options with
    numbers" vs "one different-looking row", so the untried option loses
    salience as soon as ANY arm has a decent observed rate. Separating them
    into two headed blocks makes "you have not looked at these yet" a
    structural fact of the prompt, not a sentence the model has to hold onto
    across the table.

    Tried arms render "k rewards / n trials, rate r" — D2's ordering (count
    before rate) is kept, since nothing in the D2 diagnosis implicated it.
    Untried arms are listed by NAME ONLY, no placeholder numbers at all.
    """
    n_actions = {name: 0 for name in arm_names}
    action_rewards = {name: 0.0 for name in arm_names}
    for name, reward in history:
        n_actions[name] += 1
        action_rewards[name] += reward
    tried_lines, untried_lines = [], []
    for name in arm_names:
        n = n_actions[name]
        if n == 0:
            untried_lines.append(f"- {name}")
            continue
        k = int(round(action_rewards[name]))
        rate = action_rewards[name] / n
        tried_lines.append(
            f"- {name} — {n} trial{'' if n == 1 else 's'}, "
            f"{k} reward{'' if k == 1 else 's'}, rate {rate:.2f}"
        )
    return "\n".join(tried_lines), "\n".join(untried_lines)


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
#   pv3 — 2026-07-29, --prompt_variant D2. pv2 + SAMPLING UNCERTAINTY as an
#         environment fact: rewards are random draws, few-trial estimates are
#         unreliable, and the summary reports "k rewards / n trials (observed
#         rate r)" so trial count is read before the rate. Also renames
#         "This is choice N of 50" → "Round N of 50" (the old wording induced
#         list-continuation, "1. ...", which produced most of D's 9 invalids,
#         nearly all in round 1).
#   pv4 — 2026-07-29, --prompt_variant D3. pv3 + two TIMING-STRATEGY sentences
#         (explore early / weight the best-supported option as rounds run out).
#         DIAGNOSTIC CONTROL ONLY, never the main line: those sentences supply
#         the explore→exploit schedule externally, and when to stop exploring is
#         exactly the behaviour α is hypothesised to move. If D3 succeeds where
#         D2 fails, the deficit is strategic planning, not task comprehension —
#         but an α sweep must then run on D2, not D3 (cf. IGT v4).
#
#   pv5 — 2026-07-30, --prompt_variant E-direct / E-CoT. FROZEN as of the
#         BanditExperiment_LiteratureReview.md redesign; both variants share
#         one version tag because they differ only in output-shape wording
#         (see build_prompt), not in the information given to the model, and
#         both are new — neither has stored rows to protect.
#
#         D→D2 conflated two changes at once (sampling-uncertainty prose AND
#         the tried/untried table layout), so when D2 collapsed to pure greedy
#         (adherence 1.000, 18/20 α=−4 runs byte-identical to α=0) it was not
#         possible to tell which change caused it. pv5 isolates the layout
#         change and DROPS the long epistemics paragraph D2 added:
#           - explicit TRIED / UNTRIED blocks (untried arms listed by name
#             only, never merged into the same table row as tried ones);
#           - tried arms rendered "k rewards / n trials, rate r" (D2's
#             ordering, kept — it is not implicated in the collapse);
#           - the arm menu is NOT repeated a second time before the commit
#             line (D2's line reused it twice for no informational gain);
#           - D's original exploration–exploitation affordance sentence is
#             RESTORED verbatim ("weigh the value of learning about an
#             UNTRIED option against using the best-supported TRIED option") —
#             D2 replaced it with pure epistemics and never re-stated that
#             exploring is a live option, which is the leading suspect for the
#             collapse;
#           - no "no number" instruction: the round-prefix continuation bug
#             (90/99 of D's α=−4 invalids) is instead handled by a tolerant
#             parser (`_strip_round_prefix`, applied before both pv5 parsers)
#             rather than by an instruction that adds another sentence to an
#             already-oversaturated commit line.
#
#         Two output interfaces share this prompt body, differing only after
#         "Observed evidence":
#           E-direct — `Choice: ` prefilled, parsed by parse_choice_exact
#                      (unchanged whole-reply strictness).
#           E-CoT    — NOT prefilled; the model reasons first, then commits on
#                      a final `Choice:` line, parsed by
#                      parse_choice_rationale_final (see that function's
#                      docstring for why parse_choice_exact cannot be reused).
#         Steering token differs by interface — E-direct injects on the
#         trailing space of the prefilled anchor, E-CoT injects on the LAST
#         PROMPT TOKEN before free generation starts (there is no prefill to
#         anchor to). The two are not the same intervention; α comparisons
#         must stay WITHIN an interface, never pooled across E-direct/E-CoT.
PROTOCOL_VERSION_LEGACY = "pv1"
PROTOCOL_VERSION_UNTRIED = "pv2"

# User-facing experiment condition → internal protocol version. The two are
# deliberately separate names: D2/D3/E-* identify the experimental condition
# in the launcher and the writeup, pvN exists only so the resume key changes
# when the prompt text does. Callers pass --prompt_variant, never a pv tag.
PROMPT_VARIANT_VERSION = {"D2": "pv3", "D3": "pv4",
                          "E-direct": "pv5", "E-CoT": "pv5"}


def build_prompt(
    arm_names: list[str],
    history: list[tuple[str, int]],
    use_role: bool = True,
    summary_history: bool = False,
    answer_anchor: bool = False,
    untried_semantics: bool = False,
    round_idx: int = 0,
    num_rounds: int = 50,
    prompt_variant: str | None = None,
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

    prompt_variant="D2" (pv3) → pv2 + SAMPLING UNCERTAINTY, still facts only.
    pv2 fixed what the model knew about UNTRIED arms and the discovery gate
    opened (best-never-tried 6/20 → 1/20), but coverage barely moved (3.55 →
    3.75) and OptFrac stayed bimodal, because a DIFFERENT misreading remained:
    the model treats an observed rate as a settled property regardless of n.
    D2 states three further facts — each reward is a fresh random draw, a
    few-trial result is uncertain in BOTH directions, and trial count is
    evidence strength — and renders the table as "k rewards / n trials". It
    still prescribes no policy: no forced initialization, no round split, no
    "try every option once".

    prompt_variant="D3" (pv4) → D2 + when to explore ("more useful early") and
    when to stop ("as fewer rounds remain, weight the best-supported option").
    ⚠ CONTROL ONLY. Those two sentences are strategy, not environment: they
    hand the model the explore→exploit schedule, and the timing of that switch
    is precisely what α is hypothesised to modulate. Run D3 to diagnose whether
    a D2 failure is planning rather than comprehension; never sweep α on it.
    """
    names_str = "[" + ", ".join(arm_names) + "]"
    if prompt_variant in ("E-direct", "E-CoT"):
        remaining = num_rounds - round_idx
        tried_block, untried_block = summarize_tried_untried(arm_names, history)
        lines = [
            # len(arm_names), NOT the global K — this is the branch that must
            # support K<5 (E-direct/E-CoT are the capability-ladder interface).
            f"You are playing a {num_rounds}-round stochastic decision task "
            f"with {len(arm_names)} options: {names_str}.",
            "",
            f"Round {round_idx + 1} of {num_rounds}. There are {remaining} "
            f"choices remaining, including this choice.",
            "",
            "Each option has a fixed but unknown probability of giving reward "
            "1. Rewards are stochastic, so observed rates are estimates "
            "rather than guarantees.",
            "Option names and positions contain no reward information.",
            "",
        ]
        if tried_block:
            lines += ["TRIED OPTIONS:", tried_block, ""]
        if untried_block:
            lines += ["UNTRIED OPTIONS:", untried_block, ""]
        else:
            # Every arm has been tried at least once — say so rather than
            # printing an empty header, which would read as a formatting slot
            # left blank instead of a fact about the state.
            lines += ["All options have been tried at least once.", ""]
        if untried_block:
            lines += [
                # RESTORED from D verbatim (D2 dropped this sentence when it
                # added the epistemics paragraph, and D2's collapse is the
                # leading suspect for why). This is the one sentence in the
                # whole prompt that states exploring is a live alternative to
                # exploiting. Only printed when an UNTRIED option actually
                # exists — once every arm has been tried, telling the model to
                # weigh "learning about an UNTRIED option" against a nonexistent
                # option is a stale instruction, not a fact about the state.
                "When choosing, weigh the value of learning about an UNTRIED "
                "option against using the best-supported TRIED option.",
                "",
            ]
        if prompt_variant == "E-direct":
            lines.append(
                f"Choose the option expected to maximize total reward across "
                f"the current and remaining rounds. After \"Choice:\", write "
                f"only its exact name from the list above."
            )
            # `Choice: ` is prefilled by the caller (mirrors D/D2's
            # answer_anchor path) — no worked example, no "no number": the
            # round-prefix continuation failure is handled by
            # `_strip_round_prefix` in the parser instead of a prompt
            # instruction, so this line stays purely about the decision.
        else:  # E-CoT
            lines += [
                "Briefly compare the amount of evidence, observed rewards, "
                "and remaining rounds. Use at most two short sentences.",
                "End with exactly one final line:",
                "Choice: <exact option name>",
            ]
            # Deliberately NOT prefilled with "Choice: " — the model must be
            # free to write rationale BEFORE committing. Prefilling here would
            # reproduce E-direct's anchor and give the model nowhere to put
            # the requested comparison.
        return "\n".join(lines)
    if prompt_variant in ("D2", "D3"):
        remaining = num_rounds - round_idx
        lines = [
            f"You are playing a {num_rounds}-round stochastic decision task "
            f"with {K} options: {names_str}.",
            "",
            # "Round N of 50", NOT "This is choice N of 50": the latter reads as
            # an enumerated list item and the model continued the list ("1. ..."),
            # which is where most of D's invalid replies came from — nearly all
            # in round 1, where there is no history to displace the pattern.
            f"Round {round_idx + 1} of {num_rounds}. There are {remaining} "
            f"choices remaining, including this choice.",
            "",
            "Each option has its own fixed but unknown probability of giving "
            "reward 1.",
            "Each observed reward is a new random outcome: the same option may "
            "sometimes give 1 and sometimes give 0.",
            "",
            # Symmetric on purpose. A one-sided version ("a 0 does not mean it is
            # poor") would be a nudge toward exploring; stating that a single 1
            # is equally uninformative keeps it a fact about sampling rather than
            # a hint about what to do next.
            "A result based on only a few trials is uncertain. One reward of 0 "
            "does not show that an option is poor, and one reward of 1 does not "
            "show that it is best. When comparing options, consider both the "
            "observed reward rate and the number of trials.",
            "",
            "Option names and list positions contain no reward information.",
            "",
            "Observed evidence:",
        ]
        if history:
            lines.append(
                summarize_history(arm_names, history,
                                  untried_semantics=True,
                                  trial_counts=True).lstrip("\n")
            )
        else:
            lines.append("No options have been tried yet.")
        lines += [
            "",
            "An UNTRIED option has no reward estimate yet; it does not have an "
            "estimated reward of zero.",
        ]
        if prompt_variant == "D3":
            # STRATEGY, not environment — the only difference between D2 and D3.
            lines += [
                "Exploring uncertain options is more useful early, because the "
                "information can guide later choices. As fewer rounds remain, "
                "place more weight on the best-supported option.",
            ]
        lines += [
            "",
            # ONE sentence: decision goal + output requirement. The option list
            # is NOT repeated here — it is already in the opening line, and a
            # second copy re-exposes five names right before the commit point
            # for no informational gain.
            #
            # "no number" is evidence-driven, not a stylistic rule: in the D run
            # 90 of 99 α=−4 invalid replies began with a digit EQUAL TO THE
            # CURRENT ROUND ("12: Retro Revival Sneakers" at round 12), i.e. the
            # `Choice: ` prefill was being continued as the enumerated "choice N
            # of 50" phrasing. The "Round N of 50" rename removes the trigger
            # and this removes the behaviour.
            #
            # No worked example ("Choice: Velvet Vogue Jacket"): naming one arm
            # next to the commit point can prime it, and name-choice bias would
            # survive the reward/position shuffle.
            f"Choose the option that will maximize total reward across all "
            f"{num_rounds} rounds. After \"Choice:\", write only its exact name "
            f"from the list above—no number or explanation.",
        ]
        return "\n".join(lines)
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


#  requires an explicit separator char [.():,-] before any following text —
#  bare whitespace after the digit does NOT count as a separator, so
#  "1 option with..." is content (digit directly modifies a word) and is left
#  alone, while "1. Option" / "1: Option" / "1 - Option" are enumeration.
ROUND_PREFIX_RE = re.compile(r"^\s*\d+\s*[.\-):,]\s*")


def _strip_round_prefix(line: str) -> str:
    """Strip a leading enumeration digit ("12. ", "20 - ") from one line, IF
    what remains is non-empty. Targets the D failure where the `Choice: `
    prefill got continued as "choice N of 50" list enumeration (90 of 99
    α=−4 invalid replies at α=−4 began with a digit equal to the current
    round number, e.g. "12: Retro Revival Sneakers" at round 12).

    pv5 fixes the trigger in the prompt (no "This is choice N of 50" phrasing
    to continue) but this is kept as a tolerant SECOND line of defense in the
    parser rather than relying on the prompt fix alone — the prompt fix could
    fail on some α/seed the same way "Round N of 50" alone did not fully
    eliminate leading digits in the D/D2 data (residual ~1% at α=+4 in D2).
    Only strips a digit followed by an explicit separator punctuation mark
    (`.`, `-`, `)`, `:`, `,`); "1 option with the lowest count" has no such
    separator and is left untouched, since a bare space after the digit reads
    as content ("1 option...") rather than enumeration ("1. Option...").
    Operates on a SINGLE LINE — callers must split multi-line text first.
    """
    stripped = ROUND_PREFIX_RE.sub("", line, count=1)
    return stripped if stripped.strip() else line


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
                       rng: random.Random = None,
                       strip_round_prefix: bool = False) -> tuple[str, bool, int]:
    """STRICT `Choice: <arm name>` parser (opt-in via --answer_anchor).

    `strip_round_prefix=False` by default so D/D2 stay byte-identical to
    every existing stored result. pv5 (E-direct) passes True: it drops D2's
    "no number" instruction from the prompt in favor of tolerating the
    continuation here (see `_strip_round_prefix`'s docstring) — only relevant
    on the no-Choice-line-found branch, since that is the only payload that
    can start with a bare digit under the `Choice: ` prefill.

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
        payload = _strip_round_prefix(lines[0]) if strip_round_prefix else lines[0]
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


def parse_choice_rationale_final(
    output: str, arm_names: list[str], rng: random.Random = None
) -> tuple[str, bool, int, str, bool]:
    """Parser for the E-CoT interface: rationale lines, THEN a final `Choice:`.

    NOT a relaxed parse_choice_exact — a genuinely different contract, because
    the two interfaces disagree about what "extra text" means. Under
    answer_anchor's `Choice: ` PREFILL there is no room for rationale before
    the anchor, so parse_choice_exact treats any non-Choice line as a format
    failure (whole-reply strictness). E-CoT is prefilled with NOTHING — the
    model reasons first and commits last — so rationale lines are the expected
    shape, not a violation, and applying parse_choice_exact here would score
    every legitimate reply as invalid.

    Contract:
      - the LAST non-empty line must be exactly `Choice: <arm name>`
        (case-insensitive, trailing punctuation stripped);
      - there must be exactly ONE `Choice:` line in the whole reply, and it
        must BE that last line — a `Choice:` line buried mid-reply with
        further text after it is invalid (the model kept talking after
        committing, so "the last line is the decision" no longer holds);
      - everything before it is rationale, kept verbatim for the audit in
        Step 2 (does it mention UNTRIED arms / trial counts / remaining
        rounds / declare a "best" arm before considering the others).

    Returns (chosen_name, is_valid, n_matched, rationale, menu_restatement):
      - n_matched is computed over the RATIONALE ONLY (not the Choice line),
        so a rationale that legitimately discusses several arms does not by
        itself invalidate the reply — that is expected CoT content here,
        unlike parse_choice_exact where >1 in the whole reply is a violation.
      - menu_restatement is a SEPARATE, stricter signal: True only if the
        rationale names ALL K arms (i.e. reads as reciting the option list
        rather than comparing evidence for a subset). Mentioning 2-3 arms
        while weighing evidence is exactly what "briefly compare" was asked
        for and must not be flagged.
    """
    lows = {name.lower(): name for name in arm_names}
    text = output or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    choice_line_idxs = [i for i, ln in enumerate(lines) if CHOICE_LINE_RE.match(ln)]

    if not lines:
        return (rng or random).choice(arm_names), False, 0, "", False
    if len(choice_line_idxs) != 1 or choice_line_idxs[0] != len(lines) - 1:
        # Zero Choice lines, several, or one that isn't last (model kept
        # talking after committing) — all invalid. n_matched/rationale still
        # computed over the non-Choice lines for diagnostics.
        rationale_lines = [ln for i, ln in enumerate(lines) if i not in
                           set(choice_line_idxs)]
        rationale = "\n".join(rationale_lines)
        mentioned = {name for low, name in lows.items() if low in rationale.lower()}
        return ((rng or random).choice(arm_names), False, len(mentioned),
                rationale, len(mentioned) == len(arm_names))

    rationale = "\n".join(lines[:-1])
    payload = CHOICE_LINE_RE.match(lines[-1]).group(1)
    payload = payload.strip().strip(".,;:!?\"'`*[]()").strip()
    mentioned = {name for low, name in lows.items() if low in rationale.lower()}
    menu_restatement = len(mentioned) == len(arm_names)

    pl = payload.lower()
    if pl in lows:
        return lows[pl], True, len(mentioned), rationale, menu_restatement
    # Committed line doesn't exactly match an arm name -> invalid regardless
    # of how clean the rationale was.
    return ((rng or random).choice(arm_names), False, len(mentioned),
            rationale, menu_restatement)


def audit_rationale(rationale: str, arm_names: list[str], tried: set[str]) -> dict:
    """Descriptive rationale features for the E-CoT mechanism audit (Step 2).

    NOT a validity gate — these never affect `valid`/`is_valid`. They exist so
    a later analysis can ask whether the rationale is actually integrating
    evidence (mentions UNTRIED arms, cites trial counts, reads the horizon) or
    is a post-hoc gloss on a greedy pick that was already decided (declares a
    "best" arm without ever engaging uncertainty) — the D2 failure mode,
    but caught at the reasoning layer instead of only in the final choice.

    `chosen`'s untried/empirical-best/known-worse category is NOT computed
    here — the caller already has the running per-arm means from building the
    summary and should classify it there (this function only reads text).
    Also records whether a rationale was produced at all: a reply that goes
    straight to `Choice: <name>` with no preceding line is parseable and can
    be valid, but it means the model did NOT execute the CoT protocol it was
    asked to (no comparison was written down, whatever happened internally).
    That is itself a finding about the E-CoT interface, not something to
    silently fold into "valid" — Step 2 should report rationale_present rate
    per α before trusting any rationale-content statistic.
    """
    rationale_present = bool(rationale.strip())
    rl = rationale.lower()
    untried = [a for a in arm_names if a not in tried]
    mentions_untried_arm = any(a.lower() in rl for a in untried)
    mentions_untried_word = "untried" in rl
    mentions_uncertainty = bool(re.search(
        r"uncertain|unsure|unknown|estimate|"
        r"not\s+\w*\s*(certain|sure|confident)|"       # "not certain" / "not fully confident"
        r"few trials|small sample|limited (data|evidence|trials)|"
        r"not enough (data|evidence|trials)", rl))
    mentions_trials = bool(re.search(r"\btrial", rl)) or bool(re.search(r"\d+\s*/\s*\d+", rl))
    mentions_remaining = bool(re.search(r"remaining|rounds left|rounds remain", rl))
    declares_best_early = bool(re.search(r"\bbest\b|\bcurrent best\b|\bhighest\b", rl))
    # The D2 failure signature: declares a winner but never engages ANY of the
    # things that would make declaring one premature — an untried arm still
    # exists and the rationale names neither it, the word "untried", nor
    # uncertainty/remaining-rounds in general. Requiring all four absences
    # (not just the untried-arm check) makes this the conservative reading:
    # a rationale that says "I'm not sure yet, but X looks best" is NOT
    # flagged, because it did engage uncertainty even without naming an arm.
    greedy_narration = bool(
        untried and declares_best_early
        and not mentions_untried_arm and not mentions_untried_word
        and not mentions_uncertainty and not mentions_remaining
    )
    return {
        "rationale_present":      rationale_present,
        "rationale_len":          len(rationale.strip()),
        "mentions_untried_arm":   mentions_untried_arm,
        "mentions_untried_word":  mentions_untried_word,
        "mentions_uncertainty":   mentions_uncertainty,
        "mentions_trial_counts":  mentions_trials,
        "mentions_remaining":     mentions_remaining,
        "declares_best_early":    declares_best_early,
        "greedy_narration":       greedy_narration,
    }


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
    prompt_variant: str | None = None,
    debug_tokens: bool = False,
    num_arms: int = 5,
    warm_start_pulls: int = 0,
) -> dict:
    """
    num_arms: size of the arm set (default 5 = every stored C/D/D2/D3 result).
        Forwarded to shuffle_arms; see that function's docstring for the
        byte-identity guarantee at num_arms=5.

    warm_start_pulls (K=5 utilization-only control, per
        BanditExperiment_LiteratureReview.md §4.2A): if >0, the environment
        deterministically pulls each arm exactly `warm_start_pulls` times
        BEFORE the model makes any choice. `num_rounds` is the TOTAL horizon
        (unchanged at 50) — the model gets `num_rounds - num_arms*warm_start_
        pulls` choices, not 50 additional ones. This measures utilization
        only: it CANNOT be read as evidence about autonomous exploration
        (discovery/coverage among model choices is trivially near-ceiling
        because the environment did the discovering), and must never be the
        interface an α sweep runs on — free exploration (warm_start_pulls=0)
        is what α is hypothesised to move.

        Mechanics, all load-bearing (get this wrong and adherence/late-window/
        round-index all silently misalign):
          - warm-start ORDER is shuffled by an INDEPENDENT, deterministic RNG
            (seed offset, no torch/model dependency) so all num_arms*
            warm_start_pulls rounds don't read as "arm A, arm A, arm B, arm
            B, ..." — a fixed block-repeat pattern the model could key off of
            in later rounds' history, though it never sees these rounds
            directly (see below).
          - warm-start REWARDS are drawn from the SAME `rng = Random(seed)`
            stream `get_feedback` uses for the model's own rounds, in warm-
            start order, consuming exactly num_arms*warm_start_pulls draws
            before the model's first round. This is a real cost: it shifts
            every later `rng.random()` call by that many draws relative to a
            warm_start_pulls=0 episode on the same seed, so warm-start and
            free-exploration results on the same seed are NOT reward-draw-
            paired beyond the shift — comparable in distribution, not in the
            exact per-round outcome.
          - the model's round_idx in build_prompt/torch.manual_seed is the
            ABSOLUTE environment round (starts at num_arms*warm_start_pulls,
            not 0) — "Round 11 of 50", not "Round 1 of 50" — so the horizon
            language stays truthful and the per-round torch seed does not
            collide with what a warm_start_pulls=0 run used for its own round
            0-9.
          - the model's FIRST prompt's TRIED block is seeded from the warm-
            start history, so adherence's running per-arm means (seen_n/
            seen_sum in the analysis layer) must be initialized from it too —
            scoring round 11 as if round 1 (no history) would compare the
            model's first choice against an empty information state it never
            had.
          - outcome metrics are computed on TWO views: `model_choices`/
            `model_feedbacks` (this function's own choices only, for a fair
            comparison to warm_start_pulls=0 conditions on model behaviour)
            and the full history the model actually experienced (warm-start +
            model). Both are returned; see the trailing dict.
    """
    rng = random.Random(seed)
    # Separate stream for the invalid-parse fallback. Kept independent of `rng`
    # ON PURPOSE: if both drew from one stream, a condition with more invalid
    # parses (the −α cells, up to 20%) would consume different numbers of draws
    # and desynchronise the REWARD sequence, so α conditions would no longer see
    # comparable reward luck. Two streams keep reward draws aligned per round.
    fallback_rng = random.Random(1_000_000 + seed)
    arm_map = shuffle_arms(seed, num_arms=num_arms)
    arm_names = list(arm_map.keys())   # shuffled order for this run
    opt = best_arm(arm_map)
    worst = worst_arm(arm_map)

    history = []     # list of (name, reward) — includes warm-start if any
    warm_start_history = []   # the warm-start portion alone, kept separate
    if warm_start_pulls > 0:
        # Independent, deterministic stream — never touches `rng` (that stream
        # is reserved for reward draws, consumed below in THIS exact order) or
        # `fallback_rng` (reserved for invalid-parse fallback picks).
        order_rng = random.Random(seed + 700_000)
        pulls = arm_names * warm_start_pulls
        order_rng.shuffle(pulls)
        for name in pulls:
            r = get_feedback(name, arm_map, rng)
            warm_start_history.append((name, r))
        history = list(warm_start_history)
    start_round = num_arms * warm_start_pulls
    if start_round >= num_rounds:
        raise ValueError(
            f"warm_start_pulls={warm_start_pulls} * num_arms={num_arms} = "
            f"{start_round} leaves no rounds for the model (num_rounds="
            f"{num_rounds})")

    choices = []
    feedbacks = []
    invalids = []
    valid_flags = []   # per-round: was the reply a parseable single choice?
    n_matched = []     # per-round: how many distinct arm names appeared
    raws = []          # per-round: full generation (diagnostics)
    # E-CoT only (stay empty [] for every other prompt_variant, so callers can
    # tell "not this interface" from "this interface produced nothing").
    rationales = []          # per-round: text before the committed Choice line
    menu_restatements = []   # per-round: rationale named ALL K arms
    rationale_audits = []    # per-round: audit_rationale(...) dict
    # Verbatim copies of the string actually handed to the model, AFTER the chat
    # wrapper and the `Choice: ` prefill. Round 0 (no history) and a mid-run
    # round (history present) are kept, because those are the two shapes that
    # differ. A rebuilt approximation is not attestation — this is the input.
    prompt_attest = {}

    # round_idx is the ABSOLUTE environment round (0..num_rounds-1) throughout
    # — with warm_start_pulls>0 the loop starts at start_round, not 0, so
    # "Round 11 of 50" is truthful and the per-round torch seed does not
    # collide with a warm_start_pulls=0 run's own round 0-9.
    for round_idx in range(start_round, num_rounds):
        prompt = build_prompt(arm_names, history, use_role=use_role,
                              summary_history=summary_history,
                              answer_anchor=answer_anchor,
                              untried_semantics=untried_semantics,
                              round_idx=round_idx, num_rounds=num_rounds,
                              prompt_variant=prompt_variant)
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
        # E-CoT is the one condition that must NOT get the `Choice: ` prefill:
        # the whole point of the interface is that the model writes rationale
        # BEFORE committing, and prefilling the anchor would pre-empt that
        # exactly like E-direct's anchor does on purpose. answer_anchor is
        # still True for E-CoT (it selects parse_choice_rationale_final below
        # via prompt_variant, not via answer_anchor), so this check must come
        # first.
        if answer_anchor and prompt_variant != "E-CoT":
            # Prefill the anchor so the next token IS the arm name — same
            # 施力点 logic as betting's "Bet: " and CGT's "Answer: ". Without
            # this the anchor is only an instruction and the model can preface
            # it with reasoning that eats the token budget.
            prompt = prompt + ("Choice: " if use_chat else "\nChoice: ")
        # Attest the model's FIRST round (= start_round, which is 0 unless
        # warm-started) and one ~10 rounds later, not literal round_idx==0 —
        # with warm_start_pulls>0 the model never sees round_idx 0 at all, and
        # the un-warm-started case (start_round=0) is unaffected.
        if round_idx == start_round or round_idx == min(start_round + 10, num_rounds - 1):
            prompt_attest[f"round_{round_idx}"] = prompt
        if debug_tokens and round_idx == start_round:
            # Print the token IDs ONCE per cell. Two things are being checked
            # and neither is visible in the decoded string:
            #  (1) DOUBLE BOS. apply_chat_template already emits
            #      <|begin_of_text|> (128000), and llms.regenerate tokenizes
            #      with add_special_tokens defaulting to True, so the head can
            #      read [128000, 128000, ...]. Harmless for the comparisons we
            #      make (D, D2 and every α inside them share the same chain, so
            #      it cancels in any paired contrast) but it must be KNOWN, not
            #      assumed absent.
            #  (2) WHICH TOKEN CARRIES THE STEERING. Injection is prefill-only
            #      into hs[:, -1, :], so the last id here IS the injection site.
            #      With the `Choice: ` prefill that should be 220 (" "), i.e.
            #      the anchor's trailing space — not a chat control token.
            ids = vc.tokenizer(prompt, return_tensors="pt")["input_ids"]
            head, tail = ids[0, :5].tolist(), ids[0, -10:].tolist()
            print(f"\n    [tok] head={head}"
                  f"{'  ⚠ DOUBLE BOS' if head[:2] == [128000, 128000] else ''}")
            print(f"    [tok] tail={tail}   steering token = {tail[-1]}"
                  f" ({vc.tokenizer.decode([tail[-1]])!r})"
                  f"{'' if tail[-1] == 220 else '  ⚠ not 220'}")
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
        if prompt_variant == "E-CoT":
            arm, valid, nmatch, rationale, menu = parse_choice_rationale_final(
                raw, arm_names, rng=fallback_rng)
        elif prompt_variant == "E-direct":
            arm, valid, nmatch = parse_choice_exact(
                raw, arm_names, rng=fallback_rng, strip_round_prefix=True)
            rationale, menu = "", False
        else:
            parser_fn = parse_choice_exact if answer_anchor else parse_choice
            arm, valid, nmatch = parser_fn(raw, arm_names, rng=fallback_rng)
            rationale, menu = "", False

        reward = get_feedback(arm, arm_map, rng)
        choices.append(arm)
        feedbacks.append(reward)
        invalids.append(0 if valid else 1)
        valid_flags.append(bool(valid))
        n_matched.append(nmatch)
        raws.append(raw)
        if prompt_variant == "E-CoT":
            # Audit uses the state BEFORE this round's choice — "tried" means
            # arms already tried entering this round, matching what the model
            # actually saw in the TRIED/UNTRIED blocks it read.
            tried_before = {name for name, _ in history}
            audit = audit_rationale(rationale, arm_names, tried_before)
            rationales.append(rationale)
            menu_restatements.append(bool(menu))
            rationale_audits.append(audit)
        history.append((arm, reward))

    # n_model_rounds = len(choices) = num_rounds - start_round. ALL denominators
    # below use this, not num_rounds — with warm_start_pulls>0 those differ
    # (e.g. 40 vs 50), and dividing by num_rounds would silently deflate every
    # ITT metric by the warm-start fraction. early_end/late_start are also
    # computed relative to the MODEL's own round count, matching how `choices`
    # is indexed (0 = the model's first round, i.e. absolute round
    # start_round) — NOT relative to the absolute environment round.
    n_model_rounds = len(choices)
    early_end = min(20, n_model_rounds)
    late_start = max(0, n_model_rounds - 30)

    # ── ITT (intention-to-treat): every MODEL round counts, including rounds
    # whose arm came from the invalid-parse fallback (a fallback arm still drew
    # a reward and still entered the next prompt's history). Warm-start rounds
    # are NEVER in this denominator — they are not a model decision at all, ITT
    # or otherwise; see model_/total_ outcome split at the end of this function
    # for the two views a warm-started episode needs.
    opt_frac        = sum(1 for c in choices if c == opt) / n_model_rounds
    explore_rate    = 1.0 - opt_frac
    worst_frac      = sum(1 for c in choices if c == worst) / n_model_rounds
    cum_regret      = float(sum(arm_map[opt] - arm_map[c] for c in choices))
    early_opt_frac  = sum(1 for c in choices[:early_end] if c == opt) / early_end
    late_opt_frac   = sum(1 for c in choices[late_start:] if c == opt) / (n_model_rounds - late_start)
    invalid_rate    = sum(invalids) / n_model_rounds

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

    # ---- D2 mechanism metrics --------------------------------------------
    # Three readouts aimed at the SPECIFIC failure D2 repairs. They exist
    # because coverage and best_never_tried are too coarse: a run can score
    # coverage=1 and best_never_tried=False purely by picking the best arm on
    # round 0 and never moving (seeds 1/6/8/18 in the 20-seed D run did exactly
    # that and scored well), which is a lucky first draw, not learning.
    #
    # All three are VALID-ONLY, for the same reason as coverage above: a random
    # fallback pull of the best arm is not the model deciding to try it.
    n_best_pulls = sum(1 for i in sorted(vset) if choices[i] == opt)

    # (a) best arm sampled at most once — the "one look and gone" signature.
    #     D: 6/20 runs. Includes never-tried, so read it with best_never_tried.
    best_arm_pulled_le1 = n_best_pulls <= 1

    # (b) best arm first tried after round 30 — discovered too late for the
    #     remaining rounds to pay for it. D: 3/20 (rounds 36-41).
    late_best_discovery = (first_best_trial is not None
                           and first_best_trial >= LATE_DISCOVERY_ROUND)

    # (c) after the best arm's first pull returned 0, was it ever tried again?
    #     This is the sampling-uncertainty failure in its purest form: the
    #     correct response to one 0 is that the estimate is uninformative, and
    #     seeds 0 and 2 instead abandoned the arm permanently.
    #     None (not False) when the premise does not apply — never tried, or its
    #     first pull returned 1 — so a run that was never at risk is EXCLUDED
    #     from the rate rather than counted as a success. Imputing False would
    #     understate the fix; imputing True would manufacture one.
    if first_best_trial is None or feedbacks[first_best_trial] != 0:
        revisit_after_first_zero = None
    else:
        revisit_after_first_zero = any(choices[i] == opt
                                       for i in sorted(vset)
                                       if i > first_best_trial)

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
    #
    # WARM-START: seen_n/seen_sum must be seeded from warm_start_history before
    # the loop, not left empty. Without this, the model's round 0 in `choices`
    # (= absolute round num_arms*warm_start_pulls, e.g. round 10) would be
    # scored against an EMPTY information state, when the model's own prompt
    # showed it a full TRIED table from the warm-start pulls — comparing its
    # first real choice to a history it never had. With warm_start_pulls=0,
    # warm_start_history is [] and this is a no-op (byte-identical to before).
    def _adherence(score_only_valid):
        seen_n, seen_sum = {}, {}
        for name, r in warm_start_history:
            seen_n[name] = seen_n.get(name, 0) + 1
            seen_sum[name] = seen_sum.get(name, 0.0) + r
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
    if first_best_trial is not None and first_best_trial + 1 + POST_WIN <= n_model_rounds:
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
        # E-CoT only — [] for every other prompt_variant (see the accumulator
        # comment above). rationale_audits holds the audit_rationale() dict
        # per round; kept per-round rather than pre-aggregated so an analysis
        # can slice by round/α/seed without re-parsing raws.
        "rationales":         rationales,
        "menu_restatements":  menu_restatements,
        "rationale_audits":   rationale_audits,
        "invalid_rate":    float(invalid_rate),
        "zero_match_rate": float(n_zero_match / n_model_rounds),
        "multi_match_rate": float(n_multi_match / n_model_rounds),
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
        # D2 mechanism metrics (valid-only). n_best_pulls is kept raw so the
        # ≤1 threshold can be re-cut offline without a re-run.
        "n_best_pulls":             int(n_best_pulls),
        "best_arm_pulled_le1":      bool(best_arm_pulled_le1),
        "late_best_discovery":      bool(late_best_discovery),
        # None = the run was never at risk (best arm never tried, or its first
        # pull returned 1). Exclude those from the rate; do not impute.
        "revisit_after_first_zero": revisit_after_first_zero,
        "empirical_best_adherence": float(empirical_best_adherence),
        "empirical_best_adherence_itt": float(empirical_best_adherence_itt),
        "n_adherence_rounds":       int(adhere_tot),
        "n_explore_untried":        int(explore_untried),
        # NaN = undefined (never tried, or found too late for a full window).
        "post_discovery_opt_frac":  float(post_discovery_opt_frac),
        # ── K / warm-start config + the total-task view ─────────────────
        "num_arms":          num_arms,
        "warm_start_pulls":  warm_start_pulls,
        "n_model_rounds":    int(n_model_rounds),
        "warm_start_history": [(n, int(r)) for n, r in warm_start_history],
        # DISCOVERY/coverage above is NOT the primary readout when warm-started
        # — every arm is trivially "discovered" by construction, so coverage
        # near-ceiling and best_never_tried=False are expected, not evidence of
        # autonomous exploration. The comparable-to-free-exploration metrics
        # here are opt_frac/late_opt_frac/empirical_best_adherence, all already
        # computed on MODEL rounds only (n_model_rounds, matching a
        # warm_start_pulls=0 episode's own denominators).
        #
        # total_* mirrors the same three quantities over the FULL experienced
        # history (warm-start + model), i.e. what actually happened across all
        # num_rounds — useful for a total-task outcome number, but NOT
        # comparable to a warm_start_pulls=0 run's opt_frac (different
        # denominator, and the warm-start rounds are the environment's forced
        # choices, not the model's).
        "total_opt_frac": (
            (sum(1 for n, _ in warm_start_history if n == opt) + opt_frac * n_model_rounds)
            / num_rounds),
        "total_cum_regret": float(
            sum(arm_map[opt] - arm_map[n] for n, _ in warm_start_history)
            + cum_regret),
    }


PV6_FIELDNAMES = [
    "model", "size", "alpha", "start", "end", "TOP",
    "environment", "k", "horizon", "num_runs", "iface",
    # PRIMARY reference metrics (§3.5). SuffFailFreq and K x MinFrac are the
    # two competence-gate quantities; they bracket the failure space (lock-in
    # vs flailing) and must both be read, never one alone.
    "suff_fail_freq_half", "k_min_frac_full", "k_min_frac_half",
    "mean_opt_frac", "mean_late_opt_frac", "mean_cum_regret",
    "mean_greedy_frac",
    "std_opt_frac", "std_late_opt_frac", "std_cum_regret",
    # decision-confidence layer: alpha can move preference without flipping
    # the argmax, which mean_opt_frac alone cannot show.
    "mean_choice_margin",
    # structurally 0 under candidate-only scoring; recorded so its absence is
    # attested rather than assumed.
    "mean_invalid_rate",
]


def _run_pv6_cell(vc, diff_mtx, alpha, st, en, TOP, seed_list,
                  writer, csv_file, SAVE_ROOT, args, iface):
    """One pv6 F-reference cell (all seeds at one alpha).

    Separate from the pv1-pv5 body because pv6's metric set is different: no
    parser-failure metrics (choices are constrained, so invalid is structurally
    0), and the primaries are SuffFailFreq / K x MinFrac rather than opt_frac.
    Model loading, mask/alpha construction, resume and CSV orchestration are
    all reused from the caller — this only replaces the episode + aggregation.
    """
    env = bandit_reference.get_environment(args.reference_environment)
    print(f"    env={env.name} K={env.k} T={env.horizon} probs={env.probs}")

    run_results = []
    for run_idx, seed in enumerate(seed_list):
        print(f"  Run {run_idx + 1}/{len(seed_list)} (seed={seed})",
              end=" ... ", flush=True)
        with torch.no_grad():
            result = pv6_episode.run_reference_episode(
                vc=vc,
                diff_mtx=diff_mtx,
                seed=seed,
                env=env,
                use_chat=args.use_chat,
                steering_scope=args.steering_scope,
                # Attest the first run of each cell: the prompt/token chain is
                # identical across seeds, so storing it 20 times adds nothing.
                attest=(run_idx == 0),
            )
        run_results.append(result)
        print(f"opt={result['opt_frac']:.2f}  "
              f"late={result['late_opt_frac']:.2f}  "
              f"regret={result['cum_regret']:.1f}  "
              f"kmin={result['k_min_frac']:.2f}  "
              f"sufffail={int(result['suffix_failure'])}  "
              f"bestpos={result['best_position']}")
        gc.collect()
        torch.cuda.empty_cache()

    half = env.horizon // 2
    opt = [r["opt_frac"] for r in run_results]
    lopt = [r["late_opt_frac"] for r in run_results]
    regr = [r["cum_regret"] for r in run_results]
    margins = [float(np.nanmean(r["choice_margins"])) for r in run_results]

    row = {
        "model": args.model, "size": args.size, "alpha": alpha,
        "start": st, "end": en, "TOP": TOP,
        "environment": env.name, "k": env.k, "horizon": env.horizon,
        "num_runs": len(seed_list), "iface": iface,
        "suff_fail_freq_half": round(
            bandit_reference.suff_fail_freq(run_results, half), 4),
        "k_min_frac_full": round(
            bandit_reference.mean_k_min_frac(run_results), 4),
        "k_min_frac_half": round(
            bandit_reference.mean_k_min_frac(run_results, half), 4),
        "mean_opt_frac": round(float(np.mean(opt)), 4),
        "mean_late_opt_frac": round(float(np.mean(lopt)), 4),
        "mean_cum_regret": round(float(np.mean(regr)), 4),
        "mean_greedy_frac": round(
            float(np.nanmean([r["greedy_frac"] for r in run_results])), 4),
        "std_opt_frac": round(float(np.std(opt)), 4),
        "std_late_opt_frac": round(float(np.std(lopt)), 4),
        "std_cum_regret": round(float(np.std(regr)), 4),
        "mean_choice_margin": round(float(np.nanmean(margins)), 4),
        "mean_invalid_rate": round(
            float(np.mean([r["invalid_rate"] for r in run_results])), 4),
    }
    writer.writerow(row)
    csv_file.flush()

    out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(
        out_dir, f"bandit_pv6_{env.name}_{args.size}_{TOP}_{st}_{en}.json")
    with open(detail_path, "w", encoding="utf-8") as fw:
        json.dump({
            "alpha": alpha,
            "protocol_version": bandit_reference.PROTOCOL_VERSION,
            "environment": {
                "name": env.name, "k": env.k, "probs": list(env.probs),
                "horizon": env.horizon,
                "is_reference": env.is_reference,
                "competence_eligible": env.competence_eligible,
            },
            # Counterbalance ATTESTATION, not assertion: what this run's seed
            # set actually achieved on both marginals and the cross table.
            "bank_report": bandit_reference.bank_report(
                seed_list, env),
            "config": {
                "seeds": seed_list,
                "use_chat": bool(args.use_chat),
                "rationale_max_tokens": pv6_episode.RATIONALE_MAX_TOKENS,
                "action_anchor": bandit_reference.ACTION_ANCHOR,
                "scoring": "candidate_sequence_logprob",
                # Derived, never hardcoded: this string is what a reader trusts
                # about where alpha landed, so it must track the flag.
                "steering": ("action_pass_only_last_prefill_token"
                             if args.steering_scope == "action"
                             else "rationale_and_action_pass_"
                                  "last_prefill_token_each"),
                "steering_scope": args.steering_scope,
                "steering_scope_version": pv6_episode.STEERING_SCOPE_VERSION,
                # alpha=0 registers no hook in EITHER pass, so the scope is a
                # request that had no effect; record what actually happened.
                "steered_rationale": bool(
                    args.steering_scope == "both" and diff_mtx is not None),
                "temperature": 0.0,
                "iface": iface,
            },
            "runs": run_results,
        }, fw, indent=2)
    print(f"  → {detail_path}")


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
    # pv6 writes a different schema (no parser-failure columns, primaries are
    # SuffFailFreq / K x MinFrac). Its launcher uses a separate output tree, so
    # the two schemas never share a CSV.
    if args.reference_environment:
        FIELDNAMES = PV6_FIELDNAMES

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
        #
        # Same rule for --prompt_variant: D2/D3 are configurations that have no
        # rows on disk, so they get their own `pvar` segment; without the flag
        # the tag is byte-identical to what pv1/pv2 rows already carry.
        #
        # pv5 (E-direct/E-CoT) additionally carries `k{num_arms}ws
        # {warm_start_pulls}` UNCONDITIONALLY — unlike ut/pvar this is not an
        # opt-in extension protecting old rows, because pv5 has no stored data
        # at all yet at introduction; every pv5 cell should self-describe its
        # K and warm-start setting so K=5/warm_start=0 vs K=5/warm_start=2 vs
        # K=2 never collide under one ans_file.
        # pv6 first: it is a different protocol entirely (own environment,
        # horizon and decoding), and it has no stored rows, so it may define
        # its own self-describing tag without the back-compat constraints the
        # branches below carry. env + chat + rationale cap are what change what
        # a pv6 row MEANS; K and horizon follow from the environment.
        if args.reference_environment:
            # The scope segment is APPENDED ONLY for a non-default scope, the
            # same asymmetry as `ut` above and for the same reason: the stored
            # Track A rows carry no scope segment, so emitting `scaction` here
            # would make every completed alpha=0 cell look unrun and silently
            # re-run 7 hours of GPU time. PROTOCOL_VERSION stays pv6 — scope
            # changes the intervention, not the environment/prompt/scoring/
            # seed bank/metrics that define the protocol.
            scope_seg = ""
            if args.steering_scope != pv6_episode.DEFAULT_STEERING_SCOPE:
                scope_seg = (f"sc{args.steering_scope}"
                             f"{pv6_episode.STEERING_SCOPE_VERSION}")
            return (f"{bandit_reference.PROTOCOL_VERSION}"
                    f"env{args.reference_environment}"
                    f"ch{int(args.use_chat)}"
                    f"rt{pv6_episode.RATIONALE_MAX_TOKENS}"
                    f"{scope_seg}"
                    f"sd{'-'.join(str(s) for s in seed_list)}")
        if args.prompt_variant in ("E-direct", "E-CoT"):
            return (f"{PROMPT_VARIANT_VERSION[args.prompt_variant]}"
                    f"sh{int(args.summary_history)}aa{int(args.answer_anchor)}"
                    f"ch{int(args.use_chat)}nr{int(args.no_role)}"
                    f"pvar{args.prompt_variant}"
                    f"k{args.num_arms}ws{args.warm_start_pulls}"
                    f"mt{int(args.max_new_tokens)}"
                    f"sd{'-'.join(str(s) for s in seed_list)}")
        if args.prompt_variant:
            return (f"{PROMPT_VARIANT_VERSION[args.prompt_variant]}"
                    f"sh{int(args.summary_history)}aa{int(args.answer_anchor)}"
                    f"ch{int(args.use_chat)}nr{int(args.no_role)}"
                    f"pvar{args.prompt_variant}"
                    f"mt{int(args.max_new_tokens)}"
                    f"sd{'-'.join(str(s) for s in seed_list)}")
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
                    # pv6 rows have no `num_rounds` column — the horizon is a
                    # property of the environment, not a CLI knob. Reading
                    # r["num_rounds"] would KeyError into the alpha-only
                    # fallback below, which collapses every alpha cell of a
                    # pv6 sweep onto one key and silently skips the rest.
                    nrounds = r.get("num_rounds") or r["horizon"]
                    done_keys.add(_resume_key(r["alpha"], r["start"], r["end"],
                                              r["num_runs"], nrounds,
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

    # pv6's horizon comes from the environment, not --num_rounds, and its CSV
    # stores it as `horizon`. Both sides of the resume key must use the same
    # number or every pv6 cell re-runs on resume.
    _rounds_for_key = (
        bandit_reference.get_environment(args.reference_environment).horizon
        if args.reference_environment else args.num_rounds)

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        key = _resume_key(alpha, st, en, args.num_runs, _rounds_for_key)
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

        if args.reference_environment:
            _run_pv6_cell(vc, diff_mtx if alpha else None, alpha, st, en, TOP,
                          seed_list, writer, csv_file, SAVE_ROOT, args,
                          _iface_tag())
            gc.collect()
            torch.cuda.empty_cache()
            continue

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
                    prompt_variant=args.prompt_variant,
                    num_arms=args.num_arms,
                    warm_start_pulls=args.warm_start_pulls,
                    # Once per cell: the tokenizer chain is identical across
                    # runs, so printing it 20 times says nothing new.
                    debug_tokens=(run_idx == 0),
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
                # Actual table/pool for THIS run's num_arms — not the global
                # K=5 constants. shuffle_arms(seed, num_arms=args.num_arms)
                # assigns REWARD_PROBS_BY_K[args.num_arms] to
                # CLOTHES_NAMES[:args.num_arms]; recording anything else here
                # would silently mislabel a K=2/K=3 run as K=5.
                "reward_probs_ordered": REWARD_PROBS_BY_K[args.num_arms],
                "arm_names_pool": CLOTHES_NAMES[:args.num_arms],
                "num_arms": args.num_arms,
                "warm_start_pulls": args.warm_start_pulls,
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
                    "prompt_variant":  args.prompt_variant,
                    "protocol_version": (
                        PROMPT_VARIANT_VERSION[args.prompt_variant]
                        if args.prompt_variant
                        else (PROTOCOL_VERSION_UNTRIED
                              if args.untried_semantics
                              else PROTOCOL_VERSION_LEGACY)),
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
    parser.add_argument("--num_arms",    type=int, default=5, choices=[2, 3, 5],
                        help="Arm-set size. Default 5 is byte-identical to "
                             "every stored C/D/D2/D3 result (see shuffle_arms). "
                             "2/3 are the capability-ladder step in "
                             "BanditExperiment_LiteratureReview.md §4.1 and "
                             "currently work ONLY with pv5 (--prompt_variant "
                             "E-direct/E-CoT) — the D/D2/D3 prompt branches "
                             "still hard-code the K=5 wording.")
    parser.add_argument("--warm_start_pulls", type=int, default=0,
                        help="K=5 utilization-only control (§4.2A): the "
                             "environment pulls each arm this many times "
                             "before the model chooses at all. 0 (default) = "
                             "free exploration, byte-identical to every "
                             "existing run. >0 measures utilization given "
                             "forced discovery — NOT autonomous exploration, "
                             "and must never be the α-sweep interface.")
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
    parser.add_argument("--prompt_variant",
                        choices=["D2", "D3", "E-direct", "E-CoT"], default=None,
                        help="Experiment condition (supersedes "
                             "--untried_semantics when set). D2 = pv2 + "
                             "sampling-uncertainty FACTS: rewards are random "
                             "draws, few-trial estimates are unreliable in both "
                             "directions, and the table reads 'k rewards / n "
                             "trials (observed rate r)' so evidence strength is "
                             "seen before the point estimate. D3 = D2 + explore-"
                             "early / exploit-late STRATEGY sentences — a "
                             "diagnostic control ONLY, because it supplies the "
                             "explore→exploit timing that α is meant to move; "
                             "sweep α on D2, never on D3. "
                             "E-direct/E-CoT = pv5, the post-D2 redesign: "
                             "structural TRIED/UNTRIED blocks instead of one "
                             "merged table, D's exploration-affordance sentence "
                             "restored, D2's long epistemics paragraph dropped. "
                             "E-direct keeps the `Choice: ` prefill "
                             "(parse_choice_exact + round-prefix tolerance); "
                             "E-CoT is NOT prefilled and requires a brief "
                             "rationale before a final Choice line, parsed by "
                             "parse_choice_rationale_final. The two are "
                             "DIFFERENT interventions (steering token differs) "
                             "— compare α within one interface, never pooled.")
    parser.add_argument("--reference_environment",
                        choices=["easy", "hard", "native_floor"], default=None,
                        help="pv6 F-reference protocol (§3 of "
                             "BanditExperiment_LiteratureReview.md). Selecting "
                             "an environment switches the episode runner to "
                             "bandit_pv6_episode.run_reference_episode and "
                             "IGNORES the pv1-pv5 prompt/parser flags "
                             "(--prompt_variant, --answer_anchor, "
                             "--summary_history, --untried_semantics, "
                             "--num_arms, --warm_start_pulls, "
                             "--max_new_tokens): pv6 has its own frozen prompt, "
                             "K, horizon and two-stage constrained decoding, so "
                             "those flags have no meaning here and are rejected "
                             "rather than silently ignored. easy = K=4 "
                             ".75/.25x3, hard = K=5 .60/.40x4 (both reference "
                             "environments, competence-gate eligible); "
                             "native_floor = K=2 .70/.30, a minimal stochastic-"
                             "adaptation diagnostic that is NOT a competence "
                             "anchor and carries no B1 alpha main experiment.")
    parser.add_argument("--steering_scope",
                        choices=list(pv6_episode.STEERING_SCOPES),
                        default=pv6_episode.DEFAULT_STEERING_SCOPE,
                        help="pv6 ONLY: which passes alpha is injected into. "
                             "'action' (default, the original pv6 semantics) "
                             "steers only the constrained action pass, at the "
                             "last token of the Choice anchor — it measures "
                             "whether alpha moves the arm logits GIVEN the "
                             "rationale, i.e. an action readout. 'both' also "
                             "steers the rationale pass, once at its own last "
                             "prefill token (prefill_only, tail_len=1, decode "
                             "unsteered), giving alpha two routes to the "
                             "choice: alpha->rationale text->action and "
                             "alpha->action logits. 'both' is the main "
                             "information-seeking-policy manipulation; "
                             "'action' is its mechanism ablation. Run them in "
                             "SEPARATE --ans_file dirs: the pv6 detail JSON "
                             "filename carries env/size/layers but NOT scope, "
                             "so one dir would overwrite the other. alpha=0 is "
                             "identical under both scopes (no hook is "
                             "registered either way), so an alpha=0 cell is "
                             "reusable across scopes and need not be re-run.")
    parser.add_argument("--use_chat",    action="store_true",
                        help="Wrap the prompt in the chat template. NOTE this "
                             "moves steering off the bare distribution the NMD "
                             "mask was extracted in, so a steered (α≠0) chat run "
                             "carries a known distribution mismatch: dilution is "
                             "a live explanation for a null, and a POSITIVE "
                             "result is the stronger claim for surviving it "
                             "(same stance as Betting — see CLAUDE.md).")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="20 suits the bare 'name only' protocol; raise for "
                             "--answer_anchor if replies get truncated.")
    parser.add_argument("--ans_file",    type=str, default="answer_bandit")
    parser.add_argument("--data",        type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir",    type=str, default=None)

    args = parser.parse_args()

    # pv6 owns its prompt, K, horizon and decoding, so a pv1-pv5 prompt/parser
    # flag passed alongside it is always a mistake. REJECT rather than ignore:
    # silently ignoring would produce a result file whose config block claims a
    # setting that never took effect.
    if args.reference_environment:
        _pv6_conflicts = [
            n for n, v in (
                ("--prompt_variant",     args.prompt_variant),
                ("--answer_anchor",      args.answer_anchor),
                ("--summary_history",    args.summary_history),
                ("--untried_semantics",  args.untried_semantics),
                ("--warm_start_pulls",   args.warm_start_pulls),
            ) if v
        ]
        if args.num_arms != parser.get_default("num_arms"):
            _pv6_conflicts.append("--num_arms")
        if args.max_new_tokens != parser.get_default("max_new_tokens"):
            _pv6_conflicts.append("--max_new_tokens")
        if _pv6_conflicts:
            parser.error(
                "--reference_environment (pv6) is incompatible with "
                + ", ".join(_pv6_conflicts)
                + ": pv6 defines its own prompt, arm count, horizon and "
                  "two-stage constrained decoding.")
    elif args.steering_scope != pv6_episode.DEFAULT_STEERING_SCOPE:
        # Mirror of the guard above: pv1-pv5 have a single generation stage, so
        # there is no rationale pass for a scope to select. Silently ignoring
        # would write a row whose iface tag claims a scope that never ran.
        parser.error(
            "--steering_scope is a pv6 flag and requires "
            "--reference_environment; pv1-pv5 have no separate rationale pass.")

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
