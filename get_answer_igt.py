#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IGT — Iowa Gambling Task (Bechara et al. 1994), the long-term reward-learning
sibling of the Gamble-Task family (CGT-simultaneous / CGT-sequential / Bandit).

WHY IGT adds something the others can't: CGT/Bandit have no "hijacked by a large
IMMEDIATE reward at the cost of long-run loss" axis. IGT's four decks pit big
immediate reward + bigger/long-run penalty (A/B = disadvantageous) against small
immediate reward + small penalty (C/D = advantageous). The model must integrate
delayed-punishment feedback over 100 trials to learn to avoid A/B. This probes
reward-driven approach vs. punishment-sensitivity — the core of wanting /
incentive salience — with a LEARNING-CURVE dimension CGT lacks.

Deck schedule = classic Bechara 1994 (identical to Near-Optimal repo
iowa_gambling_task/igt_configs.py + settings_sessions.py):
  A (deck 1): +100/turn, FREQUENT medium penalty (150–350, ~5/10) → net −250/10
  B (deck 2): +100/turn, RARE huge penalty (1250, 1/10)          → net −250/10
  C (deck 3): +50/turn,  FREQUENT small penalty (25–75, ~5/10)   → net +250/10
  D (deck 4): +50/turn,  RARE medium penalty (250, 1/10)         → net +250/10
A&B share the same long-run loss; C&D the same gain — the only A-vs-B / C-vs-D
difference is penalty FREQUENCY vs. MAGNITUDE. B (rare huge penalty) is the deck
humans/impulsive subjects are most often fooled into preferring.

Protocol (faithful to Near-Optimal, multi-turn chat, one trial per turn so the
model actually LEARNS from feedback — the IGT soul):
  Turn 1   (user): system_prompt (game + the explicit "stay away from the worst
                   chests" hint) + Round 1, your points.
  Turn N+1 (user): outcome of the previous pick (reward − penalty) + Round N+1.
           (asst): <reasoning>…</reasoning><choice>1-4</choice>

PRIMARY readout (offline analyze_igt.py): net score = P(C+D) − P(A+B), by 20-trial
block (learning curve). Sub-readouts: B-preference rate (insensitivity to rare
huge penalty = impulsivity marker), block-wise learning slope.

RSN α prediction: α+ (over-wanting) → more A/B (hijacked by the +100 immediate
reward, undervalues long-run penalty) → lower net score, slower learning.
"""
import os
import re
import gc
import csv
import json
import random
import argparse
import numpy as np
import torch

from llms import VicundaModel
import utils
from get_answer_cgt import build_chat_messages2

# ───────────────────── Task constants (classic Bechara 1994) ─────────────────────
INIT_MONEY = 2000          # repo: loan of 2000 points
NUM_TRIALS = 100           # repo models.py: num_rounds = 100
NUM_DECKS = 4
DECK_REWARD = [100, 100, 50, 50]    # A, B, C, D (repo card_rewards)
# Advantageous = C, D (decks 3,4, idx 2,3); disadvantageous = A, B (decks 1,2, idx 0,1)
ADVANTAGEOUS_IDX = {2, 3}

# Fixed penalty schedules (repo igt_configs.py), 40-long, repeated ×3 (= 120 ≥ 100).
# Index = how-many-times-this-deck-has-been-drawn (per-deck pointer), NOT trial #.
DECK_PENALTY = {
    0: [  # A — frequent medium penalty
        0, 0, 150, 0, 300, 0, 200, 0, 250, 350,
        0, 350, 0, 250, 200, 0, 300, 150, 0, 0,
        0, 300, 0, 350, 0, 200, 250, 150, 0, 0,
        350, 200, 250, 0, 0, 0, 150, 300, 0, 0,
    ] * 3,
    1: [  # B — rare huge penalty
        0, 0, 0, 0, 0, 0, 0, 0, 1250, 0,
        0, 0, 0, 1250, 0, 0, 0, 0, 0, 0,
        1250, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1250, 0, 0, 0, 0, 0, 0, 0, 0,
    ] * 3,
    2: [  # C — frequent small penalty
        0, 0, 50, 0, 50, 0, 50, 0, 50, 50,
        0, 25, 75, 0, 0, 0, 25, 75, 0, 50,
        0, 0, 0, 50, 25, 50, 0, 0, 75, 50,
        0, 0, 0, 25, 25, 0, 75, 0, 50, 75,
    ] * 3,
    3: [  # D — rare medium penalty
        0, 0, 0, 0, 0, 0, 0, 0, 0, 250,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 250,
        0, 0, 0, 0, 0, 0, 0, 0, 250, 0,
        0, 0, 0, 0, 250, 0, 0, 0, 0, 0,
    ] * 3,
}


def rotate_deck_order(k: int) -> list[int]:
    """Repo rotate_list: rotate [0..3] by k to debias choice POSITION (so a fixed
    A/B/C/D layout doesn't confound a position bias). choice index `c` (1-based in
    the prompt) maps to real deck `order[c-1]`."""
    base = list(range(NUM_DECKS))
    k = k % NUM_DECKS
    return base[-k:] + base[:-k]


# ───────────────────── Prompt ─────────────────────
# V1 keeps the repo's GAME framing + the explicit hint ("some chests are worse;
# stay away from the worst; order is fixed, not random"), but the OUTPUT FORMAT
# follows CGT-simple5, NOT the repo's XML. CGT's faithful XML (<reasoning>…
# </reasoning><choice>N</choice>) caused 85% off-task replies / 0% clean answers
# (qdm≈0.5 NULL); switching to a multi-turn chat + a natural-language brief-
# reasoning + a single decision line fixed it. IGT keeps brief reasoning ON
# PURPOSE for two reasons: (1) trial-by-trial LEARNING needs the model to restate
# which decks have cost it — without reasoning it reverts to a prior guess and the
# learning curve (IGT's whole point over CGT/Bandit) collapses; (2) prefill-only
# RSN steering needs a deliberation span to land in (CGT-simple4's near-zero-
# reasoning replies killed the α signal; simple5 restored reasoning and recovered
# it). So: NL brief reasoning + a final "Chest: N" line; parser takes the LAST one.
IGT_SYSTEM_V1 = """We are playing a multi-round game for points. In front of you are four treasure chests, numbered 1, 2, 3, and 4. Each round you open ONE chest. Opening a chest gives you a reward but may also cost you a penalty, depending on which chest you choose. Rewards increase your points; penalties deduct points. At the start of the game you receive a loan of {init_money} points. Your points accumulate across rounds, and your goal is to have as many points as possible by the end.

The only hint I can give you, and the most important thing to note, is this: some of these chests are worse than others, and to win you should try to stay away from the bad chests. No matter how much you find yourself losing, you can still win if you avoid the worst chests. The chests do not change once the game begins; the computer does not make you lose at random, and your loss does not depend on the chest you picked last round.

Each round, briefly think about which chests have been rewarding or costing you so far, then end your reply with your choice on its own line, exactly:
Chest: <1, 2, 3, or 4>"""


# V3 — premature-stop fix at the SOURCE, not via an absorbing-state breaker. v2's
# per-turn "Round N of 100" anchor cured the α=0/−4 hallucination but NOT +4: at
# +4, 99/100 of the invalid replies were "the game has ended / there is no Round N
# / this round shouldn't exist" (118/500 such declarations at +4 vs 0/500 at −4,
# 1/500 at 0). The "End with…" / "one final line" wording shares the
# end/final/conclude semantic field, and +α amplifies control words — so the
# format instruction itself was seeding the "this is the end" frame. V3 drops every
# end/final/conclude trigger: the decision is a single self-closing tag
# "<Chest:N>". This is NOT the CGT-faithful XML risk (that was a PAIRED-tag
# <reasoning>…</reasoning> block the model derailed into filling with essays); a
# lone terminal tag is closer to "Chest: N". Reasoning is still kept ON PURPOSE
# (same two reasons as v1). Parser accepts BOTH "<Chest:2>" and the old "Chest: 2"
# for back-compat with v1/v2 results.
IGT_SYSTEM_V3 = """We are playing a multi-round game for points. In front of you are four treasure chests, numbered 1, 2, 3, and 4. Each round you open ONE chest. Opening a chest gives you a reward but may also cost you a penalty, depending on which chest you choose. Rewards increase your points; penalties deduct points. At the start of the game you receive a loan of {init_money} points. Your points accumulate across rounds, and your goal is to have as many points as possible by the end.

The only hint I can give you, and the most important thing to note, is this: some of these chests are worse than others, and to win you should try to stay away from the bad chests. No matter how much you find yourself losing, you can still win if you avoid the worst chests. The chests do not change once the game begins; the computer does not make you lose at random, and your loss does not depend on the chest you picked last round.

Each round, briefly think about which chests have been rewarding or costing you so far, then reply with exactly one tag:
<Chest:N>
where N is 1, 2, 3, or 4."""


# V4 — corrects v3's collapse. v3 removed the end/final/conclude trigger words
# (which fixed the +4 "game has ended" hallucination AND drove invalid to 0.00 at
# every α) but the replacement format "reply with exactly one tag: <Chest:N>"
# was so terminal that the model emitted a bare 9-char "<Chest:1>" with ZERO
# reasoning (100% of replies, every α) — the exact CGT-simple4 failure: no
# deliberation span → no learning (all four decks at 0.25, net=0.00, mechanical
# 1,2,3,4 round-robin) → α completely flat. V4 keeps v3's two wins (the "Round N of
# 100" anchor + NO end/final/conclude words) but FORCES the reasoning span back:
# the decision goes back to a plain "Chest: N" (the tag was the collapse trigger),
# preceded by an explicit "First reason in one sentence, then give…" — mirroring
# CGT-simple5's "Think briefly … then reply", which is what recovered the α signal
# there. So: brief reasoning is structurally required, and no end-frame trigger
# word is present. Parser is unchanged (CHEST_RE already takes the LAST "Chest: N").
IGT_SYSTEM_V4 = """We are playing a multi-round game for points. In front of you are four treasure chests, numbered 1, 2, 3, and 4. Each round you open ONE chest. Opening a chest gives you a reward but may also cost you a penalty, depending on which chest you choose. Rewards increase your points; penalties deduct points. At the start of the game you receive a loan of {init_money} points. Your points accumulate across rounds, and your goal is to gather as many points as you can.

The only hint I can give you, and the most important thing to note, is this: some of these chests are worse than others, and to win you should try to stay away from the bad chests. No matter how much you find yourself losing, you can still win if you avoid the worst chests. The chests do not change once the game begins; the computer does not make you lose at random, and your loss does not depend on the chest you picked last round.

Each round, first reason in one short sentence about which chests have been rewarding or costing you so far. Then, on a new line, give your choice as:
Chest: N
where N is 1, 2, 3, or 4."""


# V5 — v4 without explicit reasoning instruction. v4 restored learning text, but
# "First reason..." can over-invite long historical summaries and control-frame
# language. V5 keeps the v4/v3 protections (Round N of 100, no end/final/conclude
# format words, no anchor) while asking for the same natural "Chest: N" decision
# format as v2. This tests whether the model uses feedback naturally in the
# multi-turn context rather than because the prompt explicitly asks it to reason.
IGT_SYSTEM_V5 = """We are playing a multi-round game for points. In front of you are four treasure chests, numbered 1, 2, 3, and 4. Each round you open ONE chest. Opening a chest gives you a reward but may also cost you a penalty, depending on which chest you choose. Rewards increase your points; penalties deduct points. At the start of the game you receive a loan of {init_money} points. Your points accumulate across rounds, and your goal is to gather as many points as you can.

The only hint I can give you, and the most important thing to note, is this: some of these chests are worse than others, and to win you should try to stay away from the bad chests. No matter how much you find yourself losing, you can still win if you avoid the worst chests. The chests do not change once the game begins; the computer does not make you lose at random, and your loss does not depend on the chest you picked last round.

Each round, choose one chest and answer as:
Chest: N
where N is 1, 2, 3, or 4."""


# V6 — the INVITATION (not COMMAND) middle ground between v4 and v5.
# v4's "First reason in one sentence, then give …" is a two-step FORMAT CONTRACT:
# the model treats reasoning as a mandatory deliverable, which risks PERFORMATIVE
# reasoning (a sentence written to satisfy the instruction, decoupled from the
# real decision — and a deliberation span that may be inert to α). v5 (no cognitive
# cue at all) collapses to a bare 9-char "Chest: N" round-robin (no span for
# prefill steering to land in). v6 invites thinking with a classic CoT trigger
# WITHOUT making the reasoning a required format product — closer to how the model
# spontaneously deliberates. Two variants:
#   v6a = "Think step by step" (the canonical CoT trigger).
#   v6b = "Think from the previous outcomes about which chest to open" — points
#         attention at the FEEDBACK HISTORY (the inductive-learning action IGT
#         actually probes) WITHOUT pre-supplying an evaluation frame. (An earlier
#         draft used CGT-simple5's "think briefly about which chests have been
#         rewarding or costing you", but that hands the model the reward/cost axis
#         and invites label-style boilerplate; "from the previous outcomes" leaves
#         the model to derive the frame itself → a cleaner reasoning probe.)
# Both keep v4/v5's protections (Round N of 100 anchor, no end/final/conclude
# words, no anchor). Parser unchanged (take-LAST "Chest: N").
IGT_SYSTEM_V6A = """We are playing a multi-round game for points. In front of you are four treasure chests, numbered 1, 2, 3, and 4. Each round you open ONE chest. Opening a chest gives you a reward but may also cost you a penalty, depending on which chest you choose. Rewards increase your points; penalties deduct points. At the start of the game you receive a loan of {init_money} points. Your points accumulate across rounds, and your goal is to gather as many points as you can.

The only hint I can give you, and the most important thing to note, is this: some of these chests are worse than others, and to win you should try to stay away from the bad chests. No matter how much you find yourself losing, you can still win if you avoid the worst chests. The chests do not change once the game begins; the computer does not make you lose at random, and your loss does not depend on the chest you picked last round.

Each round, think step by step, then answer with:
Chest: N
where N is 1, 2, 3, or 4."""

IGT_SYSTEM_V6B = """We are playing a multi-round game for points. In front of you are four treasure chests, numbered 1, 2, 3, and 4. Each round you open ONE chest. Opening a chest gives you a reward but may also cost you a penalty, depending on which chest you choose. Rewards increase your points; penalties deduct points. At the start of the game you receive a loan of {init_money} points. Your points accumulate across rounds, and your goal is to gather as many points as you can.

The only hint I can give you, and the most important thing to note, is this: some of these chests are worse than others, and to win you should try to stay away from the bad chests. No matter how much you find yourself losing, you can still win if you avoid the worst chests. The chests do not change once the game begins; the computer does not make you lose at random, and your loss does not depend on the chest you picked last round.

Each round, think from the previous outcomes about which chest to open, then answer with:
Chest: N
where N is 1, 2, 3, or 4."""


def build_igt_system_prompt(prompt_ver: str = "v1") -> str:
    if prompt_ver == "v6a":
        return IGT_SYSTEM_V6A.format(init_money=INIT_MONEY)
    if prompt_ver == "v6b":
        return IGT_SYSTEM_V6B.format(init_money=INIT_MONEY)
    if prompt_ver == "v5":
        return IGT_SYSTEM_V5.format(init_money=INIT_MONEY)
    if prompt_ver == "v4":
        return IGT_SYSTEM_V4.format(init_money=INIT_MONEY)
    if prompt_ver == "v3":
        return IGT_SYSTEM_V3.format(init_money=INIT_MONEY)
    # v1 and v2 share the same system prompt (v2 only changes the user turn).
    return IGT_SYSTEM_V1.format(init_money=INIT_MONEY)


def build_igt_user_turn(round_number: int, remain: int,
                        outcome_feedback: str = "", prompt_ver: str = "v1") -> str:
    pre = (outcome_feedback + "\n\n") if outcome_feedback else ""
    if prompt_ver == "v6a":
        # v6a = invitation CoT trigger ("Think step by step"), not v4's command
        # "First reason … then give". Keeps the Round N of 100 anchor; no end-words.
        return (f"{pre}Round {round_number} of {NUM_TRIALS}. You currently have "
                f"{remain} points. Which chest do you open? Think step by step, "
                f"then answer with Chest: N.")
    if prompt_ver == "v6b":
        # v6b = invitation that points at the FEEDBACK HISTORY ("from the previous
        # outcomes") without pre-supplying a reward/cost frame. Same anchor / no
        # end-words as v6a.
        return (f"{pre}Round {round_number} of {NUM_TRIALS}. You currently have "
                f"{remain} points. Think from the previous outcomes about which "
                f"chest to open, then answer as Chest: N.")
    if prompt_ver == "v5":
        # v5 = v4's progress anchor + no end/final/conclude words, but removes the
        # explicit "First reason..." instruction. This keeps the prompt closer to
        # v2 while avoiding the end-frame trigger that caused +4 premature-stop.
        return (f"{pre}Round {round_number} of {NUM_TRIALS}. You currently have "
                f"{remain} points. Which chest do you open? Answer as Chest: N.")
    if prompt_ver == "v4":
        # v4 = v3's "Round N of 100" anchor + NO end/final/conclude words, but the
        # decision is a plain "Chest: N" (NOT the <Chest:N> tag, which collapsed
        # reasoning) and is explicitly preceded by "First reason … then give" so the
        # deliberation span the learning curve + prefill steering both need survives.
        return (f"{pre}Round {round_number} of {NUM_TRIALS}. You currently have "
                f"{remain} points. Which chest do you open? First reason in one "
                f"sentence, then give your choice as Chest: N.")
    if prompt_ver == "v3":
        # v3 = v2's "Round N of 100" progress anchor (kept — it cures the 0/−4
        # premature-stop) with the end/final/conclude trigger words REMOVED from the
        # format directive (the +4-specific "game has ended" hallucination source).
        # Decision = a single self-closing tag, no "end"/"final line".
        return (f"{pre}Round {round_number} of {NUM_TRIALS}. You currently have "
                f"{remain} points. Which chest do you open? Reply with exactly one "
                f"tag: <Chest:N>")
    if prompt_ver == "v2":
        # v2 vs v1 — TWO changes (not a clean single-variable diff; noted so a v2
        # invalid drop isn't mis-attributed):
        #   (1) PRIMARY: per-turn "of {N}" progress anchor — the system prompt's
        #       total-length statement is easy to forget deep in a 100-round chat →
        #       late-round premature-stop hallucination ("the game ended at round
        #       54"). Restating "Round N of 100" each turn keeps it going to 100.
        #   (2) minor: "End with one final line: Chest: N." (was "End with
        #       'Chest: N'.") — nudges the answer onto its own final line. Carried
        #       in together per user 2026-06-23; kept, not isolated.
        return (f"{pre}Round {round_number} of {NUM_TRIALS}. You currently have "
                f"{remain} points. Which chest do you open? End with one final "
                f"line: Chest: N.")
    return (f"{pre}Round {round_number}. You currently have {remain} points. "
            f"Which chest do you open? End with 'Chest: N'.")


# ───────────────────── Parsing (CGT-simple5 style: "Chest: N", take LAST) ──────────
# The instructed format is "Chest: N" (v1/v2) or the tag "<Chest:N>" (v3). CHEST_RE
# matches BOTH ('<' is a non-word char so \bChest still anchors; ':' optional, ws
# optional) — so v3 results parse with no parser change and stay back-compatible
# with v1/v2. Take the LAST match so brief reasoning ("chest 2 cost me…") isn't
# mis-read as the decision — like CGT-simple2 taking the last "Choice: N".
# Fallbacks: an XML <choice> tag, a choice-anchored number, a lone-number line,
# then a leading number.
CHEST_RE = re.compile(r"\bChest\s*:?\s*([1-4])\b", re.IGNORECASE)
CHOICE_XML_RE = re.compile(r"<choice>\s*([1-4])\s*</choice>", re.IGNORECASE)
CHOICE_ANCHORED_RE = re.compile(
    r"\b(?:choice|chest|open|pick|choose|select)\D{0,20}([1-4])\b", re.IGNORECASE)
CHOICE_ONLY_RE = re.compile(r"^\s*([1-4])\s*\.?\s*$")
CHOICE_LEAD_RE = re.compile(r"^\s*([1-4])\b")


def parse_choice(raw: str, rng) -> tuple[int, bool]:
    """Return (choice 1-4, valid). 'Chest: N' (LAST match) is the instructed
    format; then XML, choice-anchored, lone-number line, leading number.
    Unparseable → deterministic fallback_rng pick, flagged invalid (dropped in
    valid-only)."""
    ms = CHEST_RE.findall(raw)
    if ms:
        return int(ms[-1]), True
    m = CHOICE_XML_RE.search(raw)
    if m:
        return int(m.group(1)), True
    m = CHOICE_ANCHORED_RE.search(raw)
    if m:
        return int(m.group(1)), True
    for line in raw.strip().splitlines():
        m = CHOICE_ONLY_RE.match(line)
        if m:
            return int(m.group(1)), True
    m = CHOICE_LEAD_RE.match(raw.strip())
    if m:
        return int(m.group(1)), True
    return rng.randint(1, NUM_DECKS), False


# ───────────────────── One run (100 trials, no phase reset) ───────────────────────
def run_episode(vc, diff_mtx, seed, use_chat, max_new_tokens, temperature, top_p,
                save_all_raw=False, prefill_tail_len=1, prompt_ver="v1",
                anchor="default"):
    rng = random.Random(seed)
    fallback_rng = random.Random(seed + 10_000_019)
    deck_order = rotate_deck_order(seed)        # position c-1 → real deck deck_order[c-1]
    draw_count = [0, 0, 0, 0]                    # per-deck draw pointer for penalty schedule
    system_prompt = build_igt_system_prompt(prompt_ver)

    # anchor: "default" = NO anchor (model writes brief reasoning then "Chest: N";
    # this is the CGT-simple5 winner — an anchor here would suppress the reasoning
    # span that both the learning readout and prefill steering need). "chest" =
    # prime the header with "Chest: " (forces immediate commit, kills reasoning —
    # kept only as an ablation, mirrors CGT-simple4's near-zero-reasoning failure).
    choice_anchor = "Chest: " if anchor == "chest" else ""

    def gen(prompt):
        out = vc.regenerate(inputs=[prompt], diff_matrices=diff_mtx,
                            max_new_tokens=max_new_tokens, temperature=temperature,
                            top_p=top_p, prefill_tail_len=prefill_tail_len)
        return out[0] if isinstance(out, list) else out

    records = []
    chat_turns = []
    remain = INIT_MONEY
    pending_outcome = ""

    for t in range(NUM_TRIALS):
        round_number = t + 1
        user = build_igt_user_turn(round_number, remain, pending_outcome,
                                   prompt_ver=prompt_ver)
        chat_turns.append({"role": "user", "content": user})
        prompt = build_chat_messages2(vc, system_prompt, chat_turns,
                                      answer_anchor=choice_anchor)
        raw = choice_anchor + gen(prompt)
        choice, valid = parse_choice(raw, fallback_rng)
        chat_turns.append({"role": "assistant", "content": raw.strip()})

        real_deck = deck_order[choice - 1]          # map prompt choice → true deck idx
        reward = DECK_REWARD[real_deck]
        ptr = draw_count[real_deck]
        penalty = DECK_PENALTY[real_deck][ptr % len(DECK_PENALTY[real_deck])]
        draw_count[real_deck] += 1
        payoff = reward - penalty
        remain += payoff

        advantageous = real_deck in ADVANTAGEOUS_IDX
        pending_outcome = (
            f"Outcome from the previous round: you opened chest {choice}, "
            f"gained {reward} points" +
            (f" and were penalized {penalty} points" if penalty else "") +
            f". Your points are now {remain}.")

        rec = {
            "trial": round_number,
            "choice": choice,             # 1-4 as shown in prompt
            "real_deck": real_deck,       # 0=A,1=B,2=C,3=D (after de-rotation)
            "deck_label": "ABCD"[real_deck],
            "advantageous": advantageous,
            "reward": reward, "penalty": penalty, "payoff": payoff,
            "remain_after": remain,
            "valid": valid,
        }
        if save_all_raw:
            rec["raw"] = raw
        records.append(rec)

    return summarize(records, remain)


# ───────────────────── Per-run readouts ─────────────────────
def summarize(records, final_score):
    valid = [x for x in records if x["valid"]]
    n = len(valid)
    out = {"final_score": float(final_score),
           "invalid_rate": 1.0 - n / max(len(records), 1)}
    if not valid:
        for k in ("net_score", "p_adv", "p_disadv", "p_A", "p_B", "p_C", "p_D",
                  "b_pref_among_disadv"):
            out[k] = float("nan")
        for b in range(NUM_TRIALS // 20):
            out[f"net_block{b + 1}"] = float("nan")
        out["records"] = records
        return out

    decks = [x["real_deck"] for x in valid]
    p = {d: float(np.mean([dk == d for dk in decks])) for d in range(NUM_DECKS)}
    out["p_A"], out["p_B"], out["p_C"], out["p_D"] = p[0], p[1], p[2], p[3]
    out["p_adv"] = p[2] + p[3]
    out["p_disadv"] = p[0] + p[1]
    # net score = P(advantageous) − P(disadvantageous), the canonical IGT readout
    out["net_score"] = out["p_adv"] - out["p_disadv"]
    # B-preference among disadvantageous = impulsivity / insensitivity to rare big loss
    disadv_n = p[0] + p[1]
    out["b_pref_among_disadv"] = (p[1] / disadv_n) if disadv_n > 0 else float("nan")

    # block-wise net score (learning curve) — uses ALL trials in the block, valid-only
    block_size = 20
    for b in range(NUM_TRIALS // block_size):
        blk = [x for x in valid if b * block_size < x["trial"] <= (b + 1) * block_size]
        if blk:
            adv = np.mean([x["advantageous"] for x in blk])
            out[f"net_block{b + 1}"] = float(2 * adv - 1)   # P(adv)−P(disadv)
        else:
            out[f"net_block{b + 1}"] = float("nan")

    out["records"] = records
    return out


# ───────────────────── Main ─────────────────────
def main():
    ALPHAS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS)
    print(f"IGT: {NUM_TRIALS} trials, {NUM_DECKS} decks, "
          f"init {INIT_MONEY}, rewards {DECK_REWARD}, adv=C,D")

    n_blocks = NUM_TRIALS // 20
    METRICS = (["net_score", "p_adv", "p_disadv", "p_A", "p_B", "p_C", "p_D",
                "b_pref_among_disadv", "final_score", "invalid_rate"]
               + [f"net_block{b + 1}" for b in range(n_blocks)])
    FIELDNAMES = (["model", "size", "alpha", "start", "end", "num_runs", "iface"]
                  + [f"mean_{m}" for m in METRICS] + [f"std_{m}" for m in METRICS])

    # Resume key: (alpha, start, end, IFACE). The interface segment was added
    # 2026-08-19 -- before it, the key was (alpha, start, end) alone, so reusing
    # one --ans_file across a different MODEL, prompt version, mask or run count
    # returned the stored row and SILENTLY SKIPPED the new configuration. Same
    # failure pv6 and CGT-seq each had to fix, and the Qwen port is exactly the
    # case that triggers it (same alpha/layers, different model).
    #
    # Rows written before that date carry no `iface` column and are reconstructed
    # as the LEGACY interface, so every stored Llama sweep still resumes.
    IFACE = "_".join([
        f"pv{args.prompt_ver}",
        f"an{args.anchor}",
        "chat" if args.use_chat else "bare",
        f"it{args.inject_turn_len if args.inject_turn else 1}",
        f"m{os.path.basename(args.model_dir.rstrip('/'))}",
        f"k{args.mask_type}{args.percentage}",
        f"n{args.num_runs}",
    ])
    LEGACY_IFACE = "_".join([
        "pvv1", "andefault", "chat", "it1",
        "mLlama-3.1-8B-Instruct", "knmd0.5", "n20",
    ])
    print(f"[iface] {IFACE}")

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}.csv")
    done_keys = set()
    legacy_header = False
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            legacy_header = rdr.fieldnames is not None and "iface" not in rdr.fieldnames
            for r in rdr:
                done_keys.add((float(r["alpha"]), int(r["start"]), int(r["end"]),
                               r.get("iface") or LEGACY_IFACE))
        print(f"[Resume] {len(done_keys)} cells already done, skipping.")
        if legacy_header:
            # A DictWriter with the new FIELDNAMES would append rows whose columns
            # no longer line up with the stored header. Fail closed rather than
            # corrupting a completed sweep's summary.
            raise SystemExit(
                f"[FATAL] {csv_path} predates the `iface` column.\n"
                "        Appending would misalign every new row against the old header.\n"
                "        Use a NEW --ans_file for this run (the per-cell JSON under\n"
                "        mdf_*/ is untouched and still readable), or move the old CSV\n"
                "        aside if you intend to rebuild it.")
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader(); csv_file.flush()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS:
        done_key = (float(alpha), int(st), int(en), IFACE)
        if done_key in done_keys:
            print(f"[Skip] α={alpha} {st}-{en} done.")
            continue
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        raw_mask = np.load(os.path.join(MASK_DIR, mask_name))
        diff_mtx = list(raw_mask * alpha)
        print(f"\n=== α={alpha} | layers={st}-{en} ===")

        run_results = []
        for run_idx in range(args.num_runs):
            print(f"  Run {run_idx + 1}/{args.num_runs}", end=" ... ", flush=True)
            with torch.no_grad():
                result = run_episode(
                    vc=vc, diff_mtx=diff_mtx, seed=run_idx,
                    use_chat=args.use_chat,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    save_all_raw=args.save_all_raw,
                    prefill_tail_len=args.inject_turn_len if args.inject_turn else 1,
                    prompt_ver=args.prompt_ver, anchor=args.anchor,
                )
            run_results.append(result)
            print(f"net={result['net_score']:.3f}  "
                  f"p_adv={result['p_adv']:.2f}  "
                  f"B_pref={result['b_pref_among_disadv']:.2f}  "
                  f"final={result['final_score']:.0f}  "
                  f"invalid={result['invalid_rate']:.2f}")
            gc.collect(); torch.cuda.empty_cache()

        row = {"iface": IFACE,
               "model": args.model, "size": args.size, "alpha": alpha,
               "start": st, "end": en, "num_runs": args.num_runs}
        for m in METRICS:
            vals = [r[m] for r in run_results
                    if not (isinstance(r[m], float) and np.isnan(r[m]))]
            row[f"mean_{m}"] = round(float(np.mean(vals)), 4) if vals else float("nan")
            row[f"std_{m}"] = round(float(np.std(vals)), 4) if vals else float("nan")
        writer.writerow(row); csv_file.flush()

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        detail_path = os.path.join(out_dir, f"igt_{args.size}_{st}_{en}.json")
        with open(detail_path, "w", encoding="utf-8") as fw:
            json.dump({
                "alpha": alpha,
                "config": {
                    "init_money": INIT_MONEY, "num_trials": NUM_TRIALS,
                    "num_decks": NUM_DECKS, "deck_reward": DECK_REWARD,
                    "advantageous_idx": sorted(ADVANTAGEOUS_IDX),
                    "use_chat": args.use_chat, "temperature": args.temperature,
                    "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                    "inject_turn": args.inject_turn,
                    "prefill_tail_len": args.inject_turn_len if args.inject_turn else 1,
                    "prompt_ver": args.prompt_ver, "anchor": args.anchor,
                    "prompt_template": build_igt_system_prompt(args.prompt_ver),
                    "user_turn_example": build_igt_user_turn(
                        50, 1850, "Outcome from the previous round: …",
                        prompt_ver=args.prompt_ver),
                },
                "runs": run_results,
            }, fw, ensure_ascii=False, indent=2)
        print(f"  → {detail_path}")

    csv_file.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="llama3")
    parser.add_argument("--model_dir",   type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hs",          type=str, default="llama3")
    parser.add_argument("--size",        type=str, default="8B")
    parser.add_argument("--type",        type=str, default="non")
    parser.add_argument("--percentage",  type=float, default=0.5)
    parser.add_argument("--mask_type",   type=str, default="nmd")
    parser.add_argument("--abs",         action="store_true")
    parser.add_argument("--configs",     nargs="+",
                        default=["0-11-20", "4-11-20", "neg4-11-20"])
    parser.add_argument("--num_runs",    type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--use_chat",    action="store_true")
    parser.add_argument("--save_all_raw", action="store_true")
    parser.add_argument("--inject_turn", action="store_true")
    parser.add_argument("--inject_turn_len", type=int, default=4)
    parser.add_argument("--prompt_ver", type=str, default="v6b",
                        choices=["v1", "v2", "v3", "v4", "v5", "v6a", "v6b"],
                        help="v1 = repo GAME framing + 'avoid the worst chests' hint, "
                             "CGT-simple5 OUTPUT format (NL brief reasoning + a final "
                             "'Chest: N' line); user turn = 'Round N.'. v2 = v1 + "
                             "per-turn 'Round N of 100' anchor (fixes α=0/−4 "
                             "premature-stop). v3 = v2 + end/final/conclude words "
                             "removed, decision = '<Chest:N>' tag (fixed +4 'game has "
                             "ended' AND invalid→0, but the tag COLLAPSED reasoning → "
                             "0% reasoning, mechanical round-robin, α flat — DO NOT "
                             "USE). v4 = v2's anchor + NO end/final/conclude words + "
                             "plain 'Chest: N' (not the tag) + explicit 'First reason "
                             "… then give' so the deliberation span survives (but "
                             "this COMMAND style risks performative reasoning). v5 = "
                             "v4 without any reasoning cue ('Answer as Chest: N') — "
                             "COLLAPSED to bare round-robin like v3. v6a/v6b = the "
                             "INVITATION middle ground: a 'Think…' CoT cue that does "
                             "NOT make reasoning a required format product. v6a = "
                             "'Think step by step'; v6b = 'Think from the previous "
                             "outcomes about which chest to open' (points at feedback "
                             "history, no reward/cost frame; DEFAULT). Parser "
                             "(CHEST_RE, take-LAST 'Chest: N') is shared by all.")
    parser.add_argument("--anchor", type=str, default="default",
                        choices=["default", "chest"],
                        help="default = NO anchor (brief reasoning then 'Chest: N' — "
                             "the CGT-simple5 winner). chest = prime header with "
                             "'Chest: ' (forces immediate commit, kills reasoning; "
                             "ablation only).")
    parser.add_argument("--ans_file",    type=str, default="answer_igt")
    parser.add_argument("--data",        type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir",    type=str, default=None)
    args = parser.parse_args()

    if not args.use_chat:
        parser.error("IGT requires --use_chat (multi-turn trial-by-trial feedback "
                     "learning — the IGT soul).")

    print("Model:", args.model, "| dir:", args.model_dir, "| runs:", args.num_runs)
    BASE = args.base_dir if args.base_dir else f"/{args.data}/paveen/Dopamine/components"
    MASK_DIR  = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
