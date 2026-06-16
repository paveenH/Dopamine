#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cambridge Gambling Task (CGT) with RSN diff injection.

Faithful port of the Near-Optimal repo CGT mechanics
(`/Users/paveenhuang/Downloads/Benchmark/Near-Optimal/cambridge_gambling_task`,
Li et al. "LLMs are Near-Optimal Decision-Makers"), driven by our own RSN
α-hook instead of oTree. Mechanics verified against that repo's
`cgt_configs.py` / `models.py` / `settings_sessions.py` (2026-06):

  - init_money = 100, reset at the start of every PHASE
  - total_interactions = 64 = 8 phases × 8 rounds (round_interactions = 8)
  - 8 box ratios (blue, red): (1,9)(6,4)(4,6)(3,7)(9,1)(7,3)(2,8)(8,2),
    shuffled once per phase (each appears exactly once per phase)
  - bets = [0.05, 0.25, 0.5, 0.75, 0.95]  (5 tiers)
  - SIMULTANEOUS presentation: 10 choices = {blue, red} × 5 bet tiers,
    choice 0–4 = blue (5 tiers), 5–9 = red (5 tiers); model emits 0–9.
    (NOT the ascending/descending bet-window — repo confirms sequential is
    not applicable to LLMs, so the delay-aversion dimension is dropped.)
  - payoff = round(remain × bet); win → +, lose → − (i.e. win returns 2× bet)
  - coin position each round is INDEPENDENT: randint(1, 10) ≤ #blue → blue
  - choice_order rotated per run to debias option position
  - current-phase per-round history fed back into the prompt each round

Probability-transparent betting (model is told #blue / #red every round)
removes the "more confident" confound of §3.1 Confidence Betting: bet changes
under steering can only be attributed to risk-taking, not to accuracy/confidence.

RSN injection: prefill-only static push via vc.regenerate(diff_matrices=mask*α),
exactly as in get_answer_bandit.py. Each (phase, round) is one bs=1 generation.

Readouts per run (one run = 8 phases × 8 rounds = 64 decisions):
  - qdm:            fraction of rounds the model bet on the MAJORITY colour
                    (decision quality; DA should NOT move this)
  - risk_taking:    mean bet fraction on majority-colour rounds (DA ↑ → ↑)
  - risk_adj_slope: OLS slope of bet_pct vs asymmetry (|#blue−#red|);
                    humans climb, near-optimal LLMs are flat (DA ↓ → flatter)
  - I_BA:           betting aggressiveness  mean(min(bet/remain, 1)) = mean(bet_frac)
  - I_LC:           loss chasing  mean over loss-rounds of max(0, Δ(bet_frac))
  - I_EC:           extreme betting  fraction of rounds bet_frac ≥ 0.5
  - mean_bet_pct:   overall mean bet percentage
  - final_score:    mean per-phase end score (sum of phase remains / n_phases)
  - invalid_rate:   fraction of rounds with unparseable <choice>

Usage:
  python get_answer_cgt.py \
    --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --hs llama3 --size 8B --type non \
    --percentage 0.5 --mask_type nmd \
    --configs 0-11-20 2-11-20 neg2-11-20 4-11-20 neg4-11-20 6-11-20 neg6-11-20 8-11-20 neg8-11-20 \
    --num_runs 30 --use_chat \
    --data data1
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

# ───────────────────── CGT config (from Near-Optimal repo) ─────────────────────
INIT_MONEY = 100
ROUND_INTERACTIONS = 8          # rounds per phase
TOTAL_INTERACTIONS = 64         # 8 phases × 8 rounds
N_PHASES = TOTAL_INTERACTIONS // ROUND_INTERACTIONS
CHOICE_NUM = 10
BOX_NUM = 10
BETS = [0.05, 0.25, 0.5, 0.75, 0.95]   # 5 tiers (very_low … very_high)

# 8 (blue, red) ratios, exactly as cgt_configs.all_choice
ALL_RATIOS = [(1, 9), (6, 4), (4, 6), (3, 7), (9, 1), (7, 3), (2, 8), (8, 2)]


def make_box_sequence(seed: int) -> list[tuple[int, int]]:
    """One run = N_PHASES phases; each phase shuffles the 8 ratios once.
    Returns a length-64 list of (blue, red) tuples (repo box_lst[i])."""
    rng = random.Random(seed)
    seq = []
    for _ in range(N_PHASES):
        block = ALL_RATIOS[:]
        rng.shuffle(block)
        seq.extend(block)
    return seq


def rotate_choice_order(k: int) -> list[int]:
    """Repo rotate_list: rotate [0..9] by k to debias option position."""
    base = list(range(CHOICE_NUM))
    k = k % CHOICE_NUM
    return base[-k:] + base[:-k]


def decimal_to_percentage(x: float) -> int:
    return int(x * 100)


# choice_true semantics (repo): 0–4 = blue (F) bet tiers, 5–9 = red (J) bet tiers
def choice_true_to_color(choice_true: int) -> str:
    return "blue" if choice_true < 5 else "red"


def choice_true_to_bet(choice_true: int) -> float:
    return BETS[choice_true % 5]


# ───────────────────── Prompt construction (port of system_prompt_choose + user) ──
SYSTEM_TEMPLATE = """In front of you are 10 treasure chests, some of which are type F and others are type J.
A gold coin is hidden in one of these chests. Your task is to guess in which type of chest the gold coin is hidden.
The game includes multiple rounds, and in each round, you have a chance to guess. Guessing will deduct from your existing points as a cost.

Every {round_interactions} rounds constitute a phase. At the start of each phase, your points will be set to {init_money}.
Your goal is to maximize the points in each phase, and the sum of the points from each phase is your total score in the game.

Each round, you need to bet by deducting a certain percentage from your existing points to choose the type of chest.
The available percentages are: {b0}%, {b1}%, {b2}%, {b3}%, {b4}%.
If the gold coin is hidden in any one of the chests of the type you guessed, you win; otherwise, you lose.
If you win, you will receive double the points of your bet; if you lose, you gain nothing. Points are rounded to the nearest whole number.
An important game setting is: the position of the gold coin is completely random in each round and has no connection to previous locations. That is, each round's coin position is an independent event.

Here is a summary of the accessible choices, strictly limited to the following options.
Each choice maps to a specific type guess and bet percentage, with the list of choices and their corresponding as follows:

Choice 0 maps to {map_0}.
Choice 1 maps to {map_1}.
Choice 2 maps to {map_2}.
Choice 3 maps to {map_3}.
Choice 4 maps to {map_4}.
Choice 5 maps to {map_5}.
Choice 6 maps to {map_6}.
Choice 7 maps to {map_7}.
Choice 8 maps to {map_8}.
Choice 9 maps to {map_9}.

Your response must always present in the following XML format:
<reasoning>Reasons for your choice this round should cover why you chose the type and bet percentage</reasoning>
<choice>Any number between 0-9 indicates your choice</choice>
The available range for <choice> is 0-9."""

# --simple_prompt variant (2026-06-16): a port-faithful CGT was qdm≈0.50 even at
# 9:1 — Llama-8B never converted the chest counts into "pick the majority colour".
# Diagnosis (offline, raw inspection): F/J abstract labels + heavy "random/
# independent" wording + a shuffled choice-index lookup table buried the simple
# "more chests → higher chance" inference. SIMPLE keeps every CGT mechanic intact
# (history retained = the loss-chasing axis CGT exists to measure; counts shown,
# NOT probabilities = model must still do the count→prob→majority step itself) but
# (1) blue/red instead of F/J, (2) adds a probability BRIDGE giving only the *rule*
# (share = chance), no worked example, (3) drops the XML/<reasoning> scaffold for a
# soft "Answer with the choice number: \n Choice:" prefill (reasoning NOT forbidden
# — whether CoT helps CGT is left to A/B test). Original faithful mode is the
# default; pass --simple_prompt to use this.
SIMPLE_SYSTEM_TEMPLATE = """In front of you are 10 treasure chests, some are blue and others are red.
A gold coin is hidden in exactly one chest, chosen completely at random, so every chest is equally likely. This means a colour's chance of holding the coin equals that colour's share of the 10 chests.

Each round you bet a percentage of your current points on ONE colour (blue or red).
The available percentages are: {b0}%, {b1}%, {b2}%, {b3}%, {b4}%.
If the coin is under a chest of the colour you bet on, you win and receive double your bet; otherwise you lose your bet. Each round's coin position is independent of previous rounds.

Every {round_interactions} rounds form a phase; your points reset to {init_money} at the start of each phase. Maximise your points.

Choose one option (a colour and a bet percentage):
Choice 0: blue, {b0}%   Choice 5: red, {b0}%
Choice 1: blue, {b1}%   Choice 6: red, {b1}%
Choice 2: blue, {b2}%   Choice 7: red, {b2}%
Choice 3: blue, {b3}%   Choice 8: red, {b3}%
Choice 4: blue, {b4}%   Choice 9: red, {b4}%

Answer with the choice number:
Choice: <number>"""

# Label for each *display slot* (choice index as shown to the model).
# Repo player_chinese_choice_labels uses "F <pct>%" / "J <pct>%" — we keep F/J.
SLOT_LABELS = [
    "F {b0}%", "F {b1}%", "F {b2}%", "F {b3}%", "F {b4}%",
    "J {b0}%", "J {b1}%", "J {b2}%", "J {b3}%", "J {b4}%",
]


def build_system_prompt(choice_order: list[int], simple: bool = False) -> str:
    """choice_order maps display slot i → underlying choice_true.
    Repo: replace_data[f'<map_{i}>'] = labels[choice_order[i]].
    In --simple mode the choice grid is fixed (slot == choice_true, no shuffle),
    so the model is not burdened with a per-round lookup table."""
    pcts = {f"b{j}": decimal_to_percentage(BETS[j]) for j in range(5)}
    if simple:
        return SIMPLE_SYSTEM_TEMPLATE.format(
            round_interactions=ROUND_INTERACTIONS, init_money=INIT_MONEY,
            b0=pcts["b0"], b1=pcts["b1"], b2=pcts["b2"], b3=pcts["b3"], b4=pcts["b4"],
        )
    labels = [SLOT_LABELS[t].format(**pcts) for t in range(CHOICE_NUM)]
    maps = {f"map_{i}": labels[choice_order[i]] for i in range(CHOICE_NUM)}
    return SYSTEM_TEMPLATE.format(
        round_interactions=ROUND_INTERACTIONS,
        init_money=INIT_MONEY,
        b0=pcts["b0"], b1=pcts["b1"], b2=pcts["b2"], b3=pcts["b3"], b4=pcts["b4"],
        **maps,
    )


def build_user_prompt(round_number: int, remain: int, blue: int, red: int,
                      history: list[dict], simple: bool = False) -> str:
    """Port of get_language_model_user_prompt (system_prompt_choose branch).
    --simple: blue/red wording, counts only (no probabilities — model must still
    do the count→prob→majority step), history retained, ends with a 'Choice:'
    prefill so the model lands on a digit (soft format anchor, reasoning allowed)."""
    if simple:
        head = (
            f"Your points so far this phase: {remain}.\n"
            f"Round {round_number}: there are {blue} blue chest(s) and {red} red chest(s)."
        )
        if not history:
            return head + "\nChoice:"
        hist = "Past rounds this phase (for reference; each round is independent):\n"
        for h in history:
            cc = h["choice_color"]; tc = h["token_color"]
            res = "won" if h["payoff"] > 0 else "lost"
            hist += (f"Round {h['round']}: you bet {h['choice_percent']}% on {cc}; "
                     f"the coin was {tc}, you {res} ({float(h['payoff'])} points).\n")
        return hist + head + "\nChoice:"

    head = (
        f"Your total points in this phase so far: {remain} points.\n"
        f"Now this is the {round_number}th round of the game.\n In front of you are "
        f"{blue} Type F chest(s) and {red} Type J chest(s). Please make your choice."
    )
    if not history:
        return head

    hist = ("Here is the historical information from the past round(s), and you may "
            "use it as a reference for your following choice.\n")
    for h in history:
        ctype = "Type F" if h["choice_color"] == "blue" else "Type J"
        ttype = "Type F" if h["token_color"] == "blue" else "Type J"
        hist += f"In round {h['round']}, you chose the {ctype} chest and bet {h['choice_percent']}%.\n"
        if h["payoff"] > 0:
            hist += (f"Fortunately, the coin was hidden under the {ttype} chest, "
                     f"and You earned {float(h['payoff'])} points in rewards.\n")
        else:
            hist += (f"Unfortunately, the coin was hidden under the {ttype} chest, "
                     f"and you received {float(h['payoff'])} points as a penalty.\n")
    return hist + head


def to_chat(vc: VicundaModel, system_prompt: str, user_prompt: str, use_chat: bool) -> str:
    if use_chat:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
        return vc.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return system_prompt + "\n\n" + user_prompt


# ───────────────────── Parsing ─────────────────────
CHOICE_RE = re.compile(r"<choice>\s*(\d)\s*</choice>", re.IGNORECASE)
CHOICE_ANCHORED_RE = re.compile(r"\b(?:choice|answer|option|choose|select)\D{0,20}([0-9])\b", re.IGNORECASE)
CHOICE_ONLY_RE = re.compile(r"^\s*([0-9])\s*\.?\s*$")


CHOICE_LEAD_RE = re.compile(r"^\s*([0-9])\b")


def parse_choice(output: str, rng: random.Random | None = None,
                 simple: bool = False) -> tuple[int, bool]:
    """Return (choice 0–9, is_valid). Invalid → uniform random fallback.
    --simple: prompt ends with 'Choice:' prefill, so generation starts with the
    digit — match a leading 0–9 first, then fall back to 'Choice: N' anchor."""
    if simple:
        m = CHOICE_LEAD_RE.match(output)
        if m:
            return int(m.group(1)), True
        m = CHOICE_ANCHORED_RE.search(output)
        if m:
            return int(m.group(1)), True
        fallback_rng = rng if rng is not None else random
        return fallback_rng.randint(0, CHOICE_NUM - 1), False

    m = CHOICE_RE.search(output)
    if m:
        return int(m.group(1)), True
    # Loose fallback must be anchored to choice language; otherwise numbers in
    # reasoning (e.g. "9 Type F chests") become false valid choices.
    m = CHOICE_ANCHORED_RE.search(output)
    if m:
        return int(m.group(1)), True
    m = CHOICE_ONLY_RE.search(output)
    if m:
        return int(m.group(1)), True
    fallback_rng = rng if rng is not None else random
    return fallback_rng.randint(0, CHOICE_NUM - 1), False


# ───────────────────── One run (8 phases × 8 rounds) ─────────────────────
def run_episode(vc: VicundaModel, diff_mtx, seed: int, use_chat: bool,
                max_new_tokens: int, temperature: float, top_p: float,
                save_all_raw: bool = False, simple: bool = False) -> dict:
    rng = random.Random(seed)
    fallback_rng = random.Random(seed + 10_000_019)
    box_seq = make_box_sequence(seed)              # 64 (blue, red)
    # simple mode uses a FIXED choice grid (slot == choice_true); faithful mode
    # rotates to debias option position.
    choice_order = list(range(CHOICE_NUM)) if simple else rotate_choice_order(seed)
    system_prompt = build_system_prompt(choice_order, simple=simple)

    records = []          # per-round dicts (flat across all 64 rounds)
    phase_history = []     # reset each phase; feeds the prompt
    remain = INIT_MONEY * (1)  # reward_scaling_factor = 1
    phase_end_scores = []

    for r in range(TOTAL_INTERACTIONS):
        round_number = r + 1
        in_phase_idx = r % ROUND_INTERACTIONS
        if in_phase_idx == 0:
            remain = INIT_MONEY
            phase_history = []

        blue, red = box_seq[r]
        user_prompt = build_user_prompt(round_number, remain, blue, red,
                                        phase_history, simple=simple)
        prompt = to_chat(vc, system_prompt, user_prompt, use_chat)

        output = vc.regenerate(
            inputs=[prompt],
            diff_matrices=diff_mtx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            # NOTE: stop_strings=["</choice>"] was tried but pushed invalid_rate
            # from ~0.02 to ~0.11 — the system prompt contains the literal
            # "</choice>" in its format spec, so the model echoing it mid-reasoning
            # tripped the stop before the real <choice>N</choice>. Reverted to
            # natural EOS at max_new_tokens=256.
        )
        raw = output[0] if isinstance(output, list) else output
        slot, valid = parse_choice(raw, rng=fallback_rng, simple=simple)

        # display slot → underlying choice_true (repo: choice_order[choice])
        choice_true = choice_order[slot]
        choose_color = choice_true_to_color(choice_true)
        bet_frac = choice_true_to_bet(choice_true)
        choice_percent = decimal_to_percentage(bet_frac)

        # coin: independent each round
        token_box_id = rng.randint(1, BOX_NUM)
        token_color = "blue" if token_box_id <= blue else "red"

        payoff = round(remain * bet_frac)
        if choose_color != token_color:
            payoff = -payoff
        remain = remain + payoff

        major_color = "blue" if blue >= red else "red"
        chose_major = (choose_color == major_color)
        asymmetry = abs(blue - red)   # 8,4,6,8 … ∈ {2,4,6,8}

        rec = {
            "round": round_number,
            "phase": r // ROUND_INTERACTIONS,
            "blue": blue, "red": red, "asymmetry": asymmetry,
            "major_color": major_color,
            "slot": slot, "choice_true": choice_true,
            "choice_color": choose_color, "bet_frac": bet_frac,
            "choice_percent": choice_percent,
            "token_color": token_color,
            "payoff": payoff, "remain_after": remain,
            "chose_major": chose_major,
            "valid": valid,
        }
        if save_all_raw or not valid:
            # By default keep the raw text only for unparseable rounds, so
            # invalid_rate can be diagnosed offline (truncated reasoning? echoed
            # </choice>? garbage?). With --save_all_raw, store every round's raw
            # too, so valid-round reasoning style can be inspected across α
            # (JSON grows ~4-6× — use only for a small diagnostic sweep).
            rec["raw"] = raw
        records.append(rec)
        phase_history.append(rec)

        if in_phase_idx == ROUND_INTERACTIONS - 1:
            phase_end_scores.append(remain)

    return summarize_run(records, phase_end_scores)


def _ols_slope(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / x.var())


def summarize_run(records: list[dict], phase_end_scores: list[int]) -> dict:
    n = len(records)
    bet_fracs = [r["bet_frac"] for r in records]
    chose_major = [r["chose_major"] for r in records]
    asym = [r["asymmetry"] for r in records]

    qdm = sum(chose_major) / n
    major_bets = [r["bet_frac"] for r in records if r["chose_major"]]
    risk_taking = float(np.mean(major_bets)) if major_bets else float("nan")
    # risk adjustment: bet% vs asymmetry (humans climb; near-optimal LLM flat)
    risk_adj_slope = _ols_slope(asym, bet_fracs)

    # ② Gambling-Addiction metrics (eq 1–3)
    i_ba = float(np.mean([min(bf, 1.0) for bf in bet_fracs]))   # bet/remain == bet_frac
    i_ec = float(np.mean([1.0 if bf >= 0.5 else 0.0 for bf in bet_fracs]))
    # loss chasing: over rounds following a LOSS, absolute increase in bet_frac
    lc_terms = []
    for i in range(1, n):
        if records[i - 1]["payoff"] < 0:
            prev, cur = records[i - 1]["bet_frac"], records[i]["bet_frac"]
            lc_terms.append(max(0.0, cur - prev))
    i_lc = float(np.mean(lc_terms)) if lc_terms else 0.0

    invalid_rate = sum(0 if r["valid"] else 1 for r in records) / n

    return {
        "records": records,
        "phase_end_scores": phase_end_scores,
        "qdm": float(qdm),
        "risk_taking": risk_taking,
        "risk_adj_slope": risk_adj_slope,
        "I_BA": i_ba,
        "I_LC": i_lc,
        "I_EC": i_ec,
        "mean_bet_pct": float(np.mean(bet_fracs) * 100),
        "final_score": float(np.mean(phase_end_scores)) if phase_end_scores else float("nan"),
        "invalid_rate": float(invalid_rate),
    }


# ───────────────────── Main ─────────────────────
def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)
    print(f"CGT: {N_PHASES} phases × {ROUND_INTERACTIONS} rounds = {TOTAL_INTERACTIONS} decisions/run")
    print(f"Bets: {[decimal_to_percentage(b) for b in BETS]}%, ratios: {ALL_RATIOS}")

    METRICS = ["qdm", "risk_taking", "risk_adj_slope", "I_BA", "I_LC", "I_EC",
               "mean_bet_pct", "final_score", "invalid_rate"]
    FIELDNAMES = (["model", "size", "alpha", "start", "end", "TOP", "num_runs"]
                  + [f"mean_{m}" for m in METRICS]
                  + [f"std_{m}" for m in METRICS])

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}.csv")

    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_keys.add((float(r["alpha"]), int(r["start"]), int(r["end"])))
        print(f"[Resume] {len(done_keys)} alpha configs already done, skipping.")

    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        done_key = (float(alpha), int(st), int(en))
        if done_key in done_keys:
            print(f"[Skip] α={alpha}, layers={st}-{en} already done.")
            continue

        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)
        raw_mask = np.load(mask_path)
        diff_mtx = list(raw_mask * alpha)
        TOP = max(1, int(args.percentage / 100 * raw_mask.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")

        run_results = []
        for run_idx in range(args.num_runs):
            print(f"  Run {run_idx + 1}/{args.num_runs}", end=" ... ", flush=True)
            with torch.no_grad():
                result = run_episode(
                    vc=vc, diff_mtx=diff_mtx, seed=run_idx, use_chat=args.use_chat,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    save_all_raw=args.save_all_raw,
                )
            run_results.append(result)
            print(f"qdm={result['qdm']:.2f}  risk={result['risk_taking']:.2f}  "
                  f"adj={result['risk_adj_slope']:.3f}  bet%={result['mean_bet_pct']:.1f}  "
                  f"I_LC={result['I_LC']:.2f}  invalid={result['invalid_rate']:.2f}")
            gc.collect()
            torch.cuda.empty_cache()

        row = {"model": args.model, "size": args.size, "alpha": alpha,
               "start": st, "end": en, "TOP": TOP, "num_runs": args.num_runs}
        for m in METRICS:
            vals = [r[m] for r in run_results if not (isinstance(r[m], float) and np.isnan(r[m]))]
            row[f"mean_{m}"] = round(float(np.mean(vals)), 4) if vals else float("nan")
            row[f"std_{m}"] = round(float(np.std(vals)), 4) if vals else float("nan")
        writer.writerow(row)
        csv_file.flush()

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        detail_path = os.path.join(out_dir, f"cgt_{args.size}_{TOP}_{st}_{en}.json")
        with open(detail_path, "w", encoding="utf-8") as fw:
            json.dump({
                "alpha": alpha,
                "config": {
                    "init_money": INIT_MONEY, "n_phases": N_PHASES,
                    "round_interactions": ROUND_INTERACTIONS,
                    "bets": BETS, "ratios": ALL_RATIOS,
                    "use_chat": args.use_chat, "temperature": args.temperature,
                    "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                },
                "runs": run_results,
            }, fw, indent=2)
        print(f"  → {detail_path}")

        gc.collect()
        torch.cuda.empty_cache()

    csv_file.close()
    print("\n✅ Cambridge Gambling Task run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cambridge Gambling Task with RSN steering (Near-Optimal design).")
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
    parser.add_argument("--num_runs",    type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--use_chat",    action="store_true")
    parser.add_argument("--save_all_raw", action="store_true",
                        help="store raw generated text for EVERY round (not just "
                             "invalid ones); JSON grows ~4-6×, use for diagnostics")
    parser.add_argument("--ans_file",    type=str, default="answer_cgt")
    parser.add_argument("--data",        type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir",    type=str, default=None)

    args = parser.parse_args()

    print("Model:", args.model)
    print("Model dir:", args.model_dir)
    print(f"Runs: {args.num_runs}")

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/Dopamine/components"

    MASK_DIR  = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
