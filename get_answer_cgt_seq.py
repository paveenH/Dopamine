#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGT-Sequential — the ascending/descending delay-aversion sibling of get_answer_cgt.py.

WHY a separate file: get_answer_cgt.py is the SIMULTANEOUS task (one Color+Bet
choice per round; measures risk appetite / risk adjustment). Its simple5 line
found α moves reasoning-investment (delib_tok), NOT betting. CGT-Sequential adds
the dimension simultaneous CANNOT test: DELAY AVERSION / impulsivity, via a bet
that is revealed one tier at a time so "wanting a big bet means WAITING for it".

Mechanics shared with get_answer_cgt (imported): 8 phases × 8 rounds, 100-pt reset
per phase, 8 box ratios shuffled per phase, ±bet payoff, independent coin, explicit
outcome feedback, counts-only odds, the RSN prefill-only α-hook (vc.regenerate).

Per-round protocol (clinical CGT order — chosen by user 2026-06-17):
  Turn 1  (user): Round N, M points, X blue / Y red chests. Choose a COLOUR.
          (asst): Color: blue|red
  Turn 2+ (user): system reveals bet tiers ONE AT A TIME (ascending 5→95 or
          descending 95→5). Each tier: "Current bet: P%. Accept or Wait?"
          (asst): Accept | Wait   →  Accept locks the bet at P%; Wait shows the
          next tier. Running off the last tier force-locks it.

Presentation is a RUN-LEVEL condition (--presentation asc|desc); both must be run
to compute delay_aversion_index = mean_bet_desc − mean_bet_asc. Output dirs are
keyed by --ans_file so asc/desc never overwrite.

PRIMARY readout = ACCEPT TIER (which step the model commits), NOT the bet amount
(bet amount is confounded by presentation order — desc anchors high, asc low).
  - mean_accept_step:  1..5, lower = more impulsive (commits sooner)
  - early_stop_rate:   fraction accepting at step 1 (the most impulsive marker:
                       asc step1 = settle for 5%; desc step1 = grab 95%)
  - mean_bet:          mean locked bet% (SECONDARY — presentation-confounded)
  - max_bet_first_stop (desc only): fraction grabbing 95% at step 1
  - qdm / risk by condition (control: knowing should stay flat)

RSN α prediction: α+ (over-wanting/impulsivity) → accept SOONER (lower step) in
BOTH conditions → delay_aversion_index widens.
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

# Reuse the simultaneous task's verified mechanics/constants.
from get_answer_cgt import (
    INIT_MONEY, ROUND_INTERACTIONS, TOTAL_INTERACTIONS, N_PHASES, BOX_NUM,
    BETS, ALL_RATIOS, make_box_sequence, decimal_to_percentage,
    build_chat_messages2,
)

BET_PCTS = [decimal_to_percentage(b) for b in BETS]        # [5, 25, 50, 75, 95]


# ───────────────────── Prompt ─────────────────────
# Built from simple5 (the corrected main line): betting-game framing, neutral
# reward wording, ±bet payoff matched to code, counts-only odds. The ONLY change
# is the SEQUENTIAL bet mechanic (choose colour first, then accept/wait the bet
# tiers one at a time). NO "rational/optimal/EV" wording (would turn it into a
# maths-optimisation test and pollute the wanting/impulsivity readout).
SEQ_SYSTEM_TEMPLATE = """We are playing a multi-round betting game for points. I am the host; you are the player. Win as many points as you can.

In front of you each round are 10 treasure chests, some blue and some red. A gold coin is hidden in exactly one chest, completely at random, so a colour's chance of holding the coin equals that colour's share of the 10 chests (e.g. 9 blue, 1 red → blue 90%, red 10%).

Each round has two steps. First, using the chest counts, you choose ONE colour: blue or red. Then I reveal possible bet sizes ONE AT A TIME ({order_phrase}). For each bet size I show, you either Accept it (you bet that percentage of your current points on your colour and the round ends) or Wait to see the next bet size. If your colour is correct, your points increase by the amount you bet; if your colour is wrong, your points decrease by the amount you bet. Each round's coin is independent.

Every {round_interactions} rounds form a phase; your points reset to {init_money} at the start of each phase. The sum of your points across all phases is your final score in the game — your aim is to make that final score as high as possible. Higher bets gain more points but also lose more points.

When I ask for your colour, reply with exactly:
Color: <blue or red>
When I show a bet size, reply with exactly one word:
Accept
or
Wait"""

# v2: colour/bet format SYMMETRISED — both steps are "reply with exactly one word:
# X or Y" (colour = Blue/Red, bet = Accept/Wait), to pull the two response formats
# together. Everything else (game framing, payoff, feedback, odds) identical to v1.
SEQ_SYSTEM_TEMPLATE_V2 = """We are playing a multi-round betting game for points. I am the host; you are the player. Win as many points as you can.

In front of you each round are 10 treasure chests, some blue and some red. A gold coin is hidden in exactly one chest, completely at random, so a colour's chance of holding the coin equals that colour's share of the 10 chests (e.g. 9 blue, 1 red → blue 90%, red 10%).

Each round has two steps. First, using the chest counts, you choose ONE colour: blue or red. Then I reveal possible bet sizes ONE AT A TIME ({order_phrase}). For each bet size I show, you either Accept it (you bet that percentage of your current points on your colour and the round ends) or Wait to see the next bet size. If your colour is correct, your points increase by the amount you bet; if your colour is wrong, your points decrease by the amount you bet. Each round's coin is independent.

Every {round_interactions} rounds form a phase; your points reset to {init_money} at the start of each phase. The sum of your points across all phases is your final score in the game — your aim is to make that final score as high as possible. Higher bets gain more points but also lose more points.

When I ask for your colour, reply with exactly one word:
Blue
or
Red
When I show a bet size, reply with exactly one word:
Accept
or
Wait"""

ORDER_PHRASE = {
    "asc": "from the smallest to the largest",
    "desc": "from the largest to the smallest",
}


def build_seq_system_prompt(presentation: str, prompt_ver: str = "v1") -> str:
    # v3 = v2's symmetric base (the validated v2b winner) + an explicit next-offer
    # hint added only at the bet turn (see build_bet_user_turn).
    tmpl = SEQ_SYSTEM_TEMPLATE_V2 if prompt_ver in ("v2", "v3") else SEQ_SYSTEM_TEMPLATE
    return tmpl.format(
        order_phrase=ORDER_PHRASE[presentation],
        round_interactions=ROUND_INTERACTIONS, init_money=INIT_MONEY,
    )


def build_color_user_turn(round_number, remain, blue, red,
                          phase_reset, outcome_feedback="", prompt_ver="v1"):
    pre = ""
    if outcome_feedback:
        pre += outcome_feedback + "\n\n"
    if phase_reset and round_number > 1:
        pre += f"--- New phase. Your points reset to {INIT_MONEY}. ---\n"
    choose = ("choose your colour, Blue or Red." if prompt_ver in ("v2", "v3")
              else "choose your colour (blue or red).")
    return (f"{pre}Round {round_number}. You have {remain} points. "
            f"This round: {blue} blue chest(s) and {red} red chest(s). "
            f"Use the chest counts to {choose}")


def build_bet_user_turn(color, pct, step, n_tiers, next_pct=None, prompt_ver="v1"):
    if prompt_ver != "v3":  # v1/v2 byte-identical
        return (f"You chose {color}. Bet size {step} of {n_tiers}: {pct}% of your "
                f"current points. Accept or Wait?")
    # v3 (A1): make the future offer EXPLICIT so "wait = larger/smaller bet" is
    # not something the model has to infer. asc → next is larger; desc → smaller.
    # The hint goes BEFORE "Accept or Wait?" so the decision question stays the
    # LAST sentence (prefill steering injects at the final token).
    if next_pct is None:
        hint = "This is the last bet size; there is no later offer. "
    else:
        rel = "larger" if next_pct > pct else "smaller"
        hint = f"If you Wait, the next bet size will be {next_pct}% ({rel}). "
    return (f"You chose {color}. Bet size {step} of {n_tiers}: {pct}% of your "
            f"current points. {hint}Accept or Wait?")


# ───────────────────── Parsing ─────────────────────
COLOR_RE = re.compile(r"\bColor\s*:?\s*(blue|red)\b", re.IGNORECASE)
BARE_COLOR_RE = re.compile(r"\b(blue|red)\b", re.IGNORECASE)
ACCEPT_RE = re.compile(r"\baccept\b", re.IGNORECASE)
WAIT_RE = re.compile(r"\bwait\b", re.IGNORECASE)


def parse_color(raw, rng):
    m = COLOR_RE.search(raw) or BARE_COLOR_RE.search(raw)
    if m:
        return m.group(1).lower(), True
    return rng.choice(["blue", "red"]), False


def parse_accept_wait(raw, rng):
    """Return (accept: bool, valid: bool). Take the FIRST decisive token so a
    later 'wait for next round' aside doesn't override an opening 'Accept'."""
    a = ACCEPT_RE.search(raw)
    w = WAIT_RE.search(raw)
    if a and w:
        return (a.start() < w.start()), True
    if a:
        return True, True
    if w:
        return False, True
    return rng.random() < 0.5, False      # unparseable → coin-flip, flagged invalid


# ───────────────────── One run (8 phases × 8 rounds) ─────────────────────
def run_episode(vc, diff_mtx, seed, presentation, use_chat,
                max_new_tokens, temperature, top_p,
                save_all_raw=False, prefill_tail_len=1,
                prompt_ver="v1", anchor="default"):
    rng = random.Random(seed)
    fallback_rng = random.Random(seed + 10_000_019)
    box_seq = make_box_sequence(seed)
    tiers = BET_PCTS if presentation == "asc" else list(reversed(BET_PCTS))
    n_tiers = len(tiers)
    system_prompt = build_seq_system_prompt(presentation, prompt_ver)

    # anchor: "default" = v1 byte-equivalent (colour="Color: ", bet="");
    #         "answer"  = both steps anchored on "Answer: " (A version);
    #         "none"    = both steps no anchor (B version).
    if anchor == "answer":
        color_anchor, bet_anchor = "Answer: ", "Answer: "
    elif anchor == "none":
        color_anchor, bet_anchor = "", ""
    else:  # default (v1)
        color_anchor, bet_anchor = "Color: ", ""

    records = []
    chat_turns = []
    remain = INIT_MONEY
    phase_end_scores = []
    pending_outcome = ""

    def gen(prompt):
        out = vc.regenerate(inputs=[prompt], diff_matrices=diff_mtx,
                            max_new_tokens=max_new_tokens, temperature=temperature,
                            top_p=top_p, prefill_tail_len=prefill_tail_len)
        return out[0] if isinstance(out, list) else out

    for r in range(TOTAL_INTERACTIONS):
        round_number = r + 1
        in_phase_idx = r % ROUND_INTERACTIONS
        phase_reset = (in_phase_idx == 0)
        if phase_reset:
            remain = INIT_MONEY
            chat_turns = []
            pending_outcome = ""

        blue, red = box_seq[r]
        major_color = "blue" if blue >= red else "red"
        asymmetry = abs(blue - red)

        # --- Step 1: colour ---
        user_c = build_color_user_turn(round_number, remain, blue, red,
                                       phase_reset, outcome_feedback=pending_outcome,
                                       prompt_ver=prompt_ver)
        chat_turns.append({"role": "user", "content": user_c})
        prompt_c = build_chat_messages2(vc, system_prompt, chat_turns,
                                        answer_anchor=color_anchor)
        raw_c = color_anchor + gen(prompt_c)
        choose_color, color_valid = parse_color(raw_c, fallback_rng)
        chat_turns.append({"role": "assistant", "content": raw_c.strip()})

        # --- Step 2: accept/wait the bet tiers one at a time ---
        accept_step = None
        seq_valid = True
        bet_raws = []
        for step in range(n_tiers):
            pct = tiers[step]
            next_pct = tiers[step + 1] if step + 1 < n_tiers else None
            user_b = build_bet_user_turn(choose_color, pct, step + 1, n_tiers,
                                         next_pct=next_pct, prompt_ver=prompt_ver)
            chat_turns.append({"role": "user", "content": user_b})
            prompt_b = build_chat_messages2(vc, system_prompt, chat_turns,
                                            answer_anchor=bet_anchor)
            raw_b = bet_anchor + gen(prompt_b)
            bet_raws.append(raw_b)
            chat_turns.append({"role": "assistant", "content": raw_b.strip()})
            accept, valid = parse_accept_wait(raw_b, fallback_rng)
            seq_valid = seq_valid and valid
            if accept:
                accept_step = step + 1
                break
        if accept_step is None:            # ran off the last tier → force-lock it
            accept_step = n_tiers
        locked_pct = tiers[accept_step - 1]
        bet_frac = locked_pct / 100.0

        # --- payoff (same as simultaneous) ---
        coin = "blue" if rng.randint(1, BOX_NUM) <= blue else "red"
        payoff = round(remain * bet_frac)
        if choose_color != coin:
            payoff = -payoff
        remain = remain + payoff

        won = payoff > 0
        pending_outcome = (
            f"Outcome from previous round: the coin was under a {coin} chest. "
            f"You {'won' if won else 'lost'} {abs(payoff)} points. "
            f"You now have {remain} points.")

        chose_major = (choose_color == major_color)
        rec = {
            "round": round_number, "phase": r // ROUND_INTERACTIONS,
            "blue": blue, "red": red, "asymmetry": asymmetry,
            "major_color": major_color, "presentation": presentation,
            "choice_color": choose_color, "coin": coin,
            "accept_step": accept_step, "n_tiers": n_tiers,
            "locked_pct": locked_pct, "bet_frac": bet_frac,
            "payoff": payoff, "remain_after": remain,
            "chose_major": chose_major,
            "valid": color_valid and seq_valid,
        }
        if save_all_raw:
            rec["raw_color"] = raw_c
            rec["raw_bets"] = bet_raws
        records.append(rec)

        if in_phase_idx == ROUND_INTERACTIONS - 1:
            phase_end_scores.append(remain)

    return summarize(records, phase_end_scores, presentation)


# ───────────────────── Per-run readouts ─────────────────────
def summarize(records, phase_end_scores, presentation):
    valid = [x for x in records if x["valid"]]
    n = len(valid)
    out = {"presentation": presentation,
           "invalid_rate": 1.0 - n / max(len(records), 1)}
    if not valid:
        for k in ("mean_accept_step", "early_stop_rate", "mean_bet",
                  "qdm", "risk_taking", "max_bet_first_stop",
                  "final_score", "mean_phase_score"):
            out[k] = float("nan")
        out["records"] = records
        return out

    steps = [x["accept_step"] for x in valid]
    out["mean_accept_step"] = float(np.mean(steps))
    out["early_stop_rate"] = float(np.mean([s == 1 for s in steps]))
    out["mean_bet"] = float(np.mean([x["bet_frac"] for x in valid])) * 100
    out["qdm"] = float(np.mean([x["chose_major"] for x in valid]))
    rt = [x["bet_frac"] for x in valid if x["chose_major"]]
    out["risk_taking"] = float(np.mean(rt)) if rt else float("nan")
    # desc-only: grabbing the very first (largest) tier = 95% at step 1
    if presentation == "desc":
        out["max_bet_first_stop"] = float(np.mean(
            [x["accept_step"] == 1 for x in valid]))
    else:
        out["max_bet_first_stop"] = float("nan")
    # prompt says "the sum of your points across all phases is your final score"
    # → final_score = sum; keep mean as a separate per-phase readout.
    out["final_score"] = float(np.sum(phase_end_scores)) if phase_end_scores else float("nan")
    out["mean_phase_score"] = float(np.mean(phase_end_scores)) if phase_end_scores else float("nan")
    out["records"] = records
    return out


# ───────────────────── Main ─────────────────────
def main():
    ALPHAS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS)
    print(f"Presentation: {args.presentation}")
    print(f"CGT-Sequential: {N_PHASES} phases × {ROUND_INTERACTIONS} rounds, "
          f"{len(BET_PCTS)}-tier accept/wait, bets {BET_PCTS}%")

    METRICS = ["mean_accept_step", "early_stop_rate", "mean_bet", "qdm",
               "risk_taking", "max_bet_first_stop", "final_score",
               "mean_phase_score", "invalid_rate"]
    FIELDNAMES = (["model", "size", "alpha", "start", "end", "presentation", "num_runs"]
                  + [f"mean_{m}" for m in METRICS] + [f"std_{m}" for m in METRICS])

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}.csv")
    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_keys.add((float(r["alpha"]), int(r["start"]), int(r["end"]),
                               r.get("presentation", "")))
        print(f"[Resume] {len(done_keys)} cells already done, skipping.")
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader(); csv_file.flush()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for alpha, (st, en) in ALPHAS:
        done_key = (float(alpha), int(st), int(en), args.presentation)
        if done_key in done_keys:
            print(f"[Skip] α={alpha} {st}-{en} {args.presentation} done.")
            continue
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        raw_mask = np.load(os.path.join(MASK_DIR, mask_name))
        diff_mtx = list(raw_mask * alpha)
        print(f"\n=== α={alpha} | layers={st}-{en} | {args.presentation} ===")

        run_results = []
        for run_idx in range(args.num_runs):
            print(f"  Run {run_idx + 1}/{args.num_runs}", end=" ... ", flush=True)
            with torch.no_grad():
                result = run_episode(
                    vc=vc, diff_mtx=diff_mtx, seed=run_idx,
                    presentation=args.presentation, use_chat=args.use_chat,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    save_all_raw=args.save_all_raw,
                    prefill_tail_len=args.inject_turn_len if args.inject_turn else 1,
                    prompt_ver=args.prompt_ver, anchor=args.anchor,
                )
            run_results.append(result)
            print(f"accept_step={result['mean_accept_step']:.2f}  "
                  f"early_stop={result['early_stop_rate']:.2f}  "
                  f"bet%={result['mean_bet']:.1f}  qdm={result['qdm']:.2f}  "
                  f"invalid={result['invalid_rate']:.2f}")
            gc.collect(); torch.cuda.empty_cache()

        row = {"model": args.model, "size": args.size, "alpha": alpha,
               "start": st, "end": en, "presentation": args.presentation,
               "num_runs": args.num_runs}
        for m in METRICS:
            vals = [r[m] for r in run_results
                    if not (isinstance(r[m], float) and np.isnan(r[m]))]
            row[f"mean_{m}"] = round(float(np.mean(vals)), 4) if vals else float("nan")
            row[f"std_{m}"] = round(float(np.std(vals)), 4) if vals else float("nan")
        writer.writerow(row); csv_file.flush()

        out_dir = os.path.join(SAVE_ROOT, f"mdf_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        detail_path = os.path.join(out_dir, f"cgtseq_{args.size}_{st}_{en}_{args.presentation}.json")
        with open(detail_path, "w", encoding="utf-8") as fw:
            json.dump({
                "alpha": alpha,
                "config": {
                    "init_money": INIT_MONEY, "n_phases": N_PHASES,
                    "round_interactions": ROUND_INTERACTIONS,
                    "bets": BETS, "ratios": ALL_RATIOS,
                    "presentation": args.presentation,
                    "use_chat": args.use_chat, "temperature": args.temperature,
                    "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
                    "inject_turn": args.inject_turn,
                    "prefill_tail_len": args.inject_turn_len if args.inject_turn else 1,
                    "prompt_ver": args.prompt_ver, "anchor": args.anchor,
                    "prompt_template": (SEQ_SYSTEM_TEMPLATE_V2
                                        if args.prompt_ver == "v2"
                                        else SEQ_SYSTEM_TEMPLATE),
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
    parser.add_argument("--presentation", type=str, required=True,
                        choices=["asc", "desc"],
                        help="bet-tier reveal order: asc (5→95) or desc (95→5). "
                             "Run BOTH (separate --ans_file) to get "
                             "delay_aversion_index = mean_bet_desc − mean_bet_asc.")
    parser.add_argument("--num_runs",    type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--use_chat",    action="store_true")
    parser.add_argument("--save_all_raw", action="store_true")
    parser.add_argument("--inject_turn", action="store_true",
                        help="steer the last N prompt tokens instead of only the "
                             "last (prefill). Default OFF = tail=1 (the validated "
                             "simultaneous-CGT injection strength).")
    parser.add_argument("--inject_turn_len", type=int, default=4)
    parser.add_argument("--prompt_ver", type=str, default="v1", choices=["v1", "v2", "v3"],
                        help="v1 = validated e55b132 prompt (default, byte-equivalent). "
                             "v2 = symmetrised colour/bet format (Blue/Red ↔ Accept/Wait). "
                             "v3 = v2 base + EXPLICIT next-offer hint at each bet tier "
                             "(A1: 'if you Wait, next bet will be N% (larger/smaller)'), "
                             "so 'wait = bigger bet' is stated, not inferred — lets asc test "
                             "strategic waiting vs pure impulsivity.")
    parser.add_argument("--anchor", type=str, default="default",
                        choices=["default", "answer", "none"],
                        help="prefill anchor for BOTH steps. default = v1 "
                             "(colour='Color: ', bet=''); answer = both 'Answer: '; "
                             "none = both empty.")
    parser.add_argument("--ans_file",    type=str, default="answer_cgtseq_asc")
    parser.add_argument("--data",        type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--base_dir",    type=str, default=None)
    args = parser.parse_args()

    if not args.use_chat:
        parser.error("CGT-Sequential requires --use_chat (real multi-turn accept/wait dialogue).")

    print("Model:", args.model, "| dir:", args.model_dir, "| runs:", args.num_runs)
    BASE = args.base_dir if args.base_dir else f"/{args.data}/paveen/Dopamine/components"
    MASK_DIR  = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_ROOT = os.path.join(BASE, args.model, args.ans_file)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
