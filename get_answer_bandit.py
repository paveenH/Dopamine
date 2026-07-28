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
import csv
import json
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


def build_prompt(
    arm_names: list[str],
    history: list[tuple[str, int]],
    use_role: bool = True,
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
    """
    names_str = "[" + ", ".join(arm_names) + "]"
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
    if use_role:
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
        lines.append(f"So far you have interacted {n} times with the following choices and rewards:")
        for name, reward in history:
            lines.append(f"{name} {ACTION_UNIT}, reward {reward}")
        lines.append("")

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


def get_feedback(arm: str, arm_map: dict[str, float], rng: random.Random) -> int:
    return 1 if rng.random() < arm_map[arm] else 0


def run_episode(
    vc: VicundaModel,
    diff_mtx,
    num_rounds: int,
    seed: int,
    use_role: bool = True,
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

    for round_idx in range(num_rounds):
        prompt = build_prompt(arm_names, history, use_role=use_role)
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
            max_new_tokens=20,
            temperature=1.0,
        )
        raw = output[0] if isinstance(output, list) else output
        arm, valid, nmatch = parse_choice(raw, arm_names, rng=fallback_rng)

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

    return {
        "seed":           seed,
        "arm_map":        {k: v for k, v in arm_map.items()},
        "best_arm":       opt,
        "worst_arm":      worst,
        # 1-based display position of the best arm — lets the analysis verify
        # position counterbalancing and test for residual first-option bias.
        "best_position":  arm_names.index(opt) + 1,
        "arm_order":      arm_names,
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
    }


def main():
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("Configs:", ALPHAS_START_END_PAIRS)
    print(f"Reward probs (assigned after shuffle): {REWARD_PROBS_ORDERED}")
    print(f"Arm names pool (first {K}): {CLOTHES_NAMES[:K]}")

    FIELDNAMES = [
        "model", "size", "alpha", "start", "end", "TOP",
        "num_runs", "num_rounds",
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
    ]

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, f"summary_{args.model}_{args.size}.csv")

    # Resume key includes the settings that change what a row MEANS. Keying on
    # alpha alone silently skipped a cell when the layer range, run count or
    # round count changed within the same output dir — you would get the old
    # row back and never notice the new configuration had not run.
    def _resume_key(alpha, st, en, nruns, nrounds):
        return (float(alpha), int(st), int(en), int(nruns), int(nrounds))

    done_keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    done_keys.add(_resume_key(r["alpha"], r["start"], r["end"],
                                              r["num_runs"], r["num_rounds"]))
                except (KeyError, ValueError):
                    # Legacy row from before the key was widened: fall back to
                    # alpha-only so old dirs still resume rather than crash.
                    done_keys.add((float(r["alpha"]), None, None, None, None))
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
        if key in done_keys or (float(alpha), None, None, None, None) in done_keys:
            print(f"[Skip] α={alpha} layers={st}-{en} "
                  f"runs={args.num_runs}x{args.num_rounds} already done.")
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
                    vc=vc,
                    diff_mtx=diff_mtx,
                    num_rounds=args.num_rounds,
                    seed=run_idx,
                    use_role=not args.no_role,
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

        row = {
            "model": args.model,
            "size":  args.size,
            "alpha": alpha,
            "start": st,
            "end":   en,
            "TOP":   TOP,
            "num_runs":   args.num_runs,
            "num_rounds": args.num_rounds,
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
