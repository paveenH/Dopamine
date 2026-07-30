#!/usr/bin/env python3.10
"""Algorithmic reference curves for the Bandit environment — random / oracle /
UCB1 / Thompson Sampling — on the SAME reward structure, seeds, and arm mapping
the LLM runs use. No model, no GPU; this is a sanity check on the environment
itself, run BEFORE spending any generation budget on a new protocol.

WHY THIS EXISTS. Before asking whether Llama3-8B "does Bandit", we need to know
whether the environment is even learnable in 50 rounds with 5 well-separated
probabilities (0.7/0.5/0.4/0.3/0.1). If UCB1/TS themselves converge slowly or
never separate the middle three arms, a model failing to beat them is not
informative either way.

An OLD UCB1 number exists in prior project notes (K=5, T=50, OptFrac~0.359,
"over-explores") but it predates the 2026-07-28 shuffle_arms/parse_choice fix
and used different bookkeeping — it is NOT a valid reference for this protocol
and must not be cited. This script recomputes from scratch against the CURRENT
environment code.

⚠ FIXED (2026-07-30) — the first version of this script claimed to reuse the
current environment but did NOT:
  (a) `make_arm_map()` reimplemented shuffle_arms()'s two-step shuffle with a
      DIFFERENT rng consumption order, so from seed=1 on it assigned different
      probabilities to different arm identities than get_answer_bandit's own
      `shuffle_arms(seed)` — not the same environment instance.
  (b) reward draws used `random.Random(seed*100_003 + round_idx)`, re-seeding
      EVERY round. That formula is copied from run_episode's
      `torch.manual_seed(seed*100_003 + round_idx)` call, which seeds the
      MODEL's sampling RNG — it has nothing to do with the reward RNG. The
      actual reward stream in run_episode is ONE `rng = random.Random(seed)`
      created once per episode and advanced by exactly one `rng.random()` call
      per round via `get_feedback`. Re-seeding per round draws a DIFFERENT,
      uncorrelated sequence.
  Both are fixed below: K=5 now calls `shuffle_arms` directly (imported, not
  reimplemented) and every algorithm gets its own `random.Random(seed)`
  advanced once per round, matching `get_feedback`'s consumption exactly.

ENVIRONMENT REUSE (K=5 only, load-bearing): `shuffle_arms` is imported
directly from get_answer_bandit.py, so a K=5 baseline here uses EXACTLY the
arm_map a K=5 LLM run would see for the same seed. K=2/K=3 CANNOT reuse it —
shuffle_arms hard-codes `CLOTHES_NAMES[:K]` against the 5-element
REWARD_PROBS_ORDERED, so it has no K<5 mode. K=2/K=3 use a local shuffle that
mirrors shuffle_arms' two-step structure (name/prob assignment, independent
from display order) but is NOT byte-identical to any current LLM protocol,
because no K=2/K=3 LLM prompt exists yet. Treat K=2/K=3 numbers as
environment-difficulty calibration only, not as "the same seed the future
K=2/K=3 LLM run will see" — that guarantee only holds for K=5.

REWARD RNG (load-bearing, now matches get_feedback): ONE `random.Random(seed)`
per episode, advanced by exactly one `.random()` call per round, threshold
against the CHOSEN arm's true probability — identical in mechanism to
`get_feedback(arm, arm_map, rng)`. An algorithm that pulls a different arm at
round t than the LLM did will still consume the RNG identically (one draw per
round regardless of which arm), so the reward STREAM is aligned per seed even
though the specific 0/1 outcomes an algorithm sees depend on which arm it
chose to pull.

Usage:
    python3.10 run_bandit_algorithmic_baseline.py --k 5 --seeds 20 --rounds 50
    python3.10 run_bandit_algorithmic_baseline.py --k 2
    python3.10 run_bandit_algorithmic_baseline.py --k 3
"""

import argparse
import math
import random
import statistics as st

from get_answer_bandit import shuffle_arms as _shuffle_arms_k5

K5_PROBS = [0.7, 0.5, 0.4, 0.3, 0.1]     # == get_answer_bandit.REWARD_PROBS_ORDERED
K3_PROBS = [0.7, 0.45, 0.20]
K2_PROBS = [0.7, 0.3]

PROBS_BY_K = {2: K2_PROBS, 3: K3_PROBS, 5: K5_PROBS}


def make_arm_map(k, seed):
    """{arm_idx: true_prob}. K=5 delegates to the REAL shuffle_arms(seed) (name
    keys mapped to indices in dict order) so it is the exact environment an LLM
    run would see. K=2/K=3 use a local two-step shuffle with the SAME structure
    (probability assignment independent of any display order) since
    shuffle_arms has no K<5 mode — see the module docstring's caveat on what
    this guarantees for K=2/K=3."""
    if k == 5:
        am = _shuffle_arms_k5(seed)          # {name: prob}, in display order
        return {i: prob for i, (_name, prob) in enumerate(am.items())}
    probs = PROBS_BY_K[k][:]
    rng = random.Random(seed)
    order = list(range(k))
    rng.shuffle(order)
    return {arm: probs[i] for arm, i in zip(range(k), order)}


def run_random(arm_map, k, seed, rounds):
    reward_rng = random.Random(seed)              # matches get_feedback's rng
    action_rng = random.Random(seed + 500_000)     # separate stream for the pick
    choices = []
    for t in range(rounds):
        a = action_rng.randrange(k)
        reward_rng.random()  # consume one draw regardless of outcome use, so
                              # the reward STREAM position matches every other
                              # algorithm's at round t (only the threshold
                              # comparison below actually uses the draw)
        choices.append(a)
    return choices


def _draw(reward_rng, arm_map, arm):
    return 1 if reward_rng.random() < arm_map[arm] else 0


def run_ucb1(arm_map, k, seed, rounds):
    """Standard UCB1 with forced round-robin initialization (each arm pulled
    once first) — this is why UCB's coverage is NOT a discovery baseline for
    the LLM comparison: it is handed, not earned."""
    reward_rng = random.Random(seed)
    n = [0] * k
    s = [0.0] * k
    choices = []
    for t in range(rounds):
        if t < k:
            a = t
        else:
            ucb = [s[i] / n[i] + math.sqrt(2 * math.log(t + 1) / n[i])
                   for i in range(k)]
            a = max(range(k), key=lambda i: ucb[i])
        r = _draw(reward_rng, arm_map, a)
        n[a] += 1
        s[a] += r
        choices.append(a)
    return choices


def run_thompson(arm_map, k, seed, rounds):
    """Beta-Bernoulli TS, Beta(1,1) prior. No forced initialization — every
    arm starts with an honest uninformative prior, so TS (unlike UCB1) is a
    fair reference for "how fast can autonomous discovery converge here"."""
    reward_rng = random.Random(seed)
    action_rng = random.Random(seed + 900_000)     # separate stream for the sampler
    alpha = [1.0] * k
    beta = [1.0] * k
    choices = []
    for t in range(rounds):
        samples = [action_rng.betavariate(alpha[i], beta[i]) for i in range(k)]
        a = max(range(k), key=lambda i: samples[i])
        r = _draw(reward_rng, arm_map, a)
        if r:
            alpha[a] += 1
        else:
            beta[a] += 1
        choices.append(a)
    return choices


def run_oracle(arm_map, k, seed, rounds):
    best = max(arm_map, key=arm_map.get)
    return [best] * rounds


def _boot_ci(vals, n_boot=20000, seed=0):
    vals = list(vals)
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(st.mean(rng.choices(vals, k=n)) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def summarize(name, arm_map, choices, rounds):
    best = max(arm_map, key=arm_map.get)
    late_start = max(0, rounds - 30)
    early_end = min(20, rounds)
    coverage = len(set(choices))
    opt = sum(1 for c in choices if c == best) / rounds
    late = sum(1 for c in choices[late_start:] if c == best) / len(choices[late_start:])
    early = sum(1 for c in choices[:early_end] if c == best) / len(choices[:early_end])
    regret = sum(arm_map[best] - arm_map[c] for c in choices) / rounds
    first_best = next((i for i, c in enumerate(choices) if c == best), None)
    return dict(coverage=coverage, opt=opt, early=early, late=late,
               regret=regret, first_best=first_best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, choices=[2, 3, 5])
    ap.add_argument("--seeds", type=int, default=20, help="uses seeds 0..N-1")
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()

    k, rounds = args.k, args.rounds
    seeds = list(range(args.seeds))
    algos = {"random": run_random, "UCB1": run_ucb1,
             "Thompson": run_thompson, "oracle": run_oracle}

    print("=" * 84)
    print(f"ALGORITHMIC BASELINE  K={k}  rounds={rounds}  seeds=0..{args.seeds - 1}"
          f"  probs={PROBS_BY_K[k]}")
    if k == 5:
        print("  arm_map via get_answer_bandit.shuffle_arms(seed) — exact LLM environment")
    else:
        print("  arm_map via a local shuffle (no current LLM protocol at this K — see docstring)")
    print("=" * 84)
    print("NOTE: UCB1's coverage is via FORCED round-robin init, not discovery —")
    print("      compare its OUTCOME numbers to the LLM, not its coverage.")
    print("      Thompson Sampling has no forced init and is the fairer discovery")
    print("      reference. Treat TS as a CALIBRATION reference, not a hard pass/")
    print("      fail threshold — compare the model to it together with CI and")
    print("      policy-mechanism evidence, not as a single number to clear.\n")

    rows = {name: [] for name in algos}
    for seed in seeds:
        arm_map = make_arm_map(k, seed)
        for name, fn in algos.items():
            choices = fn(arm_map, k, seed, rounds)
            rows[name].append(summarize(name, arm_map, choices, rounds))

    hdr = (f'{"algo":<10}{"coverage":>9}{"OptFrac":>9}{"early":>8}'
           f'{"late":>8}{"late 95%CI":>16}{"regret":>9}{"1stBest(med)":>14}')
    print(hdr)
    for name in algos:
        rs = rows[name]
        cov = st.mean(r["coverage"] for r in rs)
        opt = st.mean(r["opt"] for r in rs)
        early = st.mean(r["early"] for r in rs)
        late_vals = [r["late"] for r in rs]
        late = st.mean(late_vals)
        lo, hi = _boot_ci(late_vals)
        regret = st.mean(r["regret"] for r in rs)
        fb = [r["first_best"] for r in rs if r["first_best"] is not None]
        fb_med = st.median(fb) if fb else float("nan")
        ci_str = f"[{lo:.3f},{hi:.3f}]" if lo == lo else "n/a (oracle)"
        print(f'{name:<10}{cov:9.2f}{opt:9.3f}{early:8.3f}{late:8.3f}'
              f'{ci_str:>16}{regret:9.3f}{fb_med:14.1f}')

    print()
    print("Per-seed late-OptFrac spread (algo: min / median / max over seeds):")
    for name in algos:
        lv = sorted(r["late"] for r in rows[name])
        print(f'  {name:<10} {lv[0]:.3f} / {st.median(lv):.3f} / {lv[-1]:.3f}')

    print()
    print("Read this BEFORE any LLM run on this K: if Thompson Sampling itself")
    print("needs most of the horizon to separate the arms (late << 1.0, or a")
    print("late regret far from 0), a model's late OptFrac should be judged")
    print("against TS's late value + CI above, not against 1.0 — and only as")
    print("calibration, alongside regret and the discovery/utilization/stability")
    print("mechanism metrics, never as a standalone pass/fail bar.")


if __name__ == "__main__":
    main()
