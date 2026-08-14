#!/usr/bin/env python3.10
"""PV10 offline environment prescreen.

Answers ONE question: does the candidate BAI environment leave measurable room
for alpha at T=100 -- i.e. is it neither at ceiling nor at chance? It does NOT
select a "good" environment by outcome, and it never touches model data. It runs
before any GPU time is spent and its verdict is frozen into a manifest.

FROZEN ACCEPTANCE CRITERIA (fixed before looking at any number):

    P1: Uniform final accuracy in [0.35, 0.85]
    P2: exists a in {SH, TTTS} with
            Uniform + 0.05 <= accuracy(a) <= 0.95

Both must hold to freeze the environment. Greedy and LUCB are DESCRIPTIVE
reference points and take no part in the verdict:

  * Greedy is a failure-mode comparator, NOT an upper bound. In BAI a greedy
    policy can lock onto an early-noise leader and finish BELOW Uniform, so its
    difference from Uniform is reported without a predicted direction.
  * LUCB answers "when does a fixed-confidence algorithm think it has enough
    evidence", which the PV10 prompt never asks the model for. Heavy censoring
    at n=100 would only mean statistical delta-correctness costs more than the
    budget, while the model reports a looser SUBJECTIVE commitment threshold --
    that is interpretable, not disqualifying. Widening the gap purely to let
    LUCB stop before 100 would risk pushing fixed-budget identification toward
    ceiling, so it is not done.

If a criterion fails, this script reports it. It does not silently retune the
probability vector; any change is a prescreen-driven protocol edit that goes in
the manifest with its reason (allowed, because it happens before any model data
exists).

Usage:
    python3.10 pv10_env_prescreen.py              # run + print report
    python3.10 pv10_env_prescreen.py --freeze     # also write the manifest
    python3.10 pv10_env_prescreen.py --check      # recompute + diff manifest
    python3.10 pv10_env_prescreen.py --selftest   # invariants, no full sim
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bandit_reference import Environment, RewardTape, make_arm_map, best_arm

# ───────────────────────────── frozen constants ─────────────────────────────

MANIFEST_PATH = Path(__file__).with_name("pv10_prescreen_manifest.json")

PRESCREEN_VERSION = "pv10-prescreen-v1"

# Candidate environment under test. K=4 Bernoulli, gaps 0.10 between adjacent
# arms. Written explicitly (NOT in the mu*=0.5+Delta/2 form of the pv6
# reference envs) because PV10 wants a graded ladder, not one best arm against
# a flat field -- the second arm has to be a genuine challenger for
# leader/challenger targeting to be defined at all.
CANDIDATE_PROBS = (0.60, 0.50, 0.40, 0.30)
CANDIDATE_K = 4
TOTAL_BUDGET = 100          # T_max, INCLUDING the forced initialization
FORCED_INIT_PER_ARM = 1     # environment pulls each arm once before any policy
FORCED_INIT = CANDIDATE_K * FORCED_INIT_PER_ARM   # = 4

N_SIM = 10_000

# Simulation seed stream, deliberately DISJOINT from the 20 formal episode
# seeds. The prescreen must never consume or overlap the tapes the real
# experiment will run on.
SIM_SEED_START = 900_000
SIM_SEED_END = SIM_SEED_START + N_SIM   # exclusive

# TTTS internal randomness: an independent stream, so posterior draws never
# consume the shared reward tape and the tape stays paired across algorithms.
TTTS_RNG_OFFSET = 500_000

# Terminal posterior-best estimate (TTTS recommendation only -- NOT per step).
TTTS_FINAL_MC = 10_000

BETA_PRIOR = (1.0, 1.0)     # Beta(1,1)

# LUCB, Kalyanakrishnan et al. (2012) confidence radius.
LUCB_DELTA = 0.1
LUCB_K1 = 5.0 / 4.0

# Sequential Halving, PV10 interface-adapted (see run_sh docstring).
SH_ROUNDS = 2
SH_ROUND_BUDGET = 48        # 4 + 48 + 48 = 100 exactly, no remainder

ACCEPT_UNIFORM_LO = 0.35
ACCEPT_UNIFORM_HI = 0.85
ACCEPT_MARGIN = 0.05
ACCEPT_ADAPTIVE_HI = 0.95


def candidate_environment() -> Environment:
    """The environment under prescreen.

    horizon is TOTAL_BUDGET, but tapes are built longer: LUCB concentrates its
    pulls on two arms, so a single arm can absorb far more than horizon/k draws.
    """
    return Environment(
        name="pv10_bai_candidate",
        k=CANDIDATE_K,
        probs=CANDIDATE_PROBS,
        horizon=TOTAL_BUDGET,
        is_reference=False,
        competence_eligible=False,
    )


# ───────────────────────────── shared state ─────────────────────────────────

@dataclass
class ArmState:
    """Cumulative counts for one arm, INCLUDING the forced initialization.

    The forced init observations count toward every empirical mean and every
    posterior. Excluding them would make the algorithms see a different history
    than the model does, which is the whole point of interface-adapting them.
    """
    successes: int = 0
    trials: int = 0

    @property
    def mean(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    def posterior(self) -> tuple[float, float]:
        a0, b0 = BETA_PRIOR
        return a0 + self.successes, b0 + (self.trials - self.successes)


class Run:
    """One simulated episode: shared tape + per-arm counts + a sample budget.

    `display_order` is the arm order shown in OPTIONS and is the deterministic
    tie-break for every algorithm-internal choice (pull / elimination / final
    recommendation). It is NOT the tie-tolerant convention used for analysis
    readouts -- those report a band instead.
    """

    def __init__(self, seed: int, env: Environment, tape_length: int):
        self.seed = seed
        self.env = env
        self.tape = RewardTape(seed, env, length=tape_length)
        self.arm_map = self.tape.arm_map
        self.display_order = list(self.arm_map)
        self.true_best = best_arm(self.arm_map)
        self.state = {name: ArmState() for name in self.display_order}
        self.n = 0
        self.history: list[str] = []

    def pull(self, arm: str) -> int:
        if self.n >= TOTAL_BUDGET:
            raise RuntimeError(
                f"budget exhausted: attempted pull {self.n + 1} > {TOTAL_BUDGET}")
        r = self.tape.pull(arm)
        st = self.state[arm]
        st.successes += r
        st.trials += 1
        self.n += 1
        self.history.append(arm)
        return r

    def forced_init(self) -> None:
        """Environment pulls each arm once, in display order.

        NOTE: this fixed display-order sweep is the PRESCREEN convention. The
        real PV10-B protocol randomizes `initial_pull_order` per seed and pairs
        it across alpha cells. The distinction does not affect the prescreen,
        whose algorithms are all order-invariant given the same per-arm tape:
        each arm's first observation is tape.peek(arm, 0) no matter when it is
        drawn.
        """
        for name in self.display_order:
            for _ in range(FORCED_INIT_PER_ARM):
                self.pull(name)

    # -- deterministic tie-breaks, all resolved by display order --------------

    def _argmax_display(self, score) -> str:
        best_name, best_val = None, -math.inf
        for name in self.display_order:   # display order = tie-break
            v = score(name)
            if v > best_val:
                best_name, best_val = name, v
        return best_name

    def empirical_best(self, among: list[str] | None = None) -> str:
        pool = among if among is not None else self.display_order
        best_name, best_val = None, -math.inf
        for name in self.display_order:
            if name not in pool:
                continue
            v = self.state[name].mean
            if v > best_val:
                best_name, best_val = name, v
        return best_name

    def remaining(self) -> int:
        return TOTAL_BUDGET - self.n


def _result(run: Run, recommended: str, *, tau: int, stopped: bool,
            censored: bool, extra: dict | None = None) -> dict:
    """Package one episode.

    `tau` is always a TOTAL SAMPLE COUNT including the forced initialization,
    never a round index -- LUCB pulls two arms per round, so round counts are
    not comparable across algorithms.
    """
    probs = run.arm_map
    out = {
        "seed": run.seed,
        "recommended": recommended,
        "true_best": run.true_best,
        "correct": int(recommended == run.true_best),
        "simple_regret": probs[run.true_best] - probs[recommended],
        "tau": tau,
        "stopped": bool(stopped),
        "censored": bool(censored),
        "pulls": {name: run.state[name].trials for name in run.display_order},
        "best_position": run.display_order.index(run.true_best) + 1,
    }
    if extra:
        out.update(extra)
    return out


# ───────────────────────────── algorithms ───────────────────────────────────

def run_uniform(seed: int, env: Environment) -> dict:
    """Equal allocation, then recommend the cumulative empirical best.

    Chance/lower reference point. 4 forced + 96 remaining = 24 extra per arm,
    exactly divisible, so there is no remainder rule to freeze.
    """
    run = Run(seed, env, tape_length=TOTAL_BUDGET)
    run.forced_init()
    per_arm, rem = divmod(run.remaining(), env.k)
    assert rem == 0, f"uniform expects an exact split, got remainder {rem}"
    for _ in range(per_arm):
        for name in run.display_order:
            run.pull(name)
    assert run.n == TOTAL_BUDGET
    return _result(run, run.empirical_best(), tau=TOTAL_BUDGET,
                   stopped=False, censored=False)


def run_greedy(seed: int, env: Environment) -> dict:
    """Always pull the current cumulative empirical best.

    FAILURE-MODE comparator, not an upper bound: an early-noise leader can lock
    it out of the true best arm for the whole horizon, so it may finish below
    Uniform. Direction is not predicted.
    """
    run = Run(seed, env, tape_length=TOTAL_BUDGET)
    run.forced_init()
    while run.n < TOTAL_BUDGET:
        run.pull(run.empirical_best())
    return _result(run, run.empirical_best(), tau=TOTAL_BUDGET,
                   stopped=False, censored=False)


def run_sh(seed: int, env: Environment) -> dict:
    """Sequential Halving, PV10 interface-adapted (Karnin et al., ICML 2013).

    NOT a verbatim reproduction: this adds the forced initialization and scores
    on CUMULATIVE empirical means (init observations included), so that the
    algorithm sees exactly the history the model sees.

    Exact frozen schedule for K=4, T=100 -- no remainder, no rounding rule:

        forced init : 4 arms x 1                   =  4 pulls  (cum 1 each)
        round 1     : 4 arms x 12                  = 48 pulls  (cum 13 each)
                      keep the top 2 by cumulative mean
        round 2     : 2 arms x 24                  = 48 pulls  (cum 37 each)
        recommend   : cumulative-best survivor

    Total 4 + 48 + 48 = 100. Elimination and recommendation tie-break on
    display order.
    """
    run = Run(seed, env, tape_length=TOTAL_BUDGET)
    run.forced_init()

    survivors = list(run.display_order)
    for _ in range(SH_ROUNDS):
        per_arm, rem = divmod(SH_ROUND_BUDGET, len(survivors))
        assert rem == 0, (
            f"SH round budget {SH_ROUND_BUDGET} not divisible by "
            f"{len(survivors)} survivors -- the frozen K=4/T=100 schedule "
            f"guarantees this; a different K or budget needs a new frozen rule")
        for _ in range(per_arm):
            for name in survivors:
                run.pull(name)
        if len(survivors) > 1:
            keep = max(1, len(survivors) // 2)
            ranked = sorted(
                survivors,
                key=lambda nm: (-run.state[nm].mean, run.display_order.index(nm)))
            survivors = [nm for nm in run.display_order if nm in set(ranked[:keep])]

    assert run.n == TOTAL_BUDGET, f"SH used {run.n} != {TOTAL_BUDGET}"
    assert len(survivors) == 1, f"SH ended with {len(survivors)} survivors"
    return _result(run, survivors[0], tau=TOTAL_BUDGET,
                   stopped=False, censored=False,
                   extra={"sh_final_survivor_pulls": run.state[survivors[0]].trials})


def _ttts_posterior_argmax(run: Run, rng: np.random.Generator) -> str:
    """One joint posterior draw over all arms; argmax with display tie-break."""
    best_name, best_val = None, -math.inf
    for name in run.display_order:
        a, b = run.state[name].posterior()
        theta = rng.beta(a, b)
        if theta > best_val:
            best_name, best_val = name, theta
    return best_name


def _posterior_best_probs(run: Run, rng: np.random.Generator,
                          n_mc: int) -> dict[str, float]:
    """Monte-Carlo P(arm is best), with STRUCTURAL ties collapsed.

    Beta(1,1) makes ties structural, not incidental: every 0/1 arm sits at
    exactly the same posterior and every untried arm at exactly the same one.
    MC noise would invent a difference between such arms and let it decide the
    recommendation. So arms sharing identical (successes, trials) are pooled and
    assigned the group's mean probability; the display-order tie-break then
    resolves them, exactly as everywhere else.
    """
    names = run.display_order
    draws = np.empty((n_mc, len(names)))
    for j, name in enumerate(names):
        a, b = run.state[name].posterior()
        draws[:, j] = rng.beta(a, b, size=n_mc)
    winners = draws.argmax(axis=1)
    counts = np.bincount(winners, minlength=len(names)).astype(float) / n_mc
    probs = {name: counts[j] for j, name in enumerate(names)}

    groups: dict[tuple[int, int], list[str]] = {}
    for name in names:
        st = run.state[name]
        groups.setdefault((st.successes, st.trials), []).append(name)
    for members in groups.values():
        if len(members) > 1:
            shared = sum(probs[m] for m in members) / len(members)
            for m in members:
                probs[m] = shared
    return probs


def run_ttts(seed: int, env: Environment) -> dict:
    """Top-Two Thompson Sampling, beta=0.5.

    NOTE ON THE NAME: this is TTTS, not TTPS. TTTS draws a posterior SAMPLE to
    get the leader and re-draws until a different arm appears as the challenger.
    True TTPS instead computes each arm's posterior probability of being best
    every step and takes the top two of those. An earlier draft of this
    prescreen called this algorithm TTPS; the algorithm is kept and the name
    corrected.

    Per step:
      * joint posterior draw -> I1 = argmax theta
      * with probability beta: pull I1
      * else: re-draw jointly until argmax != I1, pull that I2

    The 10,000-sample MC estimate runs ONCE at the end for the recommendation,
    not per step -- per-step MC would cost ~3.8e10 Beta variates for no gain,
    since the action rule needs only a single draw.
    """
    run = Run(seed, env, tape_length=TOTAL_BUDGET)
    run.forced_init()
    rng = np.random.default_rng(TTTS_RNG_OFFSET + seed)

    while run.n < TOTAL_BUDGET:
        i1 = _ttts_posterior_argmax(run, rng)
        if rng.random() < 0.5:
            run.pull(i1)
            continue
        # Re-draw until the argmax differs. With continuous Beta draws this
        # terminates with probability 1; the cap only guards a degenerate
        # posterior (e.g. an arm at Beta(1,1) vs an arm with huge counts).
        i2 = None
        for _ in range(1000):
            cand = _ttts_posterior_argmax(run, rng)
            if cand != i1:
                i2 = cand
                break
        run.pull(i2 if i2 is not None else i1)

    final_rng = np.random.default_rng(TTTS_RNG_OFFSET + 7_000_000 + seed)
    pb = _posterior_best_probs(run, final_rng, TTTS_FINAL_MC)
    rec = run._argmax_display(lambda nm: pb[nm])
    return _result(run, rec, tau=TOTAL_BUDGET, stopped=False, censored=False,
                   extra={"ttts_posterior_best": pb[rec]})


def _lucb_radius(u: int, t: int, k: int, delta: float) -> float:
    """Kalyanakrishnan et al. (2012): sqrt( ln(k1*K*t^4/delta) / (2u) )."""
    if u <= 0:
        return math.inf
    return math.sqrt(math.log(LUCB_K1 * k * (t ** 4) / delta) / (2.0 * u))


def run_lucb(seed: int, env: Environment, delta: float = LUCB_DELTA) -> dict:
    """LUCB fixed-confidence stopping, DESCRIPTIVE reference only.

    Frozen loop semantics:
      * t is the LUCB decision round, starting at t=1 after the forced init
      * b(t) = highest cumulative empirical mean (display tie-break)
      * c(t) = argmax of U_k over the other arms (display tie-break)
      * bounds clipped to [0, 1]
      * stopping is CHECKED FIRST, then b and c are pulled
      * a pull adds 2 to n and 1 to t
      * on reaching n=100 a FINAL stopping check runs:
            satisfied     -> tau=100, stopped=True
            not satisfied -> censored=True
      * a censored episode still reports the n=100 empirical best as a
        descriptive recommendation, but that is NOT a self-initiated stop and
        must not be pooled with stopped episodes
      * tau is a total sample count including the forced initialization

    A censored run is an interpretable outcome (delta-correctness costs more
    than the budget), not an environment failure -- which is exactly why LUCB
    does not gate the verdict.
    """
    run = Run(seed, env, tape_length=TOTAL_BUDGET)
    run.forced_init()
    k = env.k
    t = 1

    def bounds(name: str) -> tuple[float, float]:
        st = run.state[name]
        r = _lucb_radius(st.trials, t, k, delta)
        return (max(0.0, st.mean - r), min(1.0, st.mean + r))

    def stop_now() -> bool:
        b = run.empirical_best()
        lb = bounds(b)[0]
        ub_rest = max(bounds(nm)[1] for nm in run.display_order if nm != b)
        return lb > ub_rest

    while True:
        if stop_now():
            return _result(run, run.empirical_best(), tau=run.n,
                           stopped=True, censored=False,
                           extra={"lucb_rounds": t - 1})
        if run.remaining() < 2:
            break
        b = run.empirical_best()
        c = run._argmax_display(
            lambda nm: bounds(nm)[1] if nm != b else -math.inf)
        run.pull(b)
        run.pull(c)
        t += 1

    # Final check at the budget boundary.
    if stop_now():
        return _result(run, run.empirical_best(), tau=run.n,
                       stopped=True, censored=False,
                       extra={"lucb_rounds": t - 1})
    return _result(run, run.empirical_best(), tau=run.n,
                   stopped=False, censored=True,
                   extra={"lucb_rounds": t - 1})


ALGORITHMS = {
    "Uniform": run_uniform,
    "Greedy": run_greedy,
    "SH": run_sh,
    "TTTS": run_ttts,
    "LUCB": run_lucb,
}

# Only these two decide the verdict.
GATING_ALGORITHMS = ("Uniform", "SH", "TTTS")
ADAPTIVE_ALGORITHMS = ("SH", "TTTS")


# ───────────────────────────── simulation ───────────────────────────────────

def _mean_ci(values: list[float], n_boot: int = 2000,
             seed: int = 12345) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return m, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def simulate(n_sim: int = N_SIM, verbose: bool = True) -> dict:
    env = candidate_environment()
    seeds = list(range(SIM_SEED_START, SIM_SEED_START + n_sim))
    out: dict[str, list[dict]] = {name: [] for name in ALGORITHMS}

    for i, seed in enumerate(seeds):
        for name, fn in ALGORITHMS.items():
            out[name].append(fn(seed, env))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{n_sim} seeds", flush=True)

    summary = {}
    for name, results in out.items():
        acc, lo, hi = _mean_ci([r["correct"] for r in results])
        regret = float(np.mean([r["simple_regret"] for r in results]))
        stopped = [r for r in results if r["stopped"]]
        censored = [r for r in results if r["censored"]]
        pulls = {
            a: float(np.mean([r["pulls"][a] for r in results]))
            for a in out[name][0]["pulls"]
        }
        entry = {
            "accuracy": acc,
            "accuracy_ci": [lo, hi],
            "simple_regret": regret,
            "n_stopped": len(stopped),
            "n_censored": len(censored),
            "mean_pulls_by_display_position": [
                float(np.mean([list(r["pulls"].values())[j] for r in results]))
                for j in range(env.k)
            ],
        }
        if name == "LUCB":
            entry["p_stop_within_budget"] = len(stopped) / len(results)
            entry["censoring_rate"] = len(censored) / len(results)
            if stopped:
                taus = [r["tau"] for r in stopped]
                entry["tau_stopped_median"] = float(np.median(taus))
                entry["tau_stopped_mean"] = float(np.mean(taus))
                entry["tau_stopped_q25"] = float(np.percentile(taus, 25))
                entry["tau_stopped_q75"] = float(np.percentile(taus, 75))
            # A median over ALL episodes is only defined if a majority stopped;
            # otherwise it is reported as not reached rather than pinned to 100.
            if len(stopped) / len(results) > 0.5:
                entry["tau_median_all"] = float(
                    np.median([r["tau"] for r in results]))
            else:
                entry["tau_median_all"] = ">100 / not reached"
            entry["survival_curve"] = {
                str(n): float(np.mean([r["tau"] > n or r["censored"]
                                       for r in results]))
                for n in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
            }
            entry["accuracy_stopped_only"] = (
                float(np.mean([r["correct"] for r in stopped])) if stopped else None)
        summary[name] = entry

    return {"env": env, "summary": summary, "raw": out, "seeds": seeds}


def evaluate_criteria(summary: dict) -> dict:
    """P1 and P2 only. Greedy and LUCB are descriptive and never gate."""
    u = summary["Uniform"]["accuracy"]
    p1 = ACCEPT_UNIFORM_LO <= u <= ACCEPT_UNIFORM_HI

    p2_detail = {}
    p2 = False
    for name in ADAPTIVE_ALGORITHMS:
        a = summary[name]["accuracy"]
        ok = (u + ACCEPT_MARGIN) <= a <= ACCEPT_ADAPTIVE_HI
        p2_detail[name] = {"accuracy": a, "lift_over_uniform": a - u, "passes": ok}
        p2 = p2 or ok

    return {
        "P1": {
            "description": (
                f"Uniform accuracy in [{ACCEPT_UNIFORM_LO}, {ACCEPT_UNIFORM_HI}]"),
            "value": u,
            "passes": p1,
        },
        "P2": {
            "description": (
                f"exists a in {list(ADAPTIVE_ALGORITHMS)} with "
                f"Uniform+{ACCEPT_MARGIN} <= acc(a) <= {ACCEPT_ADAPTIVE_HI}"),
            "per_algorithm": p2_detail,
            "passes": p2,
        },
        "verdict": "PASS" if (p1 and p2) else "FAIL",
    }


# ───────────────────────────── reporting ────────────────────────────────────

def report(sim: dict) -> dict:
    env = sim["env"]
    s = sim["summary"]
    crit = evaluate_criteria(s)

    print()
    print("=" * 78)
    print(f"PV10 ENVIRONMENT PRESCREEN  ({PRESCREEN_VERSION})")
    print("=" * 78)
    print(f"env            : {env.name}  K={env.k}  probs={env.probs}")
    print(f"budget         : T_max={TOTAL_BUDGET} "
          f"(forced init {FORCED_INIT}, adaptive {TOTAL_BUDGET - FORCED_INIT})")
    print(f"simulations    : {len(sim['seeds'])}  "
          f"seeds [{SIM_SEED_START}, {SIM_SEED_END}) "
          f"-- disjoint from the formal episode seeds")
    print()

    print("-" * 78)
    print("GATING ALGORITHMS (decide the verdict)")
    print("-" * 78)
    print(f"{'algorithm':<10} {'accuracy':>10} {'95% CI':>18} {'simple regret':>15}")
    for name in GATING_ALGORITHMS:
        e = s[name]
        ci = f"[{e['accuracy_ci'][0]:.3f}, {e['accuracy_ci'][1]:.3f}]"
        print(f"{name:<10} {e['accuracy']:>10.4f} {ci:>18} "
              f"{e['simple_regret']:>15.4f}")

    print()
    print("-" * 78)
    print("DESCRIPTIVE REFERENCES (never gate the verdict)")
    print("-" * 78)
    g = s["Greedy"]
    u = s["Uniform"]["accuracy"]
    ci = f"[{g['accuracy_ci'][0]:.3f}, {g['accuracy_ci'][1]:.3f}]"
    print(f"{'Greedy':<10} {g['accuracy']:>10.4f} {ci:>18} "
          f"{g['simple_regret']:>15.4f}")
    print(f"           Greedy - Uniform = {g['accuracy'] - u:+.4f}  "
          f"(direction NOT predicted; Greedy is a failure-mode comparator)")

    lu = s["LUCB"]
    print()
    print(f"LUCB (delta={LUCB_DELTA}, k1={LUCB_K1}, Kalyanakrishnan et al. 2012)")
    print(f"           P(tau <= {TOTAL_BUDGET})   = {lu['p_stop_within_budget']:.4f}")
    print(f"           censoring rate    = {lu['censoring_rate']:.4f}")
    print(f"           tau median (all)  = {lu['tau_median_all']}")
    if lu["n_stopped"]:
        print(f"           tau | stopped     = median {lu['tau_stopped_median']:.1f}"
              f"  IQR [{lu['tau_stopped_q25']:.1f}, {lu['tau_stopped_q75']:.1f}]"
              f"  (n={lu['n_stopped']})")
        print(f"           accuracy | stopped= {lu['accuracy_stopped_only']:.4f}")
    print(f"           survival P(still sampling after n):")
    print("           " + "  ".join(
        f"{n}:{v:.2f}" for n, v in lu["survival_curve"].items()))

    print()
    print("-" * 78)
    print("MEAN PULLS BY DISPLAY POSITION")
    print("-" * 78)
    print(f"{'algorithm':<10} " + "  ".join(f"{'pos'+str(j+1):>8}" for j in range(env.k)))
    for name in ALGORITHMS:
        row = s[name]["mean_pulls_by_display_position"]
        print(f"{name:<10} " + "  ".join(f"{v:>8.1f}" for v in row))

    print()
    print("=" * 78)
    print("FROZEN ACCEPTANCE CRITERIA")
    print("=" * 78)
    p1 = crit["P1"]
    print(f"P1  {p1['description']}")
    print(f"    Uniform = {p1['value']:.4f}   ->  {'PASS' if p1['passes'] else 'FAIL'}")
    p2 = crit["P2"]
    print(f"P2  {p2['description']}")
    for name, d in p2["per_algorithm"].items():
        print(f"    {name:<6} acc={d['accuracy']:.4f}  "
              f"lift={d['lift_over_uniform']:+.4f}  "
              f"->  {'PASS' if d['passes'] else 'fail'}")
    print(f"    {'PASS' if p2['passes'] else 'FAIL'}")
    print()
    print(f"VERDICT: {crit['verdict']}")
    if crit["verdict"] == "FAIL":
        print()
        print("  A failing criterion is REPORTED, not silently retuned. Any change")
        print("  to the probability vector is a prescreen-driven protocol edit that")
        print("  must be recorded in the manifest with its reason.")
    print("=" * 78)
    print()
    return crit


# ───────────────────────────── manifest ─────────────────────────────────────

def build_manifest(sim: dict, crit: dict) -> dict:
    env = sim["env"]
    s = {k: {kk: vv for kk, vv in v.items()} for k, v in sim["summary"].items()}
    return {
        "prescreen_version": PRESCREEN_VERSION,
        "environment": {
            "name": env.name,
            "k": env.k,
            "probs": list(env.probs),
            "total_budget": TOTAL_BUDGET,
            "forced_init_per_arm": FORCED_INIT_PER_ARM,
            "forced_init_total": FORCED_INIT,
        },
        "simulation": {
            "n_sim": len(sim["seeds"]),
            "seed_start": SIM_SEED_START,
            "seed_end": SIM_SEED_END,
            "beta_prior": list(BETA_PRIOR),
        },
        "algorithms": {
            "Uniform": {"role": "gating", "spec": "equal allocation, 24 extra per arm"},
            "Greedy": {"role": "descriptive",
                       "spec": "always pull cumulative empirical best; "
                               "failure-mode comparator, direction not predicted"},
            "SH": {"role": "gating",
                   "spec": "PV10 interface-adapted Sequential Halving: "
                           "4 init + 4x12 + 2x24, cumulative means, "
                           "display-order tie-break",
                   "rounds": SH_ROUNDS, "round_budget": SH_ROUND_BUDGET},
            "TTTS": {"role": "gating",
                     "spec": "Top-Two Thompson Sampling beta=0.5 (NOT TTPS); "
                             "single posterior draw per action, "
                             f"{TTTS_FINAL_MC}-sample MC once at the end for the "
                             "recommendation, structural ties pooled",
                     "beta": 0.5, "rng_offset": TTTS_RNG_OFFSET},
            "LUCB": {"role": "descriptive",
                     "spec": "Kalyanakrishnan et al. (2012) radius "
                             "sqrt(ln(k1*K*t^4/delta)/(2u)); check-then-pull; "
                             "2 pulls per round; tau in total samples",
                     "delta": LUCB_DELTA, "k1": LUCB_K1},
        },
        "acceptance_criteria": {
            "P1": f"Uniform accuracy in [{ACCEPT_UNIFORM_LO}, {ACCEPT_UNIFORM_HI}]",
            "P2": (f"exists a in {list(ADAPTIVE_ALGORITHMS)} with "
                   f"Uniform+{ACCEPT_MARGIN} <= acc(a) <= {ACCEPT_ADAPTIVE_HI}"),
            "note": "Greedy and LUCB are descriptive and never gate.",
        },
        "results": s,
        "criteria_evaluation": crit,
    }


def _round_trip(obj):
    return json.loads(json.dumps(obj))


# ───────────────────────────── selftest ─────────────────────────────────────

def selftest() -> None:
    env = candidate_environment()
    seed = SIM_SEED_START

    # Budget: every algorithm respects T_max, and the non-stopping ones use it
    # exactly.
    for name in ("Uniform", "Greedy", "SH", "TTTS"):
        r = ALGORITHMS[name](seed, env)
        assert sum(r["pulls"].values()) == TOTAL_BUDGET, (
            f"{name} used {sum(r['pulls'].values())} != {TOTAL_BUDGET}")
    r = ALGORITHMS["LUCB"](seed, env)
    assert sum(r["pulls"].values()) <= TOTAL_BUDGET

    # SH schedule: survivor ends on exactly 37 cumulative pulls.
    r = run_sh(seed, env)
    assert r["sh_final_survivor_pulls"] == 37, r["sh_final_survivor_pulls"]
    assert max(r["pulls"].values()) == 37

    # Uniform is exactly balanced.
    r = run_uniform(seed, env)
    assert set(r["pulls"].values()) == {25}, r["pulls"]

    # Shared tapes: the n-th pull of an arm is the same reward for every
    # algorithm, regardless of when it happens.
    t1 = RewardTape(seed, env, length=TOTAL_BUDGET)
    t2 = RewardTape(seed, env, length=TOTAL_BUDGET)
    for arm in t1.arm_map:
        assert [t1.peek(arm, j) for j in range(20)] == \
               [t2.peek(arm, j) for j in range(20)]

    # Determinism: same seed -> same result, twice.
    for name, fn in ALGORITHMS.items():
        assert fn(seed, env) == fn(seed, env), f"{name} is not deterministic"

    # Prescreen seeds must not collide with the formal episode seeds.
    from bandit_reference import build_seed_bank, get_environment
    formal = set(build_seed_bank(get_environment("easy"), n=20))
    assert not (formal & set(range(SIM_SEED_START, SIM_SEED_END))), (
        "prescreen seed stream overlaps the formal episode seed bank")

    # LUCB tau is a total sample count including the forced init, and pulls two
    # arms per round.
    r = run_lucb(seed, env)
    assert r["tau"] == sum(r["pulls"].values())
    assert r["tau"] >= FORCED_INIT
    assert (r["tau"] - FORCED_INIT) % 2 == 0, "LUCB pulls two arms per round"
    assert r["stopped"] != r["censored"], "stopped and censored must be exclusive"

    # Structural-tie pooling: at the forced-init state, arms with identical
    # (successes, trials) must receive identical posterior-best probability.
    run = Run(seed, env, tape_length=TOTAL_BUDGET)
    run.forced_init()
    pb = _posterior_best_probs(run, np.random.default_rng(0), 2000)
    groups: dict[tuple[int, int], list[str]] = {}
    for nm in run.display_order:
        st = run.state[nm]
        groups.setdefault((st.successes, st.trials), []).append(nm)
    for members in groups.values():
        vals = {round(pb[m], 12) for m in members}
        assert len(vals) == 1, f"structural tie not pooled: {members} -> {vals}"

    # Display-order tie-break: with all arms identical, pick the first shown.
    run = Run(seed, env, tape_length=TOTAL_BUDGET)
    assert run.empirical_best() == run.display_order[0]

    print("selftest: all invariants hold")


# ───────────────────────────── main ─────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_sim", type=int, default=N_SIM)
    ap.add_argument("--freeze", action="store_true",
                    help="write the manifest after a PASS verdict")
    ap.add_argument("--check", action="store_true",
                    help="recompute and diff against the stored manifest")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    print(f"simulating {args.n_sim} seeds x {len(ALGORITHMS)} algorithms ...")
    sim = simulate(args.n_sim)
    crit = report(sim)

    if args.check:
        if not MANIFEST_PATH.exists():
            raise SystemExit(f"no manifest at {MANIFEST_PATH}; run --freeze first")
        stored = json.loads(MANIFEST_PATH.read_text())
        fresh = _round_trip(build_manifest(sim, crit))
        if stored == fresh:
            print(f"MANIFEST CHECK: OK ({MANIFEST_PATH.name} reproduces)")
        else:
            diffs = [k for k in set(stored) | set(fresh)
                     if stored.get(k) != fresh.get(k)]
            raise SystemExit(
                f"MANIFEST MISMATCH in {diffs}; the frozen basis does not "
                f"reproduce on this machine -- nothing downstream is citable")
        return

    if args.freeze:
        if crit["verdict"] != "PASS":
            raise SystemExit(
                "refusing to freeze a FAILING environment; report the failure "
                "and decide on a protocol edit first")
        MANIFEST_PATH.write_text(
            json.dumps(build_manifest(sim, crit), indent=2, sort_keys=True))
        print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
