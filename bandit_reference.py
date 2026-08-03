#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Shared implementation of the pv6 `F-reference` Bandit protocol.

Everything here is used by BOTH entry points:

    get_answer_bandit.py            model loading, RSN hook, orchestration
    run_bandit_algorithmic_baseline.py   Random / Greedy / UCB1 / TS / Oracle

That shared use is the REASON this module exists, not the line count of
get_answer_bandit.py. If each entry point built its own environment, the
algorithmic baselines would silently run on a different instance than the LLM
did — which is exactly what the per-arm reward tapes in
`BanditExperiment_LiteratureReview.md` §3.5 exist to prevent, and exactly the
bug that was found and fixed in run_bandit_algorithmic_baseline.py on
2026-07-30 (it had reimplemented `shuffle_arms` with a different RNG
consumption order and re-seeded the reward stream every round).

This module owns, per §3.9:
  - reference environment specifications
  - counterbalanced seed banks (formal + smoke)
  - per-arm reward tapes
  - F-reference prompt construction
  - rationale sanitization
  - candidate / tokenization utilities
  - reference metrics (SuffFailFreq, K x MinFrac, GreedyFrac)

It owns NO model code, NO steering, and does NOT import get_answer_bandit —
the dependency runs one way only, so pv1-pv5 stay untouched.

Design is FROZEN as of 2026-08-02 (see §3 PLAN). Changing an environment
probability, a seed bank, a tie-break rule or a metric definition here
invalidates every stored pv6 cell. Bump PROTOCOL_VERSION if that happens.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# pv6. Distinct from get_answer_bandit's pv1-pv5 so the resume key can never
# collide with a legacy cell. `graded` is deliberately NOT a pv6 environment
# (§3.9): the legacy .7/.5/.4/.3/.1 vector stays on the legacy path only.
PROTOCOL_VERSION = "pv6"

# Neutral arm labels. "Button" matches the Krishnamurthy et al. buttons
# scenario, which is the frame their one successful configuration used. The
# letters carry no reward information and, unlike the EVOLvE clothes names, no
# semantic prior that could interact with the model's preferences.
ARM_LABELS = ["Button A", "Button B", "Button C", "Button D", "Button E"]

# Offset for the Greedy tie-break RNG (§3.5). Must not coincide with any other
# stream's seeding: reward tapes are seeded per (seed, arm), the model's
# sampling uses seed*100_003 + round_idx, and legacy fallback uses
# 1_000_000 + seed. This constant is part of the frozen spec — changing it
# changes every stored Greedy baseline.
TIE_RNG_OFFSET = 2_000_000


# ─────────────────────────── environments ───────────────────────────────────

@dataclass(frozen=True)
class Environment:
    """A frozen reference environment.

    `probs` is in DESCENDING order and is NOT the display order — it is the
    probability multiset. Which arm identity gets which probability, and the
    order arms are displayed in, are both decided per-seed by `make_arm_map`.
    """
    name: str
    k: int
    probs: tuple[float, ...]
    horizon: int
    is_reference: bool          # False = not from Krishnamurthy et al.
    competence_eligible: bool   # may this environment anchor a competence gate?

    def __post_init__(self):
        assert len(self.probs) == self.k, f"{self.name}: |probs| != k"
        assert list(self.probs) == sorted(self.probs, reverse=True), (
            f"{self.name}: probs must be descending")
        assert self.probs[0] > self.probs[1], f"{self.name}: best arm not unique"


# Krishnamurthy et al. (NeurIPS 2024) parameterize their instances as
#     mu* = 0.5 + Delta/2,  mu = 0.5 - Delta/2
# so Delta=0.5 gives .75/.25 and Delta=0.2 gives .60/.40. Writing the vectors
# any other way (e.g. .5/.3/.3/.3) silently changes the gap and breaks
# comparability with the published numbers.
ENVIRONMENTS: dict[str, Environment] = {
    "easy": Environment(
        name="reference_easy", k=4,
        probs=(0.75, 0.25, 0.25, 0.25), horizon=100,
        is_reference=True, competence_eligible=True,
    ),
    "hard": Environment(
        name="reference_hard", k=5,
        probs=(0.60, 0.40, 0.40, 0.40, 0.40), horizon=100,
        is_reference=True, competence_eligible=True,
    ),
    # Not from the reference paper. Diagnostic floor only (§3.2 / Track C):
    # at K=2 "exploration" degenerates to "did it switch to the other arm",
    # which cannot support an information-seeking claim. Never a competence
    # anchor, never a B1 alpha environment.
    "native_floor": Environment(
        name="native_floor", k=2,
        probs=(0.70, 0.30), horizon=50,
        is_reference=False, competence_eligible=False,
    ),
}


def get_environment(key: str) -> Environment:
    if key not in ENVIRONMENTS:
        raise ValueError(
            f"unknown reference environment {key!r}; "
            f"expected one of {sorted(ENVIRONMENTS)} "
            f"(note: 'graded' is legacy-path only and is not a pv6 environment)")
    return ENVIRONMENTS[key]


# ──────────────────────── arm mapping (per seed) ────────────────────────────

def make_arm_map(seed: int, env: Environment) -> dict[str, float]:
    """name -> true probability, in DISPLAY ORDER.

    Two independent shuffles, mirroring the structure fixed in
    get_answer_bandit.shuffle_arms on 2026-07-28: (1) which identity is best,
    (2) where it is displayed. Coupling them is the position-leakage bug that
    invalidated every pre-2026-07-28 Bandit result — with a single shuffle the
    best arm lands at display position 1 for every seed, and OptFrac becomes
    indistinguishable from a first-option bias.

    This is a separate implementation from shuffle_arms rather than a call
    into it because shuffle_arms hard-codes CLOTHES_NAMES against the graded
    5-vector and has no K<5 or custom-probability mode. It is NOT byte-
    compatible with any pv1-pv5 cell, and must not be — pv6 is a different
    environment family with different labels and probabilities.
    """
    rng = random.Random(seed)
    names = ARM_LABELS[:env.k]
    assign = names[:]
    rng.shuffle(assign)
    name_to_prob = dict(zip(assign, env.probs))
    display = names[:]
    rng.shuffle(display)
    return {name: name_to_prob[name] for name in display}


def best_arm(arm_map: dict[str, float]) -> str:
    return max(arm_map, key=arm_map.get)


def position_of_best(seed: int, env: Environment) -> int:
    """1-based display position of the best arm (counterbalance check)."""
    am = make_arm_map(seed, env)
    return list(am).index(best_arm(am)) + 1


def identity_of_best(seed: int, env: Environment) -> str:
    return best_arm(make_arm_map(seed, env))


# ───────────────────────────── seed banks ───────────────────────────────────

def build_seed_bank(env: Environment, n: int = 20, search_limit: int = 100_000,
                    exclude: set[int] | None = None) -> list[int]:
    """Frozen counterbalanced seed bank: best-arm POSITION exactly balanced,
    best-arm IDENTITY exactly balanced (§3.5).

    n must be divisible by k so both can be exact — K=4 gives 5 seeds per
    position, K=5 gives 4. The (position, identity) CROSS table is balanced
    too, via a Latin-rectangle target pattern that is chosen first and then
    filled from the seed space (see the comment below for why a greedy scan
    cannot reach it). This rules out the residual confound a per-margin
    balance alone would leave: identity concentrating inside one position.

    Deterministic: same env + n + exclude always returns the same bank, so the
    bank can be regenerated rather than stored. Selected BEFORE any model
    behaviour is observed (§3.2) — never re-run this after seeing results.
    """
    if n % env.k:
        raise ValueError(
            f"n={n} not divisible by k={env.k}; exact position/identity "
            f"counterbalancing is impossible")
    per_margin = n // env.k
    exclude = exclude or set()
    names = ARM_LABELS[:env.k]

    # Balancing the two MARGINALS independently is not enough: a first-come
    # greedy fill satisfies "each position 4x" and "each identity 4x" while
    # still covering only 15 of 25 (position, identity) cells and repeating
    # some combination 3x, which is exactly the "identity concentrated inside
    # a position" confound §3.5 asks to rule out.
    #
    # So the cross table is capped too. There are k^2 cells and only n = k *
    # per_margin seeds, so full coverage is impossible for k > per_margin;
    # the achievable target is to spread seeds as evenly as arithmetic allows,
    # i.e. no cell used more than ceil(n / k^2) times. For n=20: K=4 gives
    # cap 2 (20 seeds over 16 cells), K=5 gives cap 1 (20 over 25, so 20
    # DISTINCT cells and no repeats at all).
    # A first-come greedy scan CANNOT reach the tight cap even when every cell
    # is reachable: it commits seeds early and strands the marginals, leaving
    # K=5 at 15/25 cells with repeats. So the target cell pattern is chosen
    # FIRST and seeds are then found to fill it.
    #
    # The pattern is a Latin rectangle: `per_margin` cyclic shifts of the
    # identity order, so row p (position) uses identities
    # {(p + r) mod k : r < per_margin}. Every position appears per_margin
    # times, every identity appears per_margin times, and no (position,
    # identity) cell repeats — the strongest cross-balance the arithmetic
    # allows (n = k*per_margin seeds over k^2 cells).
    #
    # For n=20: K=4 -> 20 seeds but only 16 cells, so per_margin(5) > k(4) and
    # the rectangle needs ceil(5/4)=2 passes; K=5 -> 20 seeds over 25 cells,
    # per_margin(4) < k(5), a single pass gives 20 DISTINCT cells.
    wanted: list[tuple[int, str]] = []
    for r in range(per_margin):
        for p in range(env.k):
            wanted.append((p + 1, names[(p + r) % env.k]))

    # Index the seed space once, then take the lowest unused seed per cell.
    by_cell: dict[tuple[int, str], list[int]] = {}
    for s in range(search_limit):
        if s in exclude:
            continue
        key = (position_of_best(s, env), identity_of_best(s, env))
        by_cell.setdefault(key, []).append(s)

    bank: list[int] = []
    used: set[int] = set()
    for cell in wanted:
        pool = by_cell.get(cell, [])
        pick = next((s for s in pool if s not in used), None)
        if pick is None:
            raise RuntimeError(
                f"no seed found for cell {cell} in {env.name} within "
                f"{search_limit} seeds")
        bank.append(pick)
        used.add(pick)

    if len(bank) != n:
        raise RuntimeError(
            f"could not build a balanced bank of {n} for {env.name} "
            f"(got {len(bank)})")
    return sorted(bank)


def bank_report(bank: list[int], env: Environment) -> dict:
    """Attestation of a bank's actual balance — stored with results so a run
    can prove what counterbalancing it ran under rather than asserting it."""
    pos: dict[int, int] = {}
    ident: dict[str, int] = {}
    cross: dict[str, int] = {}
    for s in bank:
        p = position_of_best(s, env)
        i = identity_of_best(s, env)
        pos[p] = pos.get(p, 0) + 1
        ident[i] = ident.get(i, 0) + 1
        cross[f"{p}|{i}"] = cross.get(f"{p}|{i}", 0) + 1
    return {
        "n": len(bank),
        "seeds": list(bank),
        "position_counts": dict(sorted(pos.items())),
        "identity_counts": dict(sorted(ident.items())),
        "n_cross_cells_used": len(cross),
        "n_cross_cells_total": env.k ** 2,
        "max_cell_repeat": max(cross.values()) if cross else 0,
        "position_balanced": len(set(pos.values())) == 1,
        "identity_balanced": len(set(ident.values())) == 1,
    }


def build_smoke_bank(env: Environment, n: int = 3,
                     formal_bank: list[int] | None = None) -> list[int]:
    """Smoke seeds, DISJOINT from the formal bank (§3.2).

    The formal 20 runs must stay unobserved until they are run for real, so
    smoke may not reuse them. Covers distinct best-arm positions where
    possible, but N=3 is never counterbalance evidence — this only makes the
    smoke test exercise more than one layout.
    """
    formal = set(formal_bank or build_seed_bank(env))
    picked: list[int] = []
    seen_pos: set[int] = set()
    for s in range(100_000):
        if len(picked) == n:
            break
        if s in formal:
            continue
        pos = position_of_best(s, env)
        if pos in seen_pos and len(seen_pos) < min(n, env.k):
            continue
        picked.append(s)
        seen_pos.add(pos)
    if len(picked) != n:
        raise RuntimeError(f"could not build a smoke bank of {n} for {env.name}")
    return picked


# ───────────────────────── per-arm reward tapes ─────────────────────────────

class RewardTape:
    """Pre-generated Bernoulli draws, indexed per arm (§3.5).

    The n-th pull of arm a returns tape[a][n] for EVERY policy sharing the
    seed. So alpha=-4, alpha=+4, reference-chat, reference-bare, Greedy, UCB1,
    TS and Oracle all face the same latent outcome on their n-th pull of a
    given arm, and a difference between them cannot be reward luck on that arm.

    This is strictly stronger than the legacy scheme (one RNG advanced once per
    round): there, two policies that pulled different arms at round t stayed
    "aligned" only in the number of draws consumed, not in what those draws
    were, so any per-arm outcome difference was still noise.

    Uniforms are drawn once and compared to the arm's probability on demand,
    so the tape does not depend on the probability vector — the same tape
    object stays valid if only the labels change.
    """

    def __init__(self, seed: int, env: Environment, length: int | None = None):
        self.seed = seed
        self.env = env
        self.length = length or env.horizon
        self.arm_map = make_arm_map(seed, env)
        # One independent stream per ARM (not per round), seeded from the run
        # seed plus the arm's display index. Per-arm streams are what make the
        # n-th pull reproducible regardless of when it happens.
        self._uniforms: dict[str, list[float]] = {}
        for i, name in enumerate(self.arm_map):
            rng = random.Random((seed + 1) * 1_000_003 + i)
            self._uniforms[name] = [rng.random() for _ in range(self.length)]
        self._pulls: dict[str, int] = {name: 0 for name in self.arm_map}

    def pull(self, arm: str) -> int:
        """Consume and return the next reward for `arm`."""
        n = self._pulls[arm]
        if n >= self.length:
            raise IndexError(
                f"tape exhausted for {arm!r} after {self.length} pulls "
                f"(seed={self.seed}, env={self.env.name})")
        self._pulls[arm] = n + 1
        return int(self._uniforms[arm][n] < self.arm_map[arm])

    def peek(self, arm: str, n: int) -> int:
        """The n-th (0-based) reward for `arm`, without consuming. For tests
        and for verifying two policies really did face the same tape."""
        return int(self._uniforms[arm][n] < self.arm_map[arm])

    @property
    def tape_id(self) -> str:
        """Stored with results so a run can be matched to its tape."""
        return f"{self.env.name}:s{self.seed}:L{self.length}"


# ────────────────────── F-reference prompt construction ─────────────────────

# Suggestive framing + reinforced CoT, the two prompt elements present in the
# single configuration that succeeded in Krishnamurthy et al. The reminder is
# repeated in the per-round block below ("reinforced" = it appears in the task
# description AND the per-round query, not only in the system message).
_TASK_HEADER = (
    "You are choosing among {k} buttons. Each button has a fixed but unknown "
    "probability of giving reward 1.\n"
    "A good strategy requires balancing exploration — trying buttons to "
    "estimate their rewards — with exploitation — choosing the button "
    "currently estimated to give the highest reward — to maximize total "
    "reward over all {horizon} rounds."
)

_RATIONALE_INSTRUCTION = (
    "Briefly reason about the amount of evidence, observed rewards, "
    "uncertainty, and remaining rounds in at most two sentences. "
    "Do not state a final choice yet."
)

# The action anchor. Frozen: the last prompt token after appending this is the
# single alpha injection site (§3.3), and candidate scoring continues from it.
ACTION_ANCHOR = "Choice: Button"


def render_state(arm_map: dict[str, float], history: list[tuple[str, int]],
                 round_idx: int, env: Environment) -> str:
    """The per-round state block: externally summarized sufficient statistics.

    Python computes successes / trials / empirical rate and nothing else — no
    Beta smoothing, no credible interval, no UCB bonus, no recommended arm
    (those are Track C scaffolds, §3.8, and would move this out of the
    "native capability" evidence class, §3.1).

    UNTRIED arms are listed by name in their own block and never rendered as
    0.00 — encoding unknown as a number makes an untried arm read as tied-worst.
    """
    n_arm = {name: 0 for name in arm_map}
    s_arm = {name: 0 for name in arm_map}
    for name, r in history:
        n_arm[name] += 1
        s_arm[name] += r

    tried, untried = [], []
    for name in arm_map:            # display order
        if n_arm[name] == 0:
            untried.append(f"- {name}")
        else:
            n, s = n_arm[name], s_arm[name]
            tried.append(
                f"- {name}: {s} reward{'' if s == 1 else 's'} / "
                f"{n} trial{'' if n == 1 else 's'}, empirical rate {s / n:.2f}")

    remaining = env.horizon - round_idx
    lines = [
        _TASK_HEADER.format(k=env.k, horizon=env.horizon),
        "",
        f"Round {round_idx + 1} of {env.horizon}; {remaining} rounds remain.",
        "",
    ]
    if tried:
        lines += ["TRIED OPTIONS", *tried, ""]
    if untried:
        lines += ["UNTRIED OPTIONS", *untried, ""]
    lines += [
        "Each button has a fixed but unknown probability of reward 1.",
        f"Balance exploration and exploitation to maximize total reward over "
        f"all {env.horizon} rounds.",
        "",
    ]
    return "\n".join(lines)


def build_rationale_prompt(arm_map, history, round_idx, env) -> str:
    """Stage 1: state + the reinforced CoT instruction. No anchor, no alpha."""
    return render_state(arm_map, history, round_idx, env) + _RATIONALE_INSTRUCTION


def build_action_prompt(arm_map, history, round_idx, env,
                        rationale_clean: str) -> str:
    """Stage 2: same state + sanitized rationale + the frozen action anchor.

    The returned string ENDS with ACTION_ANCHOR, so its last token is the alpha
    injection site and the next token is the arm letter. Callers must not
    append anything further.
    """
    state = render_state(arm_map, history, round_idx, env)
    block = state + _RATIONALE_INSTRUCTION
    if rationale_clean:
        block += "\n" + rationale_clean
    return block + "\n" + ACTION_ANCHOR


# ────────────────────────── rationale sanitization ──────────────────────────

# Matches a `Choice:` marker ANYWHERE in the line, not only at its start.
# Anchoring this to ^ was a bug: "I conclude Choice: Button A" survived and put
# a second anchor into the action prompt, which silently moves the alpha
# injection site off the real decision token.
_CHOICE_LINE = re.compile(r"choice\s*[:：]", re.I)


def sanitize_rationale(raw: str) -> str:
    """Frozen sanitization (§3.3).

    Drops WHOLE LINES containing a `Choice:` marker anywhere in the line, then
    strips surrounding whitespace. Nothing else — no character or sentence
    truncation (length is controlled by the stage-1 token cap instead, because
    character slicing can cut a decimal, an arm name, or a clause mid-word),
    and no semantic rewriting, reordering or summarizing.

    Removing premature `Choice:` lines is load-bearing rather than cosmetic: if
    one survived into the action prompt, that prompt would contain two anchors
    and the alpha injection site would no longer be the token preceding the
    real decision — which silently breaks the frozen steering semantics.
    """
    if not raw:
        return ""
    kept = [ln for ln in raw.splitlines() if not _CHOICE_LINE.search(ln)]
    return "\n".join(kept).strip()


# ───────────────────── candidate / tokenization utilities ───────────────────

def candidate_suffixes(env: Environment) -> list[str]:
    """Legal continuations of ACTION_ANCHOR, one per arm, in display-independent
    (label) order. `ACTION_ANCHOR + suffix` reconstructs the full arm name, so
    scoring these suffixes is scoring the arms.
    """
    return [name[len("Button"):] for name in ARM_LABELS[:env.k]]


def audit_candidate_tokenization(tokenizer, env: Environment) -> dict:
    """Tokenizer audit required by §3.3.

    Preferred case: every candidate suffix is a SINGLE token after the shared
    prefix, so a one-step argmax over K token ids is exact. If any candidate is
    multi-token, the caller must score full sequence log-probabilities instead
    of comparing first tokens — comparing only first tokens across candidates
    of different lengths is not a valid comparison of the candidates.

    Returns the audit dict to be stored with results; never raises. The caller
    decides which scoring path to take from `single_token`.
    """
    suffixes = candidate_suffixes(env)
    ids = {s: tokenizer.encode(s, add_special_tokens=False) for s in suffixes}
    single = all(len(v) == 1 for v in ids.values())
    return {
        "anchor": ACTION_ANCHOR,
        "suffixes": suffixes,
        "token_ids": ids,
        "n_tokens": {s: len(v) for s, v in ids.items()},
        "single_token": single,
        "scoring_mode": "argmax_single_token" if single else "sequence_logprob",
    }


# ───────────────────────────── reference metrics ────────────────────────────

def suffix_failure(choices: list[str], opt: str, t_start: int) -> bool:
    """SuffFail(t): the best arm is NEVER chosen in rounds [t, T].

    Krishnamurthy et al.'s primary failure statistic — it catches irrecoverable
    lock-in, which mean regret at N=20 cannot separate from bad luck because
    the underlying behaviour is bimodal.
    """
    return opt not in choices[t_start:]


def suff_fail_freq(runs: list[dict], t_start: int) -> float:
    """Fraction of runs with a suffix failure from `t_start` on."""
    if not runs:
        return float("nan")
    return sum(suffix_failure(r["choices"], r["best_arm"], t_start)
               for r in runs) / len(runs)


def k_min_frac(choices: list[str], arm_names: list[str], t: int | None = None) -> float:
    """K x MinFrac(t): least-played arm's share, rescaled to [0, 1].

    ~1 means near-uniform play (flailing, no exploitation); ~0 means at least
    one arm is being neglected, which is expected once a policy commits.
    Orthogonal to SuffFailFreq: together they bracket the two failure modes.

    Denominator is t (rounds elapsed), not the number of arms pulled, matching
    the reference definition.
    """
    t = t or len(choices)
    seg = choices[:t]
    counts = [sum(1 for c in seg if c == a) for a in arm_names]
    return len(arm_names) * (min(counts) / t)


def mean_k_min_frac(runs: list[dict], t: int | None = None) -> float:
    """Cross-run aggregate, ARITHMETIC MEAN (§3.5, frozen).

    The competence gate compares point estimates, and mean vs median can
    disagree on a right-skewed distribution, so the choice is part of the
    frozen spec rather than a reporting preference. Median/IQR are reported
    alongside but never gate.
    """
    if not runs:
        return float("nan")
    vals = [k_min_frac(r["choices"], list(r["arm_map"]), t) for r in runs]
    return sum(vals) / len(vals)


def greedy_frac(choices: list[str], feedbacks: list[int],
                arm_names: list[str]) -> float:
    """Fraction of rounds that chose an arm with the highest empirical mean at
    that moment, over rounds where at least one arm had been tried.

    Ties all count as greedy: several arms can legitimately share the top
    observed mean, especially early, and calling a tied pick non-greedy would
    overstate exploration.
    """
    seen_n: dict[str, int] = {}
    seen_s: dict[str, float] = {}
    hit = tot = 0
    for c, r in zip(choices, feedbacks):
        if seen_n:
            means = {a: seen_s[a] / seen_n[a] for a in seen_n}
            top = max(means.values())
            tot += 1
            # float tolerance: means are k/n ratios, exact ties are common and
            # must not be lost to representation error.
            if c in means and means[c] >= top - 1e-9:
                hit += 1
        seen_n[c] = seen_n.get(c, 0) + 1
        seen_s[c] = seen_s.get(c, 0.0) + r
    return hit / tot if tot else float("nan")


# ─────────────────────── algorithmic reference policies ─────────────────────

def run_greedy(seed: int, env: Environment, tape: RewardTape) -> dict:
    """Greedy baseline, frozen definition (§3.5).

    - First K rounds pull each arm once in DISPLAY ORDER. These rounds COUNT
      toward T and toward every metric — they are not warm-start rounds
      excluded from the denominator.
    - Thereafter pick the empirical-mean-best arm; on a tie choose uniformly
      AMONG THE TIED ARMS.
    - Tie-breaks draw from a dedicated `tie_rng` that never touches the reward
      tape.

    The tie-break rule is load-bearing because Greedy is the comparison base of
    competence gate rule 1, and the reference environments have identical
    suboptimal probabilities (Easy .25x3, Hard .40x4) so ties are frequent —
    especially right after initialization when every arm has n=1. A first-index
    tie-break would couple Greedy's choices to display position, contaminating
    the position counterbalancing the seed bank is built to guarantee.
    """
    arm_map = tape.arm_map
    arm_names = list(arm_map)
    tie_rng = random.Random(seed + TIE_RNG_OFFSET)
    choices: list[str] = []
    feedbacks: list[int] = []
    seen_n: dict[str, int] = {}
    seen_s: dict[str, float] = {}

    for t in range(env.horizon):
        if t < env.k:
            arm = arm_names[t]           # initialization, display order
        else:
            means = {a: seen_s[a] / seen_n[a] for a in seen_n}
            top = max(means.values())
            tied = [a for a, m in means.items() if m >= top - 1e-9]
            arm = tied[0] if len(tied) == 1 else tie_rng.choice(tied)
        r = tape.pull(arm)
        choices.append(arm)
        feedbacks.append(r)
        seen_n[arm] = seen_n.get(arm, 0) + 1
        seen_s[arm] = seen_s.get(arm, 0.0) + r

    return _package(seed, env, tape, choices, feedbacks,
                    policy="greedy",
                    extra={"tie_break": "uniform_among_tied",
                           "tie_rng": f"Random(seed+{TIE_RNG_OFFSET})",
                           "rng_version": 1,
                           "init_counts_toward_T": True})


def run_oracle(seed: int, env: Environment, tape: RewardTape) -> dict:
    """Oracle, frozen definition (§3.5): always pull the true best arm, same
    tape. A reward/regret UPPER BOUND only — it faces no exploration problem,
    so it is not a comparable policy.
    """
    opt = best_arm(tape.arm_map)
    choices, feedbacks = [], []
    for _ in range(env.horizon):
        choices.append(opt)
        feedbacks.append(tape.pull(opt))
    return _package(seed, env, tape, choices, feedbacks, policy="oracle")


def run_random(seed: int, env: Environment, tape: RewardTape) -> dict:
    """Uniform random. Provides the K x MinFrac ceiling that gate rule 2
    compares against."""
    rng = random.Random(seed + 3_000_000)
    arm_names = list(tape.arm_map)
    choices, feedbacks = [], []
    for _ in range(env.horizon):
        arm = rng.choice(arm_names)
        choices.append(arm)
        feedbacks.append(tape.pull(arm))
    return _package(seed, env, tape, choices, feedbacks, policy="random")


def _package(seed, env, tape, choices, feedbacks, policy, extra=None) -> dict:
    """Common result record. Same shape as the LLM runs produce, so the metric
    functions above apply unchanged to model and baseline alike."""
    arm_map = tape.arm_map
    opt = best_arm(arm_map)
    arm_names = list(arm_map)
    T = len(choices)
    rec = {
        "policy":        policy,
        "seed":          seed,
        "environment":   env.name,
        "protocol":      PROTOCOL_VERSION,
        "arm_map":       dict(arm_map),
        "arm_order":     arm_names,
        "best_arm":      opt,
        "best_position": arm_names.index(opt) + 1,
        "tape_id":       tape.tape_id,
        "choices":       choices,
        "feedbacks":     feedbacks,
        "opt_frac":      sum(1 for c in choices if c == opt) / T,
        "late_opt_frac": (sum(1 for c in choices[T // 2:] if c == opt)
                          / (T - T // 2)),
        "cum_regret":    float(sum(arm_map[opt] - arm_map[c] for c in choices)),
        "suffix_failure": suffix_failure(choices, opt, T // 2),
        "k_min_frac":    k_min_frac(choices, arm_names),
        "greedy_frac":   greedy_frac(choices, feedbacks, arm_names),
    }
    if extra:
        rec.update(extra)
    return rec


# ──────────────────── bootstrap + frozen baseline manifest ──────────────────

BOOTSTRAP_SEED = 20260803
BOOTSTRAP_N = 10_000
BOOTSTRAP_METHOD = "percentile"   # BCa is NOT used; see _bootstrap_ci


def _bootstrap_ci(values: list[float], stat, n_boot: int = BOOTSTRAP_N,
                  seed: int = BOOTSTRAP_SEED,
                  alpha: float = 0.05) -> dict:
    """Percentile bootstrap over RUNS (the resampling unit is the seed).

    Percentile rather than BCa: SuffFailFreq is a mean of a BINARY per-run
    indicator, so at N=20 the statistic is discrete with 21 attainable values
    and the acceleration/bias terms BCa estimates are themselves unstable.
    A percentile interval is the honest, weaker statement.

    NOTE this is the UNPAIRED interval for a single policy. The gate's real
    test is a PAIRED bootstrap on the per-seed DIFFERENCE (model - Greedy),
    which needs the model runs and therefore cannot be computed here — see
    paired_bootstrap_ci, which is what §3.7 must use.
    """
    if not values:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(n_boot):
        draws.append(stat([values[rng.randrange(n)] for _ in range(n)]))
    draws.sort()
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {
        "point":  stat(values),
        "lo":     lo,
        "hi":     hi,
        "n":      n,
        "n_boot": n_boot,
        "method": BOOTSTRAP_METHOD,
        "seed":   seed,
    }


def paired_bootstrap_ci(runs_a: list[dict], runs_b: list[dict], metric,
                        n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                        alpha: float = 0.05) -> dict:
    """Paired bootstrap on the per-seed difference metric(a) - metric(b).

    THIS is the gate statistic (§3.7). Resamples SEEDS, not runs independently,
    so the shared reward tape's variance cancels the way the paired design
    intends. Requires the two run lists to cover the same seeds; raises if not,
    because silently intersecting them would change the comparison basis.

    Duplicate seeds within EITHER list are also an error. Keying by seed would
    silently keep only the last record, so a resume that re-appended a cell, or
    a schema change that merged two conditions into one file, would quietly
    shrink n and tighten the CI instead of failing. Checked before the dicts
    are built, since building them is what destroys the evidence.
    """
    for label, runs in (("a", runs_a), ("b", runs_b)):
        seen = [r["seed"] for r in runs]
        if len(set(seen)) != len(seen):
            dupes = sorted({s for s in seen if seen.count(s) > 1})
            raise ValueError(
                f"paired bootstrap got duplicate seeds in list {label}: "
                f"{dupes} ({len(seen)} runs, {len(set(seen))} distinct) — "
                f"check for a resumed/merged result file")

    by_a = {r["seed"]: r for r in runs_a}
    by_b = {r["seed"]: r for r in runs_b}
    if set(by_a) != set(by_b):
        raise ValueError(
            f"paired bootstrap needs identical seed sets; "
            f"a-only={sorted(set(by_a) - set(by_b))} "
            f"b-only={sorted(set(by_b) - set(by_a))}")
    seeds = sorted(by_a)
    diffs = [metric(by_a[s]) - metric(by_b[s]) for s in seeds]
    out = _bootstrap_ci(diffs, lambda v: sum(v) / len(v), n_boot, seed, alpha)
    out["seeds"] = seeds
    out["paired"] = True
    return out


def _summarise_policy(runs: list[dict], env: Environment) -> dict:
    """Point estimates + unpaired bootstrap CIs for one policy."""
    T = env.horizon
    half = T // 2

    def _b(vals, stat=lambda v: sum(v) / len(v)):
        return _bootstrap_ci(vals, stat)

    return {
        "n_runs": len(runs),
        "opt_frac":      _b([r["opt_frac"] for r in runs]),
        "late_opt_frac": _b([r["late_opt_frac"] for r in runs]),
        "cum_regret":    _b([r["cum_regret"] for r in runs]),
        "greedy_frac":   _b([r["greedy_frac"] for r in runs]),
        # SuffFailFreq is the mean of a binary indicator -> bootstrap the
        # indicator directly so the CI is on the FREQUENCY, not on a per-run
        # continuous quantity.
        "suff_fail_freq_half": _b(
            [float(suffix_failure(r["choices"], r["best_arm"], half))
             for r in runs]),
        "k_min_frac_full": _b(
            [k_min_frac(r["choices"], list(r["arm_map"])) for r in runs]),
        "k_min_frac_half": _b(
            [k_min_frac(r["choices"], list(r["arm_map"]), half) for r in runs]),
    }


def build_baseline_manifest(env_keys: tuple[str, ...] = ("easy", "hard"),
                            n: int = 20) -> dict:
    """FROZEN algorithmic baseline manifest (§3.5 / §3.7).

    Computed and stored BEFORE any model is run, so the competence gate's
    comparison basis cannot drift with later model behaviour. Contains the
    protocol version, the full seed bank, its balance attestation, the
    bootstrap parameters, and Random/Greedy/Oracle summaries with CIs.

    Deterministic: no model, no GPU, no I/O. Re-running reproduces it byte for
    byte, which is what makes it checkable rather than merely archived.
    """
    manifest = {
        "protocol":         PROTOCOL_VERSION,
        "kind":             "frozen_algorithmic_baseline",
        "arm_labels":       ARM_LABELS,
        "action_anchor":    ACTION_ANCHOR,
        "tie_rng_offset":   TIE_RNG_OFFSET,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "n_boot": BOOTSTRAP_N,
            "seed":   BOOTSTRAP_SEED,
            "alpha":  0.05,
            "resampling_unit": "seed(run)",
            "note": ("unpaired per-policy CIs; the GATE statistic is the "
                     "paired per-seed difference computed after model runs "
                     "via paired_bootstrap_ci"),
        },
        "environments": {},
    }
    for key in env_keys:
        env = get_environment(key)
        bank = build_seed_bank(env, n=n)
        policies: dict[str, dict] = {}
        for pol_name, fn in (("random", run_random),
                             ("greedy", run_greedy),
                             ("oracle", run_oracle)):
            runs = [fn(s, env, RewardTape(s, env)) for s in bank]
            policies[pol_name] = _summarise_policy(runs, env)
        manifest["environments"][key] = {
            "name":        env.name,
            "k":           env.k,
            "probs":       list(env.probs),
            "horizon":     env.horizon,
            "is_reference": env.is_reference,
            "competence_eligible": env.competence_eligible,
            "seed_bank":   bank,
            "bank_report": bank_report(bank, env),
            "smoke_bank":  build_smoke_bank(env, formal_bank=bank),
            "policies":    policies,
        }
    return manifest
