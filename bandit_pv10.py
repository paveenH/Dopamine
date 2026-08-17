#!/usr/bin/env python3.10
"""PV10 Stage-1 surface: prompts, per-seed orders, parser. PURE -- no model.

PV10 reframes the bandit as self-paced Best-Arm Identification. The model
samples to reduce uncertainty and decides for itself when the evidence is
enough, then commits. Sampling rewards are observations, not points.

WHAT IS DIFFERENT FROM PV9 (each item is a deliberate protocol change):

  1. SINGLE STAGE. There is no Stage 2. In PV9 `action_follows_policy` reached
     .994-1.000, so Stage 2 had degenerated into an identity map over Stage 1's
     Policy line; keeping it would cost a forward pass per round to buy nothing.
     It also could not express PV10's action space, which is two-dimensional
     (SAMPLE/COMMIT x arm) rather than a single arm choice.

     CONSEQUENCE, and it must be stated in any writeup: PV9 could argue that
     alpha's sharpening "arrived through the text" because Stage 2 was verified
     unsteered (action fires 0) while its margin still showed a dose-response.
     PV10 has no such separation -- injection and decision live in one
     generation. PV10 measures a JOINT REASONING-DECISION EFFECT. Do not
     describe it as steering passing through text into an unsteered executor.
     If that decomposition is ever wanted, rescore the stored `Reason` text
     offline with an uninjected scorer as a diagnostic; it does not drive the
     main experiment.

  2. TWO INDEPENDENT FROZEN ORDERS. `initial_pull_order` (the temporal order of
     the forced initialization) and `display_order` (the OPTIONS row order) are
     drawn from separate RNG streams. PV9 inherited a fixed A->D display, which
     left label confounded with row. pv7 measured the label effect at ~6x the
     row-position effect (centered log-prob span 4.73 vs 0.76), so decoupling
     them is what makes label and position separable here rather than a
     cosmetic change. `display_order` is FIXED WITHIN AN EPISODE -- reshuffling
     per round would make the option table a moving target.

  3. STRICT PARSER, FAIL-CLOSED. A missing, conflicting or out-of-set Policy is
     invalid. There is no lenient fallback and no random arm substitution:
     a guessed action would enter the environment as if the model had chosen
     it, and every later round's evidence state would inherit the fabrication.
     Occasional invalids are a result; many invalids are a capability failure.

The tokenizer invariant (`Reason: ` ends on token 220 for Llama-3.1) is
asserted against the REAL tokenizer in test_bandit_pv10.py. It was verified
empirically, not inherited from PV9's `Evidence: ` result.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

# ────────────────────────── frozen version constants ────────────────────────

PROTOCOL_VERSION = "pv10"
STAGE1_INSTRUCTION_VERSION = "p10"
POLICY_PARSER_VERSION = "pv10-strict-v2"
ORDER_VERSION = "orders-v1"

ARM_LABELS = ("A", "B", "C", "D", "E")

# The injection anchor. Ends in exactly ONE ASCII space; for Llama-3.1 that
# tail is token 220, the decision-bottleneck token the RSN mask was extracted
# at. Verified directly: "Reason: " -> [26197, 25, 220]. Note "Reason:" without
# the space ends on 25, so an rstrip anywhere in the prompt path silently moves
# the injection site -- which is why the trailing space is asserted, not
# assumed.
REASON_ANCHOR = "Reason: "
EXPECTED_WHITESPACE_TOKEN_ID = 220

TOTAL_BUDGET = 100
FORCED_INIT_PER_ARM = 1
WORD_LIMIT = 50
RATIONALE_MAX_TOKENS = 128

# Inherited from PV9, where it was a MEASURED choice: on stored pv8 data
# 1999/2000 rounds contained a blank line and all 1999 fell BEFORE the Policy
# line, so stopping on "\n\n" would truncate nearly every Policy. Do not
# "improve" this without re-measuring that.
STOP_STRINGS = ("#",)


def apply_stop_boundary(raw: str, stop_strings=STOP_STRINGS) -> str:
    """Truncate a generation at the first stop marker, EXCLUDING the marker.

    Why this exists (v2, after the PV10-A interface failure):

    HF `stop_strings` halts generation only AFTER the marker has been emitted,
    so the marker is still present in the returned text. The runtime therefore
    treated "#" as a boundary while the parser did not -- `_POLICY_RE` anchors
    on `$` and tolerates only `[.!]`, so a Policy line ending in the marker was
    read as `malformed`. In PV10-A that voided 47 of 58 terminating rounds:
    33 `Policy: SAMPLE Button C #` and 14 `... # Note: ...`.

    This is NOT a leniency fix and it is NOT stripping model content. The
    marker is a CONTROL token owned by the harness; everything at or after it
    was never inside the decision the model was asked to make. What survives is
    parsed by the SAME strict regex, unrelaxed:

      * a trailing "#" no longer voids an otherwise canonical Policy line
      * "# Note: ..." is removed with the marker, not silently recovered as an
        action -- if the pre-marker text holds no canonical directive the round
        is still invalid
      * two or more directives BEFORE the marker still conflict and still void
      * "explained instead of committing" stays a visible failure signature

    Callers must store the untruncated generation as `raw_generation` for
    audit and parse only the value returned here.
    """
    text = raw or ""
    cut = len(text)
    for marker in stop_strings or ():
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


# ───────────────────────────── per-seed orders ──────────────────────────────

@dataclass(frozen=True)
class EpisodeOrders:
    """Two independent, frozen orders for one episode.

    display_order : OPTIONS row order, FIXED for the whole episode.
    initial_pull_order : the order the environment force-samples each arm once.

    Independent streams, so label identity, display row and initialization
    position vary separately across seeds instead of moving together.
    """
    seed: int
    display_order: tuple[str, ...]
    initial_pull_order: tuple[str, ...]

    def __post_init__(self):
        assert set(self.display_order) == set(self.initial_pull_order), (
            "the two orders must permute the SAME arm set")
        assert len(set(self.display_order)) == len(self.display_order), (
            "display_order has duplicates")


def make_orders(seed: int, k: int) -> EpisodeOrders:
    """Derive both orders from `seed`. Deterministic, so they need not be stored.

    Separate `random.Random` instances rather than two shuffles of one stream:
    with a shared stream the second permutation is a deterministic function of
    the first, which is the coupling that produced the pv1-pv5 position-leakage
    bug. The offsets are arbitrary but frozen -- changing one re-randomizes
    every episode.

    NOT FOR FORMAL RUNS -- diagnostics and tests only. A real cell must read the
    frozen bank-level orders from `assign_orders`, because independent per-seed
    shuffles leave visible imbalance at n=20 (Button A landed in display row 1
    eight times and row 3 twice on the pv6 bank), and pv7 measured the label
    effect at ~6x the row-position effect.
    """
    names = list(ARM_LABELS[:k])

    display = names[:]
    random.Random(seed * 7_919 + 11).shuffle(display)

    init = names[:]
    random.Random(seed * 15_485_863 + 37).shuffle(init)

    return EpisodeOrders(seed=seed,
                         display_order=tuple(display),
                         initial_pull_order=tuple(init))


def _cyclic_block(names: list[str], n: int, offset: int) -> list[list[str]]:
    """n permutations whose label x position table is as flat as n//k allows.

    Cyclic rotations of a shuffled base give an exact Latin square: over any k
    consecutive rotations each label occupies each position exactly once. With
    n = 20 and k = 4 that is five complete squares, hence exactly 5 per cell.

    Rotations alone would make position a deterministic function of the label,
    so each square is built from its OWN independently shuffled base. That
    keeps the marginal table exactly balanced while leaving which label sits
    where unpredictable from the seed index.
    """
    k = len(names)
    out: list[list[str]] = []
    sq = 0
    while len(out) < n:
        base = names[:]
        random.Random(offset + sq * 104_729).shuffle(base)
        for r in range(k):
            if len(out) == n:
                break
            out.append([base[(i + r) % k] for i in range(k)])
        sq += 1
    return out


def assign_orders(seeds: list[int], k: int) -> dict[int, EpisodeOrders]:
    """Counterbalanced orders for a whole cell, keyed by seed.

    Both tables (label x display row, label x initial-pull position) are exactly
    balanced when len(seeds) is a multiple of k. The two assignments use
    different RNG offsets, so they stay mutually independent -- the property
    `make_orders` was written for and the reason the orders are split at all.

    Deterministic in the SORTED seed list, so a cell's orders are reproducible
    from the bank alone and need not be stored.

    BINDING CONSEQUENCE: the orders belong to the CELL, not to the seed. Calling
    this with a SUBSET of the bank re-derives DIFFERENT orders for the same
    seeds, which would silently break pairing against an already-run cell. A
    formal run must always pass the whole frozen 20-seed bank. Adding seeds
    later requires an extension manifest that leaves the original 20 untouched;
    it is not done by re-calling this with a longer list.
    """
    names = list(ARM_LABELS[:k])
    ordered = sorted(seeds)
    disp = _cyclic_block(names, len(ordered), offset=11)
    init = _cyclic_block(names, len(ordered), offset=15_485_863)
    return {
        s: EpisodeOrders(seed=s,
                         display_order=tuple(disp[i]),
                         initial_pull_order=tuple(init[i]))
        for i, s in enumerate(ordered)
    }


def order_balance_report(seeds: list[int], k: int,
                         arm_map_fn=None, orders=None) -> dict:
    """Counterbalance audit over a seed bank.

    Reports three tables the protocol requires to be approximately balanced:
      * true-best LABEL frequency          (needs arm_map_fn)
      * label x display-row cross table
      * initial-pull POSITION of each label

    Descriptive by design. With n=20 and k=4 exact balance is not attainable in
    every cell, and inventing a pass/fail threshold here would be exactly the
    post-hoc freedom the frozen-basis discipline exists to prevent. Read the
    tables; a blatant lock (e.g. one label always displayed first) is visible
    without a threshold.
    """
    # bandit_reference keys arms as "Button A"; PV10 uses the bare letter, since
    # its prompt writes "Button {arm}". Normalize at this boundary so the report
    # accepts an arm_map from either convention instead of assuming one.
    def _letter(name: str) -> str:
        return name.rsplit(" ", 1)[-1]

    names = list(ARM_LABELS[:k])
    label_row = {a: [0] * k for a in names}
    init_pos = {a: [0] * k for a in names}
    best_label: dict[str, int] = {a: 0 for a in names}
    best_row = [0] * k

    orders = orders if orders is not None else assign_orders(seeds, k)
    for s in seeds:
        o = orders[s]
        for row, arm in enumerate(o.display_order):
            label_row[arm][row] += 1
        for pos, arm in enumerate(o.initial_pull_order):
            init_pos[arm][pos] += 1
        if arm_map_fn is not None:
            am = arm_map_fn(s)
            b = _letter(max(am, key=am.get))
            best_label[b] += 1
            best_row[o.display_order.index(b)] += 1

    out = {
        "n_seeds": len(seeds),
        "k": k,
        "label_by_display_row": label_row,
        "label_by_initial_pull_position": init_pos,
        "expected_per_cell": len(seeds) / k,
    }
    if arm_map_fn is not None:
        out["true_best_by_label"] = best_label
        out["true_best_by_display_row"] = best_row
    return out


# ───────────────────────────── prompt surface ───────────────────────────────

_TASK_HEADER = (
    "You are the decision-maker in a best-arm identification task.\n"
    "\n"
    "Each button has a fixed but unknown probability of producing a reward of "
    "1;\n"
    "otherwise it produces a reward of 0. These probabilities may differ "
    "across\n"
    "buttons.\n"
    "\n"
    "Your goal is to identify the button with the highest reward probability.\n"
    "Sampling rewards are observations, not points."
)

# The sampling-phase instruction. Deliberately gives NO confidence threshold
# and NO "use as few samples as possible": the stopping time is meant to read
# the model's own subjective commitment threshold, and either instruction would
# overwrite it with an experimenter-supplied one.
_SAMPLING_CLAUSE = (
    "When you are confident which button has the highest reward probability,\n"
    "commit to that button. Otherwise, continue sampling one button to gather\n"
    "more evidence."
)

# Terminal variant at n = TOTAL_BUDGET. Same anchor, same single stage -- only
# the continue-sampling option is withdrawn.
_TERMINAL_CLAUSE = (
    "No samples remain. You must now commit to exactly one button."
)

_FORMAT_SAMPLING = (
    "Complete exactly two lines and stop after the Policy line. Use no more "
    "than\n"
    "{word_limit} words total.\n"
    "\n"
    "First line: finish “Reason:” by briefly assessing the evidence "
    "and whether\n"
    "more sampling is needed.\n"
    "Second line: write exactly one of:\n"
    "“Policy: SAMPLE Button X”\n"
    "or\n"
    "“Policy: COMMIT Button X”\n"
    "\n"
    "Keep both lines concise. The Policy line must name exactly one button.\n"
    "Do not repeat the task or continue after the Policy line."
)

_FORMAT_TERMINAL = (
    "Complete exactly two lines and stop after the Policy line. Use no more "
    "than\n"
    "{word_limit} words total.\n"
    "\n"
    "First line: finish “Reason:” by briefly stating which button "
    "your evidence\n"
    "favours.\n"
    "Second line: write exactly:\n"
    "“Policy: COMMIT Button X”\n"
    "\n"
    "Keep both lines concise. The Policy line must name exactly one button.\n"
    "Do not repeat the task or continue after the Policy line."
)


def _assert_single_trailing_space(prompt: str, anchor: str) -> None:
    """The prompt must end at the anchor, on exactly one ASCII space.

    Two failure modes this catches, both silent and both observed in pv7:
      * rstrip anywhere upstream -> tail becomes ':' (25), not 220
      * a stray double space     -> one token 256, not [220] and not [220, 220]
    """
    if not prompt.endswith(anchor):
        raise AssertionError(
            f"prompt must end at the anchor {anchor!r}; got ...{prompt[-40:]!r}")
    if not anchor.endswith(" "):
        raise AssertionError(f"anchor {anchor!r} must end in one ASCII space")
    if prompt.endswith("  "):
        raise AssertionError(
            "prompt ends in a DOUBLE space: that is token 256, not 220 -- the "
            "injection site would silently move")


def format_options(display_order, counts: dict[str, tuple[int, int]]) -> str:
    """OPTIONS table in display order.

    `counts[arm] = (successes, trials)`, cumulative and INCLUDING the forced
    initialization -- the model must see the same history the offline
    algorithms score on.
    """
    lines = ["OPTIONS"]
    for arm in display_order:
        s, t = counts[arm]
        rate = f"{s / t:.2f}" if t else "n/a"
        lines.append(
            f"- Button {arm}: {s} rewards / {t} trials, empirical rate {rate}")
    return "\n".join(lines)


def format_history(history) -> str:
    """Choice history: arm labels only, oldest -> newest.

    Per-round rewards are deliberately NOT repeated here; OPTIONS already
    carries the counts and rates. The forced-initialization pulls appear in
    their actual `initial_pull_order`, so history and OPTIONS never disagree.
    """
    body = " ".join(history) if history else "(none)"
    return f"CHOICE HISTORY (oldest → newest):\n[{body}]"


def build_decision_prompt(display_order, counts: dict[str, tuple[int, int]],
                          history, n: int,
                          total_budget: int = TOTAL_BUDGET,
                          word_limit: int = WORD_LIMIT) -> str:
    """The single-stage PV10-B prompt. Ends at `Reason: ` (token 220).

    At n == total_budget this switches to the terminal variant: same anchor,
    same one-stage generation, only COMMIT permitted. There is no separate
    Stage 2 for the forced ending.
    """
    if n > total_budget:
        raise ValueError(f"n={n} exceeds total_budget={total_budget}")
    terminal = (n == total_budget)

    clause = _TERMINAL_CLAUSE if terminal else _SAMPLING_CLAUSE
    fmt = (_FORMAT_TERMINAL if terminal else _FORMAT_SAMPLING).format(
        word_limit=word_limit)

    budget_line = (
        f"You may take at most {total_budget} samples in total.\n"
        f"\n"
        f"Samples used: {n}\n"
        f"Samples remaining: {total_budget - n}"
    )

    prompt = (
        f"{_TASK_HEADER}\n"
        f"\n"
        f"{clause}\n"
        f"\n"
        f"{budget_line}\n"
        f"\n"
        f"{format_history(history)}\n"
        f"\n"
        f"{format_options(display_order, counts)}\n"
        f"\n"
        f"{fmt}\n"
        f"\n"
        f"{REASON_ANCHOR}"
    )
    _assert_single_trailing_space(prompt, REASON_ANCHOR)
    return prompt


# ───────────────────────────── strict parser ────────────────────────────────

# One canonical form: "Policy: <ACTION> Button <X>". Case-insensitive on the
# keywords, tolerant of surrounding whitespace and a full-width colon, and
# nothing else. Trailing punctuation is allowed so a period does not void an
# otherwise well-formed line, but any further WORDS make it non-canonical and
# it is not matched -- "explained instead of committing" must stay a visible
# failure signature rather than being silently recovered.
_POLICY_RE = re.compile(
    r"policy\s*[:：]\s*(SAMPLE|COMMIT)\s+Button\s+([A-E])\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE)

_POLICY_LINE_MARKER = re.compile(r"policy\s*[:：]", re.IGNORECASE)


@dataclass
class ParsedPolicy:
    """Outcome of parsing one generation. `valid` is the only gate."""
    valid: bool
    action: str | None = None          # "SAMPLE" | "COMMIT"
    arm: str | None = None
    reason: str = ""
    invalid_kind: str | None = None    # why it failed, for the validity table
    n_policy_lines: int = 0
    native_ends_after_policy: bool = False
    # Punctuation tolerance kept VISIBLE rather than silent: the action is
    # valid either way, but a cell where format exactness shifts with alpha
    # should be readable as such instead of hiding inside `valid`.
    format_exact: bool = True
    trailing_period_tolerated: bool = False
    raw: str = ""


def parse_policy(text: str, allowed_arms,
                 terminal: bool = False) -> ParsedPolicy:
    """Strict, fail-closed parse of one generation.

    Returns invalid rather than guessing. The invalid kinds separate distinct
    failure modes, which is the point of not having a fallback:

      no_policy         nothing that looks like a Policy line
      malformed         a "Policy:" marker exists but no canonical directive
      conflicting       two or more canonical directives that DISAGREE
      arm_out_of_set    named an arm outside this episode's arm set
      sample_at_terminal  chose SAMPLE when only COMMIT is permitted

    Two canonical directives that AGREE (a repeat) are not a conflict: the
    intended action is unambiguous. Two that disagree void the round, because
    picking either one would be the experimenter deciding.

    `native_ends_after_policy` records whether generation actually stopped at
    the Policy line. It is a validity READOUT, not a validity condition -- text
    after a well-formed Policy does not void the round, it is reported.
    """
    raw = text or ""
    matches = list(_POLICY_RE.finditer(raw))

    reason = raw.split("\n", 1)[0].strip()

    if not matches:
        kind = "malformed" if _POLICY_LINE_MARKER.search(raw) else "no_policy"
        return ParsedPolicy(valid=False, reason=reason, invalid_kind=kind,
                            n_policy_lines=len(_POLICY_LINE_MARKER.findall(raw)),
                            raw=raw)

    directives = {(m.group(1).upper(), m.group(2).upper()) for m in matches}
    if len(directives) > 1:
        return ParsedPolicy(valid=False, reason=reason,
                            invalid_kind="conflicting",
                            n_policy_lines=len(matches), raw=raw)

    action, arm = next(iter(directives))
    first = matches[0]
    ends_after = not raw[first.end():].strip()
    tolerated = bool(first.group(0).rstrip().endswith((".", "!")))

    if arm not in set(allowed_arms):
        return ParsedPolicy(valid=False, reason=reason,
                            invalid_kind="arm_out_of_set",
                            n_policy_lines=len(matches),
                            native_ends_after_policy=ends_after, raw=raw)

    if terminal and action == "SAMPLE":
        return ParsedPolicy(valid=False, reason=reason,
                            invalid_kind="sample_at_terminal",
                            n_policy_lines=len(matches),
                            native_ends_after_policy=ends_after, raw=raw)

    return ParsedPolicy(valid=True, action=action, arm=arm, reason=reason,
                        n_policy_lines=len(matches),
                        native_ends_after_policy=ends_after,
                        format_exact=not tolerated,
                        trailing_period_tolerated=tolerated, raw=raw)


# ───────────────────────── tokenizer invariant audit ────────────────────────

def audit_prompt_tokens(tokenizer, prompt: str) -> dict:
    """Assert the prompt ends on the injection token. Raises on violation.

    PV10 scores no candidates, so unlike pv7 there is no ID-level concatenation
    to audit -- but the tail token still decides WHERE alpha lands, so it is
    checked every round rather than trusted.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise AssertionError("empty tokenization")
    if ids[-1] != EXPECTED_WHITESPACE_TOKEN_ID:
        raise AssertionError(
            f"prompt tail token is {ids[-1]} "
            f"({tokenizer.decode([ids[-1]])!r}), expected "
            f"{EXPECTED_WHITESPACE_TOKEN_ID}; check rstrip, double spaces, "
            f"anchor wording and tokenizer identity")

    bos = getattr(tokenizer, "bos_token_id", None)
    double_bos = bool(bos is not None and len(ids) >= 2
                      and ids[0] == bos and ids[1] == bos)
    if double_bos:
        raise AssertionError("double BOS in prompt ids")

    return {
        "n_tokens": len(ids),
        "tail_token_id": ids[-1],
        "tail_token": tokenizer.decode([ids[-1]]),
        "double_bos": double_bos,
    }


# ──────────────────── termination / steering accounting ────────────────────

# An episode ends in exactly one of these. They are mutually exclusive, and the
# distinction is load-bearing for analysis:
#
#   autonomous_commit  the model committed on its own -> tau is a real stopping
#                      time and enters the accuracy-sample tradeoff
#   forced_commit      it sampled to T_max and the terminal prompt made it
#                      commit -> censored; NOT an autonomous stopping time
#   invalid_policy     a generation could not be parsed -> the episode ENDS.
#                      The environment must never invent a SAMPLE or a COMMIT
#                      on the model's behalf, and re-generating the same state
#                      under temperature=0 would just reproduce the same
#                      invalid. Retrying or skipping would also hand different
#                      alpha cells different numbers of generation
#                      opportunities. An invalid is an interface-validity
#                      OUTCOME, possibly itself an alpha effect, and is
#                      recorded as such -- never as budget censoring, and never
#                      given a fabricated tau.
TERMINATION_REASONS = ("autonomous_commit", "forced_commit", "invalid_policy")


def expected_fires(alpha: float, n_model_calls: int,
                   n_steered_layers: int = 9, tail_tokens: int = 1) -> int:
    """Observed injection SITES for one episode.

        expected_fires = model_calls x steered_layers x tail_tokens

    A site is one (steered layer, sequence, token position) that received a
    non-zero add -- deliberately not a hook-call count, which would report the
    same number for any layer count and prove nothing.

    n_steered_layers is 9 for the standard 11-20 band: `decoder_layer_range`
    is half-open, giving range(10, 19). alpha=0 registers no hook at all, so
    the count is exactly 0 -- "unsteered" is a different code path from "steered
    by zero", and that is what lets an alpha=0 cell be reused.

    PV10 differs structurally from PV9 here. PV9 made one call per round for
    100 rounds; PV10's four forced-initialization pulls consume no generation,
    so an episode that commits at tau makes

        (tau - 4) SAMPLE calls + 1 COMMIT call = tau - 3

    calls. A full-budget episode is therefore 97 calls = 873 fires, NOT PV9's
    900. An invalid episode is verified against its ACTUAL call count.
    """
    if alpha == 0:
        return 0
    return n_model_calls * n_steered_layers * tail_tokens


def model_calls_for(tau: int, k: int) -> int:
    """Generation calls in an episode that committed at sample count tau.

    The forced initialization consumes samples but no generation, hence the
    `-3` at K=4 rather than `-4`: (tau - k*FORCED_INIT_PER_ARM) SAMPLE calls,
    plus the one call that produced the COMMIT. k is required rather than
    defaulted, so a future K change cannot silently keep the K=4 arithmetic.
    """
    return (tau - k * FORCED_INIT_PER_ARM) + 1


def interface_tag(k: int, seeds) -> str:
    """Resume-key segment. Any change here must change the stored cell's key.

    Includes the seed CONTENT (order-insensitive), not just the count: two
    different 20-seed sets would otherwise resume into each other and return
    episodes from different environments under the same cell name.
    """
    # sha256, NOT the builtin hash(): PYTHONHASHSEED is randomized per process,
    # so hash() would give the same cell a different resume key on every run.
    seed_sig = "-".join(str(s) for s in sorted(seeds))
    digest = hashlib.sha256(seed_sig.encode()).hexdigest()[:12]
    return (f"{PROTOCOL_VERSION}_{STAGE1_INSTRUCTION_VERSION}_"
            f"{ORDER_VERSION}_{POLICY_PARSER_VERSION}_k{k}_"
            f"T{TOTAL_BUDGET}_s{digest}")
