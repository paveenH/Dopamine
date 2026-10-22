#!/usr/bin/env python3.10
"""PV11 prompt surface for Controlled Evidence-State Micro-Episodes. PURE.

No model, no I/O, no RNG. Everything here is a string transform, so it can be
audited by printing.

WHAT PV11 SHOWS, EXHAUSTIVELY
-----------------------------
    * an OPTIONS table: per-arm successes / trials / empirical rate
    * one line: "You may take at most H more samples."
    * the task header, the action clause and the format block
    * the `Reason: ` anchor

WHAT IT DELIBERATELY WITHHOLDS, AND WHY EACH ONE MATTERS
-------------------------------------------------------
  * CHOICE HISTORY. Frozen prohibition, not a preference. PV10 stored history
    as arm LABELS, which binds "this arm has many trials" to "this label
    appears many times in context" -- sample size and token frequency become
    the same manipulation, and PV11's entire Acquisition block is a sample-size
    contrast. It also structurally prevents reconstructing a count-based
    top-2 share from a label history, the degeneration that got
    `empirical_top2_share` banned in count-based form. A history-present
    presentation control may be run only AFTER the full +-4 main experiment.
  * TOTAL BUDGET / SAMPLES USED. PV10 printed "Samples used: n / remaining:
    T-n". PV11 cannot: the Acquisition cells differ in initial evidence
    (65 trials under low_n, 80 under matched_n), so any displayed total would
    make the sample-size manipulation ALSO a budget manipulation. Only the
    forward horizon H is shown, and H is equal across the four cells.
  * LATENT PROBABILITIES, PROBE IDENTITY, BLOCK NAME, CELL NAME. The design
    is invisible to the model. Naming the probe would replace the measurement
    with a transcription of the prompt -- the mechanism behind PV10-C's
    deprecated `.774`.

WHAT IS SHARED WITH PV10 RATHER THAN REIMPLEMENTED
--------------------------------------------------
`parse_policy`, `apply_stop_boundary`, `STOP_STRINGS`, `REASON_ANCHOR` and
`WORD_LIMIT` are imported from `bandit_pv10`. They are the SAME OBJECTS: the
v2 stop-boundary/parser contract was fixed at real cost (see
pv10_capability_amendment_02.json -- 58/60 episodes lost to a stop-parity bug
and a control-token boundary bug) and a copy here could drift from it silently.
The prompt BODY is PV11's own, because every line of it differs.

ANCHOR
------
The prompt ends at `Reason: `, one ASCII space, token 220 on the Llama-3.1
tokenizer -- the same decision-bottleneck site the RSN mask was extracted at.
`_assert_single_trailing_space` is imported from PV10 and runs on every build.
LLAMA3-ONLY, for the same reason PV10 is.

FIRE ARITHMETIC IS NOT PV10'S
-----------------------------
PV10 computes `(tau - k*FORCED_INIT_PER_ARM) + 1` because it forces one pull
per arm before generation starts. PV11 has NO forced initialization: the
synthetic counts ARE the history, and the episode opens with a real decision.
An episode that samples s times (0 <= s <= H) then commits makes s + 1 model
calls, so the per-episode ceiling is `(H + 1) * n_steered_layers`. Inheriting
PV10's formula would under-count by `k` and, because attestation is
fail-closed, raise on the first seed. See `expected_fires_max`.
"""

from __future__ import annotations

import bandit_pv10 as p10

# ────────────────────────── frozen version constants ────────────────────────
PROTOCOL_VERSION = "pv11"
STAGE1_INSTRUCTION_VERSION = "p11"
STATE_BANK_VERSION = "pv11-states-v1"
# The parser and stop boundary are PV10's v2 contract, unmodified.
POLICY_PARSER_VERSION = p10.POLICY_PARSER_VERSION

REASON_ANCHOR = p10.REASON_ANCHOR
WORD_LIMIT = p10.WORD_LIMIT
STOP_STRINGS = p10.STOP_STRINGS
ARM_LABELS = ("A", "B", "C", "D")
K = 4

parse_policy = p10.parse_policy
apply_stop_boundary = p10.apply_stop_boundary
_assert_single_trailing_space = p10._assert_single_trailing_space

# ───────────────────────────── prompt surface ───────────────────────────────

# Header. Kept close to PV10's so that a PV11-vs-PV10 difference cannot be
# attributed to a reworded task definition, minus the budget sentence (PV11
# never states a total) .
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

# Action clause while samples remain. Verbatim PV10 `_SAMPLING_CLAUSE`: it
# supplies NO confidence threshold and NO "use as few samples as possible",
# because the whole point is to read the model's OWN commitment threshold, and
# either instruction would overwrite it with an experimenter-supplied one.
# PV11 measures how that threshold moves with evidence and with alpha, so
# importing the wording keeps the threshold definition constant across the two
# protocols.
_SAMPLING_CLAUSE = p10._SAMPLING_CLAUSE

# At H == 0 the continue-sampling option is withdrawn. Same anchor, same single
# stage -- there is no separate terminal stage.
_TERMINAL_CLAUSE = p10._TERMINAL_CLAUSE

_FORMAT_SAMPLING = p10._FORMAT_SAMPLING
_FORMAT_TERMINAL = p10._FORMAT_TERMINAL


def format_options(display_order, counts) -> str:
    """OPTIONS table in display order. `counts[arm] = (successes, trials)`.

    Identical rendering to PV10's so the evidence display is not a confound
    between the protocols. In PV11 these counts are SYNTHETIC at the opening
    state and are then updated by real sampled rewards.
    """
    lines = ["OPTIONS"]
    for arm in display_order:
        s, t = counts[arm]
        rate = f"{s / t:.2f}" if t else "n/a"
        lines.append(
            f"- Button {arm}: {s} rewards / {t} trials, empirical rate {rate}")
    return "\n".join(lines)


def format_horizon(remaining: int) -> str:
    """Forward horizon only. No total, no used count -- see module docstring.

    Singular/plural is handled because "1 more samples" is a visible defect in
    a prompt that is otherwise held byte-stable across cells.
    """
    if remaining < 0:
        raise ValueError(f"remaining horizon must be >= 0, got {remaining}")
    if remaining == 0:
        return "You may take no further samples."
    if remaining == 1:
        return "You may take at most 1 more sample."
    return f"You may take at most {remaining} more samples."


def build_decision_prompt(display_order, counts, remaining: int,
                          word_limit: int = None) -> str:
    """The single-stage PV11 prompt. Ends at `Reason: ` (token 220).

    `remaining == 0` withdraws SAMPLE and permits only COMMIT. There is no
    forced-initialization phase and no CHOICE HISTORY block.
    """
    word_limit = WORD_LIMIT if word_limit is None else word_limit
    if set(counts) != set(display_order):
        raise ValueError("counts keys and display_order disagree")
    if len(set(display_order)) != len(display_order):
        raise ValueError(f"display_order has duplicates: {display_order}")

    terminal = (remaining == 0)
    clause = _TERMINAL_CLAUSE if terminal else _SAMPLING_CLAUSE
    fmt = (_FORMAT_TERMINAL if terminal else _FORMAT_SAMPLING).format(
        word_limit=word_limit)

    prompt = (
        f"{_TASK_HEADER}\n"
        f"\n"
        f"{clause}\n"
        f"\n"
        f"{format_horizon(remaining)}\n"
        f"\n"
        f"{format_options(display_order, counts)}\n"
        f"\n"
        f"{fmt}\n"
        f"\n"
        f"{REASON_ANCHOR}"
    )
    _assert_single_trailing_space(prompt, REASON_ANCHOR)
    return prompt


# ───────────────────────────── fire arithmetic ──────────────────────────────

def expected_fires_max(horizon: int, n_steered_layers: int = 9) -> int:
    """Per-episode UPPER BOUND on steering sites: (H + 1) * L.

    An episode makes one model call per decision. It may SAMPLE at most H
    times, and it always makes exactly one terminating call (the COMMIT, or
    the H == 0 terminal decision), hence H + 1 calls at most. Each call injects
    once, at the last prefill token, on each steered layer -- prefill_only,
    tail_len 1, decode never steered.

    This is a CEILING, not an equality: an episode that commits early makes
    fewer calls. Attestation must therefore compare against the episode's OWN
    realized call count, and use this only to bound it. PV10's
    `(tau - k*FORCED_INIT) + 1` does NOT apply -- PV11 has no forced init.

    L is 9 for the standard `11-20` band: `utils.decoder_layer_range(11, 20)`
    is half-open and yields `range(10, 19)`.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    return (horizon + 1) * n_steered_layers


def expected_fires_for_calls(n_model_calls: int,
                             n_steered_layers: int = 9) -> int:
    """Exact expected sites for an episode that made `n_model_calls` calls."""
    if n_model_calls < 1:
        raise ValueError(f"an episode makes >= 1 model call, got "
                         f"{n_model_calls}")
    return n_model_calls * n_steered_layers


def interface_tag(seeds_or_states, horizon_set) -> str:
    """Resume-key segment. Distinct from every PV10/PV10-C tag by construction.

    Includes the state-bank version and the horizon set: a bank edit or a
    horizon change must not resume into stored episodes under the same name.
    """
    import hashlib
    body = ",".join(str(x) for x in sorted(seeds_or_states))
    digest = hashlib.sha256(body.encode()).hexdigest()[:12]
    horizons = "-".join(str(h) for h in sorted(set(horizon_set)))
    return (f"{PROTOCOL_VERSION}_{STAGE1_INSTRUCTION_VERSION}_"
            f"{STATE_BANK_VERSION}_{POLICY_PARSER_VERSION}_"
            f"k{K}_H{horizons}_s{digest}")
