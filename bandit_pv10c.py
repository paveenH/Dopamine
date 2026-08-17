#!/usr/bin/env python3.10
"""PV10-C Stage-1 surface: PV10-B + a competitor/falsification cue. PURE.

PV10-C is an INTERVENTION on the sampling instruction and nothing else. It
exists because PV10-A v2 answered the mechanism question: forced continuation
to the full budget did NOT produce exploration (min_trials stayed at 1,
max_arm_share ~.6, A0 true_top2 .408 against a rate-based empirical_top2 .734),
so PV10-B's bottleneck is the ACQUISITION policy, not a too-low commitment
threshold. See AdaBandit.md 4.4. The question here is therefore not "sample
more" but "does naming the comparison change HOW the model samples".

WHAT CHANGES, EXHAUSTIVELY:

  * `_SAMPLING_CLAUSE` gains the competitor cue below.

WHAT DOES NOT CHANGE (each is asserted in test_bandit_pv10c.py):

  * The TERMINAL prompt is byte-identical to PV10-B's. The cue is about how to
    sample; at n == TOTAL_BUDGET there is nothing left to sample, and letting
    the cue leak there would alter the commit decision itself -- the one
    measurement PV10-A and PV10-B share with this protocol.
  * The `Reason: ` anchor still ends the prompt on token 220. The cue adds
    prompt tokens UPSTREAM of the anchor; pv7's `.rstrip()` trap is the
    precedent for how quietly an injection site can slide to `:` (25).
  * The parser, the arm labels, the orders, the environment, the seed bank and
    the reward tapes are the SAME OBJECTS, imported from bandit_pv10.
  * `expected_fires` arithmetic is unchanged: the cue adds prompt tokens, not
    model calls, so a full-budget episode is still model_calls x 9 x 1.

THE CUE NAMES THE COMPARISON, NOT AN ARM. Naming a specific button would make
`competitor_named` a transcription of the prompt rather than a measurement.

PRE-REGISTERED READING (frozen before the run, so a null cannot be re-read as
a failed run): PV7's calculator arm lifted `mentions_posterior` to .780 while
uncertainty targeting stayed at the floor, and PV9's `cue-v1` produced
cue-scaffolded untried coverage (77/79/79) rather than genuine
uncertainty-targeting (5/2/1). So the two text metrics are reported as an
ORDERED PAIR -- recognition first, then alignment CONDITIONAL on recognition.
High recognition with floor alignment is a RESULT (recognition is not the
binding constraint), not a failed intervention.
"""

from __future__ import annotations

import bandit_pv10 as p10

# ────────────────────────── frozen version constants ────────────────────────
# PROTOCOL_VERSION stays "pv10": the environment, seed bank, tapes, orders,
# parser and scoring are the same objects. What changed is the Stage-1
# instruction, which is exactly what STAGE1_INSTRUCTION_VERSION names -- and it
# propagates into the resume key, so C cannot resume into a B cell.
PROTOCOL_VERSION = p10.PROTOCOL_VERSION
STAGE1_INSTRUCTION_VERSION = "p10c"
POLICY_PARSER_VERSION = p10.POLICY_PARSER_VERSION
ORDER_VERSION = p10.ORDER_VERSION
COMPETITOR_CUE_VERSION = "cue-competitor-v1"

# The frozen cue. Accepted verbatim; do not reword without a version bump.
_COMPETITOR_CUE = (
    "Before deciding, identify the strongest alternative to the button that\n"
    "currently looks best. Consider which next sample would most clearly\n"
    "distinguish between them. If the evidence is weak, tied, or uneven, "
    "continue\n"
    "sampling. Commit only when the evidence favors one button over its "
    "strongest\n"
    "alternative."
)

_SAMPLING_CLAUSE = f"{p10._SAMPLING_CLAUSE}\n\n{_COMPETITOR_CUE}"

# Re-exported so callers never reach past this module for the pieces that are
# deliberately shared.
REASON_ANCHOR = p10.REASON_ANCHOR
TOTAL_BUDGET = p10.TOTAL_BUDGET
WORD_LIMIT = p10.WORD_LIMIT
STOP_STRINGS = p10.STOP_STRINGS
ARM_LABELS = p10.ARM_LABELS
parse_policy = p10.parse_policy
apply_stop_boundary = p10.apply_stop_boundary
format_history = p10.format_history
format_options = p10.format_options


def build_decision_prompt(display_order, counts, history, n: int,
                          total_budget: int = None,
                          word_limit: int = None) -> str:
    """PV10-B's prompt with the competitor cue in the SAMPLING clause only.

    At n == total_budget this delegates unchanged to bandit_pv10, so the
    terminal prompt is byte-identical to PV10-B's by construction rather than
    by a copied string that could drift.
    """
    total_budget = p10.TOTAL_BUDGET if total_budget is None else total_budget
    word_limit = p10.WORD_LIMIT if word_limit is None else word_limit

    if n == total_budget:
        return p10.build_decision_prompt(
            display_order, counts, history, n,
            total_budget=total_budget, word_limit=word_limit)

    original = p10._SAMPLING_CLAUSE
    p10._SAMPLING_CLAUSE = _SAMPLING_CLAUSE
    try:
        return p10.build_decision_prompt(
            display_order, counts, history, n,
            total_budget=total_budget, word_limit=word_limit)
    finally:
        p10._SAMPLING_CLAUSE = original
