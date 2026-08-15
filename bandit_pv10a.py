"""PV10-A: FIXED-BUDGET control for PV10-B. Mechanism control, not a rival.

PV10-B asks two things at once: how the model ACQUIRES evidence, and when it
decides it has enough. Its alpha=0 result confounds them -- the median episode
committed after 13 samples with some arm still at a single pull, so poor
identification could be premature stopping OR a broken acquisition policy.

PV10-A removes the stopping decision:

  * during the budget the model may only SAMPLE (no COMMIT is offered)
  * at n == total_budget the FROZEN PV10-B terminal prompt asks for the commit

Everything else is held identical to PV10-B on purpose -- environment, seed
bank, reward tapes, order bank, history format, anchor, parser, steering
semantics -- so a difference between A and B is attributable to the stopping
decision alone.

Reading:
  * still ~.6-.9 single-arm share  -> the defect is the ACQUISITION policy,
    and early stopping was a symptom rather than the cause
  * allocation and accuracy improve -> PV10-B's defect is a too-low subjective
    commitment threshold

Only then does comparing +-4 say whether RSN moves sampling, the commitment
threshold, or only the text.

PV10-B is untouched: this module imports its prompt parts and overrides one
clause. The parser, anchor and format block are the same objects, so the two
protocols cannot drift apart.
"""
from __future__ import annotations

import bandit_pv10 as p10

PROTOCOL_VERSION = "pv10a"
STAGE1_INSTRUCTION_VERSION = "p10a"

# The ONE difference from PV10-B. COMMIT is not offered while budget remains,
# so this must not hint that stopping early is possible.
_SAMPLING_ONLY_CLAUSE = (
    "Sample buttons to learn which one has the highest reward probability.\n"
    "You will choose a button at the end, after your sampling budget is used."
)

_FORMAT_SAMPLING_ONLY = (
    "Complete exactly two lines and stop after the Policy line. Use no more "
    "than\n{word_limit} words total.\n"
    "\n"
    "First line: finish “Reason:” by briefly assessing the evidence "
    "and which button\nyou want to learn more about.\n"
    "Second line: write exactly:\n"
    "“Policy: SAMPLE Button X”\n"
    "\n"
    "Keep both lines concise. The Policy line must name exactly one button.\n"
    "Do not repeat the task or continue after the Policy line."
)


def build_decision_prompt(display_order, counts, history, n: int,
                          total_budget: int = p10.TOTAL_BUDGET,
                          word_limit: int = p10.WORD_LIMIT) -> str:
    """PV10-A prompt. Identical to PV10-B except COMMIT is withheld until the
    budget is exhausted.

    At n == total_budget this delegates to the FROZEN PV10-B builder, so the
    terminal commit prompt is byte-identical between the two protocols -- the
    commit is elicited the same way and only the history differs.
    """
    if n > total_budget:
        raise ValueError(f"n={n} exceeds total_budget={total_budget}")
    if n == total_budget:
        return p10.build_decision_prompt(
            display_order, counts, history, n=n,
            total_budget=total_budget, word_limit=word_limit)

    prompt = (
        f"{p10._TASK_HEADER}\n"
        f"\n"
        f"{_SAMPLING_ONLY_CLAUSE}\n"
        f"\n"
        f"You may take at most {total_budget} samples in total.\n"
        f"\n"
        f"Samples used: {n}\n"
        f"Samples remaining: {total_budget - n}\n"
        f"\n"
        f"{p10.format_history(history)}\n"
        f"\n"
        f"{p10.format_options(display_order, counts)}\n"
        f"\n"
        f"{_FORMAT_SAMPLING_ONLY.format(word_limit=word_limit)}\n"
        f"\n"
        f"{p10.REASON_ANCHOR}"
    )
    p10._assert_single_trailing_space(prompt, p10.REASON_ANCHOR)
    return prompt


def interface_tag(k: int, seeds) -> str:
    """Distinct from PV10-B's tag so the two can never resume into each other."""
    base = p10.interface_tag(k, seeds)
    return base.replace(p10.PROTOCOL_VERSION, PROTOCOL_VERSION, 1).replace(
        p10.STAGE1_INSTRUCTION_VERSION, STAGE1_INSTRUCTION_VERSION, 1)
