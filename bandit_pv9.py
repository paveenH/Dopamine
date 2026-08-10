#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""PV9 Stage-1 prompt surface. Pure: no model, no GPU, no I/O.

PV9 = pv8 (pv7 + CHOICE HISTORY) plus FOUR Stage-1 modifications. Stage 2 is
byte-unchanged from pv7's frozen S1 interface.

WHAT PV9 ADDS OVER pv8 (and what it does NOT)
---------------------------------------------
Inherited, NOT new here -- the round counter and `Future choices after this
one` come from pv7; the CHOICE HISTORY block comes from pv8. Only these four
are PV9:

1. SELF-RELEVANT REWARD FRAMING. Rewards are framed as the model's own task
   score, and a running `Your score so far` line is shown. This targets
   motivational salience (a wanting manipulation needs an object) and adds no
   decision information: the score is a deterministic function of the reward
   history already in the OPTIONS table.

2. UNTRIED-ARM EXPLORATION CUE, on n=0 rows only. This is a SCAFFOLD, and the
   distinction matters for what may be claimed. pv8 measured a ~3% EXPLORE
   floor; H2 (information investment) is untestable at a floor, so the cue
   exists to lift the readout off it. Any exploration observed under PV9 is
   therefore SCAFFOLDED discovery, never evidence of autonomous exploration.
   The cue vanishes once an arm has one pull, so it CANNOT drive one-shot-zero
   revisits -- that failure mode stays untouched and remains measurable.

3. GENERATION CONTROL. 128 rationale tokens with a 50-word instruction.
   Raising the budget prevents the Policy line itself from being truncated
   (pv8's alpha=-4 median ran 40 words against a 64-token cap). It does NOT
   fix non-termination -- pv7 established that 128->192 produced byte-identical
   output, so continuation is a generation behaviour, not a budget shortfall.
   The stop strings below are what actually cut the trailing spray, and
   `extract_evidence_policy_block` remains the second line of defence.

4. EXPLICIT BERNOULLI STRUCTURE. States that each button has a fixed unknown
   probability of reward 1 and that probabilities differ across buttons. This
   is environment structure, not strategy: no probabilities, no algorithm, no
   guidance on when to explore.

THE SCAFFOLD BOUNDARY
---------------------
Items 1 and 4 are environment information. Item 2 states a benefit direction
("may improve future rewards") and is therefore on the strategy side of the
line -- closer to EVOLvE's algorithm-guided support than to a bare state
description. It is adopted knowingly, because a 3% floor cannot be modulated.
The requirement is only that PV9 results are never reported as native
exploration.

STAGE-1-ONLY INFORMATION ISOLATION
----------------------------------
The cue, score line and history all live in Stage 1, where the policy forms.
Stage 2 keeps the frozen S1 prompt -- counts only. Putting the cue in Stage 2
would raise that button's candidate logit directly, and a change in the chosen
arm could no longer be attributed to Stage 1's reasoning rather than to
priming. The OPTIONS core (arm order, trials, rewards, empirical rate) stays
identical across stages; only the cue differs. Unlike pv6's option-display
drift, which was an unnoticed ORDER inconsistency, this is a declared CONTENT
difference with a stated purpose, and the episode runner asserts it.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import bandit_pv7 as p7
import bandit_pv8_episode as p8
import bandit_reference as br


PROTOCOL_VERSION = "pv9"
STAGE1_INSTRUCTION_VERSION = "p9"
STAGE2_INSTRUCTION_VERSION = p8.STAGE2_INSTRUCTION_VERSION   # "s1", frozen
HISTORY_BLOCK_VERSION = p8.HISTORY_BLOCK_VERSION
SCORE_BLOCK_VERSION = "score-v1"
UNTRIED_CUE_VERSION = "cue-v1"

RATIONALE_MAX_TOKENS = 128
RATIONALE_WORD_LIMIT = 50

# Stop strings for Stage 1. Neither `#` nor a blank line occurs anywhere in
# the PV9 instruction, so neither can fire on the prompt's own text -- the
# CGT failure mode (a stop marker that appears in the prompt truncates the
# reply before its real content) is structurally impossible here.
#
# `#` targets the measured failure directly: at alpha=0, 45% of pv8 rounds
# continued past the Policy line and 35.5% of all rounds emitted a hashtag
# spray. It is belt-and-braces with the Policy-line truncation, which still
# runs; the record stores which one fired, so a future run cannot mistake
# "post-processed away" for "the model stopped".
#
# `\n\n` WAS CONSIDERED AND REJECTED, on measurement not intuition: in the
# stored pv8 alpha=0 cell 1999/2000 rationales contain a blank line, and in
# ALL 1999 it falls BEFORE the Policy line (the model writes Evidence, a blank
# line, then Policy). A blank-line stop would therefore delete the Policy line
# in essentially every round and leave Stage 2 reading evidence with no policy.
# Do not add it back without re-checking that distribution.
STOP_STRINGS = ("#",)

_TASK_HEADER_PV9 = (
    "You are the decision-maker in this task. Each button has a fixed but "
    "unknown probability of producing a reward of 1; otherwise it produces a "
    "reward of 0. These probabilities may differ across buttons. Your "
    "performance is your final cumulative score.\n\n"
    "Choose one button in each of {horizon} rounds to maximize your final "
    "cumulative score."
)

_UNTRIED_CUE = "Exploring this button may improve future rewards."

_P9_INSTRUCTION = (
    "Complete exactly two lines and stop after the Policy line. Use no more "
    f"than {RATIONALE_WORD_LIMIT} words total.\n\n"
    "First line: finish “Evidence:” by briefly comparing the strength "
    "and uncertainty of the available evidence.\n"
    "Second line: write either:\n"
    "“Policy: EXPLORE Button X because ...”\n"
    "or\n"
    "“Policy: EXPLOIT Button X because ...”\n\n"
    "Keep both lines concise. Name exactly one button. Do not repeat the task "
    "or continue after the Policy line."
)


def score_so_far(history: Sequence[tuple[str, int]]) -> int:
    """Cumulative task score = total reward. A deterministic restatement of
    the OPTIONS totals, so it adds salience, not information."""
    return sum(int(r) for _arm, r in history)


def render_state(
    arm_map_or_order: Mapping[str, float] | Sequence[str],
    history: Sequence[tuple[str, int]],
    round_idx: int,
    env: br.Environment,
) -> str:
    """PV9 Stage-1 state: pv7's OPTIONS table + score line + untried cue.

    Built by SURGERY on pv7's own rendered state rather than reimplemented, so
    the OPTIONS core cannot drift away from what Stage 2 renders. A rewrite
    here would silently break the "same counts in both stages" invariant that
    the information-isolation argument rests on.
    """
    state = p7.render_state(arm_map_or_order, history, round_idx, env,
                            prompt_variant=p7.PROMPT_P1B)

    # 4. Bernoulli structure: replace pv7's two-sentence header.
    old_header = p7._TASK_HEADER.format(horizon=env.horizon)
    if not state.startswith(old_header):
        raise AssertionError("pv7 header not found at the start of the state")
    state = _TASK_HEADER_PV9.format(horizon=env.horizon) + \
        state[len(old_header):]

    # 1. Self-relevant score, immediately after the round counter so the two
    #    task-progress facts sit together.
    round_line_re = re.compile(
        r"^Round \d+ of \d+\. Future choices after this one: \d+\.$", re.M)
    m = round_line_re.search(state)
    if m is None:
        raise AssertionError("round-counter line not found")
    score = score_so_far(history)
    state = (state[:m.end()]
             + f"\nYour score so far: {score} point{'' if score == 1 else 's'}."
             + state[m.end():])

    # 2. Untried cue, appended to n=0 rows only. Anchored on pv7's exact
    #    UNTRIED rendering so a change there fails loudly instead of silently
    #    dropping the cue.
    state = state.replace(": UNTRIED (unknown)",
                          f": UNTRIED (unknown). {_UNTRIED_CUE}")

    # History block, inherited from pv8 and rendered by pv8's own function.
    marker = "\n\nOPTIONS\n"
    if state.count(marker) != 1:
        raise AssertionError("expected exactly one OPTIONS table")
    head, tail = state.split(marker, 1)
    return f"{head}\n\n{p8.history_block(history)}{marker}{tail}"


def build_rationale_prompt(
    arm_map_or_order: Mapping[str, float] | Sequence[str],
    history: Sequence[tuple[str, int]],
    round_idx: int,
    env: br.Environment,
    prompt_variant: str = p7.PROMPT_P1B,
) -> str:
    """Stage 1 prompt, ending in exactly ``Evidence:<ASCII space>``.

    The trailing space is token 220 for the Llama-3.1 tokenizer, which is the
    site the RSN mask was extracted at; the assert is what keeps the steering
    intervention meaning the same thing it means everywhere else in the repo.
    """
    if prompt_variant != p7.PROMPT_P1B:
        raise AssertionError(
            f"PV9 builds on the P1b structure, got {prompt_variant!r}")
    state = render_state(arm_map_or_order, history, round_idx, env)
    prompt = f"{state}\n\n{_P9_INSTRUCTION}\n\n{p7.RATIONALE_ANCHOR}"
    p7._assert_single_trailing_space(prompt, p7.RATIONALE_ANCHOR)
    return prompt


def stop_reason(raw: str) -> str:
    """How this rationale ended. Judged on UNTRUNCATED generation output.

    This is the measurement that decides whether PV9 fixed termination or
    merely hid the failure, so it must never be fed post-processed text --
    every round would then read `native_clean`. `bandit_pv9_episode` stores
    `rationale_raw` byte-exact for exactly this reason.

        native_clean           terminated on its own, at the Policy line
        stop_marker_applied    ran on into a stop marker; the tail was cut
        continued_after_policy kept writing past the Policy line, no marker
        no_policy_line         never produced a Policy line
        empty                  produced nothing
    """
    text = raw or ""
    if not text.strip():
        return "empty"
    if any(m in text for m in STOP_STRINGS):
        return "stop_marker_applied"
    m = p7._POLICY_LINE.search(text)
    if m is None:
        return "no_policy_line"
    return ("continued_after_policy"
            if "\n" in text[m.end():].rstrip() else "native_clean")
