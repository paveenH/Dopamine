#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Pure prompt/state components for the pv7 Bandit protocol.

pv7 deliberately does not modify pv6.  It reuses only the frozen reference
environment identities/order supplied by :mod:`bandit_reference`; prompt
wording, rationale structure, anchors, and candidate tokens are new protocol
surface.

The load-bearing alignment rule is that BOTH generation stages end in one
literal ASCII space.  For the frozen Llama-3.1-8B-Instruct tokenizer this is
token 220, the same decision-bottleneck token used by the original RSN
``Answer: `` extraction.  Candidates must therefore be tokenized separately
as bare ``A``/``B``/... and appended at the ID level.  Building
``prompt + candidate`` as a string lets BPE merge ``<space> + A`` into the
single ``' A'`` token and silently destroys the intervention.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import bandit_reference as br


PROTOCOL_VERSION = "pv7"

PROMPT_P1 = "p1_structural"
PROMPT_P2 = "p2_light_hint"
PROMPT_P1B = "p1b_terminated"
PROMPT_VARIANTS = (PROMPT_P1, PROMPT_P2, PROMPT_P1B)

RATIONALE_ANCHOR = "Evidence: "
ACTION_ANCHOR = "Choose Button: "
TRAILING_SPACE = " "
EXPECTED_WHITESPACE_TOKEN_ID = 220
EXPECTED_CANDIDATE_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}

_TASK_HEADER = (
    "You will choose one button in each of {horizon} rounds. Each button has "
    "a fixed but unknown probability of reward 1. Maximize total reward."
)
_RATIONALE_INSTRUCTION = (
    "Write one line of no more than 35 words in the form "
    "\u201cEvidence: ...; Policy: ...\u201d. Briefly assess which estimates are "
    "well-supported or uncertain, then state whether to explore or exploit "
    "and why. The policy may name a button."
)
_LIGHT_HINT = (
    "Treat empirical rates based on very few trials as weak evidence; an "
    "untried button is unknown, not bad."
)
# P1b: P1's structure plus explicit TERMINATION and a mandatory named target.
# It fixes only Stage 1 defects measured on the frozen states -- 39.3% of P1
# policies named no button, and 17.8% of P1 generations continued into the next
# round's prompt. It deliberately adds NO decision hint (that was P2's failure:
# the hint became a policy prior and pushed the model to call tried arms
# untried). "Name exactly one button" is a FORMAT requirement, not a nudge
# toward explore or exploit.
_P1B_INSTRUCTION = (
    "Complete exactly two lines and stop after the Policy line. Use no more "
    "than 35 words total.\n\n"
    "First line: finish “Evidence:” by briefly stating which "
    "estimates are well-supported or uncertain.\n"
    "Second line: write either:\n"
    "“Policy: EXPLORE Button X because ...”\n"
    "or\n"
    "“Policy: EXPLOIT Button X because ...”\n\n"
    "Name exactly one button. Do not repeat the task or continue after the "
    "Policy line."
)


def _validate_variant(prompt_variant: str) -> None:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"unknown pv7 prompt variant {prompt_variant!r}; "
            f"expected one of {PROMPT_VARIANTS}")


def _counts(arm_order: Sequence[str], history: Sequence[tuple[str, int]]):
    trials = {arm: 0 for arm in arm_order}
    successes = {arm: 0 for arm in arm_order}
    for arm, reward in history:
        if arm not in trials:
            raise ValueError(f"history contains arm {arm!r} outside arm_order")
        if reward not in (0, 1):
            raise ValueError(f"reward must be 0 or 1, got {reward!r}")
        trials[arm] += 1
        successes[arm] += reward
    return trials, successes


def render_state(
    arm_map_or_order: Mapping[str, float] | Sequence[str],
    history: Sequence[tuple[str, int]],
    round_idx: int,
    env: br.Environment,
    prompt_variant: str = PROMPT_P1,
) -> str:
    """Render externally computed counts without exposing true probabilities.

    Rows stay in one fixed OPTIONS table and in the run's frozen display order.
    An untried arm is textual ``UNTRIED (unknown)`` rather than a numeric zero.
    ``round_idx`` is the zero-based index of the choice about to be made.
    """
    _validate_variant(prompt_variant)
    arm_order = list(arm_map_or_order)
    if len(arm_order) != env.k or len(set(arm_order)) != env.k:
        raise ValueError(
            f"arm order must contain exactly {env.k} distinct labels")
    if not 0 <= round_idx < env.horizon:
        raise ValueError(
            f"round_idx={round_idx} outside [0, {env.horizon})")
    if len(history) != round_idx:
        raise ValueError(
            f"history length {len(history)} != round_idx {round_idx}; "
            "the prompt must describe the state before the current choice")

    trials, successes = _counts(arm_order, history)
    option_lines = []
    for arm in arm_order:
        n, s = trials[arm], successes[arm]
        if n == 0:
            option_lines.append(f"- {arm}: UNTRIED (unknown)")
        else:
            option_lines.append(
                f"- {arm}: {s} reward{'' if s == 1 else 's'} / "
                f"{n} trial{'' if n == 1 else 's'}, empirical rate {s / n:.2f}")

    future = env.horizon - round_idx - 1
    lines = [
        _TASK_HEADER.format(horizon=env.horizon),
        "",
        f"Round {round_idx + 1} of {env.horizon}. "
        f"Future choices after this one: {future}.",
        "",
        "OPTIONS",
        *option_lines,
    ]
    if prompt_variant == PROMPT_P2:
        lines += ["", _LIGHT_HINT]
    return "\n".join(lines)


def build_rationale_prompt(
    arm_map_or_order: Mapping[str, float] | Sequence[str],
    history: Sequence[tuple[str, int]],
    round_idx: int,
    env: br.Environment,
    prompt_variant: str = PROMPT_P1,
) -> str:
    """Stage 1 prompt, ending in exactly ``Evidence:<ASCII space>``."""
    state = render_state(
        arm_map_or_order, history, round_idx, env, prompt_variant)
    instruction = (_P1B_INSTRUCTION if prompt_variant == PROMPT_P1B
                   else _RATIONALE_INSTRUCTION)
    prompt = f"{state}\n\n{instruction}\n\n{RATIONALE_ANCHOR}"
    # Load-bearing: the instruction is inserted BEFORE the anchor, so the tail
    # is still "Evidence: " -> token 220 and Stage 1 steering lands unchanged.
    _assert_single_trailing_space(prompt, RATIONALE_ANCHOR)
    return prompt


def sanitize_rationale(raw: str) -> str:
    """Preserve model content but remove trailing whitespace before Stage 2.

    pv7 deliberately permits the policy to name a button, so it does not drop
    semantic lines as pv6 did.  The ``rstrip`` is load-bearing: otherwise a
    rationale ending in a space followed by the action anchor can create a
    tokenizer-dependent double-space token rather than the audited token 220.
    """
    return (raw or "").rstrip()


_POLICY_LINE = re.compile(r"^\s*policy\s*[:：]", re.I | re.M)


def extract_evidence_policy_block(raw: str) -> str:
    """P1b clean text: the Evidence line plus the Policy line, and nothing after.

    RAW IS NEVER MODIFIED -- the caller stores both, so a continuation failure
    stays observable. This only decides what Stage 2 sees.

    Truncation is at the END OF THE POLICY LINE, which is what makes the
    "continue after the Policy line" failure mode harmless to Stage 2 while
    still being measurable on raw. If no Policy line exists there is nothing to
    truncate to, so the whole rationale is returned and the caller records the
    missing policy rather than silently inventing one.
    """
    text = (raw or "").rstrip()
    m = _POLICY_LINE.search(text)
    if m is None:
        return text
    end = text.find("\n", m.end())
    return (text if end == -1 else text[:end]).rstrip()


# ── policy parser (canonical home) ───────────────────────────────────────────
# Lives here, in the pure prompt/state module, so the production runner never
# has to import an analysis script to read its own output. HEURISTIC and
# POST-HOC DESCRIPTIVE: it records what Stage 1 said it intended. It NEVER
# selects the action -- that is always an argmax over the four candidate
# scores, which is what keeps `action_follows_policy` a measurement rather
# than a tautology.
#
# POLICY_PARSER_VERSION belongs in result metadata, not in the resume key:
# changing it re-labels stored text without changing a single trajectory.
POLICY_PARSER_VERSION = "pf1"

_POLICY_MARK = re.compile(r"policy\s*[:：]", re.I)
_EXPLORE_RE = re.compile(r"\bexplor\w*", re.I)
_EXPLOIT_RE = re.compile(r"\bexploit\w*", re.I)
_ARM_MENTION_RE = re.compile(r"button\s*([A-E])\b", re.I)
_DECISION_VERB_RE = re.compile(
    r"\b(explor\w*|exploit\w*|try|tries|choose|choos\w*|select\w*|pick\w*|"
    r"stick\s+with|continue\s+with|keep\s+with|switch\s+to|go\s+with)\b", re.I)


def first_decision_clause(body: str, m_pol) -> str:
    """First sentence/line after `Policy:`. Everything after it is explanation,
    and a button named there is not the policy target."""
    if m_pol is None:
        return ""
    seg = body[m_pol.end():].lstrip()
    line = seg.split("\n", 1)[0]
    m_end = re.search(r"[.!?](?:\s|$)", line)
    return line[:m_end.end()] if m_end else line


def parse_policy(clean: str, chosen: str | None = None) -> dict:
    """Stage 1's stated intent, and whether Stage 2 executed it.

    Target = the FIRST button after the decision verb in the policy's first
    clause. Not last-mention: 64-token truncation often ends mid-sentence on an
    unrelated arm, so "Policy: Explore Button D. Button C currently has the
    best estimate." would resolve to C. Measured disagreement between the two
    parsers was 20/102 rows (P1) and 46/95 (P2).

    `policy_parsed` is the strict gate used for headline statistics: an
    explicit `Policy:` marker, a clear explore/exploit stance, AND a named
    button. A loose full-text fallback is recorded for sensitivity only,
    because it will happily return a button mentioned in the Evidence half.
    """
    body = (clean or "").strip()
    explore, exploit = bool(_EXPLORE_RE.search(body)), bool(_EXPLOIT_RE.search(body))
    named = [f"Button {m.group(1).upper()}"
             for m in _ARM_MENTION_RE.finditer(body)]
    m_pol = _POLICY_MARK.search(body)

    clause_target = None
    clause = first_decision_clause(body, m_pol)
    if clause:
        m_verb = _DECISION_VERB_RE.search(clause)
        if m_verb:
            m_arm = _ARM_MENTION_RE.search(clause[m_verb.end():])
            if m_arm:
                clause_target = f"Button {m_arm.group(1).upper()}"

    if clause_target is not None:
        target, source = clause_target, "policy_first_clause"
    elif m_pol is not None:
        target, source = None, "policy_no_target"
    elif named:
        target, source = named[-1], "full_text_fallback"
    else:
        target, source = None, "none"

    stance = ("explore" if explore and not exploit else
              "exploit" if exploit and not explore else
              "both" if explore and exploit else "unclear")
    return {
        "policy_stance": stance,
        "stance_is_clear": stance in ("explore", "exploit"),
        "policy_names_button": bool(named),
        "policy_target": target,
        "policy_target_source": source,
        "policy_parsed": source == "policy_first_clause"
                         and stance in ("explore", "exploit"),
        "action_follows_policy": (None if target is None or chosen is None
                                  else target == chosen),
    }


def strip_policy_line(clean: str) -> str:
    """Drop the Policy line, keep the Evidence text verbatim.

    Used ONLY by the Stage-2 mask-policy control, which asks what the Policy
    text itself contributes once the same evidence is still present. Deleting
    exactly one line (not paraphrasing, not truncating the evidence) is what
    keeps that contrast interpretable.

    Removes the Policy line and everything after it: the frozen P1b interface
    truncates at the end of the Policy line already, so a trailing remainder is
    a continuation failure, not content the model meant Stage 2 to read.
    """
    text = (clean or "").rstrip()
    m = _POLICY_LINE.search(text)
    if m is None:
        return text
    return text[:m.start()].rstrip()


_ACTION_ANCHOR_MARKER = re.compile(r"choose\s+button\s*[:：]", re.I)
_EVIDENCE_ANCHOR_MARKER = re.compile(r"evidence\s*[:：]", re.I)


def rationale_format_flags(raw: str, clean: str | None = None) -> dict:
    """OBSERVATION ONLY -- these never clean, drop, or rewrite the rationale.

    pv7 invites the policy to name a button, so the model can emit the Stage 2
    anchor itself. That does NOT move the injection site (the prompt is still
    built to end at the anchor, and the runtime audit asserts token 220), but a
    rationale that already wrote "Choose Button: A" gives Stage 2 a self-cue,
    which would inflate exactly the action-follows-rationale readout Phase 3
    depends on. So it is recorded and reported, not filtered:

      * Report action-follows-rationale BOTH overall and on the collision-free
        subset. Silently dropping collisions creates a selection bias.
      * A materially non-zero collision rate disqualifies the PROMPT VERSION;
        it is not something to repair in post-processing.

    ``starts_with_redundant_evidence`` catches the model restating the anchor
    it was already given -- pv6 showed this model restates structure eagerly
    (all 2000 rationales spontaneously opened with "## Step 1").
    """
    raw = raw or ""
    clean = sanitize_rationale(raw) if clean is None else clean
    return {
        "empty_rationale": not clean.strip(),
        "rationale_contains_action_anchor":
            bool(_ACTION_ANCHOR_MARKER.search(clean)),
        "rationale_contains_evidence_anchor":
            bool(_EVIDENCE_ANCHOR_MARKER.search(clean)),
        "starts_with_redundant_evidence":
            bool(_EVIDENCE_ANCHOR_MARKER.match(clean.lstrip())),
    }


def build_action_prompt(
    arm_map_or_order: Mapping[str, float] | Sequence[str],
    history: Sequence[tuple[str, int]],
    round_idx: int,
    env: br.Environment,
    rationale_clean: str,
    prompt_variant: str = PROMPT_P1,
) -> str:
    """Stage 2 prompt, ending in exactly ``Choose Button:<ASCII space>``."""
    if rationale_clean != sanitize_rationale(rationale_clean):
        raise ValueError(
            "rationale_clean has trailing whitespace; call sanitize_rationale "
            "before building the action prompt")
    state = render_state(
        arm_map_or_order, history, round_idx, env, prompt_variant)
    analysis = f"Evidence: {rationale_clean}" if rationale_clean else "Evidence:"
    prompt = (
        f"{state}\n\nMODEL ANALYSIS\n{analysis}\n\n"
        "Now select one button to maximize total reward.\n"
        f"{ACTION_ANCHOR}"
    )
    _assert_single_trailing_space(prompt, ACTION_ANCHOR)
    return prompt


def candidate_suffixes(env: br.Environment) -> list[str]:
    """Bare arm letters; tokenize separately and append only at the ID level."""
    return [arm.removeprefix("Button ") for arm in br.ARM_LABELS[:env.k]]


def candidate_arm(candidate: str) -> str:
    if len(candidate) != 1 or not candidate.isalpha():
        raise ValueError(f"invalid bare candidate {candidate!r}")
    return f"Button {candidate}"


def _assert_single_trailing_space(prompt: str, anchor: str) -> None:
    if not prompt.endswith(anchor):
        raise AssertionError(f"prompt does not end in frozen anchor {anchor!r}")
    if not prompt.endswith(TRAILING_SPACE) or prompt.endswith(TRAILING_SPACE * 2):
        raise AssertionError("pv7 prompt must end in exactly one ASCII space")


def audit_id_level_continuation(tokenizer, prompt: str,
                                candidates: Sequence[str]) -> dict:
    """Fail closed if a candidate is not appended after the anchor token.

    This intentionally constructs the teacher-forcing sequence as
    ``prompt_ids + candidate_ids``.  Retokenizing ``prompt + candidate`` is
    reported only as a negative control; it must not equal the ID-level path.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise AssertionError("empty tokenized prompt")
    if prompt_ids[-1] != EXPECTED_WHITESPACE_TOKEN_ID:
        raise AssertionError(
            f"pv7 prompt tail is token {prompt_ids[-1]}, expected frozen "
            f"whitespace token {EXPECTED_WHITESPACE_TOKEN_ID}; check rstrip, "
            "double spaces, tokenizer identity, and anchor wording")
    rows = {}
    for candidate in candidates:
        cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
        expected_candidate_id = EXPECTED_CANDIDATE_TOKEN_IDS.get(candidate)
        if cand_ids != [expected_candidate_id]:
            raise AssertionError(
                f"pv7 candidate {candidate!r} tokenized as {cand_ids}, "
                f"expected [{expected_candidate_id}]")
        full_ids = prompt_ids + cand_ids
        merged_ids = tokenizer.encode(prompt + candidate, add_special_tokens=False)
        if len(prompt_ids) + len(cand_ids) != len(full_ids):
            raise AssertionError("ID-level candidate append changed sequence length")
        if full_ids[len(prompt_ids) - 1] != prompt_ids[-1]:
            raise AssertionError("prompt tail moved during ID-level append")
        if full_ids[len(prompt_ids):] != cand_ids:
            raise AssertionError("candidate IDs moved during ID-level append")
        if (full_ids[len(prompt_ids) - 1] != EXPECTED_WHITESPACE_TOKEN_ID
                or full_ids[len(prompt_ids)] != expected_candidate_id):
            raise AssertionError(
                "pv7 continuation boundary is not "
                f"[{EXPECTED_WHITESPACE_TOKEN_ID}, {expected_candidate_id}]")
        if merged_ids == full_ids:
            raise AssertionError(
                "string concatenation unexpectedly matches ID-level append; "
                "the whitespace-token negative control no longer detects merging")
        rows[candidate] = {
            "candidate_ids": cand_ids,
            "full_tail_ids": full_ids[-(len(cand_ids) + 1):],
            "string_merged_tail_ids": merged_ids[-2:],
        }
    return {
        "prompt_tail_id": prompt_ids[-1],
        "prompt_tail_token": tokenizer.decode([prompt_ids[-1]]),
        "candidates": rows,
    }
