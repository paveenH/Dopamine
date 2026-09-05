#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final-label parser for ProofWriter OWA generations. FAIL-CLOSED.

Rules (PREREG_PROOFWRITER_OWA.md S8), verbatim, apply to BOTH marker families
below:
  - parses only the generated continuation, never the input context (callers
    must pass ONLY the model's continuation, not prompt+continuation)
  - prefers the LAST valid strict marker
  - does NOT treat an ordinary true/false/unknown word in the reasoning body
    as a final answer -- only an explicit marker line counts
  - if there is no unique legal final answer, it is a PARSE FAILURE
  - no LLM judge anywhere

TWO MARKER FAMILIES (v1 "Answer: <Label>", v2 "#### <Label>"), NEITHER
REPLACES THE OTHER. v1's frozen preflight results (PREREG_PROOFWRITER_OWA.md
S0.1, CLAUDE.md's ProofWriter-OWA row) were scored against the "Answer:"
family and must remain re-derivable byte-for-byte; v2
(prompt.PROMPT_TEMPLATE_ID == "proofwriter-owa-cot-v2", human decision
2026-09-05) asks the model for a "#### <Label>" line instead. Every function
below is duplicated per family (find_all_markers / find_all_markers_v2,
etc.) rather than made polymorphic on a marker-style argument, because a
single parameterized function invites the exact failure this repo's
frozen-marker conventions exist to prevent: silently scoring a v1-prompted
generation against the v2 marker family (or vice versa) because a caller
forgot to pass the right style. `parse_final_answer` / `parse_final_answer_v2`
are separate top-level entry points for the same reason; callers must pick
the one matching the prompt version they actually used, and
`get_answer_proofwriter_owa.py` records `prompt_template_id` precisely so a
downstream consumer can tell which one applies to a given cell.

Both families are STRICT (line-anchored, own line) for scoring; a LOOSE
(non-line-anchored) variant of each is kept only as a non-scoring diagnostic
count of how many marker-like occurrences exist anywhere in the text (human
decision, Q2, 2026-09-05: "只有独占一行的 #### True/False/Unknown 才能计入
评分... inline/loose marker 只作为诊断记录，不参与准确率。若有多个严格
marker，继续采用最后一个，并记录 marker 数量").
"""

from __future__ import annotations

import re

VALID_LABELS = ("True", "False", "Unknown")

# ───────────────────────── v1: "Answer: <Label>" ─────────────────────────

# Case-insensitive on the label word itself (some models emit "answer: true"),
# but the LABEL is always normalized to the canonical capitalized form. The
# marker word "Answer" is matched case-insensitively too, since a model may
# emit "ANSWER:" -- this is marker-DETECTION leniency, not content leniency:
# a bare "true" elsewhere in the text is still never treated as a marker.
_MARKER_RE = re.compile(
    r"(?im)^\s*answer\s*:\s*(true|false|unknown)\s*\.?\s*$"
)
# A looser variant that also matches the marker when it is not alone on its
# own line (e.g. "...so the Answer: True" mid-paragraph, or trailing text
# after the label on the same line). Used only to COUNT candidate markers for
# the multiple-marker diagnostic; the strict line-anchored regex above is
# what determines the scored final answer.
_MARKER_LOOSE_RE = re.compile(r"(?i)\banswer\s*:\s*(true|false|unknown)\b")


def find_all_markers(continuation: str) -> list[str]:
    """All STRICT (line-anchored) v1 'Answer: <Label>' markers, in order,
    labels normalized to canonical case. Used for the primary v1 parse."""
    return [m.group(1).capitalize() if m.group(1).lower() != "unknown" else "Unknown"
            for m in _MARKER_RE.finditer(continuation)]


def find_all_markers_loose(continuation: str) -> list[str]:
    """All v1 markers matched anywhere (not necessarily line-anchored). Used
    only for the multiple-marker DIAGNOSTIC rate, never for scoring."""
    out = []
    for m in _MARKER_LOOSE_RE.finditer(continuation):
        lab = m.group(1)
        out.append("Unknown" if lab.lower() == "unknown" else lab.capitalize())
    return out


# ───────────────────────── v2: "#### <Label>" ─────────────────────────

# STRICT: the marker must occupy its own line (optionally surrounded by
# whitespace and/or a single trailing period), matching this repo's
# GSM8K/MATH/BBH/LogiQA/CRUXEval-O "####"-marker convention (case-insensitive
# on the label word only, exactly mirroring the v1 regex's leniency choice).
_MARKER_RE_V2 = re.compile(
    r"(?im)^\s*####\s*(true|false|unknown)\s*\.?\s*$"
)
# LOOSE: matches "#### <Label>" anywhere, not necessarily alone on its own
# line (e.g. "...so #### True is my answer" mid-paragraph). Diagnostic-only,
# exactly mirroring _MARKER_LOOSE_RE's role for v1 -- never used for scoring.
_MARKER_LOOSE_RE_V2 = re.compile(r"(?i)####\s*(true|false|unknown)\b")


def find_all_markers_v2(continuation: str) -> list[str]:
    """All STRICT (line-anchored) v2 '#### <Label>' markers, in order,
    labels normalized to canonical case. Used for the primary v2 parse."""
    return [m.group(1).capitalize() if m.group(1).lower() != "unknown" else "Unknown"
            for m in _MARKER_RE_V2.finditer(continuation)]


def find_all_markers_loose_v2(continuation: str) -> list[str]:
    """All v2 markers matched anywhere (not necessarily line-anchored). Used
    only for the multiple-marker DIAGNOSTIC rate, never for scoring."""
    out = []
    for m in _MARKER_LOOSE_RE_V2.finditer(continuation):
        lab = m.group(1)
        out.append("Unknown" if lab.lower() == "unknown" else lab.capitalize())
    return out


class ParseResult:
    __slots__ = ("label", "status", "n_strict_markers", "n_loose_markers",
                 "all_strict_labels", "is_true_last_line")

    def __init__(self, label, status, n_strict_markers, n_loose_markers,
                 all_strict_labels, is_true_last_line):
        self.label = label                      # None on failure
        self.status = status                     # "ok" | "no_marker" | "" (reserved)
        self.n_strict_markers = n_strict_markers
        self.n_loose_markers = n_loose_markers
        self.all_strict_labels = all_strict_labels
        # DIAGNOSTIC ONLY, does not affect scoring (see review finding #5,
        # 2026-09-04): whether the authoritative (last strict) marker is
        # ALSO the true last non-blank line of the continuation, i.e.
        # whether the model actually followed the frozen prompt's literal
        # instruction ("on the LAST line of your response, output exactly
        # one of..."). False means the model wrote something after its
        # scored answer (commentary, a restated theory line, a second
        # attempt at reasoning, etc.) -- still scored (parse_final_answer's
        # "prefer the last marker" rule is an intentional revision-tolerance
        # choice, not a bug), but worth auditing separately from ordinary
        # parse failures.
        self.is_true_last_line = is_true_last_line

    @property
    def is_parse_failure(self) -> bool:
        return self.label is None

    def to_dict(self) -> dict:
        return {
            "label": self.label, "status": self.status,
            "n_strict_markers": self.n_strict_markers,
            "n_loose_markers": self.n_loose_markers,
            "all_strict_labels": self.all_strict_labels,
            "is_true_last_line": self.is_true_last_line,
        }


def _last_marker_is_true_last_line(continuation: str) -> bool:
    """True iff the LAST non-blank line of `continuation` is itself a strict
    v1 'Answer: <Label>' marker. Computed independently of which marker
    parse_final_answer scores, purely to audit prompt-instruction adherence
    ("on the LAST line of your response...")."""
    lines = [ln for ln in continuation.splitlines() if ln.strip()]
    if not lines:
        return False
    return bool(_MARKER_RE.match(lines[-1]))


def _last_marker_is_true_last_line_v2(continuation: str) -> bool:
    """v2 counterpart of _last_marker_is_true_last_line: True iff the LAST
    non-blank line is itself a strict '#### <Label>' marker."""
    lines = [ln for ln in continuation.splitlines() if ln.strip()]
    if not lines:
        return False
    return bool(_MARKER_RE_V2.match(lines[-1]))


def parse_final_answer(continuation: str) -> ParseResult:
    """v1 PARSER. Fail-closed: label is None (parse failure) unless there is
    at least one STRICT line-anchored 'Answer: <Label>' marker. When >=1
    exist, the LAST one is authoritative -- this is a deliberate design
    choice (a model may revise its answer mid-reasoning and restate a
    corrected line last), not an accident of regex ordering. Whether that
    marker is ALSO the literal last line of the response (as the frozen
    prompt instructs) is recorded separately in `is_true_last_line` and
    never changes `label`.
    """
    strict = find_all_markers(continuation)
    loose = find_all_markers_loose(continuation)
    if not strict:
        return ParseResult(None, "no_marker", 0, len(loose), [], False)
    return ParseResult(strict[-1], "ok", len(strict), len(loose), strict,
                       _last_marker_is_true_last_line(continuation))


def parse_final_answer_v2(continuation: str) -> ParseResult:
    """v2 PARSER (human decision, Q2, 2026-09-05): fail-closed, scores ONLY a
    STRICT line-anchored '#### <Label>' marker -- a marker that appears
    inline/mid-line ("...so #### True is my answer") is counted by the loose
    detector for diagnostics only and is NEVER promoted to a scored answer,
    even when it is the only marker-like text in the continuation. When >=1
    strict markers exist, the LAST one is authoritative (same
    revision-tolerance rationale as v1's parse_final_answer), and the total
    count of strict markers is always recorded via n_strict_markers so
    'multiple final markers' is visible per-sample regardless of which one
    was scored.
    """
    strict = find_all_markers_v2(continuation)
    loose = find_all_markers_loose_v2(continuation)
    if not strict:
        return ParseResult(None, "no_marker", 0, len(loose), [], False)
    return ParseResult(strict[-1], "ok", len(strict), len(loose), strict,
                       _last_marker_is_true_last_line_v2(continuation))


# ───────────────── marker-family registry (single source of truth) ─────────────────
#
# Resolves a `prompt_template_id` (the exact string every generation cell's
# meta already records) to the ONE marker family that must be used for BOTH
# scoring and commitment-timing on that cell. This registry exists so
# get_answer_proofwriter_owa.py, eval_proofwriter_owa.py, commitment.py, and
# compare_canary.py all pick the marker family the SAME way -- by looking up
# the id they already have, rather than four call sites separately hardcoding
# "this script means v1" or "this script means v2" and drifting apart the
# moment a fifth prompt version is ever added. A prompt_template_id with no
# entry here is a hard stop (get_marker_family raises KeyError-with-context),
# never a silent fallback to either family.
MARKER_FAMILY_V1 = "v1"
MARKER_FAMILY_V2 = "v2"

MARKER_FAMILIES = {
    "proofwriter-owa-cot-v1": {
        "marker_family": MARKER_FAMILY_V1,
        "marker_prefix": "Answer:",
        "parse_final_answer": None,       # filled in below, after the fns exist
        "find_all_markers": None,
        "find_all_markers_loose": None,
    },
    "proofwriter-owa-cot-v2": {
        "marker_family": MARKER_FAMILY_V2,
        "marker_prefix": "####",
        "parse_final_answer": None,
        "find_all_markers": None,
        "find_all_markers_loose": None,
    },
}


def get_marker_family(prompt_template_id: str) -> dict:
    """The single lookup point every consumer must use. Raises with a clear
    message (not KeyError's bare traceback) on an unregistered
    prompt_template_id -- fail closed, never guess a family for a prompt
    version this module does not know about."""
    fam = MARKER_FAMILIES.get(prompt_template_id)
    if fam is None:
        raise ValueError(
            f"no marker family registered for prompt_template_id="
            f"{prompt_template_id!r}. Known ids: "
            f"{sorted(MARKER_FAMILIES)}. Register a new entry in "
            "answer_parser.MARKER_FAMILIES before using a new prompt "
            "version for scoring or commitment timing -- never assume a "
            "default family for an unrecognized id.")
    return fam


def normalize_label(x) -> str:
    """Normalize a gold or predicted label to the canonical 3-value form."""
    if isinstance(x, bool):
        return "True" if x else "False"
    s = str(x).strip()
    low = s.lower()
    if low == "true":
        return "True"
    if low == "false":
        return "False"
    if low == "unknown":
        return "Unknown"
    raise ValueError(f"unrecognized label {x!r}")


def is_correct(pred_label: str | None, gold_label: str) -> bool:
    if pred_label is None:
        return False
    return normalize_label(pred_label) == normalize_label(gold_label)


# Fill in MARKER_FAMILIES now that every function it references exists.
# Done here (module-bottom) rather than inline in the dict literal above
# because Python cannot reference a function defined later in the same
# module inside an earlier dict literal -- keeping the dict declaration next
# to the family constants (readable) and the population step here (correct
# forward-reference order) is the standard pattern for this kind of registry.
MARKER_FAMILIES["proofwriter-owa-cot-v1"].update({
    "parse_final_answer": parse_final_answer,
    "find_all_markers": find_all_markers,
    "find_all_markers_loose": find_all_markers_loose,
})
MARKER_FAMILIES["proofwriter-owa-cot-v2"].update({
    "parse_final_answer": parse_final_answer_v2,
    "find_all_markers": find_all_markers_v2,
    "find_all_markers_loose": find_all_markers_loose_v2,
})
