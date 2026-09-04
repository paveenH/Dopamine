#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final-label parser for ProofWriter OWA generations. FAIL-CLOSED.

Rules (PREREG_PROOFWRITER_OWA.md S8), verbatim:
  - parses only the generated continuation, never the input context (callers
    must pass ONLY the model's continuation, not prompt+continuation)
  - prefers the LAST valid "Answer: <Label>" marker
  - does NOT treat an ordinary true/false/unknown word in the reasoning body
    as a final answer -- only an explicit "Answer:" marker counts
  - if there is no unique legal final answer, it is a PARSE FAILURE
  - no LLM judge anywhere
"""

from __future__ import annotations

import re

VALID_LABELS = ("True", "False", "Unknown")

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
    """All STRICT (line-anchored) 'Answer: <Label>' markers, in order,
    labels normalized to canonical case. Used for the primary parse."""
    return [m.group(1).capitalize() if m.group(1).lower() != "unknown" else "Unknown"
            for m in _MARKER_RE.finditer(continuation)]


def find_all_markers_loose(continuation: str) -> list[str]:
    """All markers matched anywhere (not necessarily line-anchored). Used
    only for the multiple-marker DIAGNOSTIC rate, never for scoring."""
    out = []
    for m in _MARKER_LOOSE_RE.finditer(continuation):
        lab = m.group(1)
        out.append("Unknown" if lab.lower() == "unknown" else lab.capitalize())
    return out


class ParseResult:
    __slots__ = ("label", "status", "n_strict_markers", "n_loose_markers",
                 "all_strict_labels")

    def __init__(self, label, status, n_strict_markers, n_loose_markers,
                 all_strict_labels):
        self.label = label                      # None on failure
        self.status = status                     # "ok" | "no_marker" | "" (reserved)
        self.n_strict_markers = n_strict_markers
        self.n_loose_markers = n_loose_markers
        self.all_strict_labels = all_strict_labels

    @property
    def is_parse_failure(self) -> bool:
        return self.label is None

    def to_dict(self) -> dict:
        return {
            "label": self.label, "status": self.status,
            "n_strict_markers": self.n_strict_markers,
            "n_loose_markers": self.n_loose_markers,
            "all_strict_labels": self.all_strict_labels,
        }


def parse_final_answer(continuation: str) -> ParseResult:
    """Fail-closed: label is None (parse failure) unless there is at least
    one STRICT line-anchored 'Answer: <Label>' marker. When >=1 exist, the
    LAST one is authoritative -- this is a deliberate design choice (a model
    may revise its answer mid-reasoning and restate a corrected line last),
    not an accident of regex ordering.
    """
    strict = find_all_markers(continuation)
    loose = find_all_markers_loose(continuation)
    if not strict:
        return ParseResult(None, "no_marker", 0, len(loose), [])
    return ParseResult(strict[-1], "ok", len(strict), len(loose), strict)


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
