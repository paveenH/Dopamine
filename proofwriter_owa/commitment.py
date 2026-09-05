#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FROZEN ProofWriter-native commitment extractor (PREREG_PROOFWRITER_OWA.md S9).

Frozen BEFORE any non-zero-alpha result is examined. Analyzes ONLY the
generated continuation. Keyed on the explicit marker of whichever family the
prompt actually used (`answer_parser.MARKER_FAMILIES`, resolved by
`prompt_template_id` -- v1's "Answer:" or v2's "####"), never on informal
true/false/unknown words in the reasoning body.

MARKER-FAMILY CONSISTENCY (v2 revision, 2026-09-05, human decision): a v2
generation's commitment timing MUST be computed against the v2 "####" marker,
never against v1's "Answer:" -- and vice versa. Before this revision,
`first_strict_marker_start` and `per_sample_commitment` were hardcoded to v1's
marker regardless of which prompt actually produced the text, so a v2 cell
(prompt asking for "#### <Label>") would silently find ZERO "Answer:"
markers and every commitment field would come back None/False for every
sample -- not a crash, a quiet, wrong-looking-healthy result. Every function
below now takes an explicit `marker_family` argument (`"v1"` or `"v2"`,
matching `answer_parser.MARKER_FAMILY_V1` / `_V2`) with NO default -- a
caller must look up the right family via `answer_parser.get_marker_family(
prompt_template_id)["marker_family"]` and pass it explicitly, rather than
this module silently assuming one. An inline "Answer:" occurrence inside a
v2-family generation (or a "####" occurrence inside a v1-family generation)
is picked up ONLY by that OTHER family's own loose/strict detectors if the
caller separately asks for them -- it is never folded into the active
family's own counts, so a v2 cell that also happens to emit "Answer: True"
somewhere does not silently inflate v2's marker count or get scored on it.

Metrics:
  answer_first_rate            fraction where the FIRST content on the first
                                non-whitespace line is itself a strict
                                "Answer: <Label>" marker (i.e. essentially no
                                reasoning precedes any answer)
  first_answer_marker_pos      normalized char position (0..1) of the first
                                STRICT marker's start, within the continuation
  pre_answer_reasoning_chars   chars before the first strict marker
  pre_answer_reasoning_tokens  tokenizer-based token count before the first
                                strict marker; None/omitted if no tokenizer
                                is supplied -- never estimated by a chars/4
                                heuristic
  reason_before_answer_rate    1 - answer_first_rate, restated for readability
  first_final_label_agreement  bool: does the FIRST strict marker's label
                                equal the LAST strict marker's label
  label_revision_rate          1 - mean(first_final_label_agreement), over
                                samples with >=2 strict markers only (a
                                single-marker sample cannot "revise")
  multiple_answer_marker_rate  fraction with >1 LOOSE marker occurrence
                                (diagnostic: the model restated/second-guessed
                                its answer somewhere in the text)

THESE ARE DESCRIPTIVE CO-OCCURRENCE STATISTICS ONLY. Never reported as causal
mediation evidence for an accuracy effect -- if a commitment metric changes
while accuracy does not, the only licensed conclusion is that submission
behavior changed without converting into task benefit.

Detector version: `commitment-v0` (2026-09-04, never re-tuned after seeing
non-zero-alpha data; if a future re-tuning is ever needed it gets a new
version string and both are reported side by side, matching the
`earlycand-v1` convention elsewhere in this repo).
"""

from __future__ import annotations

import re

from answer_parser import (MARKER_FAMILY_V1, MARKER_FAMILY_V2,
                           get_marker_family, find_all_markers,
                           find_all_markers_loose, find_all_markers_v2,
                           find_all_markers_loose_v2)

DETECTOR_VERSION = "commitment-v0"

# Per-family "is the first non-blank line itself a strict marker line" regex,
# and the strict line-anchored marker regex itself, keyed the same way
# answer_parser.MARKER_FAMILIES is. These must stay byte-identical to
# answer_parser._MARKER_RE / _MARKER_RE_V2 -- duplicated here (rather than
# importing the private regex objects) only because the "does the FIRST LINE
# match" check needs re.match against a single already-extracted line, not
# re.finditer/re.search over the whole text, so it is a distinct call shape.
# A change to either family's marker syntax in answer_parser.py must be
# mirrored here; test_commitment_marker_family.py's cross-check against
# answer_parser.find_all_markers[_v2] output is what would catch a drift.
_FIND_ALL = {
    MARKER_FAMILY_V1: find_all_markers,
    MARKER_FAMILY_V2: find_all_markers_v2,
}
_FIND_ALL_LOOSE = {
    MARKER_FAMILY_V1: find_all_markers_loose,
    MARKER_FAMILY_V2: find_all_markers_loose_v2,
}
_FIRST_LINE_MARKER_RE = {
    MARKER_FAMILY_V1: re.compile(r"(?i)^\s*answer\s*:\s*(true|false|unknown)\s*\.?\s*$"),
    MARKER_FAMILY_V2: re.compile(r"(?i)^\s*####\s*(true|false|unknown)\s*\.?\s*$"),
}
_FIRST_MARKER_SEARCH_RE = {
    MARKER_FAMILY_V1: re.compile(r"(?im)^\s*answer\s*:\s*(true|false|unknown)\s*\.?\s*$"),
    MARKER_FAMILY_V2: re.compile(r"(?im)^\s*####\s*(true|false|unknown)\s*\.?\s*$"),
}


def _require_known_family(marker_family: str, caller: str) -> None:
    if marker_family not in _FIND_ALL:
        raise ValueError(
            f"{caller}: marker_family={marker_family!r} is not one of "
            f"{sorted(_FIND_ALL)}. There is no default -- callers must "
            "resolve the family explicitly via "
            "answer_parser.get_marker_family(prompt_template_id)"
            "['marker_family'] and pass it here, never assume one.")


def _first_nonblank_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def first_strict_marker_start(text: str, marker_family: str) -> int | None:
    """Char offset of the first STRICT marker's start for the given
    `marker_family` ("v1" or "v2", from answer_parser.MARKER_FAMILY_V1/_V2),
    or None if no strict marker of THAT family is present. A "v1" call
    against text containing only "#### True" (no "Answer:" line) returns
    None, and vice versa -- the two families are never mixed.

    Public (not underscore-prefixed): get_answer_proofwriter_owa.py imports
    this directly to compute pre_answer_reasoning_tokens with the real
    tokenizer at generation time, where a real HF tokenizer is actually
    loaded -- see the module docstring on pre_answer_reasoning_tokens for
    why this must be computed there instead of estimated offline without a
    tokenizer.

    NO DEFAULT for marker_family (human decision, 2026-09-05, review
    finding): a default would silently keep resolving to whichever family
    was convenient at the time this function was first written, exactly the
    failure this whole registry exists to prevent when a second/third prompt
    version is added later."""
    _require_known_family(marker_family, "first_strict_marker_start")
    m = _FIRST_MARKER_SEARCH_RE[marker_family].search(text)
    return m.start() if m else None


def per_sample_commitment(continuation: str, marker_family: str,
                          tokenizer=None, precomputed_tokens=None) -> dict:
    """Compute all commitment fields for ONE sample's generated continuation,
    scored against exactly ONE marker family.

    marker_family: "v1" or "v2" (answer_parser.MARKER_FAMILY_V1/_V2). NO
    DEFAULT -- the caller must resolve this from the cell's own
    prompt_template_id via answer_parser.get_marker_family(...)
    ["marker_family"], not assume one. This is the fix for the bug found
    2026-09-05: before this argument existed, this function always scored
    against v1's "Answer:" marker regardless of which prompt actually
    produced `continuation`, so every v2 ("####"-prompted) generation
    silently returned has_marker=False / all timing fields None -- a quiet,
    healthy-looking wrong result, not a crash.

    An occurrence of the OTHER family's marker syntax inside `continuation`
    (e.g. a stray "Answer: True" inside an otherwise v2 "####"-family
    generation) is invisible to this function entirely -- it is neither
    counted, nor scored, nor mixed into n_strict_markers/n_loose_markers for
    the active family. Auditing cross-family leakage (how often does a v2
    generation ALSO emit a v1-style marker) is a separate, explicit call
    with marker_family set to the OTHER family, done by a caller that wants
    that diagnostic -- never folded into this function's own output.

    tokenizer: optional object with an __call__ or .encode method compatible
    with HF tokenizers (used only to count pre_answer_reasoning_tokens). If
    None AND precomputed_tokens is also None, pre_answer_reasoning_tokens is
    None -- never estimated by a chars/4 heuristic.

    precomputed_tokens: optional int, the pre-answer token count already
    computed elsewhere (get_answer_proofwriter_owa.py computes this at
    generation time, where a real tokenizer is actually loaded, and stores it
    per row as "pre_answer_reasoning_tokens" -- the offline evaluator has no
    model/tokenizer loaded, so it passes that stored value through here
    rather than silently recomputing None). Takes precedence over `tokenizer`
    when both are supplied; passing both is not expected in practice but
    precomputed_tokens winning is the more conservative choice (prefers the
    value actually computed against the real generation-time tokenizer).
    """
    _require_known_family(marker_family, "per_sample_commitment")
    strict = _FIND_ALL[marker_family](continuation)
    loose = _FIND_ALL_LOOSE[marker_family](continuation)
    first_line = _first_nonblank_line(continuation)
    first_marker_start = first_strict_marker_start(continuation, marker_family)

    # answer_first: the FIRST non-blank line IS itself a strict marker line
    # OF THIS SAME FAMILY -- a v2-family call never treats a first line of
    # "Answer: True" as answer_first, even though that line reads as a
    # marker under the OTHER family's syntax.
    answer_first = bool(strict) and bool(
        _FIRST_LINE_MARKER_RE[marker_family].match(first_line))

    pre_chars = first_marker_start if first_marker_start is not None else None
    pos_norm = (first_marker_start / len(continuation)
                if (first_marker_start is not None and len(continuation) > 0)
                else None)

    if precomputed_tokens is not None:
        pre_tokens = precomputed_tokens
    else:
        pre_tokens = None
        if tokenizer is not None and pre_chars is not None:
            prefix = continuation[:pre_chars]
            try:
                ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                pre_tokens = len(ids)
            except Exception:
                pre_tokens = None

    first_final_agree = None
    if len(strict) >= 1:
        first_final_agree = (strict[0] == strict[-1])

    return {
        "marker_family": marker_family,
        "n_strict_markers": len(strict),
        "n_loose_markers": len(loose),
        "has_marker": bool(strict),
        "answer_first": answer_first,
        "first_answer_marker_pos": pos_norm,
        "pre_answer_reasoning_chars": pre_chars,
        "pre_answer_reasoning_tokens": pre_tokens,
        "reason_before_answer": (not answer_first) if strict else None,
        "first_final_label_agreement": first_final_agree,
        "multiple_answer_marker": len(loose) > 1,
    }


def aggregate_commitment(rows: list[dict]) -> dict:
    """rows: list of per_sample_commitment() dicts. Returns dataset-level
    summary statistics, each computed on its own well-defined denominator
    (documented per field) rather than silently coercing None to 0."""

    def mean_of(key, predicate=lambda r: True):
        vals = [r[key] for r in rows if predicate(r) and r.get(key) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    n = len(rows)
    n_with_marker = sum(1 for r in rows if r["has_marker"])

    answer_first_rate, n_af = mean_of("answer_first", lambda r: r["has_marker"])
    pos_mean, n_pos = mean_of("first_answer_marker_pos")
    chars_mean, n_chars = mean_of("pre_answer_reasoning_chars")
    tok_mean, n_tok = mean_of("pre_answer_reasoning_tokens")

    revisable = [r for r in rows if r["n_strict_markers"] >= 2]
    label_revision_rate = (
        sum(1 for r in revisable if not r["first_final_label_agreement"]) / len(revisable)
        if revisable else None)

    multi_marker_rate = sum(1 for r in rows if r["multiple_answer_marker"]) / n if n else None

    return {
        "detector_version": DETECTOR_VERSION,
        "n_samples": n,
        "n_with_marker": n_with_marker,
        "marker_coverage": n_with_marker / n if n else None,
        "answer_first_rate": {"value": answer_first_rate, "n": n_af,
                              "denominator": "samples with >=1 strict marker"},
        "reason_before_answer_rate": {
            "value": (1 - answer_first_rate) if answer_first_rate is not None else None,
            "n": n_af, "denominator": "samples with >=1 strict marker"},
        "first_answer_marker_pos_mean": {"value": pos_mean, "n": n_pos,
                                         "denominator": "samples with >=1 strict marker"},
        "pre_answer_reasoning_chars_mean": {"value": chars_mean, "n": n_chars,
                                            "denominator": "samples with >=1 strict marker"},
        "pre_answer_reasoning_tokens_mean": {"value": tok_mean, "n": n_tok,
                                             "denominator": "samples with >=1 strict marker AND a tokenizer supplied"},
        "label_revision_rate": {"value": label_revision_rate, "n": len(revisable),
                                "denominator": "samples with >=2 strict markers"},
        "multiple_answer_marker_rate": {"value": multi_marker_rate, "n": n,
                                        "denominator": "all samples"},
        "scope_note": ("descriptive co-occurrence statistics only; never "
                       "causal mediation evidence for an accuracy effect"),
    }
