#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for the marker-family fix (2026-09-05, human decision):
commitment timing and final-answer scoring must use the SAME marker family
(v1 "Answer:" or v2 "####"), resolved explicitly from a cell's own
PROMPT_TEMPLATE_ID / prompt_template_id, never a hardcoded default.

Before this fix, commitment.first_strict_marker_start and
commitment.per_sample_commitment were hardcoded to v1's "Answer:" regex
regardless of which prompt actually produced the text. Since prompt.py's v2
revision made "#### <Label>" the module's active default, every v2
generation would silently compute pre_answer_reasoning_tokens=None and every
other commitment field as if no marker existed at all -- not a crash, a
quiet, healthy-looking wrong result.

Covers the five scenarios named in the human decision, 2026-09-05:
  1. v1/v2 both find the correct first strict marker position.
  2. An inline (non-line-anchored) marker is not counted as a strict marker
     in either family.
  3. A v2-family call against text using only "Answer:" finds NO strict
     marker.
  4. A v1-family call against text using only "####" finds NO strict marker.
  5. pre_answer_reasoning_tokens is no longer None for legitimate v2 output
     (i.e. the get_answer_proofwriter_owa.py code path, simulated here with
     the same commitment.per_sample_commitment call it makes).

No GPU, no network, stdlib only.

    python3 test_commitment_marker_family.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commitment import first_strict_marker_start, per_sample_commitment
from answer_parser import (MARKER_FAMILY_V1, MARKER_FAMILY_V2,
                           get_marker_family, MARKER_FAMILIES)

FAILS = []
N = 0


def check(cond, msg):
    global N
    N += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


V1 = MARKER_FAMILY_V1
V2 = MARKER_FAMILY_V2


# ───────────────── 1. correct first strict marker position, both families ─────────────────

def test_v1_first_strict_marker_position():
    text = "abc\nAnswer: False\n"
    pos = first_strict_marker_start(text, V1)
    check(pos == len("abc\n"), f"expected offset {len('abc' + chr(10))}, got {pos}")


def test_v2_first_strict_marker_position():
    text = "abc\n#### False\n"
    pos = first_strict_marker_start(text, V2)
    check(pos == len("abc\n"), f"expected offset {len('abc' + chr(10))}, got {pos}")


def test_v1_and_v2_agree_on_position_for_equivalent_texts():
    # Same reasoning prefix, different marker syntax -- the OFFSET of the
    # marker's start should be identical since the prefixes are identical.
    prefix = "Step 1: X implies Y.\nStep 2: Y implies Z.\n"
    pos_v1 = first_strict_marker_start(prefix + "Answer: True\n", V1)
    pos_v2 = first_strict_marker_start(prefix + "#### True\n", V2)
    check(pos_v1 == len(prefix), f"v1 offset should be {len(prefix)}, got {pos_v1}")
    check(pos_v2 == len(prefix), f"v2 offset should be {len(prefix)}, got {pos_v2}")
    check(pos_v1 == pos_v2, "equivalent texts should give equal offsets across families")


# ───────────────── 2. inline marker not counted as strict, either family ─────────────────

def test_v1_inline_marker_not_strict():
    text = "So the Answer: True is what I conclude, though let me double check."
    pos = first_strict_marker_start(text, V1)
    check(pos is None, f"a mid-sentence v1 marker must not be strict, got offset {pos}")
    c = per_sample_commitment(text, V1)
    check(c["has_marker"] is False, "inline v1 marker -> has_marker False")
    check(c["n_loose_markers"] == 1, f"loose count should still see it, got {c['n_loose_markers']}")


def test_v2_inline_marker_not_strict():
    text = "So the #### True is what I conclude, though let me double check."
    pos = first_strict_marker_start(text, V2)
    check(pos is None, f"a mid-sentence v2 marker must not be strict, got offset {pos}")
    c = per_sample_commitment(text, V2)
    check(c["has_marker"] is False, "inline v2 marker -> has_marker False")
    check(c["n_loose_markers"] == 1, f"loose count should still see it, got {c['n_loose_markers']}")


# ───────────────── 3 & 4. cross-family: neither marker satisfies the other's parser ─────────────────

def test_v2_family_finds_nothing_in_v1_only_text():
    text = "reasoning here\nAnswer: True\n"
    pos = first_strict_marker_start(text, V2)
    check(pos is None,
          "a v2-family call against text with ONLY 'Answer:' must find no "
          "strict marker")
    c = per_sample_commitment(text, V2)
    check(c["has_marker"] is False, f"expected has_marker False, got {c}")
    check(c["n_strict_markers"] == 0, f"expected 0 strict markers, got {c['n_strict_markers']}")
    check(c["pre_answer_reasoning_tokens"] is None,
          "with no v2 marker present at all, pre_answer_reasoning_tokens "
          "must be None (there is nothing to be 'before')")


def test_v1_family_finds_nothing_in_v2_only_text():
    text = "reasoning here\n#### True\n"
    pos = first_strict_marker_start(text, V1)
    check(pos is None,
          "a v1-family call against text with ONLY '####' must find no "
          "strict marker")
    c = per_sample_commitment(text, V1)
    check(c["has_marker"] is False, f"expected has_marker False, got {c}")
    check(c["n_strict_markers"] == 0, f"expected 0 strict markers, got {c['n_strict_markers']}")


# ───────────────── 5. pre_answer_reasoning_tokens is no longer None for real v2 output ─────────────────

class FakeTokenizer:
    """Mimics the shape get_answer_proofwriter_owa.py's real HF tokenizer
    call uses: __call__(text, add_special_tokens=False) -> {"input_ids": [...]}"""
    def __call__(self, s, add_special_tokens=False):
        # One "token" per whitespace-separated word, deterministic and
        # trivially checkable -- the exact count doesn't matter, only that
        # it is a real int, not None.
        return {"input_ids": list(range(len(s.split())))}


def test_pre_answer_reasoning_tokens_populated_for_v2_output():
    """Simulates exactly what get_answer_proofwriter_owa.py does at
    generation time: resolve marker_family from PROMPT_TEMPLATE_ID, find the
    marker start with THAT family, then tokenize the prefix. Before the fix,
    this always used v1's marker regardless of the actual prompt, so a v2
    ("####") generation always got pre_answer_reasoning_tokens=None."""
    text = "Step 1: theory says X. Step 2: therefore Y.\n#### True\n"
    marker_family = get_marker_family("proofwriter-owa-cot-v2")["marker_family"]
    check(marker_family == V2, f"v2 template should resolve to {V2!r}, got {marker_family!r}")

    marker_start = first_strict_marker_start(text, marker_family)
    check(marker_start is not None, "v2 marker must be found in legitimate v2 output")

    tok = FakeTokenizer()
    pre_answer_tokens = None
    if marker_start is not None:
        pre_answer_tokens = len(tok(text[:marker_start], add_special_tokens=False)["input_ids"])

    check(pre_answer_tokens is not None,
          "pre_answer_reasoning_tokens must NOT be None for legitimate v2 "
          "output -- this is exactly the bug the 2026-09-05 fix addresses")
    check(pre_answer_tokens > 0,
          f"expected a positive pre-answer token count, got {pre_answer_tokens}")

    # Cross-check: the SAME computation via per_sample_commitment with
    # precomputed_tokens threaded through (the real evaluator's code path).
    c = per_sample_commitment(text, marker_family, precomputed_tokens=pre_answer_tokens)
    check(c["pre_answer_reasoning_tokens"] == pre_answer_tokens,
          f"per_sample_commitment should pass precomputed_tokens through "
          f"unchanged, got {c['pre_answer_reasoning_tokens']}")
    check(c["marker_family"] == V2, "per_sample_commitment must record which family it used")


def test_pre_answer_reasoning_tokens_still_populated_for_v1_output():
    """Symmetric check: v1 output must still get a real token count too,
    proving the fix did not merely swap which family is broken."""
    text = "Step 1: theory says X.\nAnswer: False\n"
    marker_family = get_marker_family("proofwriter-owa-cot-v1")["marker_family"]
    check(marker_family == V1, f"v1 template should resolve to {V1!r}, got {marker_family!r}")

    marker_start = first_strict_marker_start(text, marker_family)
    check(marker_start is not None, "v1 marker must be found in legitimate v1 output")

    tok = FakeTokenizer()
    pre_answer_tokens = len(tok(text[:marker_start], add_special_tokens=False)["input_ids"])
    check(pre_answer_tokens > 0, f"expected a positive token count, got {pre_answer_tokens}")


def test_wrong_family_would_have_silently_broken_v2():
    """Mutation-style check: confirm using V1 (the OLD hardcoded default)
    against real v2 output really would produce the broken behavior this fix
    exists to prevent -- i.e. this is not a strawman."""
    v2_text = "Some genuine multi-step reasoning about the theory.\n#### Unknown\n"
    wrong_family_start = first_strict_marker_start(v2_text, V1)
    check(wrong_family_start is None,
          "confirms the pre-fix hardcoded-v1 behavior against real v2 "
          "output would have found NO marker at all -- proving this was a "
          "real, silent bug, not a hypothetical one")
    c_wrong = per_sample_commitment(v2_text, V1)
    check(c_wrong["has_marker"] is False,
          "confirms per_sample_commitment(text) with the old hardcoded v1 "
          "family would have reported has_marker=False for genuine, "
          "well-formed v2 output")


def test_no_default_marker_family_raises():
    """per_sample_commitment and first_strict_marker_start must require
    marker_family explicitly -- there is no default to silently fall back
    on. Confirmed via TypeError (missing required positional argument)."""
    try:
        per_sample_commitment("Answer: True\n")  # type: ignore[call-arg]
        check(False, "per_sample_commitment must require marker_family "
              "(TypeError expected, none raised)")
    except TypeError:
        check(True, "per_sample_commitment correctly requires marker_family")

    try:
        first_strict_marker_start("Answer: True\n")  # type: ignore[call-arg]
        check(False, "first_strict_marker_start must require marker_family "
              "(TypeError expected, none raised)")
    except TypeError:
        check(True, "first_strict_marker_start correctly requires marker_family")


def test_unknown_marker_family_raises_value_error():
    try:
        per_sample_commitment("Answer: True\n", "v3-does-not-exist")
        check(False, "an unregistered marker_family must raise ValueError")
    except ValueError as e:
        check("marker_family" in str(e), f"error should mention marker_family, got: {e}")


def test_unregistered_prompt_template_id_raises():
    try:
        get_marker_family("some-future-prompt-id-v99")
        check(False, "an unregistered prompt_template_id must raise")
    except ValueError as e:
        check("MARKER_FAMILIES" in str(e) or "marker family" in str(e).lower(),
              f"error should point at the registry, got: {e}")


def test_registry_has_exactly_v1_and_v2():
    check(set(MARKER_FAMILIES) == {"proofwriter-owa-cot-v1", "proofwriter-owa-cot-v2"},
          f"expected exactly the v1/v2 template ids registered, got {sorted(MARKER_FAMILIES)}")
    for tpl_id, fam in MARKER_FAMILIES.items():
        check(callable(fam["parse_final_answer"]),
              f"{tpl_id}: parse_final_answer must be a real function, not the "
              "placeholder None left before population")
        check(callable(fam["find_all_markers"]),
              f"{tpl_id}: find_all_markers must be a real function")
        check(callable(fam["find_all_markers_loose"]),
              f"{tpl_id}: find_all_markers_loose must be a real function")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{N - len(FAILS)}/{N} checks passed")
    if FAILS:
        print(f"{len(FAILS)} FAILURES")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
