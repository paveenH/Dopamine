#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for proofwriter_owa/answer_parser.py. No GPU, no network, stdlib
only (the module itself has no non-stdlib dependency).

    python3 test_answer_parser.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answer_parser import (parse_final_answer, find_all_markers,
                            find_all_markers_loose, normalize_label, is_correct)

FAILS = []
N = 0


def check(cond, msg):
    global N
    N += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


def test_basic_last_marker_wins():
    text = ("Step 1: X implies Y.\nAnswer: True\nWait, reconsidering...\n"
            "Answer: False\n")
    r = parse_final_answer(text)
    check(r.label == "False", f"last marker should win, got {r.label!r}")
    check(r.n_strict_markers == 2, f"expected 2 strict markers, got {r.n_strict_markers}")


def test_single_marker():
    text = "Some reasoning here.\nAnswer: Unknown\n"
    r = parse_final_answer(text)
    check(r.label == "Unknown", f"got {r.label!r}")
    check(r.status == "ok", f"got status={r.status!r}")


def test_no_marker_is_parse_failure():
    text = "I think the answer is true, but I am not fully sure."
    r = parse_final_answer(text)
    check(r.is_parse_failure, "no strict marker should be a parse failure")
    check(r.label is None, f"expected label None, got {r.label!r}")


def test_informal_word_not_treated_as_marker():
    # "true"/"false"/"unknown" appearing in prose must NOT be picked up as a
    # final answer -- only the "Answer: X" marker counts.
    text = ("It is true that Bob is red. It is false that Bob is blue. "
            "The status of the rule is unknown to me here.")
    r = parse_final_answer(text)
    check(r.is_parse_failure,
          "informal true/false/unknown words must not be treated as an "
          "'Answer:' marker")


def test_marker_case_insensitive_but_normalized():
    text = "reasoning...\nanswer: true"
    r = parse_final_answer(text)
    check(r.label == "True", f"lowercase marker should normalize to 'True', got {r.label!r}")

    text2 = "reasoning...\nANSWER: FALSE"
    r2 = parse_final_answer(text2)
    check(r2.label == "False", f"got {r2.label!r}")


def test_marker_must_be_line_anchored_for_strict():
    # A marker embedded mid-sentence should NOT count as a strict marker (it
    # should not be the scored final answer), but SHOULD be counted by the
    # loose detector (diagnostic only).
    text = "So the Answer: True is what I conclude, though let me double check."
    strict = find_all_markers(text)
    loose = find_all_markers_loose(text)
    check(strict == [], f"mid-sentence marker must not count as strict, got {strict}")
    check(len(loose) == 1, f"loose detector should still see it, got {loose}")
    r = parse_final_answer(text)
    check(r.is_parse_failure,
          "a mid-sentence 'Answer:' occurrence must be a parse failure under "
          "the strict (scored) parser")


def test_trailing_punctuation_and_whitespace_tolerated():
    for text, expect in [
        ("Answer: True.", "True"),
        ("Answer: True ", "True"),
        ("  Answer:   False  ", "False"),
    ]:
        r = parse_final_answer(text)
        check(r.label == expect, f"{text!r} -> expected {expect}, got {r.label!r}")


def test_no_answer_word_after_label_confuses_nothing():
    # A line like "Answer: True (query is provable)" should NOT match the
    # strict line-anchored regex (extra trailing content after the label) --
    # this is a deliberate design choice: only a BARE "Answer: <Label>" line
    # counts as strict, so a model padding the line with commentary is a
    # parse failure under MAIN scoring (still visible via the loose count).
    text = "Answer: True (because the rule fires)"
    r = parse_final_answer(text)
    check(r.is_parse_failure,
          "a label line with trailing commentary must not be treated as a "
          "clean strict marker")
    loose = find_all_markers_loose(text)
    check(loose == ["True"], f"loose detector should still see it, got {loose}")


def test_normalize_label():
    check(normalize_label(True) == "True", "bool True -> 'True'")
    check(normalize_label(False) == "False", "bool False -> 'False'")
    check(normalize_label("unknown") == "Unknown", "'unknown' -> 'Unknown'")
    check(normalize_label("UNKNOWN") == "Unknown", "'UNKNOWN' -> 'Unknown'")
    try:
        normalize_label("maybe")
        check(False, "normalize_label('maybe') should raise ValueError")
    except ValueError:
        check(True, "normalize_label('maybe') correctly raises")


def test_is_correct():
    check(is_correct("True", "True") is True, "True==True should be correct")
    check(is_correct("True", "False") is False, "True!=False should be incorrect")
    check(is_correct(None, "True") is False, "parse failure (None) is never correct")


def test_multi_label_first_vs_last_are_distinguishable():
    text = "Answer: True\nreconsider...\nAnswer: True\n"
    r = parse_final_answer(text)
    check(r.all_strict_labels == ["True", "True"],
          f"expected two identical markers recorded, got {r.all_strict_labels}")


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
