#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the v2 "#### <Label>" marker family in
proofwriter_owa/answer_parser.py. No GPU, no network, stdlib only.

Mirrors test_answer_parser.py's v1 coverage exactly (same scenarios, same
expected behavior), plus an explicit cross-family isolation section proving
v1 and v2 markers never leak into each other's parse -- the exact failure
mode the module docstring warns a single parameterized parser would invite.

    python3 test_answer_parser_v2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answer_parser import (parse_final_answer, parse_final_answer_v2,
                            find_all_markers_v2, find_all_markers_loose_v2,
                            normalize_label, is_correct)

FAILS = []
N = 0


def check(cond, msg):
    global N
    N += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


def test_basic_first_marker_wins_v2():
    # Revised 2026-09-05 (human decision): MAIN scoring is FIRST-marker, not
    # last -- Llama3's v2 preflight showed a genuine answer followed by a
    # degenerate trailing loop, which "last wins" would have scored instead
    # of the model's actual (first) answer.
    text = ("Step 1: X implies Y.\n#### True\nWait, reconsidering...\n"
            "#### False\n")
    r = parse_final_answer_v2(text)
    check(r.label == "True", f"first marker should win, got {r.label!r}")
    check(r.n_strict_markers == 2, f"expected 2 strict markers, got {r.n_strict_markers}")
    check(r.all_strict_labels[-1] == "False",
          "all_strict_labels[-1] must still recover the last-marker label "
          "for the last-answer sensitivity analysis")


def test_single_marker_v2():
    text = "Some reasoning here.\n#### Unknown\n"
    r = parse_final_answer_v2(text)
    check(r.label == "Unknown", f"got {r.label!r}")
    check(r.status == "ok", f"got status={r.status!r}")


def test_no_marker_is_parse_failure_v2():
    text = "I think the answer is true, but I am not fully sure."
    r = parse_final_answer_v2(text)
    check(r.is_parse_failure, "no strict marker should be a parse failure")
    check(r.label is None, f"expected label None, got {r.label!r}")


def test_informal_word_not_treated_as_marker_v2():
    text = ("It is true that Bob is red. It is false that Bob is blue. "
            "The status of the rule is unknown to me here.")
    r = parse_final_answer_v2(text)
    check(r.is_parse_failure,
          "informal true/false/unknown words must not be treated as a "
          "'####' marker")


def test_marker_case_insensitive_but_normalized_v2():
    text = "reasoning...\n#### true"
    r = parse_final_answer_v2(text)
    check(r.label == "True", f"lowercase marker should normalize to 'True', got {r.label!r}")

    text2 = "reasoning...\n#### FALSE"
    r2 = parse_final_answer_v2(text2)
    check(r2.label == "False", f"got {r2.label!r}")


def test_marker_must_be_line_anchored_for_strict_v2():
    # Human decision (Q2, 2026-09-05): only a marker alone on its own line
    # counts toward scoring; inline occurrences are diagnostic-only.
    text = "So the #### True is what I conclude, though let me double check."
    strict = find_all_markers_v2(text)
    loose = find_all_markers_loose_v2(text)
    check(strict == [], f"mid-sentence marker must not count as strict, got {strict}")
    check(len(loose) == 1, f"loose detector should still see it, got {loose}")
    r = parse_final_answer_v2(text)
    check(r.is_parse_failure,
          "a mid-sentence '####' occurrence must be a parse failure under "
          "the strict (scored) v2 parser")


def test_trailing_punctuation_and_whitespace_tolerated_v2():
    for text, expect in [
        ("#### True.", "True"),
        ("#### True ", "True"),
        ("  ####   False  ", "False"),
    ]:
        r = parse_final_answer_v2(text)
        check(r.label == expect, f"{text!r} -> expected {expect}, got {r.label!r}")


def test_trailing_commentary_after_label_is_parse_failure_v2():
    # Same deliberate design choice as v1: a bare "#### <Label>" line is
    # required for the strict marker; padding after the label disqualifies
    # it from MAIN scoring (still visible via the loose count).
    text = "#### True (because the rule fires)"
    r = parse_final_answer_v2(text)
    check(r.is_parse_failure,
          "a label line with trailing commentary must not be treated as a "
          "clean strict v2 marker")
    loose = find_all_markers_loose_v2(text)
    check(loose == ["True"], f"loose detector should still see it, got {loose}")


def test_multiple_strict_markers_recorded_v2():
    text = "#### True\nreconsider...\n#### False\n#### Unknown\n"
    r = parse_final_answer_v2(text)
    check(r.n_strict_markers == 3, f"expected 3 strict markers recorded, got {r.n_strict_markers}")
    check(r.label == "True", f"first of 3 should win, got {r.label!r}")
    check(r.all_strict_labels == ["True", "False", "Unknown"],
          f"expected all three recorded in order, got {r.all_strict_labels}")


def test_degenerate_trailing_loop_does_not_overwrite_first_answer_v2():
    # The exact failure pattern found in Llama3's v2 preflight: a genuine
    # answer is submitted, then the model loops emitting further
    # marker-shaped text (a repeated True/False/Unknown enumeration, or a
    # repeated disclaimer) until the token budget is exhausted.
    text = ("Step 1: reasoning...\n#### Unknown\n"
            + "#### True\n#### False\n#### Unknown\n" * 50)
    r = parse_final_answer_v2(text)
    check(r.label == "Unknown",
          f"a trailing degenerate loop must not overwrite the first, "
          f"genuine answer, got {r.label!r}")
    check(r.n_strict_markers == 151, f"expected 151 strict markers, got {r.n_strict_markers}")


def test_is_true_last_line_v2():
    ok_text = "reasoning\n#### True\n"
    r_ok = parse_final_answer_v2(ok_text)
    check(r_ok.is_true_last_line is True,
          "marker alone on the true last line should read True")

    trailing_text = "reasoning\n#### True\nsome trailing commentary\n"
    r_trailing = parse_final_answer_v2(trailing_text)
    check(not r_trailing.is_parse_failure and r_trailing.label == "True",
          "the marker line itself is still strict and still scored -- "
          "trailing commentary on a LATER line does not retroactively "
          "invalidate an earlier strict marker")
    check(r_trailing.is_true_last_line is False,
          "but that marker is NOT the true last line, since commentary "
          "follows it on its own line")

    revised_text = "#### True\nWait, reconsidering.\n#### False\n"
    r_revised = parse_final_answer_v2(revised_text)
    # Revised 2026-09-05: MAIN scoring takes the FIRST strict marker, so
    # `label` is "True" here even though the model appears to revise to
    # "False" -- is_true_last_line is a SEPARATE, purely descriptive check
    # of whether the LAST strict marker (not necessarily the scored one)
    # sits on the true last line; it is unaffected by which marker is
    # scored and correctly still reads True here.
    check(r_revised.label == "True", f"got {r_revised.label!r}")
    check(r_revised.all_strict_labels[-1] == "False",
          "the last marker is still recoverable via all_strict_labels[-1]")
    check(r_revised.is_true_last_line is True,
          "the LAST marker here is also the true last line (this check is "
          "independent of which marker was scored)")


# ───────────────────── cross-family isolation ─────────────────────

def test_v1_markers_do_not_satisfy_v2_parser():
    text = "Some reasoning.\nAnswer: True\n"
    r2 = parse_final_answer_v2(text)
    check(r2.is_parse_failure,
          "a v1 'Answer: True' marker must NOT be picked up by the v2 "
          "'####' parser -- the two families must never silently mix")


def test_v2_markers_do_not_satisfy_v1_parser():
    text = "Some reasoning.\n#### True\n"
    r1 = parse_final_answer(text)
    check(r1.is_parse_failure,
          "a v2 '#### True' marker must NOT be picked up by the v1 "
          "'Answer:' parser -- the two families must never silently mix")


def test_both_families_present_each_parses_only_its_own():
    # A pathological but possible generation carrying both marker styles
    # (e.g. a model that saw a v1-style exemplar leak into its training
    # distribution) -- each parser must score ONLY its own family, entirely
    # ignoring occurrences of the other.
    text = "Answer: True\nreconsidering...\n#### False\n"
    r1 = parse_final_answer(text)
    r2 = parse_final_answer_v2(text)
    check(r1.label == "True", f"v1 parser should see only its own marker, got {r1.label!r}")
    check(r1.n_strict_markers == 1, f"v1 should count exactly 1, got {r1.n_strict_markers}")
    check(r2.label == "False", f"v2 parser should see only its own marker, got {r2.label!r}")
    check(r2.n_strict_markers == 1, f"v2 should count exactly 1, got {r2.n_strict_markers}")


def test_normalize_label_and_is_correct_unaffected():
    # normalize_label/is_correct are marker-family-agnostic; confirm v2's
    # parser output feeds them identically to v1's.
    check(normalize_label("unknown") == "Unknown", "'unknown' -> 'Unknown'")
    r2 = parse_final_answer_v2("#### Unknown\n")
    check(is_correct(r2.label, "Unknown") is True,
          "v2-parsed label should compare correctly via the shared is_correct")
    check(is_correct(r2.label, "True") is False,
          "v2-parsed label should compare correctly via the shared is_correct")


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
