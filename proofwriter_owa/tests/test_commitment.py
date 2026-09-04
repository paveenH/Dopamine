#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for proofwriter_owa/commitment.py. No GPU, no network, stdlib only.

    python3 test_commitment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commitment import (per_sample_commitment, aggregate_commitment,
                        first_strict_marker_start, DETECTOR_VERSION)

FAILS = []
N = 0


def check(cond, msg):
    global N
    N += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


def test_answer_first():
    text = "Answer: True\n"
    c = per_sample_commitment(text)
    check(c["answer_first"] is True, f"expected answer_first True, got {c}")
    check(c["pre_answer_reasoning_chars"] == 0,
          f"expected 0 pre-answer chars, got {c['pre_answer_reasoning_chars']}")


def test_reason_before_answer():
    text = "Fact1 and Fact2 imply Query.\nAnswer: True\n"
    c = per_sample_commitment(text)
    check(c["answer_first"] is False, f"expected answer_first False, got {c}")
    check(c["reason_before_answer"] is True, f"got {c}")
    check(c["pre_answer_reasoning_chars"] == len("Fact1 and Fact2 imply Query.\n"),
          f"got {c['pre_answer_reasoning_chars']}")


def test_no_marker_fields_are_none():
    text = "I am not sure how to answer this."
    c = per_sample_commitment(text)
    check(c["has_marker"] is False, "no marker -> has_marker False")
    check(c["answer_first"] is False, "no marker -> answer_first False (no strict marker)")
    check(c["first_answer_marker_pos"] is None, "no marker -> position None")
    check(c["pre_answer_reasoning_chars"] is None, "no marker -> chars None")
    check(c["reason_before_answer"] is None,
          "reason_before_answer must be None (undefined), not False, when there is no marker")


def test_label_revision_detected():
    text = "Answer: True\nWait, let me reconsider.\nAnswer: False\n"
    c = per_sample_commitment(text)
    check(c["n_strict_markers"] == 2, f"got {c['n_strict_markers']}")
    check(c["first_final_label_agreement"] is False,
          "first (True) != last (False) should be detected as disagreement")


def test_label_no_revision_when_consistent():
    text = "Answer: True\nTo confirm: Answer: True\n"
    c = per_sample_commitment(text)
    check(c["first_final_label_agreement"] is True,
          "identical first/last labels should agree")


def test_single_marker_agreement_is_trivially_true():
    text = "Answer: Unknown\n"
    c = per_sample_commitment(text)
    check(c["first_final_label_agreement"] is True,
          "a single marker trivially agrees with itself")


def test_multiple_answer_marker_flag_uses_loose_detector():
    # A loose (non-strict) second occurrence should still flip
    # multiple_answer_marker, even if it never becomes the scored answer.
    text = "Answer: True\nSo the Answer: True is confirmed once more inline."
    c = per_sample_commitment(text)
    check(c["multiple_answer_marker"] is True,
          f"expected multiple_answer_marker True, got {c}")


def test_precomputed_tokens_takes_precedence():
    text = "reasoning here\nAnswer: True\n"

    class FakeTok:
        def __call__(self, s, add_special_tokens=False):
            # deliberately wrong count, to prove precomputed_tokens wins
            return {"input_ids": [0] * 999}

    c_tok = per_sample_commitment(text, tokenizer=FakeTok())
    check(c_tok["pre_answer_reasoning_tokens"] == 999,
          f"tokenizer path should report 999, got {c_tok['pre_answer_reasoning_tokens']}")

    c_pre = per_sample_commitment(text, tokenizer=FakeTok(), precomputed_tokens=7)
    check(c_pre["pre_answer_reasoning_tokens"] == 7,
          f"precomputed_tokens must take precedence over tokenizer, got "
          f"{c_pre['pre_answer_reasoning_tokens']}")


def test_no_tokenizer_no_precomputed_is_none():
    text = "reasoning here\nAnswer: True\n"
    c = per_sample_commitment(text)
    check(c["pre_answer_reasoning_tokens"] is None,
          "with neither tokenizer nor precomputed_tokens, must be None -- "
          "never estimated by a chars/4 heuristic")


def test_first_strict_marker_start_matches_answer_first_logic():
    text = "Answer: True\n"
    pos = first_strict_marker_start(text)
    check(pos == 0, f"expected offset 0, got {pos}")
    text2 = "abc\nAnswer: False\n"
    pos2 = first_strict_marker_start(text2)
    check(pos2 == 4, f"expected offset 4, got {pos2}")
    text3 = "no marker here"
    check(first_strict_marker_start(text3) is None, "no marker -> None")


def test_aggregate_denominators():
    rows = [
        per_sample_commitment("Answer: True\n"),                       # answer_first
        per_sample_commitment("reasoning\nAnswer: False\n"),           # reason_before
        per_sample_commitment("no marker at all here"),                # no marker
        per_sample_commitment("Answer: True\nAnswer: False\n"),        # revision
    ]
    agg = aggregate_commitment(rows)
    check(agg["detector_version"] == DETECTOR_VERSION, "version tag present")
    check(agg["n_samples"] == 4, f"got {agg['n_samples']}")
    check(agg["n_with_marker"] == 3, f"got {agg['n_with_marker']}")
    # answer_first_rate denominator is samples WITH a marker (3), not all 4
    check(agg["answer_first_rate"]["n"] == 3,
          f"answer_first_rate denominator should be 3, got {agg['answer_first_rate']['n']}")
    check(abs(agg["answer_first_rate"]["value"] - (2 / 3)) < 1e-9,
          f"2 of 3 marker-bearing samples are answer_first, got "
          f"{agg['answer_first_rate']['value']}")
    # label_revision_rate denominator is samples with >=2 strict markers (1)
    check(agg["label_revision_rate"]["n"] == 1,
          f"expected 1 revisable sample, got {agg['label_revision_rate']['n']}")
    check(agg["label_revision_rate"]["value"] == 1.0,
          f"the one revisable sample DID revise, expected rate 1.0, got "
          f"{agg['label_revision_rate']['value']}")
    # multiple_answer_marker_rate denominator is ALL samples (4)
    check(agg["multiple_answer_marker_rate"]["n"] == 4,
          f"expected denominator 4, got {agg['multiple_answer_marker_rate']['n']}")


def test_aggregate_empty_rows_does_not_crash():
    agg = aggregate_commitment([])
    check(agg["n_samples"] == 0, "empty input -> n_samples 0")
    check(agg["answer_first_rate"]["value"] is None, "empty input -> rate None, not ZeroDivisionError")


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
