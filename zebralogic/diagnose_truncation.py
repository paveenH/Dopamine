#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZebraLogic-Easy 2048-token truncation diagnosis. READ-ONLY, NON-SCORING.

This script does NOT change, call, or import anything that affects Puzzle
Accuracy, Cell Accuracy, or JSON parsing. It reuses commitment_metrics.py's
`find_first_complete_json` (first complete JSON block, ANY content) AND
`find_first_answer_json` (first complete JSON carrying a non-empty dict
"solution" -- zebralogic-easy-amend-01) purely to locate character spans,
and reuses the already-generated, on-disk `truncated` / `stop_reason` field
from get_answer_zebralogic.py's output (itself sourced from llms.py's real
per-row terminator scan, not a padding/length heuristic -- see
llms.py:1046-1074) rather than recomputing it.

BOTH JSON notions are reported, deliberately, as two separate field sets
(`first_json_*` vs `first_answer_*`): a model can emit some OTHER complete
JSON object (e.g. a partial restatement) before the one that actually
carries "solution", in which case the two disagree. Reporting only
`find_first_complete_json`-based fields would then describe "where the
first JSON closed" while silently mislabeling it as "where the answer
closed" whenever the two differ -- this is exactly the amend-01 fix applied
to score_one_item_first and compute_commitment_metrics, and this diagnostic
script needs the same distinction for its own truncation questions (was a
graded ANSWER present before the limit, not merely was SOME JSON present).

It answers five questions about a preflight cell (originally written for the
2048-token cells; equally applicable to any budget), per CLAUDE.md's
instruction not to touch prompt/parser/scorer while diagnosing truncation.
Each question is answered on the ANSWER-JSON fields (`first_answer_*` /
`has_answer_json` / `answer_completed_before_limit`), which are what matters
for "was the puzzle actually answered before the cap" -- the parallel
GENERIC-JSON fields (`first_json_*` / `has_first_json`) are also reported so
a divergence between the two (some other JSON closed first, then the real
answer) is visible rather than silently absorbed into one number:

  1. parsed+truncated vs unparsed+truncated counts, where "parsed" means
     "has an ANSWER json" (does truncation always coincide with a failure to
     produce an answer, or does the model sometimes get cut off AFTER a
     usable answer was already written)
  2. was a complete answer JSON present before the token budget was hit
     (answer_completed_before_limit)
  3. where (in tokens) did the first complete answer JSON end
     (first_answer_end_token)
  4. how many tokens were generated AFTER that point
     (trailing_tokens_after_answer)
  5. does the strict-loop pattern (project-standard tail-recurrence
     detector) occur before or after the first answer JSON
     (loop_before_answer / loop_after_answer)

Token positions are computed by re-tokenizing the row's own `raw_text` (the
literal generated text as stored, special tokens included) with the SAME
tokenizer the model used, and mapping the first-JSON character span to a
token index via `offset_mapping`. This requires the tokenizer's fast/rust
backend (offset_mapping support); no model weights are loaded, no GPU is
needed, and no generation happens here.

Usage:
  python3.10 zebralogic/diagnose_truncation.py \
      --generations <path/to/mdf_0/zebralogic_easy_*.json> \
      --model_dir meta-llama/Llama-3.1-8B-Instruct \
      [--show_n 3]

@author: paveenhuang
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from commitment_metrics import (  # noqa: E402
    find_first_complete_json, find_first_answer_json, is_strict_loop,
)

_STRICT_LOOP_BLOCK = 40
_STRICT_LOOP_MIN_REPEATS = 4


def _loop_tail_span(text: str):
    """If is_strict_loop(text) is True, return the character span
    [start, end) of the FIRST occurrence of the recurring tail block --
    i.e. where the loop pattern first begins to repeat. Returns None if no
    strict loop is present. Read-only positional variant of
    commitment_metrics.is_strict_loop, which only returns a boolean."""
    if not is_strict_loop(text):
        return None
    tail = text[-_STRICT_LOOP_BLOCK:]
    first = text.find(tail)
    return (first, first + _STRICT_LOOP_BLOCK)


def _span_diagnostics(found, offsets, n_tok_retokenized, truncated, loop_span):
    """Shared per-span diagnostics for one located JSON span (either the
    generic-first-complete-JSON span or the first-ANSWER-JSON span). `found`
    is (obj, start, end) as returned by find_first_complete_json /
    find_first_answer_json. Returns (has, end_char, end_token,
    trailing_tokens, completed_before_limit, loop_before, loop_after) -- kept
    as one function so the two callers (generic vs answer) cannot silently
    diverge in how "completed before the limit" or "loop before/after" is
    computed."""
    obj, start, end = found
    has = obj is not None and isinstance(obj, dict)

    end_token = None
    trailing_tokens = None
    completed_before_limit = None
    if has and end is not None:
        # First token whose span starts at or after the JSON's closing brace
        # char offset (`end` is already "one past the brace").
        end_token = next(
            (i for i, (s, e) in enumerate(offsets) if s >= end),
            n_tok_retokenized,
        )
        trailing_tokens = max(0, n_tok_retokenized - end_token)
        # "Before the limit" means the model still had budget left when this
        # JSON closed, OR it stopped naturally right after (either way, this
        # JSON was not itself cut off by the cap). Read off the EXISTING
        # truncated/stop_reason field, not a re-derived length threshold --
        # a not-truncated row by definition never hit budget_exhausted, so
        # any complete JSON in it was necessarily completed before any
        # limit. For a truncated row, it is "completed before the limit" iff
        # its closing brace appears before the LAST generated token (i.e.
        # the model kept generating past it rather than being cut off
        # mid-JSON).
        completed_before_limit = True if not truncated else trailing_tokens > 0

    loop_before = None
    loop_after = None
    if loop_span is not None:
        loop_char_start = loop_span[0]
        if has and end is not None:
            loop_before = loop_char_start < end
            loop_after = loop_char_start >= end
        # else: no complete JSON of this kind at all -- the loop cannot be
        # classified relative to a JSON that was never written; leave None.

    return has, end, end_token, trailing_tokens, completed_before_limit, loop_before, loop_after


def diagnose_row(row: dict, tokenizer) -> dict:
    text = row.get("raw_text") or row.get("generated") or ""
    truncated = bool(row.get("truncated"))
    stop_reason = row.get("stop_reason")
    n_tok_generated = row.get("generated_token_count")

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n_tok_retokenized = len(offsets)

    loop_span = _loop_tail_span(text)

    # GENERIC: first syntactically complete JSON block, any content. Answers
    # "where did the first JSON close", regardless of whether it is an
    # answer -- this is what a raw truncation/loop diagnosis needs when the
    # question is about JSON-writing behavior in general.
    (has_first_json, first_json_end_char, first_json_end_token,
     trailing_after_json, json_completed_before_limit,
     loop_before_json, loop_after_json) = _span_diagnostics(
        find_first_complete_json(text), offsets, n_tok_retokenized,
        truncated, loop_span)

    # ANSWER: first complete JSON carrying a non-empty dict "solution"
    # (zebralogic-easy-amend-01's find_first_answer_json). Answers "was the
    # puzzle actually answered before the limit" -- the question this
    # diagnostic exists to answer, and DIFFERENT from has_first_json
    # whenever the model emits some other JSON (e.g. a partial restatement)
    # before its real answer.
    (has_answer_json, first_answer_end_char, first_answer_end_token,
     trailing_after_answer, answer_completed_before_limit,
     loop_before_answer, loop_after_answer) = _span_diagnostics(
        find_first_answer_json(text), offsets, n_tok_retokenized,
        truncated, loop_span)

    return {
        "id": row.get("id"),
        "sample_id": row.get("sample_id"),
        "size": row.get("size"),
        "truncated": truncated,
        "stop_reason": stop_reason,
        "generated_token_count_stored": n_tok_generated,
        "generated_token_count_retokenized": n_tok_retokenized,
        # GENERIC (any complete JSON, regardless of content)
        "has_first_json": has_first_json,
        "first_json_end_char": first_json_end_char,
        "first_json_end_token": first_json_end_token,
        "trailing_tokens_after_json": trailing_after_json,
        "json_completed_before_limit": json_completed_before_limit,
        "loop_before_json": loop_before_json,
        "loop_after_json": loop_after_json,
        # ANSWER (first JSON carrying a non-empty dict "solution")
        "has_answer_json": has_answer_json,
        "first_answer_end_char": first_answer_end_char,
        "first_answer_end_token": first_answer_end_token,
        "trailing_tokens_after_answer": trailing_after_answer,
        "answer_completed_before_limit": answer_completed_before_limit,
        "loop_before_answer": loop_before_answer,
        "loop_after_answer": loop_after_answer,
        "is_strict_loop": loop_span is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--model_dir", required=True,
                     help="HF repo id or local path; tokenizer only, no "
                          "model weights are loaded.")
    ap.add_argument("--show_n", type=int, default=3,
                     help="How many representative sample tails to print "
                          "verbatim per bucket (parsed+truncated, "
                          "unparsed+truncated).")
    ap.add_argument("--tail_chars", type=int, default=600)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if not tokenizer.is_fast:
        sys.exit(f"FAIL: {args.model_dir}'s tokenizer is not a fast "
                 "tokenizer; offset_mapping is unavailable, so token "
                 "positions cannot be computed for this diagnostic.")

    d = json.load(open(args.generations, encoding="utf-8"))
    meta, rows = d["meta"], d["data"]
    max_new_tokens = meta.get("max_new_tokens")

    diags = [diagnose_row(r, tokenizer) for r in rows]
    n = len(diags)

    n_truncated = sum(1 for x in diags if x["truncated"])
    n_not_truncated = n - n_truncated

    def _bucket(field, has, trunc):
        return [x for x in diags if x[field] == has and x["truncated"] == trunc]

    # PRIMARY bucketing is on has_answer_json -- "was the puzzle answered",
    # which is what this diagnostic exists to establish. has_first_json
    # buckets are also printed so a divergence (some other JSON closed
    # first, before the real answer) is visible.
    ans_parsed_truncated = _bucket("has_answer_json", True, True)
    ans_unparsed_truncated = _bucket("has_answer_json", False, True)
    ans_parsed_not_truncated = _bucket("has_answer_json", True, False)
    ans_unparsed_not_truncated = _bucket("has_answer_json", False, False)

    json_parsed_truncated = _bucket("has_first_json", True, True)
    json_unparsed_truncated = _bucket("has_first_json", False, True)

    print(f"\n=== TRUNCATION DIAGNOSIS: {args.generations} ===")
    print(f"model={meta.get('model')} size={meta.get('size')} "
          f"max_new_tokens={max_new_tokens} n={n}")
    print(f"\ntruncated_rate (real stop_reason==budget_exhausted): "
          f"{n_truncated}/{n} = {n_truncated/n:.3f}")
    print(f"[ANSWER JSON: first complete JSON carrying a non-empty solution]")
    print(f"  answered+truncated     : {len(ans_parsed_truncated)}/{n}")
    print(f"  unanswered+truncated   : {len(ans_unparsed_truncated)}/{n}")
    print(f"  answered+NOT truncated : {len(ans_parsed_not_truncated)}/{n}")
    print(f"  unanswered+NOT truncated: {len(ans_unparsed_not_truncated)}/{n}")
    print(f"[GENERIC JSON: first syntactically complete JSON, any content]")
    print(f"  has-json+truncated     : {len(json_parsed_truncated)}/{n}")
    print(f"  no-json+truncated      : {len(json_unparsed_truncated)}/{n}")
    n_divergent = sum(1 for x in diags if x["has_first_json"] != x["has_answer_json"])
    print(f"  rows where first-JSON and first-ANSWER-JSON disagree (some "
          f"other JSON closed before the real answer, or vice versa): "
          f"{n_divergent}/{n}")

    answered = [x for x in diags if x["has_answer_json"]]
    ans_completed_before_limit = [x for x in answered if x["answer_completed_before_limit"]]
    print(f"\nanswer JSON present at all              : {len(answered)}/{n}")
    if answered:
        print(f"  of those, completed before token limit "
              f"(i.e. not itself cut off): {len(ans_completed_before_limit)}/{len(answered)} "
              f"= {len(ans_completed_before_limit)/len(answered):.3f}")

    trailing = [x["trailing_tokens_after_answer"] for x in answered
                if x["trailing_tokens_after_answer"] is not None]
    if trailing:
        trailing_sorted = sorted(trailing)
        med = trailing_sorted[len(trailing_sorted) // 2]
        print(f"  trailing_tokens_after_answer: min={min(trailing)} "
              f"median={med} max={max(trailing)}")

    loop_rows = [x for x in diags if x["is_strict_loop"]]
    loop_before = sum(1 for x in loop_rows if x["loop_before_answer"])
    loop_after = sum(1 for x in loop_rows if x["loop_after_answer"])
    loop_undecidable = sum(
        1 for x in loop_rows
        if x["loop_before_answer"] is None and x["loop_after_answer"] is None
    )
    print(f"\nstrict-loop rows: {len(loop_rows)}/{n}")
    print(f"  loop begins BEFORE first ANSWER JSON closes : {loop_before}")
    print(f"  loop begins AFTER  first ANSWER JSON closes : {loop_after}")
    print(f"  loop present but no answer JSON to compare  : {loop_undecidable}")

    # Truncation-source sanity: confirm every truncated row's stop_reason is
    # genuinely budget_exhausted (not e.g. a missing field silently coerced
    # to False/True) and that a not-truncated row never reports
    # budget_exhausted -- catches a metadata/field-name drift, not a real
    # generation bug.
    bad_reason = [x for x in diags
                  if (x["truncated"] and x["stop_reason"] != "budget_exhausted")
                  or (not x["truncated"] and x["stop_reason"] == "budget_exhausted")]
    if bad_reason:
        print(f"\n[WARN] {len(bad_reason)} row(s) have truncated flag "
              "inconsistent with stop_reason -- inspect before trusting "
              "the truncated_rate above:")
        for x in bad_reason[:5]:
            print(f"   id={x['id']} truncated={x['truncated']} "
                  f"stop_reason={x['stop_reason']!r}")
    else:
        print("\n[OK] every row's truncated flag agrees with its own "
              "stop_reason (budget_exhausted iff truncated) -- truncation "
              "is a real generation-time signal here, not a padding or "
              "length-computation artifact.")

    # Representative tails, verbatim.
    def _show(bucket, label):
        if not bucket:
            print(f"\n--- {label}: none ---")
            return
        print(f"\n--- {label}: showing up to {args.show_n} of {len(bucket)} ---")
        for x in bucket[: args.show_n]:
            row = next(r for r in rows if r["id"] == x["id"])
            text = row.get("raw_text") or row.get("generated") or ""
            tail = text[-args.tail_chars:]
            print(f"\n  id={x['id']} size={x['size']} "
                  f"trailing_tokens_after_answer={x['trailing_tokens_after_answer']} "
                  f"is_strict_loop={x['is_strict_loop']} "
                  f"(showing last {len(tail)} chars of raw_text)")
            print("  " + "-" * 70)
            for line in tail.splitlines():
                print(f"  | {line}")
            print("  " + "-" * 70)

    _show(ans_parsed_truncated, "ANSWERED + TRUNCATED tails")
    _show(ans_unparsed_truncated, "UNANSWERED + TRUNCATED tails")

    out = {
        "generations": args.generations,
        "model": meta.get("model"), "size": meta.get("size"),
        "max_new_tokens": max_new_tokens, "n": n,
        "truncated_rate": n_truncated / n,
        # ANSWER JSON (primary): first complete JSON carrying a non-empty
        # dict "solution" (zebralogic-easy-amend-01's find_first_answer_json)
        "answered_truncated_n": len(ans_parsed_truncated),
        "unanswered_truncated_n": len(ans_unparsed_truncated),
        "answered_not_truncated_n": len(ans_parsed_not_truncated),
        "unanswered_not_truncated_n": len(ans_unparsed_not_truncated),
        "answer_present_n": len(answered),
        "answer_completed_before_limit_n": len(ans_completed_before_limit),
        "answer_completed_before_limit_rate": (
            len(ans_completed_before_limit) / len(answered) if answered else None
        ),
        "trailing_tokens_after_answer_med": (
            sorted(trailing)[len(trailing) // 2] if trailing else None
        ),
        "loop_before_answer_n": loop_before,
        "loop_after_answer_n": loop_after,
        "loop_undecidable_n": loop_undecidable,
        # GENERIC JSON: first syntactically complete JSON, any content
        "has_first_json_truncated_n": len(json_parsed_truncated),
        "no_first_json_truncated_n": len(json_unparsed_truncated),
        "first_json_vs_answer_json_divergent_n": n_divergent,
        "strict_loop_n": len(loop_rows),
        "stop_reason_inconsistent_n": len(bad_reason),
        "per_row": diags,
    }
    out_path = os.path.splitext(args.generations)[0] + ".truncation_diag.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
