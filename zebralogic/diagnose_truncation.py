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


def diagnose_row(row: dict, tokenizer) -> dict:
    text = row.get("raw_text") or row.get("generated") or ""
    truncated = bool(row.get("truncated"))
    stop_reason = row.get("stop_reason")
    n_tok_generated = row.get("generated_token_count")

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n_tok_retokenized = len(offsets)

    first_obj, first_start, first_end = find_first_complete_json(text)
    has_solution = bool(
        first_obj is not None and isinstance(first_obj, dict)
        and "solution" in first_obj
    )

    first_solution_end_token = None
    trailing_tokens_after_solution = None
    solution_completed_before_limit = None
    if has_solution and first_end is not None:
        # First token whose span starts at or after the JSON's closing
        # brace char offset (first_end is already "one past the brace").
        end_tok = next(
            (i for i, (s, e) in enumerate(offsets) if s >= first_end),
            n_tok_retokenized,
        )
        first_solution_end_token = end_tok
        trailing_tokens_after_solution = max(0, n_tok_retokenized - end_tok)
        # "Before the limit" means the model still had budget left when the
        # solution JSON closed, OR it stopped naturally right after (either
        # way, the solution was not itself cut off by the cap). We read this
        # off the EXISTING truncated/stop_reason field, not by re-deriving
        # a length threshold here -- a not-truncated row by definition never
        # hit budget_exhausted, so any complete solution JSON in it was
        # necessarily completed before any limit. For a truncated row, the
        # solution is "completed before the limit" iff the JSON's closing
        # brace appears before the LAST generated token (i.e. the model kept
        # generating past it rather than being cut off mid-JSON).
        if not truncated:
            solution_completed_before_limit = True
        else:
            solution_completed_before_limit = trailing_tokens_after_solution > 0

    loop_span = _loop_tail_span(text)
    loop_before_solution = None
    loop_after_solution = None
    if loop_span is not None:
        loop_char_start = loop_span[0]
        if has_solution and first_end is not None:
            loop_before_solution = loop_char_start < first_end
            loop_after_solution = loop_char_start >= first_end
        else:
            # No complete solution JSON at all -- the loop cannot be
            # classified relative to a solution that was never written.
            loop_before_solution = None
            loop_after_solution = None

    return {
        "id": row.get("id"),
        "sample_id": row.get("sample_id"),
        "size": row.get("size"),
        "truncated": truncated,
        "stop_reason": stop_reason,
        "generated_token_count_stored": n_tok_generated,
        "generated_token_count_retokenized": n_tok_retokenized,
        "has_solution_json": has_solution,
        "solution_completed_before_limit": solution_completed_before_limit,
        "first_solution_end_char": first_end,
        "first_solution_end_token": first_solution_end_token,
        "trailing_tokens_after_solution": trailing_tokens_after_solution,
        "is_strict_loop": loop_span is not None,
        "loop_before_solution": loop_before_solution,
        "loop_after_solution": loop_after_solution,
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

    def _bucket(has_sol, trunc):
        return [x for x in diags if x["has_solution_json"] == has_sol and x["truncated"] == trunc]

    parsed_truncated = _bucket(True, True)
    unparsed_truncated = _bucket(False, True)
    parsed_not_truncated = _bucket(True, False)
    unparsed_not_truncated = _bucket(False, False)

    print(f"\n=== TRUNCATION DIAGNOSIS: {args.generations} ===")
    print(f"model={meta.get('model')} size={meta.get('size')} "
          f"max_new_tokens={max_new_tokens} n={n}")
    print(f"\ntruncated_rate (real stop_reason==budget_exhausted): "
          f"{n_truncated}/{n} = {n_truncated/n:.3f}")
    print(f"  parsed(has solution JSON)+truncated   : {len(parsed_truncated)}/{n}")
    print(f"  unparsed(no solution JSON)+truncated  : {len(unparsed_truncated)}/{n}")
    print(f"  parsed+NOT truncated                  : {len(parsed_not_truncated)}/{n}")
    print(f"  unparsed+NOT truncated                : {len(unparsed_not_truncated)}/{n}")

    defined = [x for x in diags if x["has_solution_json"]]
    completed_before_limit = [x for x in defined if x["solution_completed_before_limit"]]
    print(f"\nsolution JSON present at all           : {len(defined)}/{n}")
    if defined:
        print(f"  of those, completed before token limit "
              f"(i.e. not itself cut off): {len(completed_before_limit)}/{len(defined)} "
              f"= {len(completed_before_limit)/len(defined):.3f}")

    trailing = [x["trailing_tokens_after_solution"] for x in defined
                if x["trailing_tokens_after_solution"] is not None]
    if trailing:
        trailing_sorted = sorted(trailing)
        med = trailing_sorted[len(trailing_sorted) // 2]
        print(f"  trailing_tokens_after_solution: min={min(trailing)} "
              f"median={med} max={max(trailing)}")

    loop_rows = [x for x in diags if x["is_strict_loop"]]
    loop_before = sum(1 for x in loop_rows if x["loop_before_solution"])
    loop_after = sum(1 for x in loop_rows if x["loop_after_solution"])
    loop_undecidable = sum(
        1 for x in loop_rows
        if x["loop_before_solution"] is None and x["loop_after_solution"] is None
    )
    print(f"\nstrict-loop rows: {len(loop_rows)}/{n}")
    print(f"  loop begins BEFORE first solution JSON closes : {loop_before}")
    print(f"  loop begins AFTER  first solution JSON closes : {loop_after}")
    print(f"  loop present but no solution JSON to compare  : {loop_undecidable}")

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
                  f"trailing_tokens_after_solution={x['trailing_tokens_after_solution']} "
                  f"is_strict_loop={x['is_strict_loop']} "
                  f"(showing last {len(tail)} chars of raw_text)")
            print("  " + "-" * 70)
            for line in tail.splitlines():
                print(f"  | {line}")
            print("  " + "-" * 70)

    _show(parsed_truncated, "PARSED + TRUNCATED tails")
    _show(unparsed_truncated, "UNPARSED + TRUNCATED tails")

    out = {
        "generations": args.generations,
        "model": meta.get("model"), "size": meta.get("size"),
        "max_new_tokens": max_new_tokens, "n": n,
        "truncated_rate": n_truncated / n,
        "parsed_truncated_n": len(parsed_truncated),
        "unparsed_truncated_n": len(unparsed_truncated),
        "parsed_not_truncated_n": len(parsed_not_truncated),
        "unparsed_not_truncated_n": len(unparsed_not_truncated),
        "solution_present_n": len(defined),
        "solution_completed_before_limit_n": len(completed_before_limit),
        "solution_completed_before_limit_rate": (
            len(completed_before_limit) / len(defined) if defined else None
        ),
        "trailing_tokens_after_solution_med": (
            sorted(trailing)[len(trailing) // 2] if trailing else None
        ),
        "strict_loop_n": len(loop_rows),
        "loop_before_solution_n": loop_before,
        "loop_after_solution_n": loop_after,
        "loop_undecidable_n": loop_undecidable,
        "stop_reason_inconsistent_n": len(bad_reason),
        "per_row": diags,
    }
    out_path = os.path.splitext(args.generations)[0] + ".truncation_diag.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
