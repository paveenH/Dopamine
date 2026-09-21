#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_gsm8k_1319_full.py -- builds the FULL GSM8K test set (n=1319), in
ORIGINAL full-test order, as a single frozen local file for the GSM8K Role
hidden-state extraction lines (gsm8k_role, gsm8k_role_abstention). Standalone,
read-only on all existing inputs, writes ONE new file + manifest. Does NOT
modify benchmark/gsm8k_test_sample.json or benchmark/gsm8k_construction_split.json.

WHY A NEW FILE INSTEAD OF RE-CONCATENATING THE TWO EXISTING ONES AT RUNTIME.
The task requires "the same 1319 questions, the same order" across four
extraction cells, unfiltered by correctness/marker/abstention behavior. The
existing 300-question file (benchmark/gsm8k_test_sample.json) and the
construction split (benchmark/gsm8k_construction_split.json, n=1019, "full
test set minus the 300, in original full-test index order" per
build_gsm8k_construction_split.py) already partition the full test set, but
concatenating them naively (300 then 1019, or interleaved by some other rule)
would NOT reproduce the dataset's own original index order -- the 300-file's
internal order is whatever produced it (verified below, not assumed), and the
construction split's order is "original order with 300 holes removed", not
"the 300 followed by the rest". This script instead re-derives the canonical
order directly from the SAME source (openai/gsm8k/main/test) both of those
files were built from, and cross-checks (not merely trusts) that this
canonical-order set is byte-identical, as a SET, to the union of the two
existing files.

FAIL-CLOSED CHECKS (all before any output is written):
  - the 300-question file loads to exactly 300 items, all unique questions
  - the construction-split file loads to exactly 1019 items, all unique
    questions, zero overlap with the 300 (re-verified here, not assumed from
    the construction-split manifest alone)
  - 300 + 1019 == 1319 (arithmetic identity, checked explicitly)
  - the full openai/gsm8k/main/test split (freshly loaded) has EXACTLY 1319
    rows, all unique questions
  - the SET of {300 questions} union {1019 construction-split questions}
    equals EXACTLY the SET of all 1319 full-test questions (no extra, no
    missing) -- this is the cross-check that catches any drift between the
    two existing files and the live dataset
  - refuses to overwrite an existing output file

OUTPUT (repo-local, uploadable to the server like any other benchmark file):
  benchmark/gsm8k_1319_full.json
    list of {"task": "gsm8k", "question": ..., "answer": ..., "solution": ...,
    "in_300_sample": bool} dicts, in the FULL TEST SET's OWN ORIGINAL index
    order (index 0..1318 of openai/gsm8k/main/test, unmodified -- NOT
    re-shuffled, NOT sorted by "300 first"). "in_300_sample" is a
    provenance/audit field ONLY (needed downstream to compute the
    300-vs-remaining-1019 stability check the task requires) and is never
    used to filter, reorder, or select which questions are extracted --
    all 1319 rows are extracted unconditionally.
  benchmark/gsm8k_1319_full_manifest.json
    full provenance: source dataset/config/split/revision, per-input-file
    sha256+count, cross-check results, ordered-question-list sha256 (full
    1319), sha256 of the written JSON file itself.

Usage (server, same interpreter/args style as build_gsm8k_construction_split.py):
  python build_gsm8k_1319_full.py --base_dir /data1/paveen/Dopamine/components
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

from datasets import load_dataset

import utils

SCRIPT_VERSION = "build_gsm8k_1319_full-v1"

EXPECTED_300_COUNT = 300
EXPECTED_CONSTRUCTION_COUNT = 1019
EXPECTED_FULL_N = 1319


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_list(items: list) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def atomic_write_json(path: str, obj) -> None:
    if os.path.exists(path):
        die(f"refusing to overwrite existing output: {path}")
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def extract_answer(solution: str) -> str:
    import re
    m = re.search(r"####\s*(.+)", solution)
    if m:
        return m.group(1).strip().replace(",", "")
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_dir", required=True,
                     help="components/ dir. Reads "
                          "<base_dir>/benchmark/{gsm8k_test_sample.json,"
                          "gsm8k_construction_split.json}, writes "
                          "<base_dir>/benchmark/gsm8k_1319_full{,_manifest}.json")
    ap.add_argument("--revision", default=None,
                     help="Optional pinned HF dataset revision for "
                          "openai/gsm8k. If omitted, the default revision is "
                          "used.")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    bench_dir = os.path.join(args.base_dir, "benchmark")
    sample_300_path = os.path.join(bench_dir, "gsm8k_test_sample.json")
    construction_path = os.path.join(bench_dir, "gsm8k_construction_split.json")
    out_path = os.path.join(bench_dir, "gsm8k_1319_full.json")
    out_manifest_path = os.path.join(bench_dir, "gsm8k_1319_full_manifest.json")

    for p in (sample_300_path, construction_path):
        if not os.path.isfile(p):
            die(f"required input not found: {p}")
    for p in (out_path, out_manifest_path):
        if os.path.exists(p):
            die(f"refusing to overwrite existing output: {p}")

    # ---- load + validate the 300-question file ----
    samples_300 = utils.load_json(sample_300_path)
    if len(samples_300) != EXPECTED_300_COUNT:
        die(f"{sample_300_path}: loaded {len(samples_300)} items, expected "
            f"exactly {EXPECTED_300_COUNT}.")
    questions_300 = [s["question"] for s in samples_300]
    if len(set(questions_300)) != EXPECTED_300_COUNT:
        die(f"{sample_300_path}: contains duplicate questions.")
    questions_300_set = set(questions_300)
    sample_300_sha256 = sha256_of_file(sample_300_path)
    print(f"Loaded 300-question file: {sample_300_path} "
          f"(sha256={sample_300_sha256}, n={len(samples_300)})")

    # ---- load + validate the construction-split file ----
    samples_construction = utils.load_json(construction_path)
    if len(samples_construction) != EXPECTED_CONSTRUCTION_COUNT:
        die(f"{construction_path}: loaded {len(samples_construction)} items, "
            f"expected exactly {EXPECTED_CONSTRUCTION_COUNT}.")
    questions_construction = [s["question"] for s in samples_construction]
    if len(set(questions_construction)) != EXPECTED_CONSTRUCTION_COUNT:
        die(f"{construction_path}: contains duplicate questions.")
    questions_construction_set = set(questions_construction)
    construction_sha256 = sha256_of_file(construction_path)
    print(f"Loaded construction-split file: {construction_path} "
          f"(sha256={construction_sha256}, n={len(samples_construction)})")

    overlap = questions_300_set & questions_construction_set
    if overlap:
        die(f"the 300-question file and the construction split OVERLAP on "
            f"{len(overlap)} question(s) -- they must partition the full "
            f"test set with zero overlap. First overlapping question: "
            f"{sorted(overlap)[0]!r}")

    if EXPECTED_300_COUNT + EXPECTED_CONSTRUCTION_COUNT != EXPECTED_FULL_N:
        die("internal arithmetic error: EXPECTED_300_COUNT + "
            "EXPECTED_CONSTRUCTION_COUNT != EXPECTED_FULL_N -- fix the "
            "constants at the top of this script.")

    union_set = questions_300_set | questions_construction_set
    if len(union_set) != EXPECTED_FULL_N:
        die(f"union of the 300-file and the construction split has "
            f"{len(union_set)} unique questions, expected exactly "
            f"{EXPECTED_FULL_N}.")

    # ---- load the full GSM8K test set fresh, from the SAME source both
    # ---- existing files were built from, to get the canonical ORIGINAL
    # ---- index order ----
    load_kwargs = dict(path="openai/gsm8k", name="main", split="test",
                        trust_remote_code=True)
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.revision:
        load_kwargs["revision"] = args.revision
    ds = load_dataset(**load_kwargs)
    n_full = len(ds)
    print(f"Loaded full GSM8K test set: openai/gsm8k/main/test, n={n_full}")
    if n_full != EXPECTED_FULL_N:
        die(f"full GSM8K test set has {n_full} rows, expected exactly "
            f"{EXPECTED_FULL_N} -- the upstream dataset appears to have "
            "changed. Refusing rather than silently building a 1319-file "
            "from an unexpected base. Pin --revision to the previously "
            "verified commit if you intend to reproduce byte-exactly.")

    full_questions = [row["question"] for row in ds]
    if len(set(full_questions)) != n_full:
        die(f"full GSM8K test set contains duplicate questions "
            f"({len(set(full_questions))} unique of {n_full}) -- refusing "
            "to proceed under this condition.")
    full_questions_set = set(full_questions)

    # ---- CROSS-CHECK: the union of the two existing files must equal
    # ---- EXACTLY the set of all live full-test questions -- no extra,
    # ---- no missing. This is the check that would catch drift between
    # ---- the frozen 300/1019 files and the live dataset. ----
    missing_from_live = union_set - full_questions_set
    extra_in_live = full_questions_set - union_set
    if missing_from_live:
        die(f"{len(missing_from_live)} question(s) present in the "
            "300-file/construction-split union but NOT found in the live "
            "full test set. First: "
            f"{sorted(missing_from_live)[0]!r}")
    if extra_in_live:
        die(f"{len(extra_in_live)} question(s) present in the live full "
            "test set but NOT covered by the 300-file/construction-split "
            "union -- the union does not actually partition the full test "
            "set. First: "
            f"{sorted(extra_in_live)[0]!r}")

    print("[cross-check] {300-file} union {construction-split} == "
          "{live full test set}, exactly (1319 questions, no extra, no "
          "missing).")

    # ---- build the output: full test set's OWN original order, unfiltered,
    # ---- annotated (not filtered) with membership in the 300-file ----
    rows = []
    for row in ds:
        q = row["question"]
        solution = row["answer"]
        rows.append({
            "task": "gsm8k",
            "question": q,
            "answer": extract_answer(solution),
            "solution": solution,
            "in_300_sample": q in questions_300_set,
        })

    if len(rows) != EXPECTED_FULL_N:
        die(f"assembled {len(rows)} rows, expected exactly {EXPECTED_FULL_N}.")
    n_in_300 = sum(r["in_300_sample"] for r in rows)
    n_not_in_300 = len(rows) - n_in_300
    if n_in_300 != EXPECTED_300_COUNT:
        die(f"assembled output has {n_in_300} rows flagged in_300_sample, "
            f"expected exactly {EXPECTED_300_COUNT}.")
    if n_not_in_300 != EXPECTED_CONSTRUCTION_COUNT:
        die(f"assembled output has {n_not_in_300} rows NOT flagged "
            f"in_300_sample, expected exactly {EXPECTED_CONSTRUCTION_COUNT}.")

    full_questions_out = [r["question"] for r in rows]
    if len(set(full_questions_out)) != EXPECTED_FULL_N:
        die("assembled output contains duplicate questions -- should be "
            "unreachable given the live-dataset uniqueness check above; "
            "treat as a hard bug.")
    full_digest = sha256_of_list(full_questions_out)
    print(f"Assembled full 1319-question file: n={len(rows)} "
          f"(in_300_sample={n_in_300}, remaining={n_not_in_300}), "
          f"ordered_sample_identity_sha256={full_digest}")

    atomic_write_json(out_path, rows)
    out_sha256 = sha256_of_file(out_path)
    print(f"wrote {out_path} (sha256={out_sha256})")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "source_dataset": {
            "path": "openai/gsm8k", "name": "main", "split": "test",
            "requested_revision": args.revision,
            "note": "requested_revision was passed to load_dataset() as "
                    "'revision' when --revision was given (None means the "
                    "HF default revision was used).",
        },
        "full_test_set_n": n_full,
        "input_300_file": {
            "path": sample_300_path, "sha256": sample_300_sha256,
            "n": EXPECTED_300_COUNT,
        },
        "input_construction_split_file": {
            "path": construction_path, "sha256": construction_sha256,
            "n": EXPECTED_CONSTRUCTION_COUNT,
        },
        "n_in_300_sample": n_in_300,
        "n_remaining_construction": n_not_in_300,
        "n_total": len(rows),
        "order": "the full test set's OWN original index order (0..1318), "
                 "unfiltered, unsorted, unshuffled -- 'in_300_sample' is a "
                 "provenance/audit annotation only and was never used to "
                 "reorder or filter",
        "ordered_sample_identity_sha256": full_digest,
        "output": {"path": out_path, "sha256": out_sha256},
        "checks_passed": {
            "300_file_exact_count_no_duplicates": True,
            "construction_split_exact_count_no_duplicates": True,
            "zero_overlap_between_300_and_construction": True,
            "arithmetic_300_plus_1019_eq_1319": True,
            "live_full_test_set_exact_1319_no_duplicates": True,
            "union_of_existing_files_equals_live_full_test_set_exactly": True,
            "assembled_output_exact_1319_no_duplicates": True,
            "assembled_output_in_300_count_matches_input": True,
        },
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out_manifest_path, manifest)
    print(f"wrote {out_manifest_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
