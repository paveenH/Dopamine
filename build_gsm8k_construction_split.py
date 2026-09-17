#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_gsm8k_construction_split.py -- builds the GSM8K "construction split":
the full GSM8K test set MINUS the existing frozen 300-question benchmark
(benchmark/gsm8k_test_sample.json), for use as a larger-N ARRSN construction
set. Standalone, read-only on all existing inputs, writes ONE new file.

WHY. The 300-question file is reserved for downstream steering/behavioral
evaluation (per explicit instruction) and must never leak into the set used
to CONSTRUCT the ARRSN direction -- otherwise a later steering evaluation on
those 300 questions would be testing a direction partly derived from the
same items it is evaluated on.

EXCLUSION METHOD. Exact `question` text match, NOT fuzzy matching. Verified
before use (this script asserts it, does not assume it) that all 1319
questions in the real openai/gsm8k main/test split are unique strings, so
exact-string exclusion cannot silently under- or over-exclude due to
duplicate questions on either side.

SOURCE OF THE FULL TEST SET: openai/gsm8k, config "main", split "test",
loaded via HuggingFace `datasets`, SAME dataset/config/split
data_gsm8k.py / data_gsm8k_sample.py already use for this repo's GSM8K work
(revision pinned via `--revision` for exact reproducibility; if omitted, the
resolved revision actually used is still recorded in the manifest).

FAIL-CLOSED CHECKS (all before any output is written):
  - the 300-question file must load to exactly 300 items
  - all 1319 (or whatever the actual full-test count is) full-test questions
    must be unique strings -- refuses otherwise, since exact-match exclusion
    would then be ambiguous
  - every one of the 300 questions must find EXACTLY ONE exact match in the
    full test set (refuses on zero matches -- the 300-file is not a subset
    of this full test set -- or NO on duplicate matches, which cannot
    happen given the uniqueness check above but is asserted anyway)
  - the resulting construction split must have EXACTLY
    len(full_test) - 300 items, with NO duplicate questions and NO overlap
    (verified by an explicit intersection check, not merely arithmetic) with
    the 300-question file
  - refuses to overwrite an existing output file

OUTPUT (repo-local, so it can be uploaded to the server like any other
benchmark file):
  benchmark/gsm8k_construction_split.json
    list of {"task": "gsm8k", "question": ..., "answer": ..., "solution": ...}
    dicts, ORDER = the full test set's own original index order with the
    300 excluded items removed (i.e. NOT re-shuffled, NOT re-sorted by any
    other key) -- this keeps the split fully reproducible from the dataset
    alone, given the exclusion list.
  benchmark/gsm8k_construction_split_manifest.json
    full provenance: full-test source dataset/config/split/revision, full
    count, excluded-300-file path+sha256+count, construction-split count,
    ordered-question-list sha256 (construction split), sha256 of the written
    JSON file itself.

Does NOT modify benchmark/gsm8k_test_sample.json or any other existing file.
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

SCRIPT_VERSION = "build_gsm8k_construction_split-v1"

EXPECTED_300_COUNT = 300
EXPECTED_FULL_TEST_N = 1319
EXPECTED_CONSTRUCTION_N = 1019


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
                          "<base_dir>/benchmark/gsm8k_test_sample.json, "
                          "writes <base_dir>/benchmark/"
                          "gsm8k_construction_split{,_manifest}.json")
    ap.add_argument("--revision", default=None,
                     help="Optional pinned HF dataset revision for "
                          "openai/gsm8k. If omitted, the default revision is "
                          "used and the resolved value is recorded in the "
                          "manifest from the dataset's own info.")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    bench_dir = os.path.join(args.base_dir, "benchmark")
    sample_300_path = os.path.join(bench_dir, "gsm8k_test_sample.json")
    out_split_path = os.path.join(bench_dir, "gsm8k_construction_split.json")
    out_manifest_path = os.path.join(
        bench_dir, "gsm8k_construction_split_manifest.json")

    if not os.path.isfile(sample_300_path):
        die(f"300-question file not found: {sample_300_path}")
    for p in (out_split_path, out_manifest_path):
        if os.path.exists(p):
            die(f"refusing to overwrite existing output: {p}")

    # ---- load the existing 300-question file ----
    samples_300 = utils.load_json(sample_300_path)
    if len(samples_300) != EXPECTED_300_COUNT:
        die(f"{sample_300_path}: loaded {len(samples_300)} items, expected "
            f"exactly {EXPECTED_300_COUNT}.")
    questions_300 = [s["question"] for s in samples_300]
    if len(set(questions_300)) != EXPECTED_300_COUNT:
        die(f"{sample_300_path}: contains duplicate questions "
            f"({len(set(questions_300))} unique of {EXPECTED_300_COUNT}) -- "
            "refusing to build an exclusion set from a file with internal "
            "duplicates.")
    questions_300_set = set(questions_300)
    sample_300_sha256 = sha256_of_file(sample_300_path)

    print(f"Loaded 300-question file: {sample_300_path} "
          f"(sha256={sample_300_sha256}, n={len(samples_300)})")

    # ---- load the full GSM8K test set ----
    load_kwargs = dict(path="openai/gsm8k", name="main", split="test",
                        trust_remote_code=True)
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.revision:
        load_kwargs["revision"] = args.revision
    ds = load_dataset(**load_kwargs)
    n_full = len(ds)
    print(f"Loaded full GSM8K test set: openai/gsm8k/main/test, n={n_full}")
    if n_full != EXPECTED_FULL_TEST_N:
        die(f"full GSM8K test set has {n_full} rows, expected exactly "
            f"{EXPECTED_FULL_TEST_N} -- the upstream dataset appears to have "
            "changed (different revision/version than previously verified). "
            "Refusing rather than silently building a construction split "
            "from an unexpected base. Pin --revision to the previously "
            "verified commit if you intend to reproduce the exact same "
            "construction split.")

    full_questions = [row["question"] for row in ds]
    if len(set(full_questions)) != n_full:
        die(f"full GSM8K test set contains duplicate questions "
            f"({len(set(full_questions))} unique of {n_full}) -- refusing "
            "to exclude by exact question-text match under this condition; "
            "exact-match exclusion requires question-text uniqueness.")

    # ---- verify the 300-question file is an EXACT subset of the full test
    # ---- set (every one of the 300 questions must appear, exactly once,
    # ---- in the full test set) ----
    full_questions_set = set(full_questions)
    not_found = [q for q in questions_300 if q not in full_questions_set]
    if not_found:
        die(f"{len(not_found)} of the 300 questions were NOT found (exact "
            "match) in the full GSM8K test set -- the 300-question file is "
            "not a subset of this full test set (dataset/config/split/"
            "revision mismatch?). First missing question: "
            f"{not_found[0]!r}")

    # ---- build the construction split: full test set minus the 300,
    # ---- preserving the full test set's own original order ----
    construction_rows = []
    for row in ds:
        q = row["question"]
        if q in questions_300_set:
            continue
        solution = row["answer"]
        construction_rows.append({
            "task": "gsm8k",
            "question": q,
            "answer": extract_answer(solution),
            "solution": solution,
        })

    expected_construction_n = n_full - EXPECTED_300_COUNT
    if len(construction_rows) != expected_construction_n:
        die(f"construction split has {len(construction_rows)} items, "
            f"expected exactly {n_full} - {EXPECTED_300_COUNT} = "
            f"{expected_construction_n}.")

    construction_questions = [r["question"] for r in construction_rows]
    if len(set(construction_questions)) != len(construction_questions):
        die("construction split contains duplicate questions -- should be "
            "unreachable given the full-test uniqueness check above; treat "
            "as a hard bug.")

    overlap = questions_300_set & set(construction_questions)
    if overlap:
        die(f"construction split OVERLAPS the excluded 300-question file on "
            f"{len(overlap)} question(s) -- should be unreachable; treat as "
            "a hard bug. First overlapping question: "
            f"{sorted(overlap)[0]!r}")

    construction_digest = sha256_of_list(construction_questions)
    print(f"Construction split: n={len(construction_rows)} "
          f"(expected {expected_construction_n}), "
          f"ordered_sample_identity_sha256={construction_digest}")

    # ---- write output ----
    atomic_write_json(out_split_path, construction_rows)
    out_split_sha256 = sha256_of_file(out_split_path)
    print(f"wrote {out_split_path} (sha256={out_split_sha256})")

    dataset_revision = None
    try:
        dataset_revision = getattr(ds, "info", None) and getattr(
            ds.info, "download_checksums", None)
    except Exception:
        dataset_revision = None

    manifest = {
        "script_version": SCRIPT_VERSION,
        "source_dataset": {
            "path": "openai/gsm8k", "name": "main", "split": "test",
            "requested_revision": args.revision,
            "note": "revision was NOT passed to load_dataset() unless "
                    "--revision was given; if omitted, this manifest "
                    "records only the requested value (None), not a "
                    "resolved commit hash -- rerun with --revision pinned "
                    "for byte-exact future reproducibility.",
        },
        "full_test_set_n": n_full,
        "excluded_300_file": {
            "path": sample_300_path,
            "sha256": sample_300_sha256,
            "n": EXPECTED_300_COUNT,
        },
        "exclusion_method": "exact question-text string match "
                             "(verified: all questions in both the "
                             "300-file and the full test set are unique "
                             "strings; no fuzzy matching)",
        "construction_split_n": len(construction_rows),
        "construction_split_n_expected": expected_construction_n,
        "construction_split_order": "original full-test-set index order, "
                                     "with the 300 excluded items removed "
                                     "(no reshuffle/resort)",
        "construction_split_ordered_sample_identity_sha256": construction_digest,
        "construction_split_output": {
            "path": out_split_path,
            "sha256": out_split_sha256,
        },
        "checks_passed": {
            "300_file_exact_count": True,
            "300_file_no_internal_duplicates": True,
            "full_test_set_no_duplicates": True,
            "all_300_found_exact_in_full_test": True,
            "construction_split_exact_expected_count": True,
            "construction_split_no_internal_duplicates": True,
            "construction_split_zero_overlap_with_300": True,
        },
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out_manifest_path, manifest)
    print(f"wrote {out_manifest_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
