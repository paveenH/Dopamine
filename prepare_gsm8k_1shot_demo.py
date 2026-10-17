#!/usr/bin/env python3
"""Fix the ONE worked demonstration for `gsm8k-base-1shot-v1`.

Rule (frozen before any cell exists, not tuned on model behaviour): the FIRST row of
official GSM8K `main/train`, in source order, whose question text does not appear in
the 300 test questions. Question and answer are kept verbatim (the official answer
already ends in `#### <number>`; nothing is rewritten or re-marked).

Run on the SERVER (downloads the dataset there; nothing to upload):
  python prepare_gsm8k_1shot_demo.py --base_dir /data1/paveen/Dopamine/components
  python prepare_gsm8k_1shot_demo.py --base_dir ... --check     # re-derive + compare

Writes <base_dir>/benchmark/gsm8k_1shot_demo_v1.json and REFUSES to overwrite it, so a
re-run can never silently swap the example. --check re-derives from the pinned
revision and dies on any difference.
"""
import argparse
import hashlib
import json
import os
import sys

DATASET, CONFIG, SPLIT = "openai/gsm8k", "main", "train"
OUT_NAME = "benchmark/gsm8k_1shot_demo_v1.json"
TEST_NAME = "benchmark/gsm8k_test_sample.json"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def content_hash(question, answer):
    return sha256_bytes(json.dumps({"question": question, "answer": answer},
                                   ensure_ascii=False, sort_keys=True).encode("utf-8"))


def norm(t):
    return " ".join(t.split())


def pick_first_non_overlapping(rows, test_questions):
    """rows: iterable of (question, answer) in source order. Returns (idx, q, a)."""
    banned = {norm(q) for q in test_questions}
    for i, (q, a) in enumerate(rows):
        if norm(q) not in banned:
            return i, q, a
    raise SystemExit("[FATAL] every train row overlaps the test set")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revision", default=None, help="pin a revision (needed by --check only if not in the file)")
    a = ap.parse_args()

    out_path = os.path.join(a.base_dir, OUT_NAME)
    test_path = os.path.join(a.base_dir, TEST_NAME)
    raw = open(test_path, "rb").read()
    test = json.loads(raw)
    if len(test) != 300:
        sys.exit(f"[FATAL] test file has {len(test)} rows, expected 300")
    test_q = [s["question"] for s in test]

    existing = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else None
    if existing is None and a.check:
        sys.exit(f"[FATAL] --check but {out_path} does not exist")
    if existing is not None and not a.check:
        sys.exit(f"[x] {out_path} already exists -- refusing to overwrite (use --check)")

    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = a.revision or (existing or {}).get("dataset_revision") \
        or HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, revision=revision)
    rows = [(r["question"], r["answer"]) for r in ds]
    idx, q, ans = pick_first_non_overlapping(rows, test_q)

    rec = {
        "protocol": "gsm8k-base-1shot-v1",
        "dataset": DATASET, "config": CONFIG, "split": SPLIT,
        "dataset_revision": revision, "train_row_index": idx, "n_train_rows": len(rows),
        "selection_rule": "first main/train row in source order whose question text is "
                          "not among the 300 test questions (whitespace-normalised)",
        "question": q, "answer": ans,
        "content_sha256": content_hash(q, ans),
        "test_file": TEST_NAME, "test_file_sha256": sha256_bytes(raw),
        "answer_has_final_marker": "####" in ans,
    }
    if not rec["answer_has_final_marker"]:
        sys.exit("[FATAL] official answer carries no '####' marker")
    if ans.count("####") != 1:
        sys.exit(f"[FATAL] demo answer has {ans.count('####')} '####' markers, expected 1")

    if a.check:
        keys = ["dataset_revision", "train_row_index", "question", "answer",
                "content_sha256", "test_file_sha256"]
        bad = [k for k in keys if rec[k] != existing.get(k)]
        if bad:
            sys.exit(f"[FATAL] --check mismatch on {bad}")
        print(f"[ok] demo re-derived identically: train row {idx}, sha {rec['content_sha256'][:16]}")
        return
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}\n  train row {idx}  rev {revision}  sha {rec['content_sha256'][:16]}")
    print("  question:", q[:100].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
