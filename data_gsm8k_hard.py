#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and format GSM-Hard for the P3 BLIND cross-task validation.

Protocol: p3-v1 (`docs/PREREG_P3.md`, frozen 2026-08-28, tag `p3-prereg-v1`).

THE GOLD ANSWERS ARE WRITTEN TO A SEPARATE, SEALED FILE.
This is the whole point of the loader. P3 is a blind validation, so the
generation path must be structurally unable to reach a label:

  <out>_questions.json   question + sample_id ONLY   -> generation reads THIS
  <out>_gold.SEALED.json gold answers + sha256       -> evaluator reads this,
                                                        and ONLY after the
                                                        prediction file exists

Splitting the files is what makes the firewall structural rather than a
convention. A single file with a `answer` field would leave the label one
attribute access away from every generation script, and the P2 experience is
that "unused" is a much weaker guarantee than "unreachable".

SAMPLE SELECTION IS BY FROZEN HASH, NEVER DATASET ORDER (protocol 2.3).
Taking the first 300 rows would make the sample an artifact of the dataset's
storage order and would not survive a reordering upstream. Dedup on exact
question text happens BEFORE selection so one question cannot take two slots.

DATASET REVISION IS PINNED. `--revision` is recorded in both outputs, so a
later upstream edit cannot silently change what "the P3 sample" means.

This script downloads data. It does NOT run a model and computes NO accuracy.

@author: paveenhuang
"""

import argparse
import hashlib
import json
import os
import sys
from decimal import Decimal, InvalidOperation

SALT = "rsn-p3-sample-v1"          # frozen in protocol 2.3
N_SAMPLE = 300
LIMIT_2_53 = 2 ** 53


def exceeds_2_53(gold) -> bool:
    """EXACT magnitude test. Deliberately never calls float().

    Auditing a float64 precision failure BY CALLING float() is self-defeating:
    float('9007199254740993') == 2**53, so the very first value that matters is
    silently reported as within range. Verified: the float form misses 2^53+1
    in every surface form (bare, negative, '.0' suffix, scientific notation).
    Decimal parses the literal exactly.
    """
    try:
        return abs(Decimal(str(gold).strip().replace(",", ""))) > LIMIT_2_53
    except (InvalidOperation, ValueError):
        return False          # non-numeric: not a large-integer hazard


def digest(q: str) -> str:
    return hashlib.sha256(f"{SALT}:{q}".encode("utf-8")).hexdigest()


def select(questions):
    """Dedup on exact text, then take the N smallest digests."""
    uniq, seen = [], set()
    for q in questions:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    if len(uniq) < N_SAMPLE:
        sys.exit(f"FAIL: {len(uniq)} unique questions, need {N_SAMPLE}")
    return sorted(uniq, key=digest)[:N_SAMPLE], len(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", default="reasoning-machines/gsm-hard")
    ap.add_argument("--revision", required=True,
                    help=("REQUIRED. Pin the dataset revision to a commit SHA. "
                          "Without it `datasets` silently follows main, so a "
                          "later upstream edit would change what 'the P3 sample' "
                          "means with no trace. Confirm HEAD before use."))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="benchmark")
    ap.add_argument("--prefix", default="gsm8k_hard_p3")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(a.hf_name, split=a.split, revision=a.revision)
    print(f"loaded {a.hf_name} split={a.split} revision={a.revision} n={len(ds)}")
    print(f"columns: {ds.column_names}")

    # GSM-Hard's fields are `input` (question) and `target` (gold, often a float).
    qcol = "input" if "input" in ds.column_names else "question"
    gcol = "target" if "target" in ds.column_names else "answer"
    rows = [{"q": r[qcol], "g": r[gcol]} for r in ds]

    chosen, n_uniq = select([r["q"] for r in rows])
    by_q = {}
    for r in rows:                       # first occurrence wins, matching dedup
        by_q.setdefault(r["q"], r["g"])

    qdigest = hashlib.sha256("\n".join(chosen).encode("utf-8")).hexdigest()

    # ---- gold audit in memory (fix 3); the sealed file is written below.
    gold = [{"sample_id": i, "gold": by_q[q]} for i, q in enumerate(chosen)]
    # Fix 3: the audit runs HERE, in memory, while gold is legitimately in
    # scope. Only the COUNT and an audit digest reach metadata, so the sealed
    # file never has to be reopened for a second audit -- reopening it is the
    # one irreversible mistake in P3.
    big = [str(g["gold"]) for g in gold if exceeds_2_53(g["gold"])]
    n_big = len(big)
    audit_digest = hashlib.sha256(
        "|".join(sorted(big)).encode("utf-8")).hexdigest() if big else "none"
    # ---- questions file: NO gold, NO correctness. Generation reads only this.
    qs = [{"task": "gsm8k_hard", "sample_id": i, "question": q}
          for i, q in enumerate(chosen)]
    qpath = os.path.join(a.out_dir, f"{a.prefix}_questions.json")
    json.dump({"meta": {"protocol": "p3-v1", "hf_name": a.hf_name,
                        "revision": a.revision, "split": a.split,
                        "salt": SALT, "n": N_SAMPLE, "n_unique": n_uniq,
                        "n_source_rows": len(rows),
                        "questions_sha256": qdigest,
                        "n_gold_exceeding_2_53": n_big,
                        "bigint_audit_digest": audit_digest,
                        "contains_labels": False},
               "data": qs}, open(qpath, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- sealed gold file: the evaluator's input, and nothing else reads it.
    gpath = os.path.join(a.out_dir, f"{a.prefix}_gold.SEALED.json")
    json.dump({"meta": {"protocol": "p3-v1", "questions_sha256": qdigest,
                        "revision": a.revision,
                        "n_gold_exceeding_2_53": n_big,
                        "bigint_audit_digest": audit_digest,
                        "WARNING": ("SEALED. Do not open before p3_predictions.json "
                                    "is frozen. Reading this early destroys the "
                                    "blind property permanently.")},
               "data": gold}, open(gpath, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\nquestions -> {qpath}   (labels: NO)")
    print(f"gold      -> {gpath}   (SEALED)")
    print(f"questions_sha256 {qdigest}")
    print(f"\n[2^53 audit] gold answers exceeding 2^53: {n_big} of {N_SAMPLE}")
    print("  -> use the ORIGINAL extractor unchanged" if not n_big else
          "  -> norm_exact is REQUIRED for this validation (protocol 2.4)")
    print("\nNo model was run. No accuracy was computed.")


if __name__ == "__main__":
    main()
