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
AUDIT_VERSION = "p3-bigint-audit-v1"
LIMIT_2_53 = 2 ** 53


def _commit_sha(v: str) -> str:
    """Reject anything that is not a full 40-hex commit SHA.

    A branch name or short SHA is not a pin: `--revision main` follows upstream
    exactly as an unset revision would, and a short SHA can become ambiguous.
    Pinning is the only thing that makes "the P3 sample" mean one fixed object.
    """
    v = v.strip()
    if len(v) != 40 or any(c not in "0123456789abcdef" for c in v.lower()):
        raise argparse.ArgumentTypeError(
            f"--revision must be a full 40-hex commit SHA, got {v!r}. "
            "Branch names (main/latest) and short SHAs are refused: they follow "
            "upstream and would silently change what the P3 sample means.")
    return v.lower()


class UnsafeFloatGold(RuntimeError):
    """Gold arrived as a float64 already in the lossy integer range."""


def assert_float_safe(gold) -> None:
    """HARD STOP if a float64 gold reaches the unsafe integer range.

    This closes a hole no amount of string parsing can: HF stores GSM-Hard's
    `target` as float64, so `datasets` hands us a PYTHON FLOAT. By then
    float(2**53 + 1) is already exactly 2**53 -- the +1 was destroyed upstream,
    before any of our code ran, and Decimal(str(x)) cannot recover it. The
    earlier tests used the STRING '9007199254740993', which never exercises
    this path and so passed while the real input would have been missed.

    Refusing is the only honest response: we cannot know whether such a value
    was exact or already rounded, and silently auditing it as "within range"
    would under-report the very hazard the audit exists to find.
    """
    if isinstance(gold, float) and abs(gold) >= LIMIT_2_53:
        raise UnsafeFloatGold(
            f"HARD STOP: float64 gold {gold!r} reaches the unsafe integer range "
            f"(>= 2**53). The exact value cannot be recovered from a float, so "
            f"the 2^53 audit cannot be trusted for this revision. Protocol 2.4 "
            f"requires an exact audit; do not proceed.")


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
    ap.add_argument("--revision", required=True, type=_commit_sha,
                    help=("REQUIRED, and must be a full 40-hex commit SHA. "
                          "`required=True` alone is not a pin: --revision main "
                          "still follows upstream. Confirm HEAD before use."))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="benchmark")
    ap.add_argument("--prefix", default="gsm_hard_p3")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(a.hf_name, split=a.split, revision=a.revision)
    print(f"loaded {a.hf_name} split={a.split} revision={a.revision} n={len(ds)}")
    print(f"columns: {ds.column_names}")

    # HARD STOP on schema (protocol 2.1). A silent fallback to question/answer
    # would let an unexpected upstream schema quietly produce a plausible-looking
    # sample built from the wrong columns; the protocol says incompatibility is a
    # stop, not something to adapt around.
    required = {"input", "target"}
    missing = required - set(ds.column_names)
    if missing:
        sys.exit(f"HARD STOP: expected columns {sorted(required)} at this revision, "
                 f"missing {sorted(missing)}; found {ds.column_names}. "
                 "Protocol 2.1: schema incompatibility is a hard stop -- do not "
                 "adapt the loader to the data.")
    rows = [{"q": r["input"], "g": r["target"]} for r in ds]

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
    for g in gold:                       # hard stop BEFORE counting
        assert_float_safe(g["gold"])
    n_big = sum(1 for g in gold if exceeds_2_53(g["gold"]))
    # The digest is derived from revision + question digest + count + audit
    # version -- NOT from raw gold. A gold-derived hash in the label-free
    # questions file would put a (weak, but real) label-dependent value on the
    # blind side of the firewall for no benefit: the count is what P3 acts on.
    audit_digest = hashlib.sha256(
        f"{AUDIT_VERSION}|{a.revision}|{qdigest}|{n_big}".encode("utf-8")).hexdigest()
    # ---- questions file: NO gold, NO correctness. Generation reads only this.
    qs = [{"task": "gsm_hard", "sample_id": i, "question": q}
          for i, q in enumerate(chosen)]
    qpath = os.path.join(a.out_dir, f"{a.prefix}_questions.json")
    json.dump({"meta": {"protocol": "p3-v1", "hf_name": a.hf_name,
                        "revision": a.revision, "split": a.split,
                        "salt": SALT, "n": N_SAMPLE, "n_unique": n_uniq,
                        "n_source_rows": len(rows),
                        "questions_sha256": qdigest,
                        "n_gold_exceeding_2_53": n_big,
                        "bigint_audit_digest": audit_digest,
                        "audit_version": AUDIT_VERSION,
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
