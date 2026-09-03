#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRUXEval-O loader for the P4c fixed-workpoint transfer. Protocol `cruxeval-p4c-v0`.

RELATION TO P4. Not a new phase. P4 asks one question -- does a workpoint
established on GSM8K still help when carried unchanged to another reasoning
task -- and this is its THIRD task, after LogiQA 2.0 (`logiqa2-p4-v0`) and BBH
numeric (`bbh-p4b-v0`). Hence P4c, not P5.

WHAT P4c REMOVES, AND WHAT IT ADDS. It removes LogiQA's option-comparison
interface and keeps the `####` submission marker, but it changes the reasoning
content AND the answer structure at once (a constructed integer becomes an
arbitrary Python literal). So it CANNOT by itself isolate which of the two
decides the transfer outcome -- it is a third, harder point on the transfer
boundary, not a controlled single-factor manipulation. That limit is in the
pre-registration, not left to be discovered in the results.

THIS IS NOT A BLIND VALIDATION. CRUXEval gold is public, so there is no seal to
keep -- P3 (GSM-Hard) was the blind test and is CLOSED. What holds here is the
OTHER half of the discipline: alpha is read from the frozen GSM8K record and
never re-searched, and the sample is frozen before any cell runs.

Generation still has no business reading gold, so the sample is emitted TWICE
from ONE selection:

    <prefix>_formal_blind.json   300 items, gold ABSENT   -> the RUNNER reads this
    <prefix>_formal.json         300 items, gold PRESENT  -> the SCORER reads this
    <prefix>_manifest.json       provenance + digests

The blind copy is whitelist-built and then asserted label-free, the same
structural firewall as `data_logiqa2.py` and `data_bbh_numeric.py`: "unused" is
a much weaker guarantee than "unreachable" (the P2 lesson).

SAMPLING. 300 of 800, ranked by sha256(salt:id:code:input). `sha256`, never
`hash()`, which is process-salted in Python 3 and would silently produce a
different sample every run. The same 300 items in the same order serve all
eight cells.

REVISION IS PINNED to a full 40-hex commit SHA. A branch name follows upstream
exactly as an unset revision would.

TWO STRUCTURAL FACTS make a single-line `#### <literal>` marker safe, and both
are ASSERTED here rather than assumed: no gold contains a newline, and `####`
appears in no code/input/output. A gold that does not `ast.literal_eval` is
likewise a hard stop, because the scorer compares parsed Python objects.

This script downloads data. It runs no model and computes no accuracy.

Usage
-----
    python data_cruxeval.py --out_dir components/benchmark
    python data_cruxeval.py --check          # re-run every assertion, write nothing
"""

import argparse
import ast
import hashlib
import json
import os
import sys
from collections import Counter

PROTOCOL = "cruxeval-p4c-v0"
HF_NAME = "cruxeval-org/cruxeval"
# HEAD of cruxeval-org/cruxeval as verified 2026-09-03 (last modified
# 2024-01-23). The repo ships test.jsonl and is read by the json builder -- it
# is NOT a loading script, so current `datasets` accepts it.
REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
SPLIT = "test"
N_POOL = 800
N_FORMAL = 300
SALT = "cruxeval-p4c-v0"

# Measured at freeze (2026-09-03) from the pinned revision. Asserted, not
# assumed: an upstream edit would otherwise produce a plausible-looking sample
# built from different content.
EXPECTED = {
    "questions_sha256": "4580b7a9a9ef6054",
    "gold_sha256": "a214d1fc7d84a2d9",
    "n_unique_gold": 235,
    "majority_gold": "[]",
    "majority_count": 13,
}

# a blind record may contain ONLY these keys ...
BLIND_ALLOWED = {"sample_id", "source_id", "code", "input", "content_sha256"}
# ... and must contain NONE of these
LABEL_FIELDS = {"output", "target", "answer", "label", "gold", "gold_answer",
                "correct", "accuracy", "solution"}

US = "\x1f"   # unit separator: joins code+input so the digest cannot be forged
              # by moving characters across the boundary


def _commit_sha(v: str) -> str:
    v = v.strip().lower()
    if len(v) != 40 or any(c not in "0123456789abcdef" for c in v):
        raise argparse.ArgumentTypeError(
            f"--revision must be a full 40-hex commit SHA, got {v!r}. A branch "
            "name is not a pin: it follows upstream exactly as an unset "
            "revision would.")
    return v


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def rank_key(r) -> str:
    """Frozen selection key. sha256, NEVER hash() -- that is process-salted in
    Python 3, so the 'frozen' sample would differ every run."""
    return hashlib.sha256(
        f"{SALT}:{r['id']}:{r['code']}:{r['input']}".encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", default=HF_NAME)
    ap.add_argument("--revision", default=REVISION, type=_commit_sha)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out_dir", default="components/benchmark")
    ap.add_argument("--prefix", default="cruxeval_p4c")
    ap.add_argument("--check", action="store_true",
                    help="re-run every assertion and print the digests; write nothing")
    a = ap.parse_args()

    if not a.check and not os.path.isdir(a.out_dir):
        sys.exit(f"FAIL: --out_dir {a.out_dir!r} does not exist. On the server run "
                 "from /data1/paveen/Dopamine so it resolves to "
                 "components/benchmark/, or pass --out_dir explicitly. Creating "
                 "it silently would risk writing to the wrong tree.")

    from datasets import load_dataset
    ds = load_dataset(a.hf_name, split=a.split, revision=a.revision)
    print(f"loaded {a.hf_name} split={a.split} revision={a.revision} n={len(ds)}")

    # ---- HARD STOPs on schema and shape. Adapting the loader to unexpected
    # data is exactly how a wrong-content sample gets built silently.
    required = {"code", "input", "output", "id"}
    missing = required - set(ds.column_names)
    if missing:
        sys.exit(f"HARD STOP: expected columns {sorted(required)} at this revision, "
                 f"missing {sorted(missing)}; found {ds.column_names}.")
    if len(ds) != N_POOL:
        sys.exit(f"HARD STOP: expected {N_POOL} rows, got {len(ds)}. The split "
                 "changed upstream; re-freeze deliberately rather than adapting.")

    pool = [{"id": r["id"], "code": r["code"], "input": r["input"],
             "output": r["output"]} for r in ds]

    if len({r["id"] for r in pool}) != N_POOL:
        sys.exit("HARD STOP: upstream ids are not unique; source_id would not "
                 "identify an item.")

    # Every gold must be a pure Python literal: the scorer compares PARSED
    # objects, so a gold it cannot parse is not scoreable at all.
    unparsed = []
    for r in pool:
        try:
            ast.literal_eval(r["output"])
        except Exception as e:
            unparsed.append((r["id"], r["output"][:60], type(e).__name__))
    if unparsed:
        sys.exit(f"HARD STOP: {len(unparsed)} gold value(s) do not "
                 f"ast.literal_eval, e.g. {unparsed[:3]}. The scorer compares "
                 "parsed Python objects; an unparseable gold is not scoreable.")

    # A newline in a gold would break the single-LINE '#### <literal>' marker.
    nl = [r["id"] for r in pool if "\n" in r["output"]]
    if nl:
        sys.exit(f"HARD STOP: {len(nl)} gold value(s) contain a newline "
                 f"(e.g. {nl[:3]}); the frozen marker is single-line.")

    # A '####' in the source would let the parser match the prompt echo rather
    # than the model's own answer.
    if any("####" in r["code"] or "####" in r["input"] or "####" in r["output"]
           for r in pool):
        sys.exit("HARD STOP: '####' appears in the source data, which would "
                 "pollute the marker parser.")

    # ---- frozen selection
    rows = sorted(pool, key=rank_key)[:N_FORMAL]
    if len({r["id"] for r in rows}) != N_FORMAL:
        sys.exit("HARD STOP: the selected sample has duplicate ids.")

    qdigest = sha16("\n".join(r["code"] + US + r["input"] for r in rows))
    gdigest = sha16("\n".join(r["output"] for r in rows))

    golds = [r["output"] for r in rows]
    c = Counter(golds)
    mode, mcount = c.most_common(1)[0]
    for name, got, want in (("questions_sha256", qdigest, EXPECTED["questions_sha256"]),
                            ("gold_sha256", gdigest, EXPECTED["gold_sha256"]),
                            ("n_unique_gold", len(c), EXPECTED["n_unique_gold"]),
                            ("majority_gold", mode, EXPECTED["majority_gold"]),
                            ("majority_count", mcount, EXPECTED["majority_count"])):
        if got != want:
            sys.exit(f"HARD STOP: {name} is {got!r}, frozen value is {want!r}. "
                     "The content or the selection changed; re-freeze "
                     "deliberately rather than accepting a different sample "
                     "under this name.")

    # DESCRIPTIVE ONLY. The type distribution is provenance: it does NOT
    # re-stratify the sample, does not adjust the selection, and is not a
    # stratification axis for any analysis. The majority-class rate is the
    # trivial constant-guess baseline; this protocol has NO accuracy gate for
    # it to move.
    types = Counter(type(ast.literal_eval(g)).__name__ for g in golds)
    maj_rate = mcount / N_FORMAL

    print("\n[sample properties -- DESCRIPTIVE provenance, they gate nothing]")
    print(f"  formal n           : {N_FORMAL} of {N_POOL}")
    print(f"  unique gold        : {len(c)}")
    print(f"  majority-class     : gold={mode!r} rate={maj_rate:.4f}  (trivial constant guess)")
    print(f"  gold type dist     : {dict(types.most_common())}")
    print(f"  code chars med     : {sorted(len(r['code']) for r in rows)[N_FORMAL // 2]}")
    print(f"  gold chars med/max : {sorted(len(g) for g in golds)[N_FORMAL // 2]}"
          f" / {max(len(g) for g in golds)}")
    print(f"\n  questions_sha256[:16] {qdigest}")
    print(f"  gold_sha256[:16]      {gdigest}")

    meta = {
        "protocol": PROTOCOL, "task": "cruxeval_o",
        "hf_name": a.hf_name, "revision": a.revision, "split": a.split,
        "n_pool": N_POOL, "n": N_FORMAL, "salt": SALT,
        "selection_rule": "rank by sha256(salt:id:code:input), take first 300",
        "questions_sha256": qdigest, "gold_sha256": gdigest,
        "n_unique_gold": len(c), "majority_gold": mode,
        "majority_class_rate": maj_rate,
        "gold_type_distribution": dict(types.most_common()),
        "gold_all_literal_eval": True,
        "scoring": ("ast.literal_eval on both sides, then Python object "
                    "equality. NOT the official exec-based pass@1: a "
                    "non-literal expression that evaluates correctly is scored "
                    "incorrect here. Never exec model-generated text."),
        "accuracy_gate": None,
        "gate_note": ("P4c has NO accuracy gate. A low baseline is recorded as "
                      "a limitation on the reading, not used to cancel the "
                      "test. Hard stops are technical only."),
        "blind_validation": False,
        "blind_note": ("CRUXEval gold is public: this is fixed-workpoint "
                       "transfer, not blind validation. alpha is read from the "
                       "frozen GSM8K record and never re-searched."),
    }

    if a.check:
        print(f"\n[--check] all assertions passed; nothing written.")
        return

    # ---- blind copy: whitelist-built, then asserted label-free.
    blind = []
    for i, r in enumerate(rows):
        rec = {"sample_id": i, "source_id": r["id"], "code": r["code"],
               "input": r["input"],
               "content_sha256": sha16(r["code"] + US + r["input"])}
        extra = set(rec) - BLIND_ALLOWED
        if extra:
            sys.exit(f"FAIL: blind record carries non-whitelisted key(s) {sorted(extra)}")
        leaked = {k for k in rec if k.lower() in LABEL_FIELDS}
        if leaked:
            sys.exit(f"FAIL: label field {sorted(leaked)} reached the blind record")
        blind.append(rec)
    # Belt and braces: assert on the SERIALIZED payload, not just the keys of
    # each record. A value scan is deliberately NOT done -- a gold literal such
    # as `[]` or `'a'` legitimately occurs inside `code`/`input` text, so
    # scanning values would fire on honest data and teach the reader to ignore
    # this guard.
    payload = json.loads(json.dumps(blind, ensure_ascii=False))
    stray = sorted({k for rec in payload for k in rec if k.lower() in LABEL_FIELDS})
    if stray:
        sys.exit(f"FAIL: label key {stray} survived into the blind payload")

    bpath = os.path.join(a.out_dir, f"{a.prefix}_formal_blind.json")
    gpath = os.path.join(a.out_dir, f"{a.prefix}_formal.json")
    mpath = os.path.join(a.out_dir, f"{a.prefix}_manifest.json")
    for p in (bpath, gpath, mpath):
        if os.path.exists(p):
            sys.exit(f"FAIL: {p} exists; refusing to overwrite a frozen artifact. "
                     "Delete it deliberately if this is a re-freeze.")

    json.dump({"meta": {**meta, "contains_labels": False}, "data": blind},
              open(bpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    gold_rows = [{"sample_id": i, "source_id": r["id"], "code": r["code"],
                  "input": r["input"], "gold": r["output"],
                  "content_sha256": sha16(r["code"] + US + r["input"])}
                 for i, r in enumerate(rows)]
    json.dump({"meta": {**meta, "contains_labels": True}, "data": gold_rows},
              open(gpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    json.dump(meta, open(mpath, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\nblind    -> {bpath}   (labels: NO -- the runner reads this)")
    print(f"gold     -> {gpath}   (labels: YES -- the scorer reads this)")
    print(f"manifest -> {mpath}")
    print("\nNo model was run. No accuracy was computed.")


if __name__ == "__main__":
    main()
