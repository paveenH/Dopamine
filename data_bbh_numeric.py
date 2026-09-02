#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBH numeric-answer loader for the P5 fixed-workpoint transfer stage-0 gate.

Protocol: `bbh-p5-v0` (stage 0 = the alpha=0 headroom gate; the workpoint
cells are only frozen if a model passes it).

TASK ALLOWLIST -- deliberately only two configs
-----------------------------------------------
    object_counting            counting / enumeration, small-integer answer
    multistep_arithmetic_two   nested arithmetic, integer answer

Both keep GSM8K's answer space (a model-CONSTRUCTED integer) and its submission
interface (`#### <integer>`, scored by the shared `extract_gsm8k_answer`), so
relative to GSM8K only the REASONING CONTENT changes. That minimal-variable
property is the whole reason these two were chosen over a third BBH task:
`dyck_languages` would additionally change the answer space to a symbol string
and require a new parser, adding a second variable and a new way to fail
technically. It is deliberately NOT implemented here; it needs its own frozen
parser and prompt if the two numeric tasks both turn out unreadable.

THIS IS NOT A BLIND VALIDATION, and the file layout says so honestly.
BBH gold is public and reachable, so there is no seal to keep -- P3 (GSM-Hard)
was the blind test and is CLOSED. What must hold here is the OTHER half of the
P3/P4 discipline: alpha is read from the frozen GSM8K record and never
re-searched, the predictor / features / marker adapter are not touched, and the
sample is frozen before any steered cell runs. Writing a gold-bearing file is
therefore correct; pretending to a seal we do not have would be theatre.

Still, generation has no business reading gold, so the sample is emitted TWICE
from ONE selection:

    <prefix>_<task>_blind.json   250 items, gold ABSENT   -> the RUNNER reads this
    <prefix>_<task>.json         250 items, gold PRESENT  -> the SCORER reads this
    <prefix>_<task>_manifest.json  provenance + digests

The blind copy is built from an explicit field WHITELIST and then asserted to
carry no forbidden key, the same structural firewall as `data_logiqa2.py`:
"unused" is a much weaker guarantee than "unreachable" (the P2 lesson).

WHOLE SPLIT, NO SAMPLING. Each config's test split is exactly 250 rows and all
250 are used, so there is no selection step and no salt. The digest is over the
question texts in frozen order, which is what a later run must reproduce.
`sha256` is used rather than `hash()`, which is process-salted in Python 3 and
would silently differ every run.

REVISION IS PINNED to a full 40-hex commit SHA. A branch name follows upstream
exactly as an unset revision would.

This script downloads data. It runs no model and computes no accuracy.

Usage
-----
    python data_bbh_numeric.py --task object_counting --out_dir components/benchmark
    python data_bbh_numeric.py --task object_counting --check     # no write
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

PROTOCOL = "bbh-p5-v0"
HF_NAME = "lukaemon/bbh"
# HEAD of lukaemon/bbh as verified 2026-09-02; both configs are pure Parquet
# with a 250-row `test` split and columns ['input', 'target'].
REVISION = "982bb89fd79532a8ac676a61fc42eb1aeec63f99"
SPLIT = "test"
N_EXPECTED = 250

# Only these two. A third task is not a config change: it needs its own frozen
# parser, prompt and gate, so it may not be smuggled in through --task.
TASKS = ("object_counting", "multistep_arithmetic_two")

# Measured at freeze (2026-09-02) from the pinned revision. Asserted, not
# assumed: an upstream edit that changed the data would otherwise produce a
# plausible-looking sample from different content.
EXPECTED = {
    "object_counting": {
        "n_unique_gold": 17, "majority_gold": "3", "majority_count": 26,
        "questions_sha256": "4cfbf739e1fe7870",
        "gold_sha256": "349c9729336e6170",
    },
    "multistep_arithmetic_two": {
        "n_unique_gold": 185, "majority_gold": "-35", "majority_count": 4,
        "questions_sha256": "e69d300b94274ce3",
        "gold_sha256": "e9bd0f8ce0bdae6a",
    },
}

# a blind record may contain ONLY these keys ...
BLIND_ALLOWED = {"task", "sample_id", "question", "content_sha256"}
# ... and must contain NONE of these
LABEL_FIELDS = {"target", "answer", "label", "gold", "gold_answer",
                "correct", "accuracy", "solution"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS,
                    help="only the two NUMERIC-answer BBH configs are allowed")
    ap.add_argument("--hf_name", default=HF_NAME)
    ap.add_argument("--revision", default=REVISION, type=_commit_sha)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out_dir", default="components/benchmark")
    ap.add_argument("--prefix", default="bbh_p5")
    ap.add_argument("--check", action="store_true",
                    help="re-run every assertion and print the digests; write nothing")
    a = ap.parse_args()

    if not a.check and not os.path.isdir(a.out_dir):
        sys.exit(f"FAIL: --out_dir {a.out_dir!r} does not exist. On the server run "
                 "from /data1/paveen/Dopamine so it resolves to "
                 "components/benchmark/, or pass --out_dir explicitly. Creating "
                 "it silently would risk writing to the wrong tree.")

    from datasets import load_dataset
    ds = load_dataset(a.hf_name, a.task, split=a.split, revision=a.revision)
    print(f"loaded {a.hf_name}:{a.task} split={a.split} revision={a.revision} n={len(ds)}")

    # ---- HARD STOPs on schema and shape. Adapting the loader to unexpected
    # data is exactly how a wrong-content sample gets built silently.
    required = {"input", "target"}
    missing = required - set(ds.column_names)
    if missing:
        sys.exit(f"HARD STOP: expected columns {sorted(required)} at this revision, "
                 f"missing {sorted(missing)}; found {ds.column_names}.")
    if len(ds) != N_EXPECTED:
        sys.exit(f"HARD STOP: expected {N_EXPECTED} rows, got {len(ds)}. The split "
                 "changed upstream; re-freeze deliberately rather than adapting.")

    rows = [{"q": r["input"], "g": str(r["target"]).strip()} for r in ds]
    qs = [r["q"] for r in rows]
    golds = [r["g"] for r in rows]

    if len(set(qs)) != N_EXPECTED:
        sys.exit(f"HARD STOP: questions are not unique ({len(set(qs))}/{N_EXPECTED}); "
                 "sample_id would not identify an item.")
    # Every gold must be an integer literal, or normalize_gsm8k's float round
    # trip is not the identity and scoring silently changes meaning.
    bad = [g for g in golds if not g.lstrip("-").isdigit()]
    if bad:
        sys.exit(f"HARD STOP: {len(bad)} non-integer gold value(s), e.g. {bad[:3]}. "
                 "The shared GSM8K normalizer assumes integer gold here.")
    # A gold that already looks like a submission marker would let the parser
    # match the prompt echo rather than the model's answer.
    if any("####" in q for q in qs):
        sys.exit("HARD STOP: a question contains '####', which would pollute the parser.")

    c = Counter(golds)
    mode, mcount = c.most_common(1)[0]
    exp = EXPECTED[a.task]
    for name, got, want in (("n_unique_gold", len(c), exp["n_unique_gold"]),
                            ("majority_gold", mode, exp["majority_gold"]),
                            ("majority_count", mcount, exp["majority_count"])):
        if got != want:
            sys.exit(f"HARD STOP: {name} is {got!r}, frozen value is {want!r}. "
                     "The data changed upstream; re-freeze deliberately.")

    qdigest = sha16("\n".join(qs))
    gdigest = sha16("\n".join(golds))
    for name, got in (("questions_sha256", qdigest), ("gold_sha256", gdigest)):
        if got != exp[name]:
            sys.exit(f"HARD STOP: {name} is {got}, frozen value is {exp[name]}. "
                     "The content changed upstream; re-freeze deliberately "
                     "rather than accepting a different sample under this name.")

    # DESCRIPTIVE ONLY. The majority-class rate is the trivial constant-guess
    # baseline, NOT a random-guess rate, and it deliberately does NOT move the
    # frozen [.30, .85] gate: letting it do so would make the gate adjustable
    # after seeing the data. It is recorded so a reader can judge how much of
    # the observed accuracy is absolute headroom.
    maj_rate = mcount / N_EXPECTED

    print(f"\n[data properties -- DESCRIPTIVE, they do NOT move the gate]")
    print(f"  unique gold values : {len(c)}")
    print(f"  gold range         : {min(int(g) for g in golds)} .. {max(int(g) for g in golds)}")
    print(f"  majority-class     : gold={mode!r} rate={maj_rate:.4f}  (trivial constant guess)")
    print(f"  question chars med : {sorted(len(q) for q in qs)[N_EXPECTED // 2]}")
    print(f"\n  questions_sha256[:16] {qdigest}")
    print(f"  gold_sha256[:16]      {gdigest}")

    meta = {
        "protocol": PROTOCOL, "task": a.task, "hf_name": a.hf_name,
        "revision": a.revision, "split": a.split, "n": N_EXPECTED,
        "questions_sha256": qdigest, "gold_sha256": gdigest,
        "n_unique_gold": len(c), "majority_gold": mode,
        "majority_class_rate": maj_rate,
        "gold_min": min(int(g) for g in golds), "gold_max": max(int(g) for g in golds),
        "stage0_gate": "alpha=0 first_acc in [0.30, 0.85], judged per model",
        "gate_note": ("majority_class_rate is DESCRIPTIVE and does not move the "
                      "gate; the interval is frozen"),
        "blind_validation": False,
        "blind_note": ("BBH gold is public: this is a fixed-workpoint transfer "
                       "test, not a blind validation. alpha is read from the "
                       "frozen GSM8K record and never re-searched."),
    }

    if a.check:
        print(f"\n[--check] all assertions passed for {a.task}; nothing written.")
        return

    # ---- blind copy: whitelist-built, then asserted label-free.
    blind = []
    for i, r in enumerate(rows):
        rec = {"task": a.task, "sample_id": i, "question": r["q"],
               "content_sha256": sha16(r["q"])}
        extra = set(rec) - BLIND_ALLOWED
        if extra:
            sys.exit(f"FAIL: blind record carries non-whitelisted key(s) {sorted(extra)}")
        leaked = {k for k in rec if k.lower() in LABEL_FIELDS}
        if leaked:
            sys.exit(f"FAIL: label field {sorted(leaked)} reached the blind record")
        blind.append(rec)

    bpath = os.path.join(a.out_dir, f"{a.prefix}_{a.task}_blind.json")
    gpath = os.path.join(a.out_dir, f"{a.prefix}_{a.task}.json")
    mpath = os.path.join(a.out_dir, f"{a.prefix}_{a.task}_manifest.json")
    for p in (bpath, gpath, mpath):
        if os.path.exists(p):
            sys.exit(f"FAIL: {p} exists; refusing to overwrite a frozen artifact. "
                     "Delete it deliberately if this is a re-freeze.")

    json.dump({"meta": {**meta, "contains_labels": False}, "data": blind},
              open(bpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    gold_rows = [{"task": a.task, "sample_id": i, "question": r["q"],
                  "gold": r["g"], "content_sha256": sha16(r["q"])}
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
