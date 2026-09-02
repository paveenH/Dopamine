#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Freeze the earlycand-v1 manual-audit item list for P4b. NO MODEL, NO GOLD.

WHY THIS RUNS BEFORE STAGE-0, NOT AFTER
---------------------------------------
`earlycand-v1` was frozen on GSM8K, where a first line that is short and
contains a number is an answer-shaped commitment. On `object_counting` the
question itself is a list of objects and the reasoning is a running count, so
a leading number may just be a restatement ("I have three apples"). The
detector's GSM8K blind audit measured precision 1.000, but that was on
arithmetic text and does not transfer by assumption.

So the detector must be validated on this task before it may be called a
commitment metric. Choosing WHICH items to audit after seeing the detector's
output would let the sample be picked to flatter it. This script therefore
fixes the 30 items from the frozen question digests alone, before any
generation exists.

VALIDATION IS BLIND IN ONE DIRECTION. The rubric is applied to the generated
text WITHOUT looking at the detector's flag; only afterwards are the two
compared. That ordering is the whole point -- reading the prediction first
turns a precision estimate into a confirmation.

OUTCOME RULE, frozen here so it cannot be renegotiated later:
  * validation passes -> early_candidate may be reported as an EXPLORATORY
    timing readout for this task, always labelled as such (it is outside the
    Holm family, and it is an OUTCOME of alpha, so stratifying accuracy on it
    is post-treatment stratification -- consistent-with evidence, never
    mediation).
  * validation fails  -> early_candidate is withdrawn as a timing metric for
    this task. Only marker/format descriptions survive. THIS DOES NOT AFFECT
    THE ACCURACY MAIN TEST, which never uses the detector.
  * fewer than 10 detector positives among the 30 -> INCONCLUSIVE. Precision
    over a handful of positives is not an estimate, so no precision claim is
    available in either direction. Report the raw counts and treat
    early_candidate as descriptive only. Do NOT re-draw the sample, enlarge it
    to chase positives, or re-tune the detector: each of those would select the
    audit on the detector's own output, which freezing the list in advance
    exists to prevent.

The detector is NOT re-tuned either way. Retuning it here would fork the
definition against every stored GSM8K and MATH number.

Usage
-----
    python freeze_p4b_earlycand_audit.py --task object_counting \
        --bench components/benchmark
    python freeze_p4b_earlycand_audit.py --task object_counting --check
"""

import argparse
import hashlib
import json
import os
import sys

PROTOCOL = "bbh-p4b-v0"
AUDIT_VERSION = "p4b-earlycand-audit-v1"
SALT = "p4b-earlycand-audit-v1"
N_AUDIT = 30

RUBRIC = [
    "Read ONLY the generated text. Do not look at the detector flag, the gold "
    "answer, or the accuracy of the sample.",
    "Label answer_first = TRUE when the model states a candidate FINAL answer "
    "to the question before doing any of the counting/arithmetic that would "
    "justify it.",
    "Label answer_first = FALSE when the leading number is part of the working "
    "-- restating the question's objects, naming a quantity being counted, or "
    "an intermediate subtotal -- even if it is the first token.",
    "A number that merely echoes the question ('I have three apples, ...') is "
    "FALSE: echoing is not committing.",
    "If the text states no candidate answer anywhere, label FALSE.",
    "Record a one-line reason for every item, so a disagreement can be "
    "re-read rather than re-litigated.",
    "Report the RAW COUNTS -- detector positives, manual positives, and the "
    "agreements between them -- not only a precision figure. Precision over a "
    "handful of positives is not an estimate.",
]


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=("object_counting", "multistep_arithmetic_two"))
    ap.add_argument("--bench", default="components/benchmark")
    ap.add_argument("--out_dir", default="docs")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    # The BLIND file, deliberately: selecting audit items must not touch gold.
    qpath = os.path.join(a.bench, f"bbh_p4b_{a.task}_blind.json")
    if not os.path.isfile(qpath):
        sys.exit(f"FAIL: {qpath} not found; run data_bbh_numeric.py --task "
                 f"{a.task} first.")
    blob = json.load(open(qpath, encoding="utf-8"))
    meta, data = blob["meta"], blob["data"]
    if meta.get("contains_labels") is not False:
        sys.exit("FAIL: audit selection must read the BLIND file")
    if meta.get("task") != a.task:
        sys.exit(f"FAIL: file declares task {meta.get('task')!r}")

    # Rank by a salted digest of the question text, NOT by sample_id order:
    # taking the first 30 would make the audit an artifact of dataset order.
    # sha256, never hash() -- hash() on str is process-salted in Python 3 and
    # would silently select a different 30 items on every run.
    ranked = sorted(data, key=lambda r: sha16(f"{SALT}:{r['question']}"))
    chosen = ranked[:N_AUDIT]
    ids = [r["sample_id"] for r in chosen]

    sel_digest = sha16("|".join(str(i) for i in ids))
    out = {
        "protocol": PROTOCOL, "audit_version": AUDIT_VERSION, "task": a.task,
        "detector": "earlycand-v1",
        "detector_change_allowed": False,
        "n_audit": N_AUDIT, "salt": SALT,
        "questions_sha256": meta["questions_sha256"],
        "revision": meta["revision"],
        "sample_ids": ids,
        "selection_digest": sel_digest,
        "cell_to_audit": "alpha=0 only (the stage-0 cell)",
        "procedure": ("Label each item by the rubric WITHOUT seeing the "
                      "detector flag; only then compute precision/recall "
                      "against earlycand-v1."),
        "rubric": RUBRIC,
        "outcome_rule": {
            "pass": ("early_candidate may be reported as an EXPLORATORY timing "
                     "readout for this task; outside Holm; it is an outcome of "
                     "alpha, so stratifying accuracy on it is post-treatment"),
            "fail": ("early_candidate is WITHDRAWN as a timing metric for this "
                     "task; only marker/format description survives"),
            "inconclusive_low_positives": (
                "If the detector flags FEWER THAN 10 of the 30 items positive, "
                "precision is NOT validated and must not be claimed. Report the "
                "raw counts (detector positives / manual positives / "
                "agreements) and label the audit INCONCLUSIVE. Treat "
                "early_candidate as descriptive only for this task. Do NOT "
                "re-draw the 30 items, do not enlarge the sample to chase "
                "positives, and do not re-tune the detector -- any of those "
                "would select the audit sample on the detector's own output, "
                "which is what freezing it in advance prevents."),
            "either_way": ("the accuracy main test is unaffected -- it never "
                           "uses the detector -- and the detector is NOT "
                           "re-tuned, which would fork the definition against "
                           "every stored GSM8K and MATH number"),
        },
        "min_positives_for_precision_claim": 10,
    }

    print(f"task={a.task}  n={N_AUDIT}  selection_digest={sel_digest}")
    print(f"sample_ids: {ids}")
    if a.check:
        print("\n[--check] nothing written.")
        return

    if not os.path.isdir(a.out_dir):
        sys.exit(f"FAIL: --out_dir {a.out_dir!r} does not exist")
    path = os.path.join(a.out_dir, f"p4b_earlycand_audit_{a.task}.json")
    if os.path.exists(path):
        sys.exit(f"FAIL: {path} exists; refusing to overwrite a frozen "
                 "artifact. Delete it deliberately if this is a re-freeze.")
    json.dump(out, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nwrote {path}")
    print("No model was run. No gold was read.")


if __name__ == "__main__":
    main()
