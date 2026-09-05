#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
READ-ONLY lister of Unknown-labeled candidates from the official ProofWriter
TRAIN split, for a human to pick ONE as the v2 fixed few-shot exemplar.

This script does not select, freeze, or write anything -- it only prints.
It exists because PREREG_PROOFWRITER_OWA.md S4 requires any exemplar to come
from the official train split, verbatim, and the v2 prompt revision requires
one short, fixed Unknown-labeled exemplar. Picking that exemplar is a human
decision (2026-09-05): "exemplar theory/question/label must come from the
official train split; if the dataset has no natural-language reasoning, a
brief explanation may be written based on the official proof but must be
clearly marked as our own construction, never called official reasoning."

WHAT THIS PRINTS, per candidate:
  - dataset (D3/D5)
  - official_theory_id, official_qid (so the exact row can be re-located)
  - split (must read "train" -- asserted, not assumed)
  - theory_text, question_text (official, byte-unchanged)
  - gold label (must read "Unknown" -- filtered on this)
  - proof_text / proofs field AS STORED (raw, unedited) -- so a human can see
    whatever the release provides for an Unknown item, if anything, before
    writing a short explanation from it
  - n_facts, n_rules, qdep (for judging "simple/short" by hand)
  - a naive char-length of theory_text+question_text, to help filter for
    SHORT candidates (the v2 requirement), sorted ascending by this length

WHAT THIS DOES NOT DO:
  - does not touch the test split or the frozen 300-item manifest at all
  - does not write any file
  - does not pick a "best" candidate automatically -- ranking is by length
    only, as a convenience; the actual selection is a human judgment call
  - does not synthesize a reasoning sentence -- that happens later, by hand,
    only after a human has chosen one row from this listing, and even then
    must be labeled as this project's own construction, never "official"

Usage (run on the server, where the official archive is already downloaded):
    python3 list_unknown_exemplar_candidates.py --archive_dir <dir> \\
        [--dataset D3] [--max_theory_question_chars 400] [--limit 20]

Reuses proofwriter_owa/data_proofwriter_owa.py's own archive parser
(parse_depth_split) rather than re-implementing schema handling -- so this
lister fails closed on the exact same schema surprises the loader does, and
never silently diverges from it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_proofwriter_owa import (  # noqa: E402
    ARCHIVE_BASENAME, DEPTH_TASKS, EXEMPLAR_SPLIT, parse_depth_split, die,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive_dir", default=None,
                    help="directory already containing the official zip")
    ap.add_argument("--archive_path", default=None,
                    help="explicit path to the zip, overrides --archive_dir")
    ap.add_argument("--dataset", choices=sorted(DEPTH_TASKS), default=None,
                    help="restrict to D3 or D5; default = both")
    ap.add_argument("--max_theory_question_chars", type=int, default=None,
                    help="only list candidates whose theory_text+question_text "
                         "combined length is <= this many characters "
                         "(convenience filter for 'must be short')")
    ap.add_argument("--limit", type=int, default=20,
                    help="print at most this many candidates per dataset "
                         "(shortest first); does not affect selection, only "
                         "how much is printed")
    return ap.parse_args()


def main():
    a = parse_args()

    zip_path = a.archive_path
    if not zip_path and a.archive_dir:
        zip_path = os.path.join(a.archive_dir, ARCHIVE_BASENAME)
    if not zip_path or not os.path.exists(zip_path):
        die(f"official archive not found (looked for {zip_path!r}). Pass "
            "--archive_path/--archive_dir pointing at an already-downloaded "
            "proofwriter-dataset-V2020.12.3.zip (this lister does not "
            "download anything itself).")

    datasets = [a.dataset] if a.dataset else sorted(DEPTH_TASKS)

    for ds in datasets:
        recs = parse_depth_split(zip_path, ds, EXEMPLAR_SPLIT)
        # parse_depth_split stamps every record's "split" field from the
        # argument it was called with, so this assertion is really "did we
        # call it correctly", not "did the archive lie" -- kept anyway as a
        # belt-and-suspenders check against a future refactor of that
        # function silently changing what it stamps.
        wrong_split = [r for r in recs if r["split"] != EXEMPLAR_SPLIT]
        if wrong_split:
            die(f"{ds}: {len(wrong_split)} record(s) do not carry "
                f"split={EXEMPLAR_SPLIT!r} as expected; refusing to list "
                "exemplar candidates from a mixed-split result.")

        unknown = [r for r in recs if r["answer"] == "Unknown"]
        for r in unknown:
            r["_combined_len"] = len(r["theory_text"]) + len(r["question_text"])
        if a.max_theory_question_chars is not None:
            unknown = [r for r in unknown
                      if r["_combined_len"] <= a.max_theory_question_chars]
        unknown.sort(key=lambda r: r["_combined_len"])

        print(f"\n{'=' * 70}")
        print(f"{ds}  (split={EXEMPLAR_SPLIT})  "
              f"{len(unknown)} Unknown candidate(s)"
              f"{' after length filter' if a.max_theory_question_chars else ''}"
              f", showing up to {a.limit}, shortest first")
        print(f"{'=' * 70}")

        for r in unknown[:a.limit]:
            print(f"\n--- {ds} theory_id={r['official_theory_id']!r} "
                  f"qid={r['official_qid']!r}  "
                  f"combined_chars={r['_combined_len']}  "
                  f"n_facts={r['n_facts']}  n_rules={r['n_rules']}  "
                  f"qdep={r['qdep']} ---")
            print(f"theory_text : {r['theory_text']}")
            print(f"question_text: {r['question_text']}")
            print(f"gold answer  : {r['answer']}")
            proof_repr = r["proof_text"] if r["proof_text"] else "(empty/absent)"
            print(f"proof_text (AS STORED, raw): {proof_repr}")

        if not unknown:
            print("(no candidates matched -- widen or drop "
                  "--max_theory_question_chars)")

    print(f"\n{'=' * 70}")
    print("Nothing was written. Pick ONE (dataset, theory_id, qid) above and "
          "hand its theory_text/question_text/proof_text back for freezing "
          "into the v2 exemplar file. If proof_text is empty/absent, a short "
          "explanation may be hand-written from the theory+question, but it "
          "must be labeled as this project's own construction -- never "
          "called the official reasoning.")


if __name__ == "__main__":
    main()
