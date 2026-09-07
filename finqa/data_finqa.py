#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinQA loader for the task-specific steering exploration (protocol `finqa-v0`).

RELATION TO P3/P4/P4b/P4c: NOT a fixed-workpoint transfer test and NOT a blind
validation. This is a task-specific dose exploration on FinQA, analogous to
ZebraLogic-Easy / ProofWriter-OWA -- alpha is swept fresh on this task, not
read from the frozen GSM8K record. FinQA's official test-split gold is public,
so there is no seal to keep; generation simply does not read gold/program
fields (a P4b-style simplification, not a structural label firewall).

DATA SOURCE. `ibm-research/finqa`, config `en` (pure Parquet, no loading
script), `test` split, 1147 rows. Columns used: `id`, `pre_text`, `post_text`,
`table`, `question`, `answer`, `program_re`. `program_re` is read ONLY here,
offline, to build the stratification manifest (operation count) -- it is NEVER
passed to the generator or the scorer.

NUMERIC-ANSWER ELIGIBILITY (the user's rule, not a naive "digits only" filter):
`$13.4 million`, `129.60%`, `-31.47,`, comma-grouped and unit-suffixed answers
are ALL numeric answers and must be normalized and kept. Only genuinely
non-numeric gold is excluded: empty string, bare `yes`/`no`, or text that does
not reduce to a single number after stripping `$`, commas, `%`, and a
million/billion/thousand/trillion unit suffix (prose, per-share fragments,
ratios like `1.371:1`, multi-value answers like `$411906848 or $411.91
million`). Measured on the full 1147-row test split at the pinned revision:
1097 eligible, 50 excluded (22 yes/no, 14 empty, 14 unconvertible
text/ratio/multi-value) -- exact breakdown recorded in the manifest and
re-printed by `--check` every run so drift from a future revision is visible
immediately rather than silently changing the sample.

STRATIFICATION. Two axes, both computed OFFLINE from fields the generator
never sees:
    reasoning steps  = count of top-level function calls in `program_re`
                        (comma-split at depth 0), bucketed 1 / 2 / 3+
    table size        = len(row['table']), bucketed small(<=6) / medium(7-12)
                        / large(>12) rows (quantile-informed, see EXPECTED)
Cross of the two gives up to 9 strata; empty strata are simply not sampled
from. Within each stratum, items are ranked by `sha256(salt:id)` (never
`hash()`, which is process-salted in Python 3.10+ and would give a different
order every run) and the frozen 300 draws a proportional share per stratum
(largest remainder), so the sample is deterministic and reproducible from the
pinned revision alone.

FROZEN 300 = the entire formal sample. The 30-item preflight is NOT a
separate draw -- it is a fixed, published subset of indices 0..29 into the
already-ordered 300, chosen to cover the majority strata (see
PREFLIGHT_INDICES below). There is no disjointness requirement here because
they are the same pool by design.

OUTPUT. One JSON with gold present (`finqa_formal.json`) -- generation reads
only `question`/`pre_text`/`post_text`/`table`/`sample_id`/`table_linear`, and
`get_answer_finqa.py` is written to touch nothing else. A stratification
manifest (`finqa_manifest.json`) records the digests, strata counts and
exclusion breakdown for provenance.

REVISION is pinned to a full 40-hex commit SHA (dataset repo, not a model
revision) so an upstream edit does not silently reshuffle the sample; an unset
or branch-name revision would follow upstream exactly as unset would.

Usage (run from the repo root, not from inside finqa/)
-------------------------------------------------------
    python finqa/data_finqa.py --out_dir components/benchmark
    python finqa/data_finqa.py --out_dir components/benchmark --check   # verify only, no write
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

PROTOCOL = "finqa-v0"
HF_NAME = "ibm-research/finqa"
CONFIG = "en"
SPLIT = "test"
# Pinned dataset-repo commit SHA (verified via HfApi().dataset_info(HF_NAME).sha
# on 2026-09-07). A branch name or unset revision follows upstream exactly as
# unset would, which would silently reshuffle the frozen 300 on a future edit.
REVISION = "1d0076a55b609744218081ff6ea693aefe824677"
N_TOTAL = 1147
N_FORMAL = 300
SALT = "finqa-v0-strata-2026"

# The 300-item formal sample is ordered by a global salted-hash sort (not
# grouped by stratum), so the preflight subset cannot be a hardcoded list of
# positions -- it is instead computed DETERMINISTICALLY from the final 300
# (see `preflight_indices_for`): the first N items of every stratum, in the
# frozen 300's own order, proportional to that stratum's share of 30, so the
# preflight covers the same strata the formal 300 does without drawing a
# second, potentially inconsistent sample. This still yields a fixed,
# reproducible list of `sample_id`s once the 300 are frozen (see the
# `preflight_indices` field the manifest records below).
N_PREFLIGHT = 30

UNIT_MULT = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def salted_key(salt: str, s: str) -> str:
    return hashlib.sha256(f"{salt}:{s}".encode("utf-8")).hexdigest()


def normalize_finqa_answer(raw: str):
    """Return a float, or None if `raw` is not a numeric FinQA answer.

    Handles: leading `$`, trailing `%` (returned as the percentage NUMBER,
    e.g. "14%" -> 14.0, matching how FinQA gold percentages are written and
    how the model is asked to answer), thousands commas, a trailing stray
    comma (`-31.47,`), and a `million/billion/thousand/trillion` unit suffix
    which multiplies the numeric part. Returns None for empty/yes/no/prose/
    ratio/multi-value strings -- i.e. anything that does not reduce to one
    unambiguous number.
    """
    if raw is None:
        return None
    s = raw.strip()
    if s == "" or s.lower() in ("yes", "no"):
        return None
    is_pct = s.endswith("%")
    s2 = s[:-1].strip() if is_pct else s
    s2 = re.sub(r"^\$\s*", "", s2).strip()
    s2 = s2.replace(",", "")
    s2 = s2.rstrip(".")  # "10 ." style trailing punctuation, if any survives
    mult = 1.0
    m = re.search(r"\b(thousand|million|billion|trillion)\b", s2, flags=re.I)
    if m:
        mult = UNIT_MULT[m.group(1).lower()]
        s2 = (s2[: m.start()] + s2[m.end():]).strip()
    s2 = s2.strip()
    # reject anything with leftover non-numeric content (ratios, "or", extra
    # words, multiple numbers) -- a clean numeric core must fully match.
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s2):
        return None
    try:
        val = float(s2) * mult
    except ValueError:
        return None
    return val  # percentages are returned as the stated number, e.g. 14.0


def count_program_ops(program_re: str) -> int:
    """Approximate reasoning-step count: number of top-level function calls
    in program_re, e.g. "divide(subtract(5829,5735),5735)" -> 2 (divide,
    subtract). Counts every `<name>(` token; nesting still counts each call
    once, which is the intended "how many operations" proxy, not AST depth.
    """
    if not program_re:
        return 0
    return len(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*\(", program_re))


def step_bucket(n_ops: int) -> str:
    if n_ops <= 1:
        return "steps_1"
    if n_ops == 2:
        return "steps_2"
    return "steps_3plus"


def table_bucket(n_rows: int) -> str:
    if n_rows <= 6:
        return "table_small"
    if n_rows <= 12:
        return "table_medium"
    return "table_large"


def linearize_table(table):
    """Render a FinQA table (list[list[str]], first row = header) as clear
    lines: "<row_label> | <col header>: <cell>; <col header>: <cell>; ...".
    FinQA's first column is usually the row label and first row the header;
    both are used verbatim. No numeric reformatting -- the raw cell strings
    (which may already carry "-32 ( 32 )" style negatives) are passed through
    so the model sees exactly what is in the source filing.
    """
    if not table:
        return "(no table)"
    header = table[0]
    lines = []
    for row in table[1:]:
        label = row[0] if row else ""
        parts = []
        for h, v in zip(header[1:], row[1:]):
            h = h.strip() if h else ""
            v = v.strip() if v else ""
            if h:
                parts.append(f"{h}: {v}")
            else:
                parts.append(v)
        lines.append(f"{label} | " + "; ".join(parts))
    return "\n".join(lines)


def load_all_rows():
    from datasets import load_dataset
    ds = load_dataset(HF_NAME, CONFIG, split=SPLIT, revision=REVISION,
                       trust_remote_code=True)
    return list(ds)


def build_pool(rows):
    """Return (eligible, exclusions) where exclusions is a Counter of reasons."""
    eligible, exclusions = [], Counter()
    for r in rows:
        val = normalize_finqa_answer(r["answer"])
        if val is None:
            a = (r["answer"] or "").strip()
            if a == "":
                exclusions["empty"] += 1
            elif a.lower() in ("yes", "no"):
                exclusions["yes_no"] += 1
            else:
                exclusions["unconvertible_text"] += 1
            continue
        eligible.append((r, val))
    return eligible, exclusions


def stratify_and_sample(eligible, n_formal=N_FORMAL, salt=SALT):
    """Deterministic proportional-by-stratum sample, ranked by salted hash."""
    strata = defaultdict(list)
    for r, val in eligible:
        sb = step_bucket(count_program_ops(r.get("program_re", "")))
        tb = table_bucket(len(r.get("table", [])))
        strata[(sb, tb)].append((r, val))

    for k in strata:
        strata[k].sort(key=lambda rv: salted_key(salt, rv[0]["id"]))

    total = len(eligible)
    raw_alloc = {k: len(v) / total * n_formal for k, v in strata.items()}
    alloc = {k: int(raw_alloc[k]) for k in strata}
    remainder = n_formal - sum(alloc.values())
    # largest-remainder method, tie-broken by stratum key for determinism
    order = sorted(strata.keys(), key=lambda k: (-(raw_alloc[k] - alloc[k]), k))
    for k in order[:remainder]:
        alloc[k] += 1
    # clip to stratum size (should not trigger at the pinned revision's counts)
    for k in alloc:
        alloc[k] = min(alloc[k], len(strata[k]))

    picked = []
    for k in sorted(strata.keys()):
        picked.extend(strata[k][: alloc[k]])
    # global deterministic order across strata: by salted hash of id again,
    # so the final 300-item order does not depend on stratum iteration order
    picked.sort(key=lambda rv: salted_key(salt, rv[0]["id"]))
    return picked, {("_".join(k)): alloc[k] for k in sorted(strata)}, \
        {("_".join(k)): len(v) for k, v in sorted(strata.items())}


def preflight_indices_for(data, n_preflight=N_PREFLIGHT):
    """Deterministic subset of `data` (already frozen, in its final order)
    covering every populated stratum. Allocation is proportional (largest
    remainder, same method as the 300-item stratified sample) to each
    stratum's share of the 300; within a stratum the first items IN THE
    FROZEN 300'S OWN ORDER are taken, so this is a pure function of the
    already-written data array, not a second random draw.
    """
    by_stratum = defaultdict(list)
    for d in data:
        sb = step_bucket(d["n_program_ops"])
        tb = table_bucket(d["n_table_rows"])
        by_stratum[(sb, tb)].append(d["sample_id"])

    total = len(data)
    raw_alloc = {k: len(v) / total * n_preflight for k, v in by_stratum.items()}
    alloc = {k: max(1, int(raw_alloc[k])) if raw_alloc[k] > 0 else 0
             for k in by_stratum}
    # shrink/grow to hit n_preflight exactly, tie-broken by stratum key
    diff = n_preflight - sum(alloc.values())
    order = sorted(by_stratum.keys(), key=lambda k: (-(raw_alloc[k] - int(raw_alloc[k])), k))
    i = 0
    while diff > 0:
        k = order[i % len(order)]
        if alloc[k] < len(by_stratum[k]):
            alloc[k] += 1
            diff -= 1
        i += 1
        if i > 10000:
            break
    while diff < 0:
        k = order[-(i % len(order)) - 1]
        if alloc[k] > 0:
            alloc[k] -= 1
            diff += 1
        i += 1
        if i > 10000:
            break
    for k in alloc:
        alloc[k] = min(alloc[k], len(by_stratum[k]))

    picked = []
    for k in sorted(by_stratum.keys()):
        picked.extend(by_stratum[k][: alloc[k]])
    picked.sort()  # ascending sample_id, deterministic and easy to read off
    return picked


def build_dataset():
    rows = load_all_rows()
    eligible, exclusions = build_pool(rows)
    picked, alloc, pool_sizes = stratify_and_sample(eligible)

    data = []
    for i, (r, gold_val) in enumerate(picked):
        table = r.get("table", [])
        data.append({
            "sample_id": i,
            "id": r["id"],
            "question": r["question"],
            "pre_text": " ".join(r.get("pre_text", []) or []),
            "post_text": " ".join(r.get("post_text", []) or []),
            "table_linear": linearize_table(table),
            "n_table_rows": len(table),
            "n_program_ops": count_program_ops(r.get("program_re", "")),
            "gold_raw": r["answer"],
            "gold_value": gold_val,
        })

    q_digest = sha256_short(json.dumps(
        [d["question"] for d in data], ensure_ascii=False))
    id_digest = sha256_short(json.dumps([d["id"] for d in data]))
    gold_digest = sha256_short(json.dumps([d["gold_raw"] for d in data]))
    # FULL prompt-relevant content, not just the question: table and
    # pre/post report text also feed the prompt, and questions_sha256_16
    # alone would not notice a table or report-text change while the
    # question wording stayed identical. Hashed per item then joined so a
    # single differing item is still detectable (not swamped by 299 others).
    prompt_digest = sha256_short(json.dumps(
        [[d["question"], d["pre_text"], d["post_text"], d["table_linear"]]
         for d in data], ensure_ascii=False))
    preflight_ids = preflight_indices_for(data)

    manifest = {
        "protocol": PROTOCOL,
        "hf_dataset": HF_NAME, "config": CONFIG, "split": SPLIT,
        "revision": REVISION,
        "n_total_rows": len(rows),
        "n_eligible": len(eligible),
        "n_excluded": sum(exclusions.values()),
        "exclusion_breakdown": dict(exclusions),
        "n_formal": len(data),
        "strata_pool_sizes": pool_sizes,
        "strata_allocation": alloc,
        "salt": SALT,
        "questions_sha256_16": q_digest,
        "ids_sha256_16": id_digest,
        "gold_sha256_16": gold_digest,
        "prompt_content_sha256_16": prompt_digest,
        "preflight_indices": preflight_ids,
        "n_preflight": len(preflight_ids),
        "note": ("Numeric eligibility keeps $/%%/comma/unit-suffixed answers "
                 "(normalize_finqa_answer); excludes only empty, yes/no, and "
                 "text that does not reduce to one number. The 30-item "
                 "preflight is a fixed SUBSET of these 300 indices, not a "
                 "separate draw. prompt_content_sha256_16 covers question + "
                 "pre_text + post_text + table_linear, so a table/report-text "
                 "drift is caught even if question wording is unchanged; "
                 "questions_sha256_16 is kept for compatibility with the "
                 "convention used by the other tasks in this repo."),
    }
    return data, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="components/benchmark")
    ap.add_argument("--check", action="store_true",
                     help="rebuild and print digests only, do not write")
    args = ap.parse_args()

    data, manifest = build_dataset()

    print(f"[finqa] revision={manifest['revision']}")
    print(f"[finqa] total={manifest['n_total_rows']} "
          f"eligible={manifest['n_eligible']} excluded={manifest['n_excluded']} "
          f"{manifest['exclusion_breakdown']}")
    print(f"[finqa] formal n={manifest['n_formal']}  "
          f"questions_sha256_16={manifest['questions_sha256_16']}  "
          f"ids_sha256_16={manifest['ids_sha256_16']}  "
          f"gold_sha256_16={manifest['gold_sha256_16']}  "
          f"prompt_content_sha256_16={manifest['prompt_content_sha256_16']}")
    print(f"[finqa] strata pool sizes: {manifest['strata_pool_sizes']}")
    print(f"[finqa] strata allocation (of 300): {manifest['strata_allocation']}")

    if args.check:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    formal_path = os.path.join(args.out_dir, "finqa_formal.json")
    manifest_path = os.path.join(args.out_dir, "finqa_manifest.json")
    if os.path.exists(formal_path) or os.path.exists(manifest_path):
        sys.exit(f"[FATAL] output already exists ({formal_path} / "
                  f"{manifest_path}); refusing to overwrite")
    json.dump({"meta": manifest, "data": data},
               open(formal_path, "w", encoding="utf-8"),
               indent=2, ensure_ascii=False)
    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"),
               indent=2, ensure_ascii=False)
    print(f"[finqa] wrote {formal_path}")
    print(f"[finqa] wrote {manifest_path}")


if __name__ == "__main__":
    main()
