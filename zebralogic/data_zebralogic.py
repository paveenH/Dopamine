#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZebraLogic-Easy loader for the zebralogic-easy-v0 four-point workpoint
exploration. Protocol: docs/PREREG_ZEBRALOGIC_EASY.md.

TWO DATASETS, TWO ROLES, and the split is the point (same firewall pattern as
data_gsm_hard.py / data_bbh_numeric.py / data_logiqa2.py):

    allenai/ZebraLogicBench            PUBLIC.  grid_mode/test, 1000 rows.
                                        Verified: EVERY row's `solution` field
                                        is entirely "___" placeholders -- this
                                        is the official leaderboard submission
                                        split, gold intentionally withheld.
                                        Used ONLY for `puzzle` text, `id`, and
                                        the SHAPE of `solution` (header +
                                        house count) needed to build the
                                        per-item JSON template -- never for
                                        any cell VALUE, because there isn't
                                        one to leak.

    allenai/ZebraLogicBench-private    GATED. grid_mode/test, same 1000 ids,
                                        real solution cell values. Requires a
                                        human to accept HF's per-repo access
                                        request AND a valid HF_TOKEN for that
                                        account. This loader NEVER attempts a
                                        silent fallback if that access is
                                        missing -- see load_private_gold().

EASY TIER = the OFFICIAL WildEval/ZeroEval definition, verbatim:
    ['2*2','2*3','2*4','2*5','2*6','3*2','3*3']
(zebra_grid_eval.py, `easy_sizes`). All 40 rows of each of those seven sizes
in the public split -- 280 items total, no further sampling, no salt: the
"easy" tier already IS this exact id set within the frozen revision, so what
must be frozen is the SET of ids, not a selection procedure. `--check` re-runs
every assertion and reprints the frozen id-set digest; it downloads the public
split (network required) but never the private one.

Revision pinned to a full 40-hex commit SHA (never a branch name -- a branch
follows upstream exactly as an unset revision would).

This script downloads data. It runs no model and computes no accuracy.

Usage
-----
    python data_zebralogic.py --out_dir components/benchmark   # writes files
    python data_zebralogic.py --check                          # no write
"""

import argparse
import hashlib
import json
import os
import sys

PROTOCOL = "zebralogic-easy-v0"
HF_PUBLIC = "allenai/ZebraLogicBench"
HF_PRIVATE = "allenai/ZebraLogicBench-private"
CONFIG = "grid_mode"
SPLIT = "test"
# HEAD of allenai/ZebraLogicBench `main`, verified 2026-09-03.
REVISION = "2f94a445d7079f20146f5443e2606049de8543e0"
N_FULL = 1000
N_PER_SIZE = 40

# The OFFICIAL easy tier, copied verbatim from WildEval/ZeroEval
# `src/evaluation/zebra_grid_eval.py`'s `easy_sizes` list. Do not edit.
EASY_SIZES = ("2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3")
N_EASY = len(EASY_SIZES) * N_PER_SIZE  # 280

# Measured at freeze (2026-09-03/04) at the pinned REVISION above, over the
# easy-tier item set in the frozen deterministic order this loader produces
# (sorted by (EASY_SIZES rank, id)). Asserted, not assumed -- an upstream
# edit to the dataset content (without a revision bump, or a different
# subset resolving under the same call) would otherwise produce a
# plausible-looking sample built from different puzzles or a different id
# set, matching data_bbh_numeric.py's EXPECTED convention.
EXPECTED_EASY_IDS_SHA256 = "b22536887230294f"
EXPECTED_EASY_PUZZLES_SHA256 = "dd972dff8d36c923"

# The full official difficulty partition (same source), printed by --check
# and by the launcher's distribution-confirmation gate -- descriptive only,
# never used to select items beyond EASY_SIZES.
OFFICIAL_HARD_SIZES = (
    "3*4", "3*5", "4*2", "3*6", "4*3", "4*4", "5*2", "6*2",
    "4*5", "4*6", "5*3", "5*4", "5*5", "5*6", "6*3", "6*4", "6*5", "6*6",
)
OFFICIAL_SMALL_SIZES = ("2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2")
OFFICIAL_MEDIUM_SIZES = ("3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2")
OFFICIAL_LARGE_SIZES = ("4*5", "5*3", "4*6", "5*4", "6*3")
OFFICIAL_XL_SIZES = ("5*5", "6*4", "5*6", "6*5", "6*6")

# a blind (public) record may contain ONLY these keys ...
BLIND_ALLOWED = {"id", "sample_id", "size", "puzzle", "solution_shape",
                  "content_sha256"}
# ... and must contain NONE of these (no cell values reach the blind file)
LABEL_FIELDS = {"solution", "answer", "gold", "gold_answer", "correct",
                 "accuracy", "target"}


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


def sha256_full(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def solution_shape(solution: dict) -> dict:
    """The label-free SHAPE of a solution scaffold: header + row count, never
    cell values. This is all apply_lgp_grid_template needs to build the
    per-item JSON answer template, and it is identical whether read from the
    public (blank) or private (filled) dataset -- both ship the same header
    and house count for a given id."""
    return {"header": list(solution["header"]), "n_rows": len(solution["rows"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", default=HF_PUBLIC)
    ap.add_argument("--revision", default=REVISION, type=_commit_sha)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--out_dir", default="components/benchmark")
    ap.add_argument("--prefix", default="zebralogic_easy")
    ap.add_argument("--check", action="store_true",
                    help="re-run every assertion and print the digests; write nothing")
    a = ap.parse_args()

    if not a.check and not os.path.isdir(a.out_dir):
        sys.exit(f"FAIL: --out_dir {a.out_dir!r} does not exist. On the server "
                 "run from /data1/paveen/Dopamine so it resolves to "
                 "components/benchmark/, or pass --out_dir explicitly.")

    from datasets import load_dataset
    ds = load_dataset(a.hf_name, CONFIG, split=a.split, revision=a.revision)
    print(f"loaded {a.hf_name}:{CONFIG} split={a.split} revision={a.revision} n={len(ds)}")

    # ---- HARD STOPs on schema and shape.
    required = {"id", "size", "puzzle", "solution"}
    missing = required - set(ds.column_names)
    if missing:
        sys.exit(f"HARD STOP: expected columns {sorted(required)} at this "
                 f"revision, missing {sorted(missing)}; found {ds.column_names}.")
    if len(ds) != N_FULL:
        sys.exit(f"HARD STOP: expected {N_FULL} rows, got {len(ds)}. The split "
                 "changed upstream; re-freeze deliberately rather than adapting.")

    rows = list(ds)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != N_FULL:
        sys.exit(f"HARD STOP: ids are not unique ({len(set(ids))}/{N_FULL}).")

    # ---- verify EVERY row's solution is blank (this is the whole reason the
    # private dataset exists). A single non-blank row would mean the public
    # split changed upstream and might now carry real gold -- worth a hard
    # stop and a deliberate re-freeze, not a silent "great, less friction".
    non_blank = [r["id"] for r in rows
                 if any(c != "___" for row in r["solution"]["rows"] for c in row)]
    if non_blank:
        sys.exit(f"HARD STOP: {len(non_blank)} row(s) have a NON-BLANK public "
                 f"solution (e.g. {non_blank[:3]}) -- the public split's "
                 "gold-withheld property no longer holds as assumed. This "
                 "changes the private-gold-required design; do not proceed "
                 "without re-reading docs/PREREG_ZEBRALOGIC_EASY.md.")

    by_size = {}
    for r in rows:
        by_size.setdefault(r["size"], []).append(r)
    got_easy_sizes = tuple(sorted(s for s in by_size if s in EASY_SIZES))
    if sorted(by_size) != sorted(set(EASY_SIZES) | set(OFFICIAL_HARD_SIZES)):
        sys.exit(f"HARD STOP: size set changed upstream. Got {sorted(by_size)}, "
                 f"expected easy+hard = {sorted(set(EASY_SIZES) | set(OFFICIAL_HARD_SIZES))}.")
    for sz in EASY_SIZES:
        n = len(by_size.get(sz, []))
        if n != N_PER_SIZE:
            sys.exit(f"HARD STOP: size {sz!r} has {n} rows, expected {N_PER_SIZE}.")

    easy_rows = [r for r in rows if r["size"] in EASY_SIZES]
    if len(easy_rows) != N_EASY:
        sys.exit(f"HARD STOP: easy-tier row count {len(easy_rows)} != {N_EASY}.")

    # Frozen, deterministic item ORDER: sorted by (size in EASY_SIZES order,
    # then id) so re-running this loader always reproduces the same order,
    # independent of the dataset's on-disk row order.
    size_rank = {s: i for i, s in enumerate(EASY_SIZES)}
    easy_rows.sort(key=lambda r: (size_rank[r["size"]], r["id"]))

    easy_ids = [r["id"] for r in easy_rows]
    id_digest = sha16("\n".join(easy_ids))
    puzzle_digest = sha16("\n".join(r["puzzle"] for r in easy_rows))

    for name, got, want in (
        ("easy_ids_sha256", id_digest, EXPECTED_EASY_IDS_SHA256),
        ("easy_puzzles_sha256", puzzle_digest, EXPECTED_EASY_PUZZLES_SHA256),
    ):
        if got != want:
            sys.exit(f"HARD STOP: {name} is {got}, frozen value is {want}. "
                     "The easy-tier content or id set changed upstream (or "
                     "under this pinned revision) since this loader was "
                     "frozen; re-freeze deliberately rather than accepting a "
                     "different sample under this name.")

    print(f"\n[data properties -- DESCRIPTIVE]")
    print(f"  easy-tier sizes    : {list(EASY_SIZES)}  (official WildEval/ZeroEval definition)")
    print(f"  easy-tier n        : {N_EASY}  ({N_PER_SIZE} per size x {len(EASY_SIZES)} sizes)")
    print(f"  easy_ids_sha256[:16]    {id_digest}")
    print(f"  easy_puzzles_sha256[:16] {puzzle_digest}")

    print(f"\n[official ZeroEval full difficulty partition, for the "
          f"launcher's confirmation gate -- descriptive, not used to select "
          f"items beyond EASY_SIZES]")
    for name, sizes in (("easy", EASY_SIZES), ("hard", OFFICIAL_HARD_SIZES),
                        ("small", OFFICIAL_SMALL_SIZES), ("medium", OFFICIAL_MEDIUM_SIZES),
                        ("large", OFFICIAL_LARGE_SIZES), ("xl", OFFICIAL_XL_SIZES)):
        present = [s for s in sizes if s in by_size]
        n = sum(len(by_size[s]) for s in present)
        print(f"  {name:7s} sizes={list(sizes)!s:55s} n_present={n}")

    meta = {
        "protocol": PROTOCOL, "hf_public": HF_PUBLIC, "hf_private": HF_PRIVATE,
        "config": CONFIG, "revision": a.revision, "split": a.split,
        "n_full": N_FULL, "n_per_size": N_PER_SIZE,
        "easy_sizes": list(EASY_SIZES), "n_easy": N_EASY,
        "easy_ids_sha256": id_digest, "easy_puzzles_sha256": puzzle_digest,
        "easy_tier_source": ("verbatim WildEval/ZeroEval zebra_grid_eval.py "
                             "`easy_sizes` list -- not independently chosen"),
        "public_solution_verified_blank": True,
        "official_hard_sizes": list(OFFICIAL_HARD_SIZES),
        "official_small_sizes": list(OFFICIAL_SMALL_SIZES),
        "official_medium_sizes": list(OFFICIAL_MEDIUM_SIZES),
        "official_large_sizes": list(OFFICIAL_LARGE_SIZES),
        "official_xl_sizes": list(OFFICIAL_XL_SIZES),
        "gold_source": HF_PRIVATE,
        "gold_note": ("Real gold is in the GATED allenai/ZebraLogicBench-private "
                      "dataset (grid_mode/test, same ids), requiring per-account "
                      "HF consent + a valid HF_TOKEN. Never substituted."),
        "blind_validation": False,
        "blind_note": ("Not a blind protocol in the P3 sense -- this is ordinary "
                       "gated dataset access, not a designed seal."),
    }

    if a.check:
        print(f"\n[--check] all assertions passed; nothing written.")
        return

    # ---- blind copy: whitelist-built (shape only, no cell values), then
    # asserted to carry no label field. sample_id is 0..279 in the frozen
    # order above -- this is what the runner and scorer both index by.
    blind = []
    for i, r in enumerate(easy_rows):
        rec = {"id": r["id"], "sample_id": i, "size": r["size"],
               "puzzle": r["puzzle"], "solution_shape": solution_shape(r["solution"]),
               "content_sha256": sha16(r["puzzle"])}
        extra = set(rec) - BLIND_ALLOWED
        if extra:
            sys.exit(f"FAIL: blind record carries non-whitelisted key(s) {sorted(extra)}")
        leaked = {k for k in rec if k.lower() in LABEL_FIELDS}
        if leaked:
            sys.exit(f"FAIL: label field {sorted(leaked)} reached the blind record")
        blind.append(rec)

    bpath = os.path.join(a.out_dir, f"{a.prefix}_blind.json")
    mpath = os.path.join(a.out_dir, f"{a.prefix}_manifest.json")
    for p in (bpath, mpath):
        if os.path.exists(p):
            sys.exit(f"FAIL: {p} exists; refusing to overwrite a frozen artifact. "
                     "Delete it deliberately if this is a re-freeze.")

    json.dump({"meta": {**meta, "contains_labels": False}, "data": blind},
              open(bpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(meta, open(mpath, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\nblind    -> {bpath}   (no cell VALUES -- the runner reads this)")
    print(f"manifest -> {mpath}")
    print("\nNo model was run. No accuracy was computed. Real gold requires "
          "allenai/ZebraLogicBench-private access; see load_private_gold() "
          "in zebralogic/eval_zebralogic.py.")


def load_private_gold(ids, hf_name=HF_PRIVATE, revision=REVISION, split=SPLIT,
                      config=CONFIG):
    """Load real gold for the given `ids` from the GATED private dataset.

    Hard-stops with an explanatory message (never a silent fallback) if:
      - the `datasets` library cannot authenticate/access the private repo
        (missing/invalid HF_TOKEN, or the account has not been granted
        access -- both surface as a `datasets`/`huggingface_hub` exception,
        which is caught here and re-raised as a clear, actionable message);
      - any requested id is missing from the private dataset;
      - a returned solution's shape (header, house count) disagrees with what
        the public split reported for the same id -- would mean the two
        datasets have drifted out of sync with each other.

    Returns {id: solution_dict} for exactly the requested ids -- solution_dict
    has the real "header"/"rows" (rows containing actual cell values, not
    "___").
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        sys.exit(f"FAIL: `datasets` library not importable: {e}")

    try:
        priv = load_dataset(hf_name, config, split=split, revision=revision)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any auth/
        # access failure from `datasets`/`huggingface_hub` must hard-stop with
        # a clear message, not propagate an opaque stack trace or, worse, be
        # caught somewhere upstream and silently treated as "no gold".
        sys.exit(
            f"FAIL: could not load the PRIVATE gold dataset {hf_name!r} "
            f"(config={config!r}, split={split!r}, revision={revision!r}).\n"
            f"  Underlying error: {type(e).__name__}: {e}\n\n"
            f"  This dataset is GATED. To fix:\n"
            f"    1. Visit https://huggingface.co/datasets/{hf_name} while "
            f"logged in and click through the access request.\n"
            f"    2. Set a valid HF_TOKEN environment variable for that same "
            f"account on the machine running this scorer "
            f"(or `huggingface-cli login`).\n"
            f"  This script does NOT substitute another dataset, a hand-built "
            f"solver, or a relaxed parser when this fails -- see "
            f"docs/PREREG_ZEBRALOGIC_EASY.md section 9.")

    by_id = {r["id"]: r["solution"] for r in priv}
    ids = list(ids)
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f"FAIL: {len(missing)} id(s) missing from the private gold "
                 f"dataset, e.g. {missing[:5]}. The public and private splits "
                 "have drifted out of sync.")
    return {i: by_id[i] for i in ids}


if __name__ == "__main__":
    main()
