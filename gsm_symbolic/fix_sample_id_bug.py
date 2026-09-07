#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_sample_id_bug.py — ONE-TIME offline repair for the sample_id collision
bug (2026-09).

Root cause: GSM-Symbolic's HF `id` field equals `original_id` (a CLUSTER id,
not a per-row id), so the original `sample_id = f"{config}:{id}"` collided
across every instance sharing one original_id. Generation itself was NOT
affected -- every distinct `question` really was sent to the model and a
real answer generated -- but `sample_id`-keyed downstream analysis
(score_cell, check_cell_consistency in eval_gsm_symbolic.py) silently kept
only ONE of several rows per collision, corrupting every statistic computed
so far.

This script repairs ALREADY-GENERATED result JSON files IN PLACE: for each
row in `data`, recompute `sample_id = f"{config}:{original_id}:{instance}"`
from the row's own `original_id`/`instance` fields (already present and
correct -- only the KEY built from them was wrong), and rewrite both
`data[i].sample_id` and `meta.sample_ids`. No model output, no `question`, no
`generated`, no `answer`/`pred_answer`/`correct` field is touched -- this is
a pure key-recomputation, not a re-score.

Safety:
  - Refuses to touch a file unless every row already has `original_id` and
    `instance` (both must be present -- if either is missing there is
    nothing to safely recompute from).
  - Verifies the row count is unchanged before/after.
  - Verifies the recomputed sample_id values ARE unique per file (the same
    hard check data_gsm_symbolic.assert_unique_sample_ids now enforces at
    generation time); refuses to write if the new keys still collide.
  - Writes a `.bak` backup alongside each file before overwriting, so the
    (corrupted-key, but otherwise complete) original is never destroyed.
  - `--dry_run` (default) only reports what would change; `--apply` writes.

Usage:
  python fix_sample_id_bug.py --root /path/to/RoleAnswer            # report only (default)
  python fix_sample_id_bug.py --root /path/to/RoleAnswer --apply    # actually write
"""

import argparse
import glob
import json
import os
import shutil
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "gsm_symbolic"))
from data_gsm_symbolic import assert_unique_sample_ids  # noqa: E402


def find_cell_files(root: str):
    """Find every gsm_symbolic result JSON under root/{model}/gsm_symbolic/
    (or the answer_gsm_symbolic* tree if that's what's on disk), EXCLUDING
    anything under a 'preflight' directory (preflight is out of scope here --
    fix it separately if needed, it's small enough to just re-run)."""
    patterns = [
        os.path.join(root, "*", "gsm_symbolic", "*", "mdf_*", "gsm_symbolic_*.json"),
        os.path.join(root, "*", "answer_gsm_symbolic", "*", "mdf_*", "gsm_symbolic_*.json"),
    ]
    files = []
    for pat in patterns:
        for f in glob.glob(pat):
            if "preflight" in f:
                continue
            files.append(f)
    return sorted(set(files))


def repair_file(path: str, apply: bool):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload["meta"]
    data = payload["data"]
    config = meta.get("gsm_config")
    n_before = len(data)

    missing_fields = [
        i for i, row in enumerate(data)
        if row.get("original_id") is None or row.get("instance") is None
    ]
    if missing_fields:
        raise ValueError(
            f"{path}: {len(missing_fields)} rows missing original_id/instance "
            f"(e.g. row {missing_fields[0]}) -- cannot safely recompute "
            "sample_id from data this incomplete."
        )

    old_sample_ids = [row["sample_id"] for row in data]
    n_old_unique = len(set(old_sample_ids))

    new_sample_ids = [
        f"{config}:{row['original_id']}:{row['instance']}" for row in data
    ]
    n_new_unique = len(set(new_sample_ids))

    if len(new_sample_ids) != n_before:
        raise RuntimeError(f"{path}: row count changed during recompute ({len(new_sample_ids)} != {n_before})")

    # This is the exact hard check data_gsm_symbolic.py now enforces at
    # generation time -- run it here too so a repair can never silently
    # write a file that still has collisions.
    fake_items = [{"sample_id": sid} for sid in new_sample_ids]
    assert_unique_sample_ids(fake_items, config)  # raises ValueError if not unique

    changed = old_sample_ids != new_sample_ids
    report = {
        "path": path,
        "config": config,
        "n_rows": n_before,
        "n_old_unique_sample_id": n_old_unique,
        "n_new_unique_sample_id": n_new_unique,
        "changed": changed,
    }

    if not changed:
        return report  # nothing to do, e.g. a config where id happened to be row-unique

    if apply:
        backup_path = path + ".bak"
        if os.path.exists(backup_path):
            raise FileExistsError(
                f"{backup_path} already exists -- refusing to overwrite an "
                "existing backup (this file may have already been repaired "
                "once; inspect before re-running)."
            )
        shutil.copy2(path, backup_path)

        for row, new_sid in zip(data, new_sample_ids):
            row["sample_id"] = new_sid
        meta["sample_ids"] = new_sample_ids
        meta["sample_id_repair_note"] = (
            "sample_id was recomputed 2026-09 from original_id:instance "
            "(the old key f'{config}:{id}' collided because HF's id field "
            "equals original_id). No model output, question, answer, "
            "pred_answer, or correct field was touched. See "
            "fix_sample_id_bug.py and gsm_symbolic/data_gsm_symbolic.py for "
            "the root-cause note."
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        report["backup"] = backup_path

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                     help="Directory containing {model}/gsm_symbolic/... "
                          "(e.g. RoleAnswer/, or components/ on the server)")
    ap.add_argument("--apply", action="store_true",
                     help="Actually write the repaired files (with a .bak "
                          "backup). Without this flag, only reports what "
                          "would change.")
    args = ap.parse_args()

    files = find_cell_files(args.root)
    if not files:
        print(f"[!] no gsm_symbolic result files found under {args.root}")
        return 1

    print(f"Found {len(files)} cell files.")
    n_changed = 0
    n_errors = 0
    for path in files:
        try:
            report = repair_file(path, apply=args.apply)
        except Exception as e:
            print(f"[FAIL] {path}: {type(e).__name__}: {e}")
            n_errors += 1
            continue
        status = "CHANGED" if report["changed"] else "already unique (no-op)"
        applied = " [APPLIED]" if (args.apply and report["changed"]) else ""
        print(f"  {report['path']}: n={report['n_rows']} "
              f"old_unique={report['n_old_unique_sample_id']} "
              f"new_unique={report['n_new_unique_sample_id']}  {status}{applied}")
        if report["changed"]:
            n_changed += 1

    print(f"\n{'DRY RUN -- ' if not args.apply else ''}"
          f"{n_changed}/{len(files)} files need repair, {n_errors} errors.")
    if not args.apply and n_changed:
        print("Re-run with --apply to actually write the fix (with .bak backups).")
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
