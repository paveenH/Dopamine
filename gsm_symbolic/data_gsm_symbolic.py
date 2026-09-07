#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_gsm_symbolic.py — loader for apple/GSM-Symbolic (test split, all 3 official
configs: main, p1, p2).

GSM-Symbolic is a numeric/template perturbation of GSM8K: same underlying
reasoning templates, re-instantiated with different symbolic values (and, for
p1/p2, one/two extra clauses inserted, raising difficulty). This loader is a
standalone read-only step -- no other GSM8K/MATH loader or runner is touched.

Output schema per item (JSON list), one file per config:
  {
    "id": <int>,               # OFFICIAL HF dataset row `id` field (row["id"]),
                                # NOT a re-enumerated local index
    "sample_id": <str>,        # "{config}:{id}", unique across the 3 configs
                                # combined (id alone repeats across configs)
    "instance": <int>,         # instantiation index within original_id
    "original_id": <int>,      # links back to the shared GSM8K-derived template
    "config": "main"|"p1"|"p2",
    "question": <str>,
    "answer": <str>,           # gold numeric answer (extracted from the
                                # official '#### <number>' solution field,
                                # via the SAME regex utils.extract_gsm8k_answer
                                # would apply -- kept local so this loader has
                                # no torch/heavy import)
    "solution": <str>,         # full solution text as shipped by HF
  }

FULL OFFICIAL TEST SPLIT PER CONFIG -- NOT a fixed 300-item sample. Measured
sizes (2026-09, Hub dataset-viewer): main ~1319, p1 ~5000, p2 ~2500 rows, so
the formal sweep is ~8819 items x 4 alphas x 2 models = ~70,552 generations,
NOT 300 x 8. This script only downloads + reformats + writes one JSON per
config (full split), plus (deterministically) a separate small preflight
subset for the pre-approval check.
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset

CONFIGS = ["main", "p1", "p2"]

# NOT hand-pinned to a guessed commit SHA -- I have no verified way to read
# the exact current HEAD commit from this environment, and writing a wrong
# SHA would hard-fail every load. Instead: resolve at runtime (whatever
# `datasets` picks, normally the dataset repo's current main), then READ BACK
# the actual resolved commit hash from the loaded dataset object and record it
# in every output file's meta (see `resolve_revision` below), so drift across
# runs is at least DETECTABLE even though it is not pre-pinned. If a true pin
# is needed later, set REVISION to a verified full 40-hex commit SHA here.
REVISION = None


def extract_gold(solution: str) -> str:
    """Same convention as utils.extract_gsm8k_answer's primary branch: the
    official '#### <number>' marker. GSM-Symbolic solutions carry this
    verbatim (inherited from GSM8K format)."""
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", solution)
    if m:
        return m.group(1).replace(",", "")
    return ""


def resolve_revision(ds) -> str:
    """Best-effort read-back of the ACTUAL commit the loaded dataset came
    from, so meta records what really got pulled rather than the (possibly
    unset) request. `datasets` exposes this via the dataset's `_fingerprint`-
    adjacent download_checksums / info, but the one stable, documented place
    is `huggingface_hub`'s own resolution -- fall back through a few
    attributes rather than assuming one exists across `datasets` versions."""
    for attr_path in (
        lambda: ds.info.download_checksums,
        lambda: ds.builder_name,
    ):
        try:
            v = attr_path()
            if v:
                return str(v) if not isinstance(v, dict) else json.dumps(v)[:200]
        except Exception:
            continue
    # Last resort: ask the Hub API directly for the current HEAD of `main`.
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info("apple/GSM-Symbolic", revision=REVISION or "main")
        return info.sha
    except Exception as e:
        return f"UNRESOLVED ({type(e).__name__}: {e})"


def load_config(config: str, cache_dir: str):
    ds = load_dataset(
        "apple/GSM-Symbolic",
        config,
        split="test",
        cache_dir=cache_dir,
        revision=REVISION,
    )
    resolved_revision = resolve_revision(ds)
    items = []
    for row in ds:
        gold = extract_gold(row["answer"])
        official_id = row["id"]
        items.append({
            "id": official_id,                       # official HF row id, verbatim
            "sample_id": f"{config}:{official_id}",   # unique across configs combined
            "instance": row.get("instance", None),
            "original_id": row.get("original_id", None),
            "config": config,
            "question": row["question"],
            "answer": gold,
            "solution": row["answer"],
        })
    missing_gold = sum(1 for it in items if it["answer"] == "")
    return items, missing_gold, resolved_revision


def preflight_subset(items: list, n: int = 10, seed_field: str = "original_id"):
    """Deterministic 10-item subset per config: sort by (original_id, instance,
    id) and take an evenly spaced stride so distinct original_id/instance
    values are covered rather than clustering on the first few rows."""
    ordered = sorted(items, key=lambda x: (x.get(seed_field) if x.get(seed_field) is not None else x["id"], x.get("instance") or 0, x["id"]))
    if len(ordered) <= n:
        return ordered
    stride = len(ordered) / n
    picked = []
    seen_ids = set()
    for k in range(n):
        idx = int(k * stride)
        while idx in seen_ids and idx < len(ordered) - 1:
            idx += 1
        seen_ids.add(idx)
        picked.append(ordered[idx])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="components/benchmark/gsm_symbolic")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--preflight_n", type=int, default=10)
    ap.add_argument("--check", action="store_true",
                     help="Load + validate schema only, print counts, write nothing.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not args.check:
        out_dir.mkdir(parents=True, exist_ok=True)

    preflight_all = []
    summary = {}
    resolved_revisions = {}
    for config in CONFIGS:
        # No try/except here deliberately: if a config (in particular "main",
        # whose split was not listed by the Hub dataset-viewer's config table
        # as of this writing) fails to load, this must HARD STOP with the
        # real traceback -- never silently skip a config or drop it from the
        # experiment matrix.
        items, missing, resolved_revision = load_config(config, args.cache_dir)
        resolved_revisions[config] = resolved_revision
        n = len(items)
        n_original = len({it["original_id"] for it in items})
        n_instance = len({it["instance"] for it in items})
        summary[config] = {
            "n_items": n,
            "n_unique_original_id": n_original,
            "n_unique_instance": n_instance,
            "n_missing_gold": missing,
        }
        print(f"[{config}] n={n}  unique original_id={n_original}  "
              f"unique instance={n_instance}  missing_gold={missing}")
        if missing:
            print(f"  [WARN] {missing} items in {config} have no parseable gold "
                  f"'#### <number>' marker -- inspect before using this config.")

        if args.check:
            continue

        out_path = out_dir / f"gsm_symbolic_{config}_test.json"
        if out_path.exists():
            raise FileExistsError(
                f"{out_path} already exists -- refusing to overwrite. "
                "Delete it deliberately if you intend to regenerate."
            )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "dataset": "apple/GSM-Symbolic",
                    "config": config,
                    "split": "test",
                    "revision_requested": REVISION,
                    "revision_resolved": resolved_revision,
                    "n_items": n,
                },
                "data": items,
            }, f, ensure_ascii=False, indent=2)
        print(f"  -> wrote {out_path}")

        pf = preflight_subset(items, n=args.preflight_n)
        for it in pf:
            preflight_all.append(it)

    if args.check:
        print("\n[check] OK -- schema validated for all 3 configs, nothing written.")
        return

    pf_path = out_dir / "gsm_symbolic_preflight_30.json"
    if pf_path.exists():
        raise FileExistsError(f"{pf_path} already exists -- refusing to overwrite.")
    with open(pf_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "dataset": "apple/GSM-Symbolic",
                "split": "test",
                "revision_requested": REVISION,
                "revision_resolved_by_config": resolved_revisions,
                "note": "10 items per config (main/p1/p2), deterministic stride "
                        "sample by (original_id, instance, id); NOT a random "
                        "sample of the formal run. The formal run uses the "
                        "FULL official test split per config, not a fixed "
                        "300-item sample.",
                "configs": CONFIGS,
                "n_per_config": args.preflight_n,
            },
            "data": preflight_all,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n-> wrote preflight subset ({len(preflight_all)} items) -> {pf_path}")
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
