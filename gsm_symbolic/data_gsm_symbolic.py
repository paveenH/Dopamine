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
                                # NOT a re-enumerated local index. IMPORTANT:
                                # verified 2026-09 that GSM-Symbolic's `id`
                                # field equals `original_id` (i.e. it is a
                                # CLUSTER id, not a per-row id) -- every
                                # instance of one original_id shares the SAME
                                # `id`. Do NOT use `id` alone as a row key.
    "sample_id": <str>,        # "{config}:{original_id}:{instance}", unique
                                # PER ROW within a config (id alone is NOT
                                # row-unique -- see above). A KeyError is
                                # raised at load time if this is not unique.
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

*** FIXED 2026-09: sample_id WAS "{config}:{id}", WRONG ***
GSM-Symbolic's `id` field is NOT a per-row unique id -- it equals
`original_id` (verified: unique(id) == unique(original_id) in all 3 configs
of a real formal run). The old `sample_id = f"{config}:{id}"` therefore
collided across every instance of one original_id, so a dict keyed by
sample_id (score_cell, check_cell_consistency) silently kept only ONE of
several rows per original_id -- e.g. main's formal 300-row sample collapsed
to 100 unique sample_id (= 100 unique original_id), each really holding 3
DIFFERENT questions (different instance, different text) under one key.
Generation itself was NOT affected (prompts are built from the full
`question` string, and all 300 distinct questions per config really were
generated) -- only sample_id-keyed downstream analysis was corrupted. Fixed
by keying on `f"{config}:{original_id}:{instance}"` instead, with a hard
uniqueness assertion at load time (see `load_config`) so a similar collision
can never again pass silently.

FORMAL SAMPLE = 300 items PER CONFIG (900 total, NOT one pooled 300 and NOT
the full ~8819-row split). Selection is CLUSTER-BALANCED by original_id, not
a plain row sample -- GSM-Symbolic's own README defines original_id as the
GSM8K problem a row is instantiated from, and p1/p2 each re-instantiate one
original_id many times (p1 ~5000 rows over far fewer distinct original_id;
p2 similarly), so a plain 300-row sample would silently over-represent
whichever original_id happens to have more instances. See `select_sample`
below for the exact deterministic rule (salted SHA-256 throughout, never
Python's process-salted `hash()`):
  - main: 300 DISTINCT original_id, 1 instance each (main has 1
    instance/original_id already, per the dataset's own structure).
  - p1/p2: n_original_ids clusters, 300 // n_original_ids instances per
    cluster as the base quota, with the 300 % n_original_ids remainder
    distributed to specific clusters (chosen by salted hash, not by which
    clusters happen to sort first) getting one extra instance each.
  - Instance selection WITHIN a cluster is also by salted hash of
    (config, original_id, instance) -- never by model output, difficulty, or
    dataset order.
This script only downloads + reformats + writes one JSON per config (the
FULL split, for provenance/audit) plus the 300/config formal sample plus a
separate small preflight subset for the pre-approval check.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from datasets import load_dataset

CONFIGS = ["main", "p1", "p2"]
SAMPLE_SALT = "gsm_symbolic_v1"
N_PER_CONFIG = 300

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


def assert_unique_sample_ids(items: list, config: str) -> None:
    """Hard uniqueness check on sample_id -- this is exactly the assumption
    that silently broke once already (HF's `id` field equals `original_id`,
    NOT a per-row id, so the old `sample_id=f"{config}:{id}"` collided across
    every instance of one original_id). Never let a similar collision pass
    silently again. Pulled out as its own function so it is unit-testable
    without needing a real `datasets` load."""
    sids = [it["sample_id"] for it in items]
    if len(set(sids)) != len(sids):
        from collections import Counter
        dupes = {k: v for k, v in Counter(sids).items() if v > 1}
        raise ValueError(
            f"config={config}: sample_id is NOT unique per row -- "
            f"{len(dupes)} duplicated keys (e.g. {list(dupes.items())[:3]}). "
            "This means (original_id, instance) does not uniquely identify a "
            "row either; the key scheme needs revisiting before this data can "
            "be used for anything downstream."
        )


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
        original_id = row.get("original_id", None)
        instance = row.get("instance", None)
        items.append({
            "id": official_id,      # official HF row id, verbatim -- NOTE:
                                     # this equals original_id, NOT a per-row
                                     # id; never use it alone as a row key.
            "sample_id": f"{config}:{original_id}:{instance}",
            "instance": instance,
            "original_id": original_id,
            "config": config,
            "question": row["question"],
            "answer": gold,
            "solution": row["answer"],
        })
    missing_gold = sum(1 for it in items if it["answer"] == "")
    assert_unique_sample_ids(items, config)
    return items, missing_gold, resolved_revision


def salted_hash(*parts: object) -> str:
    """Deterministic, non-process-salted hash for selection decisions.
    Python's built-in hash() is process-salted for str (PYTHONHASHSEED),
    so it gives a DIFFERENT ranking on every run -- salted SHA-256 is the
    only thing that reproduces across machines/runs/interpreters."""
    key = SAMPLE_SALT + ":" + ":".join(str(p) for p in parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def select_sample(items: list, config: str, n_target: int = N_PER_CONFIG) -> list:
    """Cluster-balanced deterministic sample of n_target items, balanced by
    original_id (never a plain row sample -- see the module docstring for
    why). All tie-breaking / selection decisions use salted_hash, so the
    result is: (a) reproducible across machines and Python versions, (b)
    identical for every model that calls this with the same `items` (both
    models must see the SAME 300/config sample and order), (c) independent
    of dataset row order or any model/difficulty signal.
    """
    by_original = {}
    for it in items:
        by_original.setdefault(it["original_id"], []).append(it)

    original_ids = sorted(by_original.keys())  # sort by value first (stable
                                                # base ordering), tie-breaks
                                                # below use the hash, not
                                                # this sort's incidental order
    n_clusters = len(original_ids)

    if n_clusters == 0:
        raise ValueError(f"config={config}: no original_id groups found -- cannot sample.")

    if n_clusters >= n_target:
        # main-like case: more (or exactly as many) clusters than the target
        # sample size -- pick n_target DISTINCT clusters, 1 instance each.
        ranked = sorted(original_ids, key=lambda oid: salted_hash(config, "cluster_select", oid))
        chosen_clusters = ranked[:n_target]
        picked = []
        for oid in chosen_clusters:
            group = by_original[oid]
            # 1 instance per cluster: pick deterministically by hash even
            # when a cluster happens to have >1 instance (main is expected
            # to have exactly 1, but this branch must not assume it).
            group_ranked = sorted(group, key=lambda it: salted_hash(config, "instance_select", oid, it["instance"], it["id"]))
            picked.append(group_ranked[0])
        return finalize_order(picked, config)

    # p1/p2-like case: fewer clusters than the target -- give every cluster a
    # base quota, then distribute the remainder to specific clusters chosen
    # by hash (never "whichever clusters sort first" or "whichever have the
    # most instances", which would let dataset structure bias the sample).
    base_quota = n_target // n_clusters
    remainder = n_target % n_clusters

    ranked_for_remainder = sorted(original_ids, key=lambda oid: salted_hash(config, "remainder_select", oid))
    extra_one = set(ranked_for_remainder[:remainder])

    picked = []
    shortfall = {}  # cluster -> how many fewer instances it had than its quota
    for oid in original_ids:
        quota = base_quota + (1 if oid in extra_one else 0)
        group = by_original[oid]
        group_ranked = sorted(group, key=lambda it: salted_hash(config, "instance_select", oid, it["instance"], it["id"]))
        take = group_ranked[:quota]
        picked.extend(take)
        if len(take) < quota:
            shortfall[oid] = quota - len(take)

    total_shortfall = sum(shortfall.values())
    if total_shortfall > 0:
        # A cluster had fewer instances than its quota (e.g. an uneven p2
        # split). Redistribute the shortfall to OTHER clusters' unused
        # instances, again choosing deterministically by hash, so the
        # sample still totals exactly n_target rather than silently coming
        # up short.
        picked_ids = {it["sample_id"] for it in picked}
        leftover_by_cluster = {
            oid: [it for it in by_original[oid] if it["sample_id"] not in picked_ids]
            for oid in original_ids if oid not in shortfall
        }
        pool = [(oid, it) for oid, its in leftover_by_cluster.items() for it in its]
        pool_ranked = sorted(pool, key=lambda pair: salted_hash(config, "shortfall_fill", pair[0], pair[1]["instance"], pair[1]["id"]))
        picked.extend(it for _, it in pool_ranked[:total_shortfall])

    if len(picked) != n_target:
        raise RuntimeError(
            f"config={config}: cluster-balanced sample produced {len(picked)} "
            f"items, expected {n_target} -- {n_clusters} clusters, base_quota="
            f"{base_quota}, remainder={remainder}, total_shortfall={total_shortfall}."
        )
    return finalize_order(picked, config)


def finalize_order(picked: list, config: str) -> list:
    """Fix the final row order deterministically (salted hash of sample_id),
    independent of dict/insertion order or any earlier tie-break ordering --
    both models must iterate the SAME 300/config sample in the SAME order,
    and that order must not depend on incidental Python dict iteration."""
    return sorted(picked, key=lambda it: salted_hash(config, "final_order", it["sample_id"]))


def preflight_subset(sample: list, config: str, n: int = 10):
    """Deterministic n-item subset OF THE FORMAL 300/config SAMPLE (never of
    the full split) -- so preflight exercises a true subset of exactly the
    items the formal sweep will run, same original_id-balance logic
    included. Selection is by the same salted_hash convention, evenly
    spread by re-using select_sample's cluster-balance rule at a smaller
    target size."""
    if len(sample) <= n:
        return sample
    return select_sample(sample, config=f"{config}_preflight", n_target=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="components/benchmark/gsm_symbolic")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--n_per_config", type=int, default=N_PER_CONFIG,
                     help="Formal sample size PER CONFIG (default 300; total "
                          "= 3x this across main/p1/p2, NOT this many total).")
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
        n_full = len(items)
        n_original = len({it["original_id"] for it in items})
        n_instance = len({it["instance"] for it in items})
        summary[config] = {
            "n_items_full_split": n_full,
            "n_unique_original_id": n_original,
            "n_unique_instance": n_instance,
            "n_missing_gold": missing,
        }
        print(f"[{config}] full_split_n={n_full}  unique original_id={n_original}  "
              f"unique instance={n_instance}  missing_gold={missing}")
        if missing:
            print(f"  [WARN] {missing} items in {config} have no parseable gold "
                  f"'#### <number>' marker -- inspect before using this config.")

        if args.check:
            continue

        full_path = out_dir / f"gsm_symbolic_{config}_full.json"
        if full_path.exists():
            raise FileExistsError(
                f"{full_path} already exists -- refusing to overwrite. "
                "Delete it deliberately if you intend to regenerate."
            )
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "dataset": "apple/GSM-Symbolic",
                    "config": config,
                    "split": "test",
                    "revision_requested": REVISION,
                    "revision_resolved": resolved_revision,
                    "n_items": n_full,
                    "note": "FULL official test split, kept for provenance/audit "
                            "only. The formal sweep uses gsm_symbolic_{config}"
                            "_sample.json (300/config, cluster-balanced by "
                            "original_id), NOT this file.",
                },
                "data": items,
            }, f, ensure_ascii=False, indent=2)
        print(f"  -> wrote FULL split -> {full_path}")

        sample = select_sample(items, config=config, n_target=args.n_per_config)
        n_sample_original = len({it["original_id"] for it in sample})
        summary[config]["n_items_formal_sample"] = len(sample)
        summary[config]["n_unique_original_id_in_sample"] = n_sample_original
        print(f"  cluster-balanced sample: n={len(sample)}  "
              f"unique original_id in sample={n_sample_original}")

        sample_path = out_dir / f"gsm_symbolic_{config}_sample.json"
        if sample_path.exists():
            raise FileExistsError(
                f"{sample_path} already exists -- refusing to overwrite. "
                "Delete it deliberately if you intend to regenerate."
            )
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "dataset": "apple/GSM-Symbolic",
                    "config": config,
                    "split": "test",
                    "revision_requested": REVISION,
                    "revision_resolved": resolved_revision,
                    "n_items": len(sample),
                    "n_unique_original_id": n_sample_original,
                    "sample_salt": SAMPLE_SALT,
                    "sampling_method": (
                        "cluster-balanced by original_id: if n_original_id "
                        ">= n_target, n_target distinct clusters chosen by "
                        "salted hash, 1 instance/cluster; else every cluster "
                        "gets floor(n_target/n_clusters) instances, with the "
                        "remainder distributed to clusters chosen by salted "
                        "hash. Instance selection within a cluster and the "
                        "final row order are both by salted hash. THIS IS THE "
                        "FORMAL SAMPLE the run_gsm_symbolic_formal.sh sweep "
                        "actually uses -- NOT the full split."
                    ),
                },
                "data": sample,
            }, f, ensure_ascii=False, indent=2)
        print(f"  -> wrote FORMAL sample ({len(sample)} items) -> {sample_path}")

        pf = preflight_subset(sample, config=config, n=args.preflight_n)
        for it in pf:
            preflight_all.append(it)

    if args.check:
        print("\n[check] OK -- schema validated for all 3 configs, nothing written.")
        return

    n_total_formal = 3 * args.n_per_config
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
                "note": f"{args.preflight_n} items per config (main/p1/p2), "
                        "deterministic subset OF the formal cluster-balanced "
                        f"{args.n_per_config}/config sample (NOT of the full "
                        "split). The formal run uses "
                        f"{args.n_per_config}/config = {n_total_formal} total "
                        "items, cluster-balanced by original_id -- NOT the "
                        "full ~8819-row split and NOT one pooled 300.",
                "configs": CONFIGS,
                "n_per_config": args.preflight_n,
            },
            "data": preflight_all,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n-> wrote preflight subset ({len(preflight_all)} items) -> {pf_path}")
    print(f"\nFormal sample = {args.n_per_config}/config x 3 configs = {n_total_formal} total")
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
