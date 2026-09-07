#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline sanity check for data_gsm_symbolic.py's cluster-balanced sampling
(select_sample / finalize_order / salted_hash). No network, no model."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_gsm_symbolic import select_sample, salted_hash, N_PER_CONFIG, assert_unique_sample_ids


def make_items(config: str, n_clusters: int, instances_per_cluster):
    """instances_per_cluster: int (uniform) or list[int] (per-cluster count).
    Builds fake items with unique sample_id = f"{config}:{running_id}"."""
    items = []
    running_id = 0
    if isinstance(instances_per_cluster, int):
        counts = [instances_per_cluster] * n_clusters
    else:
        counts = instances_per_cluster
        assert len(counts) == n_clusters
    for oid in range(n_clusters):
        for inst in range(counts[oid]):
            items.append({
                "id": running_id,
                "sample_id": f"{config}:{running_id}",
                "original_id": oid,
                "instance": inst,
                "config": config,
                "question": f"q{running_id}",
                "answer": "1",
            })
            running_id += 1
    return items


# ---- case 1: main-like -- 500 clusters, 1 instance each, target 300 ----
main_items = make_items("main_test", n_clusters=500, instances_per_cluster=1)
sample = select_sample(main_items, config="main_test", n_target=300)
assert len(sample) == 300, len(sample)
assert len({it["sample_id"] for it in sample}) == 300  # no duplicates
oids = {it["original_id"] for it in sample}
assert len(oids) == 300, f"expected 300 distinct original_id, got {len(oids)}"
print(f"[ok] main-like case: {len(sample)} items, {len(oids)} distinct original_id (target 300, 1/cluster)")

# ---- case 2: p1-like -- exactly 100 clusters, 10 instances each, target 300
#      -> 300 // 100 = 3 exactly, no remainder ----
p1_items = make_items("p1_test", n_clusters=100, instances_per_cluster=10)
sample = select_sample(p1_items, config="p1_test", n_target=300)
assert len(sample) == 300, len(sample)
assert len({it["sample_id"] for it in sample}) == 300
from collections import Counter
per_cluster_count = Counter(it["original_id"] for it in sample)
assert len(per_cluster_count) == 100, len(per_cluster_count)
assert set(per_cluster_count.values()) == {3}, (
    f"expected exactly 3 instances per cluster (300/100, no remainder), got {set(per_cluster_count.values())}"
)
print(f"[ok] p1-like case (100 clusters, no remainder): {len(sample)} items, "
      f"all clusters get exactly {list(per_cluster_count.values())[0]} instances")

# ---- case 3: p2-like -- 50 clusters, plenty of instances each, target 300
#      -> 300 // 50 = 6 exactly, no remainder ----
p2_items = make_items("p2_test", n_clusters=50, instances_per_cluster=20)
sample = select_sample(p2_items, config="p2_test", n_target=300)
assert len(sample) == 300, len(sample)
per_cluster_count = Counter(it["original_id"] for it in sample)
assert len(per_cluster_count) == 50, len(per_cluster_count)
assert set(per_cluster_count.values()) == {6}, set(per_cluster_count.values())
print(f"[ok] p2-like case (50 clusters, no remainder): {len(sample)} items, "
      f"all clusters get exactly 6 instances")

# ---- case 4: remainder distribution -- 47 clusters, target 300
#      -> base_quota = 300 // 47 = 6, remainder = 300 % 47 = 18
#      -> 18 clusters get 7, 29 clusters get 6: 18*7 + 29*6 = 126 + 174 = 300 ----
p2_uneven = make_items("p2_uneven", n_clusters=47, instances_per_cluster=20)
sample = select_sample(p2_uneven, config="p2_uneven", n_target=300)
assert len(sample) == 300, len(sample)
per_cluster_count = Counter(it["original_id"] for it in sample)
assert len(per_cluster_count) == 47, len(per_cluster_count)
counts = Counter(per_cluster_count.values())
assert counts == Counter({6: 29, 7: 18}), counts
print(f"[ok] remainder distribution (47 clusters, base=6, remainder=18): "
      f"counts={dict(counts)} (expect {{6: 29, 7: 18}})")

# ---- case 5: shortfall -- one cluster has FEWER instances than its quota
#      (uneven real-world case) -- total must still land on exactly 300 ----
counts_uneven = [20] * 46 + [2]  # 46 clusters with plenty, 1 cluster with only 2
p2_shortfall = make_items("p2_shortfall", n_clusters=47, instances_per_cluster=counts_uneven)
sample = select_sample(p2_shortfall, config="p2_shortfall", n_target=300)
assert len(sample) == 300, len(sample)
assert len({it["sample_id"] for it in sample}) == 300  # still no duplicates
# the shortfall cluster (original_id=46, quota 6 or 7 but only 2 available)
# should contribute AT MOST its available 2 instances
shortfall_cluster_count = sum(1 for it in sample if it["original_id"] == 46)
assert shortfall_cluster_count <= 2, shortfall_cluster_count
print(f"[ok] shortfall case: total={len(sample)}, "
      f"shortfall cluster (only 2 available) contributed {shortfall_cluster_count}")

# ---- determinism: calling select_sample twice on the SAME items (even if
#      the list is reordered / shuffled) must give the IDENTICAL sample and
#      order -- both models must see the same 300/config sample ----
import random
shuffled = list(p1_items)
random.Random(42).shuffle(shuffled)
sample_a = select_sample(p1_items, config="p1_test", n_target=300)
sample_b = select_sample(shuffled, config="p1_test", n_target=300)
ids_a = [it["sample_id"] for it in sample_a]
ids_b = [it["sample_id"] for it in sample_b]
assert ids_a == ids_b, "select_sample must be invariant to input row order"
print("[ok] select_sample is deterministic and invariant to input row order")

# ---- salted_hash must NOT depend on Python's process hash seed -- verify by
#      checking it's a stable hex string of the expected length (sha256) ----
h1 = salted_hash("main", "x", 1, 2)
h2 = salted_hash("main", "x", 1, 2)
assert h1 == h2
assert len(h1) == 64  # sha256 hex digest length
print(f"[ok] salted_hash is deterministic sha256: {h1[:16]}...")

# ---- N_PER_CONFIG default sanity ----
assert N_PER_CONFIG == 300
print(f"[ok] N_PER_CONFIG default = {N_PER_CONFIG}")

# ---- REGRESSION: assert_unique_sample_ids must PASS when sample_id is
#      genuinely unique per (original_id, instance), and FAIL when the OLD
#      bug's collision pattern recurs (multiple instances sharing one id/
#      sample_id, exactly what happened when id==original_id was used alone
#      as the key) ----
good_items = [
    {"sample_id": f"cfg:{oid}:{inst}"}
    for oid in range(5) for inst in range(3)
]
assert_unique_sample_ids(good_items, "cfg")  # must not raise
print("[ok] assert_unique_sample_ids passes on genuinely unique (original_id, instance) keys")

# reproduce the EXACT old bug: id equals original_id, so a sample_id built
# from id alone (ignoring instance) collides across every instance sharing
# one original_id -- this is what corrupted the real main/p1/p2 formal runs.
bad_items = [
    {"sample_id": f"cfg:{oid}"}  # note: no instance in the key -- the bug
    for oid in range(5) for _inst in range(3)
]
try:
    assert_unique_sample_ids(bad_items, "cfg")
    raise AssertionError("assert_unique_sample_ids should have raised on the old bug's collision pattern")
except ValueError as e:
    print(f"[ok] assert_unique_sample_ids correctly rejects the id-only collision bug: {e}")

print("\nALL SAMPLING SANITY CHECKS PASSED")
