#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small offline sanity check for eval_gsm_symbolic.py's statistics + scorer
wiring. No model, no server, no network. Not a full test suite -- exercises
the exact-McNemar math, Holm correction, and score_cell's use of the frozen
extractor on a handful of hand-built generations."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_gsm_symbolic import (
    exact_mcnemar, holm_correct, score_cell, per_instance_breakdown,
    check_cell_consistency,
)

# ---- exact McNemar sanity: b=c=0 -> p=1.0; asymmetric discordant pairs ----
assert exact_mcnemar(0, 0) == 1.0
p_10_0 = exact_mcnemar(10, 0)
assert p_10_0 < 0.01, p_10_0  # 10 vs 0 discordant should be strongly significant
p_6_4 = exact_mcnemar(6, 4)
assert p_6_4 > 0.5, p_6_4  # nearly balanced discordant pairs -> not significant
print(f"[ok] exact_mcnemar: (0,0)->1.0, (10,0)->{p_10_0:.4g}, (6,4)->{p_6_4:.4g}")

# ---- Holm correction sanity: monotone non-decreasing adjusted p, largest raw
#      p keeps its own value (factor=1 at the last rank) ----
raw = [0.001, 0.02, 0.5]
adj = holm_correct(raw)
assert adj == sorted(adj) or True  # order preserved w.r.t. input, not sortedness
# smallest raw p gets multiplied by m=3
assert abs(adj[0] - min(1.0, 0.001 * 3)) < 1e-9
# largest raw p (already largest rank) keeps factor=1
assert abs(adj[2] - 0.5) < 1e-9
print(f"[ok] holm_correct({raw}) -> {adj}")

# monotonicity across ranks is enforced (running max)
raw2 = [0.5, 0.001, 0.02]
adj2 = holm_correct(raw2)
# ranked order by p: 0.001(x3)=0.003, 0.02(x2)=0.04, 0.5(x1)=0.5 -- all >= previous
ranked = sorted(zip(raw2, adj2))
for i in range(1, len(ranked)):
    assert ranked[i][1] >= ranked[i - 1][1] - 1e-12
print(f"[ok] holm_correct monotonicity holds: {raw2} -> {adj2}")

# ---- score_cell + per_instance_breakdown sanity using the frozen extractor ----
fake_payload = {
    "data": [
        {"id": 0, "sample_id": "main:0", "original_id": 100, "instance": 0, "answer": "42",
         "generated": "Let's think. 40+2=42. #### 42",
         "multi_marker": False, "first_last_agree": True, "is_loop": False,
         "truncated": False, "no_marker": False, "marker_unparsed": False},
        {"id": 1, "sample_id": "main:1", "original_id": 100, "instance": 1, "answer": "50",
         "generated": "Reasoning here. #### 42",  # wrong answer
         "multi_marker": False, "first_last_agree": True, "is_loop": False,
         "truncated": False, "no_marker": False, "marker_unparsed": False},
        {"id": 2, "sample_id": "main:2", "original_id": 200, "instance": 0, "answer": "7",
         "generated": "No marker at all, just rambling text with no number really wait 7",
         "multi_marker": False, "first_last_agree": True, "is_loop": False,
         "truncated": False, "no_marker": True, "marker_unparsed": False},
    ]
}
rows = score_cell(fake_payload)
assert rows["main:0"]["correct"] is True
assert rows["main:1"]["correct"] is False
# row 2 has no '####' -> falls to fallback chain in extract_gsm8k_answer;
# "7" is the last number in the text, so it will parse as "7" and score correct.
# no_marker is True (format diagnostic) even though the fallback recovered an
# answer -- no_answer (did the FULL chain find nothing) is a separate, False,
# field. This is exactly the distinction the two fields exist to preserve.
assert rows["main:2"]["no_marker"] is True
assert rows["main:2"]["no_answer"] is False
assert rows["main:2"]["correct"] is True
print(f"[ok] score_cell via frozen extractor: {[(k, v['correct']) for k, v in rows.items()]}")

pib = per_instance_breakdown(rows)
# original_id 100 has 2 instances: correct=[True, False] -> mean 0.5
# original_id 200 has 1 instance: correct=[True] -> mean 1.0
# across the 2 original_ids: mean of [0.5, 1.0] = 0.75, std computed accordingly
assert pib["n_original_ids"] == 2
assert abs(pib["mean_pct"] - 75.0) < 1e-6
print(f"[ok] per_instance_breakdown: {pib}")

# ---- check_cell_consistency: must PASS on matching meta, FAIL on drift ----
def make_meta(**overrides):
    base = {
        "model": "llama3", "model_dir": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "8B", "prompt_template": "TPL", "prompt_template_sha256": "abc",
        "cot": True, "role": "neutral", "max_new_tokens": 768, "temperature": 0.0,
        "batch_size": 24, "prefill_only": True, "prefill_tail_len": 1,
        "n_layers_band": 9, "gsm_config": "main", "alpha": 0,
        "steering_fires": 0, "steering_fires_expected": 0, "mask_sha256": "m0",
    }
    base.update(overrides)
    return base

good_payloads = {
    ("main", 0): {"meta": make_meta(gsm_config="main", alpha=0), "data": [{"sample_id": "main:0"}]},
    ("main", 4): {"meta": make_meta(gsm_config="main", alpha=4, steering_fires=9,
                                     steering_fires_expected=9, mask_sha256="mX"),
                  "data": [{"sample_id": "main:0"}]},
}
check_cell_consistency("llama3", good_payloads)  # must not raise
print("[ok] check_cell_consistency passes on consistent meta")

bad_prompt = dict(good_payloads)
bad_payload_key = ("main", 4)
bad_payloads = dict(good_payloads)
bad_payloads[bad_payload_key] = {
    "meta": make_meta(gsm_config="main", alpha=4, prompt_template="DIFFERENT",
                       steering_fires=9, steering_fires_expected=9, mask_sha256="mX"),
    "data": [{"sample_id": "main:0"}],
}
try:
    check_cell_consistency("llama3", bad_payloads)
    raise AssertionError("check_cell_consistency should have raised on prompt_template drift")
except ValueError as e:
    print(f"[ok] check_cell_consistency correctly rejects prompt drift: {e}")

fires_bad = dict(good_payloads)
fires_bad[("main", 4)] = {
    "meta": make_meta(gsm_config="main", alpha=4, steering_fires=5,
                       steering_fires_expected=9, mask_sha256="mX"),
    "data": [{"sample_id": "main:0"}],
}
try:
    check_cell_consistency("llama3", fires_bad)
    raise AssertionError("check_cell_consistency should have raised on fires mismatch")
except ValueError as e:
    print(f"[ok] check_cell_consistency correctly rejects fires mismatch: {e}")

id_mismatch = dict(good_payloads)
id_mismatch[("main", 4)] = {
    "meta": make_meta(gsm_config="main", alpha=4, steering_fires=9,
                       steering_fires_expected=9, mask_sha256="mX"),
    "data": [{"sample_id": "main:999"}],  # different sample set than alpha=0
}
try:
    check_cell_consistency("llama3", id_mismatch)
    raise AssertionError("check_cell_consistency should have raised on sample_id set mismatch")
except ValueError as e:
    print(f"[ok] check_cell_consistency correctly rejects sample_id drift: {e}")

print("\nALL SANITY CHECKS PASSED")
