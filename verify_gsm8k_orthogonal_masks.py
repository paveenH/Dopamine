#!/usr/bin/env python3
"""Read-only server-side acceptance check for the prebuilt orthogonal masks.

--family same (default, v1): ortho_gauss_same, support identical to the actual RSN mask.
--family off  (off-v1):      ortho_gauss_off, support disjoint from the actual RSN mask;
                             --seed all additionally requires three distinct masks.
"""
import argparse
import hashlib
import json
import numpy as np
from get_answer_gsm8k_orthogonal import validate_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama3", "qwen2.5"], required=True)
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--family", choices=["same", "off"], default="same")
    ap.add_argument("--seed", choices=["1", "2", "3", "all"], required=True)
    a = ap.parse_args()
    if a.seed == "all" and a.family != "off":
        ap.error("--seed all is only defined for --family off")
    if a.model == "llama3":
        folder, stem, size, start, end = "llama3_non_logits", "llama3", "8B", 11, 20
        dense_dir = "llama3"
    else:
        folder, stem, size, start, end = "qwen2.5_non_logits", "qwen2.5", "7B", 16, 22
        dense_dir = "qwen2.5"
    mask_dir = f"{a.base_dir}/mask/{folder}"
    rsn = f"{mask_dir}/nmd_0.5_{start}_{end}_{size}.npy"
    dense = f"{a.base_dir}/hidden_states_mean/{dense_dir}/mmlue_expert_mean_{size}.npy"
    none_dense = f"{a.base_dir}/hidden_states_mean/{dense_dir}/mmlue_non_expert_mean_{size}.npy"
    alpha = -6 if a.model == "llama3" else 8
    seeds = [1, 2, 3] if a.seed == "all" else [int(a.seed)]
    reports = []
    for seed in seeds:
        ortho = f"{mask_dir}/ortho_gauss_{a.family}_0.5_{start}_{end}_{size}_seed{seed}.npy"
        report = validate_masks(ortho, rsn, dense, none_dense, start, end, True, alpha, a.family)
        report.update({"model": a.model, "seed": seed, "family": a.family, "mask_path": ortho, "rsn_path": rsn,
                       "mmlue_expert_mean_path": dense, "mmlue_non_expert_mean_path": none_dense})
        reports.append(report)
        print(json.dumps(report, indent=2))
    if a.seed == "all":
        hashes = [r["mask_sha256"] for r in reports]
        arrays = [np.load(r["mask_path"]) for r in reports]
        if len(set(hashes)) != 3 or any(np.array_equal(arrays[i], arrays[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"the three off-support seeds are not three distinct masks: {hashes}")
        print(json.dumps({"three_distinct_masks": True, "mask_sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
