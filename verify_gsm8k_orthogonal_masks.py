#!/usr/bin/env python3
"""Read-only server-side acceptance check for the prebuilt orthogonal masks."""
import argparse
import json
from get_answer_gsm8k_orthogonal import validate_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama3", "qwen2.5"], required=True)
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--seed", type=int, choices=[1, 2, 3], required=True)
    a = ap.parse_args()
    if a.model == "llama3":
        folder, stem, size, start, end = "llama3_non_logits", "llama3", "8B", 11, 20
        dense_dir = "llama3_non_logits"
    else:
        folder, stem, size, start, end = "qwen2.5_non_logits", "qwen2.5", "7B", 16, 22
        dense_dir = "qwen2.5_non_logits"
    mask_dir = f"{a.base_dir}/mask/{folder}"
    ortho = f"{mask_dir}/ortho_gauss_same_0.5_{start}_{end}_{size}_seed{a.seed}.npy"
    rsn = f"{mask_dir}/nmd_0.5_{start}_{end}_{size}.npy"
    dense = f"{a.base_dir}/hidden_states_mean/{dense_dir}/diff_mean_{size}.npy"
    none_dense = f"{a.base_dir}/hidden_states_mean/{dense_dir}/none_diff_mean_{size}.npy"
    alpha = -6 if a.model == "llama3" else 8
    report = validate_masks(ortho, rsn, dense, none_dense, start, end, True, alpha)
    report.update({"model": a.model, "seed": a.seed, "mask_path": ortho, "rsn_path": rsn,
                   "diff_mean_path": dense, "none_diff_mean_path": none_dense})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
