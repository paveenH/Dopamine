#!/usr/bin/env python3
"""
Server-side: extract per-sample NMD-projected signal scalars + metadata from
the raw-hidden-state HDF5 files, and write them out as JSON files with the
same schema used by ./signal/dopamine_signal_*.json (the neutral baseline).

This lets the local analysis scripts treat expert / non_expert outputs the
same way they treat the existing neutral baseline — no schema adapter needed.

Input  (server, ~7-9 GB each):
  /data1/paveen/RolePlaying/components/hidden_states/gsm8k/
      hs_gsm8k_8B_nocot_expert_L11-20.h5
      hs_gsm8k_8B_nocot_non_expert_L11-20.h5

Output (server staging area, ~few MB each — copy to local ./signal/):
  /data1/paveen/RolePlaying/components/llama3/
      dopamine_signal_gsm8k_8B_nocot_expert_ema0.95_L11-20.json
      dopamine_signal_gsm8k_8B_nocot_non_expert_ema0.95_L11-20.json

Per-sample JSON schema (matches ./signal baseline exactly):
  question, gold_answer, pred_answer, correct, difficulty, generated,
  x_prefill, x_prefill_per_layer, x_decode, ema_decode, x_decode_per_layer

Note: HDF5 has full-layer raw HS but only stored *aggregate* NMD scalars
on-the-fly. To produce per-layer scalars (x_*_per_layer), we re-project
HDF5's stored HS against the NMD mask, restricted to layers [11, 20).
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm


def load_mask(mask_path: str, layer_start: int, layer_end: int) -> np.ndarray:
    """Load NMD mask and slice to (n_middle, H) for hs[layer_start..layer_end).

    See utils.mask_slice_for for the layer_start-1 offset rationale.
    """
    import utils
    return utils.mask_slice_for(np.load(mask_path), layer_start, layer_end).astype(np.float32)


def project_per_layer(hs: np.ndarray, mask_mid: np.ndarray) -> list:
    """
    hs:       (num_layers, H) — single token, all layers
    mask_mid: (n_middle, H)
    Returns list of length n_middle (per-middle-layer scalars).
    """
    middle = hs[mask_mid.shape[0] * 0 :]  # placeholder; caller passes already-sliced HS
    raise NotImplementedError  # we slice in caller for clarity


def extract_per_layer(hs_layers: np.ndarray, mask_mid: np.ndarray, middle_slice: slice) -> list:
    """
    hs_layers:    (stored_layers, H) fp16 — single token's HS at stored layers
    mask_mid:     (n_middle, H)      fp32 — NMD mask, middle layers only
    middle_slice: slice telling us where middle layers live in `hs_layers`:
                    - legacy full-32 HDF5  → slice(layer_start, layer_end)
                    - new selective HDF5   → slice(0, n_middle)
    Returns: list of n_middle floats (per-layer projection scalars)
    """
    middle = hs_layers[middle_slice].astype(np.float32)  # (n_middle, H)
    projs = (middle * mask_mid).sum(axis=-1)             # (n_middle,)
    return projs.tolist()


def extract_one_file(h5_path: Path, mask_mid: np.ndarray, layer_start: int, layer_end: int):
    """Extract signal JSON from one HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        meta_attrs = dict(f["meta"].attrs)
        samples_grp = f["samples"]
        sample_keys = sorted(samples_grp.keys())

        # Decide where middle layers live in the stored HS.
        # New selective HDF5: middle layers are at storage indices [0, n_middle).
        # Legacy full-32 HDF5: middle layers are still at model-space indices.
        if "final_layer_idx_stored" in meta_attrs:
            n_middle = layer_end - layer_start
            middle_slice = slice(0, n_middle)
            print(f"  (new selective HDF5; middle layers at storage [0, {n_middle}))")
        else:
            middle_slice = slice(layer_start, layer_end)
            print(f"  (legacy full-layer HDF5; middle layers at model-space [{layer_start}, {layer_end}))")

        results = []
        for key in tqdm(sample_keys, desc=f"Extract {h5_path.name}"):
            g = samples_grp[key]

            # On-the-fly scalars (already computed by tracker, NMD on middle layers)
            x_prefill = float(g["x_prefill_proj"][()])
            x_decode = g["x_decode_proj"][:].astype(np.float32).tolist()
            ema_decode = g["ema_decode_proj"][:].astype(np.float32).tolist()

            # Per-layer scalars need reprojecting (HDF5 doesn't store these)
            prefill_hs = g["prefill_hs"][:]  # (P, num_layers, H) fp16 — full prompt
            decode_hs = g["decode_hs"][:]    # (T, num_layers, H) fp16

            # Use last prompt token for prefill per-layer (matches old tracker)
            x_prefill_per_layer = extract_per_layer(prefill_hs[-1], mask_mid, middle_slice)

            x_decode_per_layer = [
                extract_per_layer(decode_hs[t], mask_mid, middle_slice)
                for t in range(decode_hs.shape[0])
            ]

            # Metadata attrs
            results.append({
                "question":     g.attrs.get("question", ""),
                "gold_answer":  g.attrs.get("gold_answer", ""),
                "pred_answer":  g.attrs.get("pred_answer", ""),
                "correct":      bool(int(g.attrs.get("correct", 0))),
                "difficulty":   g.attrs.get("difficulty", ""),
                "generated":    g.attrs.get("generated", ""),
                "x_prefill":             x_prefill,
                "x_prefill_per_layer":   x_prefill_per_layer,
                "x_decode":              x_decode,
                "ema_decode":            ema_decode,
                "x_decode_per_layer":    x_decode_per_layer,
            })

    # Build per-difficulty stats (mirror baseline diff_stats)
    diff_stats: dict = {}
    for r in results:
        d = r["difficulty"]
        s = diff_stats.setdefault(d, {"total": 0, "correct": 0})
        s["total"] += 1
        if r["correct"]:
            s["correct"] += 1

    # Build meta block (mirror baseline schema)
    total = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    out_meta = {
        "task":        meta_attrs.get("task", ""),
        "model":       meta_attrs.get("model", ""),
        "size":        meta_attrs.get("size", ""),
        "ema_alpha":   float(meta_attrs.get("ema_alpha", 0.95)),
        "layer_start": int(meta_attrs.get("layer_start", layer_start)),
        "layer_end":   int(meta_attrs.get("layer_end", layer_end)),
        "mask":        meta_attrs.get("sanity_mask", ""),
        "n_samples":   total,
        "accuracy":    round(correct_n / max(total, 1) * 100, 2),
        "cot":         bool(meta_attrs.get("cot", False)),
        # extra info specific to role-bearing runs (not in neutral baseline)
        "role":        meta_attrs.get("role", "neutral"),
        "character":   meta_attrs.get("character", "(none)"),
        "prompt_template": meta_attrs.get("prompt_template", ""),
    }

    return {"meta": out_meta, "diff_stats": diff_stats, "data": results}


def main():
    global MASK_LAYER_START, MASK_LAYER_END
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_dir", type=str,
        default="/data1/paveen/RolePlaying/components/hidden_states/gsm8k_mathematician",
        help="Directory containing hs_*.h5 files",
    )
    parser.add_argument(
        "--mask_path", type=str,
        default="/data1/paveen/RolePlaying/components/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy",
        help="NMD mask path — must match the sanity_mask recorded in HDF5",
    )
    parser.add_argument("--layer_start", type=int, default=11)
    parser.add_argument("--layer_end", type=int, default=20)
    parser.add_argument(
        "--out_dir", type=str,
        default="/data1/paveen/RolePlaying/components/llama3",
        help="Output dir for lightweight JSON (default is the staging area "
             "intended to be copied to local).",
    )
    parser.add_argument(
        "--ema_alpha", type=float, default=0.95,
        help="Used only to name the output JSON (the actual EMA was already computed in the tracker)",
    )
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    MASK_LAYER_START = args.layer_start
    MASK_LAYER_END = args.layer_end

    print(f"Loading mask: {args.mask_path}")
    mask_mid = load_mask(args.mask_path, args.layer_start, args.layer_end)
    print(f"Mask middle shape: {mask_mid.shape}")

    # Find all HDF5 files matching hs_*.h5
    h5_files = sorted(h5_dir.glob("hs_*.h5"))
    if not h5_files:
        raise SystemExit(f"No hs_*.h5 found in {h5_dir}")
    print(f"Found {len(h5_files)} HDF5 file(s):")
    for p in h5_files:
        print(f"  {p.name}")

    for h5_path in h5_files:
        print(f"\n=== {h5_path.name} ===")
        signal_dict = extract_one_file(h5_path, mask_mid, args.layer_start, args.layer_end)

        # Output filename: dopamine_signal_<task>_<size>_<mode>_<role>_ema<alpha>_L<s>-<e>.json
        # Parse h5 filename: hs_gsm8k_8B_nocot_expert_L11-20.h5
        stem = h5_path.stem  # hs_gsm8k_8B_nocot_expert_L11-20
        # Remove leading 'hs_' and trailing '_L<s>-<e>'
        body = stem[len("hs_"):]
        # body = gsm8k_8B_nocot_expert_L11-20
        layer_suffix = f"_L{args.layer_start}-{args.layer_end}"
        if body.endswith(layer_suffix):
            body = body[: -len(layer_suffix)]
        out_name = f"dopamine_signal_{body}_ema{args.ema_alpha}_L{args.layer_start}-{args.layer_end}.json"
        out_path = out_dir / out_name

        with open(out_path, "w") as fw:
            json.dump(signal_dict, fw)
        print(f"Wrote: {out_path}  (n={signal_dict['meta']['n_samples']}, "
              f"acc={signal_dict['meta']['accuracy']}%, role={signal_dict['meta']['role']})")


if __name__ == "__main__":
    main()
