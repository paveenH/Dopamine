#!/usr/bin/env python3
"""
Re-project HDF5 hidden states under an ARBITRARY mask (NMD, random, etc.) and
emit per-sample JSON in the same schema as extract_signal_json.py.

Unlike extract_signal_json.py which reuses tracker-computed (NMD) scalars from
HDF5, this script recomputes ALL aggregates (x_prefill, x_decode, ema_decode)
from raw HS using the mask you pass in — so random-mask outputs actually reflect
the random mask.

Usage (server):
  python extract_signal_json_remask.py \
    --h5_dir   /data1/paveen/Dopamine/components/hidden_states/gsm8k \
    --mask_path /data1/paveen/Dopamine/components/mask/llama3_non_logits/diff_random_0.5_11_20_8B.npy \
    --out_dir  /data1/paveen/Dopamine/components/llama3_random \
    --out_prefix random_signal \
    --ema_alpha 0.95
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def load_mask(mask_path: str, layer_start: int, layer_end: int) -> np.ndarray:
    """Load mask → return slice (n_middle, H) fp32 for hs[layer_start..layer_end).

    See utils.mask_slice_for for the layer_start-1 offset rationale.
    """
    import utils
    return utils.mask_slice_for(np.load(mask_path), layer_start, layer_end).astype(np.float32)


def project_token(hs_layers: np.ndarray, mask_mid: np.ndarray, middle_slice: slice) -> tuple:
    """
    hs_layers:    (stored_layers, H) fp16 — one token, all stored layers
    mask_mid:     (n_middle, H)      fp32
    Returns:
      scalar  = mean over middle layers of (hs · mask)
      per_lay = list[n_middle] of per-layer (hs · mask)
    """
    middle = hs_layers[middle_slice].astype(np.float32)   # (n_middle, H)
    per_lay = (middle * mask_mid).sum(axis=-1)            # (n_middle,)
    return float(per_lay.mean()), per_lay.tolist()


def extract_one_file(h5_path: Path, mask_mid: np.ndarray,
                     layer_start: int, layer_end: int, ema_alpha: float):
    with h5py.File(h5_path, "r") as f:
        meta_attrs = dict(f["meta"].attrs)
        samples_grp = f["samples"]
        sample_keys = sorted(samples_grp.keys())

        if "final_layer_idx_stored" in meta_attrs:
            n_middle = layer_end - layer_start
            middle_slice = slice(0, n_middle)
            print(f"  (selective HDF5; middle layers at storage [0, {n_middle}))")
        else:
            middle_slice = slice(layer_start, layer_end)
            print(f"  (legacy full-layer HDF5; middle at [{layer_start}, {layer_end}))")

        results = []
        for key in tqdm(sample_keys, desc=f"Remask {h5_path.name}"):
            g = samples_grp[key]
            prefill_hs = g["prefill_hs"][:]   # (P, stored, H) fp16
            decode_hs  = g["decode_hs"][:]    # (T, stored, H) fp16

            # Prefill: use last prompt token (matches tracker)
            x_prefill, x_prefill_per_layer = project_token(prefill_hs[-1], mask_mid, middle_slice)

            # Decode: per-step scalar + per-layer
            x_decode = []
            x_decode_per_layer = []
            for t in range(decode_hs.shape[0]):
                s, pl = project_token(decode_hs[t], mask_mid, middle_slice)
                x_decode.append(s)
                x_decode_per_layer.append(pl)

            # EMA over decode scalar (init = x_prefill, same as tracker)
            ema = x_prefill
            ema_decode = []
            for s in x_decode:
                ema = ema_alpha * ema + (1.0 - ema_alpha) * s
                ema_decode.append(ema)

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

    diff_stats: dict = {}
    for r in results:
        d = r["difficulty"]
        s = diff_stats.setdefault(d, {"total": 0, "correct": 0})
        s["total"] += 1
        if r["correct"]:
            s["correct"] += 1

    total = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    out_meta = {
        "task":        meta_attrs.get("task", ""),
        "model":       meta_attrs.get("model", ""),
        "size":        meta_attrs.get("size", ""),
        "ema_alpha":   float(ema_alpha),
        "layer_start": int(meta_attrs.get("layer_start", layer_start)),
        "layer_end":   int(meta_attrs.get("layer_end", layer_end)),
        "n_samples":   total,
        "accuracy":    round(correct_n / max(total, 1) * 100, 2),
        "cot":         bool(meta_attrs.get("cot", False)),
        "role":        meta_attrs.get("role", "neutral"),
        "character":   meta_attrs.get("character", "(none)"),
        "prompt_template": meta_attrs.get("prompt_template", ""),
    }
    return {"meta": out_meta, "diff_stats": diff_stats, "data": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_dir",     required=True)
    p.add_argument("--mask_path",  required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--out_prefix", default="random_signal",
                   help="filename prefix; output = {prefix}_{body}_ema{α}_L{s}-{e}.json")
    p.add_argument("--layer_start", type=int, default=11)
    p.add_argument("--layer_end",   type=int, default=20)
    p.add_argument("--ema_alpha",   type=float, default=0.95)
    args = p.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_name = Path(args.mask_path).name
    print(f"Loading mask: {args.mask_path}")
    mask_mid = load_mask(args.mask_path, args.layer_start, args.layer_end)
    print(f"Mask middle shape: {mask_mid.shape} (mean |w|={np.abs(mask_mid).mean():.5f}, "
          f"nnz/layer≈{(mask_mid != 0).sum(axis=-1).mean():.1f})")

    h5_files = sorted(h5_dir.glob("hs_*.h5"))
    if not h5_files:
        raise SystemExit(f"No hs_*.h5 found in {h5_dir}")
    print(f"Found {len(h5_files)} HDF5 file(s):")
    for f in h5_files:
        print(f"  {f.name}")

    for h5_path in h5_files:
        print(f"\n=== {h5_path.name} ===")
        sig = extract_one_file(h5_path, mask_mid, args.layer_start, args.layer_end, args.ema_alpha)
        sig["meta"]["mask"] = mask_name

        stem = h5_path.stem
        body = stem[len("hs_"):] if stem.startswith("hs_") else stem
        suf  = f"_L{args.layer_start}-{args.layer_end}"
        if body.endswith(suf):
            body = body[: -len(suf)]
        out_name = f"{args.out_prefix}_{body}_ema{args.ema_alpha}_L{args.layer_start}-{args.layer_end}.json"
        out_path = out_dir / out_name
        with open(out_path, "w") as fw:
            json.dump(sig, fw)
        print(f"Wrote: {out_path}  (n={sig['meta']['n_samples']}, "
              f"acc={sig['meta']['accuracy']}%, role={sig['meta']['role']}, mask={mask_name})")


if __name__ == "__main__":
    main()
