#!/usr/bin/env python3
"""
Verify mask indexing semantics on the server.

Three questions we want to answer:

  Q1. Saved mask shape and which rows are non-zero?
      → tells us where nmd.py actually wrote values, after the `mask[1:]` strip.

  Q2. What is the layer count of the loaded model?
      - len(model.layers) = number of decoder layers
      - len(model(...).hidden_states) = decoder_layers + 1 (includes embedding)
      → tells us whether nmd.py's `l in range(L)` indexes HF hidden_states (L+1)
        or decoder layers (L).

  Q3. Where does regenerate inject?
      llms.regenerate does:
          for layer, diff_matrix in zip(decoder_layers, diff_matrices):
              layer.register_forward_hook(...)
      So diff_matrices[i] (loaded directly from saved mask) is applied to
      decoder_layers[i]. → confirms whether saved index `i` corresponds to
      decoder layer `i` or decoder layer `i-1`.

Run on server (does NOT load full model weights for Q1 alone; Q2/Q3 do).

  python sanity_mask_indexing.py \
    --mask_path /data1/paveen/RolePlaying/components/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy \
    --model_dir meta-llama/Llama-3.1-8B-Instruct

Set --skip_model to skip Q2/Q3 if you only want the mask info.
"""

import argparse
import numpy as np


def q1_mask(mask_path: str):
    print("=" * 70)
    print("Q1. Saved mask")
    print("=" * 70)
    mask = np.load(mask_path)
    print(f"  shape: {mask.shape}")
    nz = [i for i in range(mask.shape[0]) if (mask[i] != 0).any()]
    print(f"  non-zero saved-index rows: {nz}")
    print(f"  saved-index count: {len(nz)}")
    # density per non-zero row
    for i in nz[:3] + nz[-3:]:
        n = int((mask[i] != 0).sum())
        print(f"    saved[{i:2d}]  nnz={n}   norm={np.linalg.norm(mask[i]):.4f}")
    return mask


def q2_q3_model(model_dir: str, mask: np.ndarray):
    print()
    print("=" * 70)
    print("Q2. Model layer count (loads weights — may take a minute)")
    print("=" * 70)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Decoder layers count (HF Llama: model.model.layers)
    try:
        n_dec = len(model.model.layers)
        print(f"  len(model.model.layers) = {n_dec}   (decoder layers)")
    except Exception as e:
        print(f"  (couldn't find model.model.layers: {e})")
        n_dec = None

    # hidden_states length on a dummy forward
    inp = tok("hello world", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    n_hs = len(out.hidden_states)
    print(f"  forward(output_hidden_states=True): {n_hs} tensors   "
          f"(= decoder_layers + 1 embedding)")

    # Sanity: hs[0] = embedding output, hs[1..n_dec] = each decoder layer output
    print(f"  hs[0].shape  = {tuple(out.hidden_states[0].shape)}   (embedding)")
    print(f"  hs[-1].shape = {tuple(out.hidden_states[-1].shape)}  (final layer)")

    print()
    print("=" * 70)
    print("Q3. Saved-mask index ↔ decoder-layer mapping (via regenerate's contract)")
    print("=" * 70)
    print("  llms.regenerate requires len(diff_matrices) == len(decoder_layers).")
    print(f"  → saved mask must have {n_dec} rows to feed regenerate directly.")
    print(f"  → saved mask actual rows: {mask.shape[0]}")
    if n_dec is not None and mask.shape[0] == n_dec:
        print("  ✓ saved mask row count matches decoder layer count exactly.")
        print("  ✓ saved-index i  ↔  decoder_layers[i]  ↔  hidden_states[i+1]")
    elif n_dec is not None and mask.shape[0] == n_dec + 1:
        print("  ⚠ saved mask is one longer than decoder count — kept embedding row?")
    else:
        print(f"  ⚠ mismatch: saved rows {mask.shape[0]} vs decoder count {n_dec}")

    # Final interpretation
    print()
    print("=" * 70)
    print("Conclusion")
    print("=" * 70)
    nz = [i for i in range(mask.shape[0]) if (mask[i] != 0).any()]
    if n_dec is not None and mask.shape[0] == n_dec and nz:
        first, last = nz[0], nz[-1]
        print(f"  saved-index non-zero range: [{first}, {last}]")
        print(f"  → applies to decoder_layers[{first} .. {last}]")
        print(f"  → equivalent to hidden_states[{first+1} .. {last+1}]")
        print()
        print(f"  If you intended `--layer_start {first+1} --layer_end {last+2}`")
        print(f"  (HF hidden_states semantics, end-exclusive), then:")
        print(f"    * consumer's mask[layer_start:layer_end] = mask[{first+1}:{last+2}]")
        print(f"      → saved rows {first+1}..{last+1} (off by 1 from non-zero!)")
        print(f"    * To align: use mask[layer_start-1:layer_end-1]"
              f" = mask[{first}:{last+1}]   ✓")
        print()
        print(f"  If you intended decoder-layer-index semantics"
              f" (start={first}, end={last+1}):")
        print(f"    * consumer's mask[layer_start:layer_end] = mask[{first}:{last+1}]"
              f"  → already correct, no offset needed.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_path", required=True,
                   help="e.g. .../mask/llama3_non_logits/nmd_0.5_11_20_8B.npy")
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--skip_model", action="store_true",
                   help="Skip Q2/Q3 (no model weights loaded)")
    args = p.parse_args()

    mask = q1_mask(args.mask_path)
    if not args.skip_model:
        q2_q3_model(args.model_dir, mask)


if __name__ == "__main__":
    main()
