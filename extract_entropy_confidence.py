#!/usr/bin/env python3
"""
Server-side: extract per-step entropy / confidence / margin / info-gain from
the raw-hidden-state HDF5 files. Output JSON parallels dopamine_signal_*.json
schema so local analysis scripts can ingest it the same way.

Pipeline:
  HDF5 stores decode_hs[t, layer=31, :]      ← last decoder block output
  pipe through final RMSNorm + lm_head.weight  ← reconstructs the true logits
  per token  →  softmax → (entropy, top1_prob, margin)
  per step   →  info_gain_t = entropy_{t-1} - entropy_t

Also computes the same scalars for the **last prompt token** (prefill snapshot)
so paper-03 §3.4.2 "intuitive prior" analysis is supported.

Input  (server, full-layer fp16 HS):
  /data1/paveen/Dopamine/components/hidden_states/<task>/
      hs_<task>_<size>_<mode>_<role>_L<L_start>-<L_end>.h5

Output (server, ~few MB each — copy to local analysis workspace):
  /data1/paveen/Dopamine/components/llama3/
      metrics_<task>_<size>_<mode>_<role>_ema0.95_L<L_start>-<L_end>.json

Per-sample JSON schema:
  question, gold_answer, pred_answer, correct, difficulty, generated
  # prefill snapshot (last prompt token, after the model finished reading the prompt)
  entropy_prefill, top1_prefill, margin_prefill
  # per-decode-step trajectories
  entropy_decode  (T,)
  top1_decode     (T,)
  margin_decode   (T,)
  info_gain_decode (T,)   # H_{t-1} - H_t; element 0 = H_prefill - H_0

Notes:
- We only load `model.norm.weight` and `lm_head.weight` from HF checkpoint —
  not the full transformer — keeping memory and load time minimal.
- All computation done on GPU if available; otherwise CPU.
- Reads only the last-layer column of decode_hs (layer 31 for Llama3-8B),
  so memory cost per sample is O(T × H) instead of O(T × L × H).
"""

import os
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors


# ─────────────────────── Llama3 RMSNorm ───────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Llama-3 final RMSNorm. x shape: (..., H), weight: (H,).
    """
    # Cast to fp32 for stable computation; cast back to input dtype at the end.
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return (x32 * weight.to(torch.float32)).to(in_dtype)


# ─────────────────────── Logits → metrics ───────────────────────

def logits_to_metrics(logits: torch.Tensor):
    """
    logits: (N, V) fp32 — one row per token position to score.
    Returns four (N,) numpy fp32 arrays: entropy, top1_prob, margin, max_logit_norm
    """
    # Stable log-softmax for entropy
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)                   # (N,)
    # Top-1 and top-2
    top2_vals, _ = probs.topk(2, dim=-1)                         # (N, 2)
    top1_prob = top2_vals[:, 0]
    margin = top1_prob - top2_vals[:, 1]
    return (
        entropy.cpu().numpy().astype(np.float32),
        top1_prob.cpu().numpy().astype(np.float32),
        margin.cpu().numpy().astype(np.float32),
    )


# ─────────────────────── Load HF weights ───────────────────────

def load_lm_head_and_norm(model_dir: str, device: torch.device):
    """
    Load only model.norm.weight and lm_head.weight from a HF Llama-3 checkpoint.
    Returns (norm_w, lm_head_w) on device, fp32.
    """
    model_path = Path(model_dir)
    if not model_path.is_dir():
        # HF cache dir — pass model_dir directly to AutoModel (slower path)
        from transformers import AutoModelForCausalLM
        print(f"  ! model_dir is not a local dir; falling back to AutoModelForCausalLM.")
        m = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        norm_w = m.model.norm.weight.detach().to(device).to(torch.float32)
        lm_head_w = m.lm_head.weight.detach().to(device).to(torch.float32)
        del m
        return norm_w, lm_head_w

    # Local dir — load weights directly from .safetensors / .bin to avoid building model
    # Look for index file (sharded) or single file
    norm_w = lm_head_w = None
    for st in sorted(model_path.glob("*.safetensors")):
        sd = load_safetensors(str(st))
        for k, v in sd.items():
            if k.endswith("model.norm.weight") or k == "model.norm.weight":
                norm_w = v.detach().to(device).to(torch.float32)
            elif k == "lm_head.weight":
                lm_head_w = v.detach().to(device).to(torch.float32)
        if norm_w is not None and lm_head_w is not None:
            break
    # If lm_head was tied to embedding weights, lm_head may be absent
    if lm_head_w is None:
        for st in sorted(model_path.glob("*.safetensors")):
            sd = load_safetensors(str(st))
            for k, v in sd.items():
                if k.endswith("embed_tokens.weight") or k == "model.embed_tokens.weight":
                    lm_head_w = v.detach().to(device).to(torch.float32)
                    print(f"  (using tied embedding weight as lm_head: {k})")
                    break
            if lm_head_w is not None:
                break
    if norm_w is None or lm_head_w is None:
        raise RuntimeError(
            f"Could not find norm/lm_head weights in {model_dir}. "
            f"Try passing the HF id (e.g. meta-llama/Llama-3.1-8B-Instruct)."
        )
    print(f"  norm_w shape: {tuple(norm_w.shape)}, lm_head_w shape: {tuple(lm_head_w.shape)}")
    return norm_w, lm_head_w


# ─────────────────────── Main loop ───────────────────────

def extract_one_file(
    h5_path: Path,
    norm_w: torch.Tensor,
    lm_head_w: torch.Tensor,
    device: torch.device,
    chunk_size: int = 256,
):
    """Process one HDF5 file, return signal-style dict."""
    with h5py.File(h5_path, "r") as f:
        meta_attrs = dict(f["meta"].attrs)
        num_layers = int(meta_attrs.get("num_layers", 32))
        # Backward compat:
        #   - new HDF5 (selective storage): use `final_layer_idx_stored` (= n_middle)
        #   - old HDF5 (full 32-layer storage): final layer = num_layers - 1
        if "final_layer_idx_stored" in meta_attrs:
            last_layer_idx = int(meta_attrs["final_layer_idx_stored"])
            print(f"  (new selective HDF5; final-layer storage idx = {last_layer_idx})")
        else:
            last_layer_idx = num_layers - 1
            print(f"  (legacy full-layer HDF5; final layer = {last_layer_idx})")

        samples_grp = f["samples"]
        sample_keys = sorted(samples_grp.keys())

        results = []
        for key in tqdm(sample_keys, desc=f"  {h5_path.name}"):
            g = samples_grp[key]

            # ── prefill last-token HS (final layer) ──
            # prefill_hs shape: (P, num_layers, H)
            P = g["prefill_hs"].shape[0]
            prefill_last_hs = g["prefill_hs"][P - 1, last_layer_idx, :]   # (H,) fp16
            prefill_t = torch.from_numpy(prefill_last_hs.astype(np.float16)).to(device)
            prefill_t = rms_norm(prefill_t.unsqueeze(0), norm_w)            # (1, H) → normed
            prefill_logits = prefill_t.to(torch.float32) @ lm_head_w.T      # (1, V)
            ent_pre, top1_pre, mar_pre = logits_to_metrics(prefill_logits)

            # ── decode HS (final layer only) ──
            # decode_hs shape: (T, num_layers, H); we take [:, last_layer_idx, :]
            decode_hs = g["decode_hs"][:, last_layer_idx, :]   # (T, H) fp16
            T = decode_hs.shape[0]
            if T == 0:
                # Empty decode (shouldn't normally happen)
                entropy_dec = np.zeros((0,), dtype=np.float32)
                top1_dec    = np.zeros((0,), dtype=np.float32)
                margin_dec  = np.zeros((0,), dtype=np.float32)
            else:
                ent_list, top1_list, mar_list = [], [], []
                for s in range(0, T, chunk_size):
                    chunk = decode_hs[s:s + chunk_size]                              # (B, H) fp16
                    chunk_t = torch.from_numpy(chunk.astype(np.float16)).to(device)
                    chunk_t = rms_norm(chunk_t, norm_w)                              # (B, H) normed
                    logits = chunk_t.to(torch.float32) @ lm_head_w.T                 # (B, V)
                    e, t1, m = logits_to_metrics(logits)
                    ent_list.append(e); top1_list.append(t1); mar_list.append(m)
                entropy_dec = np.concatenate(ent_list)
                top1_dec    = np.concatenate(top1_list)
                margin_dec  = np.concatenate(mar_list)

            # ── info gain: H_{t-1} - H_t; element 0 uses H_prefill ──
            if entropy_dec.size > 0:
                prev = np.concatenate(([ent_pre[0]], entropy_dec[:-1]))
                info_gain_dec = (prev - entropy_dec).astype(np.float32)
            else:
                info_gain_dec = np.zeros((0,), dtype=np.float32)

            results.append({
                "question":     g.attrs.get("question", ""),
                "gold_answer":  g.attrs.get("gold_answer", ""),
                "pred_answer":  g.attrs.get("pred_answer", ""),
                "correct":      bool(int(g.attrs.get("correct", 0))),
                "difficulty":   g.attrs.get("difficulty", ""),
                "generated":    g.attrs.get("generated", ""),
                # prefill snapshot
                "entropy_prefill": float(ent_pre[0]),
                "top1_prefill":    float(top1_pre[0]),
                "margin_prefill":  float(mar_pre[0]),
                # per-decode-step
                "entropy_decode":   entropy_dec.tolist(),
                "top1_decode":      top1_dec.tolist(),
                "margin_decode":    margin_dec.tolist(),
                "info_gain_decode": info_gain_dec.tolist(),
            })

    # diff_stats
    diff_stats: dict = {}
    for r in results:
        d = r["difficulty"]
        s = diff_stats.setdefault(d, {"total": 0, "correct": 0})
        s["total"] += 1
        if r["correct"]: s["correct"] += 1

    total = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    out_meta = {
        "task":        meta_attrs.get("task", ""),
        "model":       meta_attrs.get("model", ""),
        "size":        meta_attrs.get("size", ""),
        "layer_start": int(meta_attrs.get("layer_start", 11)),
        "layer_end":   int(meta_attrs.get("layer_end", 20)),
        "num_layers":  num_layers,
        "n_samples":   total,
        "accuracy":    round(correct_n / max(total, 1) * 100, 2),
        "cot":         bool(meta_attrs.get("cot", False)),
        "role":        meta_attrs.get("role", "neutral"),
        "character":   meta_attrs.get("character", "(none)"),
        "prompt_template": meta_attrs.get("prompt_template", ""),
        "metric_type": "entropy_confidence",
    }
    return {"meta": out_meta, "diff_stats": diff_stats, "data": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_dir", type=str,
        default="/data1/paveen/Dopamine/components/hidden_states/gsm8k",
        help="Directory containing hs_*.h5",
    )
    parser.add_argument(
        "--model_dir", type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id or local path. Only norm + lm_head weights are loaded.",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default="/data1/paveen/Dopamine/components/llama3",
    )
    parser.add_argument("--ema_alpha", type=float, default=0.95,
                        help="Only used to name the output JSON.")
    parser.add_argument("--layer_start", type=int, default=11)
    parser.add_argument("--layer_end", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=256,
                        help="Decode tokens per logits chunk (affects GPU memory).")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")
    print(f"Loading lm_head + norm from: {args.model_dir}")
    norm_w, lm_head_w = load_lm_head_and_norm(args.model_dir, device)

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(h5_dir.glob("hs_*.h5"))
    if not h5_files:
        raise SystemExit(f"No hs_*.h5 in {h5_dir}")
    print(f"Found {len(h5_files)} HDF5 file(s):")
    for p in h5_files:
        print(f"  {p.name}")

    for h5_path in h5_files:
        print(f"\n=== {h5_path.name} ===")
        d = extract_one_file(h5_path, norm_w, lm_head_w, device, args.chunk_size)

        # output filename
        stem = h5_path.stem            # hs_gsm8k_8B_nocot_expert_L11-20
        body = stem[len("hs_"):]       # gsm8k_8B_nocot_expert_L11-20
        layer_suffix = f"_L{args.layer_start}-{args.layer_end}"
        if body.endswith(layer_suffix):
            body = body[: -len(layer_suffix)]
        out_name = f"metrics_{body}_ema{args.ema_alpha}_L{args.layer_start}-{args.layer_end}.json"
        out_path = out_dir / out_name
        with open(out_path, "w") as fw:
            json.dump(d, fw)
        m = d["meta"]
        print(f"  → {out_path}")
        print(f"    n={m['n_samples']}, acc={m['accuracy']}%, role={m['role']}")


if __name__ == "__main__":
    main()
