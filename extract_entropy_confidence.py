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
- We only load `model.norm.weight` and `lm_head.weight` from a LOCAL HF
  checkpoint — not the full transformer. A bare HF repo id is REFUSED
  (it used to fall back to building the whole model on CPU).
- All computation done on GPU if available; otherwise CPU.
- Reads only the final-layer column of decode_hs (index from
  `final_layer_idx_stored`), so memory per sample is O(T × H), not O(T × L × H).

Model-specific constants (2026-08-25):
- `rms_norm_eps` is READ FROM config.json, never defaulted. Llama-3.1-8B uses
  1e-5, Qwen2.5-7B-Instruct 1e-6. A wrong eps does not raise; it biases every
  logit and therefore every entropy/top1/margin. Since entropy/log(V) is the
  axis used to compare models, each must be normalized by its own constant.
- The metrics are computed on the FINAL layer, so `--layer_start/--layer_end`
  never enter the maths. They ARE verified against each H5's own meta and used
  in the output filename, so a name can no longer claim a band the data lacks.

Fail-closed (all of these previously passed silently):
- empty decode, length mismatch vs decode steps, non-finite values
- missing `question_idx` (the only key allowing per-question pairing)
- missing `steer_alpha` (would otherwise survive only in the filename)
- existing output (needs --allow_overwrite)
- unexpected cell count (--expect_n_cells; the input dir is a GLOB)

Acceptance checks:
- hidden-size agreement between H5 / norm.weight / lm_head.weight (always)
- `--verify_head`: hand-computed logits vs the model's native final_norm+lm_head
  on a few stored vectors. Run once per new model before the first extraction.
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

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Final RMSNorm. x shape: (..., H), weight: (H,).

    `eps` is REQUIRED and must come from the checkpoint's own config.json
    (`rms_norm_eps`) -- it is model-specific: Llama-3.1-8B uses 1e-5 while
    Qwen2.5-7B-Instruct uses 1e-6. A wrong eps does not raise; it silently
    biases every logit, and therefore every entropy/top1/margin, by a small
    systematic amount. That matters because entropy/log(V) is the axis used
    for cross-model comparison, so the two models must be normalized by their
    own constants or the comparison folds in an artifact.
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

def read_rms_norm_eps(model_dir: str) -> float:
    """
    Read `rms_norm_eps` from the checkpoint's own config.json. Fail closed:
    a missing key means we cannot know the model's constant, and guessing it
    is exactly the silent-bias failure this function exists to prevent.
    """
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(
            f"config.json not found under {model_dir}. It is required to read "
            f"rms_norm_eps (Llama 1e-5 vs Qwen 1e-6); refusing to guess."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "rms_norm_eps" not in cfg:
        raise SystemExit(f"'rms_norm_eps' absent from {cfg_path}; refusing to guess.")
    eps = float(cfg["rms_norm_eps"])
    print(f"  rms_norm_eps (from config.json): {eps:g}")
    return eps


def load_lm_head_and_norm(model_dir: str, device: torch.device):
    """
    Load ONLY model.norm.weight and lm_head.weight from a local HF checkpoint.
    Returns (norm_w, lm_head_w) on device, fp32.

    A bare HF repo id is REFUSED. The old fallback built a full
    AutoModelForCausalLM on CPU -- for a 7-8B checkpoint that materialises
    every weight just to read two tensors, and it contradicts this script's
    stated design of loading two tensors only. We also need config.json on
    disk for rms_norm_eps, so a local snapshot is required either way.
    """
    # An empty string is Path("") -> "." which IS a dir, so it slips past the
    # is_dir() guard and fails much later with a bare "config.json not found".
    # Almost always an unset shell variable.
    if not str(model_dir).strip():
        raise SystemExit(
            "--model_dir is EMPTY. An unset shell variable ($QWEN_DIR) is the "
            "usual cause; resolve the snapshot first:\n"
            "  export QWEN_DIR=$(python -c \"from huggingface_hub import "
            "snapshot_download as d; print(d('Qwen/Qwen2.5-7B-Instruct'))\")"
        )
    model_path = Path(model_dir)
    if not (model_path / "config.json").is_file():
        raise SystemExit(
            f"--model_dir '{model_dir}' has no config.json, so it is not a "
            f"checkpoint directory (resolved to: {model_path.resolve()})."
        )
    if not model_path.is_dir():
        raise SystemExit(
            f"--model_dir must be a LOCAL checkpoint directory; got '{model_dir}'.\n"
            f"A bare HF repo id would load the entire model onto CPU to read two "
            f"tensors, and config.json (for rms_norm_eps) would not be readable.\n"
            f"Resolve the snapshot first, e.g.:\n"
            f"  python -c \"from huggingface_hub import snapshot_download as d; "
            f"print(d('Qwen/Qwen2.5-7B-Instruct'))\""
        )

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

    # Tied embeddings: lm_head.weight may be absent by design.
    if lm_head_w is None:
        tied = bool(json.load(open(model_path / "config.json")).get("tie_word_embeddings", False))
        if not tied:
            raise SystemExit(
                f"lm_head.weight not found in {model_dir}, but config.json says "
                f"tie_word_embeddings=false. Refusing to substitute the embedding "
                f"matrix -- that would silently score against the wrong head."
            )
        for st in sorted(model_path.glob("*.safetensors")):
            sd = load_safetensors(str(st))
            for k, v in sd.items():
                if k.endswith("embed_tokens.weight") or k == "model.embed_tokens.weight":
                    lm_head_w = v.detach().to(device).to(torch.float32)
                    print(f"  (tie_word_embeddings=true; using {k} as lm_head)")
                    break
            if lm_head_w is not None:
                break

    if norm_w is None or lm_head_w is None:
        raise SystemExit(f"Could not find norm/lm_head weights in {model_dir}.")

    print(f"  norm_w shape: {tuple(norm_w.shape)}, lm_head_w shape: {tuple(lm_head_w.shape)}")
    return norm_w, lm_head_w


def check_dims(h5_path: Path, norm_w: torch.Tensor, lm_head_w: torch.Tensor) -> None:
    """
    Acceptance check A: H5 hidden size must agree with norm.weight and
    lm_head.weight. A mismatch means the H5 and the checkpoint are from
    different models -- the matmul would either raise deep inside a chunk
    loop or, worse, broadcast into plausible-looking numbers.
    """
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["samples"].keys())
        if not keys:
            raise SystemExit(f"{h5_path.name}: no samples.")
        H_h5 = int(f["samples"][keys[0]]["prefill_hs"].shape[-1])
    H_norm, H_head = int(norm_w.shape[0]), int(lm_head_w.shape[1])
    if not (H_h5 == H_norm == H_head):
        raise SystemExit(
            f"{h5_path.name}: hidden-size mismatch -- H5={H_h5}, "
            f"norm.weight={H_norm}, lm_head.weight[1]={H_head}."
        )
    print(f"  [dim check] hidden_size={H_h5}, vocab={int(lm_head_w.shape[0])} ok")


# ─────────────────────── Main loop ───────────────────────

def extract_one_file(
    h5_path: Path,
    norm_w: torch.Tensor,
    lm_head_w: torch.Tensor,
    device: torch.device,
    rms_eps: float,
    cli_layer_start: int,
    cli_layer_end: int,
    chunk_size: int = 256,
):
    """Process one HDF5 file, return signal-style dict."""
    with h5py.File(h5_path, "r") as f:
        meta_attrs = dict(f["meta"].attrs)
        if "num_layers" not in meta_attrs:
            raise SystemExit(f"{h5_path.name}: meta has no 'num_layers'.")
        num_layers = int(meta_attrs["num_layers"])

        # ── issue 4: the band is a FACT OF THE H5, not a CLI label ──
        # These metrics are computed on the FINAL layer, not on the band, so
        # the band never enters the maths. It does enter the filename, and a
        # filename claiming a band the data does not have is how two batches
        # get mixed. Verify rather than decorate.
        h5_ls = int(meta_attrs.get("layer_start", -1))
        h5_le = int(meta_attrs.get("layer_end", -1))
        if h5_ls < 0 or h5_le < 0:
            raise SystemExit(f"{h5_path.name}: meta lacks layer_start/layer_end.")
        if (cli_layer_start, cli_layer_end) != (h5_ls, h5_le):
            raise SystemExit(
                f"{h5_path.name}: band mismatch -- CLI says "
                f"[{cli_layer_start},{cli_layer_end}) but the H5 was collected at "
                f"[{h5_ls},{h5_le}). Pass the H5's own band, or point at the right dir."
            )
        # Backward compat:
        #   - new HDF5 (selective storage): use `final_layer_idx_stored` (= n_middle)
        #   - old HDF5 (full 32-layer storage): final layer = num_layers - 1
        if "final_layer_idx_stored" in meta_attrs:
            last_layer_idx = int(meta_attrs["final_layer_idx_stored"])
            n_stored = int(meta_attrs.get("n_stored_layers", -1))
            if n_stored > 0 and not (0 <= last_layer_idx < n_stored):
                raise SystemExit(
                    f"{h5_path.name}: final_layer_idx_stored={last_layer_idx} "
                    f"outside stored range [0,{n_stored})."
                )
            print(f"  (selective HDF5; final-layer storage idx = {last_layer_idx})")
        elif "n_stored_layers" in meta_attrs:
            # Selective file that somehow lacks the pointer: num_layers-1 would
            # index past the stored set. Fail rather than read a middle layer.
            raise SystemExit(
                f"{h5_path.name}: selective HDF5 (n_stored_layers present) but no "
                f"'final_layer_idx_stored'; cannot locate the final layer."
            )
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
            prefill_t = rms_norm(prefill_t.unsqueeze(0), norm_w, rms_eps)   # (1, H) → normed
            prefill_logits = prefill_t.to(torch.float32) @ lm_head_w.T      # (1, V)
            ent_pre, top1_pre, mar_pre = logits_to_metrics(prefill_logits)

            # ── decode HS (final layer only) ──
            # decode_hs shape: (T, num_layers, H); we take [:, last_layer_idx, :]
            decode_hs = g["decode_hs"][:, last_layer_idx, :]   # (T, H) fp16
            T = decode_hs.shape[0]
            if T == 0:
                # Was silently zero-filled. A zero-length trajectory is
                # indistinguishable from a real one downstream, and "check for
                # empty trajectories" cannot be done after the fact.
                raise SystemExit(
                    f"{h5_path.name} sample {key}: empty decode (T=0). "
                    f"The cell is incomplete; re-collect it."
                )
            else:
                ent_list, top1_list, mar_list = [], [], []
                for s in range(0, T, chunk_size):
                    chunk = decode_hs[s:s + chunk_size]                              # (B, H) fp16
                    chunk_t = torch.from_numpy(chunk.astype(np.float16)).to(device)
                    chunk_t = rms_norm(chunk_t, norm_w, rms_eps)                     # (B, H) normed
                    logits = chunk_t.to(torch.float32) @ lm_head_w.T                 # (B, V)
                    e, t1, m = logits_to_metrics(logits)
                    ent_list.append(e); top1_list.append(t1); mar_list.append(m)
                entropy_dec = np.concatenate(ent_list)
                top1_dec    = np.concatenate(top1_list)
                margin_dec  = np.concatenate(mar_list)

            # ── info gain: H_{t-1} - H_t; element 0 uses H_prefill ──
            prev = np.concatenate(([ent_pre[0]], entropy_dec[:-1]))
            info_gain_dec = (prev - entropy_dec).astype(np.float32)

            # ── issue 7: length + finiteness, per sample ──
            # Lengths must match the stored decode trajectories, and a non-finite
            # value poisons every mean it enters while looking like a number.
            for nm, arr in (("entropy", entropy_dec), ("top1", top1_dec),
                            ("margin", margin_dec), ("info_gain", info_gain_dec)):
                if arr.shape[0] != T:
                    raise SystemExit(
                        f"{h5_path.name} sample {key}: {nm} length {arr.shape[0]} != decode steps {T}."
                    )
                if not np.all(np.isfinite(arr)):
                    raise SystemExit(
                        f"{h5_path.name} sample {key}: non-finite value in {nm}."
                    )
            for nm, val in (("entropy_prefill", ent_pre[0]), ("top1_prefill", top1_pre[0]),
                            ("margin_prefill", mar_pre[0])):
                if not np.isfinite(val):
                    raise SystemExit(f"{h5_path.name} sample {key}: non-finite {nm}.")
            # Cross-check against the tracker's own per-step projections when present.
            for ref in ("x_decode_proj", "ema_decode_proj"):
                if ref in g and int(g[ref].shape[0]) != T:
                    raise SystemExit(
                        f"{h5_path.name} sample {key}: decode_hs has {T} steps but "
                        f"{ref} has {int(g[ref].shape[0])}."
                    )

            if "question_idx" not in g.attrs:
                raise SystemExit(
                    f"{h5_path.name} sample {key}: no 'question_idx'. It is the only "
                    f"key that lets these metrics be paired per-question with the "
                    f"signal JSON; refusing to emit unpairable output."
                )
            results.append({
                "question_idx": int(g.attrs["question_idx"]),
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

    # question_idx must be a duplicate-free cover, or per-question pairing is unsafe.
    qidx = [r["question_idx"] for r in results]
    if len(set(qidx)) != len(qidx):
        raise SystemExit(f"{h5_path.name}: duplicate question_idx values.")

    # ── issue 2: steer_alpha must travel WITH the data ──
    # extract_signal_json.py drops it, so α survives only in the filename and
    # every consumer has to re-parse it. Same schema, same trap; carry it here.
    if "steer_alpha" not in meta_attrs:
        raise SystemExit(
            f"{h5_path.name}: meta has no 'steer_alpha'. Without it the α of this "
            f"cell exists only in its filename."
        )
    out_meta = {
        "task":        meta_attrs.get("task", ""),
        "model":       meta_attrs.get("model", ""),
        "size":        meta_attrs.get("size", ""),
        "layer_start": int(meta_attrs["layer_start"]),
        "layer_end":   int(meta_attrs["layer_end"]),
        "num_layers":  num_layers,
        "n_samples":   total,
        "accuracy":    round(correct_n / max(total, 1) * 100, 2),
        "cot":         bool(meta_attrs.get("cot", False)),
        "role":        meta_attrs.get("role", "neutral"),
        "character":   meta_attrs.get("character", "(none)"),
        "prompt_template": meta_attrs.get("prompt_template", ""),
        "metric_type": "entropy_confidence",
        # provenance
        "steer_alpha": float(meta_attrs["steer_alpha"]),
        "steer_mode":  str(meta_attrs.get("steer_mode", "none")),
        "rms_norm_eps": float(rms_eps),
        "final_layer_idx_stored": int(last_layer_idx),
        "source_h5": h5_path.name,
    }
    return {"meta": out_meta, "diff_stats": diff_stats, "data": results}


def verify_against_native_head(model_dir: str, h5_path: Path, norm_w, lm_head_w,
                               device, rms_eps: float, n_probe: int = 4) -> None:
    """
    Acceptance check B: confirm that hand-rolled `rms_norm + lm_head` reproduces
    the model's OWN final norm + head on real stored hidden states.

    This is the check that catches a wrong eps, a transposed head, a tied-weight
    substitution, or an RMSNorm variant -- none of which raise, and all of which
    yield plausible entropies. Loads the full model briefly on CPU, so it is
    opt-in (--verify_head) and runs on a handful of vectors only.
    """
    from transformers import AutoModelForCausalLM

    print(f"\n[verify] loading full model on CPU for a one-off head comparison ...")
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map="cpu"
    ).eval()

    with h5py.File(h5_path, "r") as f:
        meta_attrs = dict(f["meta"].attrs)
        last_idx = int(meta_attrs["final_layer_idx_stored"]) if "final_layer_idx_stored" \
            in meta_attrs else int(meta_attrs["num_layers"]) - 1
        keys = sorted(f["samples"].keys())[:n_probe]
        vecs = [f["samples"][k]["decode_hs"][0, last_idx, :] for k in keys]
    hs = torch.from_numpy(np.stack(vecs).astype(np.float16))

    # ours (on device, as in the real path)
    ours = rms_norm(hs.to(device), norm_w, rms_eps).to(torch.float32) @ lm_head_w.T
    e_ours, t_ours, m_ours = logits_to_metrics(ours)

    # native
    with torch.no_grad():
        native = m.lm_head(m.model.norm(hs.to(torch.float32))).to(torch.float32)
    e_nat, t_nat, m_nat = logits_to_metrics(native.to(device))
    del m

    max_logit_dev = float(np.abs(ours.detach().cpu().numpy() - native.numpy()).max())
    for nm, a, b, tol in (("entropy", e_ours, e_nat, 1e-3),
                          ("top1", t_ours, t_nat, 1e-4),
                          ("margin", m_ours, m_nat, 1e-4)):
        dev = float(np.abs(a - b).max())
        status = "ok" if dev <= tol else "FAIL"
        print(f"  [verify] {nm:<8} max|Δ| = {dev:.3e}  (tol {tol:.0e})  {status}")
        if dev > tol:
            raise SystemExit(
                f"[verify] {nm} deviates by {dev:.3e} from the native head. "
                f"Check rms_norm_eps (currently {rms_eps:g}) and the lm_head weights."
            )
    print(f"  [verify] max|Δlogit| = {max_logit_dev:.3e} over {len(vecs)} probe vectors — PASS\n")


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
    parser.add_argument("--layer_start", type=int, default=11,
                        help="Band the H5 was collected at. VERIFIED against each "
                             "H5's own meta, not merely used for the filename.")
    parser.add_argument("--layer_end", type=int, default=20)
    parser.add_argument("--allow_overwrite", action="store_true",
                        help="Permit overwriting existing metrics_*.json. Refused by default.")
    parser.add_argument("--expect_n_cells", type=int, default=None,
                        help="Assert exactly this many hs_*.h5 in --h5_dir (Qwen backfill: 7). "
                             "Default None keeps the Llama flow unchanged.")
    parser.add_argument("--verify_head", action="store_true",
                        help="Acceptance check B: compare hand-computed logits against "
                             "the model's native final_norm+lm_head on a few stored "
                             "vectors. Loads the full model on CPU once; run it before "
                             "the first extraction on a new model.")
    parser.add_argument("--verify_n", type=int, default=4,
                        help="Probe vectors for --verify_head.")
    parser.add_argument("--rms_eps", type=float, default=None,
                        help="Override rms_norm_eps. Default reads config.json; "
                             "only pass this if you know why.")
    parser.add_argument("--chunk_size", type=int, default=256,
                        help="Decode tokens per logits chunk (affects GPU memory).")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")
    print(f"Loading lm_head + norm from: {args.model_dir}")
    norm_w, lm_head_w = load_lm_head_and_norm(args.model_dir, device)
    rms_eps = args.rms_eps if args.rms_eps is not None else read_rms_norm_eps(args.model_dir)
    if args.rms_eps is not None:
        print(f"  ! rms_norm_eps OVERRIDDEN via CLI: {rms_eps:g}")

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(h5_dir.glob("hs_*.h5"))
    if not h5_files:
        raise SystemExit(f"No hs_*.h5 in {h5_dir}")
    # issue 6: this is a GLOB. A directory holding a different number of cells
    # than intended silently yields a different cell set, which is how a batch
    # gets analysed against the wrong contents.
    if args.expect_n_cells is not None and len(h5_files) != args.expect_n_cells:
        raise SystemExit(
            f"Expected {args.expect_n_cells} hs_*.h5 in {h5_dir}, found {len(h5_files)}:\n  "
            + "\n  ".join(f.name for f in h5_files)
        )
    if args.verify_head:
        verify_against_native_head(args.model_dir, h5_files[0], norm_w, lm_head_w,
                                   device, rms_eps, n_probe=args.verify_n)

    print(f"Found {len(h5_files)} HDF5 file(s):")
    for p in h5_files:
        print(f"  {p.name}")

    # issue 5: refuse to clobber. Checked for ALL cells up-front so a run does
    # not die partway having already overwritten some.
    if not args.allow_overwrite:
        clashes = []
        for h5_path in h5_files:
            stem = h5_path.stem[len("hs_"):]
            suffix = f"_L{args.layer_start}-{args.layer_end}"
            body = stem[: -len(suffix)] if stem.endswith(suffix) else stem
            cand = out_dir / f"metrics_{body}_ema{args.ema_alpha}_L{args.layer_start}-{args.layer_end}.json"
            if cand.exists():
                clashes.append(cand.name)
        if clashes:
            raise SystemExit(
                "Refusing to overwrite existing output:\n  " + "\n  ".join(clashes)
                + "\nPass --allow_overwrite deliberately, or choose another --out_dir."
            )

    for h5_path in h5_files:
        print(f"\n=== {h5_path.name} ===")
        check_dims(h5_path, norm_w, lm_head_w)
        d = extract_one_file(
            h5_path, norm_w, lm_head_w, device,
            rms_eps=rms_eps,
            cli_layer_start=args.layer_start,
            cli_layer_end=args.layer_end,
            chunk_size=args.chunk_size,
        )

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
        print(f"    n={m['n_samples']}, acc={m['accuracy']}%, role={m['role']}, "
              f"alpha={m['steer_alpha']}, cot={m['cot']}")


if __name__ == "__main__":
    main()
