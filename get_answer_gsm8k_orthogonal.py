#!/usr/bin/env python3
"""One-cell GSM8K generation for the norm-matched orthogonal control."""
import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import torch

from llms import VicundaModel
from template import select_templates_gsm8k
import utils


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def validate_masks(mask_path, rsn_path, dense_path, none_dense_path, start, end, orthogonal, alpha=1.0):
    for p in (mask_path, rsn_path, dense_path, none_dense_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"required source file missing: {p}")
    u = np.load(mask_path, allow_pickle=False)
    m = np.load(rsn_path, allow_pickle=False)
    dense_char = np.load(dense_path, allow_pickle=False)
    dense_none = np.load(none_dense_path, allow_pickle=False)
    if dense_char.shape != dense_none.shape:
        raise ValueError(f"dense source shape mismatch: diff_mean={dense_char.shape}, none_diff_mean={dense_none.shape}")
    dense = np.squeeze(dense_char - dense_none).astype(np.float64)
    if u.shape != m.shape or u.ndim != 2 or dense.ndim != 2 or dense.shape != (u.shape[0] + 1, u.shape[1]):
        raise ValueError(f"shape mismatch: ortho={u.shape}, RSN={m.shape}, dense={dense.shape}")
    if not all(np.isfinite(x).all() for x in (u, m, dense_char, dense_none, dense)):
        raise ValueError("mask or dense role-difference contains non-finite values")
    if orthogonal and not np.array_equal(u != 0, m != 0):
        raise ValueError("orthogonal mask support differs from actual RSN mask")
    rows = []
    layer_rows = set(range(start - 1, end - 1))
    if start < 1 or end > u.shape[0] + 1:
        raise ValueError("layer band is outside saved masks (embedding row is omitted)")
    for i in range(u.shape[0]):
        if i not in layer_rows:
            if np.any(u[i]) or np.any(m[i]):
                raise ValueError(f"layer row {i} outside requested band is nonzero")
            continue
        # Saved row i corresponds to original dense-mean row i+1.
        ul, ml, dl = u[i].astype(np.float64), m[i].astype(np.float64), dense[i + 1]
        if orthogonal and not np.array_equal(ul != 0, ml != 0):
            raise ValueError(f"support mismatch in decoder layer {i+1}")
        nu, nm = np.linalg.norm(ul), np.linalg.norm(ml)
        if orthogonal and (np.count_nonzero(ul) == 0 or np.count_nonzero(ml) == 0):
            raise ValueError(f"empty in-band support in decoder layer {i+1}")
        dot = float(ul @ dl)
        if orthogonal and abs(nu - nm) >= 1e-3 * max(nm, 1e-8):
            raise ValueError(f"norm mismatch in decoder layer {i+1}: {nu} vs {nm}")
        cosine = abs(dot) / (nu * np.linalg.norm(dl)) if nu and np.linalg.norm(dl) else 0.0
        if orthogonal and cosine >= 1e-5:
            raise ValueError(f"dense role-diff orthogonality failed in decoder layer {i+1}: dot={dot}")
        rows.append({"decoder_layer": i, "norm": nu, "dot": dot, "abs_cosine": cosine})
    dtype = torch.bfloat16
    effective = torch.as_tensor(u * alpha, dtype=dtype).float().numpy()
    matched = torch.as_tensor(m * alpha, dtype=dtype).float().numpy()
    return {"shape": list(u.shape), "nonzero_count": int(np.count_nonzero(u)), "layers": rows,
            "direction_kind": "orthogonal" if orthogonal else "rsn_reference",
            "mask_sha256": sha256(mask_path), "rsn_mask_sha256": sha256(rsn_path),
            "mmlue_expert_mean_sha256": sha256(dense_path), "mmlue_non_expert_mean_sha256": sha256(none_dense_path),
            "bf16_norm_max_abs_error": float(max(abs(np.linalg.norm(effective[i]) - np.linalg.norm(matched[i])) for i in layer_rows)),
            "injected_alpha": alpha,
            "bf16_dot_abs_max": float(max(abs(float(effective[i] @ dense[i+1])) for i in layer_rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama3", "qwen2.5"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--size", choices=["8B", "7B"], required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--condition", choices=["baseline", "rsn", "seed1", "seed2", "seed3"], required=True)
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--rsn_mask_path", required=True)
    ap.add_argument("--dense_mean_path", required=True)
    ap.add_argument("--none_dense_mean_path", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--smoke_n", type=int, default=0)
    a = ap.parse_args()
    if (a.model, a.size, a.start, a.end) not in (("llama3", "8B", 11, 20), ("qwen2.5", "7B", 16, 22)):
        ap.error("model/size/layer band must match the frozen Llama or Qwen protocol")
    expected_alpha = {"llama3": -6.0, "qwen2.5": 8.0}[a.model]
    if a.condition == "baseline" and a.alpha != 0 or a.condition != "baseline" and a.alpha != expected_alpha:
        ap.error("alpha does not match the fixed condition")
    if a.condition == "baseline":
        if not os.path.isfile(a.mask_path) or os.path.realpath(a.mask_path) != os.path.realpath(a.rsn_mask_path):
            raise ValueError("baseline must explicitly use the matching existing RSN mask path")
        diff = np.zeros_like(np.load(a.rsn_mask_path, allow_pickle=False))
        mask_validation = {"baseline_zero_mask": True, "mask_sha256": sha256(a.rsn_mask_path),
                           "rsn_mask_sha256": sha256(a.rsn_mask_path),
                           "mmlue_expert_mean_sha256": sha256(a.dense_mean_path),
                           "mmlue_non_expert_mean_sha256": sha256(a.none_dense_mean_path)}
    else:
        mask_validation = validate_masks(a.mask_path, a.rsn_mask_path, a.dense_mean_path, a.none_dense_mean_path, a.start, a.end,
                                         a.condition.startswith("seed"), a.alpha)
        diff = np.load(a.mask_path, allow_pickle=False) * a.alpha

    data_path = a.test_file if os.path.isabs(a.test_file) else os.path.join(a.base_dir, a.test_file)
    samples = utils.load_json(data_path)
    if a.smoke_n:
        samples = samples[:a.smoke_n]
    elif len(samples) != 300:
        raise ValueError(f"formal run requires exactly 300 GSM8K items, got {len(samples)}")
    prompts = [select_templates_gsm8k(suite="default", cot=False, wording="plain")["neutral"].format(context=s["question"]) for s in samples]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path, manifest_path = out_dir / "result.json", out_dir / "manifest.json"
    provenance = {"protocol": "gsm8k-orthogonal-v1", "model": a.model, "model_dir": a.model_dir,
                  "condition": a.condition, "alpha": a.alpha, "start": a.start, "end": a.end,
                  "batch_size": a.batch_size, "max_new_tokens": a.max_new_tokens, "temperature": 0,
                  "top_p": 0.9, "prefill_only": True, "prefill_tail_len": 1,
                  "mask_path": os.path.abspath(a.mask_path), "mask_validation": mask_validation,
                  "data_sha256": sha256(data_path), "question_sha256": [hashlib.sha256(s["question"].encode()).hexdigest() for s in samples],
                  "prompt_sha256": [hashlib.sha256(p.encode()).hexdigest() for p in prompts],
                  "question_order_sha256": hashlib.sha256("\n".join(hashlib.sha256(s["question"].encode()).hexdigest() for s in samples).encode()).hexdigest(),
                  "host": socket.gethostname(), "platform": platform.platform(), "torch": torch.__version__,
                  "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    vc = VicundaModel(model_path=a.model_dir)
    vc.model.eval()
    tokenized = vc.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    tail_ids = tokenized.input_ids[:, -1].tolist()
    token_sequences = [vc.tokenizer(p, return_tensors=None)["input_ids"] for p in prompts]
    provenance["model_config_sha256"] = hashlib.sha256(json.dumps(vc.model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest()
    provenance["model_config_identity"] = getattr(vc.model.config, "_name_or_path", None)
    provenance["device_map"] = str(getattr(vc.model, "hf_device_map", None))
    provenance["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    provenance["transformers"] = __import__("transformers").__version__
    provenance["tokenizers"] = __import__("tokenizers").__version__
    try:
        provenance["accelerate"] = __import__("accelerate").__version__
    except ImportError:
        provenance["accelerate"] = None
    provenance["python"] = platform.python_version()
    provenance["cuda"] = torch.version.cuda
    provenance["tokenizer"] = {"class": type(vc.tokenizer).__name__,
                                "name_or_path": getattr(vc.tokenizer, "name_or_path", None),
                                "vocab_size": len(vc.tokenizer),
                                "padding_side": vc.tokenizer.padding_side,
                                "bos_token_id": vc.tokenizer.bos_token_id,
                                "eos_token_id": vc.tokenizer.eos_token_id}
    provenance["tokenized_prompt_ids_sha256"] = hashlib.sha256(json.dumps(token_sequences, separators=(",", ":")).encode()).hexdigest()
    provenance["tokenized_prompt_lengths"] = [len(x) for x in token_sequences]
    provenance["injection_tail_tokens"] = [list(x) for x in sorted({(int(t), vc.tokenizer.decode([int(t)])) for t in tail_ids})]
    provenance["runner_sha256"] = sha256(__file__)
    provenance["code_sha256"] = {name: sha256(os.path.join(os.path.dirname(__file__), name))
                                  for name in ("llms.py", "template.py", "utils.py")}
    provenance["model_weight_revision"] = getattr(vc.model.config, "_commit_hash", None)
    if manifest_path.exists() or result_path.exists():
        if not (manifest_path.exists() and result_path.exists()):
            raise FileExistsError(f"incomplete existing cell at {out_dir}; refusing resume/overwrite")
        old = json.loads(manifest_path.read_text())
        if canonical_json(old.get("configuration")) != canonical_json(provenance):
            raise FileExistsError(f"existing cell provenance conflict at {out_dir}; refusing overwrite")
        result = json.loads(result_path.read_text())
        rows = result.get("data")
        run = old.get("run", {})
        if (not isinstance(rows, list) or len(rows) != len(samples)
                or run.get("row_count") != len(rows)
                or run.get("result_sha256") != sha256(result_path)
                or run.get("steering_fires") != run.get("expected_steering_fires")
                or run.get("expected_steering_fires") != (0 if a.condition == "baseline" else (a.end-a.start)*len(samples))
                or [hashlib.sha256(r.get("question", "").encode()).hexdigest() for r in rows] != provenance["question_sha256"]
                or any(any(r.get(k) != sample.get(k) for k in ("question", "answer"))
                       or not all(k in r for k in ("generated_neutral", "raw_generated_neutral", "generated_token_count", "stop_reason"))
                       for r, sample in zip(rows, samples))):
            raise ValueError(f"existing cell content/hash/count/order/fire validation failed at {out_dir}")
        print(f"matching completed cell exists: {out_dir}; skipped")
        return

    print(f"injection site: last prefill token; observed tail token ids/text={provenance['injection_tail_tokens']}; tail=1")
    vc.steering_fire_count(reset=True)
    records = []
    for i in range(0, len(prompts), a.batch_size):
        batch_prompts = prompts[i:i+a.batch_size]
        metas = vc.regenerate(batch_prompts, max_new_tokens=a.max_new_tokens,
                              temperature=0.0, diff_matrices=diff, batch_size=a.batch_size,
                              prefill_only=True, prefill_tail_len=1, return_metadata=True)
        if len(metas) != len(batch_prompts):
            raise RuntimeError(f"generation batch returned {len(metas)} rows for {len(batch_prompts)} prompts")
        records.extend({**samples[j], "generated_neutral": m["text"], "raw_generated_neutral": m["raw_text"],
                        "generated_token_count": m["generated_token_count"], "stop_reason": m["stop_reason"]}
                       for j, m in zip(range(i, i+len(metas)), metas))
    fires = vc.steering_fire_count(reset=True)
    expected = 0 if a.condition == "baseline" else (len(range(a.start, a.end)) * len(samples))
    if fires != expected:
        raise RuntimeError(f"steering fire count {fires}, expected {expected}; refusing to save")
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"data": records}, ensure_ascii=False, indent=2))
    os.replace(tmp, result_path)
    mtmp = manifest_path.with_suffix(".json.tmp")
    run = {"steering_fires": fires, "expected_steering_fires": expected,
           "row_count": len(records), "result_sha256": sha256(result_path)}
    mtmp.write_text(json.dumps({"configuration": provenance, "run": run}, ensure_ascii=False, indent=2))
    os.replace(mtmp, manifest_path)
    print(f"saved {result_path} ({len(records)} rows; fires={fires})")


if __name__ == "__main__":
    main()
