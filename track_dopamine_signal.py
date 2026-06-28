#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 & 1: Real-time RSN dopamine signal tracking during generation.

For each sample, records:
  - x_prefill : RSN projection at the last prefill token (initial DA level)
  - x_decode  : RSN projection at each decode step (shape: [T])
  - ema_decode : EMA-smoothed signal curve (shape: [T])
  - generated  : decoded text
  - correct    : whether the predicted answer matches ground truth
  - difficulty : "Easy"/"Medium"/"Hard" for GSM8K, "Level X" for MATH

The RSN direction is the sparse NMD mask (nmd_0.5_11_20_{size}.npy),
layers 11-20 averaged to a single scalar per step.

Usage (GSM8K):
  python track_dopamine_signal.py \
    --task gsm8k \
    --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct --size 8B \
    --test_file benchmark/gsm8k_test_sample.json \
    --base_dir /data1/paveen/Dopamine/components \
    --ema_alpha 0.95 --max_new_tokens 512

Usage (MATH):
  python track_dopamine_signal.py \
    --task math \
    --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct --size 8B \
    --test_file benchmark/math_test_sample.json \
    --base_dir /data1/paveen/Dopamine/components \
    --ema_alpha 0.95 --max_new_tokens 1024 --cot
"""

import os
import re
import gc
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_gsm8k, build_math_suite
import utils


# Answer extraction / correctness / difficulty are imported from utils (canonical).
# Re-exported here so track_hidden_states.py can keep importing them from this module.
from utils import (
    extract_gsm8k_answer,
    normalize_gsm8k,
    is_correct_gsm8k,
    extract_boxed,
    extract_math_answer,
    normalize_math,
    is_correct_math,
    gsm8k_difficulty,
)


# ─────────────────────── Signal tracker ───────────────────────

class DopamineTracker:
    """
    Registers read-only hooks on middle layers to record RSN projections
    during both prefill and decode.

    rsn_mask: np.ndarray, shape (num_model_layers, hidden_dim)
              The sparse NMD mask (nmd_0.5_{start}_{end}_{size}.npy).
              Rows outside [layer_start, layer_end) are all-zero.
    layer_start, layer_end: int — the RSN middle-layer range (e.g. 11, 20)
    ema_alpha: float — EMA smoothing coefficient (0 < ema_alpha < 1)
    """

    def __init__(
        self,
        rsn_mask: np.ndarray,
        layer_start: int,
        layer_end: int,
        ema_alpha: float = 0.95,
        steer_alpha: float = 0.0,
    ):
        self.rsn_mask = rsn_mask          # (L_model, H)
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.ema_alpha = ema_alpha

        # Raw mask vectors (no normalization) — same scale as steering.
        # See utils.mask_slice_for for the layer_start-1 offset rationale.
        self.directions = utils.mask_slice_for(rsn_mask, layer_start, layer_end).astype(np.float32)

        # Optional prefill-only steering. steer_alpha==0 (default) → pure
        # observer (byte-identical to the old read-only tracker). When != 0,
        # inject α × mask[l] into the last prompt token of each middle layer's
        # OUTPUT (L>1 only) — same output-side injection as
        # get_answer_regenerate_*.py / llms._regenerate_prefill_only, so the
        # mask (extracted from decoder_layers[l] OUTPUT) lands on the matching
        # layer (NO one-layer offset; a pre-hook would misalign by one layer).
        # Injection happens INSIDE the observation hook, before projecting, so
        # the recorded x_t reflects the POST-injection signal (matches the
        # co-design identity x_{t+1} += α·‖mask‖²). steer_dirs == directions.
        self.steer_alpha = steer_alpha
        self.steer_dirs = self.directions if steer_alpha != 0.0 else None

        # Runtime state (reset per sample)
        self._hooks = []
        self._step_buffer: list[float] = []   # raw projection per decode step
        self._ema_buffer: list[float] = []
        self._ema_val: float = 0.0
        self._prefill_proj: float | None = None
        self._layer_projs: list[list[float]] = []  # per-step, per-middle-layer

    # ── internal helpers ──

    def _project(self, hs_np: np.ndarray) -> float:
        """
        hs_np: (n_middle, H) — one hidden state vector per middle layer.
        Returns: mean projection across middle layers (scalar).
        """
        return utils.project_rsn_numpy(hs_np, self.directions)

    def _project_per_layer(self, hs_np: np.ndarray) -> list[float]:
        projs = np.sum(hs_np * self.directions, axis=-1)  # (n_middle,)
        return projs.tolist()

    # ── hook registration ──

    def attach(self, decoder_layers: list):
        """Register read-only hooks on middle layers."""
        self._hooks = []

        n_middle = self.layer_end - self.layer_start

        # Collect hidden states from middle layers per forward call
        # We use a shared dict keyed by layer index to accumulate within one step
        self._pending: dict[int, np.ndarray] = {}
        self._n_middle = n_middle

        def make_hook(local_idx: int):
            def hook(_module, _input, output):
                hs = output[0] if isinstance(output, tuple) else output
                # hs: (B=1, L, H)  — batch_size forced to 1 during tracking
                modified = False

                # Prefill-only steering: inject α × mask[l] into the last prompt
                # token of THIS layer's output (L>1), in-place so it propagates
                # downstream. Done before projecting → x_t observes post-injection.
                if self.steer_alpha != 0.0 and hs.shape[1] > 1:
                    diff = torch.as_tensor(
                        self.steer_alpha * self.steer_dirs[local_idx],
                        device=hs.device, dtype=hs.dtype,
                    )
                    hs[:, -1, :] += diff
                    modified = True

                vec = hs[0, -1, :].detach().float().cpu().numpy()  # (H,)
                self._pending[local_idx] = vec

                # Once all middle layers collected for this forward call:
                if len(self._pending) == self._n_middle:
                    hs_np = np.stack(
                        [self._pending[i] for i in range(self._n_middle)], axis=0
                    )  # (n_middle, H)

                    is_prefill = hs[0].shape[0] > 1  # L > 1 → prefill

                    if is_prefill:
                        # Record prefill snapshot (last token)
                        self._prefill_proj = self._project(hs_np)
                        self._prefill_per_layer = self._project_per_layer(hs_np)
                        # Seed EMA with prefill value
                        self._ema_val = self._prefill_proj
                    else:
                        # Decode step
                        raw = self._project(hs_np)
                        per_layer = self._project_per_layer(hs_np)
                        self._ema_val = utils.ema_update(self._ema_val, raw, self.ema_alpha)
                        self._step_buffer.append(raw)
                        self._ema_buffer.append(self._ema_val)
                        self._layer_projs.append(per_layer)

                    self._pending.clear()

                if modified:
                    if isinstance(output, tuple):
                        return (hs,) + output[1:]
                    return hs
                return None

            return hook

        # See utils.decoder_layer_range for the layer_start-1 offset rationale.
        for local_idx, global_idx in enumerate(utils.decoder_layer_range(self.layer_start, self.layer_end)):
            h = decoder_layers[global_idx].register_forward_hook(make_hook(local_idx))
            self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset(self):
        self._step_buffer = []
        self._ema_buffer = []
        self._ema_val = 0.0
        self._prefill_proj = None
        self._prefill_per_layer = []
        self._layer_projs = []
        self._pending = {}

    def get_signal(self) -> dict:
        return {
            "x_prefill": self._prefill_proj,
            "x_prefill_per_layer": getattr(self, "_prefill_per_layer", []),
            "x_decode": self._step_buffer,
            "ema_decode": self._ema_buffer,
            "x_decode_per_layer": self._layer_projs,
        }


# ─────────────────────── Generation with tracking ───────────────────────

def generate_with_tracking(
    vc: VicundaModel,
    prompt: str,
    tracker: DopamineTracker,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Run greedy/sampling generation for a single prompt while tracking RSN signal.
    Batch size is fixed to 1 so that hook logic is straightforward.
    """
    decoder_layers = vc._find_decoder_layers()
    tracker.reset()
    tracker.attach(decoder_layers)

    try:
        text = vc.generate_one(prompt, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_p=top_p)
    finally:
        tracker.detach()

    return text


# ─────────────────────── Main ───────────────────────

def main():
    # ── load model ──
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # ── load RSN mask ──
    mask_name = f"{args.mask_type}_{args.percentage}_{args.layer_start}_{args.layer_end}_{args.size}.npy"
    mask_path = os.path.join(MASK_DIR, mask_name)
    rsn_mask = np.load(mask_path)  # (num_model_layers, H)
    print(f"Loaded RSN mask: {mask_path}, shape={rsn_mask.shape}")

    # ── load data ──
    all_samples = utils.load_json(DATA_DIR)
    samples = all_samples[: args.n_samples]
    print(f"Loaded {len(samples)} samples from {DATA_DIR}")

    # ── build prompt template ──
    if args.task == "gsm8k":
        templates = select_templates_gsm8k(suite="default", cot=args.cot)
    else:
        templates = build_math_suite(cot=args.cot)

    # ── role selection ──
    prompt_template, character = utils.select_role_prompt(templates, args.role)
    print(f"Role: {args.role} (character={character})")

    # ── tracker ──
    tracker = DopamineTracker(
        rsn_mask=rsn_mask,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        ema_alpha=args.ema_alpha,
        steer_alpha=args.alpha,
    )
    if args.alpha != 0.0:
        print(f"Prefill-only steering ON: α={args.alpha} × {args.mask_type} mask, "
              f"output-side, layers {args.layer_start}-{args.layer_end}")

    results = []

    for sample in tqdm(samples, desc=f"Tracking [{args.task}|{args.role}]"):
        prompt = utils.render_role_prompt(prompt_template, sample["question"], character)

        with torch.no_grad():
            generated = generate_with_tracking(
                vc=vc,
                prompt=prompt,
                tracker=tracker,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        signal = tracker.get_signal()

        # ── correctness & difficulty ──
        if args.task == "gsm8k":
            pred = extract_gsm8k_answer(generated)
            correct = is_correct_gsm8k(pred, sample["answer"])
            difficulty = gsm8k_difficulty(sample["question"])
        else:
            gold_boxed = extract_boxed(sample["answer"])
            pred = extract_math_answer(generated)
            correct = is_correct_math(pred, gold_boxed) if gold_boxed else False
            difficulty = sample.get("level", "")

        results.append({
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "pred_answer": pred,
            "correct": correct,
            "difficulty": difficulty,
            "generated": generated,
            # signal fields
            "x_prefill": signal["x_prefill"],
            "x_prefill_per_layer": signal["x_prefill_per_layer"],
            "x_decode": signal["x_decode"],
            "ema_decode": signal["ema_decode"],
            "x_decode_per_layer": signal["x_decode_per_layer"],
        })

    # ── accuracy summary ──
    total = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    print(f"\nAccuracy: {correct_n}/{total} = {correct_n/total*100:.2f}%")

    # ── per-difficulty breakdown ──
    diff_stats: dict[str, dict] = {}
    for r in results:
        d = r["difficulty"]
        if d not in diff_stats:
            diff_stats[d] = {"correct": 0, "total": 0}
        diff_stats[d]["total"] += 1
        if r["correct"]:
            diff_stats[d]["correct"] += 1
    for d, s in sorted(diff_stats.items()):
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {d}: {s['correct']}/{s['total']} = {pct:.1f}%")

    # ── save ──
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = f"cot" if args.cot else "nocot"
    # filename: include role + mask_type to disambiguate the validation runs.
    # Backward compat: when role=neutral and mask_type=nmd, omit those tags so
    # legacy filenames keep working.
    role_tag = "" if args.role == "neutral" else f"_{args.role}"
    mask_tag = "" if args.mask_type == "nmd" else f"_{args.mask_type}"
    # α-tag: omitted at α=0 (legacy/baseline filenames unchanged). Matches the
    # _aneg{n}/_a{n} convention used by the old track_hidden_states.py.
    if args.alpha == 0.0:
        alpha_tag = ""
    elif args.alpha < 0:
        alpha_tag = f"_aneg{abs(args.alpha):g}"
    else:
        alpha_tag = f"_a{args.alpha:g}"
    out_path = os.path.join(
        SAVE_DIR,
        f"dopamine_signal_{args.task}_{args.size}_{tag}{role_tag}{alpha_tag}{mask_tag}"
        f"_ema{args.ema_alpha}_L{args.layer_start}-{args.layer_end}.json",
    )
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(
            {
                "meta": {
                    "task": args.task,
                    "model": args.model,
                    "size": args.size,
                    "ema_alpha": args.ema_alpha,
                    "layer_start": args.layer_start,
                    "layer_end": args.layer_end,
                    "mask": mask_name,
                    "mask_type": args.mask_type,
                    "role": args.role,
                    "character": character,
                    "steer_alpha": args.alpha,
                    "steer_mode": "prefill_only_output" if args.alpha != 0.0 else "none",
                    "n_samples": len(results),
                    "accuracy": round(correct_n / total * 100, 2),
                    "cot": args.cot,
                    "prompt_template": prompt_template,
                },
                "diff_stats": diff_stats,
                "data": results,
            },
            fw,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved → {out_path}")

    del results
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: RSN dopamine signal tracking")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "math"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True, help="Hidden state prefix for mask dir")
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--layer_start", type=int, default=11)
    parser.add_argument("--layer_end", type=int, default=20)
    parser.add_argument("--ema_alpha", type=float, default=0.95,
                        help="EMA smoothing coefficient (higher = smoother)")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Prefill-only steering strength (× mask). 0 = pure "
                             "observer (default, byte-identical to legacy runs); "
                             "!=0 = output-side injection on layers [start,end).")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument(
        "--role",
        type=str,
        default="neutral",
        choices=["neutral", "expert", "non_expert", "primary_teacher"],
        help="neutral: no role; expert: 'an expert'; non_expert: 'a non expert'; "
             "primary_teacher: 'a primary school teacher'.",
    )
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    args = parser.parse_args()

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/Dopamine/components"

    DATA_DIR = os.path.join(BASE, args.test_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_DIR = os.path.join(BASE, args.model, "dopamine_signal")

    main()
