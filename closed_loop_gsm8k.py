#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Closed-loop RSN dopamine signal control during GSM8K generation.

Implements three flow-imitation waveform plans (A / B / C) plus baselines:
  - none  : no steering (α=0)
  - static: fixed α applied every step (e.g. α=-4)
  - A     : Tonic Floor Control — hold EMA ≥ x_prefill × floor_ratio
  - B     : Plateau Imitation  — EMA tracks a gently declining target curve
  - C     : Tonic + Micro-Phasic — Plan B tonic + spike-ratio damping
  - D     : Waveform Smoothing — α = -k1*(x_t-ema)/x_prefill, bidirectional, no target
  - E     : EMA Homeostasis — α = -k1*(ema-target)/xp, target=floor_ratio*xp
  - F     : Dual-EMA Filter — α = -k1*(fast_ema-slow_ema)/xp, MACD-style
  - G     : Bang-Bang Dead-Zone — fixed ±k1 outside ±k2*xp band around xp, else 0
  - H1    : Early Peak Boost — first 100 steps inject +k1 if ema < floor_ratio*xp, else 0

Hook design: each middle layer gets two hooks.
  - pre-hook  : injects α_{t-1} × nmd_mask[l] before K/V computation
  - post-hook : reads layer output; last middle layer computes x_t, updates
                EMA, decides α_t for use at next decode step (1-step lag).

The 1-step lag is the smallest causal lag possible: x_t can only be observed
after the layer outputs of step t are available, but at that point K/V for
step t are already cached. We therefore use α derived from x_t to steer
step t+1.

All plans use relative deviation (deviation / x_prefill) so hyperparameters
are scale-invariant across tasks and samples.

Usage:
  python closed_loop_gsm8k.py \\
    --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct --size 8B \\
    --test_file benchmark/gsm8k_test_sample.json \\
    --base_dir /data1/paveen/Dopamine/components \\
    --plan B --k1 2.0 --n_samples 300

@author: paveenhuang
"""

import os
import re
import gc
import json
import argparse
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_gsm8k
import utils


# Answer extraction / correctness / difficulty are imported from utils (canonical).
from utils import extract_gsm8k_answer, is_correct_gsm8k, gsm8k_difficulty


# ─────────────────────── Closed-loop controller ───────────────────────

class ClosedLoopController:
    """
    Two hooks per middle layer:
      pre-hook  : in-place add α_cur × dir[l] to last-token hidden state
                  before the layer runs, so K/V cache reflects the injection.
      post-hook : capture layer output's last-token vector. When the last
                  middle layer's post-hook fires, we have h_t[l] for all l,
                  so we compute x_t, update EMA, and decide α for next step.

    Time alignment:
      decode step t: pre-hooks inject α_cur (decided at step t-1) into all
                     middle layers; post-hooks collect h_t[l]; last-layer
                     post-hook computes x_t, updates EMA, sets α_cur for t+1.
      step 0: α_cur=0 (no signal yet). Prefill seeds EMA with x_prefill.

    Plans:
      none   – no injection
      static – fixed α every step
      A      – tonic floor: steer if EMA < x_prefill * floor_ratio
      B      – plateau: steer toward a gently declining target curve
      C      – plateau tonic + spike-ratio damping
      D      – waveform smoothing: α = -k1*(x_t-ema)/x_prefill, bidirectional, no target
      E      – EMA homeostasis: α = -k1*(ema-target)/x_prefill, target=x_prefill*floor_ratio (default 0.85)
    """

    def __init__(
        self,
        rsn_mask: np.ndarray,
        layer_start: int,
        layer_end: int,
        ema_alpha: float,
        plan: str,
        static_alpha: float = 0.0,
        k1: float = 2.0,
        k2: float = 1.0,
        floor_ratio: float = 0.65,
        plateau_end_ratio: float = 0.65,
        avg_gen_len: int = 400,
    ):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.ema_alpha = ema_alpha
        self.plan = plan
        self.static_alpha = static_alpha
        self.k1 = k1
        self.k2 = k2
        self.floor_ratio = floor_ratio
        self.plateau_end_ratio = plateau_end_ratio
        self.plateau_slope = (1.0 - plateau_end_ratio) / max(avg_gen_len, 1)
        self.avg_gen_len = avg_gen_len

        n_middle = layer_end - layer_start
        # directions: (n_middle, H) — raw NMD mask, same scale as steering.
        # See utils.mask_slice_for for the layer_start-1 offset rationale.
        self.directions = utils.mask_slice_for(rsn_mask, layer_start, layer_end).astype(np.float32)
        # Precompute torch tensors per layer for fast in-place injection
        # Stored as list[(H,)] on CPU; moved to device on first use
        self._dir_tensors: Optional[list] = None
        self._n_middle = n_middle

        # Runtime state (reset per sample)
        self._hooks: list = []
        self._pending: dict = {}
        self._ema_val: float = 0.0
        self._fast_ema_val: float = 0.0   # Plan F: fast EMA (α=0.7)
        self._x_prefill: float = 0.0
        self._step: int = 0
        self._alpha_cur: float = 0.0

        # Recording
        self.x_decode: list = []
        self.ema_decode: list = []
        self.alpha_t: list = []

    # ── helpers ──

    def _compute_alpha(self, x_t: float) -> float:
        plan = self.plan
        if plan == "none":
            return 0.0
        if plan == "static":
            return self.static_alpha

        xp = self._x_prefill
        if xp <= 0:
            return 0.0

        ema = self._ema_val

        if plan == "A":
            target = xp * self.floor_ratio
            deviation = target - ema
            return self.k1 * deviation / xp if deviation > 0 else 0.0

        if plan == "B":
            t = self._step
            target = xp * max(self.floor_ratio, 1.0 - self.plateau_slope * t)
            deviation = target - ema
            return self.k1 * deviation / xp if deviation > 0 else 0.0

        if plan == "C":
            t = self._step
            target = xp * max(self.floor_ratio, 1.0 - self.plateau_slope * t)
            deviation = target - ema
            alpha = self.k1 * deviation / xp if deviation > 0 else 0.0
            # Spike damping: if raw x_t is too large relative to EMA, negative steer
            if ema > 0:
                spike_ratio = x_t / ema
                if spike_ratio > 1.5:
                    alpha -= self.k2 * (spike_ratio - 1.5)
            return alpha

        if plan == "D":
            # Waveform Smoothing: push x_t toward EMA bidirectionally.
            # spike (x_t > ema) → negative α (suppress); dip (x_t < ema) → positive α (boost).
            # Clamp to [-8, 8] to prevent runaway feedback when x_t has large spikes.
            deviation = x_t - ema
            alpha = -self.k1 * deviation / xp
            return float(np.clip(alpha, -8.0, 8.0))

        if plan == "E":
            # EMA Homeostasis: steer EMA toward a fixed target (bidirectional).
            # Uses EMA (slow signal) as input — immune to 1-step lag oscillation.
            # target = x_prefill * floor_ratio (recommended: 0.85)
            target = xp * self.floor_ratio
            deviation = ema - target        # EMA above target → negative α; below → positive α
            return -self.k1 * deviation / xp

        if plan == "H1":
            # Early Peak Boost (task-onset dopamine burst):
            # During the first `boost_window` decode steps, inject +k1 whenever
            # ema is below `floor_ratio * xp` (here floor_ratio reused as peak target,
            # e.g. 1.5). After the boost window, no intervention — let natural
            # dynamics propagate the early commitment via KV cache.
            #
            # Rationale: dopamine task-onset burst (Schultz 1997) commits the
            # system to engagement; sustained intervention is unnecessary and may
            # disrupt downstream reasoning (cf. G k1=4.0 acc=58.3%).
            #
            # Hyperparams reused: k1 = injection magnitude, floor_ratio = peak
            # target ratio (set 1.5 for early peak), avg_gen_len defines window
            # via boost_window = min(100, avg_gen_len // 4).
            boost_window = 100
            peak_target = self.floor_ratio * xp
            if self._step < boost_window:
                if ema < peak_target:
                    return self.k1
                return 0.0
            return 0.0

        if plan == "H2":
            # Trapezoid Tracking: bidirectional proportional control toward a
            # piecewise-linear target trajectory.
            #   t in [0, 50)         : linear rise   1.00·xp -> peak·xp
            #   t in [50, 200)       : plateau      peak·xp
            #   t in [200, T)        : linear decay peak·xp -> end·xp
            #   t >= T               : clamp at end·xp
            # where T = avg_gen_len, peak = floor_ratio (default 1.25),
            # end = plateau_end_ratio (default 0.75).
            # Control law: alpha_t = k1 * (target_t - ema_t) / xp, then global
            # clamp. Bidirectional — pushes ema both up (when below target) and
            # down (when above target) to fit the prescribed waveform.
            rise_end = 50
            plateau_end = 200
            T = max(plateau_end + 1, int(self.avg_gen_len))
            peak = self.floor_ratio        # default 1.25
            end_lvl = self.plateau_end_ratio  # default 0.75
            t = self._step
            if t < rise_end:
                target_ratio = 1.0 + (peak - 1.0) * (t / rise_end)
            elif t < plateau_end:
                target_ratio = peak
            elif t < T:
                target_ratio = peak - (peak - end_lvl) * (t - plateau_end) / (T - plateau_end)
            else:
                target_ratio = end_lvl
            target = target_ratio * xp
            return self.k1 * (target - ema) / xp

        if plan == "H3":
            # Trapezoid Tracking v2 — Steeper Rise & Longer Plateau.
            # Aimed at fitting CoT-like shape more tightly than H2.
            #   t in [0, 30)         : linear rise   1.00·xp -> peak·xp
            #   t in [30, 120)       : plateau      peak·xp
            #   t in [120, T)        : linear decay peak·xp -> end·xp
            #   t >= T               : clamp at end·xp
            # where T = avg_gen_len, peak = floor_ratio (default 1.35 for H3),
            # end = plateau_end_ratio (default 0.75).
            # Same control law as H2: alpha_t = k1 * (target_t - ema_t) / xp.
            rise_end = 30
            plateau_end = 120
            T = max(plateau_end + 1, int(self.avg_gen_len))
            peak = self.floor_ratio
            end_lvl = self.plateau_end_ratio
            t = self._step
            if t < rise_end:
                target_ratio = 1.0 + (peak - 1.0) * (t / rise_end)
            elif t < plateau_end:
                target_ratio = peak
            elif t < T:
                target_ratio = peak - (peak - end_lvl) * (t - plateau_end) / (T - plateau_end)
            else:
                target_ratio = end_lvl
            target = target_ratio * xp
            return self.k1 * (target - ema) / xp

        if plan == "G":
            # Bang-Bang Dead-Zone: fixed ±k1 pulse outside a ±k2*xp band around xp.
            # Inside the band α=0 (preserves phasic structure).
            # Direction is automatic: ema > xp+band → -k1; ema < xp-band → +k1.
            # Bang-bang decouples α magnitude from deviation magnitude — this directly
            # addresses the "α too small to affect reasoning" failure mode of D/E/F.
            band = self.k2 * xp
            if ema > xp + band:
                return -self.k1
            if ema < xp - band:
                return +self.k1
            return 0.0

        if plan == "F":
            # Dual-EMA Waveform Filter (MACD-style):
            # fast_ema (α=0.7) tracks short-term trend; slow_ema (α=0.95) tracks baseline.
            # deviation = fast_ema - slow_ema: positive → current trend above baseline → suppress.
            # Using fast_ema (not raw x_t) as input reduces lag sensitivity vs Plan D:
            # fast_ema changes ~70% of x_t per step, so 1-step lag effect is buffered.
            # Stable when k1 ≲ 2.0 (open-loop gain c = 0.7 × k1 / xp × β < 1).
            fast_ema = self._fast_ema_val
            deviation = fast_ema - ema      # fast above slow → above-baseline trend
            return float(np.clip(-self.k1 * deviation / xp, -8.0, 8.0))

        return 0.0

    # ── hook registration ──

    def attach(self, decoder_layers: list, device):
        self._hooks = []
        self._pending = {}

        # Probe model dtype from the first decoder layer we'll hook
        # (use the same offset as decoder_layer_range below).
        first_hooked = utils.decoder_layer_range(self.layer_start, self.layer_end)[0]
        sample_param = next(decoder_layers[first_hooked].parameters())
        model_dtype = sample_param.dtype

        # Build direction tensors matching model dtype/device
        self._dir_tensors = [
            torch.tensor(self.directions[i], dtype=model_dtype, device=device)
            for i in range(self._n_middle)
        ]

        def make_pre_hook(local_idx: int):
            def pre_hook(_module, args):
                hs = args[0]
                is_prefill = hs.shape[1] > 1
                if is_prefill:
                    return
                # Inject α decided at previous decode step
                if self._alpha_cur != 0.0:
                    hs[0, -1, :].add_(self._dir_tensors[local_idx], alpha=self._alpha_cur)
            return pre_hook

        def make_post_hook(local_idx: int):
            is_last = (local_idx == self._n_middle - 1)

            def post_hook(_module, _args, output):
                hs = output[0] if isinstance(output, tuple) else output
                is_prefill = hs.shape[1] > 1

                # Capture last-token hidden state for projection.
                # Keep on device as float32 to avoid per-token GPU→CPU sync.
                self._pending[local_idx] = hs[0, -1, :].detach().to(torch.float32)

                if not is_last:
                    return

                # Last middle layer: all h_t[l] collected.
                vecs = torch.stack(
                    [self._pending[i] for i in range(self._n_middle)], dim=0
                )  # (n_middle, H), float32
                # x_t = mean_l( h_t[l] · mask[l] )
                dirs = torch.stack(self._dir_tensors, dim=0).to(torch.float32)
                x_t = float((vecs * dirs).sum(dim=-1).mean().item())
                self._pending.clear()

                if is_prefill:
                    self._x_prefill = x_t
                    self._ema_val = x_t
                    self._fast_ema_val = x_t
                    self._step = 0
                    self._alpha_cur = 0.0
                    return

                # Decode step t: update slow EMA and fast EMA with observed x_t
                self._ema_val = utils.ema_update(self._ema_val, x_t, self.ema_alpha)
                self._fast_ema_val = utils.ema_update(self._fast_ema_val, x_t, 0.3)
                # Decide α for next decode step (1-step lag)
                alpha_next = self._compute_alpha(x_t)

                # Record signal at step t (alpha_t recorded is what was injected
                # into THIS step, i.e. decided at step t-1)
                self.x_decode.append(x_t)
                self.ema_decode.append(float(self._ema_val))
                self.alpha_t.append(float(self._alpha_cur))

                self._alpha_cur = alpha_next
                self._step += 1

            return post_hook

        # See utils.decoder_layer_range for the layer_start-1 offset rationale.
        for local_idx, global_idx in enumerate(utils.decoder_layer_range(self.layer_start, self.layer_end)):
            layer = decoder_layers[global_idx]
            h_pre = layer.register_forward_pre_hook(make_pre_hook(local_idx))
            h_post = layer.register_forward_hook(make_post_hook(local_idx))
            self._hooks.append(h_pre)
            self._hooks.append(h_post)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset(self):
        self._pending = {}
        self._ema_val = 0.0
        self._fast_ema_val = 0.0
        self._x_prefill = 0.0
        self._step = 0
        self._alpha_cur = 0.0
        self.x_decode = []
        self.ema_decode = []
        self.alpha_t = []

    def get_signal(self) -> dict:
        return {
            "x_prefill": self._x_prefill,
            "x_decode": self.x_decode,
            "ema_decode": self.ema_decode,
            "alpha_t": self.alpha_t,
        }


# ─────────────────────── Generation with closed-loop control ───────────────────────

def generate_with_control(
    vc: VicundaModel,
    prompt: str,
    controller: ClosedLoopController,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    decoder_layers = vc._find_decoder_layers()
    device = vc.model.device

    controller.reset()
    controller.attach(decoder_layers, device)

    try:
        text = vc.generate_one(prompt, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_p=top_p)
    finally:
        controller.detach()

    return text


# ─────────────────────── Main ───────────────────────

def main():
    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load RSN mask
    mask_name = (
        f"{args.mask_type}_{args.percentage}"
        f"_{args.layer_start}_{args.layer_end}_{args.size}.npy"
    )
    mask_path = os.path.join(MASK_DIR, mask_name)
    rsn_mask = np.load(mask_path)
    print(f"Loaded RSN mask: {mask_path}, shape={rsn_mask.shape}")

    # Load data
    all_samples = utils.load_json(DATA_DIR)
    samples = all_samples[: args.n_samples]
    print(f"Loaded {len(samples)} samples | plan={args.plan}")

    # Templates
    templates = select_templates_gsm8k(suite="default", cot=args.cot)
    prompt_template, character = utils.select_role_prompt(templates, "neutral")

    # Controller
    controller = ClosedLoopController(
        rsn_mask=rsn_mask,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        ema_alpha=args.ema_alpha,
        plan=args.plan,
        static_alpha=args.static_alpha,
        k1=args.k1,
        k2=args.k2,
        floor_ratio=args.floor_ratio,
        plateau_end_ratio=args.plateau_end_ratio,
        avg_gen_len=args.avg_gen_len,
    )

    results = []

    for sample in tqdm(samples, desc=f"GSM8K closed-loop [{args.plan}]"):
        prompt = utils.render_role_prompt(prompt_template, sample["question"], character)

        with torch.no_grad():
            generated = generate_with_control(
                vc=vc,
                prompt=prompt,
                controller=controller,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        signal = controller.get_signal()
        pred = extract_gsm8k_answer(generated)
        correct = is_correct_gsm8k(pred, sample["answer"])
        difficulty = gsm8k_difficulty(sample["question"])

        results.append({
            "question": sample["question"],
            "gold_answer": sample["answer"],
            "pred_answer": pred,
            "correct": correct,
            "difficulty": difficulty,
            "generated": generated,
            "x_prefill": signal["x_prefill"],
            "x_decode": signal["x_decode"],
            "ema_decode": signal["ema_decode"],
            "alpha_t": signal["alpha_t"],
        })

    # Accuracy summary
    total = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    acc = correct_n / total * 100 if total else 0.0
    print(f"\n[Plan={args.plan}] Accuracy: {correct_n}/{total} = {acc:.2f}%")

    diff_stats: dict = {}
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

    # Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    cot_tag = "cot" if args.cot else "nocot"
    out_name = (
        f"closed_loop_gsm8k_{args.size}_{cot_tag}"
        f"_plan{args.plan}_k1{args.k1}_k2{args.k2}"
        f"_ema{args.ema_alpha}_L{args.layer_start}-{args.layer_end}.json"
    )
    out_path = os.path.join(SAVE_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(
            {
                "meta": {
                    "task": "gsm8k",
                    "model": args.model,
                    "size": args.size,
                    "plan": args.plan,
                    "static_alpha": args.static_alpha,
                    "k1": args.k1,
                    "k2": args.k2,
                    "floor_ratio": args.floor_ratio,
                    "plateau_end_ratio": args.plateau_end_ratio,
                    "avg_gen_len": args.avg_gen_len,
                    "ema_alpha": args.ema_alpha,
                    "layer_start": args.layer_start,
                    "layer_end": args.layer_end,
                    "mask": mask_name,
                    "n_samples": total,
                    "accuracy": round(acc, 2),
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
    print(f"Saved → {out_path}")

    del results
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Closed-loop RSN dopamine control for GSM8K"
    )
    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True)
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    # Mask
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--layer_start", type=int, default=11)
    parser.add_argument("--layer_end", type=int, default=20)
    # Data
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--cot", action="store_true")
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    # Tracking
    parser.add_argument("--ema_alpha", type=float, default=0.95)
    # Control plan
    parser.add_argument(
        "--plan", type=str, default="B",
        choices=["none", "static", "A", "B", "C", "D", "E", "F", "G", "H1", "H2", "H3"],
        help=(
            "none=no steering, static=fixed alpha, "
            "A=tonic floor, B=plateau imitation, C=tonic+phasic, "
            "D=waveform smoothing (bidirectional, no target), "
            "E=EMA homeostasis (bidirectional, target=xp*floor_ratio), "
            "F=dual-EMA filter (fast_ema-slow_ema, MACD-style), "
            "G=bang-bang dead-zone (fixed ±k1 outside ±k2*xp band, else 0), "
            "H1=early peak boost (first 100 steps, +k1 if ema<floor_ratio*xp)"
        ),
    )
    parser.add_argument("--static_alpha", type=float, default=0.0,
                        help="Alpha for plan=static")
    parser.add_argument("--k1", type=float, default=2.0,
                        help="Tonic correction gain")
    parser.add_argument("--k2", type=float, default=1.0,
                        help="Spike damping gain (Plan C only)")
    parser.add_argument("--floor_ratio", type=float, default=0.65,
                        help="Plan A/E: EMA target as fraction of x_prefill (use 0.85 for Plan E)")
    parser.add_argument("--plateau_end_ratio", type=float, default=0.65,
                        help="Plan B/C: target EMA at end of generation as fraction of x_prefill")
    parser.add_argument("--avg_gen_len", type=int, default=400,
                        help="Expected generation length used to compute plateau slope")
    # Paths
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument("--ans_file", type=str, default="closed_loop")

    args = parser.parse_args()

    if args.base_dir:
        BASE = args.base_dir
    else:
        BASE = f"/{args.data}/paveen/Dopamine/components"

    DATA_DIR = os.path.join(BASE, args.test_file)
    MASK_DIR = os.path.join(BASE, "mask", f"{args.hs}_{args.type}_logits")
    SAVE_DIR = os.path.join(BASE, args.model, args.ans_file)

    main()
