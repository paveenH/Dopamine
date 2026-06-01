#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hidden-state recorder for RSN signal proxy validation.

For each generated sample, stores raw last-token hidden states at every middle
layer for both prefill and decode steps. Output is a single HDF5 file per
(role × task × mode) combination, intended for offline reanalysis with
arbitrary projection / aggregation strategies (NMD mask, random mask,
per-layer ablation, alternative aggregations, etc.).

Output structure:
  hs_<task>_<size>_<mode>_<role><mask_tag>.h5
  /meta                       (file-level attrs: role, character, ema_alpha,
                               num_layers, n_stored_layers,
                               final_layer_idx_stored, stored_layer_indices, ...)
  /samples
      /0000
          prefill_hs          (P, n_stored, H)  fp16
              — every prompt token HS, but only at the *selected* layers:
                middle layers [layer_start, layer_end) + final decoder layer.
                Storage index `final_layer_idx_stored` (= n_middle) is the
                final layer; the first n_middle slots are the middle layers.
          decode_hs           (T, n_stored, H)  fp16
              — every decode step HS at the same selected layers
          x_prefill_proj      ()    fp32  — NMD-projected prefill scalar (sanity)
          x_decode_proj       (T,)  fp32  — NMD-projected per-step scalar (sanity)
          ema_decode_proj     (T,)  fp32  — EMA of x_decode_proj (sanity)
          attrs:
            correct       : int (0/1)
            generated     : str
            pred_answer   : str
            gold_answer   : str
            difficulty    : str
            question      : str
            question_idx  : int
      /0001 ...

The "sanity" projections use the loaded mask (default NMD); they let downstream
analysis verify offline-recomputed projections match the on-the-fly tracker.

Usage (matches track_dopamine_signal.py argument layout):
  python track_hidden_states.py \
    --task gsm8k --model llama3 --size 8B --hs llama3 \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --test_file benchmark/gsm8k_test_sample.json \
    --base_dir /data1/paveen/Dopamine/components \
    --role expert --max_new_tokens 768 --n_samples 300
"""

import os
import re
import gc
import argparse

import numpy as np
import h5py
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_gsm8k, build_math_suite
import utils
from utils import (
    extract_gsm8k_answer,
    is_correct_gsm8k,
    extract_boxed,
    extract_math_answer,
    is_correct_math,
    gsm8k_difficulty,
)


# ─────────────────────── Hidden-state recorder ───────────────────────

class HiddenStateRecorder:
    """
    Captures hidden states at **middle layers + the final decoder layer** for
    both prefill and decode. Also computes a sanity NMD-projected scalar + EMA
    on the fly using the supplied mask (only the middle layers participate in
    the projection).

    Stored layers (in HDF5):
      indices 0 .. n_middle-1   → middle layers [layer_start, layer_end)
      index   n_middle          → final decoder layer (used for logits)
    → total `n_stored = n_middle + 1` layers

    Layout for each sample:
      prefill_hs : (P, n_stored, H) fp16   — every prompt token
      decode_hs  : (T, n_stored, H) fp16   — every generated token

    Meta stores `final_layer_idx` (= n_middle) so downstream scripts can find
    the final-layer HS without ambiguity.
    """

    def __init__(
        self,
        rsn_mask: np.ndarray,
        layer_start: int,
        layer_end: int,
        ema_alpha: float = 0.95,
    ):
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.n_middle = layer_end - layer_start
        self.ema_alpha = ema_alpha

        # Sanity-projection direction (n_middle, H) on CPU fp32 — used only for
        # NMD-projected sanity scalar. Stored HS keeps the full per-layer state
        # of middle layers so any other mask can be re-projected offline.
        # See utils.mask_slice_for for the layer_start-1 offset rationale.
        self.directions = utils.mask_slice_for(rsn_mask, layer_start, layer_end).astype(np.float32)

        # Discovered at attach() time
        self.num_layers: int | None = None       # total decoder layers in model
        self.final_layer_idx_model: int | None = None   # model-space index of final layer
        self.layer_indices: list[int] = []       # model-space layer indices we hook
        self.final_layer_idx_stored: int | None = None  # storage-space index of final layer

        # Reset per sample
        self._hooks = []
        self._pending: dict[int, np.ndarray] = {}     # storage_idx → (L, H)
        self._prefill_hs: np.ndarray | None = None    # (P, n_stored, H)
        self._decode_hs: list[np.ndarray] = []        # list of (n_stored, H)
        self._x_prefill_proj: float | None = None
        self._x_decode_proj: list[float] = []
        self._ema_decode_proj: list[float] = []
        self._ema_val: float = 0.0

    # ── projection helper (middle-layer NMD sanity scalar) ──
    def _project_middle_last(self, stored_hs_last: np.ndarray) -> float:
        """
        stored_hs_last: (n_stored, H) — last-token HS at each stored layer.
        First n_middle entries are the middle layers; uses those for NMD proj.
        """
        middle = stored_hs_last[: self.n_middle]  # (n_middle, H)
        return utils.project_rsn_numpy(middle, self.directions)

    # ── hook registration: hook middle layers + final layer only ──
    def attach(self, decoder_layers):
        self._hooks = []
        self._pending = {}
        self.num_layers = len(decoder_layers)
        self.final_layer_idx_model = self.num_layers - 1

        # Build the list of decoder-layer indices we want to capture:
        #   middle layers + final layer
        # See utils.decoder_layer_range for the layer_start-1 offset rationale.
        middle_idxs = list(utils.decoder_layer_range(self.layer_start, self.layer_end))
        if self.final_layer_idx_model in middle_idxs:
            # Final layer is inside middle range — still store it twice for
            # schema stability (last slot is always the final layer).
            pass
        self.layer_indices = middle_idxs + [self.final_layer_idx_model]
        self.final_layer_idx_stored = len(self.layer_indices) - 1   # always n_middle

        # Map model-space index → storage-space index. If final layer is also
        # in middle range, that model index maps to multiple storage slots —
        # only register hook once per model index but copy into both slots.
        model_to_storage: dict[int, list[int]] = {}
        for storage_i, model_i in enumerate(self.layer_indices):
            model_to_storage.setdefault(model_i, []).append(storage_i)

        def make_hook(model_idx: int):
            def hook(_module, _input, output):
                hs = output[0] if isinstance(output, tuple) else output
                # hs: (B=1, L, H). Store ALL tokens of this forward (fp16).
                hs_np = hs[0].detach().to(torch.float16).cpu().numpy()
                for storage_i in model_to_storage[model_idx]:
                    self._pending[storage_i] = hs_np

                # When all storage slots filled, process the step.
                if len(self._pending) == len(self.layer_indices):
                    layers_LH = np.stack(
                        [self._pending[i] for i in range(len(self.layer_indices))],
                        axis=0,
                    )  # (n_stored, L, H)
                    L = layers_LH.shape[1]
                    is_prefill = L > 1

                    if is_prefill:
                        # Reshape to (P, n_stored, H) for token-major access
                        self._prefill_hs = layers_LH.transpose(1, 0, 2).copy()

                        last_tok = layers_LH[:, -1, :].astype(np.float32)
                        self._x_prefill_proj = self._project_middle_last(last_tok)
                        self._ema_val = self._x_prefill_proj
                    else:
                        last_tok = layers_LH[:, -1, :].astype(np.float32)
                        self._decode_hs.append(layers_LH[:, -1, :].copy())

                        proj = self._project_middle_last(last_tok)
                        self._ema_val = utils.ema_update(self._ema_val, proj, self.ema_alpha)
                        self._x_decode_proj.append(proj)
                        self._ema_decode_proj.append(self._ema_val)

                    self._pending.clear()

            return hook

        # Register hook once per unique model layer index
        for model_idx in sorted(model_to_storage.keys()):
            h = decoder_layers[model_idx].register_forward_hook(make_hook(model_idx))
            self._hooks.append(h)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset(self):
        self._pending = {}
        self._prefill_hs = None
        self._decode_hs = []
        self._x_prefill_proj = None
        self._x_decode_proj = []
        self._ema_decode_proj = []
        self._ema_val = 0.0

    def get_record(self) -> dict:
        H = self.directions.shape[-1]
        n_stored = len(self.layer_indices) if self.layer_indices else 0
        decode_hs = (
            np.stack(self._decode_hs, axis=0)             # (T, n_stored, H)
            if self._decode_hs
            else np.zeros((0, n_stored, H), dtype=np.float16)
        )
        return {
            "prefill_hs":      self._prefill_hs,           # (P, n_stored, H) fp16
            "decode_hs":       decode_hs,                  # (T, n_stored, H) fp16
            "x_prefill_proj":  np.float32(self._x_prefill_proj or 0.0),
            "x_decode_proj":   np.array(self._x_decode_proj, dtype=np.float32),
            "ema_decode_proj": np.array(self._ema_decode_proj, dtype=np.float32),
        }


# ─────────────────────── Generation ───────────────────────

def generate_with_recording(
    vc: VicundaModel,
    prompt: str,
    recorder: HiddenStateRecorder,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    decoder_layers = vc._find_decoder_layers()
    recorder.reset()
    recorder.attach(decoder_layers)
    try:
        text = vc.generate_one(prompt, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_p=top_p)
    finally:
        recorder.detach()
    return text


# ─────────────────────── Main ───────────────────────

def main():
    # ── load model ──
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # ── load RSN mask (only for sanity projection — analysis re-projects offline) ──
    mask_name = f"{args.mask_type}_{args.percentage}_{args.layer_start}_{args.layer_end}_{args.size}.npy"
    mask_path = os.path.join(MASK_DIR, mask_name)
    rsn_mask = np.load(mask_path)
    print(f"Loaded sanity mask: {mask_path}, shape={rsn_mask.shape}")

    # ── load data ──
    all_samples = utils.load_json(DATA_DIR)
    samples = all_samples[: args.n_samples]
    print(f"Loaded {len(samples)} samples from {DATA_DIR}")

    # ── prompt template ──
    if args.task == "gsm8k":
        templates = select_templates_gsm8k(suite="default", cot=args.cot)
    else:
        templates = build_math_suite(cot=args.cot)

    prompt_template, character = utils.select_role_prompt(templates, args.role)
    print(f"Role: {args.role} (character={character})")

    # ── recorder ──
    recorder = HiddenStateRecorder(
        rsn_mask=rsn_mask,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        ema_alpha=args.ema_alpha,
    )

    # ── output path ──
    os.makedirs(SAVE_DIR, exist_ok=True)
    mode_tag = "cot" if args.cot else "nocot"
    role_tag = "" if args.role == "neutral" else f"_{args.role}"
    mask_tag = "" if args.mask_type == "nmd" else f"_{args.mask_type}"
    out_path = os.path.join(
        SAVE_DIR,
        f"hs_{args.task}_{args.size}_{mode_tag}{role_tag}{mask_tag}"
        f"_L{args.layer_start}-{args.layer_end}.h5",
    )

    # ── HDF5 file ──
    correct_n = 0
    total = 0
    diff_stats: dict[str, dict] = {}

    with h5py.File(out_path, "w") as fh:
        # File-level metadata
        meta = fh.create_group("meta")
        meta.attrs["task"] = args.task
        meta.attrs["model"] = args.model
        meta.attrs["size"] = args.size
        meta.attrs["role"] = args.role
        meta.attrs["character"] = character if character else "(none)"
        meta.attrs["prompt_template"] = prompt_template
        meta.attrs["cot"] = bool(args.cot)
        meta.attrs["layer_start"] = args.layer_start
        meta.attrs["layer_end"] = args.layer_end
        meta.attrs["ema_alpha"] = args.ema_alpha
        meta.attrs["sanity_mask"] = mask_name
        meta.attrs["max_new_tokens"] = args.max_new_tokens
        meta.attrs["n_samples_planned"] = len(samples)

        samples_grp = fh.create_group("samples")

        for idx, sample in enumerate(tqdm(samples, desc=f"Recording [{args.task}|{args.role}]")):
            prompt = utils.render_role_prompt(prompt_template, sample["question"], character)

            with torch.no_grad():
                generated = generate_with_recording(
                    vc=vc, prompt=prompt, recorder=recorder,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                )

            rec = recorder.get_record()

            # correctness + difficulty
            if args.task == "gsm8k":
                pred = extract_gsm8k_answer(generated)
                correct = is_correct_gsm8k(pred, sample["answer"])
                difficulty = gsm8k_difficulty(sample["question"])
            else:
                gold_boxed = extract_boxed(sample["answer"])
                pred = extract_math_answer(generated)
                correct = is_correct_math(pred, gold_boxed) if gold_boxed else False
                difficulty = sample.get("level", "")

            total += 1
            if correct: correct_n += 1
            d_stat = diff_stats.setdefault(difficulty, {"total": 0, "correct": 0})
            d_stat["total"] += 1
            if correct: d_stat["correct"] += 1

            # write per-sample group
            g = samples_grp.create_group(f"{idx:04d}")
            g.create_dataset("prefill_hs", data=rec["prefill_hs"], compression="gzip", compression_opts=4)
            g.create_dataset("decode_hs",  data=rec["decode_hs"],  compression="gzip", compression_opts=4)
            g.create_dataset("x_prefill_proj",  data=rec["x_prefill_proj"])
            g.create_dataset("x_decode_proj",   data=rec["x_decode_proj"])
            g.create_dataset("ema_decode_proj", data=rec["ema_decode_proj"])
            g.attrs["correct"]      = int(correct)
            g.attrs["generated"]    = generated[:4000]
            g.attrs["pred_answer"]  = str(pred)
            g.attrs["gold_answer"]  = str(sample.get("answer", ""))
            g.attrs["difficulty"]   = difficulty
            g.attrs["question"]     = sample["question"][:2000]
            g.attrs["question_idx"] = idx

            # incremental flush every 50 samples to reduce loss-on-crash risk
            if (idx + 1) % 50 == 0:
                fh.flush()
                gc.collect()

        # final accuracy attrs
        meta.attrs["n_samples_done"] = total
        meta.attrs["accuracy"]       = round(correct_n / max(total, 1) * 100, 2)
        if recorder.num_layers is not None:
            meta.attrs["num_layers"]              = recorder.num_layers
            meta.attrs["final_layer_idx_model"]   = recorder.final_layer_idx_model
            meta.attrs["final_layer_idx_stored"]  = recorder.final_layer_idx_stored
            meta.attrs["stored_layer_indices"]    = np.array(recorder.layer_indices, dtype=np.int32)
            meta.attrs["n_stored_layers"]         = len(recorder.layer_indices)

    # ── stdout summary ──
    print(f"\nAccuracy: {correct_n}/{total} = {correct_n/max(total,1)*100:.2f}%")
    for d, s in sorted(diff_stats.items()):
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {d}: {s['correct']}/{s['total']} = {pct:.1f}%")
    print(f"\nSaved → {out_path}")


# ─────────────────────── CLI ───────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1b: raw hidden-state recorder")
    parser.add_argument("--task", type=str, required=True, choices=["gsm8k", "math"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--hs", type=str, required=True, help="hidden state prefix for mask dir")
    parser.add_argument("--size", type=str, required=True)
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--layer_start", type=int, default=11)
    parser.add_argument("--layer_end", type=int, default=20)
    parser.add_argument("--ema_alpha", type=float, default=0.95)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])
    parser.add_argument(
        "--role",
        type=str,
        default="neutral",
        choices=["neutral", "expert", "non_expert", "primary_teacher"],
        help="neutral: no role; expert: 'an expert'; "
             "non_expert: 'a non expert'; primary_teacher: 'a primary school teacher'.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override default HS output dir (default: {base_dir}/hidden_states/{task}/)",
    )
    args = parser.parse_args()

    # Resolve directories (mirror track_dopamine_signal.py layout)
    BASE_DIR = (
        args.base_dir
        if args.base_dir
        else f"/{args.data}/paveen/Dopamine/components"
    )
    MASK_DIR = os.path.join(BASE_DIR, "mask", f"{args.hs}_{args.type}_logits")
    DATA_DIR = os.path.join(BASE_DIR, args.test_file)
    SAVE_DIR = (
        args.save_dir
        if args.save_dir
        else os.path.join(BASE_DIR, "hidden_states", args.task)
    )

    print(f"DATA_DIR = {DATA_DIR}")
    print(f"MASK_DIR = {MASK_DIR}")
    print(f"SAVE_DIR = {SAVE_DIR}")

    main()
