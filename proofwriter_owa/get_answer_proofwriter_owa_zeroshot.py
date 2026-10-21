#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_proofwriter_owa_zeroshot.py -- ProofWriter-OWA CHAT ZERO-SHOT
generation (protocol `proofwriter-owa-zs-chat-v1`).

INDEPENDENT EXPERIMENT. It does not touch, reuse the outputs of, or redefine
the closed bare-string line (`proofwriter-owa-v0`) or the 1-shot chat line
(`proofwriter-owa-chat-v1`). Only the 300-item blind manifest, SYSTEM_RULES
and the frozen parser are shared.

DESIGN (all fixed in code, none a CLI knob)
  prompt   : prompt_zeroshot.build_user_prompt -- 0-shot, no exemplar.
             No-CoT = rules + Theory/Question + answer-line instruction;
             CoT    = the same string + "Let's think step by step.\n".
  wrapping : ONE user message through tokenizer.apply_chat_template(...,
             add_generation_prompt=True); no assistant prefix. A leading BOS
             already serialised as text is stripped (vc.regenerate re-adds it
             via add_special_tokens=True) and double-BOS is a hard stop.
  cells    : per model, "a0" (alpha=0) and "wp" (the GSM8K workpoint:
             llama3 -6 band [11,20), qwen2.5 +8 band [16,22)). No other alpha
             can be requested. The workpoint cell is refused until a PASSED
             format-gate file exists for that model/condition (gold-free; see
             eval_proofwriter_owa_zeroshot.py --stage gate).
  generate : greedy, temperature 0, top_p 1, 1024 new tokens, batch 8,
             prefill-only, tail 1 -> injection on the LAST token of the
             wrapped prompt. alpha=0 takes the same path with an all-zero
             full-length diff matrix. Fire counts must be exactly
             0 / L*300 (llama -6: 2700, qwen +8: 1800) or the cell is not
             written.

INJECTION SITE. The final prompt token id/text and the tail token sequence are
READ from the real tokenizer for all 300 prompts and stored in
meta.injection_site; the two models' assistant-header end tokens are NOT
assumed equal. The final id must be identical across all prompts.

PROVENANCE. meta records n_shot=0, the true cot flag, protocol, model, mask /
manifest / prompt / chat-template hashes, generation config, fire counts, host,
GPU names, sharding and library versions. An existing output is reused only if
validate_cell() passes against this run's expected config (identity, order,
hashes, fires); otherwise the run aborts -- it never skips on path alone and
never overwrites. Files are written atomically.

Modes: `--mode check` loads the TOKENIZER only (no weights, no GPU) and prints
the prompt / injection-site report; `--mode run` generates.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zs_common as zc  # noqa: E402
import prompt_zeroshot as pz  # noqa: E402


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def strip_leading_bos(tokenizer, text: str) -> str:
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def wrap_prompts(tokenizer, bare_prompts):
    """Single user message -> chat template with generation prompt, nothing
    appended after it."""
    out = []
    for p in bare_prompts:
        w = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True)
        out.append(strip_leading_bos(tokenizer, w))
    return out


def injection_site_report(tokenizer, chat_prompts, n_tail: int = 8) -> dict:
    """Tokenize EXACTLY as vc.regenerate does (add_special_tokens=True) and
    read out where the single-token injection lands. Hard stops: double BOS,
    BOS missing where the tokenizer defines one, non-uniform final token."""
    bos_id = getattr(tokenizer, "bos_token_id", None)
    finals, tails, bos_counts = set(), set(), set()
    first_ids = None
    for i, text in enumerate(chat_prompts):
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if bos_id is not None:
            if len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
                raise zc.ZSError(f"double BOS in prompt {i} (head={ids[:4]})")
            if ids[0] != bos_id:
                raise zc.ZSError(f"prompt {i} does not start with BOS id {bos_id}")
            bos_counts.add(sum(1 for t in ids if t == bos_id))
        finals.add(ids[-1])
        tails.add(tuple(ids[-n_tail:]))
        if first_ids is None:
            first_ids = ids
    if len(finals) != 1:
        raise zc.ZSError(f"final prompt token is not uniform across prompts: {sorted(finals)}")
    tail = list(next(iter(tails)))
    if len(tails) != 1:
        # the header suffix must be identical; only the last `n_tail` are compared
        raise zc.ZSError(f"last-{n_tail}-token sequence is not uniform: {len(tails)} variants")
    final_id = next(iter(finals))
    return {
        "final_token_id": final_id,
        "final_token_text": tokenizer.decode([final_id]),
        "tail_token_ids": tail,
        "tail_tokens": tokenizer.convert_ids_to_tokens(tail),
        "tail_decoded": tokenizer.decode(tail),
        "bos_token_id": bos_id,
        "bos_count_per_prompt": sorted(bos_counts) if bos_counts else None,
        "n_prompts_checked": len(chat_prompts),
        "tokenization": "add_special_tokens=True (as vc.regenerate)",
    }


def sharded_flag(vc) -> bool:
    dm = getattr(vc.model, "hf_device_map", None)
    return bool(dm) and len({str(v) for v in dm.values()}) > 1


def provenance(vc=None) -> dict:
    p = {"host": platform.node(), "platform": platform.platform(),
         "python": platform.python_version(),
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        p["torch"] = torch.__version__
        p["gpu_names"] = [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())]
    except Exception:
        p["torch"], p["gpu_names"] = None, []
    try:
        import transformers
        p["transformers"] = transformers.__version__
    except Exception:
        p["transformers"] = None
    try:
        import numpy
        p["numpy"] = numpy.__version__
    except Exception:
        p["numpy"] = None
    p["sharded"] = sharded_flag(vc) if vc is not None else None
    return p


def check_mask(raw_mask, model: str, decoder_range, n_decoder_layers: int):
    """Full-length mask; non-zero rows must be EXACTLY the band's decoder
    layers (a row COUNT check would pass a wrong band)."""
    import numpy as np
    c = zc.MODEL_CONFIG[model]
    if raw_mask.ndim != 2 or raw_mask.shape[0] != n_decoder_layers:
        raise zc.ZSError(f"mask shape {raw_mask.shape} != ({n_decoder_layers}, H)")
    nz = [int(i) for i in np.nonzero(np.any(raw_mask != 0, axis=1))[0]]
    want = list(decoder_range)
    if nz != want:
        raise zc.ZSError(f"mask non-zero rows {nz} != decoder layers {want} for "
                         f"band [{c['layer_start']},{c['layer_end']})")
    return nz


def generate_cell(vc, *, model, condition, alpha, samples, chat_prompts,
                  raw_mask, mask_nz_rows, mask_sha, mask_path, site,
                  chat_template_hash, prompt_sha, manifest_meta, out_path,
                  prov):
    """Generate one cell, verify fires, write atomically. Returns meta."""
    c = zc.MODEL_CONFIG[model]
    g = zc.GEN_FIXED
    diff = raw_mask * alpha                       # full length; alpha=0 -> all-zero
    vc.steering_fire_count(reset=True)
    gen = []
    for i in range(0, len(chat_prompts), g["batch_size"]):
        gen.extend(vc.regenerate(
            chat_prompts[i: i + g["batch_size"]],
            max_new_tokens=g["max_new_tokens"], temperature=g["temperature"],
            top_p=g["top_p"], diff_matrices=list(diff),
            batch_size=g["batch_size"], return_metadata=True))
    if len(gen) != len(samples):
        raise zc.ZSError(f"generation returned {len(gen)} rows for {len(samples)} prompts")
    fires = vc.steering_fire_count()
    want = zc.expected_fires(model, alpha, len(samples))
    if fires != want:
        raise zc.ZSError(f"steering_fires {fires} != expected {want} "
                         f"(L={c['L']}, n={len(samples)}, alpha={alpha}); cell not usable")
    rows = []
    for s, r in zip(samples, gen):
        rows.append({
            "sample_id": s["sample_id"], "key": s["key"], "dataset": s["dataset"],
            "official_theory_id": s["official_theory_id"],
            "official_qid": s["official_qid"],
            "generated": r["text"],
            "generated_token_count": r["generated_token_count"],
            "stop_reason": r["stop_reason"],
            "truncated": r["stop_reason"] == "budget_exhausted",
        })
    meta = zc.expected_fixed_meta(model, condition, alpha)
    meta.update({
        "steering_fires": fires,
        "cot_note": ("own construction, NOT an official ProofWriter LLM prompt; "
                     "0-shot, no exemplar"),
        "mask_path": mask_path, "mask_sha256": mask_sha,
        "mask_nonzero_rows": mask_nz_rows,
        "prompt_sha256": prompt_sha,
        "bare_prompt_sha256": zc.bare_prompt_sha256(samples, condition),
        "chat_template_hash": chat_template_hash,
        "injection_site": site,
        "manifest_sha256_16": manifest_meta.get("manifest_sha256_16"),
        "n_truncated": sum(1 for x in rows if x["truncated"]),
        "provenance": prov,
        "not_a_transfer_test": True,
        "workpoint_source": "GSM8K frozen record; not re-selected on ProofWriter",
    })
    zc.atomic_write_json(out_path, {"meta": meta, "data": rows})
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "run"), required=True)
    p.add_argument("--model", choices=sorted(zc.MODEL_CONFIG), required=True)
    p.add_argument("--condition", choices=sorted(pz.TEMPLATE_IDS), default="nocot")
    p.add_argument("--cell", choices=zc.CELLS, required=True,
                   help="a0 = alpha 0; wp = this model's GSM8K workpoint")
    p.add_argument("--manifest", required=True, help="manifest_blind.json")
    p.add_argument("--mask_path", help="required for --mode run")
    p.add_argument("--out_root", help="components dir; cells go under "
                   "<out_root>/<model>/proofwriter_owa_zs/ (run mode)")
    p.add_argument("--gate_file", help="passed gate JSON; required for --cell wp")
    p.add_argument("--model_dir", help="override HF id (default: model's own)")
    p.add_argument("--report", help="check mode: also write the report JSON here")
    return p.parse_args(argv)


def load_gate(path, model, condition, a0_path):
    g = json.load(open(path, encoding="utf-8"))
    if (g.get("protocol") != zc.PROTOCOL or g.get("model") != model
            or g.get("condition") != condition or g.get("stage") != "gate"):
        raise zc.ZSError(f"{path}: not a gate file for {model}/{condition}")
    if g.get("passed") is not True:
        raise zc.ZSError(f"{path}: format gate did NOT pass; the workpoint cell "
                         "is not run (no prompt/budget/scoring changes are made "
                         "automatically)")
    if g.get("thresholds") != zc.GATE_THRESHOLDS:
        raise zc.ZSError(f"{path}: gate thresholds differ from the frozen ones")
    if g.get("cell_sha256") != zc.sha256_file(a0_path):
        raise zc.ZSError(f"{path}: gate was computed on a different alpha=0 file")
    return g


def main(argv=None):
    args = parse_args(argv)
    cfg = zc.MODEL_CONFIG[args.model]
    alpha = zc.alpha_of(args.model, args.cell)
    model_dir = args.model_dir or cfg["model_dir"]
    manifest_meta, samples = zc.load_blind_manifest(args.manifest)
    bare = zc.bare_prompts_for(samples, args.condition)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    chat_prompts = wrap_prompts(tok, bare)
    try:
        site = injection_site_report(tok, chat_prompts)
    except zc.ZSError as e:
        die(str(e))
    prompt_sha = zc.sha256_text("\n".join(chat_prompts))
    template_hash = zc.sha256_text(getattr(tok, "chat_template", None) or "")
    report = {"protocol": zc.PROTOCOL, "model": args.model,
              "condition": args.condition, "model_dir": model_dir,
              "prompt_template_id": pz.TEMPLATE_IDS[args.condition],
              "injection_site": site, "prompt_sha256": prompt_sha,
              "bare_prompt_sha256": zc.bare_prompt_sha256(samples, args.condition),
              "chat_template_hash": template_hash,
              "example_wrapped_prompt_0": chat_prompts[0],
              "example_wrapped_tail_0": chat_prompts[0][-120:]}
    if args.mode == "check":
        print(json.dumps({k: v for k, v in report.items()
                          if k != "example_wrapped_prompt_0"},
                         ensure_ascii=False, indent=2))
        if args.report:
            zc.atomic_write_json(args.report, report)
        return 0

    # ───────────────────────── run ─────────────────────────
    if not (args.mask_path and args.out_root):
        die("--mask_path and --out_root are required for --mode run")
    out_path = os.path.join(args.out_root, zc.cell_relpath(args.model, args.condition, alpha))
    a0_path = os.path.join(args.out_root, zc.cell_relpath(args.model, args.condition, 0))
    if args.cell == "wp":
        if not args.gate_file:
            die("--gate_file is required for the workpoint cell")
        if not os.path.exists(a0_path):
            die(f"alpha=0 cell not found: {a0_path}")
        try:
            load_gate(args.gate_file, args.model, args.condition, a0_path)
        except zc.ZSError as e:
            die(str(e))

    import numpy as np
    import utils
    mask_sha = zc.sha256_file(args.mask_path)
    expected_extra = {"mask_sha256": mask_sha, "prompt_sha256": prompt_sha,
                      "chat_template_hash": template_hash,
                      "manifest_sha256_16": manifest_meta.get("manifest_sha256_16"),
                      "mask_path": args.mask_path, "injection_site": site}
    if os.path.exists(out_path):
        try:
            zc.validate_cell(out_path, args.model, args.condition, alpha,
                             samples, expected_extra)
        except zc.ZSError as e:
            die(f"existing output is NOT reusable ({e}); it is left untouched. "
                "Move it away deliberately if you intend to regenerate.")
        print(f"reuse validated existing cell {out_path}")
        return 0

    from llms import VicundaModel
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        die(f"padding_side is {vc.tokenizer.padding_side!r}, expected 'left'")
    raw_mask = np.load(args.mask_path)
    try:
        nz = check_mask(raw_mask, args.model,
                        utils.decoder_layer_range(cfg["layer_start"], cfg["layer_end"]),
                        len(vc._find_decoder_layers()))
        meta = generate_cell(
            vc, model=args.model, condition=args.condition, alpha=alpha,
            samples=samples, chat_prompts=chat_prompts, raw_mask=raw_mask,
            mask_nz_rows=nz, mask_sha=mask_sha, mask_path=args.mask_path,
            site=site, chat_template_hash=template_hash, prompt_sha=prompt_sha,
            manifest_meta=manifest_meta, out_path=out_path, prov=provenance(vc))
        # same validator the analysis uses, on the file just written
        zc.validate_cell(out_path, args.model, args.condition, alpha, samples,
                         expected_extra)
    except zc.ZSError as e:
        die(str(e))
    print(f"wrote {out_path}  steering_fires={meta['steering_fires']}  "
          f"truncated={meta['n_truncated']}/{meta['n']}")
    print("NO accuracy computed (generation is gold-free by construction).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
