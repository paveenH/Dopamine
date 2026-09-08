#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_proofwriter_owa_chat.py -- ProofWriter OWA v2, FORMAL four-point
interface-condition sweep under the HF chat template. Two models, SAME
protocol string, distinguished by meta.prompt_wrapper_id (and by
meta.model/meta.size):
    llama3    -> protocol "proofwriter-owa-chat-v1", prompt_wrapper_id
                 "llama3-chat-template-v1"
    qwen2.5   -> protocol "proofwriter-owa-chat-v1", prompt_wrapper_id
                 "qwen2.5-chat-template-v1"
(Qwen support added 2026-09-08 by parameterizing this SAME script over
--model/--size, rather than forking a second copy -- see the --model
argument below. Llama3's call path, output filenames, protocol string and
EXPECTED_CELLS are UNCHANGED from the original llama3-only version; every
existing llama3 invocation of this script is byte-identical in behavior. An
earlier draft of this script used a SEPARATE protocol string
"proofwriter-owa-chat-v11" for Qwen -- that was a typo/naming mistake,
corrected 2026-09-08: BOTH models share ONE chat-interface protocol,
"proofwriter-owa-chat-v1"; model identity is carried by prompt_wrapper_id,
not by the protocol string.)

THIS IS AN INDEPENDENT INTERFACE-CONDITION EXPERIMENT, NOT A NEW WORKPOINT
SEARCH AND NOT A REPLACEMENT for either model's frozen bare-string result
(proofwriter_owa/results/formal_sweep_v2.json, COMPLETE + CLOSED 2026-09-05).
The 30-item llama3 diagnostic (chat_v2_mdf_0, see diag_chat_template.py)
showed HF chat-template wrapping sharply reduces llama3's
no_answer_rate/loop_rate/truncation_rate at alpha=0, and the llama3 formal
4-point chat sweep confirmed this at N=300 (no Holm-significant workpoint
after the interface confound was controlled for; see
results/formal_sweep_chat_v1.json). Qwen's own bare-string sweep did NOT show
llama3's severe no-answer/loop pathology (see formal_sweep_v2.json's qwen2.5
row), so this Qwen run answers a narrower version of the same interface
question: under Qwen's own chat template, does steering (alpha in
{-6,0,+6,+8}) produce a stable, Holm-significant accuracy change?

WHAT IS HELD IDENTICAL to each model's own bare formal sweep
(get_answer_proofwriter_owa.py, components/<model>/proofwriter_owa/mdf_*/):
  - the SAME 300-item formal manifest, SAME sample_id order (manifest_blind.json)
  - the SAME v2 1-shot Unknown exemplar (exemplar_unknown_v2.json)
  - the SAME prompt BODY and "#### <Label>" marker convention
    (prompt.build_prompt, PROMPT_TEMPLATE_ID = proofwriter-owa-cot-v2 --
    UNCHANGED: only the wrapping around this exact string differs)
  - the SAME mask file and band as that model's bare sweep:
      llama3   mask/llama3_non_logits/nmd_0.5_11_20_8B.npy    band [11,20)
               alpha in {-6,-4,0,+4}
      qwen2.5  mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy   band [16,22)
               alpha in {-6,0,+6,+8}
  - temperature=0.0 (greedy), max_new_tokens=1024 (MAX_NEW_TOKENS_FROZEN,
    imported from the bare generator, never a separately hardcoded literal)
  - batch_size=8, prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation path as every other alpha (an all-zero
    diff matrix; vc.regenerate(diff_matrices=raw_mask*0) -- there is no
    separate vc.generate() branch for alpha=0)
  - the SAME parser and gold (eval_proofwriter_owa.py, FIRST-strict-marker
    main scoring, no-marker counted incorrect); this script does not score
    anything, it only writes generation cells

THE ONLY EXPERIMENTAL VARIABLE, held constant across all four alpha of one
model, is how the identical prompt STRING is wrapped before tokenization:
each rendered build_prompt(...) string is placed as the sole user-turn
content in tokenizer.apply_chat_template([{"role": "user", "content":
prompt}], tokenize=False, add_generation_prompt=True), with a duplicated
leading BOS stripped before vc.regenerate() tokenizes it (add_special_tokens
=True internally) -- reusing bandit_pv6_episode.py's _strip_leading_bos /
double-BOS-hazard pattern rather than reinventing it. This wrapping logic is
model-agnostic (Qwen's tokenizer has no BOS token at all -- bos_token is None
-- so strip_leading_bos / assert_no_double_bos are simply no-ops for it, the
same "double-BOS cannot occur here" fact already recorded in this repo's
Qwen GSM8K/CGT-seq check scripts).

Output is written to a SEPARATE directory tree PER MODEL (the model's own
components/<model>/proofwriter_owa/ subtree), so neither model's bare formal
sweep, nor llama3's 30-item chat diagnostic, nor the other model's chat
sweep is ever touched or ambiguous with this run:
    components/<model_dir>/proofwriter_owa/formal_chat_v1_mdf_<alpha>/
        proofwriter_owa_<size>_<ls>_<le>.json
meta.protocol = "proofwriter-owa-chat-v1" for BOTH models -- NOT either
model's bare-sweep protocol ("proofwriter-owa-v0"), and NOT two different
chat protocol strings: model identity within this ONE chat protocol is
carried by meta.prompt_wrapper_id (and meta.model/meta.size), not by the
protocol string. meta.prompt_template_id / marker_family stay
"proofwriter-owa-cot-v2" / "v2" UNCHANGED for both models (the prompt body
and marker convention did not change, only the wrapping). Additional fields
record the wrapping condition explicitly and are cross-checked for
consistency across alpha (of the SAME model) by eval_proofwriter_owa.py's
CONSISTENCY_FIELDS:
    meta.prompt_wrapper_id     = "llama3-chat-template-v1" / "qwen2.5-chat-template-v1"
    meta.chat_template_applied = true
    meta.chat_template_hash    = sha256 of tokenizer.chat_template's own
                                  string (attests WHICH chat template was
                                  used, not just that some template was used)

Both models' cell families are scored with the SAME --protocol value:
    python eval_proofwriter_owa.py --protocol proofwriter-owa-chat-v1 ...
(eval_proofwriter_owa.py's --protocol flag, added for the llama3 run and
reused unchanged here; parser/scoring/statistics are unchanged. A single
eval call must still not mix the two models' chat cells together --
load_cell groups cells by meta.model, and the launcher's `eval` stage scores
one model's four cells per invocation, writing to that model's own output
file: formal_sweep_chat_v1.json for llama3 [UNCHANGED name/path from the
original llama3-only script], formal_sweep_chat_v1_qwen25.json for qwen2.5.)

No steering-dose search beyond each model's own frozen four-point set.

@author: proofwriter_owa task (chat-template formal sweep)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt import build_prompt, PROMPT_TEMPLATE_ID  # noqa: E402
from commitment import first_strict_marker_start  # noqa: E402
from answer_parser import get_marker_family  # noqa: E402
from get_answer_proofwriter_owa import MAX_NEW_TOKENS_FROZEN  # noqa: E402

MARKER_FAMILY = get_marker_family(PROMPT_TEMPLATE_ID)["marker_family"]

FORBIDDEN_KEYS = ("answer", "label", "gold", "gold_answer", "correct",
                  "accuracy", "proof", "proofs", "target")

# Per-model config: own frozen dose set (SAME four alpha as that model's
# bare sweep, same band -- this is the SAME dose family, wrapped
# differently, not a new search), own protocol string, own prompt-wrapper
# id. Mirrors get_answer_proofwriter_owa.py's EXPECTED_CELLS / model-keyed
# structure exactly, so an out-of-family alpha is rejected rather than
# silently run.
MODEL_CONFIG = {
    "llama3": {
        "protocol": "proofwriter-owa-chat-v1",
        "prompt_wrapper_id": "llama3-chat-template-v1",
        "expected_cells": {(-6, 11, 20), (-4, 11, 20), (0, 11, 20), (4, 11, 20)},
    },
    "qwen2.5": {
        "protocol": "proofwriter-owa-chat-v11",
        "prompt_wrapper_id": "qwen2.5-chat-template-v1",
        "expected_cells": {(-6, 16, 22), (0, 16, 22), (6, 16, 22), (8, 16, 22)},
    },
}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3", choices=list(MODEL_CONFIG),
                   help="default 'llama3' preserves this script's original "
                        "call signature and output paths byte-for-byte; "
                        "pass 'qwen2.5' for the Qwen chat sweep (added "
                        "2026-09-08).")
    p.add_argument("--size", default="8B",
                   help="record-only label used in the output filename and "
                        "meta.size, matching the bare generator's "
                        "convention (llama3->8B, qwen2.5->7B). Not "
                        "cross-checked against the actual model weights.")
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Hugging Face repo id (never a local path unless "
                        "the caller explicitly overrides this). Default is "
                        "llama3's; pass --model_dir Qwen/Qwen2.5-7B-Instruct "
                        "together with --model qwen2.5.")
    p.add_argument("--manifest", required=True,
                   help="manifest_blind.json -- the full 300-item formal "
                        "manifest, SAME file that model's bare formal sweep "
                        "used")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask file as that model's own bare formal "
                        "sweep (llama3: mask/llama3_non_logits/"
                        "nmd_0.5_11_20_8B.npy; qwen2.5: mask/"
                        "qwen2.5_non_logits/nmd_0.5_16_22_7B.npy)")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20 neg6-11-20 neg4-11-20 4-11-20 (llama3) "
                        "or 0-16-22 neg6-16-22 6-16-22 8-16-22 (qwen2.5)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--exemplar_file", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tag", default="formal_chat_v1",
                   help="output subdir prefix; default keeps this run in "
                        "its own formal_chat_v1_mdf_<alpha> tree, isolated "
                        "from the bare mdf_<alpha> tree and (for llama3) "
                        "the 30-item chat_v2_mdf_0 diagnostic")
    return p.parse_args()


def load_manifest(path):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("manifest file does not declare contains_labels=false. Point "
            "--manifest at manifest_blind.json, never at manifest_gold.json.")
    for s in data:
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            die(f"label field {bad} present in the manifest file; "
                "generation must be label-free")
    return meta, data


def load_exemplars(path):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d.get("meta", {}), d["data"]
    if meta.get("split") != "train":
        die(f"exemplar file declares split={meta.get('split')!r}, must be "
            "'train'")
    if len(data) < 1:
        die("exemplar file is empty; v2 needs exactly 1 exemplar")
    return data[:1]


def strip_leading_bos(vc, text: str) -> str:
    """Remove a BOS string apply_chat_template already serialized as text.

    Byte-identical logic to diag_chat_template.py's strip_leading_bos /
    bandit_pv6_episode.py's _strip_leading_bos -- see either docstring for
    why this is required (not cosmetic) on a model with a BOS token: vc.
    regenerate's _regenerate_prefill_only tokenizes with
    add_special_tokens=True, so an un-stripped chat-templated string yields
    two leading BOS ids. On Qwen2.5 (bos_token is None) this is a no-op --
    the same "double-BOS cannot occur here" fact this repo's Qwen GSM8K/
    CGT-seq check scripts already record."""
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def assert_no_double_bos(vc, text: str, label: str) -> None:
    """Hard invariant, same check as diag_chat_template.py /
    check_cgt_seq_qwen.py: tokenize with add_special_tokens=True (the same
    call vc.regenerate makes internally) and refuse if the first two ids are
    both BOS. A no-op (never fires) on a tokenizer with bos_token_id=None."""
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS text before this add_special_tokens=True call.")


def main():
    args = parse_args()
    cfg = MODEL_CONFIG[args.model]
    protocol = cfg["protocol"]
    prompt_wrapper_id = cfg["prompt_wrapper_id"]
    expected_cells = cfg["expected_cells"]

    meta, samples = load_manifest(args.manifest)
    exemplars = load_exemplars(args.exemplar_file)

    import utils  # noqa: E402  (repo root on sys.path via the insert above)
    cfgs_raw = args.configs
    cfgs = utils.parse_configs(cfgs_raw)
    got = {(al, ls, le) for al, (ls, le) in cfgs}
    if not got.issubset(expected_cells):
        die(f"{args.model}: cells {sorted(got)} are outside this script's "
            f"OWN frozen dose set for this model {sorted(expected_cells)}. "
            "This is the SAME dose family as that model's bare formal "
            "sweep, wrapped differently -- it does not search new alpha.")

    from llms import VicundaModel  # noqa: E402

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        die(f"tokenizer.padding_side is {vc.tokenizer.padding_side!r}, "
            "expected 'left'.")

    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    chat_template_str = getattr(vc.tokenizer, "chat_template", None) or ""
    chat_template_hash = hashlib.sha256(
        chat_template_str.encode("utf-8")).hexdigest()

    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    # Rendered ONCE, shared by every alpha of this sweep -- byte-identical
    # chat prompts across the whole curve, matching the bare generator's
    # prompt_sha256 convention (get_answer_proofwriter_owa.py) so
    # eval_proofwriter_owa.py's CONSISTENCY_FIELDS check can verify no drift
    # occurred between alpha cells.
    bare_prompts = [build_prompt(s["theory_text"], s["question_text"], exemplars)
                    for s in samples]
    chat_prompts = []
    for i, p in enumerate(bare_prompts):
        wrapped = vc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        wrapped = strip_leading_bos(vc, wrapped)
        assert_no_double_bos(vc, wrapped, f"sample_id={samples[i]['sample_id']}")
        chat_prompts.append(wrapped)

    prompt_sha256 = hashlib.sha256(
        "\n".join(chat_prompts).encode("utf-8")).hexdigest()
    bare_prompt_sha256 = hashlib.sha256(
        "\n".join(bare_prompts).encode("utf-8")).hexdigest()

    for alpha, (ls, le) in cfgs:
        tag = f"mdf_{alpha}".replace("-", "neg")
        subdir = f"{args.tag}_{tag}"
        out = os.path.join(args.out_dir, subdir,
                           f"proofwriter_owa_{args.size}_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        # regenerate() requires one diff row per decoder layer in the WHOLE
        # model (32 for Llama3-8B, 28 for Qwen2.5-7B). raw_mask is already
        # full-length with zero rows outside [layer_start, layer_end) --
        # slicing it to the band here is the exact bug the 30-item
        # diagnostic hit and fixed ("diff_matrices length (9) != layers
        # (32)"); `raw_mask * alpha` matches get_answer_proofwriter_owa.py's
        # convention exactly for both models.
        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                      desc=f"proofwriter-owa-chat[{args.model}] a={alpha}"):
            batch = chat_prompts[i: i + args.batch_size]
            gen.extend(vc.regenerate(
                batch,
                max_new_tokens=MAX_NEW_TOKENS_FROZEN,
                temperature=0.0,
                top_p=1.0,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
                return_metadata=True,
            ))

        if len(gen) != len(samples):
            die(f"generation returned {len(gen)} rows for {len(samples)} "
                f"prompts at alpha={alpha}; zip() would silently drop "
                f"{abs(len(gen) - len(samples))} sample(s).")

        fires = vc.steering_fire_count()
        n_layers = len(utils.decoder_layer_range(ls, le))
        expect = 0 if alpha == 0 else n_layers * len(samples)
        if fires != expect:
            die(f"steering_fires {fires} != {expect} "
                f"(L={n_layers}, n={len(samples)}, alpha={alpha}); "
                "the intervention is unverified, so the cell is not usable.")

        rows = []
        n_truncated = 0
        for s, g in zip(samples, gen):
            truncated = (g["stop_reason"] == "budget_exhausted")
            n_truncated += int(truncated)
            text = g["text"]
            marker_start = first_strict_marker_start(text, MARKER_FAMILY)
            pre_answer_tokens = None
            post_answer_tokens = None
            if marker_start is not None:
                try:
                    pre_answer_tokens = len(vc.tokenizer(
                        text[:marker_start], add_special_tokens=False)["input_ids"])
                    post_answer_tokens = max(
                        0, g["generated_token_count"] - pre_answer_tokens)
                except Exception:
                    pre_answer_tokens = None
                    post_answer_tokens = None
            rows.append({
                "sample_id": s["sample_id"], "key": s["key"],
                "dataset": s["dataset"],
                "official_theory_id": s["official_theory_id"],
                "official_qid": s["official_qid"],
                "generated": text,
                "generated_token_count": g["generated_token_count"],
                "pre_answer_reasoning_tokens": pre_answer_tokens,
                "post_answer_tokens": post_answer_tokens,
                "stop_reason": g["stop_reason"],
                "truncated": truncated,
            })

        json.dump({"meta": {
            "protocol": protocol, "prompt_template_id": PROMPT_TEMPLATE_ID,
            "marker_family": MARKER_FAMILY,
            "model": args.model, "size": args.size,
            "alpha": alpha, "layer_start": ls, "layer_end": le, "L": n_layers,
            "mask_path": args.mask_path, "mask_sha256": mask_sha,
            "max_new_tokens": MAX_NEW_TOKENS_FROZEN,
            "temperature": 0.0, "top_p": 1.0,
            "batch_size": args.batch_size,
            "n_shot": 1,
            "cot": True, "cot_note": ("own construction, NOT an official "
                                      "ProofWriter LLM prompt"),
            "chat_template": True, "prefill_only": True, "prefill_tail_len": 1,
            "prompt_wrapper_id": prompt_wrapper_id,
            "chat_template_applied": True,
            "chat_template_hash": chat_template_hash,
            "steering_fires": fires,
            "prompt_sha256": prompt_sha256,
            "bare_prompt_sha256": bare_prompt_sha256,
            "padding_side": vc.tokenizer.padding_side,
            "manifest_sha256_16": meta.get("manifest_sha256_16"),
            "n": len(rows),
            "n_truncated": n_truncated,
            "truncation_rate": n_truncated / len(rows) if rows else None,
            "contains_labels": False, "accuracy_computed": False,
            "provenance": device_note,
            "not_a_transfer_test": True,
            "interface_condition": {
                "purpose": ("formal four-point steering sweep under the "
                           "chat-template interface condition. Independent "
                           "interface condition, NOT a redefinition of the "
                           "ProofWriter-OWA workpoint (CLOSED 2026-09-05) "
                           "and NEVER a replacement for that model's own "
                           "formal_sweep_v2.json bare-string cells."),
                "compares_against": (f"components/{args.model}/proofwriter_owa/"
                                     f"mdf_<alpha>/proofwriter_owa_{args.size}_"
                                     f"{ls}_{le}.json (that model's bare "
                                     "formal sweep)"),
                "same_300_items": True, "same_order": True,
                "same_exemplar": True, "same_marker_convention": True,
                "same_mask": True, "same_alpha_set": True,
                "only_variable": "prompt wrapping (bare-string vs chat "
                                 "template)",
            },
        }, "data": rows}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {out}  steering_fires={fires}  "
              f"truncation_rate={n_truncated}/{len(rows)}")

    print("\nGeneration complete. NO accuracy was computed -- by construction.")
    print(f"Next: python proofwriter_owa/eval_proofwriter_owa.py "
          f"--protocol {protocol} (the only script that reads gold)")


if __name__ == "__main__":
    main()
