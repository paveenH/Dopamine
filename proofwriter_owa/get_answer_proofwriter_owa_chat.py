#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_proofwriter_owa_chat.py -- ProofWriter OWA v2, Llama3.1-8B ONLY,
FORMAL four-point interface-condition sweep under the HF chat template.
Protocol `proofwriter-owa-chat-v1`.

THIS IS AN INDEPENDENT INTERFACE-CONDITION EXPERIMENT, NOT A NEW WORKPOINT
SEARCH AND NOT A REPLACEMENT for the frozen bare-string result
(proofwriter_owa/results/formal_sweep_v2.json, COMPLETE + CLOSED 2026-09-05).
The 30-item diagnostic (chat_v2_mdf_0, see diag_chat_template.py) showed HF
chat-template wrapping sharply reduces llama3's no_answer_rate/loop_rate/
truncation_rate at alpha=0. This script asks: once that interface confound is
controlled for, does steering (alpha in {-6,-4,0,+4}) produce a stable,
Holm-significant accuracy or submission-rate change?

WHAT IS HELD IDENTICAL to the bare formal sweep
(get_answer_proofwriter_owa.py, components/llama3/proofwriter_owa/mdf_*/):
  - the SAME 300-item formal manifest, SAME sample_id order (manifest_blind.json)
  - the SAME v2 1-shot Unknown exemplar (exemplar_unknown_v2.json)
  - the SAME prompt BODY and "#### <Label>" marker convention
    (prompt.build_prompt, PROMPT_TEMPLATE_ID = proofwriter-owa-cot-v2 --
    UNCHANGED: only the wrapping around this exact string differs)
  - the SAME mask file (mask/llama3_non_logits/nmd_0.5_11_20_8B.npy),
    band [11,20), alpha in {-6,-4,0,+4}
  - temperature=0.0 (greedy), max_new_tokens=1024 (MAX_NEW_TOKENS_FROZEN,
    imported from the bare generator, never a separately hardcoded literal)
  - batch_size=8, prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation path as every other alpha (an all-zero
    diff matrix; vc.regenerate(diff_matrices=raw_mask*0) -- there is no
    separate vc.generate() branch for alpha=0)
  - the SAME parser and gold (eval_proofwriter_owa.py, FIRST-strict-marker
    main scoring, no-marker counted incorrect); this script does not score
    anything, it only writes generation cells

THE ONLY EXPERIMENTAL VARIABLE, held constant across all four alpha, is how
the identical prompt STRING is wrapped before tokenization: each rendered
build_prompt(...) string is placed as the sole user-turn content in
tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
tokenize=False, add_generation_prompt=True), with a duplicated leading BOS
stripped before vc.regenerate() tokenizes it (add_special_tokens=True
internally) -- reusing bandit_pv6_episode.py's _strip_leading_bos /
double-BOS-hazard pattern rather than reinventing it. This is byte-identical
wrapping logic to diag_chat_template.py; this script differs from it only by
running the full 300-item manifest across all four frozen alpha instead of
30 items at alpha=0 only.

Output is written to a SEPARATE directory tree and under a SEPARATE protocol
string, so neither the bare v2 formal sweep nor the 30-item chat diagnostic
is ever touched or ambiguous with this run:
    components/llama3/proofwriter_owa/formal_chat_v1_mdf_<alpha>/
        proofwriter_owa_8B_11_20.json
meta.protocol = "proofwriter-owa-chat-v1" (NOT the bare sweep's
"proofwriter-owa-v0"); meta.prompt_template_id / marker_family stay
"proofwriter-owa-cot-v2" / "v2" UNCHANGED (the prompt body and marker
convention did not change, only the wrapping). Two additional fields record
the wrapping condition explicitly and are cross-checked for consistency
across alpha by eval_proofwriter_owa.py's CONSISTENCY_FIELDS:
    meta.prompt_wrapper_id     = "llama3-chat-template-v1"
    meta.chat_template_applied = true
    meta.chat_template_hash    = sha256 of tokenizer.chat_template's own
                                  string (attests WHICH chat template was
                                  used, not just that some template was used)

Because meta.protocol differs from the bare sweep's, this cell family must be
scored with:
    python eval_proofwriter_owa.py --protocol proofwriter-owa-chat-v1 ...
(eval_proofwriter_owa.py's --protocol flag, added alongside this script --
the only change made to that file; parser/scoring/statistics are unchanged).

No Qwen, no other benchmark. Llama3.1-8B only, four alpha, one sweep.

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
PROTOCOL = "proofwriter-owa-chat-v1"
PROMPT_WRAPPER_ID = "llama3-chat-template-v1"

# Own frozen dose set: SAME four alpha as the bare sweep's llama3 row, same
# band. This is the SAME dose family, wrapped differently -- not a new
# search. Kept as an explicit constant (mirroring
# get_answer_proofwriter_owa.py's EXPECTED_CELLS) so an out-of-family alpha
# is rejected rather than silently run.
EXPECTED_CELLS = {(-6, 11, 20), (-4, 11, 20), (0, 11, 20), (4, 11, 20)}

FORBIDDEN_KEYS = ("answer", "label", "gold", "gold_answer", "correct",
                  "accuracy", "proof", "proofs", "target")


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--manifest", required=True,
                   help="manifest_blind.json -- the full 300-item formal "
                        "manifest, SAME file the bare formal sweep used")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask file as the bare formal sweep "
                        "(mask/llama3_non_logits/nmd_0.5_11_20_8B.npy)")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20 neg6-11-20 neg4-11-20 4-11-20")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--exemplar_file", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tag", default="formal_chat_v1",
                   help="output subdir prefix; default keeps this run in "
                        "its own formal_chat_v1_mdf_<alpha> tree, isolated "
                        "from the bare mdf_<alpha> tree and the 30-item "
                        "chat_v2_mdf_0 diagnostic")
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
    why this is required (not cosmetic): vc.regenerate's
    _regenerate_prefill_only tokenizes with add_special_tokens=True, so an
    un-stripped chat-templated string yields two leading BOS ids."""
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def assert_no_double_bos(vc, text: str, label: str) -> None:
    """Hard invariant, same check as diag_chat_template.py /
    check_cgt_seq_qwen.py: tokenize with add_special_tokens=True (the same
    call vc.regenerate makes internally) and refuse if the first two ids are
    both BOS."""
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS text before this add_special_tokens=True call.")


def main():
    args = parse_args()
    meta, samples = load_manifest(args.manifest)
    exemplars = load_exemplars(args.exemplar_file)

    import utils  # noqa: E402  (repo root on sys.path via the insert above)
    cfgs_raw = args.configs
    cfgs = utils.parse_configs(cfgs_raw)
    got = {(al, ls, le) for al, (ls, le) in cfgs}
    if not got.issubset(EXPECTED_CELLS):
        die(f"cells {sorted(got)} are outside this script's OWN frozen dose "
            f"set {sorted(EXPECTED_CELLS)}. This is the SAME dose family as "
            "the bare formal sweep, wrapped differently -- it does not "
            "search new alpha.")

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
                           f"proofwriter_owa_8B_{ls}_{le}.json")
        if os.path.exists(out):
            print(f"skip existing {out}")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)

        # regenerate() requires one diff row per decoder layer in the WHOLE
        # model (32 for Llama3-8B). raw_mask is already full-length with
        # zero rows outside [layer_start, layer_end) -- slicing it to the
        # band here is the exact bug the 30-item diagnostic hit and fixed
        # ("diff_matrices length (9) != layers (32)"); `raw_mask * alpha`
        # matches get_answer_proofwriter_owa.py's convention exactly.
        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                      desc=f"proofwriter-owa-chat a={alpha}"):
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
            "protocol": PROTOCOL, "prompt_template_id": PROMPT_TEMPLATE_ID,
            "marker_family": MARKER_FAMILY,
            "model": "llama3", "size": "8B",
            "alpha": alpha, "layer_start": ls, "layer_end": le, "L": n_layers,
            "mask_path": args.mask_path, "mask_sha256": mask_sha,
            "max_new_tokens": MAX_NEW_TOKENS_FROZEN,
            "temperature": 0.0, "top_p": 1.0,
            "batch_size": args.batch_size,
            "n_shot": 1,
            "cot": True, "cot_note": ("own construction, NOT an official "
                                      "ProofWriter LLM prompt"),
            "chat_template": True, "prefill_only": True, "prefill_tail_len": 1,
            "prompt_wrapper_id": PROMPT_WRAPPER_ID,
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
                           "chat-template interface condition, following up "
                           "the 30-item alpha=0 diagnostic that showed "
                           "chat-template wrapping sharply reduces "
                           "no_answer_rate/loop_rate/truncation_rate on the "
                           "bare-string v2 prompt. Independent interface "
                           "condition, NOT a redefinition of the "
                           "ProofWriter-OWA workpoint (CLOSED 2026-09-05) "
                           "and NEVER a replacement for "
                           "formal_sweep_v2.json."),
                "compares_against": ("components/llama3/proofwriter_owa/"
                                     "mdf_<alpha>/proofwriter_owa_8B_11_20."
                                     "json (the bare formal sweep)"),
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
    print("Next: python proofwriter_owa/eval_proofwriter_owa.py "
          "--protocol proofwriter-owa-chat-v1 "
          "(the only script that reads gold)")


if __name__ == "__main__":
    main()
