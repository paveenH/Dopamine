#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_chat_template.py -- ProofWriter OWA v2, Llama3.1-8B ONLY: interface
diagnostic comparing bare-string vs HF chat-template wrapping at alpha=0.

THIS IS AN INTERFACE DIAGNOSTIC, NOT A NEW EXPERIMENT AND NOT A WORKPOINT
SEARCH. It does not redefine the ProofWriter-OWA workpoint (COMPLETE + CLOSED
2026-09-05, see CLAUDE.md / AdaDopamine_gsm8k.md Sec 5.5c) and its output is
never substituted for the frozen formal_sweep_v2.json result. Question being
asked: is llama3's high no_answer_rate/loop_rate under the v2 prompt driven
mainly by the BARE-STRING prompt wrapping (an interface/termination issue), or
does it persist under the model's own native chat interface (which would argue
for a task/logic-capacity limitation instead)?

WHAT IS HELD IDENTICAL to the existing bare v2 alpha=0 preflight cell
(components/llama3/proofwriter_owa/preflight_v2_mdf_0/):
  - the SAME 30 items, in the SAME order (preflight_blind_llama3.json)
  - the SAME v2 1-shot Unknown exemplar (exemplar_unknown_v2.json)
  - the SAME prompt body and "#### <Label>" marker convention
    (prompt.build_prompt, PROMPT_TEMPLATE_ID = proofwriter-owa-cot-v2)
  - temperature=0.0 (greedy, no sampling introduced)
  - max_new_tokens=1024 (MAX_NEW_TOKENS_FROZEN, imported, never hardcoded
    separately)
  - alpha=0 (an all-zero diff matrix, matching get_answer_proofwriter_owa.py's
    convention exactly -- steering hooks register and fire on the zero rows,
    steering_fires must read 0)
  - the SAME parser and gold (proofwriter_owa/eval_proofwriter_owa.py; this
    script does not score anything itself, it only writes a generation cell)

THE ONLY EXPERIMENTAL VARIABLE is how the identical prompt STRING is wrapped
before tokenization:
  bare : build_prompt(...) fed to vc.regenerate() as-is (bare-string; this is
         exactly what get_answer_proofwriter_owa.py already does -- this
         script does NOT regenerate the bare cell, it is read from the
         existing preflight_v2_mdf_0 file for comparison)
  chat : tokenizer.apply_chat_template([{"role": "user", "content":
         build_prompt(...)}], tokenize=False, add_generation_prompt=True),
         with a duplicated leading BOS stripped before it is handed to
         vc.regenerate() (which tokenizes with add_special_tokens=True
         internally) -- the exact double-BOS hazard and fix already
         documented in bandit_pv6_episode.py's _strip_leading_bos /
         audit_prompt_tokens, reused here rather than reinvented.

vc.regenerate(prefill_only=True) is the SAME generation path used for the
bare cell (same terminators via VicundaModel._build_terminators, which
already includes Llama-3.1's <|eot_id|>; same truncation detection via
stop_reason=="budget_exhausted"; same batch tokenization/padding). Only the
INPUT STRING differs, so any bare-vs-chat difference in no_answer_rate/
loop_rate/truncation_rate is attributable to the prompt wrapping, not to a
different generation code path.

Output is written to a SEPARATE path so the existing bare v2 preflight file
is never touched:
    components/llama3/proofwriter_owa/chat_v2_mdf_0/proofwriter_owa_8B_11_20.json
Row/meta schema matches get_answer_proofwriter_owa.py's output exactly (same
keys, same score_cell()-required fields) so eval_proofwriter_owa.py can score
it completely unmodified via --allow_partial_alphas (matching how the bare
preflight cell is already scored). meta.chat_template=True and
meta.prompt_template_id/marker_family are UNCHANGED (still
proofwriter-owa-cot-v2 / v2) since the prompt BODY and marker convention are
byte-identical to the bare cell -- only the wrapping differs, which is
recorded in a chat-specific meta.diagnostic block.

No steering sweep. Llama3.1-8B only.

@author: proofwriter_owa task (chat-template diagnostic)
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

MARKER_FAMILY = get_marker_family(PROMPT_TEMPLATE_ID)["marker_family"]
PROTOCOL = "proofwriter-owa-v0"
MAX_NEW_TOKENS_FROZEN = 1024  # matches get_answer_proofwriter_owa.py exactly
FORBIDDEN_KEYS = ("answer", "label", "gold", "gold_answer", "correct",
                  "accuracy", "proof", "proofs", "target")


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--manifest", required=True,
                   help="preflight_blind_llama3.json -- the SAME 30-item "
                        "blind subset the existing bare v2 preflight cell "
                        "used")
    p.add_argument("--mask_path", required=True,
                   help="llama3 nmd mask (used only to size the all-zero "
                        "alpha=0 diff matrix and assert L; alpha is fixed "
                        "at 0, no steering)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--exemplar_file", required=True)
    p.add_argument("--layer_start", type=int, default=11)
    p.add_argument("--layer_end", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def load_manifest(path):
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("manifest file does not declare contains_labels=false. Point "
            "--manifest at the blind preflight subset, never at a gold file.")
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

    REQUIRED, not cosmetic: vc.regenerate's _regenerate_prefill_only
    tokenizes with the tokenizer's default add_special_tokens=True, so
    without this a chat-templated string that already contains the literal
    "<|begin_of_text|>" text would be tokenized to two leading BOS ids. Same
    fix as bandit_pv6_episode.py's _strip_leading_bos, reused rather than
    reinvented -- see that module's docstring for why (1) apply_chat_template
    on Llama-3.x emits BOS as text and (2) the downstream tokenizer call adds
    another with add_special_tokens=True."""
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def assert_no_double_bos(vc, text: str, label: str) -> None:
    """Hard invariant (matches check_cgt_seq_qwen.py's double-BOS check):
    tokenize with add_special_tokens=True (the same call vc.regenerate makes
    internally) and refuse if the first two ids are both BOS. A silent
    double-BOS would make the chat cell's prefix disagree with the bare
    cell's in a way that is invisible in the decoded string."""
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
    from llms import VicundaModel  # noqa: E402

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        die(f"tokenizer.padding_side is {vc.tokenizer.padding_side!r}, "
            "expected 'left'.")

    raw_mask = np.load(args.mask_path)
    mask_sha = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    n_layers = len(utils.decoder_layer_range(args.layer_start, args.layer_end))
    if raw_mask.shape[0] < args.layer_end:
        die(f"mask has {raw_mask.shape[0]} rows, need >= {args.layer_end}")

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

    out_dir = os.path.join(args.out_dir, "chat_v2_mdf_0")
    out = os.path.join(out_dir, "proofwriter_owa_8B_11_20.json")
    if os.path.exists(out):
        die(f"{out} exists; refusing to overwrite (this is a diagnostic "
            "artifact, not a resumable sweep cell).")
    os.makedirs(out_dir, exist_ok=True)

    # regenerate() requires one diff row per decoder layer in the WHOLE model
    # (matches get_answer_proofwriter_owa.py exactly: `diff = raw_mask *
    # alpha`) -- raw_mask is already full-length (32 rows for Llama3-8B) with
    # zero rows outside [layer_start, layer_end); slicing it to the band here
    # was wrong and raised "diff_matrices length (9) != layers (32)".
    diff = raw_mask * 0  # alpha=0, all-zero
    vc.steering_fire_count(reset=True)

    gen = []
    for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                  desc="proofwriter-owa chat-diag a=0"):
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
            f"prompts; zip() would silently drop "
            f"{abs(len(gen) - len(samples))} sample(s).")

    fires = vc.steering_fire_count()
    if fires != 0:
        die(f"steering_fires {fires} != 0 at alpha=0; the all-zero diff "
            "matrix should never register a fire.")

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

    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    json.dump({"meta": {
        "protocol": PROTOCOL, "prompt_template_id": PROMPT_TEMPLATE_ID,
        "marker_family": MARKER_FAMILY,
        "model": "llama3", "size": "8B",
        "alpha": 0, "layer_start": args.layer_start, "layer_end": args.layer_end,
        "L": n_layers,
        "mask_path": args.mask_path, "mask_sha256": mask_sha,
        "max_new_tokens": MAX_NEW_TOKENS_FROZEN,
        "temperature": 0.0, "top_p": 1.0,
        "batch_size": args.batch_size,
        "n_shot": 1,
        "cot": True, "cot_note": ("own construction, NOT an official "
                                  "ProofWriter LLM prompt"),
        "chat_template": True, "prefill_only": True, "prefill_tail_len": 1,
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
        "diagnostic": {
            "purpose": ("interface diagnostic ONLY: does HF chat-template "
                       "wrapping (apply_chat_template + "
                       "add_generation_prompt=True) reduce no_answer_rate/"
                       "loop_rate/truncation_rate vs the bare-string v2 "
                       "preflight cell at alpha=0. Does NOT redefine the "
                       "ProofWriter-OWA workpoint (CLOSED 2026-09-05) and is "
                       "never substituted for formal_sweep_v2.json."),
            "compare_against": ("components/llama3/proofwriter_owa/"
                                "preflight_v2_mdf_0/"
                                "proofwriter_owa_8B_11_20.json"),
            "same_30_items": True, "same_order": True,
            "same_exemplar": True, "same_marker_convention": True,
            "same_temperature": True, "same_max_new_tokens": True,
            "only_variable": "prompt wrapping (bare-string vs chat template)",
        },
    }, "data": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"wrote {out}  steering_fires={fires}  "
          f"truncation_rate={n_truncated}/{len(rows)}")
    print("\nNext: python proofwriter_owa/eval_proofwriter_owa.py "
          "--gold <preflight_gold_llama3.json> --generations "
          f"{out} --out <chat_eval.json> --allow_partial_alphas")


if __name__ == "__main__":
    main()
