#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
get_answer_gsm_hard_chat_sweep.py -- GSM-Hard No-CoT, chat-template INTERFACE
condition, full 9-point alpha sweep. INDEPENDENT of the frozen bare-string
GSM-Hard P3 line (run_gsm_hard_llama3.sh / get_answer_gsm_hard_blind.py) --
neither is touched, read, or overwritten by this script.

Sibling of get_answer_gsm8k_chat_sweep.py (protocol "gsm8k-chat-sweep-v1"),
which this follows structurally rather than reuses: the two tasks differ in
sample file, sample identity (GSM-Hard rows carry a `sample_id`; GSM8K rows do
not), output tree and protocol name, and a shared script would invite exactly
the cross-task parameter mixing this repo has been bitten by. The GSM8K
script is left byte-unchanged.

THIS IS NOT A WORKPOINT SEARCH. alpha in {-8,-6,-4,-2,0,2,4,6,8} is a FIXED
nine-point family declared up front, not re-selected from any result.

  ** SCOPE NOTE, LOAD-BEARING FOR THE BARE COMPARISON **
  The frozen bare GSM-Hard P3 tree carries only FIVE No-CoT doses
  (-8/-6/-4/0/+4; CONFIGS in run_gsm_hard_llama3.sh). So four of this sweep's
  nine cells (-2/+2/+6/+8) have NO bare counterpart and CANNOT be paired
  against one. That is a property of the frozen bare tree, not a defect here:
  this sweep is a complete NINE-POINT CHAT curve read against its OWN chat
  alpha=0 baseline, and the bare-vs-chat table is a five-row DESCRIPTIVE
  subset. The analyzer states this explicitly rather than silently dropping
  the four unpaired rows.

WHAT IS HELD IDENTICAL to the frozen bare P3 cells (so the samples, prompt
body and decoding are the same and only the wrapper differs):
  - the SAME 300-question label-free sample file
    (components/benchmark/gsm_hard_p3_questions.json, questions_sha256
    48cc7635..., dataset revision pinned in docs/PREREG_P3.md), in the SAME
    ORDER, carrying the SAME sample_id values
  - the SAME neutral, No-CoT, plain-wording prompt BODY
    (template.select_templates_gsm8k(suite="default", cot=False,
    wording="plain")["neutral"]) -- the GSM8K body, which is what the frozen
    bare GSM-Hard cells also use
  - the SAME mask file and band: mask/llama3_non_logits/nmd_0.5_11_20_8B.npy,
    layers [11,20), L=9
  - the SAME generation params: max_new_tokens=768, temperature=0.0 (greedy),
    top_p=1.0, batch_size=24, prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0)), but is
    RE-RUN under the chat wrapper rather than reused from the bare baseline
    (the bare alpha=0 cell used a different prompt string and cannot serve as
    this sweep's own baseline)

THE ONLY PLANNED INTERFACE CHANGE, held constant across all nine alpha, is
that the identical rendered prompt STRING is wrapped with

    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

with a duplicated leading BOS stripped before vc.regenerate() tokenizes it
(add_special_tokens=True internally) -- reusing the same strip_leading_bos /
assert_no_double_bos pattern get_answer_gsm8k_chat_sweep.py and
proofwriter_owa/get_answer_proofwriter_owa_chat.py use.

LABEL-FREE, AND STRUCTURALLY SO. GSM-Hard gold lives in a SEPARATE sealed file
(gsm_hard_p3_gold.SEALED.json); this script reads only the questions file,
refuses any input not declaring contains_labels=false, and scans every row for
a leaked label key -- the same firewall get_answer_gsm_hard_blind.py has. It
therefore writes NO correctness field and prints NO accuracy: unlike the GSM8K
chat sweep (whose sample file carries gold inline, so inline process-state
fields are free), GSM-Hard accuracy exists only offline. P3's blind property
was already spent when the gold was unsealed on 2026-08-30, so this is no
longer protecting a seal -- it is keeping this generator on the same
label-unreachable code path as the frozen bare cells it will be compared with,
so the two trees differ in the wrapper and nothing else.

KNOWN CONFOUND -- MEASURED, NOT ASSUMED, AND IT MUST TRAVEL WITH EVERY RESULT.
apply_chat_template applies Jinja `| trim` to the user-turn content, which
STRIPS the trailing space of the prompt body's "Answer: " anchor. So the last
prompt token -- the ONLY position prefill-only tail=1 steering injects into --
differs between the two conditions. Both tails are READ OUT at run time (never
assumed to be 271) and stored per cell as meta.injection_tail_tokens_chat /
_bare_reference / meta.injection_site_moved_vs_bare. This experiment therefore
changes the interface AND the injection site TOGETHER; that is intrinsic to
chat wrapping (the generation header is always last) and cannot be removed
while still testing a chat interface, but it means a chat-vs-bare difference
may NOT be attributed to "the wrapper" alone.

Output tree (independent from the bare gsm_hard_p3 tree):
    components/llama3/gsm_hard_chat_v1/mdf_<alpha>/
        gsm_hard_chat_8B_11_20.json

Existing output files are NEVER overwritten -- fail closed (die()), matching
the P3/P4/P4b/P4c/ProofWriter-chat convention. There is no --allow_overwrite
escape hatch. All nine paths are checked BEFORE the first cell runs, so a
nine-hour sweep cannot die on cell 9 after eight cells are already written.

@author: GSM-Hard chat-template interface sweep (2026-09-11)
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

import utils
from llms import VicundaModel
from template import select_templates_gsm8k

PROTOCOL = "gsm-hard-chat-sweep-v1"
PROMPT_WRAPPER_ID = "llama3-chat-template-v1"

# Fixed, frozen nine-point dose family. NOT a free CLI list: --configs is
# validated against this set below so an out-of-family alpha cannot be silently
# run under this protocol name.
EXPECTED_ALPHAS = {-8, -6, -4, -2, 0, 2, 4, 6, 8}

# Band frozen to the bare P3 cells' own band.
BAND = (11, 20)

# The frozen GSM-Hard questions digest (docs/PREREG_P3.md, p3-v1). Checked so a
# swapped or regenerated sample file names itself instead of producing a
# plausible-looking curve over different questions.
EXPECTED_QUESTIONS_SHA256 = (
    "48cc763545d2ee23835833f5165456b90db194863420a10b6741a57cff781d02")

# Same firewall key list as get_answer_gsm_hard_blind.py.
FORBIDDEN_KEYS = ("answer", "gold", "gold_answer", "correct", "accuracy",
                  "target")


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM-Hard No-CoT chat-template interface sweep "
                    "(independent of the frozen bare P3 line)")
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention.")
    p.add_argument("--size", default="8B")
    p.add_argument("--questions", required=True,
                   help="components/benchmark/gsm_hard_p3_questions.json -- "
                        "the SAME label-free 300-question file the bare P3 "
                        "cells use")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask as the bare cells: "
                        "mask/llama3_non_logits/nmd_0.5_11_20_8B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="0-11-20 neg8-11-20 ... 8-11-20 (exactly the frozen "
                        "nine-point dose set)")
    p.add_argument("--out_dir", required=True,
                   help="components/llama3/gsm_hard_chat_v1")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--expect_questions_sha256",
                   default=EXPECTED_QUESTIONS_SHA256,
                   help="Frozen digest of the P3 question set. Pass '' only to "
                        "deliberately bypass (not recommended).")
    return p.parse_args()


def load_questions(path, expect_sha):
    """Label-free loader, byte-equivalent in intent to
    get_answer_gsm_hard_blind.py's: refuse anything that does not declare
    itself label-free, and scan every row for a leaked label key."""
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("questions file does not declare contains_labels=false; "
            "generation must be label-free.")
    for i, s in enumerate(data):
        bad = [k for k in s if k.lower() in FORBIDDEN_KEYS]
        if bad:
            die(f"label field {bad} present at row {i} of the questions file; "
                "generation must be label-free.")
    got = meta.get("questions_sha256")
    if expect_sha and got != expect_sha:
        die(f"questions_sha256 {got} != the frozen P3 digest {expect_sha}. "
            "This sweep must run on the SAME 300 questions as the bare cells; "
            "a different sample would make every bare-vs-chat pairing invalid.")
    ids = [s["sample_id"] for s in data]
    if len(set(ids)) != len(ids):
        die("duplicate sample_id in the questions file; per-item pairing "
            "against the bare cells would be ambiguous.")
    return meta, data


def strip_leading_bos(vc, text: str) -> str:
    """Same logic as get_answer_gsm8k_chat_sweep.py / the ProofWriter chat
    runner: vc.regenerate tokenizes with add_special_tokens=True, so an
    un-stripped chat-templated string that already serialized a BOS yields two
    leading BOS ids."""
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def assert_no_double_bos(vc, text: str, label: str) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS before this add_special_tokens=True call.")


def assert_chat_template_effective(vc, bare_prompt: str, wrapped: str) -> None:
    if wrapped == bare_prompt:
        die("chat template had NO effect on the first sample (wrapped text is "
            "byte-identical to the bare prompt) -- apply_chat_template did not "
            "actually run, or the tokenizer has no chat_template.")
    if not getattr(vc.tokenizer, "chat_template", None):
        die("tokenizer.chat_template is empty/None -- cannot apply a chat "
            "template that does not exist.")
    header = "<|start_header_id|>assistant<|end_header_id|>"
    if header not in wrapped:
        die(f"expected assistant generation header {header!r} not found in the "
            "first wrapped prompt -- add_generation_prompt=True did not produce "
            "the expected Llama-3 chat header.")


def main():
    args = parse_args()

    cfgs = utils.parse_configs(args.configs)
    # EXACT match on the frozen nine-point family, as a SORTED LIST -- not a
    # set-subset, and with NO int() coercion. utils.parse_configs accepts float
    # alpha tokens, so int(2.5) would silently read as 2 and pass a subset
    # check; a subset check also admits a partial curve or a repeated dose.
    got_alphas = sorted(al for al, _ in cfgs)
    want_alphas = sorted(EXPECTED_ALPHAS)
    if got_alphas != want_alphas:
        die(f"--configs alphas {got_alphas} != this protocol's frozen "
            f"nine-point dose set {want_alphas}. Exact match required: a "
            "non-integer dose (parse_configs accepts floats), a missing dose, "
            "a duplicate, or an extra dose all land here. A partial curve must "
            "not be written under this protocol name.")

    qmeta, samples = load_questions(args.questions, args.expect_questions_sha256)
    n = len(samples)
    print(f"Loaded {n} GSM-Hard questions from {args.questions} "
          f"(digest {qmeta['questions_sha256'][:16]}, label-free check OK)")

    templates = select_templates_gsm8k(suite="default", cot=False,
                                       wording="plain")
    neutral_tmpl = templates["neutral"]
    prompt_body_sha256 = hashlib.sha256(
        neutral_tmpl.encode("utf-8")).hexdigest()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    if vc.tokenizer.padding_side != "left":
        die(f"tokenizer.padding_side is {vc.tokenizer.padding_side!r}, "
            "expected 'left'.")

    chat_template_str = getattr(vc.tokenizer, "chat_template", None) or ""
    chat_template_hash = hashlib.sha256(
        chat_template_str.encode("utf-8")).hexdigest()

    raw_mask = np.load(args.mask_path)
    mask_sha256 = hashlib.sha256(open(args.mask_path, "rb").read()).hexdigest()
    os.makedirs(args.out_dir, exist_ok=True)

    # vc.regenerate requires ONE diff row per decoder layer in the WHOLE model
    # (32 for Llama3-8B); raw_mask is already full-length with zero rows outside
    # the band. Slicing it to the band is the exact bug the ProofWriter chat
    # diagnostic hit. Checked HERE so it fails before the first multi-minute
    # cell rather than during it.
    n_decoder = len(vc._find_decoder_layers())
    if raw_mask.shape[0] != n_decoder:
        die(f"mask has {raw_mask.shape[0]} rows but the model has {n_decoder} "
            "decoder layers; regenerate() needs one row per layer (zero rows "
            "outside the band). Do NOT slice the mask to the band.")
    nz_rows = sorted(int(i) for i in np.nonzero(np.any(raw_mask != 0, axis=1))[0])
    want_rows = sorted(utils.decoder_layer_range(*BAND))
    if nz_rows != want_rows:
        die(f"mask non-zero rows {nz_rows} != decoder_layer_range{BAND} "
            f"{want_rows}; the mask does not match this protocol's band.")

    # Provenance: record the device rather than constrain it (repo-wide rule).
    # All nine cells share this record because they share ONE model load.
    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "param_devices": devs,
                   "sharded": len(devs) > 1}

    # Render every bare prompt ONCE, wrap it ONCE -- shared byte-identically by
    # every alpha, so the digests below attest "same prompt across the curve".
    bare_prompts = [neutral_tmpl.format(context=s["question"]) for s in samples]
    chat_prompts = []
    for i, p in enumerate(bare_prompts):
        wrapped = vc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        wrapped = strip_leading_bos(vc, wrapped)
        if i == 0:
            assert_chat_template_effective(vc, p, wrapped)
        assert_no_double_bos(vc, wrapped, f"sample_idx={i}")
        chat_prompts.append(wrapped)

    # INJECTION SITE -- READ OUT, NEVER ASSUMED (the cross-model pre-flight
    # rule). apply_chat_template applies `| trim`, which strips the trailing
    # space of the body's "Answer: " anchor, so prefill-only steering does not
    # land on the same token in the two conditions. Recorded per cell.
    def _tail_ids(text, k=4):
        ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"][-k:]
        return [{"id": int(i), "text": vc.tokenizer.decode([i])} for i in ids]

    inject_tail_chat = _tail_ids(chat_prompts[0])
    inject_tail_bare = _tail_ids(bare_prompts[0])
    print(f"injection tail (chat) = {inject_tail_chat}")
    print(f"injection tail (bare) = {inject_tail_bare}")
    if inject_tail_chat[-1]["id"] == inject_tail_bare[-1]["id"]:
        print("[note] chat and bare end on the SAME token id -- the anchor "
              "survived the chat template's trim.")
    else:
        print("[note] chat and bare end on DIFFERENT token ids -- the chat "
              "template trimmed the 'Answer: ' anchor, so the injection site "
              "moved to the assistant header. Recorded in meta.")

    prompt_sha256 = hashlib.sha256(
        "\n".join(chat_prompts).encode("utf-8")).hexdigest()
    bare_prompt_sha256 = hashlib.sha256(
        "\n".join(bare_prompts).encode("utf-8")).hexdigest()
    questions_file_sha256 = hashlib.sha256(
        open(args.questions, "rb").read()).hexdigest()
    sample_order_sha256 = hashlib.sha256(
        "\n".join(str(s["sample_id"]) for s in samples).encode("utf-8")
    ).hexdigest()

    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"chat_template_hash={chat_template_hash}")
    print(f"chat_prompt_sha256={prompt_sha256}")
    print(f"bare_prompt_sha256={bare_prompt_sha256}")
    print(f"sample_order_sha256={sample_order_sha256}")
    print("First wrapped prompt (repr, truncated to 400 chars):")
    print(repr(chat_prompts[0][:400]))

    # Band + overwrite are checked for EVERY cell BEFORE the first cell runs.
    out_paths = {}
    for alpha, (ls, le) in cfgs:
        if (ls, le) != BAND:
            die(f"layer band {(ls, le)} != {BAND} -- this protocol is frozen at "
                "the bare cells' own band; do not pass a different band under "
                "this protocol name.")
        # Tag matches the bare tree's convention (mdf_neg8, not mdf_-8), which
        # is what get_answer_gsm_hard_blind.py writes.
        tag = f"mdf_{alpha}".replace("-", "neg")
        out_paths[alpha] = os.path.join(
            args.out_dir, tag, f"gsm_hard_chat_{args.size}_{ls}_{le}.json")
    for alpha, op in out_paths.items():
        if os.path.exists(op):
            die(f"{op} already exists -- refusing to overwrite a frozen "
                "chat-sweep cell. Delete it deliberately first if a re-run is "
                "truly intended.")

    n_layers = len(utils.decoder_layer_range(*BAND))

    for alpha, (ls, le) in cfgs:
        out_path = out_paths[alpha]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                      desc=f"gsm-hard-chat a={alpha}"):
            batch = chat_prompts[i: i + args.batch_size]
            gen.extend(vc.regenerate(
                batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                diff_matrices=list(diff),
                batch_size=args.batch_size,
                return_metadata=True,
            ))

        if len(gen) != n:
            die(f"generation returned {len(gen)} rows for {n} prompts at "
                f"alpha={alpha}; zip() would silently drop "
                f"{abs(len(gen) - n)} sample(s).")

        fires = vc.steering_fire_count()
        expect = 0 if alpha == 0 else n_layers * n
        if fires != expect:
            die(f"steering_fires {fires} != {expect} (L={n_layers}, n={n}, "
                f"alpha={alpha}); the intervention is unverified, so the cell "
                "is not usable.")

        # NO correctness field, NO accuracy print -- the gold is not reachable
        # from this script by construction (see the module docstring).
        rows = [{"sample_id": s["sample_id"], "question": s["question"],
                 "generated": g["text"],
                 "generated_token_count": g["generated_token_count"],
                 "stop_reason": g["stop_reason"]}
                for s, g in zip(samples, gen)]

        json.dump({
            "meta": {
                "protocol": PROTOCOL,
                "task": "gsm_hard",
                "model": "llama3", "size": args.size,
                "alpha": alpha, "layer_start": ls, "layer_end": le,
                "L": n_layers,
                "mask_path": args.mask_path, "mask_sha256": mask_sha256,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "batch_size": args.batch_size,
                "suite": "default", "cot": False, "fmt_wording": "plain",
                "role": "neutral",
                "prefill_only": True, "prefill_tail_len": 1,
                "prompt_wrapper_id": PROMPT_WRAPPER_ID,
                "chat_template_applied": True,
                "chat_template_hash": chat_template_hash,
                "prompt_template": neutral_tmpl,
                "prompt_body_sha256": prompt_body_sha256,
                "prompt_sha256": prompt_sha256,
                "bare_prompt_sha256": bare_prompt_sha256,
                "questions_sha256": qmeta["questions_sha256"],
                "questions_file_sha256": questions_file_sha256,
                "sample_order_sha256": sample_order_sha256,
                "steering_fires": fires,
                "padding_side": vc.tokenizer.padding_side,
                "injection_tail_tokens_chat": inject_tail_chat,
                "injection_tail_tokens_bare_reference": inject_tail_bare,
                "injection_site_moved_vs_bare": (
                    inject_tail_chat[-1]["id"] != inject_tail_bare[-1]["id"]),
                "contains_labels": False,
                "n": n,
                "provenance": device_note,
                "not_a_workpoint_search": True,
                # NOT MEASURED, and deliberately not faked. track_dopamine_signal.py's
                # hook reads hs[0] and stores ONE scalar per forward, i.e. it is
                # bs=1-only; this sweep runs bs=24 to stay identical to the bare
                # curve's generation path. Reusing it would mean either bs=1 (a
                # DIFFERENT padding regime than the cell it is meant to attest) or
                # rewriting the hook -- both change the generation path the
                # protocol exists to hold fixed. The closed form
                # alpha*mean_l||m_l||^2 is NOT a substitute: that identity holds
                # ONLY at the FIRST steered layer, because each later layer also
                # carries the propagated residue of the layers below it.
                # steering_fires attests the intervention FIRED; it does not
                # measure its magnitude.
                "g_prefill_measured": False,
                "g_prefill_omitted_reason": (
                    "signal hook is bs=1-only; measuring it would change the "
                    "generation path this cell exists to hold fixed"),
                "bare_counterpart": (
                    f"components/llama3/gsm_hard_p3/mdf_{alpha}/"
                    f"gsm_hard_{args.size}_{ls}_{le}.json".replace(
                        f"mdf_{alpha}", f"mdf_{alpha}".replace("-", "neg"))
                    if alpha in (-8, -6, -4, 0, 4) else None),
                "bare_counterpart_exists": alpha in (-8, -6, -4, 0, 4),
                "bare_coverage_note": (
                    "The frozen bare GSM-Hard P3 tree carries only five No-CoT "
                    "doses (-8/-6/-4/0/+4). alpha in (-2,+2,+6,+8) has NO bare "
                    "counterpart, so those cells are chat-only and cannot enter "
                    "a paired bare-vs-chat contrast. Bare-vs-chat is DESCRIPTIVE "
                    "only and is never pooled with the chat family's statistics."),
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  steering_fires={fires}  n={n}")

    print("\nAll GSM-Hard chat-sweep cells finished. "
          "NO accuracy was computed -- by construction.")
    print("Next: python3.10 RoleAnswer/analyze_gsm_hard_chat_sweep.py")


if __name__ == "__main__":
    main()
