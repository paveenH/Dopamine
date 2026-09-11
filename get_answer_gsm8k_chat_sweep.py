#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_gsm8k_chat_sweep.py -- GSM8K No-CoT, chat-template INTERFACE
condition, full 9-point alpha sweep. INDEPENDENT of the frozen bare-string
GSM8K main line (run_gsm8k.sh / get_answer_regenerate_gsm8k.py) -- neither is
touched, read, or overwritten by this script.

Motivation (see CLAUDE.md's ProofWriter-OWA chat-template diagnostic line for
the precedent this follows): the frozen bare-string GSM8K curve shows Llama3
looping heavily at every alpha (§2.3 raw loop rate 74-88%, no clean alpha
trend) and CLAUDE.md's chat-template-alignment rule says bare-string is the
steering-aligned default because the NMD mask/diff vectors were extracted on
bare prompts. This experiment asks a narrower, orthogonal question: does
wrapping the SAME neutral GSM8K prompt body in the HF chat template reduce
that loop baseline, and if so, does the alpha=-6-region commitment/accuracy
workpoint survive the interface change?

THIS IS NOT A WORKPOINT SEARCH. alpha in {-8,-6,-4,-2,0,2,4,6,8} is the SAME
dose set as the frozen bare 9-point No-CoT sweep (GSM8K_DIRS mdf_-8..mdf_8 in
RoleAnswer/analyze_first_last_acc.py) -- not re-selected, not narrowed.

WHAT IS HELD IDENTICAL to the frozen bare run_gsm8k.sh / mdf_0..mdf_8 cells:
  - the SAME 300-question fixed sample (benchmark/gsm8k_test_sample.json)
  - the SAME neutral, No-CoT, plain-wording prompt BODY
    (template.build_gsm8k_default_suite(cot=False, wording="plain")["neutral"])
  - the SAME mask file and band: mask/llama3_non_logits/nmd_0.5_11_20_8B.npy,
    layers [11,20)
  - the SAME generation params: max_new_tokens=768, temperature=0.0 (greedy),
    top_p defaults to 1.0 under greedy (regenerate() ignores top_p when
    temperature<=0, matching get_answer_regenerate_gsm8k.py's do_sample logic)
  - batch_size=24, prefill_only=True, prefill_tail_len=1
  - alpha=0 uses the SAME generation code path as every other alpha (an
    all-zero diff matrix via vc.regenerate(diff_matrices=raw_mask*0) -- no
    separate vc.generate() branch), but is RE-RUN under the chat wrapper
    rather than reused from the bare baseline (the bare alpha=0 cell used a
    different prompt string and cannot serve as this sweep's own baseline)
  - the SAME answer extractor (utils.extract_gsm8k_answer /
    utils.is_correct_gsm8k) for the inline process-state fields; the
    AUTHORITATIVE first_acc/last_acc reading is computed offline by
    RoleAnswer/analyze_gsm8k_chat_sweep.py, which imports the frozen
    extractors from RoleAnswer/analyze_first_last_acc.py rather than
    redefining them (per CLAUDE.md's "reuse the offline extractor" rule)

THE ONLY EXPERIMENTAL VARIABLE, held constant across all nine alpha, is how
the identical prompt STRING is wrapped before tokenization: the fully
rendered neutral GSM8K prompt (question inserted, "Provide your final
numeric answer after '####'." directive included, ending in "Answer: ") is
placed as the sole user-turn content in

    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )

with a duplicated leading BOS stripped before vc.regenerate() tokenizes it
(add_special_tokens=True internally) -- reusing the same
strip_leading_bos / assert_no_double_bos pattern
proofwriter_owa/get_answer_proofwriter_owa_chat.py uses (itself reusing
bandit_pv6_episode.py's _strip_leading_bos / double-BOS-hazard check), rather
than reinventing it.

KNOWN CONFOUND -- MEASURED, NOT ASSUMED, AND IT MUST TRAVEL WITH EVERY RESULT
FROM THIS SWEEP. apply_chat_template applies Jinja `| trim` to the user-turn
content, which STRIPS the trailing space of the frozen prompt body's
"Answer: " anchor. Verified on the real meta-llama/Llama-3.1-8B-Instruct
tokenizer, the last prompt token -- the ONLY position prefill-only tail=1
steering injects into -- therefore differs between the two conditions:

    bare : id 220 ' '     <- the "Answer: " anchor (the decision bottleneck
                             CGT/PV10 deliberately anchor on)
    chat : id 271 '\n\n'   <- the assistant generation header

So this experiment changes the interface AND the injection site TOGETHER.
That is intrinsic to chat wrapping (the generation header is always last;
this is the same Llama 271 / Qwen 198 header-token fact check_igt_qwen.py
records), so it cannot be removed while still testing a chat interface -- but
it means a chat-vs-bare difference may NOT be attributed to "the wrapper"
alone. Both tails are read out at run time and stored per cell as
meta.injection_tail_tokens_chat / _bare_reference /
meta.injection_site_moved_vs_bare, so the fact is in the artifact rather than
only in prose. CLAUDE.md separately records that a weak-looking injection
site is NOT a priori weak (the header token still carries the whole context's
hidden state) and that the remedy for a flat response is an ANCHOR change,
never a bigger alpha or a wider tail (tail-widening is over-steering).

NOTE per CLAUDE.md: get_answer_regenerate_gsm8k.py declares --use_chat but
never actually branches on it in prompt construction -- that flag is
effectively dead. This script does NOT add a --use_chat flag to the bare
script; it is a wholly separate script/launcher pair so the frozen bare
pipeline is never at risk of a "looks enabled but isn't" regression.

Output tree (independent from the bare answer_mdf_gsm8k tree):
    components/llama3/answer_mdf_gsm8k_chat_v1/mdf_<alpha>/
        gsm8k_chat_8B_answers_11_20.json

Existing output files are NEVER overwritten -- fail closed (die()), matching
the P3/P4/P4b/P4c/ProofWriter-chat convention in this repo. There is no
--allow_overwrite escape hatch (CLAUDE.md: "a generator of frozen artifacts
must ship no --allow_overwrite escape hatch").

All nine alpha are driven by ONE launcher invocation, ONE model load, so the
same-machine/same-GPU pairing CLAUDE.md requires for a dose curve holds
structurally rather than by convention.

@author: GSM8K chat-template interface sweep (2026-09-11)
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
from template import build_gsm8k_default_suite
from utils import extract_gsm8k_answer, is_correct_gsm8k

PROTOCOL = "gsm8k-chat-sweep-v1"
PROMPT_WRAPPER_ID = "llama3-chat-template-v1"

# Fixed, frozen -- the SAME nine-point No-CoT dose set as the bare main line
# (GSM8K_DIRS mdf_-8..mdf_8 in RoleAnswer/analyze_first_last_acc.py). Not a
# free CLI list: --configs is validated against this set below so an
# out-of-family alpha cannot be silently run under this protocol name.
EXPECTED_ALPHAS = {-8, -6, -4, -2, 0, 2, 4, 6, 8}


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM8K No-CoT chat-template interface sweep (independent "
                    "of the frozen bare run_gsm8k.sh line)")
    p.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Hugging Face repo id, per the repo-wide convention "
                        "(never a local filesystem path unless the caller "
                        "explicitly overrides this).")
    p.add_argument("--size", default="8B")
    p.add_argument("--test_file", required=True,
                   help="benchmark/gsm8k_test_sample.json -- the SAME 300-"
                        "question fixed sample the bare sweep uses")
    p.add_argument("--mask_path", required=True,
                   help="SAME mask file as the bare sweep: "
                        "mask/llama3_non_logits/nmd_0.5_11_20_8B.npy")
    p.add_argument("--configs", required=True, nargs="+",
                   help="e.g. 0-11-20 neg8-11-20 neg6-11-20 ... 8-11-20 "
                        "(alpha restricted to the frozen -8..+8 dose set)")
    p.add_argument("--out_dir", required=True,
                   help="components/llama3/answer_mdf_gsm8k_chat_v1")
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--fmt_wording", default="plain", choices=["plain"],
                   help="Fixed to 'plain' (the main-line wording) -- 'pushy' "
                        "is not part of this experiment's scope.")
    return p.parse_args()


def strip_leading_bos(vc, text: str) -> str:
    """Byte-identical logic to proofwriter_owa/get_answer_proofwriter_owa_chat.py's
    strip_leading_bos (itself reusing bandit_pv6_episode.py's
    _strip_leading_bos): vc.regenerate's _regenerate_prefill_only tokenizes
    with add_special_tokens=True, so an un-stripped chat-templated string
    that already serialized a BOS string yields two leading BOS ids."""
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def assert_no_double_bos(vc, text: str, label: str) -> None:
    """Hard invariant: tokenize with add_special_tokens=True (the same call
    vc.regenerate makes internally) and refuse if the first two ids are both
    BOS."""
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and len(ids) >= 2 and ids[:2] == [bos_id, bos_id]:
        die(f"double BOS in the {label} chat prompt (head={ids[:4]}) -- "
            "strip_leading_bos did not remove the chat template's own "
            "serialized BOS text before this add_special_tokens=True call.")


def assert_chat_template_effective(vc, bare_prompt: str, wrapped: str) -> None:
    """First-sample assertions the task explicitly requires: the chat
    template actually fired (wrapped != bare-string tokenizer.apply would be
    a no-op template), and the assistant generation header is present."""
    if wrapped == bare_prompt:
        die("chat template had NO effect on the first sample (wrapped text "
            "is byte-identical to the bare prompt) -- apply_chat_template "
            "did not actually run, or the tokenizer has no chat_template.")
    chat_template_str = getattr(vc.tokenizer, "chat_template", None)
    if not chat_template_str:
        die("tokenizer.chat_template is empty/None -- cannot apply a chat "
            "template that does not exist.")
    header = "<|start_header_id|>assistant<|end_header_id|>"
    if header not in wrapped:
        die(f"expected assistant generation header {header!r} not found in "
            "the first wrapped prompt -- add_generation_prompt=True did not "
            "produce the expected Llama-3 chat header.")


def main():
    args = parse_args()

    cfgs = utils.parse_configs(args.configs)
    got_alphas = {int(al) for al, _ in cfgs}
    if not got_alphas.issubset(EXPECTED_ALPHAS):
        die(f"alpha values {sorted(got_alphas - EXPECTED_ALPHAS)} are outside "
            f"this protocol's frozen dose set {sorted(EXPECTED_ALPHAS)}. This "
            "sweep reuses the bare main line's own 9-point dose family and "
            "does not search new alpha.")

    samples = utils.load_json(args.test_file)
    n = len(samples)
    print(f"Loaded {n} GSM8K samples from {args.test_file}")

    templates = build_gsm8k_default_suite(cot=False, wording=args.fmt_wording)
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
    # (32 for Llama3-8B) -- raw_mask is already full-length with zero rows
    # outside the band. Slicing it to the band is the exact bug the ProofWriter
    # 30-item chat diagnostic hit ("diff_matrices length (9) != layers (32)").
    # Checked HERE rather than left to llms.py so it fails before the first
    # multi-minute cell instead of during it.
    n_decoder = len(vc._find_decoder_layers())
    if raw_mask.shape[0] != n_decoder:
        die(f"mask has {raw_mask.shape[0]} rows but the model has {n_decoder} "
            "decoder layers; regenerate() needs one row per layer (zero rows "
            "outside the band). Do NOT slice the mask to the band.")
    nz_rows = sorted(int(i) for i in np.nonzero(np.any(raw_mask != 0, axis=1))[0])
    want_rows = sorted(utils.decoder_layer_range(11, 20))
    if nz_rows != want_rows:
        die(f"mask non-zero rows {nz_rows} != decoder_layer_range(11,20) "
            f"{want_rows}; the mask does not match this protocol's band.")

    # Provenance, per CLAUDE.md's repo-wide rule: record the device rather than
    # constrain it. All nine cells share this record because they share ONE
    # model load in ONE invocation.
    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "param_devices": devs,
                   "sharded": len(devs) > 1}

    # Render every bare prompt ONCE, wrap it ONCE -- shared byte-identically
    # by every alpha of this sweep, matching the ProofWriter-chat convention
    # so the digests below attest "same prompt across the whole curve".
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

    # INJECTION SITE -- READ OUT, NEVER ASSUMED (CLAUDE.md's cross-model
    # pre-flight rule). This is load-bearing and is NOT cosmetic:
    # apply_chat_template applies `| trim` to the user content, which STRIPS
    # the trailing space of the frozen body's "Answer: " anchor. So prefill-
    # only steering does NOT land on the same token in the two conditions:
    #   bare : id 220 ' '   <- the "Answer: " anchor (decision bottleneck)
    #   chat : id 271 '\n\n' <- the assistant generation header
    # i.e. the chat condition moves the wrapper AND the injection site
    # together. That is unavoidable for any chat wrapping (the header is
    # always last) but it must be reported with the result, never silently
    # folded into "the interface changed".
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
    test_file_sha256 = hashlib.sha256(
        open(args.test_file, "rb").read()).hexdigest()

    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"chat_template_hash={chat_template_hash}")
    print(f"chat_prompt_sha256={prompt_sha256}")
    print(f"bare_prompt_sha256={bare_prompt_sha256}")
    print(f"test_file_sha256={test_file_sha256}")
    print("First wrapped prompt (repr, truncated to 400 chars):")
    print(repr(chat_prompts[0][:400]))

    # Band + overwrite are checked for EVERY cell BEFORE the first cell runs.
    # Checking them per-cell inside the loop would let a nine-hour sweep die on
    # cell 9 after eight usable cells were already written.
    out_paths = {}
    for alpha, (ls, le) in cfgs:
        if (ls, le) != (11, 20):
            die(f"layer band {(ls, le)} != (11,20) -- this protocol is "
                "frozen at the bare sweep's own band; do not pass a "
                "different band under this protocol name.")
        out_paths[alpha] = os.path.join(
            args.out_dir, f"mdf_{alpha}",
            f"gsm8k_chat_{args.size}_answers_{ls}_{le}.json")
    for alpha, op in out_paths.items():
        if os.path.exists(op):
            die(f"{op} already exists -- refusing to overwrite a frozen "
                "chat-sweep cell. Delete it deliberately first if a re-run "
                "is truly intended.")

    n_layers = len(utils.decoder_layer_range(11, 20))

    for alpha, (ls, le) in cfgs:
        out_path = out_paths[alpha]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        diff = raw_mask * alpha
        vc.steering_fire_count(reset=True)

        gen = []
        for i in tqdm(range(0, len(chat_prompts), args.batch_size),
                      desc=f"gsm8k-chat-sweep a={alpha}"):
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
                f"alpha={alpha}); the intervention is unverified, so the "
                "cell is not usable.")

        rows = []
        for s, g in zip(samples, gen):
            text = g["text"]
            pred_answer = extract_gsm8k_answer(text)
            correct = is_correct_gsm8k(pred_answer, s["answer"])
            rows.append({
                "question": s["question"],
                "answer": s["answer"],
                "generated_neutral": text,
                "pred_answer_neutral": pred_answer,
                "correct_neutral": correct,
                "generated_token_count": g["generated_token_count"],
                "stop_reason": g["stop_reason"],
            })

        n_correct = sum(r["correct_neutral"] for r in rows)
        accuracy_pct = round(n_correct / n * 100, 2) if n else 0.0

        json.dump({
            "meta": {
                "protocol": PROTOCOL,
                "model": "llama3", "size": args.size,
                "alpha": alpha, "layer_start": ls, "layer_end": le, "L": n_layers,
                "mask_path": args.mask_path, "mask_sha256": mask_sha256,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_p": args.top_p,
                "batch_size": args.batch_size,
                "suite": "default", "cot": False, "fmt_wording": args.fmt_wording,
                "role": "neutral",
                "prefill_only": True, "prefill_tail_len": 1,
                "prompt_wrapper_id": PROMPT_WRAPPER_ID,
                "chat_template_applied": True,
                "chat_template_hash": chat_template_hash,
                "prompt_body_sha256": prompt_body_sha256,
                "prompt_sha256": prompt_sha256,
                "bare_prompt_sha256": bare_prompt_sha256,
                "test_file_sha256": test_file_sha256,
                "steering_fires": fires,
                "padding_side": vc.tokenizer.padding_side,
                "injection_tail_tokens_chat": inject_tail_chat,
                "injection_tail_tokens_bare_reference": inject_tail_bare,
                "injection_site_moved_vs_bare": (
                    inject_tail_chat[-1]["id"] != inject_tail_bare[-1]["id"]),
                "n": n,
                "accuracy_inline_pct": accuracy_pct,
                "provenance": device_note,
                "not_a_workpoint_search": True,
                # NOT MEASURED, and deliberately not faked. TrackDopamineSignal's
                # hook in track_dopamine_signal.py reads hs[0] and stores ONE
                # scalar per forward, i.e. it is bs=1-only; this sweep runs
                # bs=24 to stay identical to the bare curve's generation path.
                # Reusing it would mean either bs=1 (a DIFFERENT padding regime
                # than the cell it is meant to attest) or rewriting the hook --
                # both change the generation path, which the protocol forbids.
                # The closed form alpha*mean_l||m_l||^2 is NOT a substitute: per
                # CLAUDE.md that identity holds ONLY at the FIRST steered layer,
                # because each layer also carries the propagated residue of the
                # layers below it. steering_fires below attests the intervention
                # FIRED; it does not measure its magnitude.
                "g_prefill_measured": False,
                "g_prefill_omitted_reason": (
                    "signal hook is bs=1-only; measuring it would change the "
                    "generation path this cell exists to hold fixed"),
                "compares_against": (
                    f"components/llama3/answer_mdf_gsm8k/mdf_{alpha}/"
                    f"gsm8k_{args.size}_answers_20_{ls}_{le}.json "
                    "(frozen bare sweep, neutral role, same alpha/band; "
                    "DESCRIPTIVE paired contrast only -- bare and chat are "
                    "never pooled into one statistical family)"),
            },
            "data": rows,
        }, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}  steering_fires={fires}  "
              f"inline_acc={accuracy_pct}%")

    print("\nAll chat-sweep cells finished.")
    print("Next: python RoleAnswer/analyze_gsm8k_chat_sweep.py "
          "(offline, reuses the frozen extractors; does NOT read gold here "
          "beyond the inline process-state fields already written)")


if __name__ == "__main__":
    main()
