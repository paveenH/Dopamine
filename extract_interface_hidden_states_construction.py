#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
extract_interface_hidden_states_construction.py -- sibling of
extract_interface_hidden_states.py, scoped to GSM8K ONLY, reading the LARGER
GSM8K construction split (components/benchmark/gsm8k_construction_split.json,
n~1019, the same set the formal construction-split ARRSN is built from)
instead of the frozen 300-question benchmark file.

WHY THIS SCRIPT EXISTS. To compare the formal construction-split ARRSN
(ARRSN_final = gsm8k_abstention_expert_mean - gsm8k_abstention_non_expert_mean,
built from gsm8k_construction_split.json) against a Chat direction
(Chat - Bare) on a sample-composition-matched basis, the Chat direction must
be built from the IDENTICAL 1019 questions in the IDENTICAL order -- not the
existing frozen 300-question chat/bare H5 files, whose sample set differs
from the construction split by construction (the 300 are the RESERVED
questions, deliberately excluded from the construction split). Comparing
ARRSN_final against the frozen 300-question Chat direction would confound the
comparison with sample-composition differences.

WHAT THIS SCRIPT IS AND IS NOT (identical framing to the parent script). It
is a pure forward-pass HS dumper: tokenize one rendered prompt, run ONE
`model(**tokens, output_hidden_states=True)` call (no `model.generate`, no
forward hooks of any kind), and save the last valid prefill-token hidden
state at EVERY layer (embedding output + every decoder-layer output). It is
NOT a fork of any generation script and never calls vc.regenerate/vc.generate
or any hook-registering method on VicundaModel.

WHAT IS IDENTICAL TO THE PARENT SCRIPT (extract_interface_hidden_states.py),
copied rather than re-derived:
  - GSM8K prompt body: template.build_gsm8k_default_suite(cot=False,
    wording="plain")["neutral"] -- same function, same suite, same wording.
  - Bare condition: the rendered body used as a plain string, no chat
    template applied.
  - Chat condition: tokenizer.apply_chat_template([{"role": "user",
    "content": body}], tokenize=False, add_generation_prompt=True), then
    strip_leading_bos -- byte-identical logic to the parent script's
    render_chat_prompt()/strip_leading_bos().
  - Model configs (MODELS dict): SAME model dirs, SAME
    expected_tail_bare/expected_tail_chat token id/text pairs, SAME
    expected_n_layers_out/expected_hidden_size -- these are properties of the
    MODEL and the PROMPT SHAPE (neutral GSM8K body + chat wrapping), not of
    which question set is used, so they carry over unchanged.
  - Fail-closed checks: n must equal exactly the manifest's own recorded
    construction_split_n (not hardcoded to 300 -- see below); chat vs bare
    must share the identical ordered question list (pairing-digest
    cross-check via --print_pairing_digest_only/--expect_pairing_digest);
    the final prefill token id is read out and asserted for EVERY sample; no
    double-BOS; all extracted hidden states finite; atomic write.
  - One forward pass per sample, bs=1, prefill-only, last valid prefill
    token's hidden state at every layer, cast to float16.

WHAT IS DIFFERENT FROM THE PARENT SCRIPT:
  - GSM8K ONLY. No --task flag; no MATH, no GSM-Hard. The instruction this
    script implements is explicitly scoped to GSM8K only, "other tasks not
    included for now".
  - Reads components/benchmark/gsm8k_construction_split.json (n~1019) by
    default instead of gsm8k_test_sample.json (n=300). n is NOT hardcoded;
    it is read from the loaded file AND cross-checked against
    construction_split_n recorded in gsm8k_construction_split_manifest.json
    (same self-consistency pattern
    extract_gsm8k_abstention_role_hs_construction.py uses against that same
    manifest -- not re-derived from memory, read from that script's own
    convention).
  - Writes to a SEPARATE output subtree,
    <base_dir>/hidden_states/<model>/gsm8k_construction/<condition>_<size>.h5,
    which NEVER collides with the parent script's
    <base_dir>/hidden_states/<model>/gsm8k/<condition>_<size>.h5 (the
    300-question chat/bare files) or with
    <base_dir>/hidden_states/<model>/gsm8k_abstention_role_construction/
    (the ARRSN construction H5 pair) -- three disjoint directories, three
    disjoint purposes.
  - EXPECT_N is read from the construction-split manifest at runtime, not a
    module constant, since the construction split's size is itself a derived
    quantity (full GSM8K test count minus 300), not a frozen literal.

SAMPLE FILES ARE REUSED VERBATIM, NEVER REGENERATED OR RE-SORTED.
components/benchmark/gsm8k_construction_split.json is read via utils.load_json
with NO reordering, matching build_gsm8k_construction_split.py's own
documented order ("original full-test-set index order, with the 300 excluded
items removed") and the pairing-digest convention it also uses
(sha256("\n".join(questions))).

NO STEERING, ANYWHERE. Same guarantee as the parent script -- see that
script's own module docstring for the verification detail (VicundaModel
methods this script calls register no hooks).

Output:
  <base_dir>/hidden_states/<model>/gsm8k_construction/<condition>_<size>.h5
  <base_dir>/hidden_states/<model>/gsm8k_construction/<condition>_<size>.manifest.json

@author: interface hidden-state extraction pipeline, construction-split
sibling (2026-09-17)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
import torch

import utils
from llms import VicundaModel
from template import build_gsm8k_default_suite

SCRIPT_VERSION = "extract_interface_hidden_states_construction-v1"

MODELS = {
    "llama3": {
        "model_dir_default": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "8B",
        "prompt_wrapper_id": "llama3-chat-template-v1",
        "expected_tail_bare": (220, " "),
        "expected_tail_chat": (271, "\n\n"),
        # embedding + 32 decoder layers = 33; hidden_size = 4096.
        "expected_n_layers_out": 33,
        "expected_hidden_size": 4096,
    },
    "qwen2.5": {
        "model_dir_default": "Qwen/Qwen2.5-7B-Instruct",
        "size": "7B",
        "prompt_wrapper_id": "qwen2.5-chat-template-v1",
        "expected_tail_bare": (220, " "),
        "expected_tail_chat": (198, "\n"),
        # embedding + 28 decoder layers = 29; hidden_size = 3584.
        "expected_n_layers_out": 29,
        "expected_hidden_size": 3584,
    },
}

CONDITIONS = ("chat", "bare")


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prefill-only, no-generation, no-steering hidden-state "
                    "extraction for one {model,condition} cell, on the "
                    "LARGER GSM8K construction split (GSM8K only).")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--model_dir", default=None,
                   help="HF repo id. Defaults to this model's standard repo "
                        "id if omitted.")
    p.add_argument("--base_dir", required=True,
                   help="components/ dir. Reads "
                        "<base_dir>/benchmark/gsm8k_construction_split.json "
                        "+ ..._manifest.json. Output goes to "
                        "<base_dir>/hidden_states/<model>/"
                        "gsm8k_construction/<condition>_<size>.h5")
    p.add_argument("--split_file", default=None,
                   help="Default: <base_dir>/benchmark/"
                        "gsm8k_construction_split.json")
    p.add_argument("--split_manifest_file", default=None,
                   help="Default: <base_dir>/benchmark/"
                        "gsm8k_construction_split_manifest.json")
    p.add_argument("--out_dir", default=None,
                   help="Default: <base_dir>/hidden_states/<model>/"
                        "gsm8k_construction/")
    p.add_argument("--expect_pairing_digest", default=None,
                   help="If given, the ordered-sample-identity sha256 "
                        "computed here must equal this value. Used to "
                        "cross-check chat vs bare share the identical "
                        "question set/order.")
    p.add_argument("--print_pairing_digest_only", action="store_true",
                   help="Print the ordered-sample-identity sha256 and exit 0 "
                        "without loading any model.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Prompt construction -- IDENTICAL logic to
# extract_interface_hidden_states.py's GSM8K path (reimplemented here, not
# imported, for the same standalone-sibling reason
# extract_gsm8k_abstention_role_hs_construction.py gives for its own
# duplication: each construction-split sibling stays fully independent of
# its 300-question counterpart).
# --------------------------------------------------------------------------

def render_bare_prompt(question: str, templates: dict) -> str:
    return templates["neutral"].format(context=question)


def strip_leading_bos(vc, text: str) -> str:
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def render_chat_prompt(vc, bare_prompt: str) -> str:
    wrapped = vc.tokenizer.apply_chat_template(
        [{"role": "user", "content": bare_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    return strip_leading_bos(vc, wrapped)


def assert_chat_template_effective(vc, bare_prompt: str, wrapped: str) -> None:
    if wrapped == bare_prompt:
        die("chat template had NO effect on the first sample (wrapped text "
            "is byte-identical to the bare prompt) -- apply_chat_template "
            "did not actually run, or the tokenizer has no chat_template.")
    if not getattr(vc.tokenizer, "chat_template", None):
        die("tokenizer.chat_template is empty/None -- cannot apply a chat "
            "template that does not exist.")


# --------------------------------------------------------------------------
# Tokenization + fail-closed checks -- identical to the parent script.
# --------------------------------------------------------------------------

def tokenize_single(vc, text: str):
    return vc.tokenizer(text, return_tensors="pt", add_special_tokens=True)


def assert_no_double_bos(vc, input_ids_row) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = input_ids_row.tolist()
    if bos_id is not None and len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
        die(f"double BOS detected (head={ids[:4]}) -- strip_leading_bos did "
            "not remove a chat-template-serialized BOS before "
            "add_special_tokens=True tokenization.")


def assert_final_token(model_key: str, condition: str, tok_id: int,
                       tok_text: str, sample_idx: int) -> None:
    expected_id, expected_text = MODELS[model_key][
        f"expected_tail_{condition}"]
    if tok_id != expected_id or tok_text != expected_text:
        die(f"sample_idx={sample_idx}: final prefill token is "
            f"id={tok_id} text={tok_text!r}, expected id={expected_id} "
            f"text={expected_text!r} for model={model_key} "
            f"condition={condition}. Refusing to extract hidden states "
            "under an unverified prefill boundary.")


# --------------------------------------------------------------------------
# Pairing digest -- identical convention to the parent script AND to
# build_gsm8k_construction_split.py's own sha256_of_list().
# --------------------------------------------------------------------------

def sha256_of_list(items: list) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    model_key = args.model
    condition = args.condition
    mcfg = MODELS[model_key]
    size = mcfg["size"]
    model_dir = args.model_dir or mcfg["model_dir_default"]

    base_dir = args.base_dir
    split_file = args.split_file or os.path.join(
        base_dir, "benchmark", "gsm8k_construction_split.json")
    split_manifest_file = args.split_manifest_file or os.path.join(
        base_dir, "benchmark", "gsm8k_construction_split_manifest.json")
    out_dir = args.out_dir or os.path.join(
        base_dir, "hidden_states", model_key, "gsm8k_construction")

    if not os.path.isfile(split_file):
        die(f"construction split file not found: {split_file} -- run "
            "build_gsm8k_construction_split.py first.")
    if not os.path.isfile(split_manifest_file):
        die(f"construction split manifest not found: {split_manifest_file} "
            "-- run build_gsm8k_construction_split.py first.")

    # ---- Load samples (no model needed for this part) ----
    samples = utils.load_json(split_file)
    templates = build_gsm8k_default_suite(cot=False, wording="plain")
    question_key = "question"

    n = len(samples)

    with open(split_manifest_file, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    expected_n_from_manifest = split_manifest.get("construction_split_n")
    if expected_n_from_manifest is None:
        die(f"{split_manifest_file} has no 'construction_split_n' key -- "
            "cannot cross-check n against the split builder's own record.")
    if int(expected_n_from_manifest) != n:
        die(f"loaded {n} construction-split samples from {split_file}, but "
            f"{split_manifest_file} records construction_split_n="
            f"{expected_n_from_manifest}. Refusing to proceed on a "
            "mismatched sample count.")

    pairing_ids = [s[question_key] for s in samples]
    pairing_digest = sha256_of_list(pairing_ids)
    manifest_digest = split_manifest.get(
        "construction_split_ordered_sample_identity_sha256")
    if not manifest_digest or manifest_digest != pairing_digest:
        die(f"recomputed ordered_sample_identity_sha256={pairing_digest} != "
            f"the split builder's own recorded "
            f"construction_split_ordered_sample_identity_sha256="
            f"{manifest_digest!r}. Either the key is missing/empty in "
            f"{split_manifest_file} or the construction split file has been "
            "modified since it was built; refusing to proceed either way.")

    print(f"[pairing] task=gsm8k_construction n={n} "
          f"ordered_sample_identity_sha256={pairing_digest}")

    if args.print_pairing_digest_only:
        print(pairing_digest)
        return

    if args.expect_pairing_digest:
        if pairing_digest != args.expect_pairing_digest:
            die(f"ordered_sample_identity_sha256 {pairing_digest} != "
                f"--expect_pairing_digest {args.expect_pairing_digest}. "
                "The chat and bare conditions must operate on the "
                "IDENTICAL ordered question list; refusing to proceed with "
                "a mismatched pairing.")

    questions_file_sha256 = hashlib.sha256(
        open(split_file, "rb").read()).hexdigest()

    # ---- Overwrite preflight for THIS CELL's own output files only, ----
    # ---- BEFORE model load. ----
    out_path = os.path.join(out_dir, f"{condition}_{size}.h5")
    meta_path = os.path.join(out_dir, f"{condition}_{size}.manifest.json")
    existing = [p for p in (out_path, meta_path) if os.path.exists(p)]
    if existing:
        die(f"refusing to run model={model_key} task=gsm8k_construction "
            f"condition={condition}: target file(s) already exist:\n  " +
            "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    print(f"model={model_key} task=gsm8k_construction condition={condition} "
          f"n={n} out_path={out_path}")

    # ---- Model load ----
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()

    chat_template_str = getattr(vc.tokenizer, "chat_template", None) or ""
    chat_template_hash = hashlib.sha256(
        chat_template_str.encode("utf-8")).hexdigest()
    if condition == "chat" and not chat_template_str:
        die("condition=chat but tokenizer.chat_template is empty/None -- "
            "cannot apply a chat template that does not exist.")

    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get(
                       "CUDA_VISIBLE_DEVICES"),
                   "param_devices": devs,
                   "sharded": len(devs) > 1}

    n_decoder = len(vc._find_decoder_layers())
    n_layers_out = n_decoder + 1  # embedding + every decoder layer output

    expect_layers = mcfg["expected_n_layers_out"]
    expect_hidden = mcfg["expected_hidden_size"]
    actual_hidden = int(vc.model.config.hidden_size)
    if n_layers_out != expect_layers or actual_hidden != expect_hidden:
        die(f"model shape mismatch for --model {model_key} "
            f"(--model_dir {model_dir}): got n_layers_out={n_layers_out} "
            f"hidden_size={actual_hidden}, expected "
            f"n_layers_out={expect_layers} hidden_size={expect_hidden}. "
            "Refusing to write output under a model key whose loaded "
            "architecture does not match.")

    # ---- Render prompts ----
    bare_prompts = [
        render_bare_prompt(s[question_key], templates) for s in samples]
    prompt_body_sha256 = hashlib.sha256(
        templates["neutral"].encode("utf-8")).hexdigest()

    if condition == "chat":
        rendered_prompts = []
        for i, p in enumerate(bare_prompts):
            wrapped = render_chat_prompt(vc, p)
            if i == 0:
                assert_chat_template_effective(vc, p, wrapped)
            rendered_prompts.append(wrapped)
    else:
        rendered_prompts = bare_prompts

    prompt_sha256 = hashlib.sha256(
        "\n".join(rendered_prompts).encode("utf-8")).hexdigest()

    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"prompt_sha256={prompt_sha256}")
    print(f"chat_template_hash={chat_template_hash}")
    print(f"chat_template_applied={condition == 'chat'}")
    print("First rendered prompt (repr, truncated to 400 chars):")
    print(repr(rendered_prompts[0][:400]))

    # ---- Extraction loop: ONE forward pass per sample, bs=1, prefill only,
    # ---- NO hooks, NO generation. ----
    hs_array = np.zeros((n, n_layers_out, vc.model.config.hidden_size),
                        dtype=np.float16)
    final_tok_ids = []
    final_tok_texts = []

    device = vc.model.device
    t0 = time.time()
    with torch.no_grad():
        for i, prompt in enumerate(rendered_prompts):
            enc = tokenize_single(vc, prompt)
            input_ids = enc["input_ids"][0]
            assert_no_double_bos(vc, input_ids)

            last_id = int(input_ids[-1].item())
            last_text = vc.tokenizer.decode([last_id])
            assert_final_token(model_key, condition, last_id, last_text, i)
            final_tok_ids.append(last_id)
            final_tok_texts.append(last_text)

            enc = {k: v.to(device) for k, v in enc.items()}
            out = vc.model(**enc, return_dict=True,
                           output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states
            if len(hidden_states) != n_layers_out:
                die(f"sample_idx={i}: got {len(hidden_states)} hidden_states "
                    f"layers, expected {n_layers_out} (embedding + "
                    f"{n_decoder} decoder layers).")

            for l, layer_hs in enumerate(hidden_states):
                vec = layer_hs[0, -1, :].detach().float().cpu().numpy()
                if not np.all(np.isfinite(vec)):
                    die(f"sample_idx={i} layer={l}: non-finite hidden "
                        "state values (NaN/Inf) encountered.")
                hs_array[i, l, :] = vec.astype(np.float16)

            if (i + 1) % 100 == 0 or i == n - 1:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{n}] elapsed={elapsed:.1f}s")

    if not np.all(np.isfinite(hs_array.astype(np.float32))):
        die("post-hoc finiteness check failed on the assembled hidden-state "
            "array -- this should be unreachable given the per-sample check "
            "above, so treat this as a hard bug rather than data.")

    final_tail_unique = sorted(set(final_tok_ids))
    print(f"final prefill token ids observed across all {n} samples: "
          f"{final_tail_unique}")
    if len(final_tail_unique) != 1:
        die(f"final prefill token id is NOT constant across samples "
            f"({final_tail_unique}) -- every sample was already individually "
            "asserted against the expected id above, so this indicates the "
            "assertion itself is unreliable; treat as a hard bug.")

    # ---- Atomic write ----
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_h5_path = tempfile.mkstemp(
        suffix=".h5.tmp", dir=out_dir, prefix=f"{condition}_{size}_")
    os.close(fd)
    try:
        with h5py.File(tmp_h5_path, "w") as f:
            ds = f.create_dataset(
                "hidden_states", data=hs_array, dtype="float16",
                chunks=(1, n_layers_out, vc.model.config.hidden_size))
            ds.attrs["shape_note"] = (
                "(n_samples, n_layers, hidden_size); n_layers index 0 = "
                "embedding output, 1..N = decoder layer outputs "
                "(output_hidden_states=True convention)")
            f.create_dataset(
                "final_token_id",
                data=np.array(final_tok_ids, dtype=np.int64))
            f.create_dataset(
                "final_token_text",
                data=np.array(final_tok_texts,
                             dtype=h5py.string_dtype(encoding="utf-8")))
            f.create_dataset(
                "question",
                data=np.array([s[question_key] for s in samples],
                             dtype=h5py.string_dtype(encoding="utf-8")))

            meta = f.attrs
            meta["script_version"] = SCRIPT_VERSION
            meta["git_commit"] = git_commit()
            meta["model"] = model_key
            meta["model_dir"] = model_dir
            meta["model_revision"] = (
                "unpinned, uses default HF revision")
            meta["task"] = "gsm8k_construction"
            meta["condition"] = condition
            meta["n_samples"] = n
            meta["n_samples_done"] = n
            meta["hidden_state_shape"] = json.dumps(
                [n, n_layers_out, int(vc.model.config.hidden_size)])
            meta["hidden_state_dtype"] = "float16"
            meta["n_layers"] = n_layers_out
            meta["n_decoder_layers"] = n_decoder
            meta["hidden_size"] = int(vc.model.config.hidden_size)
            meta["questions_sha256"] = questions_file_sha256
            meta["ordered_sample_identity_sha256"] = pairing_digest
            meta["construction_split_manifest_digest"] = manifest_digest or ""
            meta["prompt_body_sha256"] = prompt_body_sha256
            meta["prompt_sha256"] = prompt_sha256
            meta["chat_template_applied"] = condition == "chat"
            meta["chat_template_hash"] = chat_template_hash
            meta["prompt_wrapper_id"] = (
                mcfg["prompt_wrapper_id"] if condition == "chat" else "bare")
            meta["final_prefill_token_id"] = final_tail_unique[0]
            meta["final_prefill_token_text"] = final_tok_texts[0]
            meta["steering_applied"] = False
            meta["generation_performed"] = False
            meta["prefill_only"] = True
            meta["batch_size"] = 1
            meta["extraction_timestamp_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta["provenance"] = json.dumps(device_note)

        os.replace(tmp_h5_path, out_path)
    except Exception:
        if os.path.exists(tmp_h5_path):
            os.remove(tmp_h5_path)
        raise

    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": git_commit(),
        "model": model_key,
        "model_dir": model_dir,
        "model_revision": "unpinned, uses default HF revision",
        "task": "gsm8k_construction",
        "condition": condition,
        "n_samples": n,
        "n_samples_done": n,
        "hidden_state_shape": [n, n_layers_out,
                               int(vc.model.config.hidden_size)],
        "hidden_state_dtype": "float16",
        "questions_sha256": questions_file_sha256,
        "ordered_sample_identity_sha256": pairing_digest,
        "construction_split_manifest_digest": manifest_digest or "",
        "prompt_body_sha256": prompt_body_sha256,
        "prompt_sha256": prompt_sha256,
        "chat_template_applied": condition == "chat",
        "chat_template_hash": chat_template_hash,
        "prompt_wrapper_id": (
            mcfg["prompt_wrapper_id"] if condition == "chat" else "bare"),
        "final_prefill_token_id": final_tail_unique[0],
        "final_prefill_token_text": final_tok_texts[0],
        "steering_applied": False,
        "generation_performed": False,
        "prefill_only": True,
        "batch_size": 1,
        "extraction_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": device_note,
        "out_path": out_path,
        "note": "GSM8K-only construction-split sibling of "
                "extract_interface_hidden_states.py, built to sample-"
                "composition-match the formal construction-split ARRSN "
                "(same gsm8k_construction_split.json question set/order). "
                "Other tasks (MATH, GSM-Hard) are deliberately NOT included.",
    }
    tmp_manifest = meta_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_manifest, meta_path)

    print(f"wrote {out_path}")
    print(f"wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
