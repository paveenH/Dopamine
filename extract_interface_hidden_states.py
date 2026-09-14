#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
extract_interface_hidden_states.py -- standalone, prefill-only, NO-generation,
NO-steering hidden-state extraction for the {chat,bare} INTERFACE condition,
for one (model, task, condition) cell at a time.

WHAT THIS SCRIPT IS AND IS NOT. It is a pure forward-pass HS dumper: tokenize
one rendered prompt, run ONE `model(**tokens, output_hidden_states=True)`
call (no `model.generate`, no forward hooks of any kind -- see the "no
steering" section below), and save the last valid prefill-token hidden state
at EVERY layer (embedding output + every decoder-layer output, i.e. every
entry of `outputs.hidden_states`). It is NOT a fork of, or a replacement for,
any existing get_answer_*.py generation script, and it never calls
vc.regenerate / vc.generate / any hook-registering method on VicundaModel.

PROMPT REUSE, NOT PROMPT RETYPING. This script imports the exact prompt
BODY builders the frozen chat-sweep / bare experiments use, rather than
retyping any prompt string:
  - GSM8K / GSM-Hard body: template.build_gsm8k_default_suite(cot=False,
    wording="plain")["neutral"] -- the SAME function
    get_answer_gsm8k_chat_sweep.py (line 239) and
    get_answer_gsm_hard_chat_sweep.py (via select_templates_gsm8k, line 254)
    both call. GSM-Hard reuses the GSM8K body verbatim (confirmed by reading
    get_answer_gsm_hard_chat_sweep.py's own docstring and its
    select_templates_gsm8k(suite="default", ...) call), it does not have its
    own body builder.
  - MATH body: template.build_math_suite(cot=False)["neutral"] -- the SAME
    function get_answer_math_chat_sweep.py calls (line 220).
  - chat wrapping: tokenizer.apply_chat_template([{"role": "user",
    "content": body}], tokenize=False, add_generation_prompt=True), followed
    by strip_leading_bos -- byte-identical in shape to the pattern used in
    get_answer_gsm8k_chat_sweep.py (lines 171-180, 292-301),
    get_answer_math_chat_sweep.py (lines 161-165, 260-270), and
    get_answer_gsm_hard_chat_sweep.py (lines 197-205, 304-314). Reimplemented
    here (not imported) because those functions are private, non-exported
    helpers duplicated verbatim across all three sibling scripts -- this
    script follows that repo's existing pattern (each task script owns a
    small local copy) rather than inventing new cross-script coupling; the
    LOGIC is copied, not retyped from scratch, and the resulting wrapped
    string is byte-identical to what those scripts would produce for the
    same input, verified structurally (same apply_chat_template call, same
    strip_leading_bos rule).
  - bare condition: the SAME rendered prompt body, used as a plain string
    with no chat template applied at all -- matching the frozen bare-string
    main lines (run_gsm8k.sh / run_math.sh / the P3 GSM-Hard line), which
    never call apply_chat_template.

SAMPLE FILES ARE REUSED VERBATIM, NEVER REGENERATED. GSM8K:
benchmark/gsm8k_test_sample.json (same file both run_gsm8k.sh and
run_gsm8k_qwen25.sh point at -- confirmed by reading both launchers, so it is
model-independent). MATH: benchmark/math_test_sample.json, truncated to the
first n_samples=300 EXACTLY as get_answer_regenerate_math.py and
get_answer_math_chat_sweep.py do (all_samples[:n]). GSM-Hard: the already
label-free, frozen P3 file components/benchmark/gsm_hard_p3_questions.json
(digest 48cc763545d2ee23835833f5165456b90db194863420a10b6741a57cff781d02,
docs/PREREG_P3.md) -- this script reuses it AS-IS, never re-downloads or
re-derives it, and never touches the sealed gold file.

NO STEERING, ANYWHERE. No mask is loaded, no diff matrix is built, no forward
hook (register_forward_hook / register_forward_pre_hook) is ever registered
by this script or by any VicundaModel method it calls
(VicundaModel.get_logits registers no hooks -- verified by reading llms.py;
hooks are registered only inside the regenerate()/_apply_diff_hooks family,
none of which this script calls).

ONE FORWARD PASS PER SAMPLE, bs=1, PREFILL ONLY.
`vc.model(**tokens, output_hidden_states=True, use_cache=False)` -- no
`model.generate`, so there is no decode step at all; "prefill-only" here
means "the only forward pass IS the prefill pass". `outputs.hidden_states`
is a tuple of length num_decoder_layers+1: index 0 is the embedding output,
indices 1..N are decoder-layer outputs (the same HF hidden_states indexing
CLAUDE.md's "Layer indexing convention" section describes for
`output_hidden_states=True`). This script stores ALL of them, not a
selective middle band (unlike track_hidden_states.py, whose selective
[LAYER_START,LAYER_END)+final-layer storage this script deliberately does
NOT follow -- only its fp16-cast-before-store pattern is reused, per the
task's own framing).

LAST VALID PREFILL TOKEN. Because this is a bs=1 forward pass on a
left-or-right padded tokenizer with no padding at all present (batch of one
prompt => `padding="longest"` never inserts a pad token), the last token of
`input_ids[0]` is unambiguously the last real token of the prompt; its
hidden state at every layer is `hidden_states[l][0, -1, :]`.

FAIL-CLOSED VALIDATION (see the module docstring sections below for detail):
  - overwrite preflight, BEFORE model load, for THIS CELL's own out_path and
    meta_path only. The launcher separately does a one-shot check of all 6
    of a model's target files (.h5 AND .manifest.json) before starting the
    first cell, so a fresh model run still refuses to clobber anything; but
    the extractor itself must not re-check the other 5 cells on every
    invocation, since cells 2..6 legitimately run after cell 1's files
    already exist;
  - n must equal exactly 300;
  - chat vs bare must share the identical ordered question list (sha256
    equality is asserted before EITHER condition's run starts, via a
    separate `--print_pairing_digest_only`/`--expect_pairing_digest` pass --
    see below for how the two independent process invocations coordinate
    this);
  - the final prefill token id is read out at runtime for EVERY sample (not
    just the first) and asserted against the expected id;
  - no double-BOS;
  - all extracted hidden states are finite;
  - atomic write: build the full (300, L, H) array in memory, verify it, THEN
    write to a temp path and os.replace() it into place, so a killed job
    never leaves a partial file at the final path.

@author: interface hidden-state extraction pipeline (2026-09-14)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from template import build_gsm8k_default_suite, build_math_suite

SCRIPT_VERSION = "extract_interface_hidden_states-v1"

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

TASKS = ("gsm8k", "gsm_hard", "math")
CONDITIONS = ("chat", "bare")

N_EXPECT = 300

# The frozen GSM-Hard P3 questions digest (docs/PREREG_P3.md, protocol p3-v1),
# reused here as a sanity check that the same file this repo already freezes
# is the one being read -- not re-derived.
GSM_HARD_QUESTIONS_SHA256 = (
    "48cc763545d2ee23835833f5165456b90db194863420a10b6741a57cff781d02")

FORBIDDEN_GSM_HARD_KEYS = ("answer", "gold", "gold_answer", "correct",
                          "accuracy", "target")


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prefill-only, no-generation, no-steering hidden-state "
                    "extraction for one {model,task,condition} cell.")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--model_dir", default=None,
                   help="HF repo id. Defaults to this model's standard repo "
                        "id (meta-llama/Llama-3.1-8B-Instruct or "
                        "Qwen/Qwen2.5-7B-Instruct) if omitted.")
    p.add_argument("--base_dir", required=True,
                   help="components/ dir, e.g. /data1/paveen/Dopamine/"
                        "components. Sample files are read from "
                        "<base_dir>/benchmark/ and "
                        "<base_dir>/benchmark/ (GSM-Hard). Output goes to "
                        "<base_dir>/hidden_states/<model>/<task>/"
                        "<condition>_<size>.h5")
    p.add_argument("--gsm8k_file", default=None,
                   help="Default: <base_dir>/benchmark/gsm8k_test_sample.json")
    p.add_argument("--math_file", default=None,
                   help="Default: <base_dir>/benchmark/math_test_sample.json")
    p.add_argument("--gsm_hard_file", default=None,
                   help="Default: <base_dir>/benchmark/"
                        "gsm_hard_p3_questions.json (the frozen P3 "
                        "label-free file, reused as-is)")
    p.add_argument("--out_dir", default=None,
                   help="Default: <base_dir>/hidden_states/<model>/<task>/")
    p.add_argument("--expect_pairing_digest", default=None,
                   help="If given, the ordered-sample-identity sha256 "
                        "computed here must equal this value. Used to "
                        "cross-check chat vs bare share the identical "
                        "question set/order without requiring one process "
                        "to see the other's in-memory state -- pass the "
                        "digest printed by the OTHER condition's run (or "
                        "precomputed by --print_pairing_digest_only).")
    p.add_argument("--print_pairing_digest_only", action="store_true",
                   help="Print the ordered-sample-identity sha256 for this "
                        "(task) and exit 0 without loading any model. Use "
                        "this to obtain the digest to pass as "
                        "--expect_pairing_digest to both conditions' runs, "
                        "or to eyeball-verify chat/bare will match before "
                        "spending GPU time.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Prompt construction -- REUSES the frozen body builders (see module
# docstring for exact citations). Chat wrapping follows the SAME
# apply_chat_template + strip_leading_bos shape used in
# get_answer_{gsm8k,math,gsm_hard}_chat_sweep.py.
# --------------------------------------------------------------------------

def load_gsm8k_samples(path: str):
    samples = utils.load_json(path)
    return samples


def load_math_samples(path: str, n: int):
    all_samples = utils.load_json(path)
    if len(all_samples) < n:
        die(f"MATH sample file {path} holds only {len(all_samples)} "
            f"problems but {n} are required.")
    return all_samples[:n]


def load_gsm_hard_samples(path: str, expect_sha: str):
    """Label-free loader, mirroring get_answer_gsm_hard_chat_sweep.py's
    load_questions(): refuse anything that does not declare itself
    label-free, and scan every row for a leaked label key. This script reads
    ONLY the already-frozen P3 questions file -- it never downloads or
    regenerates GSM-Hard data itself."""
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("GSM-Hard questions file does not declare contains_labels=false; "
            "this extractor must stay on a label-free code path, matching "
            "the frozen P3 generation scripts.")
    for i, s in enumerate(data):
        bad = [k for k in s if k.lower() in FORBIDDEN_GSM_HARD_KEYS]
        if bad:
            die(f"label field {bad} present at row {i} of the GSM-Hard "
                "questions file; refusing to proceed on a label-free "
                "extraction path.")
    got = meta.get("questions_sha256")
    if expect_sha and got != expect_sha:
        die(f"GSM-Hard questions_sha256 {got} != the frozen P3 digest "
            f"{expect_sha}. This must be the SAME 300-question P3 file "
            "used by the frozen bare/chat GSM-Hard experiments; a "
            "different file would break every downstream comparison.")
    ids = [s["sample_id"] for s in data]
    if len(set(ids)) != len(ids):
        die("duplicate sample_id in the GSM-Hard questions file.")
    return meta, data


def render_bare_prompt(task: str, question: str, templates: dict) -> str:
    return templates["neutral"].format(context=question)


def strip_leading_bos(vc, text: str) -> str:
    """Byte-identical logic (reimplemented, not imported -- see module
    docstring) to the strip_leading_bos() function duplicated across
    get_answer_gsm8k_chat_sweep.py, get_answer_math_chat_sweep.py, and
    get_answer_gsm_hard_chat_sweep.py: vc.regenerate/tokenizer calls with
    add_special_tokens=True, so an un-stripped chat-templated string that
    already serialized a BOS string yields two leading BOS ids."""
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
# Tokenization + fail-closed checks
# --------------------------------------------------------------------------

def tokenize_single(vc, text: str):
    """Tokenize ONE prompt with add_special_tokens=True (the standard HF
    default and the same convention every regenerate()/chat-sweep call in
    this repo uses for its own internal tokenization checks). Returns the
    HF BatchEncoding on CPU (moved to device by the caller)."""
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
# Pairing digest -- computed identically by BOTH conditions of the same
# (model, task), so a caller can assert chat and bare will operate on the
# identical ordered question list without either process needing to see the
# other's in-memory state. The digest covers question CONTENT and ORDER
# (and, for GSM-Hard, sample_id) but NOT which condition is being run, so it
# is condition-independent by construction.
# --------------------------------------------------------------------------

def ordered_identity_list(task: str, samples: list) -> list[str]:
    if task == "gsm_hard":
        return [f"{s['sample_id']}\x1f{s['question']}" for s in samples]
    return [s["question"] for s in samples]


def sha256_of_list(items: list[str]) -> str:
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
    task = args.task
    condition = args.condition
    mcfg = MODELS[model_key]
    size = mcfg["size"]
    model_dir = args.model_dir or mcfg["model_dir_default"]

    base_dir = args.base_dir
    gsm8k_file = args.gsm8k_file or os.path.join(
        base_dir, "benchmark", "gsm8k_test_sample.json")
    math_file = args.math_file or os.path.join(
        base_dir, "benchmark", "math_test_sample.json")
    gsm_hard_file = args.gsm_hard_file or os.path.join(
        base_dir, "benchmark", "gsm_hard_p3_questions.json")
    out_dir = args.out_dir or os.path.join(
        base_dir, "hidden_states", model_key, task)

    # ---- Load samples + render prompts (no model needed for this part) ----
    if task == "gsm8k":
        samples = load_gsm8k_samples(gsm8k_file)
        templates = build_gsm8k_default_suite(cot=False, wording="plain")
        question_key = "question"
    elif task == "math":
        samples = load_math_samples(math_file, N_EXPECT)
        templates = build_math_suite(cot=False)
        question_key = "question"
    elif task == "gsm_hard":
        qmeta, samples = load_gsm_hard_samples(
            gsm_hard_file, GSM_HARD_QUESTIONS_SHA256)
        templates = build_gsm8k_default_suite(cot=False, wording="plain")
        question_key = "question"
    else:
        die(f"unknown task {task}")

    n = len(samples)
    if n != N_EXPECT:
        die(f"{task}: loaded {n} samples, expected exactly {N_EXPECT}. "
            "Refusing to extract on a wrong-size sample.")

    pairing_ids = ordered_identity_list(task, samples)
    pairing_digest = sha256_of_list(pairing_ids)
    print(f"[pairing] task={task} ordered_sample_identity_sha256="
          f"{pairing_digest}")

    if args.print_pairing_digest_only:
        print(pairing_digest)
        return

    if args.expect_pairing_digest:
        if pairing_digest != args.expect_pairing_digest:
            die(f"ordered_sample_identity_sha256 {pairing_digest} != "
                f"--expect_pairing_digest {args.expect_pairing_digest}. "
                "The chat and bare conditions for this (model,task) must "
                "operate on the IDENTICAL ordered question list; refusing "
                "to proceed with a mismatched pairing.")

    questions_file_sha256 = None
    for fpath, tk in ((gsm8k_file, "gsm8k"), (math_file, "math"),
                      (gsm_hard_file, "gsm_hard")):
        if tk == task:
            questions_file_sha256 = hashlib.sha256(
                open(fpath, "rb").read()).hexdigest()

    # ---- Overwrite preflight for THIS CELL's own output files only, ----
    # ---- BEFORE model load. The whole-model, all-6-file check is the ----
    # ---- launcher's job (run once, before cell 1, covering .h5 AND ----
    # ---- .manifest.json) -- checking all 6 here would make cell 2 refuse
    # ---- to run merely because cell 1 already wrote its own files.
    out_path = os.path.join(out_dir, f"{condition}_{size}.h5")
    meta_path = os.path.join(out_dir, f"{condition}_{size}.manifest.json")
    existing = [p for p in (out_path, meta_path) if os.path.exists(p)]
    if existing:
        die(f"refusing to run model={model_key} task={task} "
            f"condition={condition}: target file(s) already exist:\n  " +
            "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    print(f"model={model_key} task={task} condition={condition} "
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

    # ---- Model-shape lock: a wrong --model_dir (wrong architecture/size
    # for the requested --model key) must not silently write into this
    # model's formal output path. Assert BEFORE any prompt rendering or
    # forward pass.
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
        render_bare_prompt(task, s[question_key], templates) for s in samples]
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
    n_finite_checks = 0
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
            # Read out at EVERY sample, not just the first -- the task
            # requires this rather than a first-sample-only spot check.
            assert_final_token(model_key, condition, last_id, last_text, i)
            final_tok_ids.append(last_id)
            final_tok_texts.append(last_text)

            enc = {k: v.to(device) for k, v in enc.items()}
            out = vc.model(**enc, return_dict=True,
                           output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states  # tuple, len = n_decoder+1
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
                n_finite_checks += 1

            if (i + 1) % 50 == 0 or i == n - 1:
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

    # ---- Atomic write: build in memory, verify, THEN write to a temp path
    # ---- and os.replace() into place. n_samples_done is also written into
    # ---- meta so any reader can confirm ==300 rather than trust file
    # ---- existence alone. ----
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
            if task == "gsm_hard":
                f.create_dataset(
                    "sample_id",
                    data=np.array([str(s["sample_id"]) for s in samples],
                                 dtype=h5py.string_dtype(encoding="utf-8")))

            meta = f.attrs
            meta["script_version"] = SCRIPT_VERSION
            meta["git_commit"] = git_commit()
            meta["model"] = model_key
            meta["model_dir"] = model_dir
            meta["model_revision"] = (
                "unpinned, uses default HF revision")
            meta["task"] = task
            meta["condition"] = condition
            meta["n_samples"] = n
            meta["n_samples_done"] = n
            meta["hidden_state_shape"] = json.dumps(
                [n, n_layers_out, int(vc.model.config.hidden_size)])
            meta["hidden_state_dtype"] = "float16"
            meta["n_layers"] = n_layers_out
            meta["n_decoder_layers"] = n_decoder
            meta["hidden_size"] = int(vc.model.config.hidden_size)
            meta["questions_sha256"] = questions_file_sha256 or ""
            meta["ordered_sample_identity_sha256"] = pairing_digest
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
        "task": task,
        "condition": condition,
        "n_samples": n,
        "n_samples_done": n,
        "hidden_state_shape": [n, n_layers_out,
                               int(vc.model.config.hidden_size)],
        "hidden_state_dtype": "float16",
        "questions_sha256": questions_file_sha256,
        "ordered_sample_identity_sha256": pairing_digest,
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
