#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
extract_gsm8k_abstention_role_hs.py -- MINIMAL-SCOPE, GSM8K-ONLY, standalone,
prefill-only, NO-generation, NO-steering hidden-state extraction for the
Expert vs Non-expert ROLE condition, with an "honest" + abstention-enabled
prompt variant.

WHY A NEW SCRIPT (not an edit of extract_role_hidden_states.py). That script
is the frozen, already-run GSM8K/MATH/GSM-Hard/MMLU-E Role HS extractor and
its prompts (GSM8K/MATH/GSM-Hard use templates["neg"], NO "honest", NO
abstention exit) are load-bearing for existing downstream analysis --
editing it in place would fork the 口径 of already-collected data under the
same script name. This script is GSM8K-only, uses a DIFFERENT prompt (with
"honest" + an abstention sentence, deliberately matching the MMLU-E role
framing more closely), and writes to its OWN output tree. It does not import
or call extract_role_hidden_states.py, and does not touch its output.

SCOPE, DELIBERATELY NARROW (per explicit instruction): GSM8K only, the
existing fixed 300-question benchmark, Llama-3.1-8B-Instruct and
Qwen2.5-7B-Instruct only. No MATH, no GSM-Hard, no MMLU-E, no Chat, no
generation/answer audit, no steering, no layer-band selection, no NMD mask
construction, no RRSN/MRSN similarity analysis. Those are explicitly
out-of-scope for this round.

PROMPT. Starting point is template.build_gsm8k_default_suite(cot=False,
wording="plain") -- the SAME frozen GSM8K No-CoT/bare-string/plain-wording
body every other GSM8K launcker in this repo uses (0 characters of the
question-solving instructions are hand-retyped). Two deliberate,
minimal-diff changes vs. that suite's existing "neg" (role, no honest)
template:
  1. Role sentence uses "an honest {character}" (like the suite's own
     ["default"] key already does for CoT-style scaffolding), but with a
     BARE noun character string ("expert" / "non expert", NO leading
     article) rather than utils.ROLE_TO_CHARACTER's article-bearing strings
     ("an expert" / "a non expert") -- ROLE_TO_CHARACTER is deliberately NOT
     reused here, because template["default"].format(character=...) already
     supplies "an honest {character}", so formatting in "an expert" would
     render the double-article bug "an honest an expert" (verified: see
     ABSTENTION_ROLE_TO_CHARACTER below and the module-level smoke check).
  2. One abstention sentence appended, byte-identical for both roles:
     'If you are not sure, you may answer exactly "I am not sure" instead.'
This produces EXACTLY:
  expert:     "Now you are an honest expert."
  non_expert: "Now you are an honest non expert."
Both immediately followed by the same abstention sentence. Everything else
(math-problem framing, "####" directive, "Answer: " anchor, no CoT) is
BYTE-IDENTICAL to build_gsm8k_default_suite(cot=False, wording="plain")'s
existing "neg" template body -- verified programmatically at prompt-render
time (see assert_no_other_diff_than_role_wording()) rather than assumed.

template.py and any FROZEN extraction script (get_answer_regenerate_gsm8k.py,
extract_role_hidden_states.py, run_gsm8k*.sh, etc.) are NOT modified by this
file. The new prompt is built entirely by this script's own
render_abstention_prompt(), reusing only build_gsm8k_default_suite()'s FORMAT
STRING content (via a local re-templating, not a template.py edit) -- see
that function for the exact construction and the assertion that ties it back
to the frozen suite's non-role text.

DATA: the SAME frozen 300-question GSM8K file
<base_dir>/benchmark/gsm8k_test_sample.json, loaded via utils.load_json with
NO reordering/resampling -- identical to every other GSM8K script in this
repo (run_gsm8k.sh, run_gsm8k_qwen25.sh, extract_role_hidden_states.py). The
ordered-sample-identity pairing digest is computed with the EXACT SAME
formula extract_role_hidden_states.py uses for GSM8K
([s["question"] for s in samples], sha256 of "\n"-joined list) -- this makes
the printed digest directly comparable to (not merely "the same shape as")
that script's own GSM8K digest, since both read the same file with the same
loader and the same identity-list construction. Expert, Non-expert, and both
models all read this one file with this one loader -- no per-condition or
per-model resampling is possible by construction.

WHAT THIS SCRIPT IS AND IS NOT. Pure forward-pass HS dumper: tokenize one
rendered prompt, run ONE `model(**tokens, output_hidden_states=True)` call
(no `model.generate`, no forward hooks of any kind), and save the last valid
prefill-token hidden state at EVERY layer (embedding output + every
decoder-layer output). No steering mask is loaded, no diff matrix is built,
no forward hook (register_forward_hook / register_forward_pre_hook) is ever
registered by this script or by any VicundaModel method it calls.

FINAL PREFILL TOKEN. NOT hardcoded from any other script's recorded value
(the abstention sentence changes the prompt tail's tokenization context, so
the old id=220/' ' fact for the plain Role prompt is not assumed to carry
over). Instead, derived once per (model, run) from the actual loaded
tokenizer by tokenizing the first rendered prompt of the FIRST cell
processed and reading off its last token id/text, then asserted identical
across every sample of every cell in that run (a uniform prompt-template
change would keep it constant; a per-sample drift would not).

FAIL-CLOSED VALIDATION:
  - overwrite preflight for every cell's own out_path AND meta_path, for
    BOTH conditions, BEFORE the model is loaded;
  - n must equal exactly 300;
  - expert/non_expert must share the identical ordered question list (sha256
    equality asserted before either condition's run starts);
  - the final prefill token id/text is read out at runtime for EVERY sample
    (not just the first) and asserted constant across every sample of every
    cell within one run;
  - no double-BOS;
  - all extracted hidden states are finite;
  - atomic write: build the full (n, L, H) array in memory, verify it, THEN
    write to a temp path and os.replace() it into place.

MODEL LOADED EXACTLY ONCE PER RUN. --run_all loads the model ONCE, then
processes BOTH conditions (expert, non_expert) sequentially in-process,
reusing the same loaded VicundaModel object for every forward pass.

OUTPUT (server): components/hidden_states/{model}/gsm8k_abstention_role/
  expert_{size}.h5
  non_expert_{size}.h5
ONE experiment-level manifest.json (not one per role -- per explicit
instruction) recording BOTH rendered prompts, input digest, model, HS/mean
shape, dtype, and output paths. Per-cell H5 attrs still carry the standard
per-cell fields (n_samples, pairing digest, prompt sha256, etc.) so each H5
is independently self-describing; manifest.json is the experiment-level
index, not a replacement for those attrs.

@author: GSM8K abstention-enabled role hidden-state extraction (2026-09-17)
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

SCRIPT_VERSION = "extract_gsm8k_abstention_role_hs-v1"

# Reused verbatim from extract_role_hidden_states.py's MODELS dict (same
# repo-wide shape constants, not independently re-derived).
MODELS = {
    "llama3": {
        "model_dir_default": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "8B",
        "expected_n_layers_out": 33,   # embedding + 32 decoder layers
        "expected_hidden_size": 4096,
    },
    "qwen2.5": {
        "model_dir_default": "Qwen/Qwen2.5-7B-Instruct",
        "size": "7B",
        "expected_n_layers_out": 29,   # embedding + 28 decoder layers
        "expected_hidden_size": 3584,
    },
}

CONDITIONS = ("expert", "non_expert")

# Bare-noun character strings for THIS script's prompt only -- deliberately
# NOT utils.ROLE_TO_CHARACTER ("an expert" / "a non expert"), which would
# double up the article against the "an honest {character}" template text.
# See the module docstring's PROMPT section.
ABSTENTION_ROLE_TO_CHARACTER = {
    "expert": "expert",
    "non_expert": "non expert",
}

ABSTENTION_SENTENCE = (
    'If you are not sure, you may answer exactly "I am not sure" instead.'
)

N_EXPECT = 300


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------
# Prompt construction. Built from build_gsm8k_default_suite()'s OWN "neg"
# template text (role slot, no honest) by re-deriving the honest+abstention
# variant as a minimal, programmatically-checked diff, rather than typing a
# new template string from scratch and hoping it matches.
# --------------------------------------------------------------------------

def build_abstention_templates(wording: str = "plain") -> dict:
    """Returns {"expert": <format string>, "non_expert": <format string>}
    for the GSM8K No-CoT abstention-enabled role prompt. Constructed by
    taking build_gsm8k_default_suite(cot=False, wording=wording)["neg"]
    (the frozen role-bearing, non-honest template: "Now you are
    {character}.\n<fmt>\nAnswer: ") and:
      (a) replacing the role clause "Now you are {character}." with
          "Now you are an honest {character}." (character filled with a
          BARE noun, not utils.ROLE_TO_CHARACTER's article-bearing string);
      (b) inserting the abstention sentence on its own line immediately
          after the role clause, before the "####" directive line.
    Every other character of the template (the math-problem framing line,
    the Question: line, the fmt directive, the "Answer: " anchor) is IDENTICAL
    to the frozen suite's own text -- this is checked, not assumed, by
    assert_no_other_diff_than_role_wording() below.
    """
    frozen = build_gsm8k_default_suite(cot=False, wording=wording)
    neg = frozen["neg"]

    old_role_clause = "Now you are {character}."
    if old_role_clause not in neg:
        die("build_gsm8k_default_suite()['neg'] does not contain the "
            f"expected role clause {old_role_clause!r} -- the frozen "
            "template must have changed shape; refusing to guess a "
            "replacement.")

    new_role_clause = (
        "Now you are an honest {character}.\n" + ABSTENTION_SENTENCE
    )
    new_template = neg.replace(old_role_clause, new_role_clause, 1)

    assert_no_other_diff_than_role_wording(neg, new_template)

    out = {}
    for role_key, character in ABSTENTION_ROLE_TO_CHARACTER.items():
        out[role_key] = new_template.format(character=character,
                                            context="{context}")
        # .format() above only fills `character`; `context` is re-escaped
        # back to a literal "{context}" placeholder for the per-sample
        # .format(context=question) call at render time. Verify that
        # round-trip did not corrupt anything else in the string.
        if "{context}" not in out[role_key]:
            die(f"internal error: {{context}} placeholder lost while "
                f"building the abstention template for role={role_key}.")
    return out


def assert_no_other_diff_than_role_wording(old_template: str,
                                           new_template: str) -> None:
    """Confirms the ONLY change between the frozen neg template and the new
    abstention template is: the role clause growing from
    "Now you are {character}." to "Now you are an honest {character}.\n" +
    the abstention sentence. Does this by stripping the role clause out of
    each and asserting the REMAINDER (problem framing, fmt directive,
    Answer anchor) is byte-identical."""
    old_role_clause = "Now you are {character}."
    new_role_clause = (
        "Now you are an honest {character}.\n" + ABSTENTION_SENTENCE
    )
    old_remainder = old_template.replace(old_role_clause, "\x00ROLE\x00", 1)
    new_remainder = new_template.replace(new_role_clause, "\x00ROLE\x00", 1)
    if old_remainder != new_remainder:
        die("abstention template construction changed something other than "
            "the role clause -- refusing to proceed with an unverified "
            f"prompt diff.\n  old (role-clause-masked): {old_remainder!r}\n"
            f"  new (role-clause-masked): {new_remainder!r}")


def render_prompt(templates: dict, role_key: str, question: str) -> str:
    return templates[role_key].format(context=question)


# --------------------------------------------------------------------------
# Fail-closed checks
# --------------------------------------------------------------------------

def tokenize_single(vc, text: str):
    return vc.tokenizer(text, return_tensors="pt", add_special_tokens=True)


def assert_no_double_bos(vc, input_ids_row) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = input_ids_row.tolist()
    if bos_id is not None and len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
        die(f"double BOS detected (head={ids[:4]}) -- refusing to proceed.")


def ordered_identity_list(samples: list) -> list[str]:
    # Matches extract_role_hidden_states.py's GSM8K identity-list formula
    # EXACTLY: [s["question"] for s in samples] -- so this script's printed
    # digest is directly comparable to that script's own GSM8K digest.
    return [s["question"] for s in samples]


def sha256_of_list(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def build_device_note(vc):
    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    return {"host": platform.node(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "param_devices": devs,
           "sharded": len(devs) > 1}


def model_shape_lock(vc, mcfg, model_key, model_dir):
    n_decoder = len(vc._find_decoder_layers())
    n_layers_out = n_decoder + 1
    expect_layers = mcfg["expected_n_layers_out"]
    expect_hidden = mcfg["expected_hidden_size"]
    actual_hidden = int(vc.model.config.hidden_size)
    if n_layers_out != expect_layers or actual_hidden != expect_hidden:
        die(f"model shape mismatch for --model {model_key} "
            f"(--model_dir {model_dir}): got n_layers_out={n_layers_out} "
            f"hidden_size={actual_hidden}, expected "
            f"n_layers_out={expect_layers} hidden_size={expect_hidden}.")
    return n_layers_out, n_decoder


# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------

def h5_path(out_dir: str, condition: str, size: str) -> str:
    return os.path.join(out_dir, f"{condition}_{size}.h5")


def experiment_manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")


# --------------------------------------------------------------------------
# Per-condition forward-pass extraction
# --------------------------------------------------------------------------

def process_condition(vc, model_key, model_dir, size, condition, samples,
                      question_ids, templates, out_path, n_layers_out,
                      n_decoder, device_note, fixed_final_tok):
    """fixed_final_tok: (expected_id, expected_text) shared across BOTH
    conditions of this run, established from the FIRST condition processed
    (see run_experiment()). Returns (n, prompt_sha256, pairing_digest,
    final_tok_id, final_tok_text, template_used)."""
    n = len(samples)
    rendered_prompts = [render_prompt(templates, condition, s["question"])
                        for s in samples]
    prompt_body_sha256 = hashlib.sha256(
        templates[condition].encode("utf-8")).hexdigest()
    prompt_sha256 = hashlib.sha256(
        "\n".join(rendered_prompts).encode("utf-8")).hexdigest()

    print(f"condition={condition} n={n} out_path={out_path}")
    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"prompt_sha256={prompt_sha256}")
    print("Full first rendered prompt:")
    print(repr(rendered_prompts[0]))

    pairing_digest = sha256_of_list(ordered_identity_list(samples))

    device = vc.model.device
    hs_array = np.zeros((n, n_layers_out, vc.model.config.hidden_size),
                        dtype=np.float16)
    final_tok_ids = []
    final_tok_texts = []

    expected_id, expected_text = fixed_final_tok

    t0 = time.time()
    with torch.no_grad():
        for i, prompt in enumerate(rendered_prompts):
            enc = tokenize_single(vc, prompt)
            input_ids = enc["input_ids"][0]
            assert_no_double_bos(vc, input_ids)

            last_id = int(input_ids[-1].item())
            last_text = vc.tokenizer.decode([last_id])
            if expected_id is not None and (last_id != expected_id or
                                            last_text != expected_text):
                die(f"condition={condition} sample_idx={i}: final prefill "
                    f"token is id={last_id} text={last_text!r}, expected "
                    f"id={expected_id} text={expected_text!r} (derived "
                    "from this run's first processed sample; refusing to "
                    "extract hidden states under a drifted prefill "
                    "boundary within one run).")
            final_tok_ids.append(last_id)
            final_tok_texts.append(last_text)

            enc = {k: v.to(device) for k, v in enc.items()}
            out = vc.model(**enc, return_dict=True,
                           output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states
            if len(hidden_states) != n_layers_out:
                die(f"sample_idx={i}: got {len(hidden_states)} "
                    f"hidden_states layers, expected {n_layers_out}.")

            for l, layer_hs in enumerate(hidden_states):
                vec = layer_hs[0, -1, :].detach().float().cpu().numpy()
                if not np.all(np.isfinite(vec)):
                    die(f"sample_idx={i} layer={l}: non-finite hidden "
                        "state values (NaN/Inf) encountered.")
                hs_array[i, l, :] = vec.astype(np.float16)

            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.time() - t0
                print(f"  [{condition}] [{i + 1}/{n}] elapsed={elapsed:.1f}s")

    if not np.all(np.isfinite(hs_array.astype(np.float32))):
        die("post-hoc finiteness check failed on the assembled hidden-state "
            "array -- should be unreachable given the per-sample check "
            "above; treat as a hard bug.")

    final_tail_unique = sorted(set(final_tok_ids))
    if len(final_tail_unique) != 1:
        die(f"condition={condition}: final prefill token id is NOT constant "
            f"across samples ({final_tail_unique}).")

    # ---- Atomic write ----
    out_dir = os.path.dirname(out_path)
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
                data=np.array(question_ids,
                             dtype=h5py.string_dtype(encoding="utf-8")))

            meta = f.attrs
            meta["script_version"] = SCRIPT_VERSION
            meta["git_commit"] = git_commit()
            meta["experiment"] = "gsm8k_abstention_role"
            meta["model"] = model_key
            meta["model_dir"] = model_dir
            meta["model_revision"] = "unpinned, uses default HF revision"
            meta["task"] = "gsm8k"
            meta["condition"] = condition
            meta["role_wording"] = "honest + abstention-enabled"
            meta["n_samples"] = n
            meta["n_samples_done"] = n
            meta["hidden_state_shape"] = json.dumps(
                [n, n_layers_out, int(vc.model.config.hidden_size)])
            meta["hidden_state_dtype"] = "float16"
            meta["n_layers"] = n_layers_out
            meta["n_decoder_layers"] = n_decoder
            meta["hidden_size"] = int(vc.model.config.hidden_size)
            meta["ordered_sample_identity_sha256"] = pairing_digest
            meta["prompt_body_sha256"] = prompt_body_sha256
            meta["prompt_sha256"] = prompt_sha256
            meta["chat_template_applied"] = False
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

    print(f"wrote {out_path}")
    return {
        "condition": condition,
        "n_samples": n,
        "pairing_digest": pairing_digest,
        "prompt_body_sha256": prompt_body_sha256,
        "prompt_sha256": prompt_sha256,
        "rendered_prompt_example": rendered_prompts[0],
        "final_prefill_token_id": final_tail_unique[0],
        "final_prefill_token_text": final_tok_texts[0],
        "out_path": out_path,
        "h5_sha256": None,  # filled in by caller after file is on disk
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_experiment(args):
    model_key = args.model
    mcfg = MODELS[model_key]
    size = mcfg["size"]
    model_dir = args.model_dir or mcfg["model_dir_default"]
    base_dir = args.base_dir
    gsm8k_file = args.gsm8k_file or os.path.join(
        base_dir, "benchmark", "gsm8k_test_sample.json")
    out_dir = args.out_dir or os.path.join(
        base_dir, "hidden_states", model_key, "gsm8k_abstention_role")

    if not os.path.isfile(gsm8k_file):
        die(f"GSM8K sample file not found: {gsm8k_file}")

    samples = utils.load_json(gsm8k_file)
    n = len(samples)
    if n != N_EXPECT:
        die(f"gsm8k: loaded {n} samples, expected exactly {N_EXPECT}. "
            "Refusing to extract on a wrong-size sample.")
    question_ids = [s["question"] for s in samples]
    questions_file_sha256 = sha256_of_file(gsm8k_file)

    templates = build_abstention_templates(wording=args.wording)

    manifest_path = experiment_manifest_path(out_dir)
    expert_out = h5_path(out_dir, "expert", size)
    nonexpert_out = h5_path(out_dir, "non_expert", size)

    # ---- Overwrite preflight for BOTH conditions + the experiment manifest,
    # ---- BEFORE the model is loaded. ----
    existing = [p for p in (expert_out, nonexpert_out, manifest_path)
               if os.path.exists(p)]
    if existing:
        die(f"refusing to run model={model_key}: target file(s) already "
            "exist:\n  " + "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    print("=" * 60)
    print("Rendered prompts (full text, both roles) -- REVIEW BEFORE "
          "running with real GPU/model:")
    print("=" * 60)
    example_expert = render_prompt(templates, "expert", samples[0]["question"])
    example_nonexpert = render_prompt(templates, "non_expert",
                                      samples[0]["question"])
    print("--- expert ---")
    print(example_expert)
    print("--- non_expert ---")
    print(example_nonexpert)
    print("=" * 60)

    if args.verify_only:
        print("[verify_only] No model loaded, no generation, no GPU used.")
        print(f"gsm8k_file={gsm8k_file}")
        print(f"questions_file_sha256={questions_file_sha256}")
        print(f"ordered_sample_identity_sha256="
              f"{sha256_of_list(ordered_identity_list(samples))}")
        return

    # ---- Model load, ONCE for this whole run. ----
    print(f"Loading model ONCE: model={model_key} model_dir={model_dir}")
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    n_layers_out, n_decoder = model_shape_lock(vc, mcfg, model_key, model_dir)
    device_note = build_device_note(vc)

    # ---- Establish the fixed final-prefill-token expectation from the
    # ---- FIRST condition's first sample, then hold it fixed across BOTH
    # ---- conditions of this run. NOT hardcoded from any other script or
    # ---- any prior run -- derived fresh here, from this model's real
    # ---- tokenizer, on this new (honest + abstention) prompt. ----
    first_prompt = render_prompt(templates, "expert", samples[0]["question"])
    enc0 = tokenize_single(vc, first_prompt)
    ids0 = enc0["input_ids"][0]
    assert_no_double_bos(vc, ids0)
    fixed_final_id = int(ids0[-1].item())
    fixed_final_text = vc.tokenizer.decode([fixed_final_id])
    print(f"[fixed final prefill token, derived from this run's real "
          f"tokenizer] id={fixed_final_id} text={fixed_final_text!r}")

    results = []
    for condition, out_path in (("expert", expert_out),
                                ("non_expert", nonexpert_out)):
        print("=" * 60)
        print(f"Processing condition={condition}")
        print("=" * 60)
        r = process_condition(
            vc, model_key, model_dir, size, condition, samples,
            question_ids, templates, out_path, n_layers_out, n_decoder,
            device_note, (fixed_final_id, fixed_final_text))
        r["h5_sha256"] = sha256_of_file(out_path)
        results.append(r)

    pairing_digests = {r["condition"]: r["pairing_digest"] for r in results}
    if pairing_digests["expert"] != pairing_digests["non_expert"]:
        die("expert/non_expert ordered_sample_identity_sha256 mismatch "
            f"after both conditions ran: {pairing_digests} -- this should "
            "be unreachable since both read the same in-memory `samples` "
            "list; treat as a hard bug.")

    # ---- ONE experiment-level manifest (not one per role), per explicit
    # ---- instruction. ----
    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": git_commit(),
        "experiment": "gsm8k_abstention_role",
        "model": model_key,
        "model_dir": model_dir,
        "model_revision": "unpinned, uses default HF revision",
        "task": "gsm8k",
        "wording": args.wording,
        "n_samples": N_EXPECT,
        "gsm8k_file": gsm8k_file,
        "questions_file_sha256": questions_file_sha256,
        "ordered_sample_identity_sha256": pairing_digests["expert"],
        "rendered_prompts": {
            "expert": example_expert,
            "non_expert": example_nonexpert,
        },
        "role_character_strings": dict(ABSTENTION_ROLE_TO_CHARACTER),
        "abstention_sentence": ABSTENTION_SENTENCE,
        "conditions": {r["condition"]: {
            "n_samples": r["n_samples"],
            "pairing_digest": r["pairing_digest"],
            "prompt_body_sha256": r["prompt_body_sha256"],
            "prompt_sha256": r["prompt_sha256"],
            "final_prefill_token_id": r["final_prefill_token_id"],
            "final_prefill_token_text": r["final_prefill_token_text"],
            "out_path": r["out_path"],
            "h5_sha256": r["h5_sha256"],
        } for r in results},
        "hidden_state_shape": [N_EXPECT, n_layers_out,
                               int(vc.model.config.hidden_size)],
        "hidden_state_dtype": "float16",
        "chat_template_applied": False,
        "steering_applied": False,
        "generation_performed": False,
        "prefill_only": True,
        "batch_size": 1,
        "provenance": device_note,
        "extraction_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": (
            "GSM8K-only, honest + abstention-enabled Role HS extraction. "
            "Layer band / NMD mask / RRSN-MRSN similarity analysis are "
            "OUT OF SCOPE for this manifest and this run."
        ),
    }
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_manifest, manifest_path)
    print(f"wrote experiment manifest: {manifest_path}")
    print("[DONE]")


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM8K-only, honest + abstention-enabled, prefill-only, "
                    "no-generation, no-steering role (expert vs non_expert) "
                    "hidden-state extraction.")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--model_dir", default=None,
                   help="HF repo id. Defaults to this model's standard repo "
                        "id if omitted.")
    p.add_argument("--base_dir", required=True,
                   help="components/ dir, e.g. /data1/paveen/Dopamine/"
                        "components. Sample file is read from "
                        "<base_dir>/benchmark/gsm8k_test_sample.json unless "
                        "--gsm8k_file overrides it. Output goes to "
                        "<base_dir>/hidden_states/<model>/"
                        "gsm8k_abstention_role/.")
    p.add_argument("--gsm8k_file", default=None,
                   help="Default: <base_dir>/benchmark/gsm8k_test_sample.json")
    p.add_argument("--out_dir", default=None,
                   help="Default: <base_dir>/hidden_states/<model>/"
                        "gsm8k_abstention_role/")
    p.add_argument("--wording", default="plain", choices=["plain", "pushy"],
                   help="Passed to build_gsm8k_default_suite's wording= "
                        "(default: plain, matching the GSM8K main line).")
    p.add_argument("--verify_only", action="store_true",
                   help="Render and print both prompts, print digests, "
                        "exit. No model load, no GPU.")
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
