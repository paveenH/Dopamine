#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
extract_gsm8k_role_hs.py -- standalone, prefill-only, NO-generation,
NO-steering hidden-state extraction for TWO GSM8K Role directions, over the
FULL GSM8K test set (n=1319):

  gsm8k_role             -- no abstention exit, expert / non_expert.
  gsm8k_role_abstention  -- v2 abstention-enabled prompt, expert / non_expert.

Deliberately NOT named "RRSN" or any variant of it -- per CLAUDE.md, both
prior RRSN/ARRSN lines were retired 2026-09-18 and the name is reserved
unless explicitly reinstated. These are two independent, plainly-named GSM8K
Role directions.

STRUCTURAL TEMPLATE. This script mirrors extract_role_hidden_states.py's
overwrite-guard pattern, atomic H5 writing, manifest schema, model-shape
assertions, pairing-digest logic, and fail-closed validation ordering
(itself a mirror of extract_interface_hidden_states.py's conventions). It is
a SEPARATE script (not an edit of extract_role_hidden_states.py) because:
  (a) the sample set here is the full 1319-question benchmark/
      gsm8k_1319_full.json (built by build_gsm8k_1319_full.py), not the
      frozen 300-question benchmark/gsm8k_test_sample.json every existing
      GSM8K role/interface HS extractor uses;
  (b) it adds a SECOND prompt family (the v2 abstention prompt) that no
      existing extractor renders;
  (c) it is scoped to GSM8K ONLY -- no MATH, no GSM-Hard, no MMLU-E.
extract_role_hidden_states.py and extract_interface_hidden_states*.py are
NOT imported, edited, or otherwise touched by this script.

PROMPT REUSE, NOT PROMPT RETYPING -- two prompt families, each copied from an
exact existing code path:

  (1) gsm8k_role (NO abstention exit): template.build_gsm8k_default_suite(
      cot=False, wording="plain")["neg"].format(character=
      utils.ROLE_TO_CHARACTER[role], context=question). This is the EXACT
      routing get_answer_regenerate_gsm8k.py (line 66) and
      extract_role_hidden_states.py's own render_reasoning_prompt() use for
      GSM8K non-neutral roles -- the SAME "Bare" behavior-experiment prompt
      family this repo's frozen GSM8K dose-response/role work already runs
      (e.g. the mdf_0 answer_mdf_gsm8k/ tree). No "honest" framing (GSM8K has
      no E-option); templates["neg"] is used directly, bypassing
      construct_prompt's default(+honest) branch, exactly as
      extract_role_hidden_states.py's own module docstring documents.

  (2) gsm8k_role_abstention (v2 abstention-enabled prompt): imports
      PROMPT_TEMPLATE, ROLE_EXPERT ("an honest expert"), ROLE_NON_EXPERT
      ("an honest non expert") DIRECTLY from
      get_answer_gsm8k_role_abstention_v2.py (the script that already ran
      this exact prompt as a behavioral, no-steering pilot on the 300-
      question set -- see CLAUDE.md's "GSM8K role-abstention prompt v2"
      entry). Not re-typed from memory: `from
      get_answer_gsm8k_role_abstention_v2 import PROMPT_TEMPLATE,
      ROLE_EXPERT, ROLE_NON_EXPERT` and the SAME `PROMPT_TEMPLATE.format(
      question=question, role=role)` call that script's own build_prompt()
      makes.

Both families end their bare-string prefill in a literal trailing space
("...Answer: "), which both tokenizers used here (Llama-3.1, and, if this
script is ever pointed at Qwen2.5, Qwen2.5) render as a single trailing ' '
token (id=220 for Llama-3.1) -- the SAME fixed global expectation
extract_role_hidden_states.py hardcodes (EXPECTED_FINAL_TOKEN_ID/TEXT), reused
verbatim here rather than re-derived, since both prompt families share the
identical tail.

SAMPLES ARE REUSED VERBATIM, NEVER REGENERATED OR RE-SELECTED. Both gsm8k_role
and gsm8k_role_abstention read the SAME frozen
<base_dir>/benchmark/gsm8k_1319_full.json (built once by
build_gsm8k_1319_full.py; NOT built or modified by this script), in that
file's own stored order (the full GSM8K test set's original index order,
unfiltered). All 1319 rows are extracted unconditionally -- no filtering by
correctness, marker presence, or abstention behavior of any kind (there is no
generation happening here to filter on in the first place). The file's
"in_300_sample" field is read through ONLY as a provenance/audit annotation
carried into each cell's H5 (per-sample dataset) for the downstream stability
check the task requires; it is NEVER used to reorder, subset, or skip rows
during extraction.

NO STEERING, ANYWHERE. No mask is loaded, no diff matrix is built, no forward
hook (register_forward_hook / register_forward_pre_hook) is ever registered
by this script or by any VicundaModel method it calls (a raw
`vc.model(**enc, output_hidden_states=True)` call registers no hooks --
hooks are registered only inside the regenerate()/_apply_diff_hooks family,
none of which this script calls). No `model.generate` call anywhere, so
there is no decode step at all.

ONE FORWARD PASS PER SAMPLE, bs=1, PREFILL ONLY.
`vc.model(**tokens, output_hidden_states=True, use_cache=False)`.
`outputs.hidden_states` is a tuple of length num_decoder_layers+1: index 0 is
the embedding output, indices 1..N are decoder-layer outputs (HF
hidden_states convention). This script stores ALL of them (33 for Llama3.1-
8B, giving (n,33,4096)) -- the expected_n_layers_out/expected_hidden_size
values are reused verbatim from extract_role_hidden_states.py's MODELS dict.

LAST VALID PREFILL TOKEN. bs=1, no padding is ever inserted, so the last
token of input_ids[0] is unambiguously the last real prompt token; its hidden
state at every layer is hidden_states[l][0, -1, :].

EXPERT/NON-EXPERT PAIRING. Both conditions of one (model, gsm8k_task) cell
must share the identical ordered question list -- enforced the same way
extract_role_hidden_states.py enforces it: a condition-independent ordered-
sample-identity sha256, computable via --print_pairing_digest_only without
loading any model, and asserted before either condition's extraction
happens. Since both conditions read the SAME gsm8k_1319_full.json file, this
digest is expected to be IDENTICAL across gsm8k_role and gsm8k_role_
abstention too (both prompt families are built from the same underlying
1319-question list, same order) -- this cross-task equality is also checked
and printed, though it is not itself a fail-closed gate (the two tasks are
independent extraction runs and may legitimately be run separately).

FAIL-CLOSED VALIDATION:
  - overwrite preflight for THIS MODEL's/THIS TASK's every cell's own
    out_path/meta_path -- ALL of them, BEFORE the model is loaded;
  - n must equal exactly 1319;
  - all 1319 questions must be unique strings (refuses on any duplicate);
  - expert/non_expert must share the identical ordered question list (sha256
    equality asserted before EITHER condition's run starts);
  - the final prefill token id is read out at runtime for EVERY sample (not
    just the first) and asserted against the fixed expected id;
  - no double-BOS;
  - all extracted hidden states are finite;
  - atomic write: build the full (n, L, H) array in memory, verify it, THEN
    write to a temp path and os.replace() it into place.

MODEL LOADED EXACTLY ONCE PER LAUNCHER RUN (--run_all mode). Loads the model
ONCE, then runs BOTH conditions (expert, non_expert) of the ONE task given by
--task sequentially in-process, reusing the same loaded VicundaModel object.
--run_all with --task gsm8k_role runs only that task's 2 cells; --run_all
with --task gsm8k_role_abstention runs only that task's 2 cells. Running
BOTH tasks (4 cells total) in one process is deliberately NOT offered as a
single flag -- the task instructs running gsm8k_role on GPU 0 and
gsm8k_role_abstention on GPU 1 in parallel, which requires two separate
processes/devices regardless, so a combined "--run_all_tasks" mode would
only be a false convenience.

TWO-PASS ORDERING (fail-closed convention, CLAUDE.md). In --run_all mode this
script does a full upfront pass over both of the task's 2 cells -- computing
each cell's pairing digest and checking its output+meta files do not already
exist -- BEFORE the model is loaded or any output is written for even the
first cell.

@author: GSM8K Role hidden-state extraction pipeline (2026-09-21)
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

# The v2 abstention prompt is imported, not retyped -- see module docstring.
from get_answer_gsm8k_role_abstention_v2 import (
    PROMPT_TEMPLATE as ABSTENTION_PROMPT_TEMPLATE,
    ROLE_EXPERT as ABSTENTION_ROLE_EXPERT,
    ROLE_NON_EXPERT as ABSTENTION_ROLE_NON_EXPERT,
)

SCRIPT_VERSION = "extract_gsm8k_role_hs-v1"

# Reused verbatim from extract_role_hidden_states.py's MODELS dict.
MODELS = {
    "llama3": {
        "model_dir_default": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "8B",
        "expected_n_layers_out": 33,
        "expected_hidden_size": 4096,
    },
    "qwen2.5": {
        "model_dir_default": "Qwen/Qwen2.5-7B-Instruct",
        "size": "7B",
        "expected_n_layers_out": 29,
        "expected_hidden_size": 3584,
    },
}

TASKS = ("gsm8k_role", "gsm8k_role_abstention")
CONDITIONS = ("expert", "non_expert")

N_EXPECT_FULL = 1319

# Fixed global final-prefill-token expectation, matching
# extract_role_hidden_states.py's own convention: every rendered prompt in
# both task families ends in a literal trailing space ("...Answer: "),
# which the Llama-3.1 tokenizer renders as a single trailing ' ' token
# (id=220). NOT re-derived per-cell.
EXPECTED_FINAL_TOKEN_ID = 220
EXPECTED_FINAL_TOKEN_TEXT = " "


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prefill-only, no-generation, no-steering GSM8K Role "
                    "hidden-state extraction over the full 1319-question "
                    "test set (gsm8k_role / gsm8k_role_abstention).")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--condition", default=None, choices=CONDITIONS,
                   help="Required in single-cell mode; ignored (both "
                        "conditions are run) in --run_all mode.")
    p.add_argument("--run_all", action="store_true",
                   help="Load the model ONCE, then run both conditions "
                        "(expert, non_expert) for --task/--model "
                        "sequentially in this one process.")
    p.add_argument("--model_dir", default=None,
                   help="HF repo id. Defaults to this model's standard repo "
                        "id if omitted.")
    p.add_argument("--base_dir", required=True,
                   help="components/ dir, e.g. /data1/paveen/Dopamine/"
                        "components. Reads "
                        "<base_dir>/benchmark/gsm8k_1319_full.json. Output "
                        "goes to <base_dir>/hidden_states/<model>/<task>/... .")
    p.add_argument("--gsm8k_1319_file", default=None,
                   help="Default: <base_dir>/benchmark/gsm8k_1319_full.json")
    p.add_argument("--out_dir", default=None,
                   help="Single-cell mode only. Default: "
                        "<base_dir>/hidden_states/<model>/<task>/. In "
                        "--run_all mode, out_dir is always derived from "
                        "--base_dir and this flag is rejected.")
    p.add_argument("--expect_pairing_digest", default=None,
                   help="Single-cell mode only. If given, the "
                        "ordered-sample-identity sha256 computed here must "
                        "equal this value.")
    p.add_argument("--print_pairing_digest_only", action="store_true",
                   help="Print the ordered-sample-identity sha256 for this "
                        "task and exit 0 without loading any model.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Prompt construction -- two families, each a direct reuse of an existing
# code path (see module docstring for exact citations).
# --------------------------------------------------------------------------

ROLE_KEY_TO_NEG_CHARACTER = {
    # utils.ROLE_TO_CHARACTER's exact strings -- the gsm8k_role (no-
    # abstention) family's role routing.
    "expert": utils.ROLE_TO_CHARACTER["expert"],
    "non_expert": utils.ROLE_TO_CHARACTER["non_expert"],
}

ROLE_KEY_TO_ABSTENTION_ROLE_STRING = {
    "expert": ABSTENTION_ROLE_EXPERT,
    "non_expert": ABSTENTION_ROLE_NON_EXPERT,
}


def render_gsm8k_role_prompt(neg_template: str, question: str, role_key: str) -> str:
    """gsm8k_role (NO abstention exit): templates["neg"].format(character=
    utils.ROLE_TO_CHARACTER[role_key], context=question). Matches
    get_answer_regenerate_gsm8k.py line 66 /
    extract_role_hidden_states.py::render_reasoning_prompt() for GSM8K."""
    character = ROLE_KEY_TO_NEG_CHARACTER[role_key]
    return neg_template.format(character=character, context=question)


def render_gsm8k_role_abstention_prompt(question: str, role_key: str) -> str:
    """gsm8k_role_abstention: the imported v2 PROMPT_TEMPLATE, exactly as
    get_answer_gsm8k_role_abstention_v2.py::build_prompt() renders it."""
    role_string = ROLE_KEY_TO_ABSTENTION_ROLE_STRING[role_key]
    return ABSTENTION_PROMPT_TEMPLATE.format(question=question, role=role_string)


def load_gsm8k_1319(path: str):
    samples = utils.load_json(path)
    n = len(samples)
    if n != N_EXPECT_FULL:
        die(f"{path}: loaded {n} samples, expected exactly {N_EXPECT_FULL} "
            "(the full GSM8K test set built by build_gsm8k_1319_full.py). "
            "Refusing to extract on a wrong-size sample.")
    questions = [s["question"] for s in samples]
    if len(set(questions)) != n:
        die(f"{path}: contains duplicate questions -- refusing to proceed.")
    if "in_300_sample" not in samples[0]:
        die(f"{path}: rows do not carry an 'in_300_sample' field -- this "
            "must be the output of build_gsm8k_1319_full.py, not a "
            "different GSM8K file.")
    return samples


# --------------------------------------------------------------------------
# Tokenization + fail-closed checks (verbatim convention from
# extract_role_hidden_states.py)
# --------------------------------------------------------------------------

def tokenize_single(vc, text: str):
    return vc.tokenizer(text, return_tensors="pt", add_special_tokens=True)


def assert_no_double_bos(vc, input_ids_row) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = input_ids_row.tolist()
    if bos_id is not None and len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
        die(f"double BOS detected (head={ids[:4]}) -- refusing to proceed.")


def assert_final_token(tok_id: int, tok_text: str, sample_idx: int, task: str,
                       condition: str) -> None:
    if tok_id != EXPECTED_FINAL_TOKEN_ID or tok_text != EXPECTED_FINAL_TOKEN_TEXT:
        die(f"task={task} condition={condition} sample_idx={sample_idx}: "
            f"final prefill token is id={tok_id} text={tok_text!r}, "
            f"expected id={EXPECTED_FINAL_TOKEN_ID} "
            f"text={EXPECTED_FINAL_TOKEN_TEXT!r} (fixed global expectation). "
            "Refusing to extract hidden states under an unverified/drifted "
            "prefill boundary.")


# --------------------------------------------------------------------------
# Pairing digest
# --------------------------------------------------------------------------

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
# Output paths -- components/hidden_states/{model}/{task}/{condition}_{size}.h5
# (identical layout convention to extract_role_hidden_states.py's reasoning-
# task paths, with task in {gsm8k_role, gsm8k_role_abstention} instead of
# {gsm8k, math, gsm_hard}.)
# --------------------------------------------------------------------------

def h5_path(out_dir: str, condition: str, size: str) -> str:
    return os.path.join(out_dir, f"{condition}_{size}.h5")


def meta_path(out_dir: str, condition: str, size: str) -> str:
    return os.path.join(out_dir, f"{condition}_{size}.manifest.json")


class Cell:
    __slots__ = ("task", "condition", "samples", "out_path", "meta_path",
                "pairing_digest", "questions_file_sha256")

    def __init__(self, task, condition, samples, out_path, meta_path,
                pairing_digest, questions_file_sha256):
        self.task = task
        self.condition = condition
        self.samples = samples
        self.out_path = out_path
        self.meta_path = meta_path
        self.pairing_digest = pairing_digest
        self.questions_file_sha256 = questions_file_sha256


def build_cell(task, condition, out_dir, size, gsm8k_1319_file,
              expect_pairing_digest=None):
    samples = load_gsm8k_1319(gsm8k_1319_file)
    questions_file_sha256 = hashlib.sha256(
        open(gsm8k_1319_file, "rb").read()).hexdigest()

    pairing_digest = sha256_of_list([s["question"] for s in samples])
    print(f"[pairing] task={task} ordered_sample_identity_sha256="
          f"{pairing_digest}")

    if expect_pairing_digest and pairing_digest != expect_pairing_digest:
        die(f"ordered_sample_identity_sha256 {pairing_digest} != "
            f"--expect_pairing_digest {expect_pairing_digest}. The "
            "expert and non_expert conditions for this (model,task) must "
            "operate on the IDENTICAL ordered question list; refusing to "
            "proceed with a mismatched pairing.")

    out_path = h5_path(out_dir, condition, size)
    mpath = meta_path(out_dir, condition, size)
    return Cell(task, condition, samples, out_path, mpath, pairing_digest,
               questions_file_sha256)


# --------------------------------------------------------------------------
# Per-cell forward-pass processing
# --------------------------------------------------------------------------

def process_cell(vc, model_key, model_dir, size, cell: Cell, n_layers_out,
                 n_decoder, device_note):
    task, condition = cell.task, cell.condition
    samples = cell.samples
    out_path, meta_path_ = cell.out_path, cell.meta_path
    n = len(samples)

    print(f"model={model_key} task={task} condition={condition} n={n} "
          f"out_path={out_path}")

    if task == "gsm8k_role":
        neg_template = build_gsm8k_default_suite(
            cot=False, wording="plain")["neg"]
        rendered_prompts = [
            render_gsm8k_role_prompt(neg_template, s["question"], condition)
            for s in samples]
        prompt_body_sha256 = hashlib.sha256(
            neg_template.encode("utf-8")).hexdigest()
        role_string_used = ROLE_KEY_TO_NEG_CHARACTER[condition]
    else:  # gsm8k_role_abstention
        rendered_prompts = [
            render_gsm8k_role_abstention_prompt(s["question"], condition)
            for s in samples]
        prompt_body_sha256 = hashlib.sha256(
            ABSTENTION_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
        role_string_used = ROLE_KEY_TO_ABSTENTION_ROLE_STRING[condition]

    prompt_sha256 = hashlib.sha256(
        "\n".join(rendered_prompts).encode("utf-8")).hexdigest()

    print(f"role_string_used={role_string_used!r}")
    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"prompt_sha256={prompt_sha256}")
    print("First rendered prompt (repr, truncated to 400 chars):")
    print(repr(rendered_prompts[0][:400]))

    # ---- Fixed global final-prefill-token expectation, checked against
    # ---- every sample below. ----
    enc0 = tokenize_single(vc, rendered_prompts[0])
    assert_no_double_bos(vc, enc0["input_ids"][0])
    print(f"expected_final_prefill_token: id={EXPECTED_FINAL_TOKEN_ID} "
          f"text={EXPECTED_FINAL_TOKEN_TEXT!r} (fixed global expectation; "
          "asserted against every sample below)")

    hs_array = np.zeros((n, n_layers_out, vc.model.config.hidden_size),
                        dtype=np.float16)
    final_tok_ids = []
    final_tok_texts = []
    in_300_sample_flags = [bool(s.get("in_300_sample", False)) for s in samples]

    device = vc.model.device
    t0 = time.time()
    with torch.no_grad():
        for i, prompt in enumerate(rendered_prompts):
            enc = tokenize_single(vc, prompt)
            input_ids = enc["input_ids"][0]
            assert_no_double_bos(vc, input_ids)

            last_id = int(input_ids[-1].item())
            last_text = vc.tokenizer.decode([last_id])
            assert_final_token(last_id, last_text, i, task, condition)
            final_tok_ids.append(last_id)
            final_tok_texts.append(last_text)

            enc = {k: v.to(device) for k, v in enc.items()}
            out = vc.model(**enc, return_dict=True,
                           output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states
            if len(hidden_states) != n_layers_out:
                die(f"sample_idx={i}: got {len(hidden_states)} "
                    f"hidden_states layers, expected {n_layers_out} "
                    f"(embedding + {n_decoder} decoder layers).")

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
            "array -- unreachable given the per-sample check above; treat "
            "as a hard bug.")

    final_tail_unique = sorted(set(final_tok_ids))
    print(f"final prefill token ids observed across all {n} samples: "
          f"{final_tail_unique}")
    if len(final_tail_unique) != 1:
        die(f"final prefill token id is NOT constant across samples "
            f"({final_tail_unique}) -- every sample was already "
            "individually asserted above; treat as a hard bug.")

    n_in_300 = sum(in_300_sample_flags)
    n_remaining = n - n_in_300
    print(f"provenance split (audit-only, NOT used to filter): "
          f"in_300_sample={n_in_300} remaining_construction={n_remaining}")

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
                data=np.array([s["question"] for s in samples],
                             dtype=h5py.string_dtype(encoding="utf-8")))
            f.create_dataset(
                "in_300_sample",
                data=np.array(in_300_sample_flags, dtype=np.bool_))

            meta = f.attrs
            meta["script_version"] = SCRIPT_VERSION
            meta["git_commit"] = git_commit()
            meta["model"] = model_key
            meta["model_dir"] = model_dir
            meta["model_revision"] = "unpinned, uses default HF revision"
            meta["task"] = task
            meta["condition"] = condition
            meta["role_string_used"] = role_string_used
            meta["n_samples"] = n
            meta["n_samples_done"] = n
            meta["n_in_300_sample"] = n_in_300
            meta["n_remaining_construction"] = n_remaining
            meta["hidden_state_shape"] = json.dumps(
                [n, n_layers_out, int(vc.model.config.hidden_size)])
            meta["hidden_state_dtype"] = "float16"
            meta["n_layers"] = n_layers_out
            meta["n_decoder_layers"] = n_decoder
            meta["hidden_size"] = int(vc.model.config.hidden_size)
            meta["questions_sha256"] = cell.questions_file_sha256 or ""
            meta["ordered_sample_identity_sha256"] = cell.pairing_digest
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

    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": git_commit(),
        "model": model_key,
        "model_dir": model_dir,
        "model_revision": "unpinned, uses default HF revision",
        "task": task,
        "condition": condition,
        "role_string_used": role_string_used,
        "n_samples": n,
        "n_samples_done": n,
        "n_in_300_sample": n_in_300,
        "n_remaining_construction": n_remaining,
        "hidden_state_shape": [n, n_layers_out,
                               int(vc.model.config.hidden_size)],
        "hidden_state_dtype": "float16",
        "questions_sha256": cell.questions_file_sha256,
        "ordered_sample_identity_sha256": cell.pairing_digest,
        "prompt_body_sha256": prompt_body_sha256,
        "prompt_sha256": prompt_sha256,
        "chat_template_applied": False,
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
    tmp_manifest = meta_path_ + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_manifest, meta_path_)

    print(f"wrote {out_path}")
    print(f"wrote {meta_path_}")
    return n


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


def build_device_note(vc):
    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    return {"host": platform.node(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "param_devices": devs,
           "sharded": len(devs) > 1}


def run_all(args, model_key, mcfg, model_dir, size, base_dir, task,
           gsm8k_1319_file):
    out_dir = os.path.join(base_dir, "hidden_states", model_key, task)

    print("=" * 60)
    print(f"PASS 1/2: upfront validation of both conditions for "
          f"model={model_key} task={task} (no model loaded yet)")
    print("=" * 60)

    expert_cell = build_cell(task, "expert", out_dir, size, gsm8k_1319_file)
    nonexpert_cell = build_cell(
        task, "non_expert", out_dir, size, gsm8k_1319_file,
        expect_pairing_digest=expert_cell.pairing_digest)

    existing_files = []
    for c in (expert_cell, nonexpert_cell):
        for p in (c.out_path, c.meta_path):
            if os.path.exists(p):
                existing_files.append(p)
    if existing_files:
        die("refusing to run any cell for model="
            f"{model_key} task={task}: target file(s) already exist:\n  " +
            "\n  ".join(existing_files) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    print(f"PASS 1 complete: 2 cells validated, no output files exist yet, "
          f"n={len(expert_cell.samples)} each, pairing_digest="
          f"{expert_cell.pairing_digest}")

    print("=" * 60)
    print(f"Loading model ONCE: model={model_key} model_dir={model_dir}")
    print("=" * 60)
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    n_layers_out, n_decoder = model_shape_lock(vc, mcfg, model_key, model_dir)
    device_note = build_device_note(vc)

    print("=" * 60)
    print(f"PASS 2/2: processing 2 cells for task={task} with the one "
          "loaded model")
    print("=" * 60)
    for idx, cell in enumerate((expert_cell, nonexpert_cell)):
        print("=" * 60)
        print(f"[{idx + 1}/2] task={cell.task} condition={cell.condition} "
              f"time={time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        process_cell(vc, model_key, model_dir, size, cell, n_layers_out,
                    n_decoder, device_note)

    print("=" * 60)
    print(f"[DONE] Both cells finished for {model_key} task={task}.")
    print("=" * 60)


def main():
    args = parse_args()
    model_key = args.model
    mcfg = MODELS[model_key]
    size = mcfg["size"]
    model_dir = args.model_dir or mcfg["model_dir_default"]
    task = args.task

    base_dir = args.base_dir
    gsm8k_1319_file = args.gsm8k_1319_file or os.path.join(
        base_dir, "benchmark", "gsm8k_1319_full.json")

    if args.run_all:
        if args.condition is not None:
            die("--run_all is mutually exclusive with --condition; it runs "
                "both conditions for --model/--task in one process.")
        if args.out_dir is not None:
            die("--out_dir is not accepted with --run_all; the output "
                "directory is always derived from --base_dir/--task.")
        if args.expect_pairing_digest is not None:
            die("--expect_pairing_digest is not accepted with --run_all; "
                "pairing is checked internally.")
        if args.print_pairing_digest_only:
            die("--print_pairing_digest_only is not accepted with "
                "--run_all.")
        run_all(args, model_key, mcfg, model_dir, size, base_dir, task,
               gsm8k_1319_file)
        return

    # ---- Single-cell CLI mode ----
    out_dir = args.out_dir or os.path.join(
        base_dir, "hidden_states", model_key, task)

    cell = build_cell(task, args.condition if args.condition else "expert",
                      out_dir, size, gsm8k_1319_file,
                      expect_pairing_digest=None)

    if args.print_pairing_digest_only:
        print(cell.pairing_digest)
        return

    if args.condition is None:
        die("--condition is required unless --print_pairing_digest_only "
            "is given.")
    if args.condition != "expert":
        cell = build_cell(task, args.condition, out_dir, size,
                          gsm8k_1319_file, expect_pairing_digest=None)

    if args.expect_pairing_digest:
        if cell.pairing_digest != args.expect_pairing_digest:
            die(f"ordered_sample_identity_sha256 {cell.pairing_digest} != "
                f"--expect_pairing_digest {args.expect_pairing_digest}.")

    existing = [p for p in (cell.out_path, cell.meta_path)
               if os.path.exists(p)]
    if existing:
        die(f"refusing to run model={model_key} task={task} "
            f"condition={args.condition}: target file(s) already exist:\n  "
            + "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    n_layers_out, n_decoder = model_shape_lock(vc, mcfg, model_key, model_dir)
    device_note = build_device_note(vc)

    process_cell(vc, model_key, model_dir, size, cell, n_layers_out,
                n_decoder, device_note)
    print("Done.")


if __name__ == "__main__":
    main()
