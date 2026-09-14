#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
extract_role_hidden_states.py -- standalone, prefill-only, NO-generation,
NO-steering hidden-state extraction for the Expert vs Non-expert ROLE
condition, across four tasks (GSM8K, MATH, GSM-Hard, MMLU-E).

STRUCTURAL TEMPLATE. This script mirrors extract_interface_hidden_states.py's
overwrite-guard pattern, atomic H5 writing, manifest schema, model-shape
assertions, pairing-digest logic, and fail-closed validation ordering. It is
a SEPARATE script (not a fork/edit of that file) because the condition axis
here is role (expert/non_expert), not interface (chat/bare), and MMLU-E adds
a fourth task with a completely different per-subject output layout that
extract_interface_hidden_states.py has no code path for.

WHAT THIS SCRIPT IS AND IS NOT. Pure forward-pass HS dumper: tokenize one
rendered prompt, run ONE `model(**tokens, output_hidden_states=True)` call
(no `model.generate`, no forward hooks of any kind), and save the last valid
prefill-token hidden state at EVERY layer (embedding output + every
decoder-layer output). It never calls vc.regenerate / vc.generate / any
hook-registering method on VicundaModel -- no steering, no mask, no diff
matrix, anywhere.

PROMPT REUSE, NOT PROMPT RETYPING. Exact code paths traced and reused,
never hand-typed:
  - GSM8K / GSM-Hard: template.build_gsm8k_default_suite(cot=False,
    wording="plain")["neg"].format(character=utils.ROLE_TO_CHARACTER[role],
    context=question) -- the SAME routing get_answer_regenerate_gsm8k.py
    (line 66) uses for non-neutral roles: GSM8K has no E-option, so roles
    must NOT carry "honest" framing; bypass construct_prompt's
    default(+honest) branch entirely and go straight to templates["neg"].
    GSM-Hard reuses this GSM8K body verbatim, exactly as
    extract_interface_hidden_states.py's own docstring documents (GSM-Hard
    has no body builder of its own).
  - MATH: template.build_math_suite(cot=False)["neg"].format(
    character=utils.ROLE_TO_CHARACTER[role], context=question) -- the SAME
    routing get_answer_regenerate_math.py (line 52-54) uses: "Now you are
    {character}." with NO "honest" (MATH has no E-option either).
  - MMLU-E: utils.construct_prompt(vc, templates, ctx, role, use_chat=False)
    with templates = template.select_templates("default", use_E=True,
    cot=False) and role a string that is NEITHER "neutral"/"norole" NOR
    contains "not"/"confident" -- construct_prompt's own branching (utils.py
    lines 328-353) then takes the "else" branch: `templates["default"].
    format(character=role, context=ctx)`, i.e. templates["default"] is
    template_mmlu_E = 'Now you are an honest {character}, your answer
    among "A, B, C, D, E" is: ' with the "E) I am not sure." option line
    inserted above by the template itself. This is the EXACT construct_prompt
    call get_answer_logits.py (the historical MMLU-E extractor,
    run_hidden_mmlue_confidence_hs.sh's own driver) makes for every role in
    its custom --roles list. role strings here are literally "{subject}
    expert" / "non {subject} expert" (rendered via utils.make_characters
    with custom_roles=["{task} expert", "non {task} expert"], the same
    function get_answer_logits.py calls when --roles is supplied), so
    construct_prompt renders "Now you are an honest {subject} expert, ..."
    / "Now you are an honest non {subject} expert, ..." -- the WITH-honest
    wording the task spec requires for MMLU-E, produced by the actual
    existing code path, never a hand-written sentence.

SAMPLE FILES ARE REUSED VERBATIM, NEVER REGENERATED. GSM8K:
<base_dir>/benchmark/gsm8k_test_sample.json, n=300 (same file the frozen
run_gsm8k.sh / run_gsm8k_qwen25.sh point at). MATH:
<base_dir>/benchmark/math_test_sample.json, first 300 (all_samples[:300],
matching get_answer_regenerate_math.py / get_answer_math_chat_sweep.py).
GSM-Hard: the already label-free, frozen P3 file
<base_dir>/benchmark/gsm_hard_p3_questions.json (digest
48cc763545d2ee23835833f5165456b90db194863420a10b6741a57cff781d02,
docs/PREREG_P3.md / docs/p3_result_20260830.json) -- reused AS-IS, never
re-downloaded or re-derived, and the sealed gold file
(gsm_hard_p3_gold.SEALED.json) is never opened, read, or even referenced by
path anywhere in this script -- this extraction needs no label at all.
MMLU-E: <base_dir>/mmlu/{subject}.json for all 57 subjects in
detection/task_list.py -- the exact same per-task JSON files
run_hidden_mmlue_confidence_hs.sh / get_answer_logits.py already read
(schema: list of {"task","text","label"}; "text" already contains the
rendered A)/B)/C)/D) choice block per data_mmlu.py).

NO STEERING, ANYWHERE. No mask is loaded, no diff matrix is built, no
forward hook (register_forward_hook / register_forward_pre_hook) is ever
registered by this script or by any VicundaModel method it calls
(VicundaModel.get_logits / a raw vc.model(**enc) call registers no hooks --
hooks are registered only inside the regenerate()/_apply_diff_hooks family,
none of which this script calls).

ONE FORWARD PASS PER SAMPLE, bs=1, PREFILL ONLY.
`vc.model(**tokens, output_hidden_states=True, use_cache=False)`. No
`model.generate`, so there is no decode step at all.
`outputs.hidden_states` is a tuple of length num_decoder_layers+1: index 0
is the embedding output, indices 1..N are decoder-layer outputs (HF
hidden_states convention, matching CLAUDE.md's "Layer indexing convention"
section and extract_interface_hidden_states.py's identical framing). This
script stores ALL of them (33 for Llama giving (n,33,4096), 29 for Qwen
giving (n,29,3584)) -- the expected_n_layers_out/expected_hidden_size values
are reused verbatim from extract_interface_hidden_states.py's MODELS dict,
never recomputed.

LAST VALID PREFILL TOKEN. bs=1, no padding is ever inserted
(padding="longest" on a batch of one prompt never adds a pad token), so the
last token of input_ids[0] is unambiguously the last real prompt token; its
hidden state at every layer is hidden_states[l][0, -1, :].

EXPERT/NON-EXPERT PAIRING. Both conditions of one (model, task[, subject])
cell must share the identical ordered question list -- enforced the same
way extract_interface_hidden_states.py enforces chat/bare pairing: a
condition-independent ordered-sample-identity sha256, computable via
--print_pairing_digest_only without loading any model, and asserted before
either condition's extraction happens.

CROSS-MODEL DIGEST. The SAME pairing digest (question content + order, and
for GSM-Hard/MMLU-E, sample identity) is computed identically regardless of
--model, so a caller can confirm both models were fed byte-identical
question text/order for a given task/subject by comparing the two models'
printed digests -- this script does not itself compare across models (that
would require both to have already run), it prints the digest so a
consumer can.

FAIL-CLOSED VALIDATION:
  - overwrite preflight for THIS MODEL's every cell's own out_path(s) and
    meta_path(s) -- ALL of them, BEFORE the model is loaded (see "MODEL
    LOADED ONCE" below for exactly what "all" means in each mode);
  - n must equal exactly 300 (reasoning tasks) or the frozen per-subject
    counts summing to 14,042 (MMLU-E, computed from the real subject files
    at runtime, never hardcoded per-subject);
  - expert/non_expert must share the identical ordered question list
    (sha256 equality asserted before EITHER condition's run starts);
  - the final prefill token id is read out at runtime for EVERY sample (not
    just the first) and asserted against the expected id -- the expected id
    itself is READ FROM THE ACTUAL TOKENIZER at startup (never hardcoded as
    a bare literal), and the per-sample assertion compares against that
    tokenizer-derived value;
  - no double-BOS;
  - all extracted hidden states are finite;
  - GSM-Hard: refuses unless the questions file declares
    contains_labels=false, scans every row for a leaked label key, and never
    reads or references the sealed gold file;
  - atomic write: build the full (n, L, H) array in memory, verify it, THEN
    write to a temp path and os.replace() it into place, so a killed job
    never leaves a partial file at the final path.

MODEL LOADED EXACTLY ONCE PER LAUNCHER RUN (--run_all mode). This is the
PRIMARY mode the launcher scripts use. `--run_all` loads the model and
tokenizer ONE time, then iterates ALL 6 reasoning cells (gsm8k/math/
gsm_hard x expert/non_expert) and ALL 114 MMLU-E cells (57 subjects x
expert/non_expert) sequentially in-process, reusing the same loaded
VicundaModel object (vc) for every cell's forward passes -- no
`VicundaModel(...)` call happens more than once in the whole process.

TWO-PASS ORDERING (fail-closed convention, CLAUDE.md). In --run_all mode
this script does A FULL UPFRONT PASS over every one of the 120 cells --
loading each cell's sample file, computing its pairing digest, and checking
its output+meta files do not already exist -- BEFORE the model is loaded or
any output is written for even the first cell. Only after every cell in
this pass has been validated does the model load, and only then does the
per-cell processing loop begin (writing each cell's H5/manifest as it goes,
one cell fully written -- atomically -- before the next cell starts). This
is deliberately NOT "validate cell N right before writing cell N": the
overwrite guard is a single whole-model, all-cells check run once up front,
matching the launcher's previous one-time preflight and the repo's
documented "validate ALL inputs for ALL cells before writing ANY output"
convention. A per-subject MMLU-E JSON is loaded once during this upfront
pass and (for cheapness) NOT re-read from disk during the per-cell
processing pass -- the parsed sample list from the validation pass is
reused directly.

The single-cell CLI mode (--task <one task>, no --run_all) is KEPT WORKING
for debugging/re-running one cell in isolation without loading the model
120 times. It performs the SAME per-cell overwrite/pairing/model-shape/
finite/atomic-write checks as before, unchanged; the only difference in
--run_all mode is that the per-cell body runs many times against ONE loaded
model instead of once against a freshly-loaded model per process.

MMLU-E FILENAME CONVENTION (documented design choice, not derived from any
reference script). extract_interface_hidden_states.py has no expert/
non_expert naming to copy for MMLU-E, and get_answer_logits.py's own
`safe_role = role.replace(" ", "_").replace("-", "_")` convention would
produce long, subject-duplicated filenames from a full rendered role string
(e.g. "college_biology_expert_college_biology_8B.h5"). This script instead
uses the FIXED CONDITION KEY ("expert" / "non_expert") as the filename
prefix -- matching the SHAPE of run_hidden_mmlue_confidence_hs.sh's own
confident_/unconfident_ convention (a short, fixed, condition-keyed prefix,
not the full rendered role string), while keeping the literal condition
name rather than confident/unconfident since the axis here is role, not
confidence. Filenames:
  components/hidden_states/{model}/mmlue/expert_{subject}_{size}.h5
  components/hidden_states/{model}/mmlue/non_expert_{subject}_{size}.h5
each with a sibling `.manifest.json`. This convention is fixed by
mmlue_h5_path()/mmlue_meta_path() below and must not be changed casually --
it determines every consumer's glob pattern.

TREE-LEVEL MMLU-E MANIFEST. In --run_all mode, after all 114 MMLU-E cells
have been written, this script writes ONE additional aggregate manifest at
<out_dir>/manifest.json (components/hidden_states/{model}/mmlue/
manifest.json) satisfying the letter of the original spec's example output
layout (a single tree-wide manifest.json for the mmlue tree). It is a
lightweight INDEX, not a duplication of every per-cell manifest's full
metadata: subjects list, conditions, per-cell relative h5 path + n_samples,
and the overall total sample count (asserted == 14,042). See
write_mmlue_tree_manifest() for the exact schema. This step does not run in
single-cell CLI mode (there is no "all 114 cells" to index from a single
invocation).

@author: role hidden-state extraction pipeline (2026-09-14)
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
from template import build_gsm8k_default_suite, build_math_suite, select_templates
from detection.task_list import TASKS as MMLU_SUBJECTS

SCRIPT_VERSION = "extract_role_hidden_states-v2"

# Reused verbatim from extract_interface_hidden_states.py's MODELS dict --
# the task instructs "don't recompute", so these shape constants are the
# same numbers, not independently derived.
MODELS = {
    "llama3": {
        "model_dir_default": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "8B",
        # embedding + 32 decoder layers = 33; hidden_size = 4096.
        "expected_n_layers_out": 33,
        "expected_hidden_size": 4096,
    },
    "qwen2.5": {
        "model_dir_default": "Qwen/Qwen2.5-7B-Instruct",
        "size": "7B",
        # embedding + 28 decoder layers = 29; hidden_size = 3584.
        "expected_n_layers_out": 29,
        "expected_hidden_size": 3584,
    },
}

REASONING_TASKS = ("gsm8k", "math", "gsm_hard")
TASKS = ("gsm8k", "math", "gsm_hard", "mmlue")
CONDITIONS = ("expert", "non_expert")
ROLE_KEY_TO_ROLE_STRING = {
    # utils.ROLE_TO_CHARACTER's exact strings, used verbatim for GSM8K/MATH/
    # GSM-Hard's "neg" (no-honest) template path.
    "expert": utils.ROLE_TO_CHARACTER["expert"],
    "non_expert": utils.ROLE_TO_CHARACTER["non_expert"],
}

N_EXPECT_REASONING = 300
N_EXPECT_MMLUE_TOTAL = 14042

# The frozen GSM-Hard P3 questions digest (docs/PREREG_P3.md, protocol
# p3-v1; also recorded identically in extract_interface_hidden_states.py) --
# reused here as a sanity check that the same already-frozen file is read,
# never re-derived.
GSM_HARD_QUESTIONS_SHA256 = (
    "48cc763545d2ee23835833f5165456b90db194863420a10b6741a57cff781d02")

FORBIDDEN_GSM_HARD_KEYS = ("answer", "gold", "gold_answer", "correct",
                          "accuracy", "target")


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prefill-only, no-generation, no-steering, role "
                    "(expert vs non_expert) hidden-state extraction.")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--task", default=None, choices=TASKS,
                   help="Single-cell CLI mode: extract exactly one "
                        "{task[,subject],condition} cell. Mutually "
                        "exclusive with --run_all.")
    p.add_argument("--subject", default=None,
                   help="MMLU subject name (required iff --task mmlue; must "
                        "be one of detection.task_list.TASKS).")
    p.add_argument("--condition", default=None, choices=CONDITIONS,
                   help="Required in single-cell mode; ignored (all "
                        "conditions are run) in --run_all mode.")
    p.add_argument("--run_all", action="store_true",
                   help="Load the model ONCE, then run all 6 reasoning "
                        "cells + all 114 MMLU-E cells for --model "
                        "sequentially in this one process. Mutually "
                        "exclusive with --task/--subject/--condition.")
    p.add_argument("--model_dir", default=None,
                   help="HF repo id. Defaults to this model's standard repo "
                        "id if omitted.")
    p.add_argument("--base_dir", required=True,
                   help="components/ dir, e.g. /data1/paveen/Dopamine/"
                        "components. Sample files are read from "
                        "<base_dir>/benchmark/ (gsm8k/math/gsm_hard) or "
                        "<base_dir>/mmlu/ (mmlue). Output goes to "
                        "<base_dir>/hidden_states/<model>/<task>/... .")
    p.add_argument("--gsm8k_file", default=None,
                   help="Default: <base_dir>/benchmark/gsm8k_test_sample.json")
    p.add_argument("--math_file", default=None,
                   help="Default: <base_dir>/benchmark/math_test_sample.json")
    p.add_argument("--gsm_hard_file", default=None,
                   help="Default: <base_dir>/benchmark/"
                        "gsm_hard_p3_questions.json (the frozen P3 "
                        "label-free file, reused as-is)")
    p.add_argument("--mmlu_dir", default=None,
                   help="Default: <base_dir>/mmlu")
    p.add_argument("--out_dir", default=None,
                   help="Single-cell mode only. Default: "
                        "<base_dir>/hidden_states/<model>/<task>/ "
                        "(reasoning tasks) or "
                        "<base_dir>/hidden_states/<model>/mmlue/ (mmlue). "
                        "In --run_all mode, out_dir is always derived from "
                        "--base_dir per task and this flag is rejected.")
    p.add_argument("--expect_pairing_digest", default=None,
                   help="Single-cell mode only. If given, the "
                        "ordered-sample-identity sha256 computed here must "
                        "equal this value.")
    p.add_argument("--print_pairing_digest_only", action="store_true",
                   help="Single-cell mode only. Print the "
                        "ordered-sample-identity sha256 for this "
                        "(task[,subject]) and exit 0 without loading any "
                        "model.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Prompt construction -- REUSES the frozen role-routing code paths (see
# module docstring for exact citations to the sibling scripts each path
# matches).
# --------------------------------------------------------------------------

def load_gsm8k_samples(path: str):
    return utils.load_json(path)


def load_math_samples(path: str, n: int):
    all_samples = utils.load_json(path)
    if len(all_samples) < n:
        die(f"MATH sample file {path} holds only {len(all_samples)} "
            f"problems but {n} are required.")
    return all_samples[:n]


def load_gsm_hard_samples(path: str, expect_sha: str):
    """Label-free loader -- same contract as
    extract_interface_hidden_states.py's load_gsm_hard_samples(): refuse
    anything that does not declare itself label-free, and scan every row for
    a leaked label key. Reads ONLY the already-frozen P3 questions file --
    never downloads/regenerates GSM-Hard data, and never opens the sealed
    gold file (its path is not even constructed anywhere in this script)."""
    d = json.load(open(path, encoding="utf-8"))
    meta, data = d["meta"], d["data"]
    if meta.get("contains_labels") is not False:
        die("GSM-Hard questions file does not declare contains_labels=false; "
            "this extractor must stay on a label-free code path.")
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
            "used by the frozen bare/chat GSM-Hard experiments.")
    ids = [s["sample_id"] for s in data]
    if len(set(ids)) != len(ids):
        die("duplicate sample_id in the GSM-Hard questions file.")
    return meta, data


def render_reasoning_prompt(task: str, templates: dict, question: str,
                            role_key: str) -> str:
    """GSM8K / GSM-Hard / MATH, role-bearing (non-neutral) prompt, matching
    get_answer_regenerate_gsm8k.py (line 66) / get_answer_regenerate_math.py
    (lines 46-54)'s exact routing: templates["neg"].format(character=
    utils.ROLE_TO_CHARACTER[role_key], context=question). No "honest" is
    ever inserted -- these tasks have no E-option."""
    character = ROLE_KEY_TO_ROLE_STRING[role_key]
    return templates["neg"].format(character=character, context=question)


def mmlue_role_string(role_key: str, subject_formatted: str) -> str:
    """Matches utils.make_characters(task_name=subject, custom_roles=
    ["{task} expert", "non {task} expert"]) -- get_answer_logits.py's own
    --roles parsing path when given a custom_roles list containing "{task}"
    placeholders (utils.py lines 282-312: task_name lower()+replace('_',' '),
    then role.format(task=task_name_formatted) per role string)."""
    if role_key == "expert":
        template = "{task} expert"
    else:
        template = "non {task} expert"
    return utils.make_characters(subject_formatted, [template])[0]


def render_mmlue_prompt(vc, templates: dict, ctx: str, role_string: str) -> str:
    """MMLU-E, WITH honest -- the exact utils.construct_prompt(vc, templates,
    ctx, role, use_chat=False) call get_answer_logits.py makes for every
    role in its --roles list (get_answer_logits.py line 64). role_string
    here (e.g. "college biology expert") contains neither "not" nor
    "confident" and is not "neutral"/"norole", so construct_prompt's own
    branching (utils.py lines 344-349) takes the honest-default branch:
    templates["default"].format(character=role_string, context=ctx)."""
    return utils.construct_prompt(vc, templates, ctx, role_string,
                                  use_chat=False)


# --------------------------------------------------------------------------
# Tokenization + fail-closed checks
# --------------------------------------------------------------------------

def tokenize_single(vc, text: str):
    return vc.tokenizer(text, return_tensors="pt", add_special_tokens=True)


def assert_no_double_bos(vc, input_ids_row) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = input_ids_row.tolist()
    if bos_id is not None and len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
        die(f"double BOS detected (head={ids[:4]}) -- refusing to proceed.")


# Hard global invariant: every rendered prompt in this pipeline (both
# models, all four tasks, both role conditions) ends its bare-string prefill
# in a literal trailing space right before generation would begin (e.g.
# "...Answer: " / "...is: "), which every tokenizer used here (Llama-3.1 and
# Qwen2.5, both bare-string, no chat template) renders as a single trailing
# ' ' token. This was independently confirmed against the REAL cached
# tokenizer for both models across all four prompt types during development
# (see the module's own smoke-test history) and matches
# extract_interface_hidden_states.py's own frozen `expected_tail_bare`
# convention (id=220, text=' '). It is intentionally NOT re-derived
# per-cell from each cell's own first sample -- deriving it that way would
# let a uniform template drift (e.g. every prompt losing its trailing
# space) pass silently, since every sample in the drifted cell would still
# agree with each other. Every sample of every cell is instead checked
# against this ONE fixed, hardcoded expectation.
EXPECTED_FINAL_TOKEN_ID = 220
EXPECTED_FINAL_TOKEN_TEXT = " "


def derive_expected_final_token(vc, first_rendered_prompt: str):
    """Tokenize the first rendered prompt of a cell (only to run the
    double-BOS guard on a real encoding) and return the GLOBAL fixed
    expectation (EXPECTED_FINAL_TOKEN_ID, EXPECTED_FINAL_TOKEN_TEXT) rather
    than whatever this cell's own first sample happens to produce -- see the
    module-level comment above. Kept as a function (not inlined at call
    sites) so every caller still goes through one place."""
    enc = tokenize_single(vc, first_rendered_prompt)
    input_ids = enc["input_ids"][0]
    assert_no_double_bos(vc, input_ids)
    return EXPECTED_FINAL_TOKEN_ID, EXPECTED_FINAL_TOKEN_TEXT


def assert_final_token(expected_id: int, expected_text: str, tok_id: int,
                       tok_text: str, sample_idx: int, task: str,
                       subject: str, condition: str) -> None:
    if tok_id != expected_id or tok_text != expected_text:
        die(f"task={task} subject={subject} condition={condition} "
            f"sample_idx={sample_idx}: final prefill token is "
            f"id={tok_id} text={tok_text!r}, expected id={expected_id} "
            f"text={expected_text!r} (this is a FIXED global expectation, "
            "not derived per-cell -- see EXPECTED_FINAL_TOKEN_ID/TEXT and "
            "the comment above derive_expected_final_token()). Refusing to "
            "extract hidden states under an unverified/drifted prefill "
            "boundary.")


# --------------------------------------------------------------------------
# Pairing digest -- computed identically by BOTH conditions of the same
# (model, task[, subject]), so expert/non_expert pairing (and cross-model
# question identity) can be asserted without either process needing to see
# the other's in-memory state.
# --------------------------------------------------------------------------

def ordered_identity_list(task: str, samples: list) -> list[str]:
    if task == "gsm_hard":
        return [f"{s['sample_id']}\x1f{s['question']}" for s in samples]
    if task == "mmlue":
        return [f"{i}\x1f{s['text']}\x1f{s['label']}"
               for i, s in enumerate(samples)]
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
# Output path helpers -- see the "MMLU-E FILENAME CONVENTION" module
# docstring section for the documented rationale of this naming choice.
#   reasoning tasks: components/hidden_states/{model}/{task}/{condition}_{size}.h5
#   mmlue: components/hidden_states/{model}/mmlue/{expert|non_expert}_{subject}_{size}.h5
# --------------------------------------------------------------------------

def mmlue_h5_path(out_dir: str, condition: str, subject: str, size: str) -> str:
    prefix = "expert" if condition == "expert" else "non_expert"
    return os.path.join(out_dir, f"{prefix}_{subject}_{size}.h5")


def mmlue_meta_path(out_dir: str, condition: str, subject: str, size: str) -> str:
    return mmlue_h5_path(out_dir, condition, subject, size) + ".manifest.json"


def reasoning_h5_path(out_dir: str, condition: str, size: str) -> str:
    return os.path.join(out_dir, f"{condition}_{size}.h5")


def reasoning_meta_path(out_dir: str, condition: str, size: str) -> str:
    return os.path.join(out_dir, f"{condition}_{size}.manifest.json")


# --------------------------------------------------------------------------
# Cell descriptor: everything needed to validate and later process one
# {task[,subject],condition} cell, resolved ONCE during the upfront
# validation pass and reused, unchanged, during the processing pass.
# --------------------------------------------------------------------------

class Cell:
    __slots__ = (
        "task", "subject", "condition", "samples", "templates",
        "question_key", "out_path", "meta_path", "pairing_digest",
        "questions_file_sha256",
    )

    def __init__(self, task, subject, condition, samples, templates,
                question_key, out_path, meta_path, pairing_digest,
                questions_file_sha256):
        self.task = task
        self.subject = subject
        self.condition = condition
        self.samples = samples
        self.templates = templates
        self.question_key = question_key
        self.out_path = out_path
        self.meta_path = meta_path
        self.pairing_digest = pairing_digest
        self.questions_file_sha256 = questions_file_sha256


def load_task_samples(task, subject, gsm8k_file, math_file, gsm_hard_file,
                      mmlu_dir):
    """Load one task[,subject]'s sample list + templates + question_key +
    questions_file_sha256. Shared by both single-cell mode and the
    --run_all upfront validation pass, so the loading logic exists in
    exactly one place."""
    question_key = "question"
    data_path = None
    if task == "gsm8k":
        samples = load_gsm8k_samples(gsm8k_file)
        templates = build_gsm8k_default_suite(cot=False, wording="plain")
        data_path = gsm8k_file
    elif task == "math":
        samples = load_math_samples(math_file, N_EXPECT_REASONING)
        templates = build_math_suite(cot=False)
        data_path = math_file
    elif task == "gsm_hard":
        qmeta, samples = load_gsm_hard_samples(
            gsm_hard_file, GSM_HARD_QUESTIONS_SHA256)
        templates = build_gsm8k_default_suite(cot=False, wording="plain")
        data_path = gsm_hard_file
    elif task == "mmlue":
        templates = select_templates("default", use_E=True, cot=False)
        data_path = os.path.join(mmlu_dir, f"{subject}.json")
        if not os.path.isfile(data_path):
            die(f"MMLU-E subject file not found: {data_path}")
        samples = utils.load_json(data_path)
        question_key = "text"
    else:
        die(f"unknown task {task}")

    n = len(samples)
    if task in REASONING_TASKS:
        if n != N_EXPECT_REASONING:
            die(f"{task}: loaded {n} samples, expected exactly "
                f"{N_EXPECT_REASONING}. Refusing to extract on a "
                "wrong-size sample.")
    else:  # mmlue: per-subject count is whatever the real file holds; the
        # 14,042-total check happens once ALL 57 subjects are loaded (see
        # the --run_all upfront pass / launcher for the reasoning-task
        # equivalent). A single subject must still be non-empty.
        if n == 0:
            die(f"mmlue subject={subject}: loaded 0 samples.")

    questions_file_sha256 = hashlib.sha256(
        open(data_path, "rb").read()).hexdigest()

    return samples, templates, question_key, questions_file_sha256


def build_cell(task, subject, condition, out_dir_for_task, size,
              gsm8k_file, math_file, gsm_hard_file, mmlu_dir,
              expect_pairing_digest=None):
    """Load one cell's samples, compute its pairing digest, resolve its
    output paths -- everything the overwrite-preflight and the later
    forward-pass loop need, EXCEPT the model. Used by both single-cell mode
    and the --run_all upfront validation pass."""
    samples, templates, question_key, questions_file_sha256 = \
        load_task_samples(task, subject, gsm8k_file, math_file,
                          gsm_hard_file, mmlu_dir)

    pairing_ids = ordered_identity_list(task, samples)
    pairing_digest = sha256_of_list(pairing_ids)
    pairing_id_key = task if task != "mmlue" else f"mmlue:{subject}"
    print(f"[pairing] task={pairing_id_key} "
          f"ordered_sample_identity_sha256={pairing_digest}")

    if expect_pairing_digest and pairing_digest != expect_pairing_digest:
        die(f"ordered_sample_identity_sha256 {pairing_digest} != "
            f"--expect_pairing_digest {expect_pairing_digest}. "
            "The expert and non_expert conditions for this "
            "(model,task[,subject]) must operate on the IDENTICAL "
            "ordered question list; refusing to proceed with a "
            "mismatched pairing.")

    if task == "mmlue":
        out_path = mmlue_h5_path(out_dir_for_task, condition, subject, size)
        meta_path = mmlue_meta_path(out_dir_for_task, condition, subject, size)
    else:
        out_path = reasoning_h5_path(out_dir_for_task, condition, size)
        meta_path = reasoning_meta_path(out_dir_for_task, condition, size)

    return Cell(task, subject, condition, samples, templates, question_key,
               out_path, meta_path, pairing_digest, questions_file_sha256)


# --------------------------------------------------------------------------
# Per-cell forward-pass processing. Takes an ALREADY-LOADED vc; never
# constructs a VicundaModel itself. This is the body shared by single-cell
# CLI mode (called once) and --run_all mode (called up to 120 times against
# the SAME vc).
# --------------------------------------------------------------------------

def process_cell(vc, mcfg, model_key, model_dir, size, cell: Cell,
                 n_layers_out, n_decoder, device_note):
    task, subject, condition = cell.task, cell.subject, cell.condition
    samples, templates = cell.samples, cell.templates
    question_key = cell.question_key
    out_path, meta_path = cell.out_path, cell.meta_path
    n = len(samples)

    print(f"model={model_key} task={task} subject={subject or ''} "
          f"condition={condition} n={n} out_path={out_path}")

    if task in ("gsm8k", "math", "gsm_hard"):
        rendered_prompts = [
            render_reasoning_prompt(task, templates, s[question_key], condition)
            for s in samples]
        prompt_body_sha256 = hashlib.sha256(
            templates["neg"].encode("utf-8")).hexdigest()
        role_string_used = ROLE_KEY_TO_ROLE_STRING[condition]
    else:  # mmlue
        subject_formatted = subject.lower().replace("_", " ")
        role_string_used = mmlue_role_string(condition, subject_formatted)
        rendered_prompts = [
            render_mmlue_prompt(vc, templates, s[question_key],
                                role_string_used)
            for s in samples]
        prompt_body_sha256 = hashlib.sha256(
            templates["default"].encode("utf-8")).hexdigest()

    prompt_sha256 = hashlib.sha256(
        "\n".join(rendered_prompts).encode("utf-8")).hexdigest()

    print(f"role_string_used={role_string_used!r}")
    print(f"prompt_body_sha256={prompt_body_sha256}")
    print(f"prompt_sha256={prompt_sha256}")
    print("First rendered prompt (repr, truncated to 400 chars):")
    print(repr(rendered_prompts[0][:400]))

    # ---- Fixed global final-prefill-token expectation (id=220, text=' '),
    # ---- checked against every sample below -- NOT re-derived per cell, so
    # ---- a uniform template drift cannot pass silently. See
    # ---- EXPECTED_FINAL_TOKEN_ID/TEXT and derive_expected_final_token(). ----
    expected_id, expected_text = derive_expected_final_token(
        vc, rendered_prompts[0])
    print(f"expected_final_prefill_token: id={expected_id} "
          f"text={expected_text!r} (fixed global expectation; asserted "
          "against every sample below)")

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
            assert_final_token(expected_id, expected_text, last_id,
                               last_text, i, task, subject or "", condition)
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
            f"({final_tail_unique}) -- every sample was already "
            "individually asserted against the tokenizer-derived expected "
            "id above, so this indicates the assertion itself is "
            "unreliable; treat as a hard bug.")

    # ---- Atomic write: build in memory, verify, THEN write to a temp path
    # ---- and os.replace() into place. ----
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    fname_prefix = (f"{('expert' if condition == 'expert' else 'non_expert')}_"
                    f"{subject}_" if task == "mmlue"
                    else f"{condition}_{size}_")
    fd, tmp_h5_path = tempfile.mkstemp(
        suffix=".h5.tmp", dir=out_dir, prefix=fname_prefix)
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
            if task == "mmlue":
                f.create_dataset(
                    "label",
                    data=np.array([int(s["label"]) for s in samples],
                                 dtype=np.int64))

            meta = f.attrs
            meta["script_version"] = SCRIPT_VERSION
            meta["git_commit"] = git_commit()
            meta["model"] = model_key
            meta["model_dir"] = model_dir
            meta["model_revision"] = "unpinned, uses default HF revision"
            meta["task"] = task
            meta["subject"] = subject or ""
            meta["condition"] = condition
            meta["role_string_used"] = role_string_used
            meta["n_samples"] = n
            meta["n_samples_done"] = n
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
        "subject": subject or "",
        "condition": condition,
        "role_string_used": role_string_used,
        "n_samples": n,
        "n_samples_done": n,
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
    tmp_manifest = meta_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_manifest, meta_path)

    print(f"wrote {out_path}")
    print(f"wrote {meta_path}")
    return n


def model_shape_lock(vc, mcfg, model_key, model_dir):
    """Compute (n_layers_out, n_decoder) from the loaded model and assert
    them against this --model key's expected shape. Called once, right
    after model load, in both single-cell and --run_all mode."""
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


def write_mmlue_tree_manifest(out_dir, model_key, model_dir, size,
                              mmlue_cells_written):
    """Write ONE tree-level manifest.json for the whole mmlue output tree,
    after all 114 MMLU-E cells have been written. This is an INDEX over the
    per-cell manifests already on disk (relative h5 path + n_samples per
    subject/condition), not a duplication of their full metadata -- see the
    module docstring's "TREE-LEVEL MMLU-E MANIFEST" section. Schema:

    {
      "script_version": ..., "git_commit": ..., "model": ..., "model_dir": ...,
      "n_subjects": 57, "conditions": ["expert", "non_expert"],
      "n_unique_samples": 14042,
      "n_samples_by_condition": {"expert": 14042, "non_expert": 14042},
      "total_rows_stored": 28084,
      "cells": [
        {"subject": ..., "condition": ..., "h5_path": "<basename>.h5",
         "n_samples": ...},
        ...
      ],
      "extraction_timestamp_utc": ...
    }

    Three distinct counts are reported, deliberately NOT collapsed into one
    ambiguous "total_n_samples" field: n_unique_samples is the number of
    distinct MMLU-E QUESTIONS (14042, the number asserted per-condition
    elsewhere in this script as N_EXPECT_MMLUE_TOTAL); n_samples_by_condition
    breaks that same 14042 down per role condition (both entries equal
    n_unique_samples, since expert and non_expert are the SAME 14042
    questions asked twice, not 14042 combined across both); and
    total_rows_stored is the actual number of HDF5 rows written across all
    114 cells (2 x 14042 = 28084), i.e. what you would get by summing
    n_samples over every entry in "cells" below. Do not read
    n_samples_by_condition's values as summing to a grand total -- they are
    each independently equal to n_unique_samples.
    """
    cells_by_condition = {"expert": 0, "non_expert": 0}
    entries = []
    for subject, condition, n_samples, h5_path in mmlue_cells_written:
        cells_by_condition[condition] += n_samples
        entries.append({
            "subject": subject,
            "condition": condition,
            "h5_path": os.path.basename(h5_path),
            "n_samples": n_samples,
        })

    n_unique_by_condition = set(cells_by_condition.values())
    n_unique_samples = (n_unique_by_condition.pop()
                        if len(n_unique_by_condition) == 1 else None)
    if n_unique_samples is None:
        die("MMLU-E tree manifest: expert and non_expert sample counts "
            f"disagree ({cells_by_condition}) -- cannot report a single "
            "n_unique_samples value. This means the two conditions did not "
            "cover the same question set; refusing to write a manifest "
            "that would silently hide that mismatch.")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": git_commit(),
        "model": model_key,
        "model_dir": model_dir,
        "size": size,
        "n_subjects": len(set(e["subject"] for e in entries)),
        "conditions": list(CONDITIONS),
        "n_unique_samples": n_unique_samples,
        "n_samples_by_condition": cells_by_condition,
        "total_rows_stored": sum(cells_by_condition.values()),
        "cells": entries,
        "extraction_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(out_path):
        die(f"refusing to overwrite existing tree-level manifest: {out_path}")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    print(f"wrote tree-level MMLU-E manifest: {out_path}")
    return manifest


# --------------------------------------------------------------------------
# --run_all driver: upfront validation of EVERY cell (all 6 reasoning + all
# 114 mmlue), THEN one model load, THEN sequential in-process processing of
# every cell with that one loaded model.
# --------------------------------------------------------------------------

def run_all(args, model_key, mcfg, model_dir, size, base_dir,
           gsm8k_file, math_file, gsm_hard_file, mmlu_dir):
    reasoning_out_dirs = {
        t: os.path.join(base_dir, "hidden_states", model_key, t)
        for t in REASONING_TASKS}
    mmlue_out_dir = os.path.join(base_dir, "hidden_states", model_key, "mmlue")

    if not os.path.isdir(mmlu_dir):
        die(f"MMLU_DIR not found: {mmlu_dir}")
    missing = [t for t in MMLU_SUBJECTS
              if not os.path.isfile(os.path.join(mmlu_dir, f"{t}.json"))]
    if missing:
        die("Missing MMLU task JSON files under "
            f"{mmlu_dir}:\n  " + "\n  ".join(missing))

    # ======================================================================
    # PASS 1 (upfront validation, BEFORE any model load): build every one
    # of the 6 reasoning cells and 114 mmlue cells (Cell objects: samples
    # loaded, pairing digest computed and cross-checked between conditions,
    # output paths resolved), and check NONE of their output/meta files
    # already exist. This is the single whole-model, all-cells overwrite
    # guard -- it runs once, for ALL cells, before the model is touched.
    # ======================================================================
    print("=" * 60)
    print(f"PASS 1/2: upfront validation of all cells for model={model_key} "
          "(no model loaded yet)")
    print("=" * 60)

    all_cells: list[Cell] = []
    existing_files = []

    for t in REASONING_TASKS:
        print(f"--- validating task={t} ---")
        expert_cell = build_cell(t, None, "expert", reasoning_out_dirs[t],
                                 size, gsm8k_file, math_file, gsm_hard_file,
                                 mmlu_dir)
        nonexpert_cell = build_cell(
            t, None, "non_expert", reasoning_out_dirs[t], size, gsm8k_file,
            math_file, gsm_hard_file, mmlu_dir,
            expect_pairing_digest=expert_cell.pairing_digest)
        for c in (expert_cell, nonexpert_cell):
            all_cells.append(c)
            for p in (c.out_path, c.meta_path):
                if os.path.exists(p):
                    existing_files.append(p)

    mmlue_total_by_condition = {"expert": 0, "non_expert": 0}
    for subj in MMLU_SUBJECTS:
        expert_cell = build_cell(
            "mmlue", subj, "expert", mmlue_out_dir, size, gsm8k_file,
            math_file, gsm_hard_file, mmlu_dir)
        nonexpert_cell = build_cell(
            "mmlue", subj, "non_expert", mmlue_out_dir, size, gsm8k_file,
            math_file, gsm_hard_file, mmlu_dir,
            expect_pairing_digest=expert_cell.pairing_digest)
        for c in (expert_cell, nonexpert_cell):
            all_cells.append(c)
            mmlue_total_by_condition[c.condition] += len(c.samples)
            for p in (c.out_path, c.meta_path):
                if os.path.exists(p):
                    existing_files.append(p)

    mmlue_manifest_path = os.path.join(mmlue_out_dir, "manifest.json")
    if os.path.exists(mmlue_manifest_path):
        existing_files.append(mmlue_manifest_path)

    if existing_files:
        die("refusing to run any cell for model="
            f"{model_key}: target file(s) already exist:\n  " +
            "\n  ".join(existing_files) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    for cond, total in mmlue_total_by_condition.items():
        if total != N_EXPECT_MMLUE_TOTAL:
            die(f"MMLU-E total sample count for condition={cond} is "
                f"{total}, expected {N_EXPECT_MMLUE_TOTAL}.")

    n_reasoning_cells = len(REASONING_TASKS) * len(CONDITIONS)
    n_mmlue_cells = len(MMLU_SUBJECTS) * len(CONDITIONS)
    print(f"PASS 1 complete: {n_reasoning_cells} reasoning cells + "
          f"{n_mmlue_cells} mmlue cells = {len(all_cells)} cells validated, "
          "no output files exist yet, MMLU-E totals "
          f"{mmlue_total_by_condition} == {N_EXPECT_MMLUE_TOTAL} each.")

    # ======================================================================
    # Model load -- EXACTLY ONCE for this whole run, after ALL cells above
    # have been validated.
    # ======================================================================
    print("=" * 60)
    print(f"Loading model ONCE: model={model_key} model_dir={model_dir}")
    print("=" * 60)
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    n_layers_out, n_decoder = model_shape_lock(vc, mcfg, model_key, model_dir)
    device_note = build_device_note(vc)

    # ======================================================================
    # PASS 2: sequential in-process processing of every validated cell,
    # reusing the SAME vc for every forward pass. Each cell is fully
    # written (atomically) before the next cell starts.
    # ======================================================================
    print("=" * 60)
    print(f"PASS 2/2: processing {len(all_cells)} cells with the one loaded "
          "model")
    print("=" * 60)

    mmlue_cells_written = []
    for idx, cell in enumerate(all_cells):
        print("=" * 60)
        print(f"[{idx + 1}/{len(all_cells)}] task={cell.task} "
              f"subject={cell.subject or ''} condition={cell.condition} "
              f"time={time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        n_written = process_cell(vc, mcfg, model_key, model_dir, size, cell,
                                 n_layers_out, n_decoder, device_note)
        if cell.task == "mmlue":
            mmlue_cells_written.append(
                (cell.subject, cell.condition, n_written, cell.out_path))

    # ======================================================================
    # Tree-level MMLU-E manifest, written once all 114 mmlue cells are done.
    # ======================================================================
    write_mmlue_tree_manifest(mmlue_out_dir, model_key, model_dir, size,
                              mmlue_cells_written)

    print("=" * 60)
    print(f"[DONE] All {len(all_cells)} cells finished for {model_key} "
          "in a single model load.")
    print("=" * 60)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    model_key = args.model
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
    mmlu_dir = args.mmlu_dir or os.path.join(base_dir, "mmlu")

    if args.run_all:
        if args.task is not None or args.condition is not None:
            die("--run_all is mutually exclusive with --task/--condition "
                "(and --subject); it runs every task/subject/condition for "
                "--model in one process.")
        if args.subject is not None:
            die("--subject is not accepted with --run_all.")
        if args.out_dir is not None:
            die("--out_dir is not accepted with --run_all; output "
                "directories are always derived per-task from --base_dir.")
        if args.expect_pairing_digest is not None:
            die("--expect_pairing_digest is not accepted with --run_all; "
                "pairing is checked internally for every cell.")
        if args.print_pairing_digest_only:
            die("--print_pairing_digest_only is not accepted with "
                "--run_all.")
        run_all(args, model_key, mcfg, model_dir, size, base_dir,
               gsm8k_file, math_file, gsm_hard_file, mmlu_dir)
        return

    # ---- Single-cell CLI mode (unchanged behavior from v1) ----
    task = args.task
    condition = args.condition
    if task is None:
        die("--task is required unless --run_all is given.")

    if task == "mmlue":
        if not args.subject:
            die("--subject is required when --task mmlue.")
        if args.subject not in MMLU_SUBJECTS:
            die(f"--subject {args.subject!r} is not in "
                "detection.task_list.TASKS.")
    elif args.subject:
        die(f"--subject was given but --task {task} != mmlue; "
            "--subject only applies to mmlue.")

    if task == "mmlue":
        out_dir = args.out_dir or os.path.join(
            base_dir, "hidden_states", model_key, "mmlue")
    else:
        out_dir = args.out_dir or os.path.join(
            base_dir, "hidden_states", model_key, task)

    # ---- Load samples + build the role-bearing prompt list. Loaded before
    # ---- the overwrite preflight (matching v1's ordering), so a
    # ---- data-loading failure never wastes time on the preflight check
    # ---- for a cell whose input cannot even be read. ----
    cell = build_cell(task, args.subject, condition if condition else
                      "expert", out_dir, size, gsm8k_file, math_file,
                      gsm_hard_file, mmlu_dir,
                      expect_pairing_digest=None)

    if args.print_pairing_digest_only:
        print(cell.pairing_digest)
        return

    if condition is None:
        die("--condition is required unless --print_pairing_digest_only "
            "is given.")
    if condition != "expert":
        # Recompute the cell for the real requested condition -- build_cell
        # above defaulted to "expert" solely to resolve out_path/pairing
        # without requiring --condition when only the digest is wanted.
        cell = build_cell(task, args.subject, condition, out_dir, size,
                          gsm8k_file, math_file, gsm_hard_file, mmlu_dir,
                          expect_pairing_digest=None)

    if args.expect_pairing_digest:
        if cell.pairing_digest != args.expect_pairing_digest:
            die(f"ordered_sample_identity_sha256 {cell.pairing_digest} != "
                f"--expect_pairing_digest {args.expect_pairing_digest}. "
                "The expert and non_expert conditions for this "
                "(model,task[,subject]) must operate on the IDENTICAL "
                "ordered question list; refusing to proceed with a "
                "mismatched pairing.")

    # ---- Overwrite preflight for THIS CELL's own output files only, ----
    # ---- BEFORE model load. ----
    existing = [p for p in (cell.out_path, cell.meta_path)
               if os.path.exists(p)]
    if existing:
        die(f"refusing to run model={model_key} task={task} "
            f"subject={args.subject or ''} condition={condition}: target "
            "file(s) already exist:\n  " + "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    # ---- Model load ----
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    n_layers_out, n_decoder = model_shape_lock(vc, mcfg, model_key, model_dir)
    device_note = build_device_note(vc)

    process_cell(vc, mcfg, model_key, model_dir, size, cell, n_layers_out,
                n_decoder, device_note)
    print("Done.")


if __name__ == "__main__":
    main()
