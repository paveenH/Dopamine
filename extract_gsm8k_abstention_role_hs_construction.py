#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
extract_gsm8k_abstention_role_hs_construction.py -- SAME prompt/model/HS-
extraction logic as extract_gsm8k_abstention_role_hs.py, but reading the
LARGER GSM8K "construction split" (benchmark/gsm8k_construction_split.json,
built by build_gsm8k_construction_split.py -- the full GSM8K test set minus
the existing frozen 300-question benchmark) instead of the locked
300-question file, and writing to its OWN, separate output tree.

WHY A SEPARATE SCRIPT (not a flag on extract_gsm8k_abstention_role_hs.py).
That script hard-asserts n==300 and a specific known pairing digest as a
correctness guard for the existing 300-question pilot extraction, which is
frozen output that must keep reproducing byte-identically. This script
targets a DIFFERENT, larger sample (n~1019, NOT hardcoded -- the actual
count is read from the construction-split file and recorded, never
assumed), so it needs its own sample-count and pairing-digest expectations.
Rather than parameterize the frozen script's guards into optionality (which
would weaken them for its existing frozen use), this is a standalone
sibling script.

IDENTICAL, UNCHANGED FROM THE 300-QUESTION SCRIPT:
  - model configs (MODELS dict, shape locks)
  - prompt construction (build_abstention_templates(),
    assert_no_other_diff_than_role_wording()) -- same "an honest {character}"
    + abstention-sentence prompt, same bare-noun role strings, same
    programmatic diff-from-frozen-suite check
  - forward-pass-only HS extraction mechanics (prefill-only, no generation,
    no hooks, no steering, fixed-final-token derivation and enforcement,
    atomic H5 write, per-cell attrs)
  - CRASH-RECOVERY behaviour and caveat (same as the 300-question script)

CHANGED:
  - input file: <base_dir>/benchmark/gsm8k_construction_split.json instead
    of gsm8k_test_sample.json
  - n is NOT hardcoded to 300 -- read from the loaded file, asserted > 0,
    and cross-checked against the construction-split manifest's own
    recorded count (fail-closed on mismatch)
  - pairing-digest check is NOT against the old 300-question extraction's
    known value (that value belongs to a DIFFERENT, disjoint sample) --
    instead this script recomputes the construction split's own digest via
    the SAME formula and cross-checks it against
    gsm8k_construction_split_manifest.json's own recorded digest
    (construction_split_ordered_sample_identity_sha256), which was written
    by build_gsm8k_construction_split.py from the SAME full-test-set
    download, independent of this script re-reading the JSON file --
    catching drift between the JSON file on disk and the manifest that
    accompanies it.
  - output tree: components/hidden_states/{model}/
    gsm8k_abstention_role_construction/ (separate from
    gsm8k_abstention_role/, which holds the 300-question pilot extraction --
    the pilot output is NEVER read, modified, or deleted by this script)

SCOPE, same as the 300-question script: GSM8K only, Llama-3.1-8B-Instruct
and Qwen2.5-7B-Instruct only. No MATH, no GSM-Hard, no MMLU-E, no Chat, no
generation/answer audit, no steering, no layer-band selection, no NMD mask
construction, no RRSN/MRSN similarity analysis in this script.

template.py, extract_gsm8k_abstention_role_hs.py, extract_role_hidden_states.py,
and every other existing script are NOT modified or imported for execution
(the prompt-building logic is duplicated here rather than imported, to keep
this script fully standalone and not create a runtime dependency on the
300-question script's module-level constants).

OUTPUT (server): components/hidden_states/{model}/
gsm8k_abstention_role_construction/
  expert_{size}.h5
  non_expert_{size}.h5
  manifest.json  (ONE experiment-level manifest, same shape as the
  300-question script's manifest.json, with experiment field
  "gsm8k_abstention_role_construction")

@author: GSM8K abstention-enabled role hidden-state extraction, construction
split (2026-09-17)
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

SCRIPT_VERSION = "extract_gsm8k_abstention_role_hs_construction-v1"

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

CONDITIONS = ("expert", "non_expert")

ABSTENTION_ROLE_TO_CHARACTER = {
    "expert": "expert",
    "non_expert": "non expert",
}

ABSTENTION_SENTENCE = (
    'If you are not sure, you may answer exactly "I am not sure" instead.'
)

EXPERIMENT_NAME = "gsm8k_abstention_role_construction"


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------
# Prompt construction -- byte-identical logic to
# extract_gsm8k_abstention_role_hs.py, duplicated (not imported) to keep
# this script standalone.
# --------------------------------------------------------------------------

def build_abstention_templates(wording: str = "plain") -> dict:
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
        if "{context}" not in out[role_key]:
            die(f"internal error: {{context}} placeholder lost while "
                f"building the abstention template for role={role_key}.")
    return out


def assert_no_other_diff_than_role_wording(old_template: str,
                                           new_template: str) -> None:
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
# Fail-closed checks -- same logic as the 300-question script
# --------------------------------------------------------------------------

def tokenize_single(vc, text: str):
    return vc.tokenizer(text, return_tensors="pt", add_special_tokens=True)


def assert_no_double_bos(vc, input_ids_row) -> None:
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    ids = input_ids_row.tolist()
    if bos_id is not None and len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
        die(f"double BOS detected (head={ids[:4]}) -- refusing to proceed.")


def ordered_identity_list(samples: list) -> list:
    return [s["question"] for s in samples]


def sha256_of_list(items: list) -> str:
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
# Per-condition forward-pass extraction -- same mechanics as the
# 300-question script, generalized to variable n.
# --------------------------------------------------------------------------

def process_condition(vc, model_key, model_dir, size, condition, samples,
                      question_ids, templates, out_path, n_layers_out,
                      n_decoder, device_note, fixed_final_tok):
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
            meta["experiment"] = EXPERIMENT_NAME
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
        "h5_sha256": None,
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
    split_file = args.split_file or os.path.join(
        base_dir, "benchmark", "gsm8k_construction_split.json")
    split_manifest_file = args.split_manifest_file or os.path.join(
        base_dir, "benchmark", "gsm8k_construction_split_manifest.json")
    out_dir = args.out_dir or os.path.join(
        base_dir, "hidden_states", model_key, EXPERIMENT_NAME)

    if not os.path.isfile(split_file):
        die(f"GSM8K construction split file not found: {split_file} "
            "(build it first with build_gsm8k_construction_split.py)")
    if not os.path.isfile(split_manifest_file):
        die(f"GSM8K construction split manifest not found: "
            f"{split_manifest_file} (build it first with "
            "build_gsm8k_construction_split.py)")

    with open(split_manifest_file, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    samples = utils.load_json(split_file)
    n = len(samples)
    if n <= 0:
        die(f"gsm8k construction split: loaded {n} samples, expected > 0.")

    expected_n_from_manifest = split_manifest.get("construction_split_n")
    if expected_n_from_manifest is None or int(expected_n_from_manifest) != n:
        die(f"gsm8k construction split: loaded {n} samples from "
            f"{split_file}, but its own manifest "
            f"{split_manifest_file} records "
            f"construction_split_n={expected_n_from_manifest} -- refusing "
            "to proceed on a file/manifest mismatch.")

    question_ids = [s["question"] for s in samples]
    questions_file_sha256 = sha256_of_file(split_file)
    expected_file_sha256 = split_manifest.get(
        "construction_split_output", {}).get("sha256")
    if expected_file_sha256 and questions_file_sha256 != expected_file_sha256:
        die(f"{split_file}: sha256={questions_file_sha256} does not match "
            f"its own manifest's recorded sha256={expected_file_sha256} -- "
            "the split file may have been modified since it was built. "
            "Refusing to proceed.")

    pairing_digest = sha256_of_list(ordered_identity_list(samples))
    expected_pairing_digest = split_manifest.get(
        "construction_split_ordered_sample_identity_sha256")
    if not expected_pairing_digest or pairing_digest != expected_pairing_digest:
        die("ordered_sample_identity_sha256 for the loaded construction "
            f"split is {pairing_digest}, but its own manifest "
            f"{split_manifest_file} records {expected_pairing_digest!r}. "
            "Refusing to proceed -- this indicates the split file has "
            "drifted from what build_gsm8k_construction_split.py actually "
            "produced.")

    templates = build_abstention_templates(wording=args.wording)

    manifest_path = experiment_manifest_path(out_dir)
    expert_out = h5_path(out_dir, "expert", size)
    nonexpert_out = h5_path(out_dir, "non_expert", size)

    existing = [p for p in (expert_out, nonexpert_out, manifest_path)
               if os.path.exists(p)]
    if existing:
        die(f"refusing to run model={model_key}: target file(s) already "
            "exist:\n  " + "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    print("=" * 60)
    print(f"GSM8K construction split: n={n}, source={split_file}")
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
        print(f"split_file={split_file}")
        print(f"n_samples={n}")
        print(f"questions_file_sha256={questions_file_sha256}")
        print(f"ordered_sample_identity_sha256={pairing_digest}")
        return

    print(f"Loading model ONCE: model={model_key} model_dir={model_dir}")
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()
    n_layers_out, n_decoder = model_shape_lock(vc, mcfg, model_key, model_dir)
    device_note = build_device_note(vc)

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

    manifest = {
        "script_version": SCRIPT_VERSION,
        "git_commit": git_commit(),
        "experiment": EXPERIMENT_NAME,
        "model": model_key,
        "model_dir": model_dir,
        "model_revision": "unpinned, uses default HF revision",
        "task": "gsm8k",
        "wording": args.wording,
        "n_samples": n,
        "split_file": split_file,
        "split_manifest_file": split_manifest_file,
        "questions_file_sha256": questions_file_sha256,
        "ordered_sample_identity_sha256": pairing_digests["expert"],
        "excluded_300_note": (
            "This construction split EXCLUDES the 300-question benchmark "
            "(benchmark/gsm8k_test_sample.json), which is reserved for "
            "downstream steering/behavioral evaluation and does NOT "
            "participate in this ARRSN construction. See "
            "split_manifest_file for the exact exclusion provenance."
        ),
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
        "hidden_state_shape": [n, n_layers_out,
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
            "GSM8K-only, honest + abstention-enabled Role HS extraction on "
            "the LARGER construction split (full test minus the frozen "
            "300-question benchmark). Layer band / NMD mask / RRSN-MRSN "
            "similarity analysis are OUT OF SCOPE for this manifest and "
            "this run."
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
                    "hidden-state extraction on the LARGER construction "
                    "split (excludes the 300-question benchmark).")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--model_dir", default=None)
    p.add_argument("--base_dir", required=True,
                   help="components/ dir. Split file read from "
                        "<base_dir>/benchmark/gsm8k_construction_split.json "
                        "unless --split_file overrides it. Output goes to "
                        "<base_dir>/hidden_states/<model>/"
                        "gsm8k_abstention_role_construction/.")
    p.add_argument("--split_file", default=None,
                   help="Default: <base_dir>/benchmark/"
                        "gsm8k_construction_split.json")
    p.add_argument("--split_manifest_file", default=None,
                   help="Default: <base_dir>/benchmark/"
                        "gsm8k_construction_split_manifest.json")
    p.add_argument("--out_dir", default=None,
                   help="Default: <base_dir>/hidden_states/<model>/"
                        "gsm8k_abstention_role_construction/")
    p.add_argument("--wording", default="plain", choices=["plain"])
    p.add_argument("--verify_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
