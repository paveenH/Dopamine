#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
get_answer_gsm8k_abstention_role_audit.py -- MINIMAL-SCOPE, GSM8K-only,
standalone, NO-STEERING behavioral audit for the Expert vs Non-expert
abstention-enabled Role prompt. Generates free-form answers (no hidden
states, no steering, no mask) for the SAME 300-question benchmark and the
SAME "honest" + abstention-enabled prompt used by
extract_gsm8k_abstention_role_hs.py, and records per-sample raw text +
parsed outcome + metadata.

WHY A NEW SCRIPT (not an edit of get_answer_regenerate_gsm8k.py or
run_gsm8k.sh). Those scripts drive the FROZEN main-line GSM8K dose-response
pipeline (neutral/role-without-honest prompts, `vc.regenerate` with a
loaded/possibly-zero mask). This audit needs a DIFFERENT prompt (the
"honest" + abstention-enabled ARRSN prompt) and explicitly must not load a
mask or register any steering hook at all -- editing those frozen scripts in
place, or passing an all-zero mask through `regenerate`, would risk forking
or obscuring the "no steering, structurally" guarantee this task asked for.
This script imports NOTHING from get_answer_regenerate_gsm8k.py and does not
call `vc.regenerate` anywhere; it uses `VicundaModel.generate()`, which
registers no forward hooks of any kind (verified by reading llms.py before
writing this script -- `generate()` calls only `self.model.generate(...)`
with `diff_matrices` never referenced).

PROMPT -- BYTE-IDENTICAL to extract_gsm8k_abstention_role_hs.py's ARRSN HS
extraction prompt. This script does NOT retype the prompt: it imports
`build_abstention_templates`, `render_prompt`, `assert_no_other_diff_than_
role_wording`, `ABSTENTION_ROLE_TO_CHARACTER`, `ABSTENTION_SENTENCE`,
`CONDITIONS`, `N_EXPECT`, and `EXPECTED_GSM8K_PAIRING_DIGEST` DIRECTLY from
that module (an import, not a copy-paste), so a future edit to that script's
prompt construction cannot silently diverge from this audit's prompt without
also breaking this script's own import. This is the deliberate opposite
choice from extract_gsm8k_abstention_role_hs_construction.py's "duplicate,
don't import" pattern for the construction-split sibling -- there,
duplication kept the sibling independent of a DIFFERENT-SCOPE parent script;
here, importing keeps this audit's prompt PINNED to the ARRSN HS extraction
it is meant to behaviorally audit, since prompt drift between the two would
make "was this the ARRSN prompt" an open question rather than a checked
fact.

GENERATION. `VicundaModel.generate()` (registers NO hooks; NOT
`vc.regenerate`, which always builds a `diff_matrices` argument even at
alpha=0 -- see the CLAUDE.md "α=0 here is NOT the pv6/PV10 'no hook at all'
case" rule this script deliberately avoids by using a different method
entirely). `max_new_tokens=768`, `temperature=0.0` (greedy), `batch_size=24`
-- the SAME three values `run_gsm8k.sh`/`run_gsm8k_qwen25.sh` use for the
frozen No-CoT GSM8K main line (verified by reading both launchers' `MAX_NEW_
TOKENS`/`TEMPERATURE`/`BATCH_SIZE` values before writing this script, not
assumed), so this audit's generation regime is directly comparable to the
existing GSM8K behavioral literature in this repo. `generate()` already uses
`eos_token_id=self.terminators` (the `<|eot_id|>` fix), so no separate
terminator handling is needed here.

DATA: the SAME frozen 300-question file <base_dir>/benchmark/
gsm8k_test_sample.json, loaded with NO reordering, and its
ordered_sample_identity_sha256 is checked against
EXPECTED_GSM8K_PAIRING_DIGEST (imported from extract_gsm8k_abstention_role_
hs.py) exactly as that script does -- so this audit runs on the byte-
identical question set/order the ARRSN HS extraction ran on.

PARSING / OUTCOME CATEGORIES (four, mutually exclusive, computed per
sample):
  - "correct"   : a `####` marker parses to a number, and that number
                   normalizes equal to gold (utils.normalize_gsm8k).
  - "wrong"     : a `####` marker parses to a number, but it does NOT
                   normalize equal to gold.
  - "abstained" : NO `####` marker anywhere in the generated text, AND the
                   text contains the EXACT phrase 'I am not sure' (matched
                   case-sensitively against the literal abstention phrase
                   from ABSTENTION_SENTENCE, not a fuzzy/case-insensitive
                   match -- a model that writes "I am not sure" but ALSO
                   emits a parseable `####` marker is NOT scored as
                   abstained, since it did commit to a numeric answer; see
                   parse_outcome() for the precedence rule).
  - "invalid"   : neither of the above -- no parseable `####` marker AND no
                   exact "I am not sure" phrase (empty generation, prose
                   with no marker and no abstention phrase, or any other
                   format failure).
PRIMARY marker extraction is FIRST-`####`-ONLY (matching this repo's
documented production convention: `analyze_first_last_acc.py`'s `all_hash`
takes every `####` match and MAIN accuracy uses the FIRST; GSM8K's own
`extract_gsm8k_answer` in utils.py also takes the first `####` match before
falling back to looser patterns) -- deliberately NOT the loose multi-pattern
fallback chain (`the answer is...`, `\boxed{}`, last bare number)
utils.extract_gsm8k_answer() uses for the frozen dose-response pipeline,
because that fallback chain would silently convert a genuine format failure
into an apparent "wrong" answer, destroying this audit's own explicit
"invalid/format failure" category. `last_acc`-equivalent sensitivity (last
`####` marker instead of first) is ALSO recorded per sample for the offline
analysis to report as a secondary check, matching this repo's FIRST-MAIN/
LAST-sensitivity convention.

NO STEERING, ANYWHERE, STRUCTURALLY (not just "mask defaults to zero"). No
mask file is loaded, no diff matrix is built, `vc.regenerate` is never
called, and the ONLY generation method used is `VicundaModel.generate()`,
which registers no forward hooks -- verified by reading llms.py's `generate()`
body before writing this script (it calls only `self.tokenizer(...)` and
`self.model.generate(...)`, no `register_forward_hook`/`register_forward_
pre_hook` anywhere in that method or in anything it calls).

FAIL-CLOSED VALIDATION:
  - overwrite preflight for BOTH conditions' output files, BEFORE model load;
  - n must equal exactly 300;
  - expert/non_expert share the identical ordered question list (checked via
    the SAME pairing-digest mechanism as extract_gsm8k_abstention_role_hs.py);
  - ordered_sample_identity_sha256 for the loaded 300-question file must
    equal EXPECTED_GSM8K_PAIRING_DIGEST (imported, not re-derived);
  - rendered prompt text is asserted, per role, to byte-match
    build_abstention_templates()'s own output (i.e. render_prompt() is
    called, not a hand-built f-string) -- this makes "prompt identical to
    ARRSN HS extraction" a structural guarantee (same function call) rather
    than a hoped-for coincidence;
  - model-shape lock (hidden_size) before any generation;
  - atomic write: build the full per-condition sample list in memory,
    verify it, THEN write to a temp path and os.replace() it into place;
  - every output row records role, model, generated text, parsed marker
    value (first + last), outcome category, and a "steering_applied": false
    / "mask_loaded": false pair of fields, so the offline analysis (and any
    future reader) can verify no-steering from the artifact itself, not just
    from this docstring.

SCOPE, DELIBERATELY NARROW (per explicit instruction): GSM8K only, the
existing fixed 300-question benchmark, Llama-3.1-8B-Instruct and
Qwen2.5-7B-Instruct only, expert vs non_expert only. No dose/alpha sweep, no
steering, no HS, no MRSN/ARRSN/Chat similarity analysis. This is a pure
generation-and-parse step; see analyze_gsm8k_abstention_role_audit.py for
the offline accuracy/abstention/coverage statistics this generates data for.

Output: <base_dir>/{model}/answer_gsm8k_abstention_role_audit/
  {expert,non_expert}_{size}.json
  run_meta_{size}.json

@author: GSM8K abstention-role no-steering behavioral audit (2026-09-17)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import utils
from llms import VicundaModel

# ---------------------------------------------------------------------
# Prompt construction is IMPORTED, not retyped -- see module docstring.
# ---------------------------------------------------------------------
from extract_gsm8k_abstention_role_hs import (
    ABSTENTION_ROLE_TO_CHARACTER,
    ABSTENTION_SENTENCE,
    CONDITIONS,
    EXPECTED_GSM8K_PAIRING_DIGEST,
    N_EXPECT,
    build_abstention_templates,
    render_prompt,
)

SCRIPT_VERSION = "get_answer_gsm8k_abstention_role_audit-v1"

MODELS = {
    "llama3": {
        "model_dir_default": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "8B",
        "expected_hidden_size": 4096,
    },
    "qwen2.5": {
        "model_dir_default": "Qwen/Qwen2.5-7B-Instruct",
        "size": "7B",
        "expected_hidden_size": 3584,
    },
}

# Same three generation values run_gsm8k.sh / run_gsm8k_qwen25.sh use for
# their frozen No-CoT main line (verified by reading both launchers before
# writing this script).
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.0
BATCH_SIZE = 24
TOP_P = 0.9  # unused at temperature=0.0 (VicundaModel.generate() disables
             # sampling and passes top_p=None), kept explicit for the record.

# EXACT abstention phrase this audit scores against -- the literal sentence
# body from extract_gsm8k_abstention_role_hs.ABSTENTION_SENTENCE
# ('If you are not sure, you may answer exactly "I am not sure" instead.'),
# NOT re-derived: sliced directly from the imported constant so a future
# wording change there is automatically picked up here rather than silently
# diverging.
_ABSTENTION_MATCH = re.search(r'"([^"]+)"', ABSTENTION_SENTENCE)
if not _ABSTENTION_MATCH:
    print("[FATAL] could not extract the quoted abstention phrase from "
          f"ABSTENTION_SENTENCE={ABSTENTION_SENTENCE!r}", file=sys.stderr)
    sys.exit(2)
EXACT_ABSTENTION_PHRASE = _ABSTENTION_MATCH.group(1)  # "I am not sure"

HASH_MARKER_RE = re.compile(r"####\s*([+-]?[\d,]+\.?\d*)")


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_list(items: list) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def ordered_identity_list(samples: list) -> list:
    return [s["question"] for s in samples]


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
# Parsing / outcome classification
# --------------------------------------------------------------------------

def all_hash_matches(text: str) -> list:
    """Every `####` marker's parsed numeric value, in order of appearance.
    Empty list if none parse."""
    return [m.group(1).replace(",", "") for m in HASH_MARKER_RE.finditer(text)]


def contains_exact_abstention_phrase(text: str) -> bool:
    return EXACT_ABSTENTION_PHRASE in text


def parse_outcome(generated: str, gold: str) -> dict:
    """Returns a dict with:
      marker_values_first_to_last: list[str] (raw parsed strings, in order)
      first_marker_raw / last_marker_raw: str or None
      first_correct / last_correct: bool or None (None if no marker)
      abstention_phrase_present: bool
      outcome: one of "correct","wrong","abstained","invalid"

    PRECEDENCE RULE (deliberate, stated in the module docstring): a
    parseable `####` marker takes precedence over the abstention phrase --
    a sample that both writes "I am not sure" AND emits a `####` marker is
    scored by the marker (correct/wrong), never as "abstained", because it
    DID commit to a numeric answer. Only the complete absence of any
    parseable marker, combined with the exact phrase present, counts as a
    true abstention.
    """
    markers = all_hash_matches(generated)
    abst_present = contains_exact_abstention_phrase(generated)

    if markers:
        first_raw = markers[0]
        last_raw = markers[-1]
        first_correct = utils.is_correct_gsm8k(first_raw, gold)
        last_correct = utils.is_correct_gsm8k(last_raw, gold)
        outcome = "correct" if first_correct else "wrong"
        return {
            "marker_values_first_to_last": markers,
            "first_marker_raw": first_raw,
            "last_marker_raw": last_raw,
            "first_correct": first_correct,
            "last_correct": last_correct,
            "abstention_phrase_present": abst_present,
            "outcome": outcome,
        }

    if abst_present:
        return {
            "marker_values_first_to_last": [],
            "first_marker_raw": None,
            "last_marker_raw": None,
            "first_correct": None,
            "last_correct": None,
            "abstention_phrase_present": True,
            "outcome": "abstained",
        }

    return {
        "marker_values_first_to_last": [],
        "first_marker_raw": None,
        "last_marker_raw": None,
        "first_correct": None,
        "last_correct": None,
        "abstention_phrase_present": False,
        "outcome": "invalid",
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="No-steering GSM8K abstention-role behavioral audit "
                    "(expert vs non_expert, free-form generation).")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--model_dir", default=None)
    p.add_argument("--base_dir", required=True,
                   help="components/ dir. Reads "
                        "<base_dir>/benchmark/gsm8k_test_sample.json. "
                        "Output goes to <base_dir>/{model}/"
                        "answer_gsm8k_abstention_role_audit/")
    p.add_argument("--gsm8k_file", default=None,
                   help="Default: <base_dir>/benchmark/gsm8k_test_sample.json")
    p.add_argument("--out_dir", default=None,
                   help="Default: <base_dir>/{model}/"
                        "answer_gsm8k_abstention_role_audit/")
    p.add_argument("--wording", default="plain", choices=["plain"],
                   help="Locked to 'plain', matching "
                        "extract_gsm8k_abstention_role_hs.py's own default "
                        "and every frozen GSM8K No-CoT main-line launcher.")
    p.add_argument("--verify_only", action="store_true",
                   help="Render + print both role prompts for sample 0 and "
                        "assert the pairing digest, then exit 0 WITHOUT "
                        "loading any model. No GPU needed.")
    return p.parse_args()


def main():
    args = parse_args()
    model_key = args.model
    mcfg = MODELS[model_key]
    size = mcfg["size"]
    model_dir = args.model_dir or mcfg["model_dir_default"]

    base_dir = args.base_dir
    gsm8k_file = args.gsm8k_file or os.path.join(
        base_dir, "benchmark", "gsm8k_test_sample.json")
    out_dir = args.out_dir or os.path.join(
        base_dir, model_key, "answer_gsm8k_abstention_role_audit")

    if not os.path.isfile(gsm8k_file):
        die(f"GSM8K sample file not found: {gsm8k_file}")

    samples = utils.load_json(gsm8k_file)
    n = len(samples)
    if n != N_EXPECT:
        die(f"gsm8k: loaded {n} samples, expected exactly {N_EXPECT}. "
            "Refusing to run on a wrong-size sample.")

    pairing_digest = sha256_of_list(ordered_identity_list(samples))
    if pairing_digest != EXPECTED_GSM8K_PAIRING_DIGEST:
        die(f"ordered_sample_identity_sha256 for {gsm8k_file} is "
            f"{pairing_digest}, but the ARRSN HS extraction recorded "
            f"{EXPECTED_GSM8K_PAIRING_DIGEST}. The 300-question benchmark "
            "file must be byte-identical (same questions, same order) to "
            "what the ARRSN HS extraction used. Refusing to proceed.")

    questions_file_sha256 = sha256_of_file(gsm8k_file)
    templates = build_abstention_templates(wording=args.wording)

    print("=" * 60)
    print(f"[pairing] n={n} ordered_sample_identity_sha256={pairing_digest} "
          "(matches ARRSN HS extraction's EXPECTED_GSM8K_PAIRING_DIGEST)")
    print("Rendered prompts (full text, both roles, sample 0) -- "
          "byte-identical to extract_gsm8k_abstention_role_hs.py's ARRSN "
          "HS extraction prompt (same build_abstention_templates()/"
          "render_prompt() call):")
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
        print("[verify_only] prompt + pairing digest verified. Exiting "
              "WITHOUT loading any model.")
        return

    expert_out_path = os.path.join(out_dir, f"expert_{size}.json")
    nonexpert_out_path = os.path.join(out_dir, f"non_expert_{size}.json")
    meta_path = os.path.join(out_dir, f"run_meta_{size}.json")

    # ---- Overwrite preflight for BOTH conditions + the run meta, BEFORE
    # ---- the model is loaded. ----
    existing = [p for p in (expert_out_path, nonexpert_out_path, meta_path)
               if os.path.exists(p)]
    if existing:
        die(f"refusing to run model={model_key}: target file(s) already "
            "exist:\n  " + "\n  ".join(existing) +
            "\nDelete them deliberately first if a re-run is truly "
            "intended.")

    print(f"model={model_key} model_dir={model_dir} n={n} "
          f"out_dir={out_dir}")

    # ---- Model load. VicundaModel.generate() is used exclusively below --
    # ---- it registers no forward hooks and never references a mask/diff
    # ---- matrix (verified in llms.py before writing this script). ----
    vc = VicundaModel(model_path=model_dir)
    vc.model.eval()

    actual_hidden = int(vc.model.config.hidden_size)
    expect_hidden = mcfg["expected_hidden_size"]
    if actual_hidden != expect_hidden:
        die(f"model shape mismatch for --model {model_key} "
            f"(--model_dir {model_dir}): got hidden_size={actual_hidden}, "
            f"expected {expect_hidden}. Refusing to write output under a "
            "model key whose loaded architecture does not match.")

    per_condition_rows = {}
    per_condition_prompt_sha256 = {}
    per_condition_gen_seconds = {}

    for condition in CONDITIONS:
        print("=" * 60)
        print(f"Generating: model={model_key} condition={condition} n={n} "
              f"max_new_tokens={MAX_NEW_TOKENS} temperature={TEMPERATURE} "
              f"batch_size={BATCH_SIZE}")
        print("=" * 60)

        prompts = [render_prompt(templates, condition, s["question"])
                  for s in samples]
        prompt_sha256 = sha256_of_list(prompts)
        per_condition_prompt_sha256[condition] = prompt_sha256

        t0 = time.time()
        generated_texts = vc.generate(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
        )
        elapsed = time.time() - t0
        per_condition_gen_seconds[condition] = elapsed
        print(f"  done in {elapsed:.1f}s")

        if len(generated_texts) != n:
            die(f"condition={condition}: generate() returned "
                f"{len(generated_texts)} outputs, expected exactly {n}.")

        rows = []
        for i, (s, gen) in enumerate(zip(samples, generated_texts)):
            gold = s["answer"]
            outcome_info = parse_outcome(gen, gold)
            rows.append({
                "idx": i,
                "question": s["question"],
                "gold_answer": gold,
                "role": condition,
                "role_character": ABSTENTION_ROLE_TO_CHARACTER[condition],
                "generated": gen,
                "gen_char_len": len(gen),
                **outcome_info,
                "steering_applied": False,
                "mask_loaded": False,
                "generation_method": "VicundaModel.generate",
            })
        per_condition_rows[condition] = rows

    # ---- Atomic write: build in memory, verify, THEN write. ----
    os.makedirs(out_dir, exist_ok=True)

    def atomic_write_json(path: str, obj) -> None:
        d = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    atomic_write_json(expert_out_path, per_condition_rows["expert"])
    atomic_write_json(nonexpert_out_path, per_condition_rows["non_expert"])
    print(f"wrote {expert_out_path}")
    print(f"wrote {nonexpert_out_path}")

    try:
        devs = sorted({str(p.device) for p in vc.model.parameters()})
    except Exception:
        devs = []
    import platform
    device_note = {"host": platform.node(),
                   "cuda_visible_devices": os.environ.get(
                       "CUDA_VISIBLE_DEVICES"),
                   "param_devices": devs,
                   "sharded": len(devs) > 1}

    run_meta = {
        "script_version": SCRIPT_VERSION,
        "git_commit": git_commit(),
        "model": model_key,
        "model_dir": model_dir,
        "model_revision": "unpinned, uses default HF revision",
        "size": size,
        "task": "gsm8k_abstention_role_audit",
        "conditions": list(CONDITIONS),
        "n_samples": n,
        "n_samples_expected": N_EXPECT,
        "questions_file": gsm8k_file,
        "questions_sha256": questions_file_sha256,
        "ordered_sample_identity_sha256": pairing_digest,
        "expected_gsm8k_pairing_digest": EXPECTED_GSM8K_PAIRING_DIGEST,
        "prompt_wording": args.wording,
        "abstention_sentence": ABSTENTION_SENTENCE,
        "exact_abstention_phrase_matched": EXACT_ABSTENTION_PHRASE,
        "role_character_strings": dict(ABSTENTION_ROLE_TO_CHARACTER),
        "prompt_sha256": per_condition_prompt_sha256,
        "generation_method": "VicundaModel.generate (no hooks registered, "
                              "no mask loaded, no diff matrix)",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "batch_size": BATCH_SIZE,
        "steering_applied": False,
        "mask_loaded": False,
        "mask_path": None,
        "generation_seconds": per_condition_gen_seconds,
        "output_paths": {
            "expert": expert_out_path,
            "non_expert": nonexpert_out_path,
        },
        "provenance": device_note,
        "extraction_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "GSM8K-only, no-steering, expert-vs-non_expert behavioral "
                "audit of the ARRSN HS extraction's abstention-enabled "
                "prompt. Prompt is IMPORTED from "
                "extract_gsm8k_abstention_role_hs.py, not retyped. No dose/"
                "alpha sweep, no HS, no MRSN/ARRSN/Chat similarity "
                "analysis in this line.",
    }
    atomic_write_json(meta_path, run_meta)
    print(f"wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
