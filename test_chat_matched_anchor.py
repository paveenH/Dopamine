#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_chat_matched_anchor.py -- local, no-GPU test suite for
chat_matched_anchor_lib.py and the two matched-anchor generators. Exits
non-zero on failure (see main()'s exit code), prints its own [ok]/[FAIL]
lines.

THREE GROUPS, deliberately separated:

Group A (FAKE TOKENIZER, always runs, zero dependencies beyond the standard
library): exercises every fail-closed branch in chat_matched_anchor_lib.py's
anchor/BOS/tail-token machinery directly, using a minimal fake object
exposing the same apply_chat_template/__call__/decode/bos_token(_id)/
chat_template surface a real HF tokenizer does. This tests the ASSERTION
LOGIC itself -- both the pass path and every failure path -- independent of
whether the real tokenizer happens to be cached on this machine.

Group B (REAL TOKENIZER, runs only if
meta-llama/Llama-3.1-8B-Instruct's tokenizer is available in the local HF
cache -- loaded with local_files_only=True, so this NEVER hits the network
and NEVER loads model weights or a GPU): exercises the FULL real mechanism
end to end on the actual MATH and GSM8K prompt templates, and empirically
reproduces the three-way tail-token comparison this whole experiment is
built on:
    bare            -> id 220 ' '     (the "Answer: " anchor)
    native chat     -> id 271 '\\n\\n' (the assistant header, un-anchored)
    matched-anchor  -> id 220 ' '     (this experiment's mechanism)
If the tokenizer is not cached, Group B prints a clear SKIP notice (not a
silent no-op, and not a failure) and the suite's exit code depends only on
Groups A and C.

Group C (ALPHA-FAMILY VALIDATION, always runs, zero dependencies): exercises
chat_matched_anchor_lib.assert_alpha_family() -- the SAME function both
get_answer_math_chat_matched_anchor.py and get_answer_gsm8k_chat_matched_
anchor.py call for their `--configs` validation (refactored out of each
generator's main() into this one shared function precisely so it is
directly testable and cannot silently drift between the two tasks). Covers
the full nine-point family {-8,-6,-4,-2,0,2,4,6,8}: a missing dose, a
duplicate dose, an extra dose, and a float/off-grid dose must all be
rejected; a shuffled (reordered) but otherwise complete nine-point family
must be accepted.

No GPU is used anywhere in this file. No model weights are loaded. No
network access is required for Groups A and C; Group B only ever reads from
an existing local HF cache with local_files_only=True.
"""

from __future__ import annotations

import random
import sys

import chat_matched_anchor_lib as CMA

FAILURES = []


def check(label, cond):
    if cond:
        print(f"[ok] {label}")
    else:
        print(f"[FAIL] {label}")
        FAILURES.append(label)


def expect_die(label, fn, *args, **kwargs):
    """CMA.die() calls sys.exit(2). Catch it and confirm it actually fired,
    rather than the call succeeding silently."""
    try:
        fn(*args, **kwargs)
    except SystemExit as e:
        check(f"{label} (died with code {e.code})", e.code == 2)
    else:
        check(f"{label} (expected die(), got no exception)", False)


def expect_ok(label, fn, *args, **kwargs):
    """The inverse: the call must NOT call die()/sys.exit."""
    try:
        result = fn(*args, **kwargs)
    except SystemExit as e:
        check(f"{label} (unexpectedly died with code {e.code})", False)
        return None
    else:
        check(f"{label} (no exception)", True)
        return result


# ============================================================ Group A ======
class FakeTokenizer:
    """Mimics only the surface chat_matched_anchor_lib.py touches:
    apply_chat_template, __call__(text, add_special_tokens=True), decode,
    bos_token, bos_token_id, chat_template. Tokenization is a crude,
    deterministic stand-in -- it is NOT meant to be BPE-realistic (Group B
    covers realism); it exists only to drive every branch of the assertion
    functions under test."""

    def __init__(self):
        self.bos_token = "<BOS>"
        self.bos_token_id = 999
        self.chat_template = "FAKE_TEMPLATE"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        content = messages[0]["content"]
        # Mirrors a real chat template: serializes its own BOS text, wraps
        # the (untrimmed -- this fake does not simulate Jinja |trim, since
        # chat_matched_anchor_lib.py's whole point is to control the anchor
        # BEFORE wrapping) content, and appends an assistant header.
        return (f"{self.bos_token}<|start_header_id|>user"
                f"<|end_header_id|>\n\n{content}<|eot_id|>"
                f"{CMA.ASSISTANT_HEADER}\n\n")

    def __call__(self, text, add_special_tokens=True):
        ids = [self.bos_token_id] if add_special_tokens else []
        # Simulate the real double-BOS hazard: if the text ITSELF still
        # starts with the serialized BOS string (i.e. strip_leading_bos was
        # not applied), the tokenizer would encode that leading BOS text as
        # another bos_id right after the one add_special_tokens adds.
        if text.startswith(self.bos_token):
            ids.append(self.bos_token_id)
        ids += [1, 2]
        if text.endswith(" ") and not text.endswith("  "):
            ids.append(220)          # single trailing ASCII space
        elif text.endswith("!"):
            ids.append(777)
        else:
            ids.append(333)
        return {"input_ids": ids}

    def decode(self, ids):
        table = {220: " ", 777: "!", 333: "X", self.bos_token_id: "<BOS>"}
        return "".join(table.get(i, f"<{i}>") for i in ids)


class FakeVC:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


def run_group_a():
    print("\n=== Group A: fake tokenizer, assertion-logic coverage ===")
    vc = FakeVC()

    # --- strip_trailing_answer_anchor ---------------------------------------
    got = expect_ok(
        "strip_trailing_answer_anchor: well-formed body",
        CMA.strip_trailing_answer_anchor,
        "Solve the following math problem.\nQuestion: 2+2?\nAnswer: ",
        "unit-test")
    check("strip_trailing_answer_anchor: correct suffix removed",
          got == "Solve the following math problem.\nQuestion: 2+2?\n")

    expect_die(
        "strip_trailing_answer_anchor: body NOT ending in the anchor",
        CMA.strip_trailing_answer_anchor,
        "Solve the following math problem.\nQuestion: 2+2?\nFoo: ",
        "unit-test")

    expect_die(
        "strip_trailing_answer_anchor: anchor present but not at the very end",
        CMA.strip_trailing_answer_anchor,
        "Answer: is a word that appears mid-body, not at the end.",
        "unit-test")

    # --- build_matched_anchor_prompt + no-double-BOS ------------------------
    body_no_anchor = "Solve the following math problem.\nQuestion: 2+2?\n"
    wrapped = expect_ok(
        "build_matched_anchor_prompt: happy path",
        CMA.build_matched_anchor_prompt, vc, body_no_anchor)
    check("build_matched_anchor_prompt: leading BOS text stripped",
          isinstance(wrapped, str) and not wrapped.startswith(vc.tokenizer.bos_token))
    check("build_matched_anchor_prompt: assistant header present",
          isinstance(wrapped, str) and CMA.ASSISTANT_HEADER in wrapped)

    expect_ok(
        "assert_no_double_bos: correctly-stripped text passes",
        CMA.assert_no_double_bos, vc, wrapped, "unit-test")

    unstripped = vc.tokenizer.apply_chat_template(
        [{"role": "user", "content": body_no_anchor}],
        tokenize=False, add_generation_prompt=True)
    expect_die(
        "assert_no_double_bos: un-stripped text (still has leading BOS) dies",
        CMA.assert_no_double_bos, vc, unstripped, "unit-test")

    # --- assert_chat_template_effective --------------------------------------
    expect_ok(
        "assert_chat_template_effective: happy path",
        CMA.assert_chat_template_effective, vc, body_no_anchor, wrapped)
    expect_die(
        "assert_chat_template_effective: wrapped == reference (no-op template)",
        CMA.assert_chat_template_effective, vc, body_no_anchor, body_no_anchor)
    expect_die(
        "assert_chat_template_effective: missing assistant header",
        CMA.assert_chat_template_effective, vc, body_no_anchor,
        "some wrapped text with no header at all")

    class NoTemplateTokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.chat_template = None

    class NoTemplateVC:
        def __init__(self):
            self.tokenizer = NoTemplateTokenizer()

    expect_die(
        "assert_chat_template_effective: chat_template is None",
        CMA.assert_chat_template_effective, NoTemplateVC(), body_no_anchor,
        wrapped)

    # --- the full matched-anchor assembly + tail assertion ------------------
    final_text = wrapped + CMA.ANSWER_ANCHOR
    check("matched-anchor final text ends in the literal anchor",
          final_text.endswith(CMA.ANSWER_ANCHOR))

    tail = expect_ok(
        "assert_matched_anchor_tail: happy path (single trailing space)",
        CMA.assert_matched_anchor_tail, vc, final_text, "unit-test")
    check("assert_matched_anchor_tail: returned id == 220",
          tail is not None and tail["id"] == 220)
    check("assert_matched_anchor_tail: returned text == single space",
          tail is not None and tail["text"] == " ")

    # Failure path: text does NOT end in a single ASCII space.
    bad_text = wrapped + "Answer:"    # no trailing space
    expect_die(
        "assert_matched_anchor_tail: missing trailing space dies",
        CMA.assert_matched_anchor_tail, vc, bad_text, "unit-test")

    bad_text2 = wrapped + "Answer! "
    # ends in a space too, so the FakeTokenizer's crude rule would still map
    # to 220 unless the preceding char changes the story -- use a text that
    # ends in "!" instead to hit the fake's 777 branch cleanly.
    bad_text3 = wrapped + "Answer!"
    expect_die(
        "assert_matched_anchor_tail: wrong token id (fake '!' branch) dies",
        CMA.assert_matched_anchor_tail, vc, bad_text3, "unit-test")

    # --- tail_ids ------------------------------------------------------------
    tids = CMA.tail_ids(vc, final_text, k=3)
    check("tail_ids: returns k entries with id/text", len(tids) == 3
          and all("id" in t and "text" in t for t in tids))
    check("tail_ids: last entry matches assert_matched_anchor_tail",
          tids[-1]["id"] == 220 and tids[-1]["text"] == " ")


# ============================================================ Group B ======
def run_group_b():
    print("\n=== Group B: real Llama-3.1-8B-Instruct tokenizer (CPU-only, "
          "local cache) ===")
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"[skip] transformers not importable ({e!r}); Group B skipped, "
              "not counted as failure.")
        return

    try:
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)
    except Exception as e:
        print(f"[skip] meta-llama/Llama-3.1-8B-Instruct tokenizer is not in "
              f"the local HF cache ({type(e).__name__}: {e}); Group B "
              "skipped, not counted as failure. This is expected on a "
              "machine that has never downloaded this tokenizer, and does "
              "NOT indicate a problem with chat_matched_anchor_lib.py.")
        return

    if tok.padding_side != "left":
        # Not fatal for this tokenizer-only smoke test (padding side matters
        # for batched generation, not for a single-sequence tail check), but
        # worth a note since the real launchers assert this.
        print(f"[note] tokenizer.padding_side={tok.padding_side!r} (the real "
              "generator scripts set/require 'left'; irrelevant to this "
              "single-sequence tail check).")

    class RealVC:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    vc = RealVC(tok)

    from template import build_math_suite, build_gsm8k_default_suite

    question = "What is 2 + 2?"
    math_body = build_math_suite(cot=False)["neutral"].format(context=question)
    gsm8k_body = build_gsm8k_default_suite(
        cot=False, wording="plain")["neutral"].format(context=question)

    for task_name, body in (("MATH", math_body), ("GSM8K", gsm8k_body)):
        print(f"\n--- {task_name} ---")
        check(f"{task_name}: rendered body ends with the literal anchor",
              body.endswith(CMA.ANSWER_ANCHOR))

        # (i) BARE reference: tokenize the body AS-IS (anchor kept, no chat
        # wrapping at all) -- this is exactly what the frozen bare-string
        # generators do.
        bare_tail = CMA.tail_ids(vc, body, k=1)[0]
        print(f"{task_name} bare tail: {bare_tail}")
        check(f"{task_name}: bare condition's last token is id 220 ' '",
              bare_tail["id"] == 220 and bare_tail["text"] == " ")

        # (ii) NATIVE CHAT reference: wrap the body WITH the anchor still
        # inside the user turn (i.e. do NOT strip it first) -- this
        # reproduces get_answer_{math,gsm8k}_chat_sweep.py's own construction
        # exactly, so this test independently re-verifies their documented
        # confound on the real tokenizer, not just by citation.
        native_wrapped = tok.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True)
        native_wrapped = CMA.strip_leading_bos(vc, native_wrapped)
        native_tail = CMA.tail_ids(vc, native_wrapped, k=1)[0]
        print(f"{task_name} native-chat tail: {native_tail}")
        check(f"{task_name}: native-chat condition's last token is id 271 "
              "'\\n\\n' (the documented trim-induced shift)",
              native_tail["id"] == 271 and native_tail["text"] == "\n\n")

        # (iii) MATCHED-ANCHOR: this experiment's actual mechanism.
        body_no_anchor = expect_ok(
            f"{task_name}: strip_trailing_answer_anchor",
            CMA.strip_trailing_answer_anchor, body, task_name)
        wrapped = expect_ok(
            f"{task_name}: build_matched_anchor_prompt",
            CMA.build_matched_anchor_prompt, vc, body_no_anchor)
        expect_ok(
            f"{task_name}: assert_chat_template_effective",
            CMA.assert_chat_template_effective, vc, body_no_anchor, wrapped)
        expect_ok(
            f"{task_name}: assert_no_double_bos on matched-anchor wrapped text",
            CMA.assert_no_double_bos, vc, wrapped, task_name)
        final_text = wrapped + CMA.ANSWER_ANCHOR
        matched_tail = expect_ok(
            f"{task_name}: assert_matched_anchor_tail (the load-bearing gate)",
            CMA.assert_matched_anchor_tail, vc, final_text, task_name)
        print(f"{task_name} matched-anchor tail: {matched_tail}")
        check(f"{task_name}: matched-anchor last token == bare last token "
              "(id 220, single ASCII space) -- the experiment's whole point",
              matched_tail is not None
              and matched_tail["id"] == bare_tail["id"] == 220
              and matched_tail["text"] == bare_tail["text"] == " ")
        check(f"{task_name}: matched-anchor last token != native-chat last "
              "token (271) -- confirms the anchor genuinely moved, not a "
              "no-op",
              matched_tail is not None and matched_tail["id"] != native_tail["id"])

        print(f"{task_name} rendered matched-anchor prompt, last 60 chars:")
        print(f"  {final_text[-60:]!r}")


# ============================================================ Group C ======
NINE_POINT = [-8, -6, -4, -2, 0, 2, 4, 6, 8]


def run_group_c():
    print("\n=== Group C: alpha-family validation (shared by both "
          "generators) ===")

    # --- happy path: exact nine-point family, in order -----------------------
    expect_ok(
        "assert_alpha_family: exact nine-point family, in order",
        CMA.assert_alpha_family, list(NINE_POINT), NINE_POINT, "test-protocol")

    # --- happy path: SHUFFLED order must still be accepted --------------------
    shuffled = list(NINE_POINT)
    rng = random.Random(0)
    rng.shuffle(shuffled)
    check("assert_alpha_family: shuffled order differs from NINE_POINT "
          "(sanity check on the test fixture itself)",
          shuffled != NINE_POINT)
    expect_ok(
        "assert_alpha_family: shuffled nine-point family is ACCEPTED "
        "(order-independent)",
        CMA.assert_alpha_family, shuffled, NINE_POINT, "test-protocol")

    # --- failure: missing one dose --------------------------------------------
    missing_one = [a for a in NINE_POINT if a != -6]
    expect_die(
        "assert_alpha_family: missing dose (-6 absent) dies",
        CMA.assert_alpha_family, missing_one, NINE_POINT, "test-protocol")

    # --- failure: duplicate dose (same length as the real family, but one
    # dose repeated and another therefore absent) -----------------------------
    duplicated = [a for a in NINE_POINT if a != 8] + [0]
    check("assert_alpha_family: duplicated fixture has the same length as "
          "the real family (sanity check on the test fixture itself)",
          len(duplicated) == len(NINE_POINT))
    expect_die(
        "assert_alpha_family: duplicate dose (0 twice, +8 absent) dies",
        CMA.assert_alpha_family, duplicated, NINE_POINT, "test-protocol")

    # --- failure: extra dose (all nine plus one more) -------------------------
    extra = list(NINE_POINT) + [10]
    expect_die(
        "assert_alpha_family: extra dose (+10 appended) dies",
        CMA.assert_alpha_family, extra, NINE_POINT, "test-protocol")

    # --- failure: float / off-grid dose ----------------------------------------
    float_dose = [a for a in NINE_POINT if a != 2] + [2.5]
    expect_die(
        "assert_alpha_family: float/off-grid dose (2.5 instead of 2) dies",
        CMA.assert_alpha_family, float_dose, NINE_POINT, "test-protocol")

    # --- the exact real-generator invocation shape, both tasks -----------------
    # Mirrors get_answer_math_chat_matched_anchor.py's / get_answer_gsm8k_
    # chat_matched_anchor.py's actual call: EXPECTED_ALPHAS is a set, and the
    # generators pass `got_alphas = [al for al, _ in cfgs]` (unsorted, since
    # sorting happens INSIDE assert_alpha_family now).
    import get_answer_math_chat_matched_anchor as MATH_GEN
    import get_answer_gsm8k_chat_matched_anchor as GSM8K_GEN

    check("MATH generator: EXPECTED_ALPHAS is the full nine-point family",
          sorted(MATH_GEN.EXPECTED_ALPHAS) == NINE_POINT)
    check("GSM8K generator: EXPECTED_ALPHAS is the full nine-point family",
          sorted(GSM8K_GEN.EXPECTED_ALPHAS) == NINE_POINT)
    expect_ok(
        "MATH generator's EXPECTED_ALPHAS accepted via the shared function",
        CMA.assert_alpha_family, list(NINE_POINT), MATH_GEN.EXPECTED_ALPHAS,
        MATH_GEN.PROTOCOL)
    expect_ok(
        "GSM8K generator's EXPECTED_ALPHAS accepted via the shared function",
        CMA.assert_alpha_family, list(NINE_POINT), GSM8K_GEN.EXPECTED_ALPHAS,
        GSM8K_GEN.PROTOCOL)


def main():
    run_group_a()
    run_group_b()
    run_group_c()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("[ok] all checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
