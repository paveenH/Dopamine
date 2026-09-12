#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
chat_matched_anchor_lib.py -- shared, delicate anchor/injection-site handling
for the MATCHED-ANCHOR chat-template control condition
(get_answer_math_chat_matched_anchor.py / get_answer_gsm8k_chat_matched_anchor.py).

WHY A SHARED MODULE RATHER THAN TWO COPIES (unlike the native chat-sweep
generators, which deliberately duplicate task-specific templating/extraction
to avoid cross-task parameter mixing -- see get_answer_math_chat_sweep.py's
own docstring): the logic here is the SAME invariant regardless of task --
strip the SAME literal "Answer: " anchor, wrap with the SAME chat-template
call, re-attach the SAME anchor, and assert the SAME token id on the last
prefill position. A second, independently-typed copy of this logic is exactly
how the two tasks' injection sites could silently drift apart, which is the
one thing this whole experiment exists to prevent. This mirrors
chat_sweep_common.py's own stated rationale for existing (shared statistical
primitives so two analyzers can't silently apply different rules) applied to
the generator side instead.

WHAT THIS MODULE DOES NOT DO: it does not import numpy, torch, transformers,
llms, utils, or template -- it is pure string/tokenizer-interface manipulation
with zero heavy dependencies, so it can be imported and unit-tested (with a
fake tokenizer) with NO model, NO GPU, and NO server. `vc` below means "any
object exposing a HuggingFace-tokenizer-like `.tokenizer` attribute with
`.apply_chat_template`, `.bos_token`, `.bos_token_id`, `.chat_template`,
`__call__`, and `.decode`" -- llms.VicundaModel satisfies this, and so does a
minimal fake used by test_chat_matched_anchor.py.

THE MATCHED-ANCHOR CONSTRUCTION (the experiment's entire mechanism):
  1. The caller's rendered prompt BODY (identical body text to the native
     bare-string and native-chat conditions) ends in the literal anchor
     "Answer: " (with its trailing space). strip_trailing_answer_anchor()
     removes it -- this is step (1) of the user's spec: "user message 中不要
     保留末尾的 Answer:，避免重复".
  2. The body WITHOUT the anchor is wrapped via
       tokenizer.apply_chat_template(
           [{"role": "user", "content": body_without_anchor}],
           tokenize=False, add_generation_prompt=True,
       )
     exactly as the native-chat sweep does, so the rest of the chat template
     (system defaults, turn markers, BOS) is IDENTICAL to the native-chat
     condition -- the ONLY planned difference from native chat is where the
     "Answer: " anchor lands.
  3. A duplicated leading BOS (the chat template serializes its own BOS text,
     and vc.regenerate's internal tokenization call uses
     add_special_tokens=True) is stripped, exactly as the native chat-sweep
     generators do.
  4. The anchor "Answer: " is re-appended directly after the wrapped text
     (i.e. directly after the assistant generation header
     "<|start_header_id|>assistant<|end_header_id|>\n\n"), putting the
     trailing space of the anchor back at the very end of the prefill --
     the SAME position prefill-only tail=1 steering injects into under the
     bare-string condition.
  5. assert_matched_anchor_tail() then tokenizes the EXACT final string with
     add_special_tokens=True (matching vc.regenerate's own internal
     tokenization) and FAILS CLOSED unless the last token is id 220, decoded
     text a single ASCII space -- the current meta-llama/Llama-3.1-8B-Instruct
     tokenizer's id for a standalone " ". This is spec requirement (4): "若不
     符则 fail closed，不要继续生成." No approximate match, no substring
     check, no "close enough" -- exact id AND exact decoded text.

Both native conditions this control sits between, for reference (verified
elsewhere in this repo on the real tokenizer, not re-derived here):
  bare        : last prefill token id 220 ' ' (the "Answer: " anchor)
  native chat : last prefill token id 271 '\n\n' (the assistant header,
                because apply_chat_template's Jinja `| trim` strips the
                anchor's trailing space from the user-turn content)
  matched-anchor (this module) : re-creates id 220 ' ' UNDER the full chat
                template, by moving the anchor to AFTER the header instead of
                inside the (trimmed) user turn.
"""

from __future__ import annotations

import sys

ANSWER_ANCHOR = "Answer: "
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"

# The current meta-llama/Llama-3.1-8B-Instruct tokenizer's id for a standalone
# ASCII space, matching the bare condition's "Answer: " trailing-space anchor.
# This is a FACT ABOUT THE TOKENIZER, verified at runtime by
# assert_matched_anchor_tail() below -- never assumed without the check.
EXPECTED_TAIL_TOKEN_ID = 220
EXPECTED_TAIL_TOKEN_TEXT = " "


def die(msg: str):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def strip_leading_bos(vc, text: str) -> str:
    """Byte-identical logic to get_answer_{gsm8k,math}_chat_sweep.py's
    strip_leading_bos (itself reusing proofwriter_owa /
    bandit_pv6_episode.py's pattern): vc.regenerate's internal tokenization
    uses add_special_tokens=True, so an un-stripped chat-templated string that
    already serialized a BOS string yields two leading BOS ids."""
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
        die(f"double BOS in the {label} matched-anchor prompt "
            f"(head={ids[:4]}) -- strip_leading_bos did not remove the chat "
            "template's own serialized BOS text before this "
            "add_special_tokens=True call.")


def strip_trailing_answer_anchor(body: str, label: str) -> str:
    """Spec (1): the user message must NOT retain the trailing 'Answer: ' --
    it is re-attached AFTER the chat template wraps the rest of the body, so
    keeping it in the user turn would duplicate it (one copy inside the
    (trimmed) user turn, one copy after the header).

    Fails closed rather than guessing: if the body does not end EXACTLY with
    this literal anchor, the body is not the template this experiment was
    designed against (template.build_math_suite / build_gsm8k_default_suite
    both end every variant in "Answer: ", verified at build time), and
    blindly slicing the last 8 characters off a differently-shaped body would
    silently corrupt the prompt."""
    if not body.endswith(ANSWER_ANCHOR):
        die(f"{label}: prompt body does not end with the expected "
            f"{ANSWER_ANCHOR!r} anchor (tail={body[-20:]!r}) -- refusing to "
            "strip blindly. This protocol requires the body to end in the "
            "literal 'Answer: ' anchor, matching the bare/native-chat "
            "templates it is designed to sit between.")
    return body[: -len(ANSWER_ANCHOR)]


def assert_chat_template_effective(vc, reference_text: str, wrapped: str) -> None:
    """First-sample assertion, mirroring the native chat-sweep generators:
    the chat template actually fired (wrapped != a no-op) and produced the
    expected Llama-3 assistant generation header."""
    if wrapped == reference_text:
        die("chat template had NO effect on the first sample (wrapped text "
            "is byte-identical to the reference body) -- apply_chat_template "
            "did not actually run, or the tokenizer has no chat_template.")
    if not getattr(vc.tokenizer, "chat_template", None):
        die("tokenizer.chat_template is empty/None -- cannot apply a chat "
            "template that does not exist.")
    if ASSISTANT_HEADER not in wrapped:
        die(f"expected assistant generation header {ASSISTANT_HEADER!r} not "
            "found in the first wrapped prompt -- add_generation_prompt=True "
            "did not produce the expected Llama-3 chat header.")


def build_matched_anchor_prompt(vc, body_without_anchor: str):
    """Spec (2)+(3): wrap the anchor-free body via apply_chat_template with
    add_generation_prompt=True, then strip a duplicated leading BOS.

    Returns (wrapped_before_anchor, final_text_with_anchor) -- the caller
    re-attaches ANSWER_ANCHOR itself (spec (3): "在 assistant header 后拼接
    Answer: ") so the exact concatenation point is visible at the call site
    rather than hidden inside this helper.
    """
    wrapped = vc.tokenizer.apply_chat_template(
        [{"role": "user", "content": body_without_anchor}],
        tokenize=False, add_generation_prompt=True,
    )
    wrapped = strip_leading_bos(vc, wrapped)
    return wrapped


def tail_ids(vc, text: str, k: int = 4):
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"][-k:]
    return [{"id": int(i), "text": vc.tokenizer.decode([i])} for i in ids]


def assert_matched_anchor_tail(vc, text: str, label: str) -> dict:
    """Spec (4), the load-bearing runtime gate of this entire experiment:
    the LAST prefill token -- the only position vc.regenerate's prefill-only
    tail=1 steering injects into -- must be a single ASCII space, token id
    220 on the current Llama-3.1 tokenizer. Tokenized with
    add_special_tokens=True, matching vc.regenerate's own internal
    tokenization call (_regenerate_prefill_only), so this checks the EXACT
    sequence the model will actually see, not an approximation of it.

    FAILS CLOSED: any mismatch (wrong id, or right id but the decode does not
    read back as a lone space -- e.g. a tokenizer where the same id is
    reused for something else) aborts before any generation happens. There
    is no tolerance and no fallback anchor; a mismatch means the matched-
    anchor construction did not do what this experiment requires, and
    generating anyway would silently run a DIFFERENT, unverified injection
    site under this protocol's name."""
    ids = vc.tokenizer(text, add_special_tokens=True)["input_ids"]
    if not ids:
        die(f"{label}: tokenization of the matched-anchor prompt produced "
            "zero tokens.")
    last_id = int(ids[-1])
    last_text = vc.tokenizer.decode([last_id])
    if last_id != EXPECTED_TAIL_TOKEN_ID or last_text != EXPECTED_TAIL_TOKEN_TEXT:
        die(f"{label}: last prefill token is id={last_id} text={last_text!r}, "
            f"expected id={EXPECTED_TAIL_TOKEN_ID} "
            f"text={EXPECTED_TAIL_TOKEN_TEXT!r} (a single ASCII space, "
            "matching the bare condition's 'Answer: ' anchor). Refusing to "
            "generate under an unverified injection site -- reproducing the "
            "bare anchor's injection site under the full chat template is "
            "this experiment's entire mechanism.")
    return {"id": last_id, "text": last_text}


def assert_alpha_family(got_alphas, want_alphas, protocol_label):
    """SHARED between both generators (previously each had its own inline
    copy of this check) so the frozen dose-family validation cannot silently
    drift between MATH and GSM8K.

    EXACT match on the frozen alpha family, as SORTED LISTS -- not a
    set-subset, and with NO int() coercion (utils.parse_configs accepts
    float alpha tokens, e.g. '0.5-11-20', so int(2.5) would silently read as
    2 and pass a naive subset check). Order-INDEPENDENT: a shuffled
    --configs is accepted, since sorting both sides before comparing is
    exactly what makes this a SET-equality check rather than a sequence
    check. Any of a missing dose, a duplicate dose, an extra dose, or a
    non-integer/off-grid dose fails closed here."""
    got_sorted = sorted(got_alphas)
    want_sorted = sorted(want_alphas)
    if got_sorted != want_sorted:
        die(f"--configs alphas {got_sorted} != {protocol_label}'s frozen "
            f"dose set {want_sorted}. Exact match required: a non-integer "
            "dose, a missing dose, a duplicate, or an extra dose all land "
            "here. A partial family must not be written under this "
            "protocol name.")
