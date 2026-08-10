#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""PV9 full online episodes. Stage 1 rebuilt; Stage 2 byte-unchanged.

See `bandit_pv9` for the four Stage-1 modifications and the scaffold boundary.
This module is the runner: it reuses pv7's loop wholesale so steering
semantics, fire counting, reward tapes, Stage-2 scoring and the record schema
cannot drift, and changes exactly two things.

    1. the Stage-1 prompt  (patched in, as pv8 does)
    2. stop strings on the Stage-1 generation

WHERE THE STOP RULE IS APPLIED, AND WHY NOT AT GENERATION
---------------------------------------------------------
The rule wraps `p7.extract_evidence_policy_block` -- the single point where
Stage 1's text becomes what Stage 2 reads -- and NOT `generate`/`regenerate`.

Wrapping the generation calls was the first implementation and it was WRONG:
the loop stores the returned string as `rationale_raw`, so truncating there
overwrites the raw record. The hashtag tail became unrecoverable, `stop_reason`
could never see a `#`, and every round would report `clean` whether the model
terminated natively or was cut in post -- destroying the one measurement that
distinguishes "PV9 fixed termination" from "post-processing hid it".

Wrapping the extractor keeps both properties that matter:

  * `rationale_raw` is byte-exact generation output, on BOTH paths.
  * The two paths still share one post-processing rule. That symmetry is
    load-bearing: unsteered calls `vc.generate`, which has no `stop_strings`
    parameter at all (llms.py:690), while steered calls `vc.regenerate`, which
    does (llms.py:845). Passing it only where it exists would make stopping an
    alpha-correlated generation setting and no cross-alpha comparison would be
    readable.

Applying the rule in post also avoids HF `stop_strings` semantics, which halt
on the marker appearing ANYWHERE in the output -- the trap that cost CGT a
sweep (invalid_rate 0.02 -> 0.11 when a prompt token was used as a marker).

WHY NOT ALSO STOP STAGE 2
-------------------------
Stage 2 does not generate. It is a constrained teacher-forced scoring over
candidate letters, so `invalid_rate` is structurally 0 and there is nothing to
truncate.

ALPHA=0 MUST BE RE-RUN
----------------------
The Stage-1 prompt changed, so no stored pv7/pv8 alpha=0 cell is this
protocol's baseline. All three alphas run here on the same seed bank and the
same reward tapes, each into its own directory.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import bandit_pv7 as p7
import bandit_pv7_episode as p7ep
import bandit_pv8_episode as p8
import bandit_pv9 as p9


PROTOCOL_VERSION = "pv9"
STAGE1_INSTRUCTION_VERSION = p9.STAGE1_INSTRUCTION_VERSION
STAGE2_INSTRUCTION_VERSION = p9.STAGE2_INSTRUCTION_VERSION
HISTORY_BLOCK_VERSION = p9.HISTORY_BLOCK_VERSION
SCORE_BLOCK_VERSION = p9.SCORE_BLOCK_VERSION
UNTRIED_CUE_VERSION = p9.UNTRIED_CUE_VERSION
STOP_STRINGS_VERSION = "stop-hash-v1"
POLICY_PARSER_VERSION = p7ep.POLICY_PARSER_VERSION
RATIONALE_MAX_TOKENS = p9.RATIONALE_MAX_TOKENS


def apply_stop(text: str) -> tuple[str, str | None]:
    """Cut at the earliest stop marker. Returns (text, marker_that_fired)."""
    if not text:
        return text, None
    hits = [(text.find(s), s) for s in p9.STOP_STRINGS if s in text]
    if not hits:
        return text, None
    idx, marker = min(hits)
    return text[:idx], marker


class _StopShim:
    """Apply the stop rule where the rationale becomes Stage-2 input.

    Patches `bandit_pv7.extract_evidence_policy_block` for the duration of one
    episode and restores it in a `finally`, the same containment pattern pv8
    uses for the prompt. Generation is left untouched, so `rationale_raw`
    stays byte-exact and the truncation stays auditable.

    Stage 2 does NOT generate -- it is constrained teacher-forced scoring --
    so there is nothing else to wrap.
    """

    def __init__(self):
        self.original = p7.extract_evidence_policy_block

    def __enter__(self):
        original = self.original

        def _extract(raw: str) -> str:
            # Stop marker first, then the Policy-line truncation: cutting the
            # spray before extraction is what keeps a trailing "#..." from
            # being read as a continuation of the Policy line.
            return original(apply_stop(raw)[0])

        p7.extract_evidence_policy_block = _extract
        return self

    def __exit__(self, *exc):
        p7.extract_evidence_policy_block = self.original
        return False


def run_pv9_episode(vc, diff_mtx, seed: int, env, rationale_alpha: float = 0.0,
                    action_alpha: float = 0.0,
                    rationale_max_tokens: int = RATIONALE_MAX_TOKENS,
                    attest: bool = False) -> dict:
    """One PV9 episode. pv7's loop, PV9's Stage-1 prompt, stop strings on."""
    original = p7.build_rationale_prompt

    def _p9(arm_map_or_order, history, round_idx, environment,
            prompt_variant=p7.PROMPT_P1):
        if prompt_variant != p7.PROMPT_P1B:
            raise AssertionError(
                f"PV9 expects the P1b variant, got {prompt_variant!r}")
        return p9.build_rationale_prompt(
            list(arm_map_or_order), history, round_idx, environment)

    p7.build_rationale_prompt = _p9
    try:
        with _StopShim():
            rec = p7ep.run_pv7_episode(
                vc, diff_mtx, seed=seed, env=env,
                rationale_alpha=rationale_alpha, action_alpha=action_alpha,
                rationale_max_tokens=rationale_max_tokens, attest=attest)
    finally:
        p7.build_rationale_prompt = original

    rec["protocol"] = PROTOCOL_VERSION
    rec["stage1_instruction_version"] = STAGE1_INSTRUCTION_VERSION
    rec["history_block_version"] = HISTORY_BLOCK_VERSION
    rec["score_block_version"] = SCORE_BLOCK_VERSION
    rec["untried_cue_version"] = UNTRIED_CUE_VERSION
    rec["stop_strings_version"] = STOP_STRINGS_VERSION
    rec["stop_strings"] = list(p9.STOP_STRINGS)
    rec["rationale_word_limit"] = p9.RATIONALE_WORD_LIMIT

    # Which mechanism ended each rationale. Recorded per round so that
    # "PV9 fixed termination" and "post-processing hid the continuation" stay
    # distinguishable -- without this the two are observationally identical.
    for rd in rec["rounds"]:
        # `rationale_raw` is untouched generation output; `rationale_stopped`
        # is what the extractor actually saw. Storing both is what makes the
        # termination question answerable after the fact.
        stopped, marker = apply_stop(rd["rationale_raw"])
        rd["rationale_stopped"] = stopped
        rd["stop_marker"] = marker
        rd["stop_reason"] = p9.stop_reason(rd["rationale_raw"])
    reasons = [rd["stop_reason"] for rd in rec["rounds"]]
    rec["stop_reason_counts"] = {r: reasons.count(r) for r in sorted(set(reasons))}

    # STAGE-1-ONLY INFORMATION ISOLATION, asserted rather than trusted: the
    # entire attribution argument (a changed action came from Stage 1's
    # reasoning, not from priming Stage 2's candidate logits) rests on it.
    for name, rd in rec.get("attestation", {}).items():
        rp, ap = rd["rationale_prompt"], rd["action_prompt"]
        for token, label in ((p9._UNTRIED_CUE, "untried cue"),
                             ("Your score so far", "score line"),
                             ("CHOICE HISTORY", "history block")):
            if token in ap:
                raise AssertionError(f"{label} leaked into the PV9 Stage-2 prompt")
        if "Your score so far" not in rp:
            raise AssertionError("PV9 Stage-1 prompt lacks the score line")
        if "CHOICE HISTORY" not in rp:
            raise AssertionError("PV9 Stage-1 prompt lacks the history block")
        # At round 0 every arm is untried, so the cue MUST be present. Only
        # checking that it stays out of Stage 2 would leave a silently
        # cue-less Stage 1 undetected -- and a missing cue is the difference
        # between this protocol and pv8.
        if name == "round_0" and p9._UNTRIED_CUE not in rp:
            raise AssertionError(
                "PV9 round-0 Stage-1 prompt lacks the untried-arm cue, but "
                "every arm is untried at round 0")
    return rec


def resume_key(env_name: str, rationale_alpha: float, action_alpha: float,
               layer_start: int, layer_end: int,
               seeds: Sequence[int],
               model_config: dict | None = None) -> str:
    """Distinct for anything that changes the measurement.

    Beyond pv8's key this carries the PV9 protocol tag and the score-block,
    untried-cue and stop-string versions. Each of those changes the Stage-1
    prompt or the text Stage 2 receives, so a stored row from a different
    setting must not be resumable as this cell.

    `env_name` distinguishes easy from neartie, so the two environments cannot
    collide even though they share a seed bank.
    """
    ordered = sorted(int(s) for s in seeds)
    digest = hashlib.sha1(",".join(map(str, ordered)).encode()).hexdigest()[:10]
    key = (f"{PROTOCOL_VERSION}_{env_name}"
           f"_s1v{STAGE1_INSTRUCTION_VERSION}_s2v{STAGE2_INSTRUCTION_VERSION}"
           f"_h{HISTORY_BLOCK_VERSION}_sc{SCORE_BLOCK_VERSION}"
           f"_cu{UNTRIED_CUE_VERSION}_st{STOP_STRINGS_VERSION}"
           f"_ra{rationale_alpha:g}_aa{action_alpha:g}"
           f"_L{layer_start}-{layer_end}_n{len(ordered)}_sd{digest}")
    if model_config:
        payload = ",".join(f"{k}={model_config[k]}" for k in sorted(model_config))
        key += f"_cfg{hashlib.sha1(payload.encode()).hexdigest()[:10]}"
    return key


model_config_fingerprint = p8.model_config_fingerprint
expected_fires = p7ep.expected_fires
