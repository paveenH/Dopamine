#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""PV9 full online episodes. Stage 1 rebuilt; Stage 2 byte-unchanged.

See `bandit_pv9` for the four Stage-1 modifications and the scaffold boundary.
This module is the runner: it reuses pv7's loop wholesale so steering
semantics, fire counting, reward tapes, Stage-2 scoring and the record schema
cannot drift, and changes exactly two things.

    1. the Stage-1 prompt  (patched in, as pv8 does)
    2. stop strings on the Stage-1 generation

WHY THE STOP RULE IS APPLIED IN POST, TO BOTH PATHS
---------------------------------------------------
The two Stage-1 paths are DIFFERENT METHODS: unsteered calls `vc.generate`,
which has no `stop_strings` parameter at all (llms.py:690); steered calls
`vc.regenerate`, which does (llms.py:845). Two consequences drove this design:

  * pv7's loop never passes `stop_strings` to `regenerate`, so no cell gets
    model-side stopping today.
  * If it were passed only where the parameter exists, the steered cells would
    stop decoding early and the alpha=0 cell would not. Cross-alpha
    comparisons would then confound the intervention with a generation
    setting -- unreadable, and exactly what this protocol cannot afford.

So the rule is applied UNIFORMLY as post-truncation of the returned text, by
wrapping BOTH methods for the duration of an episode. Identical
post-condition on every path, at the cost of decoding some tokens that are
then discarded. Stage 2 reads the same thing either way, which is what has to
match; the discarded tail stays visible in `rationale_raw`.

This also avoids HF's `stop_strings` semantics, which halt on the marker
appearing ANYWHERE in the output -- the trap that cost CGT a sweep
(invalid_rate 0.02 -> 0.11 when a prompt token was used as a stop marker).
Post-truncation on a marker that cannot occur in the prompt is strictly safer.

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
    """Post-truncate Stage-1 output at the first stop marker, on BOTH paths.

    Wraps `generate` AND `regenerate`, because the unsteered and steered cells
    go through different methods and must end up with the same
    post-condition; wrapping only one would make the stop rule an alpha-
    correlated generation setting. Patches the bound methods for the duration
    of one episode and restores them in a `finally`, the same containment
    pattern pv8 uses for the prompt.

    Stage 2 does NOT generate -- it is constrained teacher-forced scoring --
    so it is deliberately not wrapped.
    """

    _METHODS = ("generate", "regenerate")

    def __init__(self, vc):
        self.vc = vc
        self.originals = {m: getattr(vc, m) for m in self._METHODS}

    @staticmethod
    def _wrap(original):
        def _call(*args, **kwargs):
            out = original(*args, **kwargs)
            if isinstance(out, list):
                return [apply_stop(t)[0] for t in out]
            return apply_stop(out)[0]
        return _call

    def __enter__(self):
        for name, original in self.originals.items():
            setattr(self.vc, name, self._wrap(original))
        return self

    def __exit__(self, *exc):
        for name, original in self.originals.items():
            setattr(self.vc, name, original)
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
        with _StopShim(vc):
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
        rd["stop_reason"] = p9.stop_reason(rd["rationale_raw"])
    reasons = [rd["stop_reason"] for rd in rec["rounds"]]
    rec["stop_reason_counts"] = {r: reasons.count(r) for r in sorted(set(reasons))}

    # STAGE-1-ONLY INFORMATION ISOLATION, asserted rather than trusted: the
    # entire attribution argument (a changed action came from Stage 1's
    # reasoning, not from priming Stage 2's candidate logits) rests on it.
    for rd in rec.get("attestation", {}).values():
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
