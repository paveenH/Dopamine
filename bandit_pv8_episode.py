#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""pv8 = pv7 + CHOICE HISTORY in Stage 1. Full online episodes.

ONE change from pv7, and it is in Stage 1 only:

    Round 51 of 100. Future choices after this one: 49.

    CHOICE HISTORY (oldest -> newest):
    [A A B C C C C ...]

    OPTIONS
    - Button A: 1 reward / 3 trials, empirical rate 0.33
    ...

Stage 2 is the frozen S1 prompt, byte-unchanged from pv7: no history block,
counts only. Putting the history in Stage 2 as well would let a repeated label
prime the candidate logits directly, and a change in the chosen arm could no
longer be attributed to the model's own reasoning.

WHY THIS IS A NEW PROTOCOL, NOT A pv7 FLAG
------------------------------------------
The Stage-1 prompt changes and its length now grows with the round, so pv8
trajectories are not poolable with pv7 ones. New PROTOCOL_VERSION, new resume
key, new output tree. pv7 and pv6 files are byte-unchanged.

WHY H1 AND NOT THE CALCULATOR
-----------------------------
Measured on the frozen states (alpha=0, n=123):

  * history did not change decisions -- targets_one_shot_zero .017 -> .017,
    chose_last_chosen +0.008 -- but it did stabilise the output format
    (hashtags .683 -> .496, policy_parsed .837 -> .919)
  * the Beta calculator moved the primary metric only 2/120 -> 4/120, all in
    3 seeds, with NO signal in the 103 non-re-ranking states, while raising
    posterior-greedy alignment (.675 -> .715) and cutting untried targeting
    (.057 -> .008)

So history is behaviourally near-neutral and format-positive, whereas the
calculator demonstrably strengthens greedy. Adding the calculator here would
confound "what alpha does" with "what the scaffold does to greediness".

WHAT THE FROZEN STATES COULD NOT ANSWER
---------------------------------------
A frozen state yields ONE choice and no feedback, so exploring there can never
pay off: the information a revisit buys is never collected. Only a full
episode closes that loop. A null on the frozen states therefore does not
settle whether alpha changes exploration when exploration has value -- which
is exactly what this run measures.

ALPHA=0 MUST BE RE-RUN
----------------------
H1 changes the Stage-1 prompt, so the stored pv7 alpha=0 cell is NOT the
baseline for this protocol. All three alphas run here, on the same seed bank
and the same reward tapes, each into its own directory.
"""
from __future__ import annotations

import hashlib
import re
from typing import Sequence

import bandit_pv7 as p7
import bandit_pv7_episode as p7ep


PROTOCOL_VERSION = "pv8"
STAGE1_INSTRUCTION_VERSION = "p1b"          # unchanged wording
STAGE2_INSTRUCTION_VERSION = "s1"           # unchanged
HISTORY_BLOCK_VERSION = "hist-letters-v1"
POLICY_PARSER_VERSION = p7ep.POLICY_PARSER_VERSION
RATIONALE_MAX_TOKENS = p7ep.RATIONALE_MAX_TOKENS


# The history block lives HERE, in the generation chain, not in the analysis
# script that first prototyped it. `eval_pv7_history_ablation` is an offline
# analysis file; importing it from a production module would make every future
# edit to that analysis silently change the prompts a trajectory is generated
# under. The analysis file now imports this definition instead, so there is
# exactly one renderer and the dependency points the safe way.
def _letters(history) -> list[str]:
    """`Button A` -> `A`. Raises on any label that is not `Button <X>`."""
    out = []
    for arm, _reward in history:
        m = re.fullmatch(r"Button ([A-Z])", arm)
        if not m:
            raise ValueError(f"unexpected arm label {arm!r}")
        out.append(m.group(1))
    return out


def history_block(history) -> str:
    """The CHOICE HISTORY block. `history` is a sequence of (arm, reward).

    Letters only, no per-round reward: the OPTIONS table already carries the
    reward totals, and adding outcomes here would make this a full-transcript
    condition -- several changes at once instead of one.
    """
    if not history:
        return "CHOICE HISTORY: none"
    return ("CHOICE HISTORY (oldest → newest):\n"
            f"[{' '.join(_letters(history))}]")


def build_rationale_prompt_h1(arm_order, history, round_idx, env) -> str:
    """pv7's P1b Stage-1 prompt with the CHOICE HISTORY block inserted.

    Built by string surgery on pv7's OWN rendered state, deliberately: a
    re-implementation would let pv8 drift away from pv7 on some future edit,
    and the whole design rests on the two differing ONLY by this block.
    """
    state = p7.render_state(arm_order, history, round_idx, env,
                            prompt_variant=p7.PROMPT_P1B)
    marker = "\n\nOPTIONS\n"
    if state.count(marker) != 1:
        raise AssertionError("expected exactly one OPTIONS table")
    head, tail = state.split(marker, 1)
    state_h1 = f"{head}\n\n{history_block(history)}{marker}{tail}"
    prompt = f"{state_h1}\n\n{p7._P1B_INSTRUCTION}\n\n{p7.RATIONALE_ANCHOR}"
    # The anchor invariant is what makes Stage-1 steering land on token 220.
    p7._assert_single_trailing_space(prompt, p7.RATIONALE_ANCHOR)
    return prompt


def run_pv8_episode(vc, diff_mtx, seed: int, env, rationale_alpha: float = 0.0,
                    action_alpha: float = 0.0,
                    rationale_max_tokens: int = RATIONALE_MAX_TOKENS,
                    attest: bool = False) -> dict:
    """One pv8 episode. Reuses pv7's loop with the H1 Stage-1 prompt.

    The loop itself -- steering semantics, fire counting, tape handling,
    Stage-2 scoring, record schema -- is pv7's, so the two protocols cannot
    drift apart on anything except the prompt. The swap is done by patching
    the module attribute pv7's loop calls, then restoring it, rather than by
    copying ~140 lines that would then need to be kept in sync.
    """
    original = p7.build_rationale_prompt

    def _h1(arm_map_or_order, history, round_idx, environment,
            prompt_variant=p7.PROMPT_P1):
        if prompt_variant != p7.PROMPT_P1B:
            raise AssertionError(
                f"pv8 expects the P1b variant, got {prompt_variant!r}")
        return build_rationale_prompt_h1(
            list(arm_map_or_order), history, round_idx, environment)

    p7.build_rationale_prompt = _h1
    try:
        rec = p7ep.run_pv7_episode(
            vc, diff_mtx, seed=seed, env=env,
            rationale_alpha=rationale_alpha, action_alpha=action_alpha,
            rationale_max_tokens=rationale_max_tokens, attest=attest)
    finally:
        p7.build_rationale_prompt = original

    rec["protocol"] = PROTOCOL_VERSION
    rec["history_block_version"] = HISTORY_BLOCK_VERSION
    # Stage 2 must NOT have seen the history. Asserted on the stored
    # attestation rather than trusted, because the whole Stage-1-only claim
    # rests on it.
    for rd in rec.get("attestation", {}).values():
        if "CHOICE HISTORY" not in rd["rationale_prompt"]:
            raise AssertionError("pv8 Stage-1 prompt lacks the history block")
        if "CHOICE HISTORY" in rd["action_prompt"]:
            raise AssertionError("history leaked into the pv8 Stage-2 prompt")
    return rec


def resume_key(env_name: str, rationale_alpha: float, action_alpha: float,
               layer_start: int, layer_end: int,
               seeds: Sequence[int],
               model_config: dict | None = None) -> str:
    """Distinct for anything that changes the measurement.

    Beyond pv7's key this carries the pv8 protocol tag, the history-block
    version (a change to how the block is rendered changes the prompt and
    therefore the trajectory) and, when supplied, a hash of the MODEL AND MASK
    configuration.

    The model/mask hash matters for an unattended run: without it, changing
    `model_dir`, `hs`, `mask_type`, `percentage`, the layer band or the mask
    FILE ITSELF leaves the key identical, and a resume silently returns
    episodes generated under a different intervention. The mask is hashed by
    content, so a regenerated mask with the same filename is a different key.

    `model_config=None` reproduces the older key and is kept only so a stored
    cell written before this field can still be read; new runs always pass it.
    """
    ordered = sorted(int(s) for s in seeds)
    digest = hashlib.sha1(
        ",".join(map(str, ordered)).encode()).hexdigest()[:10]
    key = (f"{PROTOCOL_VERSION}_{env_name}"
           f"_s1v{STAGE1_INSTRUCTION_VERSION}_s2v{STAGE2_INSTRUCTION_VERSION}"
           f"_h{HISTORY_BLOCK_VERSION}"
           f"_ra{rationale_alpha:g}_aa{action_alpha:g}"
           f"_L{layer_start}-{layer_end}_n{len(ordered)}_sd{digest}")
    if model_config:
        payload = ",".join(f"{k}={model_config[k]}"
                           for k in sorted(model_config))
        key += f"_cfg{hashlib.sha1(payload.encode()).hexdigest()[:10]}"
    return key


def model_config_fingerprint(model_dir: str, hs: str, type_: str,
                             mask_type: str, percentage: float, size: str,
                             mask_path: str | None) -> dict:
    """The model/mask identity that a stored cell depends on.

    `mask_sha256` is the content hash of the .npy actually loaded, so a
    regenerated mask under the same name cannot be mistaken for the old one.
    """
    cfg = {"model_dir": model_dir, "hs": hs, "type": type_,
           "mask_type": mask_type, "percentage": f"{percentage:g}",
           "size": size}
    if mask_path:
        h = hashlib.sha256()
        with open(mask_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        cfg["mask_sha256"] = h.hexdigest()
    return cfg


expected_fires = p7ep.expected_fires
