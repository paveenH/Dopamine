#!/usr/bin/env python3
"""pv7 two-stage Bandit episode runner. INDEPENDENT of pv6 -- nothing here
imports from or mutates `bandit_pv6_episode`, and no pv6 file changes.

WHY pv7 IS A SEPARATE PROTOCOL
------------------------------
pv6 stays citable as a boundary finding, but its numbers cannot be pooled with
pv7's: the environment is shared, everything around it is not. pv7 changes the
prompt (single OPTIONS table, no TRIED/UNTRIED drift), both anchors (token 220
instead of ` Button`/period), the candidates (bare `A` = 32, not ` A` = 362),
and now the Stage 2 instruction. A pv7 cell is a new measurement.

THE TWO FROZEN INTERFACE CHOICES, AND THE EVIDENCE FOR THEM
-----------------------------------------------------------
Stage 1 = P1b, Stage 2 = S1. Both were selected on the frozen state bank
(`eval_pv7_frozen_states.py`, `eval_pv7_stage2_ablation.py`) BEFORE any
trajectory ran, on validity / grounding / completion / cost -- never on reward.

  * P1b beat P1/P2: policy parse rate 60.7% -> 100%, lowest grounding error.
  * S1 ("Follow the Policy above...") vs S0, matched on state x rotation:
    non-A-policy overridden by A fell 54.3% -> 1.9% (McNemar n=428,
    discordant 178/0, p<1e-4), Button C follow 10.9% -> 95.7%, and choice
    became rotation-invariant (0/107 states change vs 14/107 under S0).
    Margin ROSE (+0.83) while entropy FELL (-0.10): S1 moves probability mass
    onto the target rather than flattening the distribution.

WHAT A pv7 GATE PASS WOULD AND WOULD NOT SHOW  (frozen wording)
---------------------------------------------------------------
It would show: Llama3-8B has Bandit competence UNDER A STRUCTURED,
PARSER-ASSISTED rationale interface with Policy-following constrained action.

It would NOT show native free-generation competence. P1b's native termination
still fails -- 119/120 frozen-state rationales hit the token cap -- and the
clean Policy that Stage 2 reads is recovered by an extractor, not emitted
cleanly by the model. Any writeup must carry that qualifier.

CONSEQUENCE FOR alpha, DECIDED IN ADVANCE
-----------------------------------------
S1 leaves Stage 2 almost no room to overrule the Policy (1.9%). So pv7's
action-only cell is expected to be near-null, and that null reads as
"the executor is robust", NOT "alpha does not affect the action". pv6's
action-vs-both contrast is not transferable: it was informative precisely
because pv6's Stage 2 DID overrule the rationale. pv7 puts the interesting
alpha at Stage 1.

alpha is therefore split into `rationale_alpha` / `action_alpha` rather than
inheriting pv6's `steering_scope`, which cannot express the (alpha, 0) cell
that is now the primary experiment.

SITE-COUNTER ACCEPTANCE (L=9 for the standard 11-20 band, K arms, T rounds)
---------------------------------------------------------------------------
    rationale_alpha=0, action_alpha=0 -> {rationale: 0,   action: 0}
    rationale-only (alpha, 0)         -> {rationale: L*T, action: 0}
    action-only    (0, alpha)         -> {rationale: 0,   action: L*K*T}
For Llama3-8B Easy (L=9, K=4, T=100): 900 and 3600. A count of 3200 means
zero rows are being counted; `action == rationale` means the K factor is
missing. `steered_*` records intent, `steering_fires` records observation --
they can disagree only if there is a bug, which is why both exist.
"""

from __future__ import annotations

import torch

import bandit_reference as br
import bandit_pv7 as p7

PROTOCOL_VERSION = "pv7"

# Versioned so a LATER wording change cannot be mistaken for this measurement.
# Both are behaviour-affecting and belong in the resume key.
STAGE1_INSTRUCTION_VERSION = "p1b"
STAGE2_INSTRUCTION_VERSION = "s1"
# The Policy extractor/parser. Bumping it changes `policy_target`, which is a
# RECORDED field only (it never picks the action), so it does not belong in the
# resume key -- but it must be in metadata, or two runs' policy statistics
# become silently incomparable.
POLICY_PARSER_VERSION = "pf1"

RATIONALE_MAX_TOKENS = 64

# S1, frozen. Names no button and supplies no strategy: it only asks Stage 2 to
# use the model's OWN Policy line.
_ACTION_INSTRUCTION = "Follow the Policy above and select one button."


def build_action_prompt_s1(arm_order, history, round_idx, env, clean) -> str:
    """pv7 Stage 2 under S1. Ends in exactly `Choose Button: ` (token 220).

    Built here rather than in bandit_pv7.build_action_prompt so the frozen
    Phase-3 comparison keeps reproducing its own S0 baseline byte for byte.
    """
    state = p7.render_state(arm_order, history, round_idx, env,
                            prompt_variant=p7.PROMPT_P1B)
    analysis = f"Evidence: {clean}" if clean else "Evidence:"
    prompt = (f"{state}\n\nMODEL ANALYSIS\n{analysis}\n\n"
              f"{_ACTION_INSTRUCTION}\n{p7.ACTION_ANCHOR}")
    p7._assert_single_trailing_space(prompt, p7.ACTION_ANCHOR)
    return prompt


def audit_pv7_prompt(vc, prompt: str, stage: str) -> dict:
    """HARD invariants. A violation stops the run instead of being recorded.

    By the time a wrong injection site is read off a stored record, the whole
    episode was steered at the wrong token. pv7's site is token 220 -- the
    single trailing space -- for BOTH stages, which is the alignment the RSN
    mask was extracted at.
    """
    ids = vc.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    if bos_id is not None and ids[:2] == [bos_id, bos_id]:
        raise AssertionError(
            f"double BOS in the pv7 {stage} prompt (head={ids[:4]}); a "
            "serialized chat BOS was not stripped before add_special_tokens")
    if ids[-1] != p7.EXPECTED_WHITESPACE_TOKEN_ID:
        raise AssertionError(
            f"pv7 {stage} prompt tail is token {ids[-1]}, expected "
            f"{p7.EXPECTED_WHITESPACE_TOKEN_ID}; check rstrip, double spaces, "
            "tokenizer identity and anchor wording")
    return {
        "stage": stage,
        "n_tokens": len(ids),
        "head_ids": ids[:5],
        "tail_ids": ids[-8:],
        "double_bos": ids[:2] == [128000, 128000],
        "injection_token_id": ids[-1],
        "injection_is_token_220": ids[-1] == p7.EXPECTED_WHITESPACE_TOKEN_ID,
    }


def score_candidates_pv7(vc, prompt: str, env: br.Environment, diff_mtx):
    """Constrained action: summed log-prob of each bare arm letter.

    THE ACTION IS ALWAYS AN ARGMAX OVER THESE FOUR SCORES. The Policy parser
    never selects it -- the parser only builds the clean rationale text and
    records what the model said it intended. That separation is what lets
    `action_follows_policy` be a measurement rather than a tautology.

    Candidates are independent batch rows, so there is no candidate-ordering
    effect; concatenation is at the ID level (a string concat would let BPE
    merge 220+A back into ' A'=362 and silently move the injection site).
    """
    import numpy as np

    sfx = p7.candidate_suffixes(env)
    labels = [p7.candidate_arm(s) for s in sfx]
    p7.audit_id_level_continuation(vc.tokenizer, prompt, sfx)
    tok_ids = [vc.tokenizer.encode(s, add_special_tokens=False) for s in sfx]
    per_tok = vc.regenerate_logits_teacher_forcing(
        prompts=[prompt] * len(sfx), answer_token_ids=tok_ids,
        diff_matrices=diff_mtx)
    scores = {}
    for lab, ids, lg in zip(labels, tok_ids, per_tok):
        lp = 0.0
        for k, tid in enumerate(ids):
            row = torch.from_numpy(np.asarray(lg[k], dtype=np.float32))
            lp += float(torch.log_softmax(row, dim=-1)[tid])
        scores[lab] = lp
    return scores, max(scores, key=scores.get)


def run_pv7_episode(
    vc,
    diff_mtx,
    seed: int,
    env: br.Environment,
    rationale_alpha: float = 0.0,
    action_alpha: float = 0.0,
    rationale_max_tokens: int = RATIONALE_MAX_TOKENS,
    attest: bool = False,
) -> dict:
    """One pv7 episode. Both stages see the SAME fixed OPTIONS order.

    Rotation belongs to the Stage-2 ablation only; a real trajectory must not
    rotate, or Stage 1 and Stage 2 would disagree about the display.

    `diff_mtx` is the direction; the two alphas select which passes receive it.
    Passing diff_mtx=None (or alpha 0) registers NO hook for that pass --
    unsteered is deliberately not the same code path as steered-by-zero.
    """
    steer_rationale = bool(diff_mtx is not None and rationale_alpha != 0.0)
    steer_action = bool(diff_mtx is not None and action_alpha != 0.0)
    if steer_rationale and steer_action and rationale_alpha != action_alpha:
        # One diff matrix carries one alpha. Two different non-zero alphas in
        # one episode need two scaled matrices; failing loudly here beats
        # silently applying the same magnitude to both stages.
        raise ValueError(
            "rationale_alpha and action_alpha differ and are both non-zero; "
            "pass a pre-scaled diff matrix per stage instead")

    tape = br.RewardTape(seed, env)
    arm_map = tape.arm_map
    arm_order = list(arm_map)          # frozen for the whole episode
    history: list[tuple[str, int]] = []
    choices: list[str] = []
    feedbacks: list[int] = []
    per_round: list[dict] = []
    attestation: dict = {}

    fired_rationale = fired_action = 0
    _can_count = hasattr(vc, "steering_fire_count")
    if _can_count:
        vc.steering_fire_count(reset=True)

    for round_idx in range(env.horizon):
        # ---- Stage 1: rationale (P1b) ------------------------------------
        r_prompt = p7.build_rationale_prompt(
            arm_order, history, round_idx, env, prompt_variant=p7.PROMPT_P1B)
        audit_r = audit_pv7_prompt(vc, r_prompt, "rationale")

        torch.manual_seed(seed * 100_003 + round_idx)
        if steer_rationale:
            out = vc.regenerate(
                inputs=[r_prompt], diff_matrices=diff_mtx,
                prefill_only=True, prefill_tail_len=1,
                max_new_tokens=rationale_max_tokens, temperature=0.0)
        else:
            # generate registers no hooks at all; regenerate raises on
            # diff_matrices=None, so it is not a way to express "unsteered".
            out = vc.generate(inputs=[r_prompt],
                              max_new_tokens=rationale_max_tokens,
                              temperature=0.0)
        if _can_count:
            fired_rationale += vc.steering_fire_count(reset=True)
        raw = out[0] if isinstance(out, list) else out
        clean = p7.extract_evidence_policy_block(raw)

        # ---- Stage 2: constrained action (S1) ----------------------------
        a_prompt = build_action_prompt_s1(arm_order, history, round_idx, env,
                                          clean)
        audit_a = audit_pv7_prompt(vc, a_prompt, "action")
        scores, arm = score_candidates_pv7(
            vc, a_prompt, env, diff_mtx if steer_action else None)
        if _can_count:
            fired_action += vc.steering_fire_count(reset=True)

        reward = tape.pull(arm)

        # DECISION QUALITY (Stage 1) and EXECUTION CONSISTENCY (Stage 2) are
        # recorded separately, because S1 makes them different questions: a
        # bad episode can now come from a bad Policy or from a failure to
        # execute a good one, and only these two fields tell them apart.
        pol = _policy_record(clean, arm)
        fmt = p7.rationale_format_flags(raw, clean)
        ordered = sorted(scores.values(), reverse=True)
        per_round.append({
            "round": round_idx,
            "rationale_raw": raw,
            "rationale_clean": clean,
            "format_flags": fmt,
            **pol,
            "action": arm,
            "reward": reward,
            # RAW floats: the decision is an argmax over these and alpha is
            # hypothesised to move them, so rounding would discard exactly the
            # near-tie resolution a small alpha effect lives in.
            "candidate_scores": dict(scores),
            "margin": (ordered[0] - ordered[1] if len(ordered) > 1
                       else float("nan")),
            "norm_entropy": _norm_entropy(list(scores.values())),
        })

        if attest and round_idx in (0, min(10, env.horizon - 1)):
            attestation[f"round_{round_idx}"] = {
                "rationale_prompt": r_prompt,
                "action_prompt": a_prompt,
                "rationale_raw": raw,
                "rationale_clean": clean,
                "tokens_rationale": audit_r,
                "tokens_action": audit_a,
            }

        history.append((arm, reward))
        choices.append(arm)
        feedbacks.append(reward)

    parsed = [r for r in per_round if r["policy_parsed"]]
    followed = [r for r in parsed if r["action_follows_policy"]]
    rec = br._package(
        seed, env, tape, choices, feedbacks, policy="model",
        extra={
            # `choices` / `feedbacks` / `arm_map` / `best_arm` come from
            # _package, which is what evaluate_competence_gate reads. Every
            # pv7 field below is ADDITIVE: the frozen gate evaluator is not
            # modified and simply ignores what it does not read.
            "protocol": PROTOCOL_VERSION,
            "stage1_instruction_version": STAGE1_INSTRUCTION_VERSION,
            "stage2_instruction_version": STAGE2_INSTRUCTION_VERSION,
            "policy_parser_version": POLICY_PARSER_VERSION,
            "rounds": per_round,
            "invalid_rate": 0.0,     # structural: constrained scoring
            "rationale_max_tokens": rationale_max_tokens,
            "policy_parse_rate": len(parsed) / env.horizon,
            "action_follows_policy_rate": (len(followed) / len(parsed)
                                           if parsed else float("nan")),
            "rationale_alpha": float(rationale_alpha),
            "action_alpha": float(action_alpha),
            "steered_rationale": steer_rationale,
            "steered_action": steer_action,
            # OBSERVED sites, not hook calls. See the module docstring for the
            # acceptance numbers; a config claiming a steered stage with 0
            # fires here is a bug that only this pair can reveal.
            "steering_fires": ({"rationale": fired_rationale,
                                "action": fired_action}
                               if _can_count else None),
        })
    if attest:
        rec["attestation"] = attestation
    return rec


def _policy_record(clean: str, chosen: str) -> dict:
    """What Stage 1 intended, and whether Stage 2 did it.

    Uses the same first-decision-clause parser as the frozen-state evaluation
    so policy statistics stay comparable across pv7 phases. RECORD ONLY -- it
    never chooses the action.

    The import is function-local and guarded: `eval_pv7_frozen_states` pulls in
    `bandit_pv6_episode` for its P0 arm, and a trajectory runner must not
    acquire a pv6 dependency at module import. If that module is ever removed
    or made pv6-only, the runner keeps running and records the parse as failed
    rather than taking the whole episode down.
    """
    try:
        from eval_pv7_frozen_states import policy_flags
    except ImportError:                                         # pragma: no cover
        return {"policy_stance": "unclear", "policy_target": None,
                "policy_target_source": "parser_unavailable",
                "policy_parsed": False, "action_follows_policy": None}
    p = policy_flags(clean, chosen)
    parsed = (p["policy_target_source"] == "policy_first_clause"
              and p["stance_is_clear"])
    return {
        "policy_stance": p["policy_stance"],
        "policy_target": p["policy_target"],
        "policy_target_source": p["policy_target_source"],
        "policy_parsed": parsed,
        "action_follows_policy": p["action_follows_policy"],
    }


def _norm_entropy(vals: list[float]) -> float:
    import math
    m = max(vals)
    e = [math.exp(v - m) for v in vals]
    s = sum(e)
    p = [x / s for x in e]
    return -sum(q * math.log(q) for q in p if q > 0) / math.log(len(p))


def resume_key(env_name: str, rationale_alpha: float, action_alpha: float,
               layer_start: int, layer_end: int, n_seeds: int) -> str:
    """Distinct for anything that changes the measurement.

    Both instruction versions are in it: a Stage 2 wording change produces
    different trajectories, so reusing a stored row across it would silently
    skip the new configuration -- the exact failure the pv6 `iface` segment was
    added to prevent.
    """
    return (f"{PROTOCOL_VERSION}_{env_name}"
            f"_s1v{STAGE1_INSTRUCTION_VERSION}_s2v{STAGE2_INSTRUCTION_VERSION}"
            f"_ra{rationale_alpha:g}_aa{action_alpha:g}"
            f"_L{layer_start}-{layer_end}_n{n_seeds}")


def expected_fires(rationale_alpha: float, action_alpha: float,
                   n_layers: int, k: int, horizon: int) -> dict:
    """Acceptance numbers for `steering_fires`; see the module docstring."""
    return {
        "rationale": n_layers * horizon if rationale_alpha else 0,
        "action": n_layers * k * horizon if action_alpha else 0,
    }
