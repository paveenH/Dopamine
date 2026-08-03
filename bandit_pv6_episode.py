"""pv6 F-reference episode runner: two-stage constrained choice.

Separate from get_answer_bandit.run_episode ON PURPOSE. That function carries
pv1-pv5 invariants that pv6 deliberately breaks — a single `rng` stream shared
with `get_feedback`, a random fallback arm on parse failure, temperature=1.0
sampling, warm-start bookkeeping. Threading pv6 through it as more flags would
put those invariants one boolean away from each other. pv6 instead uses:

  - per-arm reward TAPES (bandit_reference.RewardTape) instead of one stream
  - a two-stage generate-then-score protocol instead of one sampled generation
  - candidate-only scoring, so there is no invalid parse and no fallback
  - temperature 0 throughout (the scoring stage is a deterministic argmax)

FROZEN STEERING SEMANTICS (§3.3), the load-bearing part of this file:
  - the RATIONALE pass gets NO alpha (diff_matrices=None). Steering the free
    reasoning would confound "alpha changed the decision" with "alpha changed
    the text the decision was conditioned on".
  - the ACTION pass injects alpha EXACTLY ONCE, at the last prefill token,
    which is the final token of ACTION_ANCHOR. Every action prompt therefore
    ends at that anchor and nothing may be appended after it.
"""
from __future__ import annotations

import numpy as np
import torch

import bandit_reference as br

RATIONALE_MAX_TOKENS = 64          # §3.3 frozen cap


# ─────────────────────────── token / prompt audit ───────────────────────────

def audit_prompt_tokens(vc, prompt: str, use_chat: bool) -> dict:
    """Attest what the model is actually fed and WHERE alpha lands.

    Two things are invisible in the decoded string and must be recorded rather
    than assumed (same checks as get_answer_bandit's debug_tokens block):

     (1) DOUBLE BOS — apply_chat_template emits <|begin_of_text|> and the
         tokenizer then adds another with add_special_tokens=True. It cancels
         in paired contrasts but must be KNOWN.
     (2) THE INJECTION SITE — steering is prefill-only into hs[:, -1, :], so
         the LAST token id here is where alpha is applied. For pv6 it must be
         the tail of ACTION_ANCHOR ("Button"), never a chat control token.
         A chat control token in that slot means the anchor did not end the
         prompt and the frozen semantics are broken.
    """
    ids = vc.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    head, tail = ids[:5], ids[-8:]
    last = ids[-1]
    decoded_last = vc.tokenizer.decode([last])
    return {
        "use_chat": use_chat,
        "n_tokens": len(ids),
        "head_ids": head,
        "tail_ids": tail,
        "double_bos": head[:2] == [128000, 128000],
        "injection_token_id": last,
        "injection_token": decoded_last,
        # The anchor ends with "Button"; the injection token must be part of
        # it. A leading space is normal for SentencePiece/BPE tokenizers.
        "injection_is_anchor_tail": decoded_last.strip() in ("Button", "button"),
        "prompt_endswith_anchor": prompt.endswith(br.ACTION_ANCHOR),
    }


def _wrap_chat(vc, prompt: str, use_chat: bool, anchor: bool) -> str:
    """Bare string by default (the NMD mask was extracted bare).

    With use_chat, the state goes in one user turn. The action anchor must sit
    AFTER the generation prompt so it is still the final text — putting it
    inside the user turn would leave a chat control token as the last token and
    move the injection site.
    """
    if not use_chat:
        return prompt
    if anchor:
        assert prompt.endswith(br.ACTION_ANCHOR), \
            "action prompt must end at the anchor before chat wrapping"
        body = prompt[: -len(br.ACTION_ANCHOR)].rstrip("\n")
    else:
        body = prompt
    wrapped = vc.tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True,
    )
    return wrapped + br.ACTION_ANCHOR if anchor else wrapped


# ───────────────────────────── candidate scoring ────────────────────────────

def score_candidates(vc, prompt: str, env: br.Environment,
                     diff_mtx) -> tuple[dict[str, float], str]:
    """Sequence log-probability of each arm label, given the action prompt.

    Returns (per-label logprob, argmax label). Deterministic: no sampling, no
    temperature, no fallback — the choice is always one of the K legal arms, so
    invalid_rate is structurally 0 rather than merely small.

    Scoring is FULL SEQUENCE log-prob (sum over the candidate's tokens) even
    when the audit says every candidate is single-token. Comparing first tokens
    only would be invalid the moment one arm label tokenizes differently, and
    for single-token candidates the sum is identical to the single-step logit
    comparison anyway — so the general path costs nothing and cannot silently
    become wrong if labels or tokenizer change.
    """
    suffixes = br.candidate_suffixes(env)
    tok_ids = [vc.tokenizer.encode(s, add_special_tokens=False) for s in suffixes]

    per_token_logits = vc.regenerate_logits_teacher_forcing(
        prompts=[prompt] * len(suffixes),
        answer_token_ids=tok_ids,
        diff_matrices=diff_mtx,
    )

    scores: dict[str, float] = {}
    for suffix, ids, logits in zip(suffixes, tok_ids, per_token_logits):
        lp = 0.0
        for k, tid in enumerate(ids):
            row = torch.from_numpy(np.asarray(logits[k], dtype=np.float32))
            lp += float(torch.log_softmax(row, dim=-1)[tid])
        # candidate_suffixes strips exactly the "Button" prefix, so this
        # reconstructs the arm label the environment keys on.
        scores[f"Button{suffix}"] = lp

    assert set(scores) == set(arm_labels := set(br.ARM_LABELS[:env.k])), \
        f"scored labels {set(scores)} != environment arms {arm_labels}"
    best = max(scores, key=scores.get)
    return scores, best


# ─────────────────────────────── the episode ────────────────────────────────

def run_reference_episode(
    vc,
    diff_mtx,
    seed: int,
    env: br.Environment,
    use_chat: bool = False,
    rationale_max_tokens: int = RATIONALE_MAX_TOKENS,
    attest: bool = False,
) -> dict:
    """One pv6 F-reference episode.

    diff_mtx is applied ONLY in the action pass. Pass None for alpha=0.
    """
    tape = br.RewardTape(seed, env)
    arm_map = tape.arm_map
    history: list[tuple[str, int]] = []
    choices: list[str] = []
    feedbacks: list[int] = []
    rationales_raw: list[str] = []
    rationales_clean: list[str] = []
    score_trace: list[dict] = []
    attestation: dict = {}

    if attest:
        attestation["candidate_tokenization"] = br.audit_candidate_tokenization(
            vc.tokenizer, env)

    for round_idx in range(env.horizon):
        # ---- Stage 1: free rationale, NO alpha ----------------------------
        r_prompt = _wrap_chat(
            vc, br.build_rationale_prompt(arm_map, history, round_idx, env),
            use_chat, anchor=False)
        # temperature is not passed: the rationale is greedy so an episode is
        # reproducible from (seed, env, alpha) alone.
        torch.manual_seed(seed * 100_003 + round_idx)
        out = vc.regenerate(
            inputs=[r_prompt],
            diff_matrices=None,                 # frozen: no alpha here
            max_new_tokens=rationale_max_tokens,
        )
        raw = out[0] if isinstance(out, list) else out
        clean = br.sanitize_rationale(raw)

        # ---- Stage 2: constrained action, alpha at the anchor -------------
        a_prompt = _wrap_chat(
            vc, br.build_action_prompt(arm_map, history, round_idx, env, clean),
            use_chat, anchor=True)

        if attest and round_idx in (0, min(10, env.horizon - 1)):
            attestation[f"round_{round_idx}"] = {
                "rationale_prompt": r_prompt,
                "action_prompt": a_prompt,
                "rationale_raw": raw,
                "rationale_clean": clean,
                "tokens": audit_prompt_tokens(vc, a_prompt, use_chat),
            }

        scores, arm = score_candidates(vc, a_prompt, env, diff_mtx)
        reward = tape.pull(arm)

        history.append((arm, reward))
        choices.append(arm)
        feedbacks.append(reward)
        rationales_raw.append(raw)
        rationales_clean.append(clean)
        score_trace.append({k: round(v, 4) for k, v in scores.items()})

    rec = br._package(seed, env, tape, choices, feedbacks, policy="model",
                      extra={
                          "rationales_raw": rationales_raw,
                          "rationales_clean": rationales_clean,
                          "candidate_scores": score_trace,
                          "invalid_rate": 0.0,   # structural, not measured
                          "use_chat": use_chat,
                          "rationale_max_tokens": rationale_max_tokens,
                      })
    if attest:
        rec["attestation"] = attestation
    return rec
