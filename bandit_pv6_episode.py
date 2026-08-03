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

    # HARD invariants, not observations. pv6 freezes the injection site, so a
    # violation must stop the run rather than be recorded and read later —
    # by then the whole episode was steered at the wrong token.
    bos_id = getattr(vc.tokenizer, "bos_token_id", None)
    if bos_id is not None and head[:2] == [bos_id, bos_id]:
        raise AssertionError(
            f"double BOS in the pv6 prompt (head={head[:4]}) — the chat "
            f"template's serialized BOS was not stripped before a tokenizer "
            f"call with add_special_tokens=True")
    if not prompt.endswith(br.ACTION_ANCHOR):
        raise AssertionError(
            "pv6 action prompt does not end at ACTION_ANCHOR; the alpha "
            "injection site is not the decision token")

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


def _strip_leading_bos(vc, text: str) -> str:
    """Remove a BOS string that apply_chat_template already serialized.

    REQUIRED, not cosmetic. Both downstream paths tokenize with
    add_special_tokens=True — vc.generate via its plain tokenizer call, and
    regenerate_logits_teacher_forcing at llms.py:505 (hardcoded, no flag to
    turn it off). So a chat template that emits <|begin_of_text|> as TEXT
    yields ids [128000, 128000, ...]. pv1-pv5 tolerated this because it
    cancels in paired contrasts, but pv6 attests the absence of double BOS as
    a hard invariant, so it must actually be prevented here rather than
    merely reported.
    """
    bos = getattr(vc.tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        return text[len(bos):]
    return text


def _wrap_chat(vc, prompt: str, use_chat: bool, anchor: bool) -> str:
    """Bare string by default (the NMD mask was extracted bare).

    FROZEN ROLE STRUCTURE (§3.3) — decided now, before any behavioural result,
    because it is a design choice and not something to pick from outcomes:

        user turn      = task state ONLY (the externally summarized statistics)
        assistant turn = the model's own rationale, then the action anchor

    The rationale is the model's OWN prior output, so it belongs in the
    assistant turn; feeding it back inside the user turn would present the
    model's reasoning as if the environment had said it, which is a different
    (and misleading) conditioning structure. Stage 2 is therefore a genuine
    assistant-turn CONTINUATION: the anchor extends the assistant message that
    already contains the rationale, and the next token is the arm letter.

    The anchor must be the final text so its last token is the alpha injection
    site; nothing may follow it, including chat control tokens.
    """
    if not use_chat:
        return prompt

    if not anchor:
        # Stage 1: state in the user turn, model writes the assistant turn.
        return _strip_leading_bos(vc, vc.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        ))

    assert prompt.endswith(br.ACTION_ANCHOR), \
        "action prompt must end at the anchor before chat wrapping"
    body = prompt[: -len(br.ACTION_ANCHOR)].rstrip("\n")
    # build_action_prompt appends the sanitized rationale after the state, so
    # split them back apart to place each in its proper role.
    state, sep, rationale = body.partition(br._RATIONALE_INSTRUCTION)
    if sep:
        user_content = state + sep
        assistant_content = rationale.strip()
    else:                      # empty rationale -> nothing to attribute
        user_content, assistant_content = body, ""

    wrapped = _strip_leading_bos(vc, vc.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    ))
    # The assistant turn is CONTINUED, never closed: no eot, so the anchor and
    # the arm letter extend the same assistant message as the rationale.
    if assistant_content:
        wrapped += assistant_content + "\n"
    return wrapped + br.ACTION_ANCHOR


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
    margins: list[float] = []
    attestation: dict = {}

    if attest:
        attestation["candidate_tokenization"] = br.audit_candidate_tokenization(
            vc.tokenizer, env)

    for round_idx in range(env.horizon):
        # ---- Stage 1: free rationale, NO alpha ----------------------------
        r_prompt = _wrap_chat(
            vc, br.build_rationale_prompt(arm_map, history, round_idx, env),
            use_chat, anchor=False)
        # vc.generate, NOT vc.regenerate: regenerate REQUIRES diff_matrices and
        # raises ValueError on None, so "pass None to disable steering" is not
        # a real API. generate registers no hooks at all, which is exactly the
        # frozen semantics — the rationale pass must be unsteered, not
        # steered-by-zero. temperature=0.0 makes generate greedy (do_sample
        # False), so an episode is reproducible from (seed, env, alpha) alone.
        torch.manual_seed(seed * 100_003 + round_idx)
        out = vc.generate(
            inputs=[r_prompt],
            max_new_tokens=rationale_max_tokens,
            temperature=0.0,
        )
        raw = out[0] if isinstance(out, list) else out
        clean = br.sanitize_rationale(raw)

        # ---- Stage 2: constrained action, alpha at the anchor -------------
        a_prompt = _wrap_chat(
            vc, br.build_action_prompt(arm_map, history, round_idx, env, clean),
            use_chat, anchor=True)

        # Audit EVERY round, not only the attested ones: the assertions inside
        # are what enforce the frozen injection site, and a violation that
        # first appears at round 30 must stop the run there. Only the verbose
        # record is limited to the attested rounds.
        tok_audit = audit_prompt_tokens(vc, a_prompt, use_chat)
        if attest and round_idx in (0, min(10, env.horizon - 1)):
            attestation[f"round_{round_idx}"] = {
                "rationale_prompt": r_prompt,
                "action_prompt": a_prompt,
                "rationale_raw": raw,
                "rationale_clean": clean,
                "tokens": tok_audit,
            }

        scores, arm = score_candidates(vc, a_prompt, env, diff_mtx)
        reward = tape.pull(arm)

        history.append((arm, reward))
        choices.append(arm)
        feedbacks.append(reward)
        rationales_raw.append(raw)
        rationales_clean.append(clean)
        # RAW floats, not rounded: the decision is an argmax over these, and
        # alpha is hypothesised to move them. Rounding to 4dp discards exactly
        # the near-tie resolution that would show a small alpha effect, and a
        # tie at 4dp is not a tie in the argmax that actually ran.
        ordered = sorted(scores.values(), reverse=True)
        margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else float("nan"))
        score_trace.append(dict(scores))

    rec = br._package(seed, env, tape, choices, feedbacks, policy="model",
                      extra={
                          "rationales_raw": rationales_raw,
                          "rationales_clean": rationales_clean,
                          "candidate_scores": score_trace,
                          # top1 - top2 sequence-logprob gap per round. The
                          # decision's confidence margin: a change here is
                          # visible even when the argmax does not flip, so it
                          # separates "alpha moved the policy" from "alpha
                          # moved the preference but not enough to switch".
                          "choice_margins": margins,
                          "invalid_rate": 0.0,   # structural, not measured
                          "use_chat": use_chat,
                          "rationale_max_tokens": rationale_max_tokens,
                      })
    if attest:
        rec["attestation"] = attestation
    return rec
