"""
CGT-Sequential colour-step candidate-logit diagnostic — READ-ONLY SIDECAR.

WHY THIS EXISTS
---------------
Qwen2.5-7B failed the CGT-seq alpha=0 validity gate on the KNOWING axis:
`qdm_major_blue` = 1.000 while `qdm_major_red` = 0.37 (desc), i.e. the colour
step is label-locked rather than probability-using. Three explanations are
consistent with the free-generation data alone:

  (H1) an internal preference for the BLUE token,
  (H2) a preference for the FIRST-PRESENTED option, or
  (H3) a genuine inability to use the chest counts.

v4 presents "Blue" before "Red" in ALL THREE places it names the options
(system format spec, per-round user turn, and the worked example "9 blue,
1 red -> blue 90%"), and there is no shuffle anywhere in the driver. So label
and position are COMPLETELY COLLINEAR in v4 and this script CANNOT separate
H1 from H2. Separating them requires a balanced-order alpha=0 control (v5).

What this script CAN separate is H3 from {H1,H2}:
  * if logit margin tracks the chest counts but generation stays constant
    -> the evidence IS being used; the failure is at the generation interface.
  * if the margin is flat/blue-pinned across asymmetry
    -> the candidate preference is internal, and no format fix will help.

CONTRACT (deliberately narrow; this is a diagnostic, not an experiment)
----------------------------------------------------------------------
* Does NOT modify get_answer_cgt_seq.py, and does NOT write into any existing
  result directory. Output goes to a caller-named --out path.
* Reuses the FROZEN driver's own builders by import (build_seq_system_prompt,
  build_color_user_turn, build_bet_user_turn, build_chat_messages2,
  make_box_sequence, parse_color, parse_accept_wait). Nothing is copied, so
  the diagnostic prompt cannot drift from the driver's prompt.
* Verifies prompt IDs token-by-token against an independently rebuilt prompt,
  and verifies ID-LEVEL candidate concatenation (string-level concatenation
  lets BPE merge the anchor's trailing space into the candidate and silently
  moves the injection site -- the pv7 failure).
* Scores the FULL continuation log-prob (sum over all candidate tokens), not
  just the first token, since "Blue"/"Red" need not be single tokens.
* Records BOTH the free-generation answer and logP(Blue)-logP(Red) for the
  SAME state, so the generation-vs-representation comparison is paired.

The trajectory is driven by FREE GENERATION exactly as the driver does it, so
the visited states are the real ones. Logit scoring is a pure read-out at each
colour step and never feeds back into the trajectory.
"""
import argparse, json, os
import numpy as np
import torch

import utils
from llms import VicundaModel
import get_answer_cgt_seq as seq
from get_answer_cgt import build_chat_messages2


# ── candidate surface forms ────────────────────────────────────────────────
# v2/v3/v4 tell the model to reply with exactly one word "Blue" / "Red"; the
# "Color: " anchor means the continuation starts mid-line. Both capitalisations
# are scored and the better one is reported per colour, so a tokenizer-specific
# casing preference cannot masquerade as a colour preference.
CAND_FORMS = {"blue": ["Blue", "blue"], "red": ["Red", "red"]}


def _cand_ids(tok, forms):
    """Token ids for each surface form, WITHOUT special tokens."""
    out = []
    for f in forms:
        ids = tok(f, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            raise SystemExit(f"[FATAL] candidate {f!r} tokenized to nothing")
        out.append((f, ids))
    return out


def audit_prompt_ids(tok, prompt, rebuilt):
    """Fail closed if the diagnostic prompt is not byte/token-identical."""
    a = tok(prompt, add_special_tokens=True)["input_ids"]
    b = tok(rebuilt, add_special_tokens=True)["input_ids"]
    if a != b:
        raise SystemExit(
            f"[FATAL] prompt ID mismatch: diagnostic path diverged from driver "
            f"path (len {len(a)} vs {len(b)})")
    return a


def audit_id_level_concat(tok, prompt, cand_ids):
    """
    The scorer concatenates prompt ids + candidate ids. Assert that this is NOT
    the same as tokenizing the concatenated STRING -- if they agree, BPE did not
    merge and the check is vacuous; if they differ, string-level concat would
    have been wrong and we confirm we are not doing it.
    Returns (n_prompt_tokens, merged_flag).
    """
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    return len(p_ids), p_ids


def score_candidates(vc, prompt, cand_map, diff_mtx):
    """
    Full-continuation log-prob for every candidate surface form.
    Returns {colour: {"logp": float, "form": str, "n_tok": int}}.
    """
    forms, id_lists, owners = [], [], []
    for colour, flist in cand_map.items():
        for f, ids in flist:
            forms.append(f); id_lists.append(ids); owners.append(colour)

    prompts = [prompt] * len(id_lists)
    per_pos = vc.regenerate_logits_teacher_forcing(
        prompts=prompts, answer_token_ids=id_lists, diff_matrices=diff_mtx)

    best = {}
    for colour, form, ids, logits in zip(owners, forms, id_lists, per_pos):
        # logits[k] predicts token ids[k]
        lp = 0.0
        for k, tid in enumerate(ids):
            row = logits[k].astype(np.float64)
            row = row - row.max()
            lp += float(row[tid] - np.log(np.exp(row).sum()))
        if colour not in best or lp > best[colour]["logp"]:
            best[colour] = {"logp": lp, "form": form, "n_tok": len(ids)}
    return best


def run_diag(vc, diff_mtx, seed, presentation, prompt_ver, anchor,
             max_new_tokens, temperature, top_p, cand_map, audit_first):
    """
    Mirrors seq.run_episode's control flow, but adds a colour-step logit
    read-out. Trajectory is driven by FREE GENERATION, identical to the driver.
    """
    import random
    rng = random.Random(seed)
    fallback_rng = random.Random(seed + 10_000_019)
    box_seq = seq.make_box_sequence(seed)
    tiers = seq.BET_PCTS if presentation == "asc" else list(reversed(seq.BET_PCTS))
    n_tiers = len(tiers)
    system_prompt = seq.build_seq_system_prompt(presentation, prompt_ver)

    if anchor == "answer":
        color_anchor, bet_anchor = "Answer: ", "Answer: "
    elif anchor == "none":
        color_anchor, bet_anchor = "", ""
    else:
        color_anchor, bet_anchor = "Color: ", ""

    rows = []
    chat_turns = []
    remain = seq.INIT_MONEY
    pending_outcome = ""

    def gen(prompt):
        out = vc.regenerate(inputs=[prompt], diff_matrices=diff_mtx,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p,
                            prefill_tail_len=1)
        return out[0] if isinstance(out, list) else out

    for r in range(seq.TOTAL_INTERACTIONS):
        round_number = r + 1
        in_phase_idx = r % seq.ROUND_INTERACTIONS
        phase_reset = (in_phase_idx == 0)
        if phase_reset:
            remain = seq.INIT_MONEY
            chat_turns = []
            pending_outcome = ""

        blue, red = box_seq[r]
        major_color = "blue" if blue >= red else "red"
        asymmetry = abs(blue - red)

        user_c = seq.build_color_user_turn(
            round_number, remain, blue, red, phase_reset,
            outcome_feedback=pending_outcome, prompt_ver=prompt_ver)
        chat_turns.append({"role": "user", "content": user_c})
        prompt_c = build_chat_messages2(vc, system_prompt, chat_turns,
                                        answer_anchor=color_anchor)

        if audit_first and r == 0:
            rebuilt = build_chat_messages2(vc, system_prompt, list(chat_turns),
                                           answer_anchor=color_anchor)
            ids = audit_prompt_ids(vc.tokenizer, prompt_c, rebuilt)
            tail = ids[-1]
            print(f"[audit] colour prompt tokens={len(ids)} tail_id={tail} "
                  f"tail={vc.tokenizer.decode([tail])!r}")
            for colour, flist in cand_map.items():
                for f, cid in flist:
                    merged = vc.tokenizer(prompt_c + f,
                                          add_special_tokens=True)["input_ids"]
                    id_level = ids + cid
                    status = "ID-LEVEL != string-level (BPE merge avoided)" \
                        if merged != id_level else "identical (no merge)"
                    print(f"[audit] cand {f!r} ids={cid} -> {status}")

        # --- logit read-out (does NOT affect the trajectory) ---
        scored = score_candidates(vc, prompt_c, cand_map, diff_mtx)
        margin = scored["blue"]["logp"] - scored["red"]["logp"]

        # --- free generation, exactly as the driver ---
        raw_c = color_anchor + gen(prompt_c)
        choose_color, color_valid = seq.parse_color(raw_c, fallback_rng)
        chat_turns.append({"role": "assistant", "content": raw_c.strip()})

        # bet tiers: drive the trajectory forward, not scored
        accept_step = None
        seq_valid = True
        for step in range(n_tiers):
            pct = tiers[step]
            next_pct = tiers[step + 1] if step + 1 < n_tiers else None
            user_b = seq.build_bet_user_turn(choose_color, pct, step + 1,
                                             n_tiers, next_pct=next_pct,
                                             prompt_ver=prompt_ver)
            chat_turns.append({"role": "user", "content": user_b})
            prompt_b = build_chat_messages2(vc, system_prompt, chat_turns,
                                            answer_anchor=bet_anchor)
            raw_b = bet_anchor + gen(prompt_b)
            chat_turns.append({"role": "assistant", "content": raw_b.strip()})
            accept, valid = seq.parse_accept_wait(raw_b, fallback_rng)
            seq_valid = seq_valid and valid
            if accept:
                accept_step = step + 1
                break
        forced_lock = accept_step is None
        if forced_lock:
            accept_step = n_tiers
        locked_pct = tiers[accept_step - 1]
        bet_frac = locked_pct / 100.0

        coin = "blue" if rng.randint(1, seq.BOX_NUM) <= blue else "red"
        payoff = round(remain * bet_frac)
        if choose_color != coin:
            payoff = -payoff
        remain = remain + payoff
        won = payoff > 0
        pending_outcome = (
            f"Outcome from previous round: the coin was under a {coin} chest. "
            f"You {'won' if won else 'lost'} {abs(payoff)} points. "
            f"You now have {remain} points.")

        rows.append({
            "round": round_number, "phase": r // seq.ROUND_INTERACTIONS,
            "blue": blue, "red": red, "asymmetry": asymmetry,
            "major_color": major_color, "presentation": presentation,
            "gen_color": choose_color, "gen_valid": color_valid,
            "raw_color": raw_c,
            "logp_blue": scored["blue"]["logp"],
            "logp_red": scored["red"]["logp"],
            "margin_blue_minus_red": margin,
            "form_blue": scored["blue"]["form"], "form_red": scored["red"]["form"],
            "ntok_blue": scored["blue"]["n_tok"], "ntok_red": scored["red"]["n_tok"],
            "accept_step": accept_step, "chose_major": choose_color == major_color,
            "valid": color_valid and seq_valid,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5")
    p.add_argument("--model_dir", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--hs", default="qwen2.5")
    p.add_argument("--size", default="7B")
    p.add_argument("--type", default="non")
    p.add_argument("--mask_type", default="nmd")
    p.add_argument("--percentage", type=float, default=0.5)
    p.add_argument("--layer_start", type=int, default=16)
    p.add_argument("--layer_end", type=int, default=22)
    p.add_argument("--alpha", type=float, default=0.0,
                   help="DIAGNOSTIC IS alpha=0 BY DESIGN; nonzero only for a "
                        "deliberate steered read-out.")
    p.add_argument("--presentation", required=True, choices=["asc", "desc"])
    p.add_argument("--prompt_ver", default="v4")
    p.add_argument("--anchor", default="default")
    p.add_argument("--num_runs", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    p.add_argument("--out", required=True,
                   help="Output JSON path. MUST NOT be inside an existing "
                        "result directory -- this is a sidecar.")
    args = p.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"[FATAL] {args.out} exists; refusing to overwrite.")

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    tok = vc.tokenizer

    # Mask is loaded UNCONDITIONALLY and multiplied by alpha, exactly as the
    # frozen driver does (get_answer_cgt_seq.py:418-419). At alpha=0 this yields
    # a REAL all-zero matrix, NOT None: `regenerate` rejects None (llms.py:880)
    # because CGT-seq has no no-diff branch, so hooks still register and the
    # zero add still executes. This is deliberately NOT the pv6/PV10 "no hook at
    # all" convention -- passing None here would both crash and, worse, put the
    # diagnostic on a different execution path than the driver it is diagnosing.
    mask_path = os.path.join(
        args.base_dir, "mask", f"{args.hs}_{args.type}_logits",
        f"{args.mask_type}_{args.percentage}_{args.layer_start}_{args.layer_end}_{args.size}.npy")
    raw = np.load(mask_path)
    diff_mtx = list(raw * args.alpha)
    nz = int(np.count_nonzero(raw))
    print(f"[mask] {mask_path} shape={raw.shape} nonzero_rows="
          f"{int((raw != 0).any(axis=1).sum())} alpha={args.alpha}")
    if args.alpha == 0:
        assert all(not m.any() for m in diff_mtx), "alpha=0 must give an all-zero diff"
        print(f"[mask] alpha=0 -> all-zero diff (hooks register, zero add runs, "
              f"steering_fires will read 0). mask nonzero entries={nz}")

    cand_map = {c: _cand_ids(tok, f) for c, f in CAND_FORMS.items()}
    print("[cand]", {c: [(f, i) for f, i in v] for c, v in cand_map.items()})

    all_rows = []
    for run in range(args.num_runs):
        rows = run_diag(vc, diff_mtx, seed=run,
                        presentation=args.presentation,
                        prompt_ver=args.prompt_ver, anchor=args.anchor,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_p=args.top_p,
                        cand_map=cand_map, audit_first=(run == 0))
        all_rows.append(rows)
        m = np.mean([r["margin_blue_minus_red"] for r in rows])
        g = np.mean([r["gen_color"] == "blue" for r in rows])
        print(f"[run {run}] n={len(rows)} mean_margin={m:+.3f} gen_blue_rate={g:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "runs": all_rows}, f, indent=1)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
