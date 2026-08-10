#!/usr/bin/env python3
"""PV9 episode driver. Thin: loads model + mask, loops seeds, writes JSON.

PV9 = pv8 plus four Stage-1 modifications (self-relevant score, untried-arm
exploration cue, 128-token/50-word generation control, explicit Bernoulli
structure) and a `#` stop string. Stage 2 is byte-unchanged. Everything
protocol-specific lives in `bandit_pv9` / `bandit_pv9_episode`.

TWO ENVIRONMENTS, DIFFERENT JOBS -- do not read them the same way
------------------------------------------------------------------
`easy` (.75/.25x3) is the competence-eligible environment and carries
discovery, one-shot-zero revisit, information investment and outcome.

`neartie` (.60/.55/.25/.25) exists because `easy` structurally has no states
where two arms compete on close empirical value, so a precision effect has
nowhere to show. It is `competence_eligible=False`: with a true gap of 0.05
and ~25 pulls per arm, the empirical standard error is DOUBLE the gap, so a
high SuffFail there measures the environment, not the policy. Its gate output
is diagnostic only, and its online alphas will diverge into different states
within a few rounds -- so it gives ECOLOGICAL evidence (does alpha change
outcomes in a near-tie world), not matched-state mechanism evidence. The
matched-state precision test is the frozen probe, which is a later step.

OUTPUT NAMING -- PV9 files are never disguised as pv6/pv7/pv8
--------------------------------------------------------------
Files are `bandit_pv9_{env}_{size}_{ls}_{le}.json` in a PV9-only tree. The
pv6/pv7/pv8 gate loaders glob their own protocol's names, so PV9 results are
invisible to them by construction -- intended, since the protocols are not
poolable. The gate RULES are frozen and reused; only the LOADER is new.

`environment.name` stays `reference_{env}` because the environments are
genuine `bandit_reference` entries, which is what keeps the frozen seed banks
and algorithmic baselines the correct comparison basis.

BASELINE MANIFEST
-----------------
PV9 uses `bandit_pv9_baseline_manifest.json`, which covers easy AND neartie
and leaves the pv6 manifest byte-unchanged. Its `easy` block is asserted
identical to pv6's, so PV9-Easy and pv8-Easy are judged against one basis.

ALPHA=0 IS NOT INHERITED
------------------------
The Stage-1 prompt changed, so no stored pv7/pv8 alpha=0 cell is this
protocol's baseline. Run all three alphas here, same seeds, same tapes, one
directory per alpha.

Usage
-----
    python run_bandit_pv9_episodes.py --reference_environment easy \\
        --seeds 6 12 13 --ans_file .../pv9_easy_bare_smoke --attest
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import bandit_reference as br
import bandit_pv9_episode as ep


def load_mask(base_dir, hs, type_, mask_type, percentage, size, ls, le, alpha):
    """The NMD mask scaled by alpha. Returns (diff or None, TOP, n_layers).

    Returns None at alpha=0 so NO hook is registered anywhere -- "unsteered"
    must not be the same code path as "steered by zero".
    """
    mask_dir = os.path.join(base_dir, "mask", f"{hs}_{type_}_logits")
    name = f"{mask_type}_{percentage}_{ls}_{le}_{size}.npy"
    mask_path = os.path.join(mask_dir, name)
    raw = np.load(mask_path)
    top = max(1, int(percentage / 100 * raw.shape[1]))
    n_layers = int(sum(1 for row in raw if np.any(row != 0)))
    if alpha == 0:
        return None, top, n_layers, mask_path
    return list(raw * alpha), top, n_layers, mask_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--hs", default="llama3")
    ap.add_argument("--type", default="non")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--layers", default="11-20",
                    help="mask layer band, e.g. 11-20 (llama3) / 16-22 (qwen)")
    ap.add_argument("--reference_environment", default="easy",
                    choices=["easy", "neartie", "hard", "native_floor"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rationale_alpha", type=float, default=0.0)
    ap.add_argument("--action_alpha", type=float, default=0.0)
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--ans_file", required=True,
                    help="output DIRECTORY; one alpha cell per directory")
    ap.add_argument("--attest", action="store_true",
                    help="store full prompts/tokens for rounds 0 and 10")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ls, le = (int(x) for x in args.layers.split("-"))
    env = br.get_environment(args.reference_environment)
    out_dir = Path(args.ans_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (f"bandit_pv9_{args.reference_environment}"
                          f"_{args.size}_{ls}_{le}.json")

    # The mask path is needed for the resume key, so resolve it BEFORE the
    # stored-file check. Building the key without the model/mask fingerprint
    # would let a changed mask or model_dir resume into this cell silently.
    mask_dir = os.path.join(args.base_dir, "mask", f"{args.hs}_{args.type}_logits")
    mask_path = os.path.join(
        mask_dir, f"{args.mask_type}_{args.percentage}_{ls}_{le}_{args.size}.npy")
    if not os.path.exists(mask_path):
        raise SystemExit(f"mask not found: {mask_path}")
    model_cfg = ep.model_config_fingerprint(
        args.model_dir, args.hs, args.type, args.mask_type,
        args.percentage, args.size, mask_path)
    key = ep.resume_key(args.reference_environment, args.rationale_alpha,
                        args.action_alpha, ls, le, args.seeds,
                        model_config=model_cfg)
    done = {}
    if out_file.exists() and not args.overwrite:
        prev = json.loads(out_file.read_text())
        if prev.get("resume_key") == key:
            done = {r["seed"]: r for r in prev.get("runs", [])}
            print(f"[resume] {len(done)} episodes already stored for {key}")
        else:
            raise SystemExit(
                f"{out_file} holds a DIFFERENT configuration\n"
                f"  stored:  {prev.get('resume_key')}\n"
                f"  current: {key}\n"
                "Use a separate --ans_file per alpha cell, or --overwrite.")

    todo = [s for s in args.seeds if s not in done]
    if not todo:
        print("nothing to do; every seed is already stored")
        return

    from llms import VicundaModel
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()          # matches get_answer_bandit.py:1772
    alpha_for_mask = args.rationale_alpha or args.action_alpha
    diff, top, n_layers, _mask_path = load_mask(
        args.base_dir, args.hs, args.type, args.mask_type, args.percentage,
        args.size, ls, le, alpha_for_mask)

    expect = ep.expected_fires(args.rationale_alpha, args.action_alpha,
                               n_layers, env.k, env.horizon)
    print(f"env={env.name} K={env.k} T={env.horizon} layers={ls}-{le} "
          f"(L={n_layers}) TOP={top}")
    print(f"rationale_alpha={args.rationale_alpha} "
          f"action_alpha={args.action_alpha}")
    print(f"expected steering_fires per episode: {expect}")

    runs = list(done.values())
    for i, seed in enumerate(todo, 1):
        t0 = time.time()
        rec = ep.run_pv9_episode(
            vc, diff, seed=seed, env=env,
            rationale_alpha=args.rationale_alpha,
            action_alpha=args.action_alpha,
            # ALWAYS attest the first episode of a cell. The Stage-1-only
            # isolation assertions live in run_pv9_episode and only fire when
            # an attestation exists, so an unattended cell would otherwise
            # never verify that the cue/score/history reached Stage 1 and
            # stayed out of Stage 2. `--attest` additionally stores round 10.
            attest=args.attest or i == 1)
        fires = rec.get("steering_fires")
        # Attest per episode, not once at the end: a hook that stops firing at
        # episode 7 must stop the run there rather than be discovered after
        # the full budget is spent.
        #
        # `None` (the model object has no counter) is a FAILURE, not a pass.
        # Skipping the check when the counter is absent would mean the cells
        # with no attestation are exactly the ones that never get attested --
        # the intervention would be unverified precisely where it is
        # unverifiable, which is the opposite of fail-closed.
        if fires != expect:
            raise SystemExit(
                f"seed {seed}: steering_fires {fires} != expected {expect}"
                + (" -- the model object exposes no steering_fire_count(), so "
                   "the injection cannot be attested at all" if fires is None
                   else "") +
                ". The intervention is not what the config claims; behaviour "
                "read off this cell would be uninterpretable.")
        runs.append(rec)
        print(f"  [{i}/{len(todo)}] seed={seed} "
              f"late_opt={rec['late_opt_frac']:.3f} "
              f"regret={rec['cum_regret']:.1f} "
              f"policy_parse={rec['policy_parse_rate']:.2f} "
              f"follows={rec['action_follows_policy_rate']:.2f} "
              f"stop={rec['stop_reason_counts']} "
              f"({time.time() - t0:.0f}s)", flush=True)

        doc = {
            "protocol": ep.PROTOCOL_VERSION,
            "resume_key": key,
            "environment": {"name": f"reference_{args.reference_environment}"
                            if args.reference_environment != "native_floor"
                            else "native_floor",
                            "k": env.k, "horizon": env.horizon},
            "alpha": alpha_for_mask,
            "rationale_alpha": args.rationale_alpha,
            "action_alpha": args.action_alpha,
            "layers": [ls, le],
            "n_steered_layers": n_layers,
            "config": {
                "model": args.model, "size": args.size,
                "mask_type": args.mask_type, "percentage": args.percentage,
                "top": top,
                "stage1_instruction_version": ep.STAGE1_INSTRUCTION_VERSION,
                "stage2_instruction_version": ep.STAGE2_INSTRUCTION_VERSION,
                "policy_parser_version": ep.POLICY_PARSER_VERSION,
                "rationale_max_tokens": ep.RATIONALE_MAX_TOKENS,
                "expected_steering_fires": expect,
                "history_block_version": ep.HISTORY_BLOCK_VERSION,
                "score_block_version": ep.SCORE_BLOCK_VERSION,
                "untried_cue_version": ep.UNTRIED_CUE_VERSION,
                "stop_strings_version": ep.STOP_STRINGS_VERSION,
                "stop_strings": list(ep.p9.STOP_STRINGS),
                "rationale_word_limit": ep.p9.RATIONALE_WORD_LIMIT,
                "competence_eligible": env.competence_eligible,
                "baseline_manifest": "bandit_pv9_baseline_manifest.json",
                "model_config": model_cfg,
            },
            "seeds": sorted(r["seed"] for r in runs),
            "runs": runs,
        }
        # Atomic: write a temp file then rename. A plain write interrupted
        # mid-flush leaves a truncated JSON that the next resume would read as
        # the authoritative checkpoint -- the exact risk of an unattended run.
        tmp = out_file.with_suffix(out_file.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(out_file)

    print(f"\nwrote {out_file}  ({len(runs)} episodes)")


if __name__ == "__main__":
    main()
