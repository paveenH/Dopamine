#!/usr/bin/env python3
"""pv7 episode driver. Thin: loads model + mask, loops seeds, writes JSON.

Deliberately standalone rather than another branch inside the ~2100-line
`get_answer_bandit.py`. pv7 shares only the model and the mask with pv1-pv6;
its prompts, anchors, candidates, record schema and output tree are all
different, so a branch there would mean threading pv7 flags through a resume
key, a CSV schema and an argument surface built for a different protocol.
Everything protocol-specific lives in `bandit_pv7_episode`.

OUTPUT NAMING -- pv7 files are never disguised as pv6
-----------------------------------------------------
Files are `bandit_pv7_{env}_{size}_{ls}_{le}.json` under a pv7-only directory.
`evaluate_competence_gate.load_result_dir` globs `bandit_pv6_*.json`, so pv7
results are INVISIBLE to it by construction -- which is intended. The gate
RULES are frozen and reusable; the LOADER is not, so pv7 gets its own wrapper
(`evaluate_competence_gate_pv7.py`) that reads these files and calls the same
frozen `evaluate()`. Renaming a pv7 file to look like pv6 would make the two
protocols silently poolable, which they are not.

`environment.name` stays `reference_easy`: the environment genuinely did not
change between pv6 and pv7, which is why the frozen algorithmic baselines and
seed banks remain valid for pv7's gate.

RESUME
------
Keyed on `bandit_pv7_episode.resume_key`, which includes both instruction
versions. A stored episode is reused only when every behaviour-affecting knob
matches; changing the Stage 2 wording produces a different key rather than
silently returning the old trajectory.

Usage
-----
    python run_bandit_pv7_episodes.py --reference_environment easy \\
        --seeds 6 12 13 --ans_file .../pv7_easy_bare_smoke --attest
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import bandit_reference as br
import bandit_pv7_episode as ep


def load_mask(base_dir, hs, type_, mask_type, percentage, size, ls, le, alpha):
    """The NMD mask scaled by alpha. Returns (diff or None, TOP, n_layers).

    Returns None at alpha=0 so NO hook is registered anywhere -- "unsteered"
    must not be the same code path as "steered by zero".
    """
    mask_dir = os.path.join(base_dir, "mask", f"{hs}_{type_}_logits")
    name = f"{mask_type}_{percentage}_{ls}_{le}_{size}.npy"
    raw = np.load(os.path.join(mask_dir, name))
    top = max(1, int(percentage / 100 * raw.shape[1]))
    n_layers = int(sum(1 for row in raw if np.any(row != 0)))
    if alpha == 0:
        return None, top, n_layers
    return list(raw * alpha), top, n_layers


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
                    choices=["easy", "hard", "native_floor"])
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
    out_file = out_dir / (f"bandit_pv7_{args.reference_environment}"
                          f"_{args.size}_{ls}_{le}.json")

    key = ep.resume_key(args.reference_environment, args.rationale_alpha,
                        args.action_alpha, ls, le, args.seeds)
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
    diff, top, n_layers = load_mask(
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
        rec = ep.run_pv7_episode(
            vc, diff, seed=seed, env=env,
            rationale_alpha=args.rationale_alpha,
            action_alpha=args.action_alpha,
            attest=args.attest and i == 1)
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
            },
            "seeds": sorted(r["seed"] for r in runs),
            "runs": runs,
        }
        out_file.write_text(json.dumps(doc, indent=1))   # checkpoint each seed

    print(f"\nwrote {out_file}  ({len(runs)} episodes)")


if __name__ == "__main__":
    main()
