#!/usr/bin/env python3.10
"""PV10-A episode driver (fixed-budget control). Thin: verifies the frozen basis, loads model + mask,
loops seeds, writes JSON.

Identical to the PV10-B driver except it passes bandit_pv10a as the prompt
module, so COMMIT is withheld until the budget is spent. Environment, seed
bank, order bank, tapes and steering semantics are the SAME objects -- that is
what makes an A-vs-B difference attributable to the stopping decision alone.

PV10-A is a MECHANISM CONTROL, not a competence claim: forcing 100 samples
removes the model's stopping decision, so its accuracy is not comparable to a
self-paced PV10-B number.

Order of operations is deliberate and must not be reordered:

  1. verify the capability manifest reproduces  (no GPU time spent yet)
  2. resolve the mask and build the resume key   (fingerprints the model+mask)
  3. load the model
  4. run seeds, attesting steering per episode

Step 1 comes first so a changed evaluator, criteria, seed bank, order bank or
prescreen basis stops the run BEFORE hours of GPU time are spent producing
results that would not be citable anyway.

The orders come from the frozen bank-level `assign_orders` over the WHOLE seed
bank. Passing a subset would re-derive different display/initial-pull orders for
the same seeds and silently break pairing against an already-run cell, so the
driver requires the full bank and refuses a subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import bandit_pv10 as p10
import bandit_pv10a as p10a
import bandit_pv10_episode as ep
import bandit_reference as br
import evaluate_pv10_capability as cap


def load_mask(base_dir, hs, type_, mask_type, percentage, size, ls, le, alpha):
    """The NMD mask scaled by alpha. Returns (diff or None, TOP, n_layers, path).

    alpha == 0 returns None, so `run_pv10_episode` takes the generate() path and
    registers no hook at all. A zero-valued matrix would still register hooks
    and is NOT the same thing.
    """
    mask_dir = os.path.join(base_dir, "mask", f"{hs}_{type_}_logits")
    name = f"{mask_type}_{percentage}_{ls}_{le}_{size}.npy"
    mask_path = os.path.join(mask_dir, name)
    raw = np.load(mask_path)
    n_layers = int((np.abs(raw).sum(axis=1) > 0).sum())
    top = int((np.abs(raw) > 0).sum(axis=1).max())
    if alpha == 0:
        return None, top, n_layers, mask_path
    return list(raw * alpha), top, n_layers, mask_path


def model_config_fingerprint(model_dir, hs, type_, mask_type, percentage,
                             size, mask_path) -> str:
    """Model + mask identity for the resume key.

    Hashes the mask CONTENT, not just its filename: a regenerated mask under the
    same name is a different intervention and must not resume into this cell.
    """
    h = hashlib.sha256()
    h.update(Path(mask_path).read_bytes())
    payload = {
        "model_dir": model_dir, "hs": hs, "type": type_,
        "mask_type": mask_type, "percentage": percentage, "size": size,
        "mask_sha256": h.hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resume_key(alpha, ls, le, seeds, model_config) -> str:
    """Cell identity. Seed CONTENT is hashed, not merely its length.

    Two different 20-seed sets would otherwise resume into each other and return
    episodes from different environments under this cell's name.
    """
    return (f"{p10a.interface_tag(4, seeds)}_a{alpha}_L{ls}-{le}_"
            f"m{model_config}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--model_dir", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--hs", default="llama3")
    ap.add_argument("--type", default="non")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", type=float, default=0.5)
    ap.add_argument("--layers", default="11-20")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--base_dir", default="/data1/paveen/Dopamine/components")
    ap.add_argument("--ans_file", required=True,
                    help="output DIRECTORY; one per alpha cell (the detail "
                         "JSON name carries no alpha, so sharing a dir would "
                         "overwrite)")
    ap.add_argument("--max_new_tokens", type=int,
                    default=p10.RATIONALE_MAX_TOKENS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_basis_check", action="store_true",
                    help="DEBUG ONLY; a run with this flag is not citable")
    args = ap.parse_args()

    # ---- 1. frozen basis, BEFORE any GPU time --------------------------------
    if args.skip_basis_check:
        print("WARNING: --skip_basis_check set; this run is NOT citable")
    else:
        if not cap.MANIFEST_PATH.exists():
            raise SystemExit(
                f"no capability manifest at {cap.MANIFEST_PATH}. Freeze it "
                f"BEFORE running any cell: "
                f"python3.10 evaluate_pv10_capability.py --freeze")
        diffs = cap.verify_basis(json.loads(cap.MANIFEST_PATH.read_text()))
        if diffs:
            raise SystemExit(
                f"BASIS MISMATCH in {diffs}. The frozen capability basis does "
                f"not reproduce, so nothing this run produces would be "
                f"citable. Fix the divergence before spending GPU time.")
        print(f"[basis] {cap.MANIFEST_PATH.name} reproduces")

    ls, le = (int(x) for x in args.layers.split("-"))
    # PV10's own prescreened probability vector -- NOT a pv6 reference env.
    # The seed bank is still drawn from pv6 `easy` because both are K=4 and
    # `build_seed_bank` depends only on (seed, k), which is what keeps the two
    # environments' arm maps and counterbalancing identical.
    env = br.Environment(
        name="pv10_bai_candidate", k=4,
        probs=(0.60, 0.50, 0.40, 0.30), horizon=p10.TOTAL_BUDGET,
        is_reference=False, competence_eligible=False)

    # ---- the seed bank must be complete -------------------------------------
    bank = br.build_seed_bank(br.get_environment("easy"), n=20)
    if sorted(args.seeds) != sorted(bank):
        raise SystemExit(
            f"--seeds must be the WHOLE frozen 20-seed bank.\n"
            f"  expected: {bank}\n"
            f"  got:      {sorted(args.seeds)}\n"
            f"Orders are assigned at the CELL level, so a subset re-derives "
            f"different display/initial-pull orders for the same seeds and "
            f"would silently break pairing against other alpha cells.")
    orders = p10.assign_orders(bank, env.k)

    out_dir = Path(args.ans_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bandit_pv10a_{env.name}_{args.size}_{ls}_{le}.json"

    # ---- 2. mask + resume key, BEFORE loading the model ---------------------
    mask_dir = os.path.join(args.base_dir, "mask", f"{args.hs}_{args.type}_logits")
    mask_path = os.path.join(
        mask_dir, f"{args.mask_type}_{args.percentage}_{ls}_{le}_{args.size}.npy")
    if not os.path.exists(mask_path):
        raise SystemExit(f"mask not found: {mask_path}")
    model_cfg = model_config_fingerprint(
        args.model_dir, args.hs, args.type, args.mask_type,
        args.percentage, args.size, mask_path)
    key = resume_key(args.alpha, ls, le, bank, model_cfg)

    done: dict[int, dict] = {}
    if out_file.exists() and not args.overwrite:
        prev = json.loads(out_file.read_text())
        if prev.get("resume_key") == key:
            done = {r["seed"]: r for r in prev.get("runs", [])}
            print(f"[resume] {len(done)} episodes already stored")
        else:
            raise SystemExit(
                f"{out_file} holds a DIFFERENT configuration\n"
                f"  stored:  {prev.get('resume_key')}\n"
                f"  current: {key}\n"
                f"Use a separate --ans_file per alpha cell, or --overwrite.")

    todo = [s for s in bank if s not in done]
    if not todo:
        print("nothing to do; every seed is already stored")
        return

    # ---- 3. model ----------------------------------------------------------
    from llms import VicundaModel
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    diff, top, n_layers, _ = load_mask(
        args.base_dir, args.hs, args.type, args.mask_type, args.percentage,
        args.size, ls, le, args.alpha)

    tag = p10a.interface_tag(env.k, bank)
    print(f"env={env.name} K={env.k} T={p10.TOTAL_BUDGET} "
          f"layers={ls}-{le} (L={n_layers}) TOP={top}")
    print(f"alpha={args.alpha}  interface={tag}")
    print(f"expected fires: model_calls x {n_layers} x 1 "
          f"(0 at alpha=0; 873 for a full-budget episode at L=9)")

    # ---- 4. run ------------------------------------------------------------
    runs = list(done.values())
    for i, seed in enumerate(todo, 1):
        t0 = time.time()
        # run_pv10_episode attests its own fires per episode and raises
        # EpisodeInfrastructureError on a mismatch, so a hook that stops firing
        # at episode 7 stops the run there rather than being discovered after
        # the whole budget is spent.
        rec = ep.run_pv10_episode(
            vc, seed=seed, env=env, orders=orders[seed],
            diff_mtx=diff, alpha=args.alpha,
            total_budget=p10.TOTAL_BUDGET,
            max_new_tokens=args.max_new_tokens,
            n_steered_layers=n_layers,
            interface_tag=tag,
            prompt_module=p10a)
        runs.append(rec)

        payload = {
            "resume_key": key,
            "capability_manifest_sha256": (
                cap._sha256_file(cap.MANIFEST_PATH)
                if cap.MANIFEST_PATH.exists() else None),
            "prescreen_manifest_sha256": (
                cap._sha256_file(cap.PRESCREEN_MANIFEST)
                if cap.PRESCREEN_MANIFEST.exists() else None),
            "evaluator_sha256": cap._sha256_file(
                Path(cap.__file__)),
            "config": {
                "alpha": args.alpha, "layers": [ls, le],
                "n_steered_layers": n_layers, "top": top,
                "model_dir": args.model_dir, "mask_path": mask_path,
                "model_config": model_cfg,
                "max_new_tokens": args.max_new_tokens,
                "environment": {"name": env.name, "k": env.k,
                                "probs": list(env.probs),
                                "total_budget": p10.TOTAL_BUDGET},
                "seed_bank": bank,
                "interface_tag": tag,
            },
            "runs": runs,
        }
        out_file.write_text(json.dumps(payload, indent=1))

        dt = time.time() - t0
        print(f"  [{i}/{len(todo)}] seed {seed:>3}  "
              f"{rec['termination_reason']:<18} "
              f"tau={str(rec['tau']):>4}  calls={rec['model_calls']:>3}  "
              f"fires={rec['steering_fires']:>4}  "
              f"commit={rec['committed_arm']}  "
              f"correct={rec['commit_correct']}  ({dt:.0f}s)")

    print(f"\nwrote {out_file}  ({len(runs)} episodes)")


if __name__ == "__main__":
    main()
