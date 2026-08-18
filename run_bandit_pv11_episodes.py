#!/usr/bin/env python3.10
"""PV11 driver: run the 160 frozen evidence states through the micro-episode.

Thin on purpose. Order of operations, and each step's reason:

  1. load and VERIFY the frozen state bank        (before any GPU time)
  2. resolve the mask and build the resume key    (fingerprints model + mask)
  3. load the model
  4. run, writing after EVERY episode so a crash loses at most one

ALPHA=0 ONLY, ENFORCED. `--alpha` accepts nothing else. The +-4 cells may be
run only after the alpha=0 manipulation gate passes, and if it fails the
protocol closes rather than being re-tuned -- so the driver must not make a
steered run reachable by a flag flip. Removing this guard is a protocol change,
not a convenience.

RESUME KEY
----------
Keyed on the state-bank CONTENT hash, not its length: two different 160-state
banks would otherwise resume into each other and return episodes from a
different design under this cell's name. It also carries the prompt/parser
versions and the model+mask fingerprint, so a regenerated mask under the same
filename is a different key.

FAIL-CLOSED CHECKS BEFORE THE MODEL LOADS
-----------------------------------------
The bank must reproduce from its builder and match both manifest digests. A
bank that does not reproduce makes every downstream number uninterpretable, and
finding that out after a multi-hour run is the expensive way to learn it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

import bandit_pv11 as p11
import bandit_pv11_episode as ep
import build_pv11_state_bank as bld

HERE = Path(__file__).resolve().parent


# ─────────────────────────────── bank loading ───────────────────────────────

def load_and_verify_bank(bank_path: Path, manifest_path: Path) -> tuple:
    """Load the frozen bank, or refuse to run.

    Three independent checks, because they fail for different reasons:
      * reproduces from the builder  -> the bank was not hand-edited
      * canonical digest matches     -> the manifest describes THIS bank
      * file digest matches          -> the bytes on disk are the ones hashed
    """
    if not bank_path.exists():
        raise SystemExit(f"state bank not found: {bank_path}")
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    bank = json.loads(bank_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    fresh = bld.build_bank()
    if bld.canonical(fresh) != bld.canonical(bank):
        raise SystemExit(
            f"{bank_path.name} does not reproduce from "
            f"build_pv11_state_bank.py -- the bank or the builder was edited. "
            f"Nothing downstream is citable until this is resolved.")
    if manifest.get("state_bank_canonical_sha256") != bld.sha256(bank):
        raise SystemExit(
            "manifest canonical digest does not describe this bank")
    actual_file = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    if manifest.get("state_bank_file_sha256") != actual_file:
        raise SystemExit(
            f"manifest file digest does not match the bytes on disk\n"
            f"  manifest {manifest.get('state_bank_file_sha256')}\n"
            f"  actual   {actual_file}")

    states = bank["states"]
    uids = [s["state_uid"] for s in states]
    if len(set(uids)) != len(uids):
        raise SystemExit("state bank contains duplicate state_uids")
    if len(states) != 160:
        raise SystemExit(f"expected 160 states, bank holds {len(states)}")
    return bank, manifest, states


def bank_fingerprint(bank: dict) -> str:
    return bld.sha256(bank)[:16]


# ─────────────────────────────── mask + keys ────────────────────────────────

def load_mask(base_dir, hs, type_, mask_type, percentage, size, ls, le, alpha):
    """The NMD mask scaled by alpha. Returns (diff or None, TOP, n_layers, path).

    alpha == 0 returns None, so the episode takes the generate() path and
    registers no hook at all. A zero-valued matrix would still register hooks
    and is NOT the same thing -- that distinction is what lets attestation
    assert exactly 0 fires for an unsteered cell.
    """
    mask_dir = os.path.join(base_dir, "mask", f"{hs}_{type_}_logits")
    name = f"{mask_type}_{percentage}_{ls}_{le}_{size}.npy"
    mask_path = os.path.join(mask_dir, name)
    if not os.path.exists(mask_path):
        raise SystemExit(f"mask not found: {mask_path}")
    raw = np.load(mask_path)
    n_layers = int((np.abs(raw).sum(axis=1) > 0).sum())
    top = int((np.abs(raw) > 0).sum(axis=1).max())
    if alpha == 0:
        return None, top, n_layers, mask_path
    return list(raw * alpha), top, n_layers, mask_path


def model_config_fingerprint(model_dir, hs, type_, mask_type, percentage,
                             size, mask_path) -> str:
    """Model + mask identity. Hashes the mask CONTENT, not just its filename."""
    h = hashlib.sha256()
    h.update(Path(mask_path).read_bytes())
    payload = {
        "model_dir": model_dir, "hs": hs, "type": type_,
        "mask_type": mask_type, "percentage": percentage, "size": size,
        "mask_sha256": h.hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resume_key(alpha, ls, le, bank, model_config) -> str:
    """Cell identity. Bank CONTENT is hashed, not merely its state count."""
    horizons = sorted({s["remaining_horizon"] for s in bank["states"]})
    tag = p11.interface_tag(
        (s["state_uid"] for s in bank["states"]), horizons)
    return (f"{tag}_bank{bank_fingerprint(bank)}_a{alpha}_L{ls}-{le}_"
            f"m{model_config}")


# ──────────────────────────────────── main ──────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--ans_file", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layers", default="11-20")
    ap.add_argument("--hs", default="llama3")
    ap.add_argument("--type", default="non")
    ap.add_argument("--mask_type", default="nmd")
    ap.add_argument("--percentage", default="0.5")
    ap.add_argument("--size", default="8B")
    ap.add_argument("--max_new_tokens", type=int,
                    default=ep.RATIONALE_MAX_TOKENS)
    # A block-restricted or truncated run CANNOT be gated: the gate requires
    # both complete blocks and treats a missing one as a FAILURE, not as its
    # rule not applying. These flags exist for smoke/debug only, and the
    # driver says so at startup rather than letting a partial file look like
    # a gateable result.
    ap.add_argument("--block", choices=["all", "commitment", "acquisition"],
                    default="all",
                    help="SMOKE/DEBUG ONLY -- a single block cannot be gated")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE/DEBUG ONLY -- run only the first N states; "
                         "a truncated run cannot be gated")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # ALPHA=0 ONLY. See the module docstring: the steered cells are gated on
    # the alpha=0 manipulation check, and a failing gate CLOSES the protocol.
    if args.alpha != 0.0:
        raise SystemExit(
            f"--alpha={args.alpha} refused. PV11 runs alpha=0 only until the "
            f"manipulation gate passes; if it fails the protocol closes and "
            f"no steered cells are run. Lifting this guard is a protocol "
            f"change, not a flag.")

    ls, le = (int(x) for x in args.layers.split("-"))

    # ---- 1. bank, verified BEFORE any GPU time -----------------------------
    bank, manifest, states = load_and_verify_bank(
        HERE / "pv11_state_bank.json", HERE / "pv11_state_manifest.json")
    if args.block != "all":
        states = [s for s in states if s["block"] == args.block]
    if args.limit:
        states = states[:args.limit]
    print(f"[bank] {len(states)} states  "
          f"canonical={manifest['state_bank_canonical_sha256'][:16]}")
    if args.block != "all" or args.limit:
        print(f"[WARNING] this is a PARTIAL run "
              f"(block={args.block}, limit={args.limit or 'none'}). "
              f"The gate requires BOTH complete blocks and will report FAIL "
              f"on this output -- it is for smoke/debug only, not for a gate "
              f"decision.")

    out_file = Path(args.ans_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # ---- 2. mask + resume key ---------------------------------------------
    mask_dir = os.path.join(args.base_dir, "mask",
                            f"{args.hs}_{args.type}_logits")
    mask_path = os.path.join(
        mask_dir,
        f"{args.mask_type}_{args.percentage}_{ls}_{le}_{args.size}.npy")
    if not os.path.exists(mask_path):
        raise SystemExit(f"mask not found: {mask_path}")
    model_cfg = model_config_fingerprint(
        args.model_dir, args.hs, args.type, args.mask_type,
        args.percentage, args.size, mask_path)
    key = resume_key(args.alpha, ls, le, bank, model_cfg)

    done: dict[str, dict] = {}
    if out_file.exists() and not args.overwrite:
        prev = json.loads(out_file.read_text())
        if prev.get("resume_key") == key:
            done = {r["state_uid"]: r for r in prev.get("runs", [])}
            print(f"[resume] {len(done)} episodes already stored")
        else:
            raise SystemExit(
                f"{out_file} holds a DIFFERENT configuration\n"
                f"  stored:  {prev.get('resume_key')}\n"
                f"  current: {key}\n"
                f"Use a separate --ans_file per cell, or --overwrite.")

    todo = [s for s in states if s["state_uid"] not in done]
    if not todo:
        print("nothing to do; every state is already stored")
        return

    # ---- 3. model ----------------------------------------------------------
    from llms import VicundaModel
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    diff, top, n_layers, _ = load_mask(
        args.base_dir, args.hs, args.type, args.mask_type, args.percentage,
        args.size, ls, le, args.alpha)

    horizons = sorted({s["remaining_horizon"] for s in bank["states"]})
    tag = p11.interface_tag(
        (s["state_uid"] for s in bank["states"]), horizons)
    print(f"protocol={p11.PROTOCOL_VERSION} bank={p11.STATE_BANK_VERSION} "
          f"parser={p11.POLICY_PARSER_VERSION}")
    print(f"layers={ls}-{le} (L={n_layers}) TOP={top}  alpha={args.alpha}")
    print(f"interface={tag}")
    print(f"expected fires: model_calls x {n_layers} "
          f"(0 at alpha=0; ceiling (H+1)x{n_layers})")
    print(f"[run] {len(todo)} states to go\n")

    # ---- 4. run ------------------------------------------------------------
    runs = list(done.values())
    for i, state in enumerate(todo, 1):
        t0 = time.time()
        # The episode attests its own fires and raises
        # EpisodeInfrastructureError on a mismatch, so a hook that stops firing
        # at episode 7 stops the run there rather than being discovered after
        # the whole budget is spent.
        rec = ep.run_pv11_episode(
            vc, state=state, diff_mtx=diff, alpha=args.alpha,
            max_new_tokens=args.max_new_tokens,
            n_steered_layers=n_layers,
            interface_tag=tag)
        runs.append(rec)

        payload = {
            "resume_key": key,
            "protocol_version": p11.PROTOCOL_VERSION,
            "state_bank_version": p11.STATE_BANK_VERSION,
            "policy_parser_version": p11.POLICY_PARSER_VERSION,
            "stage1_instruction_version": p11.STAGE1_INSTRUCTION_VERSION,
            "state_bank_canonical_sha256":
                manifest["state_bank_canonical_sha256"],
            "state_bank_file_sha256": manifest["state_bank_file_sha256"],
            "manifest_sha256": hashlib.sha256(
                (HERE / "pv11_state_manifest.json").read_bytes()).hexdigest(),
            "config": {
                "alpha": args.alpha, "layers": [ls, le],
                "n_steered_layers": n_layers, "top": top,
                "model_dir": args.model_dir, "mask_path": mask_path,
                "model_config": model_cfg,
                "max_new_tokens": args.max_new_tokens,
                "block": args.block,
                "interface_tag": tag,
            },
            "n_states_expected": len(states),
            "runs": runs,
        }
        out_file.write_text(json.dumps(payload, indent=1))

        fa = rec["first_action"]
        sec = rec["secondary_trajectory"]
        print(f"[{i}/{len(todo)}] {state['state_uid']:28s} "
              f"first={fa['kind']:7s} arm={fa['arm'] or '-':2s} "
              f"n_samples={sec['n_samples']:3d} "
              f"term={sec['termination_reason']:18s} "
              f"fires={rec['attestation']['steering_fires']:4d} "
              f"({time.time() - t0:.1f}s)")

    print(f"\nwrote {out_file}  ({len(runs)} episodes)")


if __name__ == "__main__":
    main()
