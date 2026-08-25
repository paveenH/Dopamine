#!/usr/bin/env python3
"""
Guard tests for extract_entropy_confidence.py. No GPU, no server, no real model.

Every guard here exists because the corresponding defect produced a LOADABLE,
plausible-looking JSON rather than an error. A test that only ran once in a
scratch directory is not evidence, so these are committed and re-runnable.

Builds a synthetic HDF5 matching the Qwen selective schema (band [16,22),
stored layers [15..20]+27, final-layer pointer 6) plus a two-tensor fake
checkpoint, then mutates one thing at a time and asserts the extractor REJECTS
it. A guard that cannot be shown to fail on a real defect is not a guard.

    python3.10 test_extract_entropy_confidence.py      # local (macOS analysis box)
    python  test_extract_entropy_confidence.py         # server conda env

Exits non-zero on the first failure.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "extract_entropy_confidence.py"
PY = sys.executable

H, V, NUM_LAYERS = 16, 32, 28
BAND_S, BAND_E = 16, 22
STORED = [15, 16, 17, 18, 19, 20, 27]     # band + the MODEL's final layer (27, not 21)
FINAL_STORED_IDX = 6
N_SAMPLES, T_STEPS = 3, 5

_pass = _fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  ok   {name}")
    else:
        _fail += 1
        print(f"  FAIL {name}" + (f"\n       {detail}" if detail else ""))


def build(root: Path, *, meta_over=None, drop_qidx=False, dup_qidx=False,
          empty_decode=False, proj_len=None, n_done=None, eps=1e-6,
          tie=False, drop_lm_head=False, hidden=H, bad_layer_axis=False) -> Path:
    """Write a synthetic checkpoint + one H5 cell; return the H5 dir."""
    root.mkdir(parents=True, exist_ok=True)
    ck = root / "ckpt"
    ck.mkdir(exist_ok=True)
    json.dump({"rms_norm_eps": eps, "tie_word_embeddings": tie,
               "num_hidden_layers": NUM_LAYERS, "hidden_size": hidden,
               "vocab_size": V}, open(ck / "config.json", "w"))
    rng = np.random.default_rng(0)
    tensors = {"model.norm.weight": torch.ones(hidden)}
    head = torch.tensor(rng.normal(0, .05, (V, hidden)), dtype=torch.float32)
    if drop_lm_head:
        tensors["model.embed_tokens.weight"] = head
    else:
        tensors["lm_head.weight"] = head
    save_file(tensors, str(ck / "model.safetensors"))

    n_layers_axis = 3 if bad_layer_axis else len(STORED)
    with h5py.File(root / f"hs_gsm8k_7B_nocot_neutral_L{BAND_S}-{BAND_E}.h5", "w") as f:
        meta = f.create_group("meta")
        attrs = {"task": "gsm8k", "model": "qwen2.5", "size": "7B", "role": "neutral",
                 "character": "(none)", "prompt_template": "T", "cot": False,
                 "layer_start": BAND_S, "layer_end": BAND_E, "ema_alpha": 0.95,
                 "steer_alpha": 8.0, "steer_mode": "prefill_only",
                 "num_layers": NUM_LAYERS, "final_layer_idx_stored": FINAL_STORED_IDX,
                 "n_stored_layers": len(STORED),
                 "n_samples_planned": N_SAMPLES,
                 "n_samples_done": N_SAMPLES if n_done is None else n_done}
        attrs.update(meta_over or {})
        for k, v in attrs.items():
            if v is not None:
                meta.attrs[k] = v
        meta.attrs["stored_layer_indices"] = np.array(STORED, dtype=np.int32)

        samples = f.create_group("samples")
        for i in range(N_SAMPLES):
            g = samples.create_group(f"{i:04d}")
            T = 0 if (empty_decode and i == 0) else T_STEPS
            g.create_dataset("prefill_hs",
                             data=rng.normal(0, 1, (4, n_layers_axis, hidden)).astype(np.float16))
            g.create_dataset("decode_hs",
                             data=rng.normal(0, 1, (T, n_layers_axis, hidden)).astype(np.float16))
            g.create_dataset("x_decode_proj",
                             data=rng.normal(0, 1, (T if proj_len is None else proj_len,)).astype(np.float32))
            for k, v in {"question": "q", "gold_answer": "1", "pred_answer": "1",
                         "correct": 1, "difficulty": "easy", "generated": "g"}.items():
                g.attrs[k] = v
            if not drop_qidx:
                g.attrs["question_idx"] = 0 if dup_qidx else i
    return root


def run(h5_dir: Path, *, model_dir=None, ls=BAND_S, le=BAND_E, extra=()):
    cmd = [PY, str(SCRIPT), "--h5_dir", str(h5_dir),
           "--model_dir", str(h5_dir / "ckpt") if model_dir is None else str(model_dir),
           "--out_dir", str(h5_dir / "out"),
           "--layer_start", str(ls), "--layer_end", str(le),
           "--device", "cpu", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="eec_test_"))
    try:
        print("[1] happy path")
        d = build(tmp / "ok")
        rc, out = run(d, extra=["--expect_n_cells", "1"])
        check("clean cell extracts", rc == 0, out[-400:])

        if rc == 0:
            j = json.load(open(d / "out" / f"metrics_gsm8k_7B_nocot_neutral_ema0.95_L{BAND_S}-{BAND_E}.json"))
            m, r = j["meta"], j["data"][0]
            check("meta.steer_alpha", m.get("steer_alpha") == 8.0)
            check("meta.steer_mode", m.get("steer_mode") == "prefill_only")
            check("meta.rms_norm_eps from config", m.get("rms_norm_eps") == 1e-6)
            check("meta.vocab_size", m.get("vocab_size") == V)
            check("meta band from H5", (m.get("layer_start"), m.get("layer_end")) == (BAND_S, BAND_E))
            check("per-sample question_idx", "question_idx" in r)
            check("qidx covers 0..n-1",
                  sorted(x["question_idx"] for x in j["data"]) == list(range(N_SAMPLES)))
            check("entropy_norm present", "entropy_norm_decode" in r)
            check("entropy_norm == entropy/log(V)",
                  np.allclose(np.array(r["entropy_norm_decode"]),
                              np.array(r["entropy_decode"]) / np.log(V), rtol=1e-5))
            check("metric length == decode steps", len(r["entropy_decode"]) == T_STEPS)

        print("[2] guards must REJECT real defects")
        cases = [
            ("empty --model_dir",        lambda: run(build(tmp/"m_empty"), model_dir=""),                    "is EMPTY"),
            ("HF repo id refused",       lambda: run(build(tmp/"m_hfid"), model_dir="Qwen/Qwen2.5-7B-Instruct"), "no config.json"),
            ("wrong band on CLI",        lambda: run(build(tmp/"m_band"), ls=11, le=20),                     "band mismatch"),
            ("missing question_idx",     lambda: run(build(tmp/"m_qidx", drop_qidx=True)),                   "question_idx"),
            ("duplicate question_idx",   lambda: run(build(tmp/"m_dup", dup_qidx=True)),                     "duplicate question_idx"),
            ("missing steer_alpha",      lambda: run(build(tmp/"m_sa", meta_over={"steer_alpha": None})),    "steer_alpha"),
            ("missing steer_mode",       lambda: run(build(tmp/"m_sm", meta_over={"steer_mode": None})),     "steer_mode"),
            ("steer_mode contradicts α", lambda: run(build(tmp/"m_smx", meta_over={"steer_mode": "none"})),  "implies steer_mode"),
            ("empty decode",             lambda: run(build(tmp/"m_empt", empty_decode=True)),                "empty decode"),
            ("proj length mismatch",     lambda: run(build(tmp/"m_proj", proj_len=99)),                      "x_decode_proj"),
            ("truncated cell",           lambda: run(build(tmp/"m_trunc", n_done=2)),                        "n_samples_done"),
            ("missing n_samples_done",   lambda: run(build(tmp/"m_nod", meta_over={"n_samples_done": None})),"n_samples_done"),
            ("final-layer idx vs shape", lambda: run(build(tmp/"m_axis", bad_layer_axis=True)),              "outside the stored layer axis"),
            ("selective w/o pointer",    lambda: run(build(tmp/"m_ptr", meta_over={"final_layer_idx_stored": None})), "final_layer_idx_stored"),
            ("hidden-size mismatch",     lambda: run(build(tmp/"m_dim", hidden=8)),                          "hidden-size mismatch"),
            ("untied but no lm_head",    lambda: run(build(tmp/"m_tie", drop_lm_head=True)),                 "tie_word_embeddings=false"),
        ]
        for name, fn, needle in cases:
            rc, out = fn()
            check(f"rejects: {name}", rc != 0 and needle.lower() in out.lower(),
                  f"rc={rc}; wanted {needle!r}\n       {out[-300:]}")

        print("[3] overwrite protection")
        d2 = build(tmp / "ow")
        run(d2, extra=["--expect_n_cells", "1"])
        rc, out = run(d2, extra=["--expect_n_cells", "1"])
        check("second run refused", rc != 0 and "refusing to overwrite" in out.lower(), out[-300:])
        rc, _ = run(d2, extra=["--expect_n_cells", "1", "--allow_overwrite"])
        check("--allow_overwrite permits", rc == 0)

        print("[4] cell-count assertion")
        rc, out = run(build(tmp / "cnt"), extra=["--expect_n_cells", "7"])
        check("wrong cell count refused", rc != 0 and "expected 7" in out.lower(), out[-300:])

        print("[5] eps is model-specific but numerically negligible at real scale")
        sys.path.insert(0, str(HERE))
        from extract_entropy_confidence import rms_norm, logits_to_metrics  # noqa: E402
        rng = np.random.default_rng(7)
        hs = torch.tensor(rng.normal(0, 1.0, (8, H)), dtype=torch.float16)   # RMS ~ 1
        w = torch.ones(H)
        head = torch.tensor(rng.normal(0, .05, (V, H)), dtype=torch.float32)
        e6 = logits_to_metrics(rms_norm(hs, w, 1e-6).float() @ head.T)[0]
        e5 = logits_to_metrics(rms_norm(hs, w, 1e-5).float() @ head.T)[0]
        check("|Δentropy| < 1e-3 at RMS~1", float(np.abs(e6 - e5).max()) < 1e-3,
              f"got {float(np.abs(e6 - e5).max()):.3e}")

        print(f"\n{_pass} passed, {_fail} failed")
        return 1 if _fail else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
