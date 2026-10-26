#!/bin/bash
# Granite-3.3-8B-Instruct MMLU-E RSN activation (Bare, No-CoT): roles "{task} expert" + "non {task} expert", alpha 0 then +4, hidden band [11,20), 57 tasks, one model load.
# Usage: nohup env CUDA_VISIBLE_DEVICES=0 PY=python bash run_mmlue_granite33_rsn.sh > mmlue_granite33_rsn.log 2>&1 &
# Mask: components/mask/granite33_non_logits/nmd_0.5_11_20_8B.npy (built from hidden_states_mean/granite33/mmlue; reused only if identical).
# Output: components/granite33/mmlue/rsn_11_20_v1/mdf_{0,4}/ ; exits if either already holds answers. Score OFFLINE (reasoningTaskAnalysis/granite33_mmlue_rsn_v1).
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
MODEL_DIR="${MODEL_DIR:-ibm-granite/granite-3.3-8b-instruct}"
ANS_FILE="mmlue/rsn_11_20_v1"
OUT="${BASE_DIR}/granite33/${ANS_FILE}"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
if compgen -G "${OUT}/mdf_0/*_answers_*.json" >/dev/null || compgen -G "${OUT}/mdf_4/*_answers_*.json" >/dev/null; then
  echo "[x] ${OUT} already holds results; refusing to overwrite"; exit 1
fi
echo "granite33 mmlue rsn 11-20 | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
mkdir -p "${OUT}"
"${PY}" - "${BASE_DIR}" "${OUT}" "${MODEL_DIR}" <<'PYEOF'
import hashlib, json, os, platform, subprocess, sys
import numpy as np
base, out, model_dir = sys.argv[1:4]
import utils
from detection.nmd import get_nmd_mask
src = os.path.join(base, "hidden_states_mean", "granite33", "mmlue")
f_exp, f_non = os.path.join(src, "diff_mean_8B.npy"), os.path.join(src, "none_diff_mean_8B.npy")
exp, non = np.load(f_exp), np.load(f_non)
if exp.shape != non.shape or exp.ndim != 4: sys.exit(f"[x] mean shapes {exp.shape} / {non.shape}")
top_k = max(1, int(exp.shape[-1] * 0.5 / 100.0))
if top_k != 20: sys.exit(f"[x] top_k={top_k}, expected 20")
built = get_nmd_mask(exp, non, top_k=top_k, start=11, end=20)    # signed Expert - Non-expert
if built.shape != (40, 4096): sys.exit(f"[x] mask shape {built.shape}, expected (40, 4096)")
nz = sorted(int(i) for i in np.nonzero(np.any(built != 0, axis=1))[0])
if nz != sorted(utils.decoder_layer_range(11, 20)): sys.exit(f"[x] mask non-zero rows {nz} != decoder_layer_range(11,20)")
mask_path = os.path.join(base, "mask", "granite33_non_logits", "nmd_0.5_11_20_8B.npy")
if os.path.exists(mask_path):
    if not np.array_equal(np.load(mask_path), built): sys.exit(f"[x] existing mask {mask_path} differs from rebuilt one; refusing to overwrite")
else:
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    tmp = mask_path + ".tmp.npy"; np.save(tmp, built); os.replace(tmp, mask_path)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
import torch, transformers
try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception: commit = None
meta = {"experiment": "granite33-mmlue-rsn-11-20-v1", "model": "granite33", "model_dir": model_dir, "size": "8B", "suite": "default",
        "use_E": True, "cot": False, "use_chat": False, "roles": "{task} expert,non {task} expert", "alphas": [0, 4], "hidden_band": [11, 20],
        "decoder_rows": nz, "top_k": top_k, "tail_len": 1, "mask_path": mask_path, "mask_sha256": sha(mask_path),
        "mask_inputs_sha256": {"diff_mean_8B.npy": sha(f_exp), "none_diff_mean_8B.npy": sha(f_non)}, "direction": "Expert - Non-expert (signed)",
        "host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "git_commit": commit}
json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
print("[mask] ok", built.shape, "nonzero rows", nz, "sha", meta["mask_sha256"][:12])
PYEOF
[ $? -eq 0 ] || { echo "[x] mask/meta step failed"; exit 1; }
"${PY}" get_answer_regenerate_logits.py \
  --model granite33 --model_dir "${MODEL_DIR}" \
  --hs granite33 --size 8B --type non \
  --percentage 0.5 --mask_type nmd \
  --configs 0-11-20 4-11-20 \
  --roles "{task} expert,non {task} expert" \
  --suite default --E --tail_len 1 \
  --base_dir "${BASE_DIR}" --ans_file "${ANS_FILE}"
rc=$?
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED rc=${rc}"
exit $rc
