#!/bin/bash
# OLMo-3-7B-Instruct MMLU-E role difference (Bare, No-CoT, no steering, no mask, no hidden states): {task} expert vs non {task} expert.
# Usage: nohup env CUDA_VISIBLE_DEVICES=0 PY=<python with transformers>=4.57> bash run_mmlue_olmo3_roles.sh > mmlue_olmo3_roles.log 2>&1 &
# Output: components/olmo3/mmlue/role_bare_v1/ (exits if it already holds results). Score OFFLINE (reasoningTaskAnalysis/olmo3_mmlue_roles_v1).
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
MODEL_DIR="${MODEL_DIR:-allenai/Olmo-3-7B-Instruct}"
ANS_FILE="mmlue/role_bare_v1"
OUT="${BASE_DIR}/olmo3/${ANS_FILE}"
ROLES="{task} expert,non {task} expert"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
if ls "${OUT}"/*_answers.json "${OUT}"/summary_*.csv >/dev/null 2>&1; then
  echo "[x] ${OUT} already holds results; refusing to overwrite"; exit 1
fi
echo "olmo3 mmlue roles bare | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
mkdir -p "${OUT}"
"${PY}" - "${OUT}" "${MODEL_DIR}" "${ROLES}" <<'PYEOF'
import json, os, platform, subprocess, sys
out, model_dir, roles = sys.argv[1:4]
import torch, transformers
try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception: commit = None
meta = {"experiment": "olmo3-mmlue-roles-bare-v1", "model_dir": model_dir, "model": "olmo3", "size": "7B", "suite": "default",
        "use_E": True, "cot": False, "use_chat": False, "steering": "none", "mask": "none", "save_hidden_states": False,
        "roles": roles, "batch_size": 1, "host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "git_commit": commit}
json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
PYEOF
[ $? -eq 0 ] || { echo "[x] could not write run_meta.json (environment import failed)"; exit 1; }
"${PY}" get_answer_logits.py \
  --model olmo3 --model_dir "${MODEL_DIR}" --size 7B --suite default --use_E \
  --roles "${ROLES}" --base_dir "${BASE_DIR}" --ans_file "${ANS_FILE}"
rc=$?
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED rc=${rc}"
exit $rc
