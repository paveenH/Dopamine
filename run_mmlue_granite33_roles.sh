#!/bin/bash
# Granite-3.3-8B-Instruct MMLU-E paired answers + full hidden states (Bare, No-CoT, no steering, no mask): {task} expert vs non {task} expert.
# Usage: nohup env CUDA_VISIBLE_DEVICES=0 PY=<python with transformers recent enough for Granite 3.3> bash run_mmlue_granite33_roles.sh > mmlue_granite33_roles.log 2>&1 &
# Output: components/answer/granite33/ (answers+summary+run_meta) and components/hidden_states/granite33/mmlue/*.h5 (last prompt token, all layers).
# Exits if either already exists; back up any earlier answer/granite33 first. Score OFFLINE (reasoningTaskAnalysis/granite33_mmlue_roles_v1).
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
MODEL_DIR="${MODEL_DIR:-ibm-granite/granite-3.3-8b-instruct}"
OUT="${BASE_DIR}/answer/granite33"
HS_DIR="${BASE_DIR}/hidden_states"
ROLES="{task} expert,non {task} expert"
cd "${WORK_DIR}" || { echo "[x] cannot cd ${WORK_DIR}"; exit 1; }
if compgen -G "${OUT}/*_answers.json" >/dev/null || compgen -G "${OUT}/summary_*.csv" >/dev/null; then
  echo "[x] ${OUT} already holds results; refusing to overwrite"; exit 1
fi
echo "granite33 mmlue roles bare | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} | host=$(hostname) | $(date)"
if compgen -G "${HS_DIR}/granite33/mmlue/*_8B.h5" >/dev/null; then
  echo "[x] ${HS_DIR}/granite33/mmlue already holds H5 files; refusing to overwrite"; exit 1
fi
mkdir -p "${OUT}"
"${PY}" - "${OUT}" "${MODEL_DIR}" "${ROLES}" <<'PYEOF'
import json, os, platform, subprocess, sys
out, model_dir, roles = sys.argv[1:4]
import torch, transformers
from transformers import AutoTokenizer
import utils
from template import select_templates
labels = select_templates(suite="default", use_E=True, cot=False)["labels"]
class _V: tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)   # same args as VicundaModel
opt_ids = utils.option_token_ids(_V, labels)      # same single-token assertion as the entry point
try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception: commit = None
meta = {"experiment": "granite33-mmlue-roles-bare-v1", "model_dir": model_dir, "model": "granite33", "size": "8B", "suite": "default",
        "use_E": True, "cot": False, "use_chat": False, "steering": "none", "mask": "none", "save_hidden_states": True, "answer_dir": out, "hs_dir": os.path.join(os.path.dirname(os.path.dirname(out)), "hidden_states", "granite33", "mmlue"),
        "roles": roles, "option_labels": labels, "option_token_ids": opt_ids, "batch_size": 1, "host": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "library_versions": {"torch": torch.__version__, "transformers": transformers.__version__}, "git_commit": commit}
json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
PYEOF
[ $? -eq 0 ] || { echo "[x] could not write run_meta.json (environment import failed)"; exit 1; }
"${PY}" get_answer_logits.py \
  --model granite33 --model_dir "${MODEL_DIR}" --size 8B --suite default --use_E \
  --roles "${ROLES}" --base_dir "${BASE_DIR}" --answer_dir "${OUT}" \
  --hs_dir "${HS_DIR}" --task_name mmlue --save
rc=$?
[ $rc -eq 0 ] && echo "[ok] finished $(date)" || echo "[x] FAILED rc=${rc}"
exit $rc
