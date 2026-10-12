#!/bin/bash
# ============= InfoQuest information-gap selectivity (infoquest-v1) ==========
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_infoquest_v1.sh llama3  pilot  > iq_llama_pilot.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_infoquest_v1.sh qwen2.5 pilot  > iq_qwen_pilot.log  2>&1 &
# Formal is a SEPARATE invocation, run only after the pilot has been scored and
# the scoring guide frozen:
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_infoquest_v1.sh llama3  formal > iq_llama_formal.log 2>&1 &
#   cat the log immediately: a wrong PY exits 127 and the log looks empty.
#
# pilot  = 30 requests x 2 conditions, alpha=0 only        (60 replies / model)
# formal = 100 requests x 2 conditions, alpha=-4,0,+4      (600 replies / model)
# Pilot does NOT chain into formal. No scoring happens on the server.
# Reads ONE file: ${BASE_DIR}/benchmark/infoquest_v1.json (reviewed export). Existing cells are resumed only with identical provenance.
set -e

MODEL="${1:?usage: run_infoquest_v1.sh model(llama3-or-qwen2.5) stage(pilot-or-formal)}"
STAGE="${2:?usage: run_infoquest_v1.sh model(llama3-or-qwen2.5) stage(pilot-or-formal)}"
case "${MODEL}" in llama3|qwen2.5) ;; *) echo "[x] unknown model '${MODEL}'"; exit 1 ;; esac
case "${STAGE}" in pilot|formal) ;; *) echo "[x] unknown stage '${STAGE}'"; exit 1 ;; esac
echo "[args] MODEL=${MODEL} STAGE=${STAGE}"

WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-${WORK_DIR}/components}"
PY="${PY:-python}"
cd "${WORK_DIR}"

"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }
BENCH_FILE="${BASE_DIR}/benchmark/infoquest_v1.json"
[ -f "${BENCH_FILE}" ] || { echo "[x] missing ${BENCH_FILE} (run on the server: prepare_infoquest_v1.py prepare, review, then export)"; exit 1; }
echo "[env] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"${PY}" get_answer_infoquest_v1.py --model "${MODEL}" --stage "${STAGE}" --base_dir "${BASE_DIR}" --benchmark "${BENCH_FILE}"
