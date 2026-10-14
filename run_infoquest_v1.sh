#!/bin/bash
# ============= InfoQuest information-gap selectivity (infoquest-v1) ==========
# Usage (server, from /data1/paveen/Dopamine):
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_infoquest_v1.sh llama3  pilot  > iq_llama_pilot.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=1 nohup bash run_infoquest_v1.sh qwen2.5 pilot  > iq_qwen_pilot.log  2>&1 &
# Formal is a SEPARATE invocation (protocol: infoquest/FORMAL_PROTOCOL_r2.md; guide-2 frozen):
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_infoquest_v1.sh llama3  formal > iq_llama_formal.log 2>&1 &
#   cat the log immediately: a wrong PY exits 127 and the log looks empty.
#
# pilot  = 30 requests x 2 conditions, alpha=0 only        (60 replies / model)
# formal = 100 requests x 2 conditions, alpha=-4,0,+4      (600 replies / model)
# Pilot does NOT chain into formal. No scoring happens on the server.
# max_new_tokens=1024 is set HERE (MAX_NEW_TOKENS) and is the same for pilot and formal; results go to <model>/infoquest_v1_r2/.
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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
cd "${WORK_DIR}"

"${PY}" -c "import numpy, torch" || { echo "[x] ${PY} cannot import numpy/torch"; exit 1; }
BENCH_FILE="${BASE_DIR}/benchmark/infoquest_v1.json"
[ -f "${BENCH_FILE}" ] || { echo "[x] missing ${BENCH_FILE} (run on the server: prepare_infoquest_v1.py prepare, review, then export)"; exit 1; }
# Frozen r2 benchmark (formal protocol): a different file would change every scenario/prompt. Override only deliberately.
EXPECT_BENCH_SHA256="${EXPECT_BENCH_SHA256:-a212fdd67577d20d322a2b8b0e37470c437fc2e0efcbd659c38da2d65f7fd0bb}"
if command -v sha256sum >/dev/null 2>&1; then GOT_SHA="$(sha256sum "${BENCH_FILE}" | cut -d' ' -f1)"; else GOT_SHA="$(shasum -a 256 "${BENCH_FILE}" | cut -d' ' -f1)"; fi
[ "${GOT_SHA}" = "${EXPECT_BENCH_SHA256}" ] || { echo "[x] benchmark sha256 ${GOT_SHA} != frozen ${EXPECT_BENCH_SHA256}"; exit 1; }
echo "[env] benchmark sha256 OK ${GOT_SHA:0:16}"
echo "[env] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[env] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"${PY}" get_answer_infoquest_v1.py --model "${MODEL}" --stage "${STAGE}" --base_dir "${BASE_DIR}" --benchmark "${BENCH_FILE}" --max_new_tokens "${MAX_NEW_TOKENS}"
