#!/bin/bash
# ============ Cambridge Gambling Task (CGT) — RSN α dose-response (Llama3) ============
# Probability-transparent betting: the model is told #blue / #red every round, so any
# bet-size shift under α can ONLY be attributed to risk-taking, not accuracy/confidence
# — this removes the "more confident" confound of §3.1 Confidence Betting.
#
# Faithful port of the Near-Optimal repo CGT (8 phases × 8 rounds, 8 box ratios,
# 5 bet tiers, simultaneous 10-choice, ×2 payoff, independent coin). Mechanics verified
# against /Users/paveenhuang/Downloads/Benchmark/Near-Optimal/cambridge_gambling_task.
# Spec lives in Ada_Dopamine.md §3.3「實作規格」.
#
# Design (confirmed 2026-06): Llama3-8B, layers 11–20, nmd mask, choose 中性版,
# −α main axis + bidirectional (Near-Optimal Fig 2B: baseline risk-taking already
# ceilinged, so +α has little headroom; signal expected on the −α side).
# Run α=0 first to confirm Llama3-8B is NOT near-random before reading the sweep.
#
# Output (per the migrated Dopamine tree):
#   ${BASE_DIR}/${MODEL}/answer_cgt/summary_llama3_8B.csv   — mean±std per α
#   ${BASE_DIR}/${MODEL}/answer_cgt/mdf_${alpha}/cgt_8B_*_11_20.json  — per-run detail
# Per-α resume: summary CSV is appended, finished α are skipped on re-run.
#
# Usage:
#   bash run_cgt.sh                 # full −α-axis + bidirectional sweep
#   bash run_cgt.sh --pilot         # α=0 only, 2 runs; writes answer_cgt_pilot
#   nohup bash run_cgt.sh > cgt.log 2>&1 &

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==================== Paths ====================
DATA="data1"
WORK_DIR="/${DATA}/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
TYPE="non"
HS_PREFIX="llama3"
MASK_TYPE="nmd"
PERCENTAGE=0.5

# ==================== α sweep (−α main axis + bidirectional) ====================
# Full dose: 0 / ±2 / ±4 / ±6 / ±8 at layers 11–20.
CONFIGS="0-11-20 neg2-11-20 2-11-20 neg4-11-20 4-11-20 neg6-11-20 6-11-20 neg8-11-20 8-11-20"
# Past sweeps (kept for reproducibility, do not delete):
# CONFIGS="0-11-20 4-11-20 neg4-11-20"          # initial ±4 direction check

# ==================== Generation ====================
# runs are INDEPENDENT repeats (seed = run_idx); n_runs only sets statistical
# precision. Start at 10 to eyeball the dose trend; bump to 30 for the final
# tighter error bars. Workflow: server is wiped + results downloaded each run,
# so summary CSV is throwaway — authoritative stats are recomputed locally from
# each run's `records` in the detail JSON.
NUM_RUNS=10              # was 30 for the full-precision sweep
MAX_NEW_TOKENS=256        # natural EOS at this cap (no stop_strings). A </choice>
                          # stop was tried but the system-prompt format spec literally
                          # contains "</choice>", so the model echoing it mid-reasoning
                          # truncated before the real answer → invalid 0.02→0.11. 256 +
                          # natural EOS keeps invalid ~0.02 (cost: ~4.2s/round).
TEMPERATURE=1.0          # Near-Optimal default
TOP_P=0.9
ANS_FILE="answer_cgt"

# ==================== Pilot override ====================
# --pilot : α=0, 2 runs, near-random sanity check → separate answer_cgt_pilot dir
if [ "$1" == "--pilot" ]; then
    CONFIGS="0-11-20"
    NUM_RUNS=2
    ANS_FILE="answer_cgt_pilot"
    echo "[PILOT] α=0 only, ${NUM_RUNS} runs — baseline near-random check (${ANS_FILE})"
fi

# ==================== Run ====================
echo "=================================================="
echo "Cambridge Gambling Task — RSN α dose-response"
echo "Model  : ${MODEL_NAME}-${MODEL_SIZE}  (layers 11–20, ${MASK_TYPE} mask)"
echo "Configs: ${CONFIGS}"
echo "Runs   : ${NUM_RUNS} × 64 decisions  (chat=on, T=${TEMPERATURE}, top_p=${TOP_P})"
echo "Output : ${BASE_DIR}/${MODEL_NAME}/${ANS_FILE}"
echo "Start  : $(date)"
echo "=================================================="

cd "${WORK_DIR}"

python get_answer_cgt.py \
    --model "${MODEL_NAME}" \
    --model_dir "${MODEL_DIR}" \
    --hs "${HS_PREFIX}" \
    --size "${MODEL_SIZE}" \
    --type "${TYPE}" \
    --percentage "${PERCENTAGE}" \
    --mask_type "${MASK_TYPE}" \
    --configs ${CONFIGS} \
    --num_runs "${NUM_RUNS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --use_chat \
    --ans_file "${ANS_FILE}" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}"

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] CGT — ${MODEL_NAME} ${MODEL_SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] CGT — ${MODEL_NAME} ${MODEL_SIZE}"
    exit 1
fi
