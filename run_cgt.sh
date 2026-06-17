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
# NOTE (2026-06): runs BARE-STRING (no --use_chat). The NMD mask / diff vectors
# were extracted on bare prompts, so steering must inject into the same activation
# distribution; apply_chat_template prepends <|start_header_id|>system… control
# tokens that shift residual-stream geometry away from where the wanting direction
# was measured, diluting the steer — a candidate explanation for the CGT α-null.
# to_chat(use_chat=False) just concatenates system + "\n\n" + user; the <choice>N
# </choice> XML format is an instruction inside SYSTEM_TEMPLATE that the model emits
# regardless of the chat wrapper, so parse_choice is unaffected. See CLAUDE.md
# chat-template caveat. The first 10-run null was collected under chat — the
# bare-vs-chat re-run is exactly the decision sweep that caveat calls for.
#
# Output (per the migrated Dopamine tree):
#   ${BASE_DIR}/${MODEL}/answer_cgt/summary_llama3_8B.csv   — mean±std per α
#   ${BASE_DIR}/${MODEL}/answer_cgt/mdf_${alpha}/cgt_8B_*_11_20.json  — per-run detail
# Per-α resume: summary CSV is appended, finished α are skipped on re-run.
#
# Usage:
#   bash run_cgt.sh                 # full −α-axis + bidirectional sweep (faithful prompt)
#   bash run_cgt.sh --pilot         # α=0 only, 2 runs; writes answer_cgt_pilot
#   bash run_cgt.sh --verify        # α=0, 5 runs, SIMPLE prompt + save_all_raw →
#                                   #   answer_cgt_simple_verify. Checks whether the
#                                   #   faithful-port qdm≈0.5 is a prompt issue: if
#                                   #   SIMPLE lifts qdm at 9:1, CGT is salvageable.
#   bash run_cgt.sh --simple        # full sweep with the SIMPLE prompt (after verify passes)
#   bash run_cgt.sh --verify2       # α=0, 5 runs, SIMPLE2 multi-turn GAME prompt (chat) +
#                                   #   save_all_raw → answer_cgt_simple2_verify. Tests
#                                   #   whether framing CGT as a chat dialogue fixes the
#                                   #   --simple word-problem failure. Check qdm@9:1.
#   bash run_cgt.sh --simple2       # full sweep with the SIMPLE2 prompt (after verify2 passes)
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
SIMPLE_FLAG=""           # set by --verify / --simple / --verify2
SAVE_RAW_FLAG=""         # set by --verify / --verify2
USE_CHAT_FLAG=""         # set by --verify2 (simple2 requires chat)

# ==================== Mode overrides ====================
# --pilot : α=0, 2 runs, near-random sanity check → separate answer_cgt_pilot dir
if [ "$1" == "--pilot" ]; then
    CONFIGS="0-11-20"
    NUM_RUNS=2
    ANS_FILE="answer_cgt_pilot"
    echo "[PILOT] α=0 only, ${NUM_RUNS} runs — baseline near-random check (${ANS_FILE})"
fi

# --verify : α=0, 5 runs, SIMPLE prompt + save every raw → separate dir.
# Diagnoses whether faithful-port qdm≈0.5 is a prompt issue. Inspect raw + qdm
# (esp. at 9:1) offline before committing to a full SIMPLE sweep.
if [ "$1" == "--verify" ]; then
    CONFIGS="0-11-20"
    NUM_RUNS=5
    ANS_FILE="answer_cgt_simple_verify"
    SIMPLE_FLAG="--simple_prompt"
    SAVE_RAW_FLAG="--save_all_raw"
    echo "[VERIFY] α=0, ${NUM_RUNS} runs, SIMPLE prompt + save_all_raw (${ANS_FILE})"
fi

# --simple : full sweep but with the SIMPLE prompt (use after --verify passes).
if [ "$1" == "--simple" ]; then
    ANS_FILE="answer_cgt_simple"
    SIMPLE_FLAG="--simple_prompt"
    echo "[SIMPLE] full sweep with SIMPLE prompt (${ANS_FILE})"
fi

# --verify2 : α=0, 5 runs, SIMPLE2 multi-turn GAME prompt (chat) + save every raw.
# The --simple verify left qdm≈0.46 at 9:1 because the model read each round as an
# isolated WORD PROBLEM (raw: 85% probability essays / examiner / Python, 0% clean
# answers). --simple2 presents CGT as an ongoing chat dialogue so the <|assistant|>
# turn frames "now it's your move"; reasoning stays allowed but the final line is
# locked to 'Choice: <number>'. REQUIRES chat. Inspect qdm@9:1 + raw offline before
# committing to a full --simple2 sweep. Steering-alignment caveat is deferred: the
# α=0 baseline is the blocker, and (per Confidence Betting) chat-split helped there.
MAX_NEW_TOKENS_OVERRIDE=""
if [ "$1" == "--verify2" ]; then
    CONFIGS="0-11-20"
    NUM_RUNS=5
    ANS_FILE="answer_cgt_simple2_verify"
    SIMPLE_FLAG="--simple2"
    SAVE_RAW_FLAG="--save_all_raw"
    USE_CHAT_FLAG="--use_chat"
    MAX_NEW_TOKENS_OVERRIDE=200   # room to reason, then commit Choice: <number>
    echo "[VERIFY2] α=0, ${NUM_RUNS} runs, SIMPLE2 multi-turn GAME prompt + chat + save_all_raw (${ANS_FILE})"
fi
[ -n "$MAX_NEW_TOKENS_OVERRIDE" ] && MAX_NEW_TOKENS=$MAX_NEW_TOKENS_OVERRIDE

# --simple2 : full −8→+8 sweep with the SIMPLE2 multi-turn game prompt (after
# --verify2 passes; verify2 gave qdm 0.81–0.91 with risk_adj_slope turning POSITIVE
# +0.044/+0.046, i.e. the model finally reads the chests and scales bets to the
# odds). 9 α × 20 runs × 64 decisions. save_all_raw kept ON across ALL cells (user
# decision 2026-06-17) so every α's reasoning/failure mode is inspectable offline
# (does +α chase losses more? does −α decohere?). CGT is probability-transparent, so
# RSN's dose-response on risk_taking / risk_adj_slope / I_LC is the headline readout.
if [ "$1" == "--simple2" ]; then
    CONFIGS="0-11-20 neg2-11-20 2-11-20 neg4-11-20 4-11-20 neg6-11-20 6-11-20 neg8-11-20 8-11-20"
    NUM_RUNS=20
    ANS_FILE="answer_cgt_simple2"
    SIMPLE_FLAG="--simple2"
    SAVE_RAW_FLAG="--save_all_raw"
    USE_CHAT_FLAG="--use_chat"
    MAX_NEW_TOKENS=200            # room to reason, then commit Choice: <number>
    echo "[SIMPLE2] full −8→+8 sweep, ${NUM_RUNS} runs, SIMPLE2 GAME prompt + chat + save_all_raw (${ANS_FILE})"
fi

# ==================== Run ====================
echo "=================================================="
echo "Cambridge Gambling Task — RSN α dose-response"
echo "Model  : ${MODEL_NAME}-${MODEL_SIZE}  (layers 11–20, ${MASK_TYPE} mask)"
echo "Configs: ${CONFIGS}"
echo "Runs   : ${NUM_RUNS} × 64 decisions  (bare-string, T=${TEMPERATURE}, top_p=${TOP_P})"
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
    --ans_file "${ANS_FILE}" \
    --data "${DATA}" \
    --base_dir "${BASE_DIR}" \
    ${SIMPLE_FLAG} \
    ${USE_CHAT_FLAG} \
    ${SAVE_RAW_FLAG}

if [ $? -eq 0 ]; then
    echo ""
    echo "[✓ Done] CGT — ${MODEL_NAME} ${MODEL_SIZE} finished at: $(date)"
else
    echo ""
    echo "[✗ Failed] CGT — ${MODEL_NAME} ${MODEL_SIZE}"
    exit 1
fi
