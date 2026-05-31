#!/bin/bash
# Phase 1b round 2: Hidden-state recording with paper-aligned roles + neutral
# baselines. HS storage is now selective (middle 9 layers + final layer = 10
# layers total) — see track_hidden_states.py HiddenStateRecorder.
#
# 5 runs total:
#   1) GSM8K No-CoT  role=expert            "an expert"
#   2) GSM8K No-CoT  role=non_expert        "a non expert"
#   3) GSM8K No-CoT  role=neutral           (no character)
#   4) GSM8K CoT     role=neutral           (no character)
#   5) GSM8K No-CoT  role=primary_teacher   "a primary school teacher"
#
# Output: HDF5 files under ${BASE_DIR}/hidden_states/${TASK}/

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"
DATA="data1"

MASK_TYPE="nmd"           # only used as the sanity-projection mask in HDF5
PERCENTAGE=0.5
LAYER_START=11
LAYER_END=20
EMA_ALPHA=0.95

MAX_NEW_TOKENS_GSM=512
N_SAMPLES=300

WORK_DIR="/${DATA}/paveen/RolePlaying"
BASE_DIR="${WORK_DIR}/components"

echo "=================================================="
echo "Phase 1b round 2: HS recording (5 runs, GSM8K)"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE}) | Middle layers: ${LAYER_START}-${LAYER_END}"
echo "HS storage: middle [${LAYER_START},${LAYER_END}) + final layer = 10 stored layers"
echo "Sanity mask: ${MASK_TYPE} (offline reanalysis can use any mask)"
echo "Start: $(date)"
echo "=================================================="

cd ${WORK_DIR}

BASE_ARGS="
  --task gsm8k
  --model ${MODEL_NAME}
  --model_dir ${MODEL_DIR}
  --hs ${HS_PREFIX}
  --size ${MODEL_SIZE}
  --type ${TYPE}
  --mask_type ${MASK_TYPE}
  --percentage ${PERCENTAGE}
  --layer_start ${LAYER_START}
  --layer_end ${LAYER_END}
  --ema_alpha ${EMA_ALPHA}
  --test_file benchmark/gsm8k_test_sample.json
  --n_samples ${N_SAMPLES}
  --max_new_tokens ${MAX_NEW_TOKENS_GSM}
  --temperature 0.0
  --base_dir ${BASE_DIR}
"

# ── Run 1: expert ──
echo ""
echo "[1/5] GSM8K No-CoT | role=expert ('an expert')"
python track_hidden_states.py ${BASE_ARGS} --role expert
echo "[Done] expert"

# ── Run 2: non_expert ──
echo ""
echo "[2/5] GSM8K No-CoT | role=non_expert ('a non expert')"
python track_hidden_states.py ${BASE_ARGS} --role non_expert
echo "[Done] non_expert"

# ── Run 3: neutral (No-CoT) ──
echo ""
echo "[3/5] GSM8K No-CoT | role=neutral"
python track_hidden_states.py ${BASE_ARGS} --role neutral
echo "[Done] neutral No-CoT"

# ── Run 4: neutral (CoT) ──
echo ""
echo "[4/5] GSM8K CoT    | role=neutral"
python track_hidden_states.py ${BASE_ARGS} --role neutral --cot
echo "[Done] neutral CoT"

# ── Run 5: primary_teacher (task-matched extra role) ──
echo ""
echo "[5/5] GSM8K No-CoT | role=primary_teacher ('a primary school teacher')"
python track_hidden_states.py ${BASE_ARGS} --role primary_teacher
echo "[Done] primary_teacher"

echo ""
echo "=================================================="
echo "All done: $(date)"
echo "Output → ${BASE_DIR}/hidden_states/gsm8k/"
echo "=================================================="
