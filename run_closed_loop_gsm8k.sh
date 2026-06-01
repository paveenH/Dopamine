#!/bin/bash
set -euo pipefail

# Phase 2: Closed-loop RSN dopamine control — GSM8K
# Runs all plans (none, static, A, B, C, D) for No-CoT and CoT conditions

MODEL_NAME="llama3"
MODEL_DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_SIZE="8B"
HS_PREFIX="llama3"
TYPE="non"

MASK_TYPE="nmd"
PERCENTAGE=0.5
LAYER_START=11
LAYER_END=20
EMA_ALPHA=0.95

MAX_NEW_TOKENS=512
N_SAMPLES=300

K1=2.0
K2=1.0
FLOOR_RATIO=0.65
PLATEAU_END_RATIO=0.65
AVG_GEN_LEN=400

WORK_DIR="/data1/paveen/Dopamine"
BASE_DIR="${WORK_DIR}/components"
MASK_PATH="${BASE_DIR}/mask/${HS_PREFIX}_${TYPE}_logits/${MASK_TYPE}_${PERCENTAGE}_${LAYER_START}_${LAYER_END}_${MODEL_SIZE}.npy"

echo "=================================================="
echo "Phase 2: Closed-Loop GSM8K"
echo "Model: ${MODEL_NAME} (${MODEL_SIZE})"
echo "Layers: ${LAYER_START}-${LAYER_END} | EMA: ${EMA_ALPHA}"
echo "k1=${K1} | k2=${K2} | floor=${FLOOR_RATIO} | plateau_end=${PLATEAU_END_RATIO}"
echo "Start: $(date)"
echo "=================================================="

cd ${WORK_DIR}

echo ""
echo "[sanity] NMD mask indexing check"
python sanity_mask_indexing.py \
    --mask_path "${MASK_PATH}" \
    --expect_layer_start ${LAYER_START} \
    --expect_layer_end ${LAYER_END} \
    --skip_model

BASE_ARGS="
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
  --max_new_tokens ${MAX_NEW_TOKENS}
  --temperature 0.0
  --base_dir ${BASE_DIR}
  --k1 ${K1}
  --k2 ${K2}
  --floor_ratio ${FLOOR_RATIO}
  --plateau_end_ratio ${PLATEAU_END_RATIO}
  --avg_gen_len ${AVG_GEN_LEN}
"

# ── No-CoT ────────────────────────────────────────────────────────

echo ""
echo "========== No-CoT =========="

# echo "[1/7] No-CoT | plan=none (baseline α=0)"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan none
# echo "[Done]"

# echo ""
# echo "[2/7] No-CoT | plan=static α=-4"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan static --static_alpha -4
# echo "[Done]"

# echo ""
# echo "[3/7] No-CoT | plan=static α=+4"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan static --static_alpha 4
# echo "[Done]"

# echo ""
# echo "[4/7] No-CoT | plan=A (Tonic Floor)"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan A
# echo "[Done]"

# echo ""
# echo "[5/7] No-CoT | plan=B (Plateau Imitation)"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan B
# echo "[Done]"

# echo ""
# echo "[6/7] No-CoT | plan=C (Tonic + Micro-Phasic)"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan C
# echo "[Done]"

# echo ""
# echo "[7/7] No-CoT | plan=D (Waveform Smoothing) k1=2.0"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan D --k1 2.0
# echo "[Done]"

# echo ""
# echo "[7b/7] No-CoT | plan=D k1=1.0"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan D --k1 1.0
# echo "[Done]"

# echo ""
# echo "[7c/7] No-CoT | plan=D k1=0.5"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan D --k1 0.5
# echo "[Done]"

# echo ""
# echo "[8/7] No-CoT | plan=E (EMA Homeostasis) k1=2.0 target=0.85"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan E --k1 2.0 --floor_ratio 0.85
# echo "[Done]"

# echo ""
# echo "[8b/7] No-CoT | plan=E k1=1.0 target=0.85"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan E --k1 1.0 --floor_ratio 0.85
# echo "[Done]"

# echo ""
# echo "[8c/7] No-CoT | plan=E k1=0.5 target=0.85"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan E --k1 0.5 --floor_ratio 0.85
# echo "[Done]"

# echo ""
# echo "[9/7] No-CoT | plan=F (Dual-EMA Filter) k1=2.0"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan F --k1 2.0
# echo "[Done]"

# echo ""
# echo "[9b/7] No-CoT | plan=F k1=1.0"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan F --k1 1.0
# echo "[Done]"

# echo ""
# echo "[9c/7] No-CoT | plan=F k1=0.5"
# python closed_loop_gsm8k.py ${BASE_ARGS} --plan F --k1 0.5
# echo "[Done]"

# ── Plan G: Bang-Bang Dead-Zone ──────────────────────────────────
# k1 = α magnitude (fixed); k2 = dead-zone half-width as fraction of xp
# Sweep: k1 ∈ {1, 2, 4} × k2 ∈ {0.1, 0.2, 0.3} = 9 runs

# for K1_G in 1.0 2.0 4.0; do
#   for K2_G in 0.2; do
#     echo ""
#     echo "[G] No-CoT | plan=G k1=${K1_G} k2=${K2_G}"
#     python closed_loop_gsm8k.py ${BASE_ARGS} --plan G --k1 ${K1_G} --k2 ${K2_G}
#     echo "[Done]"
#   done
# done

# ── Plan H1: Early Peak Boost ─────────────────────────────────────
# First 100 decode steps inject +k1 whenever ema < floor_ratio * xp.
# floor_ratio = 1.5 (peak target); k1 = injection magnitude.
# After step 100: zero intervention. Tests "task-onset commitment" hypothesis.

# for K1_H1 in 2.0 4.0 6.0; do
#   echo ""
#   echo "[H1] No-CoT | plan=H1 k1=${K1_H1} peak_target=1.5*xp window=100"
#   python closed_loop_gsm8k.py ${BASE_ARGS} --plan H1 --k1 ${K1_H1} --floor_ratio 1.5
#   echo "[Done]"
# done

# ── Plan H2: Trapezoid Tracking ───────────────────────────────────
# Bidirectional proportional control toward piecewise-linear target:
#   t<50          : linear rise 1.0·xp → 1.25·xp
#   50≤t<200      : plateau at 1.25·xp
#   t≥200         : linear decay 1.25·xp → 0.75·xp by t=avg_gen_len
# floor_ratio=1.25 (peak), plateau_end_ratio=0.75 (end), avg_gen_len=400 (T).
# Sweep k1 ∈ {1, 2, 4}.

# for K1_H2 in 1.0 2.0 4.0; do
#   echo ""
#   echo "[H2] No-CoT | plan=H2 k1=${K1_H2} peak=1.25 end=0.75 T=${AVG_GEN_LEN}"
#   python closed_loop_gsm8k.py ${BASE_ARGS} --plan H2 --k1 ${K1_H2} --floor_ratio 1.25 --plateau_end_ratio 0.75
#   echo "[Done]"
# done

# ── Plan H3: Trapezoid v2 — Steeper Rise & Longer Plateau ─────────
# CoT-shape-fitting variant: faster rise, higher peak, earlier plateau exit.
#   t<30          : linear rise 1.0·xp → 1.35·xp     (was 50→1.25 in H2)
#   30≤t<120      : plateau at 1.35·xp                (was 200 in H2)
#   t≥120         : linear decay 1.35·xp → 0.75·xp by t=avg_gen_len
# floor_ratio=1.35 (peak), plateau_end_ratio=0.75 (end), avg_gen_len=400 (T).
# Sweep k1 ∈ {4, 6, 8} — H2 k1=4 was best (62.7%), test if higher gain helps.

for K1_H3 in 4.0 6.0 8.0; do
  echo ""
  echo "[H3] No-CoT | plan=H3 k1=${K1_H3} peak=1.35 end=0.75 T=${AVG_GEN_LEN}"
  python closed_loop_gsm8k.py ${BASE_ARGS} --plan H3 --k1 ${K1_H3} --floor_ratio 1.35 --plateau_end_ratio 0.75
  echo "[Done]"
done

# ── CoT ───────────────────────────────────────────────────────────

echo ""
echo "========== CoT =========="

# echo "[7/7] CoT | plan=none (baseline α=0)"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan none
# echo "[Done]"

# echo ""
# echo "[8/7] CoT | plan=static α=-4"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan static --static_alpha -4
# echo "[Done]"

# echo ""
# echo "[9/7] CoT | plan=static α=+4"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan static --static_alpha 4
# echo "[Done]"

# echo ""
# echo "[10/7] CoT | plan=A"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan A
# echo "[Done]"

# echo ""
# echo "[11/7] CoT | plan=B"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan B
# echo "[Done]"

# echo ""
# echo "[12/7] CoT | plan=C"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan C
# echo "[Done]"

# echo ""
# echo "[13/7] CoT | plan=D k1=2.0"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan D --k1 2.0
# echo "[Done]"

# echo ""
# echo "[14/7] CoT | plan=E k1=2.0 target=0.85"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan E --k1 2.0 --floor_ratio 0.85
# echo "[Done]"

# echo ""
# echo "[15/7] CoT | plan=F k1=2.0"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan F --k1 2.0
# echo "[Done]"

# CoT Plan G — pick best k2 after No-CoT sweep, then run k1=2,4
# echo ""
# echo "[G-CoT] CoT | plan=G k1=2.0 k2=0.2"
# python closed_loop_gsm8k.py ${BASE_ARGS} --cot --plan G --k1 2.0 --k2 0.2
# echo "[Done]"

echo ""
echo "=================================================="
echo "All done: $(date)"
echo "Output → ${BASE_DIR}/${MODEL_NAME}/closed_loop/"
echo "=================================================="
