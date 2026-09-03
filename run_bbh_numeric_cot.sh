#!/usr/bin/env bash
# BBH numeric explicit-CoT follow-up (cot-transfer-followup-v0).
#
#   bash run_bbh_numeric_cot.sh llama3  object_counting
#   bash run_bbh_numeric_cot.sh qwen2.5 object_counting
#
# Runs all four CoT cells (alpha=0, workpoint, neighbour, reverse) for one
# model in one process, mirroring the frozen GSM8K workpoint/neighbour/reverse
# doses already used for the No-CoT bbh-p4b-v0 result. POST-HOC EXPLORATORY
# FOLLOW-UP -- does not replace the frozen No-CoT result.
#
# TASK IS RESTRICTED TO object_counting: that is the only task in this
# follow-up's frozen 6-comparison Holm family (see
# PREREG_COT_TRANSFER_FOLLOWUP.md S4.1). multistep_arithmetic_two has no CoT
# cell authorised here.
#
# CELLS ARE NOT REQUIRED TO SHARE A GPU (project-wide convention). host and
# CUDA_VISIBLE_DEVICES are recorded per generation file; pairing is by
# sample_id only, never by hardware identity.
set -euo pipefail

# NOTE: no braces in a ${1:?...} message -- the expansion ends at the FIRST
# '}', which once made MODEL literally 'llama3}' in a sibling launcher.
if [[ $# -lt 2 ]]; then
  echo "usage: run_bbh_numeric_cot.sh llama3|qwen2.5 object_counting" >&2
  exit 1
fi
MODEL="$1"; TASK="$2"
PY="${PY:-python}"
WORK_DIR="${WORK_DIR:-/data1/paveen/Dopamine}"
BASE_DIR="${BASE_DIR:-$WORK_DIR/components}"
BENCH="${BENCH:-$BASE_DIR/benchmark}"
OUT_ROOT="${OUT_ROOT:-$BASE_DIR/$MODEL/bbh_cot_followup/$TASK}"

case "$TASK" in
  object_counting) ;;
  *) echo "[FATAL] task must be object_counting -- this follow-up's Holm" >&2
     echo "        family does not include multistep_arithmetic_two" >&2
     exit 1 ;;
esac

"$PY" -c "import numpy, torch" >/dev/null 2>&1 || {
  echo "[FATAL] '$PY' cannot import numpy/torch. On the server the" >&2
  echo "        interpreter is 'python', not 'python3.10'." >&2; exit 1; }

case "$MODEL" in
  llama3)
    SIZE=8B
    MODEL_DIR="${MODEL_DIR:-meta-llama/Llama-3.1-8B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/llama3_non_logits/nmd_0.5_11_20_8B.npy}"
    CONFIGS=(0-11-20 neg6-11-20 neg4-11-20 4-11-20) ;;
  qwen2.5)
    SIZE=7B
    MODEL_DIR="${MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
    MASK="${MASK:-$BASE_DIR/mask/qwen2.5_non_logits/nmd_0.5_16_22_7B.npy}"
    CONFIGS=(0-16-22 8-16-22 6-16-22 neg6-16-22) ;;
  *) echo "[FATAL] unknown model '$MODEL'" >&2; exit 1 ;;
esac

QFILE="${QFILE:-$BENCH/bbh_p4b_${TASK}_blind.json}"
[[ -f "$MASK" ]]  || { echo "[FATAL] mask not found: $MASK" >&2; exit 1; }
[[ -f "$QFILE" ]] || { echo "[FATAL] blind questions not found: $QFILE" >&2
                       echo "        (same file the No-CoT bbh-p4b-v0 cells used)" >&2
                       exit 1; }
if [[ "$MODEL_DIR" == /* && ! -d "$MODEL_DIR" ]]; then
  echo "[FATAL] MODEL_DIR looks like a path but does not exist: $MODEL_DIR" >&2; exit 1
fi

mkdir -p "$OUT_ROOT"
echo "[cot-followup] BBH model=$MODEL task=$TASK configs=${CONFIGS[*]}"
echo "[cot-followup] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[cot-followup] POST-HOC EXPLORATORY -- does not replace the frozen No-CoT bbh-p4b-v0 result"
echo "[cot-followup] start $(date)"

cd "$WORK_DIR"
"$PY" get_answer_bbh_numeric_cot.py \
  --model "$MODEL" --size "$SIZE" --model_dir "$MODEL_DIR" \
  --task "$TASK" --questions "$QFILE" --mask_path "$MASK" \
  --configs "${CONFIGS[@]}" --out_dir "$OUT_ROOT"

echo "[cot-followup] done $(date)"
echo "[cot-followup] After BOTH models finish, score with the shared eval script"
echo "     (NOTE: --nocot_generations needs all FOUR No-CoT cells per model --"
echo "     alpha=0, workpoint, neighbour, reverse -- since the DiD interaction"
echo "     term reads both alpha=0 and the workpoint dose for each model):"
echo "     $PY eval_cot_transfer_followup.py --task bbh --bbh_task $TASK \\"
echo "       --generations \\"
echo "         $BASE_DIR/llama3/bbh_cot_followup/$TASK/mdf_{0,neg6,neg4,4}/bbh_${TASK}_cot_8B_11_20.json \\"
echo "         $BASE_DIR/qwen2.5/bbh_cot_followup/$TASK/mdf_{0,8,6,neg6}/bbh_${TASK}_cot_7B_16_22.json \\"
echo "       --gold_file $BENCH/bbh_p4b_${TASK}.json \\"
echo "       --nocot_generations \\"
echo "         $BASE_DIR/llama3/bbh/$TASK/mdf_{0,neg6}/bbh_${TASK}_8B_11_20.json \\"
echo "         $BASE_DIR/qwen2.5/bbh/$TASK/mdf_{0,8}/bbh_${TASK}_7B_16_22.json \\"
echo "       --nocot_evaluation docs/bbh_p4b_${TASK}_result.json \\"
echo "       --out docs/cot_followup_bbh_${TASK}_evaluation.json"
