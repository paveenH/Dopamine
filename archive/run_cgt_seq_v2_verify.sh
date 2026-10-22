#!/bin/bash
# CGT-Sequential — run all four v2 verify A/B cells sequentially in one shot.
#   v2a = symmetrised prompt + "Answer: " anchor on BOTH steps
#   v2b = symmetrised prompt + NO anchor on either step
# Each cell: α=0/±6, 5 runs. Output dirs (never overwrite v1 _verify):
#   answer_cgtseq_{asc,desc}_v2a_verify  /  _v2b_verify
#
# Usage:
#   bash run_cgt_seq_v2_verify.sh
#   nohup bash run_cgt_seq_v2_verify.sh > cgtseq_v2_verify.log 2>&1 &
#
# After it finishes, compare v2a vs v2b vs the original v1 verify with:
#   python3.10 analyze_cgt_seq.py --asc  llama3/cgt/seq_asc_v2a_verify  --desc llama3/cgt/seq_desc_v2a_verify
#   python3.10 analyze_cgt_seq.py --asc  llama3/cgt/seq_asc_v2b_verify  --desc llama3/cgt/seq_desc_v2b_verify

set -e   # stop the whole batch on the first failing run

HERE="$(cd "$(dirname "$0")" && pwd)"
MODES=(--verify_v2a_asc --verify_v2a_desc --verify_v2b_asc --verify_v2b_desc)

echo "=================================================="
echo "CGT-Sequential v2 verify batch — ${#MODES[@]} cells"
echo "Start: $(date)"
echo "=================================================="

for mode in "${MODES[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>> ${mode} <<<<<<<<<<<<<<<<<<"
    bash "${HERE}/run_cgt_seq.sh" "${mode}"
done

echo ""
echo "[✓ All done] CGT-Sequential v2 verify batch — finished at: $(date)"
