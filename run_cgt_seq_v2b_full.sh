#!/bin/bash
# CGT-Sequential — full −8→+8 sweep for the WINNING v2b config
# (symmetrised Blue/Red ↔ Accept/Wait prompt, NO anchor on either step).
#
# v2b won the verify A/B: it reproduced e55b132's ρ≈−0.94 accept_step signal
# (asc 3.26→2.03→1.31, desc 4.04→2.67→1.41, KW p=0.002) with qdm flat ≈0.75
# (clean wanting–knowing dissociation), invalid=0.00 at α=0. v2a (Answer: anchor)
# was discarded — the anchor collapsed α=0 to step1 and emitted empty "Answer: "
# at ±6, killing the multi-step Wait decision the task measures.
#
# Runs BOTH conditions (asc 5→95, desc 95→5) sequentially. delay_aversion_index =
# mean_bet_desc − mean_bet_asc requires both. −8→+8 × 20 runs each.
#
# Usage:
#   bash run_cgt_seq_v2b_full.sh
#   nohup bash run_cgt_seq_v2b_full.sh > cgtseq_v2b_full.log 2>&1 &
#
# After it finishes:
#   python3.10 analyze_cgt_seq.py --asc llama3/cgt/seq_asc_v2b --desc llama3/cgt/seq_desc_v2b
# Confirm the signal survives at MILD α (±2/±4), since ±6 already over-steers
# (invalid 0.13–0.23) before putting it in the paper.

set -e   # stop the batch on the first failing run

HERE="$(cd "$(dirname "$0")" && pwd)"
MODES=(--v2b_asc --v2b_desc)

echo "=================================================="
echo "CGT-Sequential v2b FULL sweep — ${#MODES[@]} conditions, −8→+8 × 20 runs"
echo "Start: $(date)"
echo "=================================================="

for mode in "${MODES[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>> ${mode} <<<<<<<<<<<<<<<<<<"
    bash "${HERE}/run_cgt_seq.sh" "${mode}"
done

echo ""
echo "[✓ All done] CGT-Sequential v2b full sweep — finished at: $(date)"
