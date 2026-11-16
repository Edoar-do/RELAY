#!/bin/bash
# One container per arm: Stage-1 pre-training (60k steps), then the three Ribeiro-dev
# linear evaluations (K=12, 1, 2) on the 60k checkpoint, one after the other.
#
# Usage:  bash run_arm.sh norls      (or rls)
#         SKIP_PRETRAIN=1 bash run_arm.sh rls    # only the lin-evals, checkpoint already there

set -euo pipefail
source "$(dirname "$0")/common.sh"

ARM=${1:?usage: run_arm.sh rls|rls_zeropad|norls}
CKPT=$PRETRAINED_DIR/gat_${PRETRAIN_STEPS}_stage1_${ARM}.pt

if [[ "${SKIP_PRETRAIN:-0}" != "1" ]]; then
    [[ -f "$CKPT" ]] && { echo "$CKPT already exists; refusing to overwrite (use SKIP_PRETRAIN=1)"; exit 1; }
    bash "$ABL_DIR/pretrain_stage1.sh" "$ARM"
fi

[[ -f "$CKPT" ]] || { echo "no 60k checkpoint at $CKPT (early stopping?); see $LOG_DIR/stage1_${ARM}.log"; exit 1; }

for K in 12 1 2; do
    bash "$ABL_DIR/lineval_ribeiro.sh" "$ARM" "$K" "$CKPT"
done
if [[ "$ARM" == "rls_zeropad" ]]; then
    # everything rls has: lead I / II, and lin-eval seeds 43 / 44 at K = 1, 2
    for K in I II; do
        bash "$ABL_DIR/lineval_ribeiro.sh" "$ARM" "$K" "$CKPT"
    done
    for S in 43 44; do for K in 1 2; do
        SEED=$S bash "$ABL_DIR/lineval_ribeiro.sh" "$ARM" "$K" "$CKPT"
    done; done
fi

$PYTHON "$ABL_DIR/collect_results.py"
