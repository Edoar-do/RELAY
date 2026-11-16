#!/bin/bash
# Lead I / lead II linear evaluations (probe trained and validated on that lead only) on the 60k
# checkpoints of the given arms, one after the other on this container's GPU.
#
# Several containers can run this at once: each run is claimed with an atomic mkdir on the shared FS
# (logs/claims/<run>), so every run is done by exactly one worker.
#
# Usage:  bash run_leads_queue.sh rls norls_tok norls norls_pm0217
# A failed run releases its claim; rerunning the script retries it.

set -uo pipefail
source "$(dirname "$0")/common.sh"
CLAIMS=$LOG_DIR/claims
mkdir -p "$CLAIMS"

for ARM in "$@"; do
    CKPT=$PRETRAINED_DIR/gat_${PRETRAIN_STEPS}_stage1_${ARM}.pt
    for LEAD in I II; do
        RUN=lineval_stage1_${ARM}_lead${LEAD}
        LOG=$LOG_DIR/${RUN}.log
        if [[ -f "$LOG" ]] && grep -q "Validating the model at step $LINEVAL_STEPS" "$LOG"; then
            echo "[$(date '+%F %T')] SKIP (done): $ARM lead $LEAD"
            continue
        fi
        if ! mkdir "$CLAIMS/$RUN" 2>/dev/null; then
            echo "[$(date '+%F %T')] SKIP (claimed by $(cat "$CLAIMS/$RUN/host" 2>/dev/null)): $ARM lead $LEAD"
            continue
        fi
        hostname > "$CLAIMS/$RUN/host"
        echo "[$(date '+%F %T')] START $ARM lead $LEAD on $(hostname)"
        if bash "$ABL_DIR/lineval_ribeiro.sh" "$ARM" "$LEAD" "$CKPT" > /dev/null 2>&1; then
            echo "[$(date '+%F %T')] OK $ARM lead $LEAD"
        else
            echo "[$(date '+%F %T')] FAILED $ARM lead $LEAD, see $LOG"
            rm -rf "$CLAIMS/$RUN"
        fi
    done
done
$PYTHON "$ABL_DIR/collect_results.py"
