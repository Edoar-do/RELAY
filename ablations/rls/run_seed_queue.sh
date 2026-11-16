#!/bin/bash
# Extra linear-evaluation seeds (SEED != 42, see lineval_ribeiro.sh) on the 60k checkpoints, as a queue
# that several containers can share: each run is claimed with an atomic mkdir on the shared FS
# (logs/claims/<run>), so every run is done by exactly one worker.
#
# Usage:  SEEDS="43 44" bash run_seed_queue.sh rls:1 rls:2 norls_pm0217:1 norls_pm0217:2
#         jobs are <arm>:<K>, K = 1 | 2 | 12 | I | II; seed-major order (all jobs of seed 43 first)
# A failed run releases its claim; rerunning the script retries it.

set -uo pipefail
source "$(dirname "$0")/common.sh"
CLAIMS=$LOG_DIR/claims
mkdir -p "$CLAIMS"
read -r -a SEED_LIST <<< "${SEEDS:-43 44}"

for SEED in "${SEED_LIST[@]}"; do
    for JOB in "$@"; do
        ARM=${JOB%%:*}
        K=${JOB##*:}
        CKPT=$PRETRAINED_DIR/gat_${PRETRAIN_STEPS}_stage1_${ARM}.pt
        case "$K" in I|II) RUN=lineval_stage1_${ARM}_lead${K}_s${SEED} ;; *) RUN=lineval_stage1_${ARM}_K${K}_s${SEED} ;; esac
        LOG=$LOG_DIR/${RUN}.log
        if [[ -f "$LOG" ]] && grep -q "Validating the model at step $LINEVAL_STEPS" "$LOG"; then
            echo "[$(date '+%F %T')] SKIP (done): $RUN"
            continue
        fi
        if ! mkdir "$CLAIMS/$RUN" 2>/dev/null; then
            echo "[$(date '+%F %T')] SKIP (claimed by $(cat "$CLAIMS/$RUN/host" 2>/dev/null)): $RUN"
            continue
        fi
        hostname > "$CLAIMS/$RUN/host"
        echo "[$(date '+%F %T')] START $RUN on $(hostname)"
        if SEED=$SEED bash "$ABL_DIR/lineval_ribeiro.sh" "$ARM" "$K" "$CKPT" > /dev/null 2>&1; then
            echo "[$(date '+%F %T')] OK $RUN"
        else
            echo "[$(date '+%F %T')] FAILED $RUN, see $LOG"
            rm -rf "$CLAIMS/$RUN"
        fi
    done
done
$PYTHON "$ABL_DIR/collect_results.py"
