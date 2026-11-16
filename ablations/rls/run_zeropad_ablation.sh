#!/bin/bash
# Zero-padding ablation of RLS (arm `rls_zeropad`), end to end on this container's GPU:
#
#   1. Stage-1 pre-training, 60k steps, exactly as `rls` (L ~ U{1..12} per ECG, batch 64, 8-step
#      accumulation, lr 1e-3, p_mask 0.4, config_gat.json, dynamic reg., seed 42), except that the
#      unsampled leads are zero-filled instead of dropped: every graph keeps the full 12-subgraph
#      topology, the unsampled leads becoming "null" subgraphs. Only the nodes of sampled leads are
#      masked and scored, so the masked targets per step match `rls`.
#   2. Linear probes on Ribeiro-dev (lineval_ribeiro.sh, 70k steps), trained and validated on
#      K = 12, K = 1, 2 random leads, lead I, lead II, with zero-padded unsampled leads (as in
#      pre-training); plus lin-eval seeds 43 / 44 at K = 1, 2 (as for `rls`).
#   3. Zero-shot test (eval_zeroshot.py) of every probe of `rls` and `rls_zeropad` at K = 12 (sanity),
#      K = 1, 2 random leads (lead seeds 0 / 43 / 44), lead I and lead II, with the missing leads both
#      dropped (variable graph) and zero-padded (12-lead graph with null subgraphs).
#      Output: zeroshot_zeropad/{results.csv,table.md}
#
# Every step is skipped if its output is already there, so the script can be re-run after a crash.
#
# Usage:  bash run_zeropad_ablation.sh
#         SKIP_PRETRAIN=1 bash run_zeropad_ablation.sh   # checkpoint must already exist
#         LINEVAL_SEEDS="" bash run_zeropad_ablation.sh   # no extra lin-eval seeds at K = 1, 2
#         ZS_ARMS="rls rls_zeropad norls_pm0217" bash run_zeropad_ablation.sh   # more arms in step 3
# Smoke tests of steps 1-2: SMOKE=1 bash pretrain_stage1.sh rls_zeropad; SMOKE=1 bash lineval_ribeiro.sh rls_zeropad 1

set -euo pipefail
source "$(dirname "$0")/common.sh"

ARM=rls_zeropad
CKPT=$PRETRAINED_DIR/gat_${PRETRAIN_STEPS}_stage1_${ARM}.pt
read -r -a EXTRA_SEEDS <<< "${LINEVAL_SEEDS-43 44}"
read -r -a ZS_ARM_LIST <<< "${ZS_ARMS:-rls rls_zeropad}"

lineval_done() {  # finished = the last validation was logged
    [[ -f "$1" ]] && grep -q "Validating the model at step $LINEVAL_STEPS" "$1"
}

# --- 1. pre-training ---
if [[ -f "$CKPT" ]]; then
    echo "[$(date '+%F %T')] SKIP pre-training: $CKPT exists"
elif [[ "${SKIP_PRETRAIN:-0}" == "1" ]]; then
    echo "SKIP_PRETRAIN=1 but no checkpoint at $CKPT"; exit 1
else
    echo "[$(date '+%F %T')] START pre-training $ARM"
    bash "$ABL_DIR/pretrain_stage1.sh" "$ARM"
fi
[[ -f "$CKPT" ]] || { echo "no 60k checkpoint at $CKPT (early stopping?); see $LOG_DIR/stage1_${ARM}.log"; exit 1; }

# --- 2. linear probes (zero-padded, as in pre-training) ---
PROBES=(K12 K1 K2 leadI leadII)
run_probe() {  # <K: 12|1|2|I|II> <seed> <probe name>
    local K=$1 S=$2 NAME=$3
    local LOG=$LOG_DIR/lineval_stage1_${ARM}_${NAME}.log
    if lineval_done "$LOG"; then
        echo "[$(date '+%F %T')] SKIP (done): probe $NAME"
        return
    fi
    echo "[$(date '+%F %T')] START probe $NAME"
    SEED=$S bash "$ABL_DIR/lineval_ribeiro.sh" "$ARM" "$K" "$CKPT"
}
for K in 12 1 2; do run_probe "$K" 42 "K${K}"; done
for L in I II;    do run_probe "$L" 42 "lead${L}"; done
for S in ${EXTRA_SEEDS[@]+"${EXTRA_SEEDS[@]}"}; do
    for K in 1 2; do
        run_probe "$K" "$S" "K${K}_s${S}"
        PROBES+=("K${K}_s${S}")
    done
done

# --- 3. zero-shot test, missing leads dropped and zero-padded ---
echo "[$(date '+%F %T')] START zero-shot evaluation: arms ${ZS_ARM_LIST[*]}, probes ${PROBES[*]}"
$PYTHON "$ABL_DIR/eval_zeroshot.py" \
    --arms "${ZS_ARM_LIST[@]}" \
    --probes "${PROBES[@]}" \
    --settings K12 K1 K2 leadI leadII \
    --modes drop pad \
    --out "$ABL_DIR/zeroshot_zeropad" \
    > "$LOG_DIR/eval_zeroshot_zeropad.out" 2>&1

$PYTHON "$ABL_DIR/collect_results.py"
echo
cat "$ABL_DIR/zeroshot_zeropad/table.md"
