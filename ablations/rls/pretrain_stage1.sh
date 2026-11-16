#!/bin/bash
# Stage-1 pre-training (masked node modelling, C=50 codebook) with or without random lead sampling.
#
# Usage (one container / one GPU each):
#   bash pretrain_stage1.sh rls            # L ~ U{1..12} per ECG (re-run of the original with the current code)
#   bash pretrain_stage1.sh norls          # all 12 leads, batch 64, p_mask 0.4 (step-matched: ~1.85x the
#                                          #   segments and masked targets of rls per step)
#   bash pretrain_stage1.sh norls_tok      # all 12 leads, batch 35, p_mask 0.4: token-matched to rls
#                                          #   (E[segments/step] 8,400 vs 8,320; masked 3,360 vs 3,328) [primary]
#   bash pretrain_stage1.sh norls_pm0217   # all 12 leads, batch 64, p_mask 0.2167 = 52/240: masked-target-
#                                          #   matched to rls (3,328 per step) [secondary sensitivity]
#   bash pretrain_stage1.sh rls_zeropad    # as rls (L ~ U{1..12}, batch 64, p_mask 0.4), but the unsampled
#                                          #   leads are kept as all-zero signals (full 12-lead graph) instead
#                                          #   of dropped; only sampled leads' nodes are masked/scored, so the
#                                          #   masked targets per step match rls. Zero-padding counterpart of
#                                          #   RLS (cf. random lead masking, Oh et al.): isolates the input
#                                          #   representation (variable graph vs zero padding).
#   SMOKE=1 bash pretrain_stage1.sh norls   # 200 steps, no validation, checkpoints to $LOG_DIR/smoke

set -euo pipefail
source "$(dirname "$0")/common.sh"

ARM=${1:?usage: pretrain_stage1.sh rls|rls_zeropad|norls|norls_tok|norls_pm0217}
BATCH_SIZE=64
case $ARM in
    rls)          LEAD_ARGS=() ;;
    rls_zeropad)  LEAD_ARGS=(--pad_unsampled_leads) ;;
    norls)        LEAD_ARGS=(--train_how_many_leads=12) ;;
    norls_tok)    LEAD_ARGS=(--train_how_many_leads=12); BATCH_SIZE=35 ;;
    norls_pm0217) LEAD_ARGS=(--train_how_many_leads=12); MODEL_CONFIG=$ABL_DIR/config_gat_pm0217.json ;;
    *)            echo "unknown arm: $ARM (use rls, rls_zeropad, norls, norls_tok or norls_pm0217)"; exit 1 ;;
esac

RUN_NAME=stage1_${ARM}
OUT_DIR=$PRETRAINED_DIR
STEP_ARGS=(--training_steps=$PRETRAIN_STEPS --val_frequency=5000)

if [[ "${SMOKE:-0}" == "1" ]]; then
    RUN_NAME=smoke_${RUN_NAME}
    OUT_DIR=$LOG_DIR/smoke
    mkdir -p "$OUT_DIR"
    STEP_ARGS=(--training_steps=200 --val_frequency=100 --skip_val --log_frequency=100)
fi

echo "log: $LOG_DIR/${RUN_NAME}.log"
cd "$CODE_DIR"
$PYTHON pretrain.py \
    "$KMEANS_STAGE1" \
    $DATA/ssl_train.csv \
    $DATA/train_self_supervised/ \
    $DATA/grouped_features_ssl_train/ \
    $DATA/ssl_val.csv \
    $DATA/val_self_supervised/ \
    $DATA/grouped_features_ssl_val \
    --wandb \
    --wandb_group=$WANDB_GROUP \
    --wandb_run_name=$RUN_NAME \
    --encoder_name=gat \
    --model_config_path=$MODEL_CONFIG \
    --create_edge_index \
    --edge_strategy=spatiotemporal \
    --batch_size=$BATCH_SIZE \
    --accumulation_steps=8 \
    --pretrained_models_path=$OUT_DIR \
    --dynamic_reg \
    --validations_for_penalty=1 \
    --num_workers=6 \
    --patience=10 \
    --lr=1e-3 \
    --save_last \
    "${STEP_ARGS[@]}" \
    ${LEAD_ARGS[@]+"${LEAD_ARGS[@]}"} \
    > "$LOG_DIR/${RUN_NAME}.log" 2>&1   # not `| tee`: uutils tee 0.10.0 can panic and kill the run

echo "Expected checkpoint: $OUT_DIR/gat_${PRETRAIN_STEPS}_${RUN_NAME}.pt"
