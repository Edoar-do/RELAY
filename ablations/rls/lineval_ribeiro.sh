#!/bin/bash
# Linear evaluation on Ribeiro-dev of a Stage-1 checkpoint, with K random leads (or one fixed
# lead, K = I | II) in both probe training and validation. Validation leads are fixed per record (VAL_LEAD_SEED), so
# every validation and every arm sees the same lead subsets.
#
# Usage (one container / one GPU each):
#   bash lineval_ribeiro.sh <arm> <K> [checkpoint]
#     arm        rls | norls  (only used for the run name and the default checkpoint)
#     K          1 | 2 | 12 random leads, or I | II (that lead only)
#     checkpoint defaults to $PRETRAINED_DIR/gat_60000_stage1_<arm>.pt
#   SMOKE=1 bash lineval_ribeiro.sh norls 1   # 200 steps, validation every 100
#   SEED=43 bash lineval_ribeiro.sh rls 1      # another lin-eval seed: head init, data order, training
#                                              # AND validation lead draws (--val_lead_seed=SEED); default
#                                              # SEED=42 with validation lead seed 0 = the original runs

set -euo pipefail
source "$(dirname "$0")/common.sh"

ARM=${1:?usage: lineval_ribeiro.sh rls|norls K [checkpoint]}
K=${2:?usage: lineval_ribeiro.sh rls|norls K [checkpoint]}
CKPT=${3:-$PRETRAINED_DIR/gat_${PRETRAIN_STEPS}_stage1_${ARM}.pt}

[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT"; exit 1; }
CKPT=$(readlink -f "$CKPT")   # the script cd-s into $CODE_DIR below

SEED=${SEED:-42}
if [[ "$SEED" == "42" ]]; then SUFFIX=""; VAL_SEED=$VAL_LEAD_SEED; else SUFFIX="_s${SEED}"; VAL_SEED=$SEED; fi
case "$K" in
    I|II) RUN_NAME=lineval_stage1_${ARM}_lead${K}${SUFFIX}; LEAD_ARGS=(--sampling "$K") ;;
    *)    RUN_NAME=lineval_stage1_${ARM}_K${K}${SUFFIX};    LEAD_ARGS=(--sampling random --how_many_leads=$K --val_lead_seed=$VAL_SEED) ;;
esac
# rls_zeropad: the K probe leads are zero-padded to 12 (training and validation), as in its pretraining
[[ "$ARM" == "rls_zeropad" ]] && LEAD_ARGS+=(--pad_unsampled_leads)
OUT_DIR=$FINETUNED_DIR
STEP_ARGS=(--training_steps=$LINEVAL_STEPS --val_frequency=5000)

if [[ "${SMOKE:-0}" == "1" ]]; then
    RUN_NAME=smoke_${RUN_NAME}
    OUT_DIR=$LOG_DIR/smoke
    mkdir -p "$OUT_DIR"
    STEP_ARGS=(--training_steps=200 --val_frequency=100)
fi

echo "log: $LOG_DIR/${RUN_NAME}.log"
cd "$CODE_DIR"
$PYTHON finetune.py \
    --path_to_train_dataset_csv=$DATA/ribeiro_dev_train.csv \
    --ecg_train_dir_path=./ \
    --path_to_val_dataset_csv=$DATA/ribeiro_dev_val.csv \
    --ecg_val_dir_path=./ \
    "${LEAD_ARGS[@]}" \
    --seed=$SEED \
    --wandb \
    --wandb_group=$WANDB_GROUP \
    --wandb_run_name=$RUN_NAME \
    --checkpoint_path=$CKPT \
    --encoder_name=gat \
    --model_config_path=$MODEL_CONFIG \
    --create_edge_index \
    --edge_strategy=spatiotemporal \
    --batch_size=64 \
    --accumulation_steps=8 \
    --finetuned_models_path=$OUT_DIR \
    --num_workers=6 \
    --patience=10 \
    --validations_for_penalty=5 \
    --label_start_index=3 \
    --lin_eval \
    --lr=1e-3 \
    "${STEP_ARGS[@]}" \
    > "$LOG_DIR/${RUN_NAME}.log" 2>&1   # not `| tee`: uutils tee 0.10.0 can panic and kill the run
