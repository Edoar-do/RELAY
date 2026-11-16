#!/bin/bash
# Shared settings for the Stage-1 RLS vs noRLS ablation (E1).
# Every value here mirrors the original Stage-1 run `ablation_gat_spatiotemporal_mask=0.4`
# and its Ribeiro-dev linear evaluation.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Python of the env created by setup_env.sh (override with PYTHON=...)
PYTHON=${PYTHON:-${ENV_DIR:-${HOME}/relay_env}/bin/python}

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ABL_DIR=${REPO_DIR}/ablations/rls
CODE_DIR=${REPO_DIR}/relay_code
MODEL_CONFIG=${REPO_DIR}/ablations/config_gat.json   # 2L GAT, 768 ch, 8 heads, p_mask=0.4, C=50, edge_drop=0
KMEANS_STAGE1=$CODE_DIR/kmeans_path.txt                         # k_means_50 (MFCC codebook, lead-independent)

DATA=${DATA:?set DATA to the pretraining data folder}
PRETRAINED_DIR=$DATA/ECG_pretraining/models/any_lead/pretrained
FINETUNED_DIR=$DATA/ECG_pretraining/models/any_lead/finetuned
LOG_DIR=$ABL_DIR/logs

# wandb is disabled by default (results are parsed from the logs); set WANDB_MODE=online to re-enable
export WANDB_MODE=${WANDB_MODE:-disabled}
WANDB_GROUP=${WANDB_GROUP:-rls_ablation_stage1}

# Pre-training budget: 60k steps (one step = one micro-batch of 64; 8-step accumulation -> 512 effective)
PRETRAIN_STEPS=60000
# Linear evaluation budget on Ribeiro-dev
LINEVAL_STEPS=70000
# Seed for the fixed validation lead draws (shared by every lin-eval job, so RLS and noRLS see identical leads)
VAL_LEAD_SEED=0

mkdir -p "$LOG_DIR"
