#!/bin/bash
#=============================================================================
# ZERO-SHOT inference on the point-of-care datasets with the PTB-XL All
# (12-lead, finetuning_linear) checkpoints. Inference only, nothing is trained.
#
#   cinc2017   : AliveCor single lead, placed in the lead I slot (approximates lead I)
#   icentia11k : CardioSTAT patch lead, not a standard lead: lead II slot (primary)
#                and lead I slot; ORACLE=1 adds the other 10 slots, for the
#                "best slot chosen on test" analysis (explicitly baseline-favouring)
#
# Every model gets the same single channel in a zero-filled 12-lead input
# (--single-lead-slot). RELAY and ST-MEM native (--st-mem-native-leads) detect the
# zeros and process only the present lead; the others see the zero-filled tensor.
# Each run exports per-record predictions of the 71 PTB-XL All heads, averaged
# over the model's windows; eval_poc_zero_shot.py scores the mapped heads.
#
# Outputs: zero_shot/results/<dataset>/slot_<lead>/<model>/{run.log, predictions/}
# LAUNCH:  bash run_poc_zero_shot.sh                     # all models, cinc2017 I + icentia11k II, I
#          ORACLE=1 bash run_poc_zero_shot.sh            # + icentia11k remaining 10 slots
#          MODELS="relay st_mem" SETTINGS="cinc2017:I" bash run_poc_zero_shot.sh
# Finished runs (predictions present) are skipped, so the script can be re-launched.
#=============================================================================
set -u

# Clean PyKeOps cache (as the other benchmark scripts; CPC's S4 kernel). Without a C++ compiler KeOps
# cannot build its GPU module and s42.cauchy_conj falls back to an equivalent plain-PyTorch kernel.
rm -rf ~/.cache/keops* ~/.cache/pykeops* /tmp/keops*

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_DIR=${REPO_DIR}
CHECKPOINTS_DIR=${BASE_DIR}/pretrained_models
DATASET_DIR=${BASE_DIR}
CKPT_ROOT=${BASE_DIR}/12_lead_benchmark/finetuning_linear/outputs
OUT_ROOT=${BASE_DIR}/zero_shot/results
EVAL_MODE=finetuning_linear

read -r -a MODELS <<< "${MODELS:-ecg_founder ecg_jepa_multiblock st_mem st_mem_native merl_resnet ecgfm_ked hubert_ecg_base cpc relay}"
read -r -a SETTINGS <<< "${SETTINGS:-cinc2017:I icentia11k:II icentia11k:I}"
if [ "${ORACLE:-0}" = "1" ]; then
    for lead in III aVR aVL aVF V1 V2 V3 V4 V5 V6; do SETTINGS+=("icentia11k:${lead}"); done
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x python ]; then PYTHON_BIN=python; else PYTHON_BIN=python3; fi
fi
echo "Using interpreter: ${PYTHON_BIN}"

# model arguments: taken verbatim from rls_test_12lead_models.sh (single source of truth)
eval "$(sed -n '/^get_model_args() {/,/^}/p' "${BASE_DIR}/rls_test_12lead_models.sh")"
declare -F get_model_args >/dev/null || { echo "ERROR: could not load get_model_args"; exit 1; }

declare -A FS_DATA=([cinc2017]=300 [icentia11k]=250)

log_message() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

for setting in "${SETTINGS[@]}"; do
    DATASET=${setting%%:*}
    SLOT=${setting##*:}
    for MODEL in "${MODELS[@]}"; do
        # st_mem_native = the st_mem checkpoint run with zeroed leads dropped before encoding
        CKPT_MODEL=${MODEL%_native}
        EXTRA=()
        [ "${MODEL}" = "st_mem_native" ] && EXTRA=(--st-mem-native-leads)

        CKPT=$(ls -t ${CKPT_ROOT}/${CKPT_MODEL}_ptbxl_all/version_*/best_model.ckpt 2>/dev/null | head -n 1)
        if [ -z "${CKPT}" ]; then
            log_message "SKIP (no ptbxl_all checkpoint): ${CKPT_MODEL}"
            continue
        fi

        RUN_DIR=${OUT_ROOT}/${DATASET}/slot_${SLOT}/${MODEL}
        PRED=${RUN_DIR}/predictions/${DATASET}_version_0/agg/test_0_epoch_0_agg.npz
        if [ -f "${PRED}" ]; then
            log_message "SKIP (done): ${DATASET} slot ${SLOT} ${MODEL}"
            continue
        fi
        rm -rf "${RUN_DIR}"
        mkdir -p "${RUN_DIR}"

        # CPC reads ecg_ptbxl.yaml at init time; keep it on the PTB-XL path (as the other scripts do)
        if [ "${CKPT_MODEL}" = "cpc" ]; then
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb-xl/records500"|g' \
                ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
        fi

        log_message "${DATASET} slot ${SLOT} ${MODEL}  (ckpt ${CKPT})"
        ${PYTHON_BIN} -u ${BASE_DIR}/code/main_lite.py \
            --data ${DATASET_DIR}/${DATASET} \
            --fs-data ${FS_DATA[$DATASET]} \
            --finetune-dataset ${DATASET} \
            --single-lead-slot ${SLOT} \
            $(get_model_args ${CKPT_MODEL}) \
            "${EXTRA[@]}" \
            --epochs 0 \
            --modality ecg \
            --lr 0.001 \
            --batch-size 64 \
            --finetune \
            --eval-mode ${EVAL_MODE} \
            --output-path "${RUN_DIR}/out" \
            --prediction-path "${RUN_DIR}/predictions" \
            --export-predictions \
            --eval-only "${CKPT}" \
            --no-bootstrap \
            > "${RUN_DIR}/run.log" 2>&1

        if [ -f "${PRED}" ]; then
            log_message "  OK"
        else
            log_message "  FAILED, see ${RUN_DIR}/run.log"
        fi
    done
done
log_message "DONE. Score with: ${PYTHON_BIN} ${BASE_DIR}/zero_shot/eval_poc_zero_shot.py"
