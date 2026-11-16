#!/bin/bash
#=============================================================================
# ST-MEM WITHOUT ZERO-PADDING at test time, on its existing 12-lead finetuned
# checkpoints (inference only, no training).
#
# With --st-mem-native-leads, StMemWrapper drops the zeroed leads before
# encoding and tags each remaining lead with the lead embedding of its TRUE
# index (e.g. lead II alone gets lead_embeddings[1]). Previously the 11 zeroed
# leads were encoded too and made up 11/12 of the mean-pooled tokens.
#
# Re-runs exactly the cells of the two existing tables, with the same
# checkpoints, seeds and lead-masking transforms, under the model tag
# "st_mem_native" so nothing of the zero-padded "st_mem" results is touched:
#   part "rls"   : k = 1, 2 random leads x seeds 42..46  (as rls_test_12lead_models.sh), 14 datasets
#                  -> ktest_12lead_models/<eval_mode>/{logs,test_time}/st_mem_native_<dataset>/
#   part "leads" : lead I only, lead II only              (as test_lead_I_II_12lead_models.sh), 14 datasets
#                  -> lead_I_II_test_12lead_models/<eval_mode>/{logs,test_time}/st_mem_native_<dataset>/
# Same seed + same transform + same 4 test workers => the random lead draws
# should be the very ones the zero-padded st_mem runs used (paired comparison).
#
# Dataset/lead helpers are taken verbatim from the two original scripts, so
# the dataset arguments are identical by construction.
#
# LAUNCH:  bash test_st_mem_native_12lead_models.sh            # both parts
#          PART=rls   bash test_st_mem_native_12lead_models.sh
#          PART=leads bash test_st_mem_native_12lead_models.sh
#          DATASETS="ptbxl_all ningbo" bash test_st_mem_native_12lead_models.sh
# THEN:    python analysis/compare_st_mem_native.py
#=============================================================================

set -e
set -u

BASE_DIR="${BASE_DIR:-$(pwd)}"   # repo root: launch the script from there
CHECKPOINTS_DIR="${BASE_DIR}/pretrained_models"
DATASET_DIR="${BASE_DIR}"
CKPT_ROOT="${BASE_DIR}/12_lead_benchmark"

PART=${PART:-all}
EVAL_MODES=(finetuning_linear)

# The 14 finetuned datasets of Table 1 (the other 4 rows are label subsets, see build_lead_table.py).
# SaMi-Trop is not in the paper's tables and is left out.
TABLE_DATASETS=(ptbxl_all ptbxl_sub ptbxl_super chapman ningbo sph cpsc2018 cpsc_extra echonext georgia code15_diag zzu_pecg hefei ptb)
if [ -n "${DATASETS:-}" ]; then
    read -r -a TABLE_DATASETS <<< "${DATASETS}"
fi
RLS_DATASETS=("${TABLE_DATASETS[@]}")
LEAD_DATASETS=("${TABLE_DATASETS[@]}")

TEST_LEADS=(1 2)
TEST_SEEDS=(42 43 44 45 46)
TEST_LEAD_SPECS=(I II)

MODEL_TAG="st_mem_native"   # output / log naming
CKPT_MODEL="st_mem"         # whose 12-lead finetuned checkpoints are loaded

LEARNING_RATE=0.001
BATCH_SIZE=64

# --- interpreter: the env from setup_bench_env.sh, else the rule of test_lead_I_II_12lead_models.sh ---
if [ -z "${PYTHON_BIN:-}" ] && [ -x python ]; then
    PYTHON_BIN=python
fi
if [ -z "${PYTHON_BIN:-}" ]; then
    for _cand in python3.10 python3 python; do
        if command -v "${_cand}" >/dev/null 2>&1; then PYTHON_BIN="${_cand}"; break; fi
    done
fi
command -v "${PYTHON_BIN:-}" >/dev/null 2>&1 || { echo "ERROR: no python found; set PYTHON_BIN"; exit 1; }
echo "Using interpreter: $(command -v "${PYTHON_BIN}")"

# --- helpers copied from the original scripts (single source of truth) ---
eval "$(sed -n '/^get_dataset_args() {/,/^}/p' "${BASE_DIR}/rls_test_12lead_models.sh")"
eval "$(sed -n '/^is_test_valid() {/,/^}/p' "${BASE_DIR}/rls_test_12lead_models.sh")"
eval "$(sed -n '/^lead_tag() {/,/^}/p' "${BASE_DIR}/test_lead_I_II_12lead_models.sh")"
for fn in get_dataset_args is_test_valid lead_tag; do
    declare -F "${fn}" >/dev/null || { echo "ERROR: could not load ${fn} from the original scripts"; exit 1; }
done

ARGS_MODEL=(
    --architecture st_mem
    --input-size 2.4
    --fs-model 250
    --input-channels 12
    --pretrained "${CHECKPOINTS_DIR}/st_mem/st_mem_vit_base_full.pth"
    --st-mem-native-leads
)

log_message() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

find_checkpoint() {
    local eval_mode=$1 dataset=$2 ckpt
    ckpt=$(ls -t "${CKPT_ROOT}/${eval_mode}/outputs/${CKPT_MODEL}_${dataset}/"version_*/best_model.ckpt 2>/dev/null | head -n 1)
    [ -n "${ckpt}" ] || ckpt=$(ls -t "${CKPT_ROOT}/${eval_mode}/outputs/${CKPT_MODEL}_${dataset}/"version_*/last.ckpt 2>/dev/null | head -n 1)
    echo "${ckpt}"
}

# run_one <dataset> <output_dir> <pred_dir> <log> <lead-selection args...>
run_one() {
    local dataset=$1 out_dir=$2 pred_dir=$3 log=$4; shift 4
    mkdir -p "${out_dir}" "${pred_dir}"
    if is_test_valid "${log}" "${pred_dir}" "${dataset}"; then
        log_message "  SKIP (valid): ${log##*/}"
        return
    fi
    log_message "  Testing ${log##*/} ..."
    ${PYTHON_BIN} -u ${BASE_DIR}/code/main_lite.py \
        ${ARGS_DATASET[@]} \
        "${ARGS_MODEL[@]}" \
        --epochs 0 \
        --modality ecg \
        --lr ${LEARNING_RATE} \
        --batch-size ${BATCH_SIZE} \
        --finetune \
        --eval-mode ${EVAL_MODE} \
        --output-path "${out_dir}" \
        --prediction-path "${pred_dir}" \
        --export-predictions \
        "$@" \
        --eval-only "${CHECKPOINT_PATH}" \
        --no-bootstrap \
        2>&1 | tee "${log}"
    if grep -q "native lead mode" "${log}" && grep -q "macro_auc_agg_test0" "${log}"; then
        log_message "  OK ${log##*/}"
    else
        log_message "  WARNING: ${log##*/} has no native-mode marker or no test metric, check it"
    fi
}

for EVAL_MODE in "${EVAL_MODES[@]}"; do

    # ---------------- part 1: k random leads x seeds ----------------
    if [ "${PART}" = "all" ] || [ "${PART}" = "rls" ]; then
        OUTPUT_BASE="${BASE_DIR}/ktest_12lead_models/${EVAL_MODE}"
        for DATASET in "${RLS_DATASETS[@]}"; do
            CHECKPOINT_PATH=$(find_checkpoint "${EVAL_MODE}" "${DATASET}")
            [ -f "${CHECKPOINT_PATH}" ] || { log_message "SKIP (no checkpoint): ${CKPT_MODEL}_${DATASET}"; continue; }
            log_message "RLS ${EVAL_MODE} / ${MODEL_TAG} / ${DATASET}  ckpt: ${CHECKPOINT_PATH}"
            ARGS_DATASET=($(get_dataset_args ${DATASET}))
            LOGS_DIR="${OUTPUT_BASE}/logs/${MODEL_TAG}_${DATASET}"
            mkdir -p "${LOGS_DIR}"
            for k in "${TEST_LEADS[@]}"; do
                for seed in "${TEST_SEEDS[@]}"; do
                    run_one "${DATASET}" \
                        "${OUTPUT_BASE}/test_time/${MODEL_TAG}_${DATASET}/k${k}_seed${seed}" \
                        "${OUTPUT_BASE}/test_time/${MODEL_TAG}_${DATASET}/predictions/k${k}_seed${seed}" \
                        "${LOGS_DIR}/test_k${k}_seed${seed}.log" \
                        --random_lead_sampling --k ${k} --seed ${seed}
                done
            done
        done
    fi

    # ---------------- part 2: lead I only / lead II only ----------------
    if [ "${PART}" = "all" ] || [ "${PART}" = "leads" ]; then
        OUTPUT_BASE="${BASE_DIR}/lead_I_II_test_12lead_models/${EVAL_MODE}"
        for DATASET in "${LEAD_DATASETS[@]}"; do
            CHECKPOINT_PATH=$(find_checkpoint "${EVAL_MODE}" "${DATASET}")
            [ -f "${CHECKPOINT_PATH}" ] || { log_message "SKIP (no checkpoint): ${CKPT_MODEL}_${DATASET}"; continue; }
            log_message "LEADS ${EVAL_MODE} / ${MODEL_TAG} / ${DATASET}  ckpt: ${CHECKPOINT_PATH}"
            ARGS_DATASET=($(get_dataset_args ${DATASET}))
            LOGS_DIR="${OUTPUT_BASE}/logs/${MODEL_TAG}_${DATASET}"
            mkdir -p "${LOGS_DIR}"
            for LEAD_SPEC in "${TEST_LEAD_SPECS[@]}"; do
                LEAD=$(lead_tag "${LEAD_SPEC}")
                run_one "${DATASET}" \
                    "${OUTPUT_BASE}/test_time/${MODEL_TAG}_${DATASET}/lead${LEAD}" \
                    "${OUTPUT_BASE}/test_time/${MODEL_TAG}_${DATASET}/predictions/lead${LEAD}" \
                    "${LOGS_DIR}/test_lead${LEAD}.log" \
                    --fixed-leads "${LEAD_SPEC}"
            done
        done
    fi
done

log_message "DONE. Compare with: python ${BASE_DIR}/analysis/compare_st_mem_native.py"
