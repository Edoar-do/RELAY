#!/bin/bash
#=============================================================================
# Lead I only / lead II only INFERENCE for the 12-lead-finetuned models
# (extra rows for Table 1 of the RELAY paper).
#
# Rebuttal item (Reviewer wvc3): "Nonetheless, we will add explicit
# inference-time sampling of lead I only and lead II only in Table 1, that is,
# after fairly fine-tuning all models on the complete 12 lead set."
#
# WHAT THIS DOES
#   For every Table-1 model (7 prior FMs + RELAY) and every Table-1 dataset:
#     - loads the existing 12-lead finetuned checkpoint from 12_lead_benchmark/
#     - runs TEST ONLY (--epochs 0 --eval-only <ckpt>), no training whatsoever
#     - keeps ONLY lead I (resp. only lead II) and zeroes out the other 11
#   Both settings are deterministic: the lead is fixed, not sampled, so a
#   single run per (model, dataset, lead) is enough — no seeds, no std.
#
#   Prior FMs see the usual zero-padded 12-channel tensor; RELAY's wrapper
#   detects the zeroed leads and builds a genuine 1-lead graph, exactly as in
#   the random-lead-sampling runs (rls_test_12lead_models.sh).
#
# REQUIRES (already committed alongside this script)
#   code/clinical_ts/data/time_series_dataset_transforms.py : FixedLeadMasking
#   code/main_lite.py                                       : --fixed-leads
#   code/main_lite_base.py                                  : transform wiring
#
# NOTE ON THE 4 EVALUATION-ONLY DATASETS
#   PTB-XL Diag/Form/Rhythm and Chapman Rhythm are label subsets of PTB-XL All
#   and Chapman: they are not run separately, their macro AUROC is recomputed
#   from the per-class AUCs printed in the ptbxl_all / chapman logs. Run
#   `python build_lead_table.py` after this script to get all 18 rows.
#
# LAUNCH:  bash test_lead_I_II_12lead_models.sh
#=============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Clean PyKeOps cache
rm -rf ~/.cache/keops* ~/.cache/pykeops* /tmp/keops*

#=============================================================================
# PYTHON INTERPRETER
# Nothing to activate: the dependencies live in the container's system python
# (past run logs show /usr/local/lib/python3.10/dist-packages). The first of
# python3.10 / python3 / python that exists is used; override with
#   PYTHON_BIN=/path/to/python bash test_lead_I_II_12lead_models.sh
#=============================================================================
if [ -z "${PYTHON_BIN:-}" ]; then
    for _cand in python3.10 python3 python; do
        if command -v "${_cand}" >/dev/null 2>&1; then
            PYTHON_BIN="${_cand}"
            break
        fi
    done
fi

if [ -z "${PYTHON_BIN:-}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: no python interpreter found (tried python3.10, python3, python)."
    echo "       Set PYTHON_BIN=/path/to/python and re-run."
    exit 1
fi
echo "Using interpreter: $(command -v "${PYTHON_BIN}")"

#=============================================================================
# CONFIGURATION
#=============================================================================
BASE_DIR="${BASE_DIR:-$(pwd)}"   # repo root: launch the script from there
CHECKPOINTS_DIR="${BASE_DIR}/pretrained_models"
DATASET_DIR="${BASE_DIR}"

# Where the 12-lead finetuned checkpoints live
CKPT_ROOT="${BASE_DIR}/12_lead_benchmark"

# Where this script writes logs / predictions / summary
BENCH_ROOT="${BASE_DIR}/lead_I_II_test_12lead_models"

# The 8 models of Table 1 (7 prior FMs + RELAY)
MODELS=(ecg_founder ecg_jepa_multiblock st_mem merl_resnet ecgfm_ked hubert_ecg_base cpc relay)

# The 14 finetuned datasets of Table 1. The remaining 4 rows (PTB-XL Diag,
# PTB-XL Form, PTB-XL Rhythm, Chapman Rhythm) are label subsets of ptbxl_all /
# chapman and are recovered from those logs by build_lead_table.py.
DATASETS=(ptbxl_all ptbxl_sub ptbxl_super chapman ningbo sph cpsc2018 cpsc_extra echonext georgia code15_diag zzu_pecg hefei ptb)

# Table 1 reports supervised fine-tuning; add "linear" here to also cover the
# linear-probing checkpoints under 12_lead_benchmark/linear/.
EVAL_MODES=(finetuning_linear)

# Leads to evaluate, canonical names (I = index 0, II = index 1).
# Accepts anything --fixed-leads accepts, e.g. "I,II" for a 2-lead setting.
TEST_LEAD_SPECS=(I II)

LEARNING_RATE=0.001
BATCH_SIZE=64

#=============================================================================
# HELPER FUNCTIONS
#=============================================================================

get_dataset_args() {
    local dataset=$1
    local args=()

    case $dataset in
        "ptb")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/ptb"
                "--fs-data 1000"
                "--finetune-dataset ptb"
            )
            ;;
        "ningbo")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ningbo"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/ningbo"
                "--fs-data 500"
                "--finetune-dataset ningbo"
            )
            ;;
        "cpsc2018")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/cpsc2018"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/cpsc2018"
                "--fs-data 500"
                "--finetune-dataset cpsc2018"
            )
            ;;
        "cpsc_extra")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/cpsc_extra"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/cpsc_extra"
                "--fs-data 500"
                "--finetune-dataset cpsc_extra"
            )
            ;;
        "georgia")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/georgia"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/georgia"
                "--fs-data 500"
                "--finetune-dataset georgia"
            )
            ;;
        "chapman")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/chapman/ECGData"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/chapman/ECGData"
                "--fs-data 500"
                "--finetune-dataset chapman"
            )
            ;;
        "sph")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/sph"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/sph"
                "--fs-data 500"
                "--finetune-dataset sph"
            )
            ;;
        "code15_diag")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/code15"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/code15"
                "--fs-data 400"
                "--finetune-dataset code15_diag"
            )
            ;;
        "ptbxl_super" | "ptbxl_sub" | "ptbxl_all")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb-xl/records500"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/ptb-xl/records500"
                "--fs-data 500"
                "--finetune-dataset ${dataset}"
            )
            ;;
        "echonext")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/echonext"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/echonext"
                "--fs-data 250"
                "--finetune-dataset echonext"
            )
            ;;
        "zzu_pecg")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/zzu_pecg"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/zzu_pecg"
                "--fs-data 500"
                "--finetune-dataset zzu_pecg"
            )
            ;;
        "hefei")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/hefei"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/hefei"
                "--fs-data 500"
                "--finetune-dataset hefei"
            )
            ;;
        "samitrop")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/samitrop"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/samitrop"
                "--fs-data 400"
                "--finetune-dataset samitrop"
            )
            ;;
        *)
            echo "Unknown dataset: $dataset"
            exit 1
            ;;
    esac

    echo "${args[@]}"
}

get_model_args() {
    local model=$1
    local args=()

    case $model in
        "ecg_founder")
            args+=(
                "--architecture ecg_founder"
                "--input-size 2.5"
                "--fs-model 500"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/ecg_founder/12_lead_ECGFounder.pth"
            )
            ;;
        "ecg_jepa_multiblock")
            args+=(
                "--architecture ecg_jepa"
                "--input-size 10"
                "--fs-model 250"
                "--input-channels 8"
                "--pretrained ${CHECKPOINTS_DIR}/ecg_jepa/random_epoch100.pth"
            )
            ;;
        "st_mem")
            args+=(
                "--architecture st_mem"
                "--input-size 2.4"
                "--fs-model 250"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/st_mem/st_mem_vit_base_full.pth"
            )
            ;;
        "merl_resnet")
            args+=(
                "--architecture merl"
                "--merl-backbone resnet"
                "--input-size 2.5"
                "--fs-model 500"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/merl/res18_best_encoder.pth"
            )
            ;;
        "ecgfm_ked")
            args+=(
                "--architecture ecgfm_ked"
                "--input-size 10"
                "--fs-model 500"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
            )
            ;;
        "cpc")
            args+=(
                "--architecture cpc"
                "--input-size 2.5"
                "--fs-model 240"
                "--input-channels 12"
                "--precision 32"
                "--pretrained ${CHECKPOINTS_DIR}/cpc/config_last_11597276_ckpt.yaml"
            )
            ;;
        "hubert_ecg_base")
            args+=(
                "--architecture hubert_ecg"
                "--input-size 5"
                "--fs-model 100"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/hubert_ecg/hubert_ecg_base.safetensors"
                "--hubert-variant base"
            )
            ;;
        "hubert_ecg_small")
            args+=(
                "--architecture hubert_ecg"
                "--input-size 5"
                "--fs-model 100"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/hubert_ecg/hubert_ecg_small.safetensors"
                "--hubert-variant small"
            )
            ;;
        "hubert_ecg_large")
            args+=(
                "--architecture hubert_ecg"
                "--input-size 5"
                "--fs-model 100"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/hubert_ecg/hubert_ecg_large.safetensors"
                "--hubert-variant large"
            )
            ;;
        "ecg_fm")
            args+=(
                "--architecture ecg_fm"
                "--input-size 5"
                "--fs-model 500"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
            )
            ;;
        # RELAY (GAT-based graph ECG model). RelayWrapper detects the zeroed leads
        # and builds a genuine k-lead graph per sample — no zero-feature padding,
        # so with --fixed-leads I it processes a true single-lead graph.
        "relay")
            args+=(
                "--architecture relay"
                "--input-size 5"
                "--fs-model 100"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/relay/pretrained.pt"
                "--relay-ecg-code-path ${BASE_DIR}/relay_code"
                "--relay-config-path ${CHECKPOINTS_DIR}/relay/config.json"
            )
            ;;
        *)
            echo "Unknown model: $model"
            exit 1
            ;;
    esac

    echo "${args[@]}"
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Filesystem-safe tag for a lead spec: "I" → I, "I,II" → I_II
lead_tag() {
    echo "$1" | tr -d ' ' | tr ',' '_'
}

# Check whether an existing test result is valid (contains _test0 metrics
# and prediction .npz files).  If a stale / incomplete result is found, delete
# it so the caller re-runs.
#   Returns 0 (true)  → valid, skip
#           1 (false) → missing or bad, (re)run
is_test_valid() {
    local log_file=$1
    local pred_dir=$2
    local dataset=$3
    if [ -f "${log_file}" ] && grep -q "macro_auc_agg_test0" "${log_file}"; then
        if [ -f "${pred_dir}/${dataset}_version_0/agg/test_0_epoch_0_agg.npz" ] && \
           [ -f "${pred_dir}/${dataset}_version_0/noagg/test_0_epoch_0_noagg.npz" ]; then
            return 0
        fi
    fi
    # Clean up bad / incomplete results
    [ -f "${log_file}" ] && rm -f "${log_file}"
    [ -d "${pred_dir}/${dataset}_version_0" ] && rm -rf "${pred_dir}/${dataset}_version_0"
    return 1
}

#=============================================================================
# MAIN LOOP: TEST-ONLY INFERENCE ON LEAD I ONLY / LEAD II ONLY
#=============================================================================

log_message "Starting fixed-lead test-only inference for the 12-lead finetuned models"
log_message "Models:     ${MODELS[*]}"
log_message "Datasets:   ${DATASETS[*]}"
log_message "Eval modes: ${EVAL_MODES[*]}"
log_message "Leads:      ${TEST_LEAD_SPECS[*]}  (deterministic — one run each, no seeds)"

for EVAL_MODE in "${EVAL_MODES[@]}"; do

  OUTPUT_BASE="${BENCH_ROOT}/${EVAL_MODE}"

  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      # -----------------------------------------------------------------------
      # Locate the newest trained 12-lead checkpoint for this (eval_mode, model,
      # dataset). Training may have produced multiple version_X dirs.
      # -----------------------------------------------------------------------
      CHECKPOINT_PATH=$(ls -t "${CKPT_ROOT}/${EVAL_MODE}/outputs/${MODEL}_${DATASET}/"version_*/best_model.ckpt 2>/dev/null | head -n 1)
      if [ -z "${CHECKPOINT_PATH}" ] || [ ! -f "${CHECKPOINT_PATH}" ]; then
        CHECKPOINT_PATH=$(ls -t "${CKPT_ROOT}/${EVAL_MODE}/outputs/${MODEL}_${DATASET}/"version_*/last.ckpt 2>/dev/null | head -n 1)
      fi
      if [ -z "${CHECKPOINT_PATH}" ] || [ ! -f "${CHECKPOINT_PATH}" ]; then
        log_message "SKIP (no checkpoint): ${EVAL_MODE}/${MODEL}_${DATASET}"
        continue
      fi

      log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      log_message "Testing ${EVAL_MODE} / ${MODEL} / ${DATASET}"
      log_message "Checkpoint: ${CHECKPOINT_PATH}"

      LOGS_DIR="${OUTPUT_BASE}/logs/${MODEL}_${DATASET}"
      mkdir -p "${LOGS_DIR}"

      # Get dataset and model args once for this (model, dataset) pair
      ARGS_DATASET=($(get_dataset_args $DATASET))
      ARGS_MODEL=($(get_model_args $MODEL))

      # -----------------------------------------------------------------------
      # Inner loop: one deterministic run per lead spec (lead I, then lead II).
      # -----------------------------------------------------------------------
      for LEAD_SPEC in "${TEST_LEAD_SPECS[@]}"; do

        LEAD=$(lead_tag "${LEAD_SPEC}")

        TEST_OUTPUT_DIR="${OUTPUT_BASE}/test_time/${MODEL}_${DATASET}/lead${LEAD}"
        TEST_PRED_DIR="${OUTPUT_BASE}/test_time/${MODEL}_${DATASET}/predictions/lead${LEAD}"
        TEST_LOG="${LOGS_DIR}/test_lead${LEAD}.log"

        mkdir -p "${TEST_OUTPUT_DIR}"
        mkdir -p "${TEST_PRED_DIR}"

        # Skip if a valid result is already there; delete and re-run otherwise.
        if is_test_valid "${TEST_LOG}" "${TEST_PRED_DIR}" "${DATASET}"; then
          log_message "  SKIP (valid): lead ${LEAD_SPEC}"
          continue
        fi

        log_message "  Testing lead ${LEAD_SPEC}..."

        # CPC reads ecg_ptbxl.yaml at init time; reset to PTB-XL path.
        if [ "${MODEL}" == "cpc" ]; then
          sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb-xl/records500"|g' \
            ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
        fi

        # -u: unbuffered stdout, so a killed run still leaves a complete log
        # (block-buffered stdout silently swallowed the metrics table when the
        # first launch was interrupted mid-run).
        ${PYTHON_BIN} -u ${BASE_DIR}/code/main_lite.py \
          ${ARGS_DATASET[@]} \
          ${ARGS_MODEL[@]} \
          --epochs 0 \
          --modality ecg \
          --lr ${LEARNING_RATE} \
          --batch-size ${BATCH_SIZE} \
          --finetune \
          --eval-mode ${EVAL_MODE} \
          --output-path "${TEST_OUTPUT_DIR}" \
          --prediction-path "${TEST_PRED_DIR}" \
          --export-predictions \
          --fixed-leads "${LEAD_SPEC}" \
          --eval-only "${CHECKPOINT_PATH}" \
          --no-bootstrap \
          2>&1 | tee "${TEST_LOG}"

        if [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/agg/test_0_epoch_0_agg.npz" ] && \
           [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/noagg/test_0_epoch_0_noagg.npz" ]; then
          log_message "  OK lead ${LEAD_SPEC}"
        else
          log_message "  WARNING: prediction files missing after lead ${LEAD_SPEC} — check ${TEST_LOG}"
        fi

      done

      log_message "Done: ${EVAL_MODE} / ${MODEL} / ${DATASET}"

    done
  done
done

#=============================================================================
# AGGREGATE RESULTS INTO CSV
#=============================================================================

log_message "Aggregating results..."

for EVAL_MODE in "${EVAL_MODES[@]}"; do
  OUTPUT_BASE="${BENCH_ROOT}/${EVAL_MODE}"
  RESULTS_FILE="${BENCH_ROOT}/summary_results_${EVAL_MODE}.csv"

  echo "eval_mode,model,dataset,lead,macro_auc_agg,macro_auc_noagg" > "${RESULTS_FILE}"

  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      LOGS_DIR="${OUTPUT_BASE}/logs/${MODEL}_${DATASET}"
      [ -d "${LOGS_DIR}" ] || continue

      for LEAD_SPEC in "${TEST_LEAD_SPECS[@]}"; do

        LEAD=$(lead_tag "${LEAD_SPEC}")
        LOG_FILE="${LOGS_DIR}/test_lead${LEAD}.log"
        [ -f "${LOG_FILE}" ] || continue

        auc_agg=$(grep "macro_auc_agg_test" "${LOG_FILE}" | tail -1 | grep -oP '\d+\.\d+' || echo "NA")
        auc_noagg=$(grep "macro_auc_noagg_test" "${LOG_FILE}" | tail -1 | grep -oP '\d+\.\d+' || echo "NA")

        echo "${EVAL_MODE},${MODEL},${DATASET},${LEAD},${auc_agg},${auc_noagg}" >> "${RESULTS_FILE}"

      done
    done
  done

  log_message "Results saved to: ${RESULTS_FILE}"

  # Adds the 4 label-subset rows (PTB-XL Diag/Form/Rhythm, Chapman Rhythm) and
  # writes the LaTeX rows to drop into Table 1.
  if [ -f "${BASE_DIR}/analysis/build_lead_table.py" ]; then
    ${PYTHON_BIN} "${BASE_DIR}/analysis/build_lead_table.py" --csv "${RESULTS_FILE}" || \
      log_message "WARNING: build_lead_table.py failed — the CSV above is still valid"
  fi
done

log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_message "DONE"
log_message "  Results: ${BENCH_ROOT}/"
log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
