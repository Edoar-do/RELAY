#!/bin/bash
#SBATCH --job-name=rls_test_12lead
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out

set -e  # Exit on error
set -u  # Exit on undefined variable

# Clean PyKeOps cache
rm -rf ~/.cache/keops* ~/.cache/pykeops* /tmp/keops*

#=============================================================================
# CONFIGURATION
#=============================================================================
BASE_DIR="${BASE_DIR:-$(pwd)}"   # repo root: launch the script from there
CHECKPOINTS_DIR="${BASE_DIR}/pretrained_models"
DATASET_DIR="${BASE_DIR}"

CKPT_ROOT="${BASE_DIR}/12_lead_benchmark"

MODELS=(ecg_founder ecg_jepa_multiblock st_mem merl_resnet ecgfm_ked hubert_ecg_base cpc relay)


DATASETS=(ptbxl_all ptbxl_sub ptbxl_super chapman ningbo sph cpsc2018 cpsc_extra echonext georgia code15_diag zzu_pecg hefei ptb)

EVAL_MODES=(finetuning_linear)

TEST_LEADS=(1 2 3 4 5 6 7 8 9 10 11 12)
TEST_SEEDS=(42 43 44 45 46)

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
        # RelayWrapper detects zeroed leads and builds a genuine k-lead graph per
        # sample — no zero-feature padding. RELAY processes exactly the active leads.
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

# Check whether an existing test result is valid (contains _test0 metrics
# and prediction .npz files).  If a stale _val0 result is found, delete it
# so the caller re-runs.
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
# MAIN LOOP: TEST-ONLY INFERENCE AT k=1..12 LEADS
#=============================================================================

log_message "Starting RLS test-only inference for trained 12-lead models"
log_message "Models:     ${MODELS[*]}"
log_message "Datasets:   ${DATASETS[*]}"
log_message "Eval modes: ${EVAL_MODES[*]}"
log_message "k leads:    ${TEST_LEADS[*]}"
log_message "Seeds:      ${TEST_SEEDS[*]}  (k=12 uses only first seed)"

for EVAL_MODE in "${EVAL_MODES[@]}"; do

  OUTPUT_BASE="${BASE_DIR}/ktest_12lead_models/${EVAL_MODE}"

  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      # -----------------------------------------------------------------------
      # Locate the newest trained 12-lead checkpoint for this (eval_mode, model,
      # dataset). Both local runs and Steve's symlinks live under the same
      # outputs/ tree; training may have produced multiple version_X dirs.
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
      # Inner loop: k leads × seeds
      # k=12 is deterministic (all leads always selected), so one seed suffices.
      # -----------------------------------------------------------------------
      for k in "${TEST_LEADS[@]}"; do

        if [ "${k}" -eq 12 ]; then
          seeds=("${TEST_SEEDS[0]}")
        else
          seeds=("${TEST_SEEDS[@]}")
        fi

        for seed in "${seeds[@]}"; do

          TEST_OUTPUT_DIR="${OUTPUT_BASE}/test_time/${MODEL}_${DATASET}/k${k}_seed${seed}"
          TEST_PRED_DIR="${OUTPUT_BASE}/test_time/${MODEL}_${DATASET}/predictions/k${k}_seed${seed}"
          TEST_LOG="${LOGS_DIR}/test_k${k}_seed${seed}.log"

          mkdir -p "${TEST_OUTPUT_DIR}"
          mkdir -p "${TEST_PRED_DIR}"

          # Skip if result is valid (_test0 in log + prediction files exist);
          # delete and re-run if stale _val0 result is found.
          if is_test_valid "${TEST_LOG}" "${TEST_PRED_DIR}" "${DATASET}"; then
            log_message "  SKIP (valid): k=${k} seed=${seed}"
            continue
          fi

          log_message "  Testing k=${k} seed=${seed}..."

          # CPC reads ecg_ptbxl.yaml at init time; reset to PTB-XL path.
          if [ "${MODEL}" == "cpc" ]; then
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb-xl/records500"|g' \
              ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
          fi

          python3.10 ${BASE_DIR}/code/main_lite.py \
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
            --random_lead_sampling \
            --k ${k} \
            --seed ${seed} \
            --eval-only "${CHECKPOINT_PATH}" \
            --no-bootstrap \
            2>&1 | tee "${TEST_LOG}"

          if [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/agg/test_0_epoch_0_agg.npz" ] && \
             [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/noagg/test_0_epoch_0_noagg.npz" ]; then
            log_message "  OK k=${k} seed=${seed}"
          else
            log_message "  WARNING: prediction files missing after k=${k} seed=${seed} — check ${TEST_LOG}"
          fi

        done
      done

      log_message "Done: ${EVAL_MODE} / ${MODEL} / ${DATASET}"

    done
  done
done

#=============================================================================
# AGGREGATE RESULTS INTO CSV
#=============================================================================

log_message "Aggregating results..."

RESULTS_FILE="${BASE_DIR}/ktest_12lead_models/summary_results_12lead_models.csv"

if [ ! -f "${RESULTS_FILE}" ]; then
  echo "eval_mode,model,dataset,k,seed,macro_auc_agg,macro_auc_noagg" > "${RESULTS_FILE}"
fi

for EVAL_MODE in "${EVAL_MODES[@]}"; do
  OUTPUT_BASE="${BASE_DIR}/ktest_12lead_models/${EVAL_MODE}"

  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      LOGS_DIR="${OUTPUT_BASE}/logs/${MODEL}_${DATASET}"
      [ -d "${LOGS_DIR}" ] || continue

      for k in "${TEST_LEADS[@]}"; do

        if [ "${k}" -eq 12 ]; then
          seeds=("${TEST_SEEDS[0]}")
        else
          seeds=("${TEST_SEEDS[@]}")
        fi

        for seed in "${seeds[@]}"; do

          LOG_FILE="${LOGS_DIR}/test_k${k}_seed${seed}.log"
          [ -f "${LOG_FILE}" ] || continue

          auc_agg=$(grep "macro_auc_agg_test" "${LOG_FILE}" | tail -1 | grep -oP '\d+\.\d+' || echo "NA")
          auc_noagg=$(grep "macro_auc_noagg_test" "${LOG_FILE}" | tail -1 | grep -oP '\d+\.\d+' || echo "NA")

          echo "${EVAL_MODE},${MODEL},${DATASET},${k},${seed},${auc_agg},${auc_noagg}" >> "${RESULTS_FILE}"

        done
      done
    done
  done
done

log_message "Results saved to: ${RESULTS_FILE}"

log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_message "DONE"
log_message "  Results: ${BASE_DIR}/ktest_12lead_models/"
log_message "  Summary: ${RESULTS_FILE}"
log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
