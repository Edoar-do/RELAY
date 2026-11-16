#!/bin/bash
#SBATCH --job-name=12_lead_benchmark
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

MODELS=(ecg_founder ecg_jepa_multiblock st_mem merl_resnet ecgfm_ked hubert_ecg_base cpc relay)

DATASETS=(ptbxl_all ptbxl_sub ptbxl_super chapman ningbo sph cpsc2018 cpsc_extra echonext georgia code15_diag zzu_pecg hefei ptb)

EVAL_MODES=(finetuning_linear)

TEST_SEED=42

LEARNING_RATE=0.001
BATCH_SIZE=64
EPOCHS=100
TRAIN_SEED=42  # Fixed seed for training

#=============================================================================
# HELPER FUNCTIONS
#=============================================================================

get_output_dir() {
    local eval_mode=$1
    echo "${BASE_DIR}/12_lead_benchmark/${eval_mode}"
}

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
        "samitrop")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/samitrop"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/samitrop"
                "--fs-data 400"
                "--finetune-dataset samitrop"
            )
            ;;
        "mimic")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic"
            )
            ;;
        "mimic_diagnostic")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_diagnostic"
            )
            ;;
        "mimic_deterioration")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_deterioration"
            )
            ;;
        "mimic_sex")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_sex"
            )
            ;;
        "mimic_metadata")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_metadata"
            )
            ;;
        "mimic_cardiac")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_cardiac"
            )
            ;;
        "mimic_noncardiac")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_noncardiac"
            )
            ;;
        "mimic_age")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_age"
            )
            ;;
        "mimic_biometrics")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_biometrics"
            )
            ;;
        "mimic_ecg_features")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_ecg_features"
            )
            ;;
        "mimic_lab_values")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_lab_values"
            )
            ;;
        "mimic_vital_signs")
            sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/mimic"|g' ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
            args+=(
                "--data ${DATASET_DIR}/mimic"
                "--fs-data 500"
                "--finetune-dataset mimic_vital_signs"
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
                "--pretrained ${CHECKPOINTS_DIR}/ecg_jepa/multiblock_epoch100.pth"
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
        # NOTE: --input-size and --fs-model must match your pretraining setup.
        #       --input-size * --fs-model = T (timesteps fed to HuBERT CNN),
        #       which determines S = config.nodes_per_lead in config.json.
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

#=============================================================================
# MAIN LOOP: TRAINING
#=============================================================================

log_message "🚀 Starting 12-Lead Benchmark Pipeline"
log_message "Models: ${MODELS[*]}"
log_message "Datasets: ${DATASETS[*]}"
log_message "Eval Modes: ${EVAL_MODES[*]}"

TOTAL_CONFIGS=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#EVAL_MODES[@]}))
CURRENT_CONFIG=0

for EVAL_MODE in "${EVAL_MODES[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      CURRENT_CONFIG=$((CURRENT_CONFIG + 1))

      log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      log_message "📊 CONFIG ${CURRENT_CONFIG}/${TOTAL_CONFIGS}"
      log_message "   Model: ${MODEL}"
      log_message "   Dataset: ${DATASET}"
      log_message "   Eval Mode: ${EVAL_MODE}"
      log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

      # Setup directories
      BASE_OUTPUT_DIR=$(get_output_dir $EVAL_MODE)
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/outputs/${MODEL}_${DATASET}"
      PREDICTIONS_DIR="${BASE_OUTPUT_DIR}/predictions/${MODEL}_${DATASET}"
      LOGS_DIR="${BASE_OUTPUT_DIR}/logs/${MODEL}_${DATASET}"

      mkdir -p "${OUTPUT_DIR}"
      mkdir -p "${PREDICTIONS_DIR}"
      mkdir -p "${LOGS_DIR}"

      TRAIN_LOG="${LOGS_DIR}/train_seed${TRAIN_SEED}.log"

      # Get dataset and model args
      ARGS_DATASET=($(get_dataset_args $DATASET))
      ARGS_MODEL=($(get_model_args $MODEL))

      #=========================================================================
      # PHASE 1: TRAINING
      #=========================================================================

      log_message "🏋️  TRAINING: ${MODEL} on ${DATASET} (${EVAL_MODE})"

      # Check if already trained — pick the newest version_X checkpoint.
      CHECKPOINT_PATH=$(ls -t "${OUTPUT_DIR}"/version_*/best_model.ckpt 2>/dev/null | head -n 1)
      if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH=$(ls -t "${OUTPUT_DIR}"/version_*/last.ckpt 2>/dev/null | head -n 1)
      fi

      if [ -n "$CHECKPOINT_PATH" ] && [ -f "$CHECKPOINT_PATH" ]; then
        log_message "⏭️  Checkpoint exists, skipping training: ${CHECKPOINT_PATH}"
      else
        log_message "🔄 Training from scratch..."

        # CPC's pretrained loader reads ecg_ptbxl.yaml at init time; reset it to the
        # real PTB-XL path so the ptbxl label lookup doesn't receive a foreign dataset.
        if [ "${MODEL}" == "cpc" ]; then
          sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb-xl/records500"|g' \
            ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
        fi

        # ! 12-lead: no --random_lead_sampling, no --k — train on all 12 leads
        python3.10 ${BASE_DIR}/code/main_lite.py \
          ${ARGS_DATASET[@]} \
          ${ARGS_MODEL[@]} \
          --epochs ${EPOCHS} \
          --modality ecg \
          --lr ${LEARNING_RATE} \
          --batch-size ${BATCH_SIZE} \
          --finetune \
          --eval-mode ${EVAL_MODE} \
          --output-path "${OUTPUT_DIR}" \
          --prediction-path "${PREDICTIONS_DIR}/train" \
          --seed ${TRAIN_SEED} \
          --train_only \
          2>&1 | tee "${TRAIN_LOG}"

        CHECKPOINT_PATH=$(find "${OUTPUT_DIR}" -name "best_model.ckpt" -o -name "last.ckpt" | head -n 1)

        if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
          log_message "❌ ERROR: Training failed or checkpoint not found!"
          continue
        fi

        log_message "✅ Training complete: ${CHECKPOINT_PATH}"
      fi

      #=========================================================================
      # PHASE 2: TEST-TIME EVALUATION (12 leads — deterministic, single seed)
      #=========================================================================

      log_message "🧪 TEST-TIME EVALUATION: 12 leads, seed=${TEST_SEED}"

      TEST_OUTPUT_DIR="${BASE_OUTPUT_DIR}/test_time/${MODEL}_${DATASET}/k12_seed${TEST_SEED}"
      TEST_PRED_DIR="${BASE_OUTPUT_DIR}/test_time/${MODEL}_${DATASET}/predictions/k12_seed${TEST_SEED}"
      TEST_LOG="${LOGS_DIR}/test_k12_seed${TEST_SEED}.log"

      mkdir -p "${TEST_OUTPUT_DIR}"
      mkdir -p "${TEST_PRED_DIR}"

      # Skip if predictions already exist
      if [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/agg/test_0_epoch_0_agg.npz" ] && \
         [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/noagg/test_0_epoch_0_noagg.npz" ]; then
        log_message "⏭️  Results exist for k=12, seed=${TEST_SEED}, skipping"
      else
        if [ "${MODEL}" == "cpc" ]; then
          sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb-xl/records500"|g' \
            ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
        fi

        # ! 12-lead: no --random_lead_sampling — evaluate on all 12 leads deterministically
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
          --seed ${TEST_SEED} \
          --eval-only "${CHECKPOINT_PATH}" \
          2>&1 | tee "${TEST_LOG}"

        if [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/agg/test_0_epoch_0_agg.npz" ] && \
           [ -f "${TEST_PRED_DIR}/${DATASET}_version_0/noagg/test_0_epoch_0_noagg.npz" ]; then
          log_message "✅ Done k=12, seed=${TEST_SEED}"
        else
          log_message "❌ Test finished but expected prediction files are missing for k=12, seed=${TEST_SEED}"
          log_message "   Check ${TEST_LOG} for errors"
        fi
      fi

      log_message "✅ Completed test-time evaluation for ${MODEL} on ${DATASET} (${EVAL_MODE})"

    done
  done
done

#=============================================================================
# AGGREGATE RESULTS
#=============================================================================

log_message "📈 Aggregating results..."

RESULTS_FILE="${BASE_DIR}/12_lead_benchmark/summary_results.csv"

# Write header only if the file does not already exist
if [ ! -f "${RESULTS_FILE}" ]; then
  echo "eval_mode,model,dataset,k,seed,macro_auc_agg,macro_auc_noagg" > ${RESULTS_FILE}
fi

for EVAL_MODE in "${EVAL_MODES[@]}"; do
  BASE_OUTPUT_DIR=$(get_output_dir $EVAL_MODE)

  for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      LOGS_DIR="${BASE_OUTPUT_DIR}/logs/${MODEL}_${DATASET}"

      if [ ! -d "$LOGS_DIR" ]; then
        continue
      fi

      LOG_FILE="${LOGS_DIR}/test_k12_seed${TEST_SEED}.log"

      if [ -f "$LOG_FILE" ]; then
        auc_agg=$(grep "macro_auc_agg_test" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+' || echo "NA")
        auc_noagg=$(grep "macro_auc_noagg_test" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+' || echo "NA")

        echo "${EVAL_MODE},${MODEL},${DATASET},12,${TEST_SEED},${auc_agg},${auc_noagg}" >> ${RESULTS_FILE}
      fi

    done
  done
done

log_message "✅ Results saved to: ${RESULTS_FILE}"

#=============================================================================
# SUMMARY
#=============================================================================

log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_message "🎉 PIPELINE COMPLETE!"
log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_message "📁 Results directory: ${BASE_DIR}/12_lead_benchmark/"
log_message "📊 Summary CSV: ${RESULTS_FILE}"
log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
