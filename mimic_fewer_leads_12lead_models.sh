#!/bin/bash
#SBATCH --job-name=mimic_fewer_leads
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#=============================================================================
# MIMIC-IV-ECG: TEST-ONLY inference of the 12-lead-finetuned prior FMs on
# reduced lead sets.
#
# WHAT THIS DOES
#   For every prior 12-lead FM finetuned on MIMIC-IV-ECG ("mimic" = all tasks:
#   ICD-10 diagnoses, deterioration, mortality, ICU, sex + MAE regression
#   targets) by mimic_12_lead.sh / mimic_12_lead_hubert_large.sh:
#     - loads the existing 12-lead checkpoint (NO training)
#     - runs --epochs 0 --eval-only <ckpt> on the fold-19 test set under:
#         k12     all 12 leads                        (deterministic, 1 run)
#         k1      1 random lead,  zero-masked others  (5 seeds)
#         k2      2 random leads, zero-masked others  (5 seeds)
#         leadI   lead I only                         (deterministic, 1 run)
#         leadII  lead II only                        (deterministic, 1 run)
#   k12 is re-run (rather than reusing 12_lead_benchmark/.../test_k12_seed42)
#   so that all five settings come from the same code + environment; it should
#   reproduce the existing 12-lead numbers and doubles as a sanity check.
#
# CHECKPOINTS
#   12_lead_benchmark/finetuning_linear/outputs/<model>_mimic/version_*/best_model.ckpt
#   except hubert_ecg_large, whose lr=1e-3 run collapsed (val AUROC ~0.5): it
#   uses the lr=1e-5 retry in 12_lead_benchmark/finetuning_linear_hubert_large_lr1e-5/
#   (same choice as gather_hubert_results.py / mimic_subgroup_eval.py).
#
# RELAY is not included: relay_mimic_finetune.sh (MODE=12lead) trains and tests it
# with the same settings, into the same folder.
#
# OUTPUT
#   mimic_fewer_leads_12lead_models/<eval_mode>/
#     logs/<model>_mimic/test_<setting>_seed<seed>.log
#     test_time/<model>_mimic/predictions/<setting>_seed<seed>/mimic_version_0/{agg,noagg}/*.npz
#     summary_long.csv / summary_table.csv / summary_table.md   (build_mimic_lead_table.py)
#
# LAUNCH:  bash mimic_fewer_leads_12lead_models.sh      (or sbatch)
#=============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Clean PyKeOps cache
rm -rf ~/.cache/keops* ~/.cache/pykeops* /tmp/keops*

#=============================================================================
# PYTHON INTERPRETER
# Prefers $PYTHON_BIN, then the first
# of python3.10 / python3 / python. Override with PYTHON_BIN=/path/to/python.
#=============================================================================
if [ -z "${PYTHON_BIN:-}" ]; then
    for _cand in python python3.10 python3 python; do
        if command -v "${_cand}" >/dev/null 2>&1; then
            PYTHON_BIN="${_cand}"
            break
        fi
    done
fi

if [ -z "${PYTHON_BIN:-}" ] || ! "${PYTHON_BIN}" -c "import torch, lightning" >/dev/null 2>&1; then
    echo "ERROR: no python interpreter with torch + lightning found."
    echo "       Run 'bash setup_bench_env.sh' or set PYTHON_BIN=/path/to/python."
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
# Override with CKPT_ROOT=... e.g. ${BASE_DIR}/rls_benchmark for RELAY finetuned
# with random lead sampling (relay_mimic_finetune.sh MODE=rls).
CKPT_ROOT="${CKPT_ROOT:-${BASE_DIR}/12_lead_benchmark}"

# Where this script writes logs / predictions / summary
# Override with BENCH_ROOT=... (e.g. for the bootstrapped re-run, see below)
BENCH_ROOT="${BENCH_ROOT:-${BASE_DIR}/mimic_fewer_leads_12lead_models}"

# hubert_ecg_small / hubert_ecg_large are excluded from this experiment
# (still supported by get_model_args if passed through RUN_MODELS).
MODELS=(ecg_founder ecg_jepa_multiblock ecg_jepa_8lead st_mem merl_resnet ecgfm_ked cpc hubert_ecg_base)

# ecg_jepa_8lead = the ecg_jepa_multiblock checkpoint, but the k random leads
# are drawn only from the 8 leads ECG-JEPA reads (as in
# test_ecg_jepa_8lead_12lead_models.sh). Only random settings with k < 12 are
# run for it: k12 / lead I / lead II are identical to ecg_jepa_multiblock.
JEPA_LEAD_POOL="I,II,V1,V2,V3,V4,V5,V6"

# Multi-GPU / multi-container: give each container a disjoint subset, e.g.
#   RUN_MODELS="cpc merl_resnet" bash mimic_fewer_leads_12lead_models.sh
# Never give the same model to two containers (invalid results get deleted).
if [ -n "${RUN_MODELS:-}" ]; then
    read -r -a MODELS <<< "${RUN_MODELS}"
fi

DATASET=mimic

EVAL_MODES=(finetuning_linear)

# Test settings, see header. Format: k<N> (random N leads) or lead<SPEC>
# (fixed leads, anything --fixed-leads accepts, e.g. leadI, leadII, "leadI,II").
# Override with RUN_SETTINGS="k12 ...".
SETTINGS=(k12 k1 k2 leadI leadII)
if [ -n "${RUN_SETTINGS:-}" ]; then
    read -r -a SETTINGS <<< "${RUN_SETTINGS}"
fi

# Seeds for the random-lead settings (k < 12). Deterministic settings
# (k12, lead*) use only the first one.
TEST_SEEDS=(42 43 44 45 46)

# Bootstrap CIs over ~1100 MIMIC labels make every run much slower; seeds
# already give the variability for k1/k2. Set to 1 to get per-label CIs.
# Override with BOOTSTRAP=1. Needs the regression-name fix in
# code/main_lite_base.py (2026-10-12) for correct per-target MAE names.
BOOTSTRAP=${BOOTSTRAP:-0}

LEARNING_RATE=0.001  # unused at test time, kept for argument parity
BATCH_SIZE=64

#=============================================================================
# HELPER FUNCTIONS
#=============================================================================

# Directory under CKPT_ROOT holding the 12-lead run of a model.
get_ckpt_mode_dir() {
    local eval_mode=$1
    local model=$2
    if [ "${model}" == "hubert_ecg_large" ]; then
        echo "${CKPT_ROOT}/${eval_mode}_hubert_large_lr1e-5"
    else
        echo "${CKPT_ROOT}/${eval_mode}"
    fi
}

# No sed on conf/data/ecg_ptbxl.yaml here (unlike the other scripts): only CPC
# reads that file, and it needs the PTB-XL path. Leaving it alone makes the
# script safe to run concurrently from several containers on the shared FS.
get_dataset_args() {
    local args=(
        "--data ${DATASET_DIR}/mimic"
        "--fs-data 500"
        "--finetune-dataset mimic"
    )
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
        "ecg_jepa_multiblock" | "ecg_jepa_8lead")
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
        "hubert_ecg_small" | "hubert_ecg_base" | "hubert_ecg_large")
            local variant=${model#hubert_ecg_}
            args+=(
                "--architecture hubert_ecg"
                "--input-size 5"
                "--fs-model 100"
                "--input-channels 12"
                "--pretrained ${CHECKPOINTS_DIR}/hubert_ecg/hubert_ecg_${variant}.safetensors"
                "--hubert-variant ${variant}"
            )
            ;;
        "relay")
            # Not in the default MODELS: pass RUN_MODELS=relay (see relay_mimic_finetune.sh).
            # RelayWrapper needs inputs with signal: the test set is pruned by the
            # v2 invalid-ECG mask (main_lite_ecg.py), so score the baselines on the
            # same records with build_mimic_lead_table.py --valid-mask.
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

# Extra main_lite.py args for a setting (without --seed).
get_setting_args() {
    local setting=$1
    case $setting in
        k12)    echo "" ;;
        k*)     echo "--random_lead_sampling --k ${setting#k}" ;;
        lead*)  echo "--fixed-leads ${setting#lead}" ;;
        *)      echo "Unknown setting: $setting" >&2; exit 1 ;;
    esac
}

# Seeds to run for a setting: all for random k<12, first only otherwise.
get_setting_seeds() {
    local setting=$1
    if [[ "${setting}" =~ ^k([0-9]+)$ ]] && [ "${BASH_REMATCH[1]}" -lt 12 ]; then
        echo "${TEST_SEEDS[@]}"
    else
        echo "${TEST_SEEDS[0]}"
    fi
}

# Filesystem-safe tag ("leadI,II" -> "leadI_II")
setting_tag() {
    echo "$1" | tr -d ' ' | tr ',' '_'
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# A test is valid when the log holds the metrics table (MIMIC has no macro_auc
# row; "sex" is a classification target present in every run) and both
# prediction files exist. Invalid / partial results are deleted.
is_test_valid() {
    local log_file=$1
    local pred_dir=$2
    if [ -f "${log_file}" ] && grep -aq "sex_agg_test0" "${log_file}"; then
        if [ -f "${pred_dir}/${DATASET}_version_0/agg/test_0_epoch_0_agg.npz" ] && \
           [ -f "${pred_dir}/${DATASET}_version_0/noagg/test_0_epoch_0_noagg.npz" ]; then
            return 0
        fi
    fi
    [ -f "${log_file}" ] && rm -f "${log_file}"
    [ -d "${pred_dir}/${DATASET}_version_0" ] && rm -rf "${pred_dir}/${DATASET}_version_0"
    return 1
}

#=============================================================================
# MAIN LOOP: TEST-ONLY INFERENCE
#=============================================================================

BOOTSTRAP_ARGS=""
[ "${BOOTSTRAP}" == "1" ] || BOOTSTRAP_ARGS="--no-bootstrap"

log_message "Starting MIMIC-IV-ECG reduced-lead test-only inference (12-lead finetuned ckpts)"
log_message "Models:     ${MODELS[*]}"
log_message "Settings:   ${SETTINGS[*]}"
log_message "Seeds:      ${TEST_SEEDS[*]}  (deterministic settings use only the first)"
log_message "Eval modes: ${EVAL_MODES[*]}"

for EVAL_MODE in "${EVAL_MODES[@]}"; do

  OUTPUT_BASE="${BENCH_ROOT}/${EVAL_MODE}"

  for MODEL in "${MODELS[@]}"; do

    CKPT_MODEL=${MODEL}
    [ "${MODEL}" == "ecg_jepa_8lead" ] && CKPT_MODEL=ecg_jepa_multiblock
    CKPT_DIR="$(get_ckpt_mode_dir ${EVAL_MODE} ${CKPT_MODEL})/outputs/${CKPT_MODEL}_${DATASET}"
    CHECKPOINT_PATH=$(ls -t "${CKPT_DIR}"/version_*/best_model.ckpt 2>/dev/null | head -n 1)
    if [ -z "${CHECKPOINT_PATH}" ] || [ ! -f "${CHECKPOINT_PATH}" ]; then
      CHECKPOINT_PATH=$(ls -t "${CKPT_DIR}"/version_*/last.ckpt 2>/dev/null | head -n 1)
    fi
    if [ -z "${CHECKPOINT_PATH}" ] || [ ! -f "${CHECKPOINT_PATH}" ]; then
      log_message "SKIP (no checkpoint): ${CKPT_DIR}"
      continue
    fi

    log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_message "Testing ${EVAL_MODE} / ${MODEL} / ${DATASET}"
    log_message "Checkpoint: ${CHECKPOINT_PATH}"

    LOGS_DIR="${OUTPUT_BASE}/logs/${MODEL}_${DATASET}"
    mkdir -p "${LOGS_DIR}"

    ARGS_DATASET=($(get_dataset_args))
    ARGS_MODEL=($(get_model_args $MODEL))

    for SETTING in "${SETTINGS[@]}"; do

      TAG=$(setting_tag "${SETTING}")
      ARGS_SETTING=($(get_setting_args "${SETTING}"))

      if [ "${MODEL}" == "ecg_jepa_8lead" ]; then
        if [[ ! "${SETTING}" =~ ^k([0-9]+)$ ]] || [ "${BASH_REMATCH[1]}" -ge 12 ]; then
          continue
        fi
        ARGS_SETTING+=(--rls-lead-pool "${JEPA_LEAD_POOL}")
      fi

      for seed in $(get_setting_seeds "${SETTING}"); do

        TEST_OUTPUT_DIR="${OUTPUT_BASE}/test_time/${MODEL}_${DATASET}/${TAG}_seed${seed}"
        TEST_PRED_DIR="${OUTPUT_BASE}/test_time/${MODEL}_${DATASET}/predictions/${TAG}_seed${seed}"
        TEST_LOG="${LOGS_DIR}/test_${TAG}_seed${seed}.log"

        mkdir -p "${TEST_OUTPUT_DIR}" "${TEST_PRED_DIR}"

        if is_test_valid "${TEST_LOG}" "${TEST_PRED_DIR}"; then
          log_message "  SKIP (valid): ${SETTING} seed=${seed}"
          continue
        fi

        log_message "  Testing ${SETTING} seed=${seed}..."

        # CPC reads ecg_ptbxl.yaml at init time; reset to PTB-XL path.
        if [ "${MODEL}" == "cpc" ]; then
          sed -i 's|path: ".*"|path: "'${DATASET_DIR}'/ptb-xl/records500"|g' \
            ${BASE_DIR}/code/clinical_ts/models/conf/data/ecg_ptbxl.yaml
        fi

        # -u: unbuffered stdout, so a killed run still leaves a complete log.
        # Plain redirect, NOT `| tee`: this image's tee (uutils coreutils 0.10.0)
        # can panic ("2nd pipe should have enough spare"), which truncated the
        # log and, through set -e, killed the whole script. `|| true`: a single
        # failed run must not abort the loop; the validity check below reports
        # it. Follow a run live with: tail -f <TEST_LOG>
        log_message "    log: ${TEST_LOG}"
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
          ${ARGS_SETTING[@]+"${ARGS_SETTING[@]}"} \
          --seed ${seed} \
          --eval-only "${CHECKPOINT_PATH}" \
          ${BOOTSTRAP_ARGS} \
          > "${TEST_LOG}" 2>&1 || true

        if is_test_valid "${TEST_LOG}" "${TEST_PRED_DIR}"; then
          log_message "  OK ${SETTING} seed=${seed}"
        else
          log_message "  WARNING: ${SETTING} seed=${seed} failed (log removed) — re-run to retry"
        fi

      done
    done

    log_message "Done: ${EVAL_MODE} / ${MODEL} / ${DATASET}"

  done

  #===========================================================================
  # AGGREGATE: per task-group mean AUROC / MAE, mean ± std over seeds
  #===========================================================================
  log_message "Aggregating results for ${EVAL_MODE}..."
  ${PYTHON_BIN} "${BASE_DIR}/build_mimic_lead_table.py" --root "${OUTPUT_BASE}" || \
    log_message "WARNING: build_mimic_lead_table.py failed — logs and predictions are still in ${OUTPUT_BASE}"

done

log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_message "DONE"
log_message "  Results: ${BENCH_ROOT}/"
log_message "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
