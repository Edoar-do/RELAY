#!/bin/bash
#SBATCH --job-name=relay_mimic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out
#=============================================================================
# RELAY finetuning on MIMIC-IV-ECG ("mimic" = all tasks) + test-time evaluation.
#
#   MODE=12lead  train on all 12 leads (Table 1 "ceiling"), same recipe as the
#                baselines in mimic_12_lead.sh (lr 1e-3, bs 64, 100 epochs,
#                finetuning_linear, seed 42), then test exactly like the 12-lead
#                baselines: k12, k1, k2 (5 seeds), lead I, lead II, no bootstrap,
#                into the SAME folder as the baselines so RELAY is a row of the
#                same table; then (BOOTSTRAP_K12=1, default) a k12 run with
#                bootstrap CIs for Table 1, like the HuBERT-base re-run.
#                ckpt  -> 12_lead_benchmark/finetuning_linear/outputs/relay_mimic/
#                tests -> mimic_fewer_leads_12lead_models/   (+ mimic_12lead_bootstrap_fixed/)
#   MODE=rls     train with random lead sampling (k ~ U{1..12} per sample, early
#                stopping as in sft_random_lead_sampling.sh), then test k12, k1,
#                k2 (5 seeds), lead I, lead II without bootstrap, like the
#                baselines in mimic_fewer_leads_12lead_models.sh.
#                ckpt  -> rls_benchmark/finetuning_linear/outputs/relay_mimic/
#                tests -> mimic_fewer_leads_relay_rls/
#
# INVALID ECGs
#   RELAY drops records with no usable signal in some 5 s window (all-zero, or
#   signal stopping early) from all splits — see compute_ecg_valid_mask in
#   code/main_lite_ecg.py (mask cached as mimic/relay_valid_mask_v2_w2500.npy;
#   153 + 7 of 182,076 benchmark records, 7 of them in the test fold). The
#   previous RELAY run crashed because the old mask only read memmap_0.npy.
#   Score every baseline on the same test records with
#     python build_mimic_lead_table.py --root <...> --valid-mask mimic/relay_valid_mask_v2_w2500.npy
#
# TRAINING IS NEVER REDONE once it has finished: a .train_done marker is written
# next to the checkpoint. An interrupted 12lead run resumes from last.ckpt; an
# interrupted rls run restarts from scratch, because main_lite.py disables early
# stopping when resuming, which would change the RLS protocol.
#
# LAUNCH:  MODE=12lead nohup bash relay_mimic_finetune.sh > relay_mimic_12lead.out 2>&1 &
#          MODE=rls    nohup bash relay_mimic_finetune.sh > relay_mimic_rls.out    2>&1 &
#=============================================================================

set -e
set -u

MODE=${MODE:?set MODE=12lead or MODE=rls}

if [ -z "${PYTHON_BIN:-}" ]; then
    for _cand in python python3.10 python3 python; do
        if command -v "${_cand}" >/dev/null 2>&1; then PYTHON_BIN="${_cand}"; break; fi
    done
fi
if [ -z "${PYTHON_BIN:-}" ] || ! "${PYTHON_BIN}" -c "import torch, lightning, torch_geometric" >/dev/null 2>&1; then
    echo "ERROR: no python with torch + lightning + torch_geometric. Run 'bash setup_bench_env.sh'."
    exit 1
fi
echo "Using interpreter: $(command -v "${PYTHON_BIN}")"

#=============================================================================
# CONFIGURATION
#=============================================================================
BASE_DIR="${BASE_DIR:-$(pwd)}"   # repo root: launch the script from there
CHECKPOINTS_DIR="${BASE_DIR}/pretrained_models"
EVAL_MODE=finetuning_linear
LEARNING_RATE=0.001
BATCH_SIZE=64
EPOCHS=100
TRAIN_SEED=42

case ${MODE} in
    12lead)
        CKPT_ROOT="${BASE_DIR}/12_lead_benchmark"
        RLS_ARGS=()
        TEST_ENV=(BENCH_ROOT="${BASE_DIR}/mimic_fewer_leads_12lead_models")
        BOOT_ENV=(BENCH_ROOT="${BASE_DIR}/mimic_12lead_bootstrap_fixed" RUN_SETTINGS=k12 BOOTSTRAP=1)
        ;;
    rls)
        CKPT_ROOT="${BASE_DIR}/rls_benchmark"
        RLS_ARGS=(--random_lead_sampling)
        TEST_ENV=(BENCH_ROOT="${BASE_DIR}/mimic_fewer_leads_relay_rls")
        BOOT_ENV=()
        ;;
    *)
        echo "Unknown MODE=${MODE} (12lead | rls)"; exit 1 ;;
esac

OUTPUT_DIR="${CKPT_ROOT}/${EVAL_MODE}/outputs/relay_mimic"
PREDICTIONS_DIR="${CKPT_ROOT}/${EVAL_MODE}/predictions/relay_mimic"
LOGS_DIR="${CKPT_ROOT}/${EVAL_MODE}/logs/relay_mimic"
mkdir -p "${OUTPUT_DIR}" "${PREDICTIONS_DIR}" "${LOGS_DIR}"

log_message() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

#=============================================================================
# PHASE 1: TRAINING (skipped if a finished run exists)
#=============================================================================
DONE_DIR=$(ls -td "${OUTPUT_DIR}"/version_*/ 2>/dev/null | while read d; do
    [ -f "${d}.train_done" ] && [ -f "${d}best_model.ckpt" ] && { echo "${d}"; break; }; done || true)

if [ -n "${DONE_DIR}" ]; then
    log_message "Training already finished, skipping: ${DONE_DIR}best_model.ckpt"
else
    RESUME_ARGS=()
    LAST=$(ls -t "${OUTPUT_DIR}"/version_*/last.ckpt 2>/dev/null | head -n 1 || true)
    if [ -n "${LAST}" ] && [ "${MODE}" == "12lead" ]; then
        log_message "Resuming interrupted 12-lead run from ${LAST}"
        RESUME_ARGS=(--resume "${LAST}")
    elif [ -n "${LAST}" ]; then
        log_message "Interrupted RLS run found (${LAST}); restarting from scratch (see header)"
    fi

    TRAIN_LOG="${LOGS_DIR}/train_seed${TRAIN_SEED}_$(date +%Y%m%d_%H%M%S).log"
    log_message "Training RELAY on MIMIC (${MODE}) -> ${OUTPUT_DIR}"
    log_message "  log: ${TRAIN_LOG}"

    # Plain redirect, not `| tee` (uutils tee 0.10.0 panics; see mimic_fewer_leads_12lead_models.sh).
    set +e
    ${PYTHON_BIN} -u ${BASE_DIR}/code/main_lite.py \
      --data ${BASE_DIR}/mimic --fs-data 500 --finetune-dataset mimic \
      --architecture relay --input-size 5 --fs-model 100 --input-channels 12 \
      --pretrained ${CHECKPOINTS_DIR}/relay/pretrained.pt \
      --relay-ecg-code-path ${BASE_DIR}/relay_code \
      --relay-config-path ${CHECKPOINTS_DIR}/relay/config.json \
      --epochs ${EPOCHS} --modality ecg --lr ${LEARNING_RATE} --batch-size ${BATCH_SIZE} \
      --finetune --eval-mode ${EVAL_MODE} \
      --output-path "${OUTPUT_DIR}" --prediction-path "${PREDICTIONS_DIR}/train" \
      ${RLS_ARGS[@]+"${RLS_ARGS[@]}"} ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
      --seed ${TRAIN_SEED} --train_only \
      > "${TRAIN_LOG}" 2>&1
    rc=$?
    set -e

    NEWEST=$(ls -td "${OUTPUT_DIR}"/version_*/ 2>/dev/null | head -n 1)
    if [ ${rc} -ne 0 ] || [ ! -f "${NEWEST}best_model.ckpt" ]; then
        log_message "ERROR: training failed (exit ${rc}) — see ${TRAIN_LOG}"
        exit 1
    fi
    date > "${NEWEST}.train_done"
    log_message "Training finished: ${NEWEST}best_model.ckpt"
fi

#=============================================================================
# PHASE 2: TEST-TIME EVALUATION (reuses the reduced-lead inference script;
# it picks the newest version_*/best_model.ckpt under ${OUTPUT_DIR})
#=============================================================================
log_message "Testing RELAY (${MODE}): ${TEST_ENV[*]}"
env "${TEST_ENV[@]}" CKPT_ROOT="${CKPT_ROOT}" RUN_MODELS=relay PYTHON_BIN="${PYTHON_BIN}" \
    bash "${BASE_DIR}/mimic_fewer_leads_12lead_models.sh"

if [ ${#BOOT_ENV[@]} -gt 0 ] && [ "${BOOTSTRAP_K12:-1}" == "1" ]; then
    log_message "Testing RELAY (${MODE}) k12 with bootstrap CIs: ${BOOT_ENV[*]}"
    env "${BOOT_ENV[@]}" CKPT_ROOT="${CKPT_ROOT}" RUN_MODELS=relay PYTHON_BIN="${PYTHON_BIN}" \
        bash "${BASE_DIR}/mimic_fewer_leads_12lead_models.sh"
fi

log_message "DONE (${MODE})"
