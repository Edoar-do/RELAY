# RELAY — An ECG foundation model for zero-shot transfer from hospital to point-of-care devices

This is the official repository of `An ECG foundation model for zero-shot transfer from hospital to point-of-care devices`. It contains:

- **RELAY** (`relay_code/`): a graph-based ECG foundation model. It processes any subset of leads natively, without lead-filling, lead identity or changes to the architecture.
- **A benchmarking pipeline** (`code/`), built on PyTorch Lightning. It evaluates RELAY and seven prior ECG foundation models across public ECG datasets and clinical tasks:
  - ECGFounder
  - ECG-JEPA
  - ST-MEM
  - MERL
  - ECGFM-KED
  - HuBERT-ECG
  - ECG-CPC
- **Scripts to reproduce every table:**
  - 12-lead fine-tuning;
  - reduced-lead tests (random leads, lead I and lead II);
  - MIMIC-IV-ECG;
  - zero-shot transfer to point-of-care recordings;
  - the random-lead-sampling ablation.

```
code/              benchmarking pipeline (main_lite.py) and the wrappers of all foundation models (clinical_ts/models/fm_ecg.py)
relay_code/        RELAY: model, pre-training, k-means pseudo-labels, standalone fine-tuning
*.sh               training and test entry points (one per experiment, see below)
zero_shot/         zero-shot point-of-care transfer: CinC 2017 and Icentia11k
ablations/rls/     pre-training ablation: random lead sampling vs all leads vs zero-padding
pretrained_models/ where the scripts look for the pretrained checkpoints
```

---

## 1. Environment

Everything in this repository runs in one conda environment, which uses Python 3.13, CUDA 12.6 and PyTorch 2.7.1:

```bash
conda env create -f env.yaml
conda activate lightning3
```

Without conda, use the equivalent pip setup:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`ecg_fm_env.yaml` is needed only for the optional ECG-FM baseline, which is not part of the paper's tables.

The S4 layers of ECG-CPC run faster with the CUDA Cauchy kernel: `cd code/extensions/cauchy && python setup.py install`. Without it, an equivalent PyTorch kernel is used.

The pre-training ablation (`ablations/rls/`) pins the versions of the original pre-training run. `bash ablations/rls/setup_env.sh` creates that environment (Python 3.8, torch 2.0.1, PyG 2.6.1).

---

## 2. Datasets

### Downstream benchmark (12-lead)

More on this in the paper: Methods>Ethics and Resource Availability

| Dataset | Source |
|---|---|
| PTB-XL | https://physionet.org/content/ptb-xl/1.0.3/ |
| PTB | https://physionet.org/content/ptbdb/1.0.0/ |
| CPSC2018, CPSC-Extra, Georgia, Ningbo | https://physionet.org/content/challenge-2021/1.0.3/ |
| Chapman | https://figshare.com/articles/dataset/Chapman_ECG_dataset/25558926 |
| SPH | https://figshare.com/articles/figure/SPH/22199548?file=39453271 |
| CODE-15% | https://zenodo.org/records/4916206 |
| EchoNext | https://physionet.org/content/echonext/1.1.0/ |
| ZZU pECG | https://doi.org/10.6084/m9.figshare.27078763 |
| Hefei | https://tianchi.aliyun.com/competition/entrance/231754/information |

Put every dataset in its own folder at the repository root, using the folder names that the scripts expect:
- `ptb-xl/records500`, `ptb`, `cpsc2018`, `cpsc_extra`, `georgia`, `ningbo`;
- `chapman/ECGData`, `sph`, `code15`, `echonext`, `zzu_pecg`, `hefei`.

To use another location, set `DATASET_DIR` in the scripts. Then convert each dataset to the memmap format of the pipeline:

```bash
jupyter notebook preprocess_ecg_dataset.ipynb
```

```
<dataset_dir>/
├── df_memmap.pkl                  # DataFrame: strat_fold, label, patient_id, ...
├── memmap_meta_preprocessed.npz
├── memmap_*_preprocessed.npy
└── lbl_itos.npy
```

### MIMIC-IV-ECG

MIMIC-IV-ECG requires credentialed PhysioNet access. Download the following into `mimic/`:
- the waveforms of [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/), whose `machine_measurements.csv` and `record_list.csv` go into `mimic/data/`;
- the ICD-10 labels of [MIMIC-IV-ECG-ICD](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/), i.e. `records_w_diag_icd10.csv`.

Then download into `mimic/data/`:
- from [MIMIC-IV](https://physionet.org/content/mimiciv/): `omr.csv`, `labevents.csv`, `d_labitems.csv`, `chartevents.csv` and `d_items.csv`;
- from [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/): `vitalsign.csv`;
- the deterioration labels of the MDS-ED benchmark: `mds_ed.csv`.

Convert the waveforms with the MIMIC section of `preprocess_ecg_dataset.ipynb`. Then build the label tables:

```bash
python mimic_preprocessing.py        # -> mimic/df_mimic_benchmark.pkl and the label vocabularies
```

This covers the tasks: ICD-10 diagnoses (cardiac and non-cardiac), deterioration, mortality, ICU admission, sex, and 35 regression targets (age, biometrics, ECG features, laboratory values, vital signs). Fold 19 is the test set.

### Zero-shot point-of-care datasets

Both datasets download from PhysioNet's open-data mirror without credentials:

```bash
cd zero_shot
python download.py cinc2017                     # CinC 2017 (AliveCor single lead), 1.4 GB
python build_cinc2017.py                        # -> cinc2017/  (AF vs Normal, 5,834 records)

python download.py icentia11k_atr --workers 32  # Icentia11k annotations only (~30 GB)
python scan_icentia11k.py --procs 8             # 10-s windows x rhythm / beat content
python select_icentia11k.py                     # -> icentia11k/selected_windows.csv, selected_segments.txt
python download.py icentia11k_dat --records ../icentia11k/selected_segments.txt
python build_icentia11k.py --procs 16           # -> icentia11k/  (SR, AFIB, AFLT, PVC, PAC)
```

---

## 3. Pretrained checkpoints

Download the published checkpoint of each prior model from the link given in the paper's appendix. Place the files in the following subfolders of `pretrained_models/`:

| Folder | Files |
|---|---|
| `ecg_founder/` | `12_lead_ECGFounder.pth`, `1_lead_ECGFounder.pth` |
| `ecg_jepa/`    | `multiblock_epoch100.pth` |
| `st_mem/`      | `st_mem_vit_base_full.pth` |
| `merl/`        | `res18_best_encoder.pth` |
| `ecgfm_ked/`   | `best_valid_all_increase_with_augment_epoch_3.pt` |
| `hubert_ecg/`  | `hubert_ecg_base.safetensors` |
| `cpc/`         | `last_11597276.ckpt` (`config_last_11597276_ckpt.yaml` is included) |
| `relay/`       | `pretrained.pt` (`config.json` is included) |

The RELAY checkpoint will be released soon.

---

## 4. Reproducing the experiments

Launch every script from the repository root, either with `bash <script>` or with `sbatch <script>`; each script has SLURM headers. The configuration block at the top of each script holds:
- `BASE_DIR`, which defaults to the current directory;
- `CHECKPOINTS_DIR` and `DATASET_DIR`;
- the lists of `MODELS`, `DATASETS` and seeds, set to the paper's configuration.

Runs that have already finished are skipped, so a script can be relaunched safely. Every test run saves its per-record predictions (`test_0_epoch_0_{agg,noagg}.npz`), and all tables are computed from those predictions.

All models are fine-tuned with the same recipe:
- AdamW, learning rate 1e-3 for the head and 1e-4 for the encoder, weight decay 1e-3, constant schedule;
- batch size 64, up to 100 epochs, `finetuning_linear` head;
- model selection on the validation fold, training seed 42.

### 4.1 12-lead fine-tuning (the ceiling)

```bash
bash sft_12_lead.sh                  # 8 models x 14 datasets -> 12_lead_benchmark/finetuning_linear/
```

### 4.2 Reduced leads at test time (12-lead checkpoints, no retraining)

```bash
bash rls_test_12lead_models.sh            # k = 1..12 leads drawn at random, seeds 42-46 -> ktest_12lead_models/
bash test_lead_I_II_12lead_models.sh      # lead I only, lead II only               -> lead_I_II_test_12lead_models/
bash test_st_mem_native_12lead_models.sh  # ST-MEM on the present leads only (no zero-padding)
bash test_ecg_jepa_8lead_12lead_models.sh # ECG-JEPA, random leads drawn from the 8 leads it encodes
```

The leads that are not sampled are zero-filled in the 12-lead input:
- The prior models see the zero-filled tensor.
- RELAY detects the zeroed leads and builds a graph containing only the k present leads.
- ST-MEM native does the same: it drops the zeroed leads and encodes only the present ones.

The draws depend only on the seed, so every model is tested on the same lead subsets.

### 4.3 Fine-tuning on one random lead

```bash
bash sft_1_random_lead.sh            # train and test on 1 random lead (5 seeds) -> random_1_benchmark/
```

### 4.4 MIMIC-IV-ECG

```bash
bash mimic_12_lead.sh                       # prior models, 12-lead fine-tuning on all MIMIC tasks
MODE=12lead bash relay_mimic_finetune.sh    # RELAY, then its tests at L=12, L=1, L=2, lead I and lead II
bash mimic_fewer_leads_12lead_models.sh     # prior models: tests at L=12, L=1, L=2 (5 seeds), lead I and lead II
```

### 4.5 Zero-shot transfer to point-of-care recordings

The PTB-XL All checkpoints from 4.1 are applied as they are, with no training. The single recorded lead is placed in one lead slot of a zero-filled 12-lead input:
- CinC 2017 uses the lead I slot.
- Icentia11k uses every slot, because its patch lead is not a standard lead.

```bash
ORACLE=1 bash zero_shot/run_poc_zero_shot.sh   # all models, CinC 2017 (slot I) + Icentia11k (12 slots)
python zero_shot/eval_poc_zero_shot.py         # AUROC per head + bootstrap CIs -> zero_shot/results/summary_*.csv
python zero_shot/make_poc_tables.py            # LaTeX tables
python zero_shot/build_poc_ci_csv.py           # macro + per-label AUROC with 95% CIs, all models -> poc_auroc_ci.csv
```

### 4.6 Tables, confidence intervals and operating points

**Metrics inside the pipeline.** Every test run of `code/main_lite.py` reports the macro and per-class AUROC (MAE for the MIMIC regression targets) with 95% confidence intervals from 1,000 record-level bootstrap resamples. Bootstrapping is on by default; `--no-bootstrap` turns it off (same point estimates, faster), and several test scripts pass it because the CIs can be computed afterwards from the saved predictions.

Threshold-based metrics are optional (`code/clinical_ts/utils/eval_utils_threshold.py`) and are bootstrapped in the same way:

| flag | reports |
|---|---|
| `--threshold-metrics` | sensitivity at specificity `--sens-at-spec` (default 0.90 0.95) and specificity at sensitivity `--spec-at-sens` (default 0.80 0.90), per class and macro, with the threshold chosen on the evaluated predictions |
| `--export-thresholds FILE.json` | the per-class thresholds at the `--sens-at-spec` targets (smallest score t with specificity(score > t) above the target) |
| `--fixed-thresholds FILE.json` | sensitivity, specificity, balanced accuracy and PPV (at the sample prevalence) of every matching class at those fixed thresholds |

For example, to fix thresholds on the PTB-XL All validation fold and apply them unchanged to CinC 2017 (the threshold-transfer experiment), with `<model args>` as in `rls_test_12lead_models.sh`:

```bash
python code/main_lite.py <model args> --data ptb-xl/records500 --fs-data 500 --finetune-dataset ptbxl_all \
       --finetune --epochs 0 --eval-only <ptbxl_all best_model.ckpt> --eval-split val \
       --export-thresholds thresholds/<model>.json        # add --fixed-leads I for thresholds chosen on lead I only
python code/main_lite.py <model args> --data cinc2017 --fs-data 300 --finetune-dataset cinc2017 --single-lead-slot I \
       --finetune --epochs 0 --eval-only <ptbxl_all best_model.ckpt> --fixed-thresholds thresholds/<model>.json
```

### 4.7 Pre-training ablation: random lead sampling and zero-padding

Pretraining requires access to all pretraining sources as listed in the paper.
This ablation pre-trains for 60k steps and then runs linear probes on Ribeiro-dev. The pre-training arms are:
- random lead sampling;
- all 12 leads, matched to random lead sampling by token count or by masked-target count;
- random lead sampling with zero-padded leads instead of dropped leads.

It needs the pre-training corpus described in the paper. Set `DATA` to the folder that holds the corpus and its k-means codebook, which `relay_code/cluster.py` creates.

```bash
bash ablations/rls/run_arm.sh rls                     # pre-training + probes at K = 12, 1, 2 (also: norls_tok, norls_pm0217)
bash ablations/rls/run_zeropad_ablation.sh            # zero-padded arm, end to end
python ablations/rls/eval_zeroshot.py                 # 12-lead probes tested on fewer leads, missing leads dropped
python ablations/rls/eval_zeroshot.py --pad           # ... and zero-padded
python ablations/rls/collect_results.py
```

---

## 5. RELAY on its own

`relay_code/` also runs outside the benchmarking pipeline:

```bash
cd relay_code
python cluster.py [args]           # k-means pseudo-labels for masked node modelling
python pretrain.py [args]          # pre-training with masked node modelling
python finetune.py \
    --checkpoint_path ../pretrained_models/relay/pretrained.pt \
    --model_config_path ../pretrained_models/relay/config.json \
    --encoder_name gat \
    --use_benchmarking_format \
    --benchmarking_folder <dataset_dir> \
    --sampling random --how_many_leads 12 \
    --training_steps 50000
```

Each ECG lead becomes a subgraph of 20 nodes: the CNN feature frames of a 5-s window at 100 Hz. Within a lead, the nodes are fully connected. Across leads, nodes at the same time position are connected. Because the graph has no lead-specific parameters and no fixed 12-lead template, any subset of leads gives a valid input.
