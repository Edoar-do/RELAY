"""
ECG-specific Main_Lite_ECG class for the fm-benchmarking project.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from main_lite_base import Main_Lite, multihot_encode
from clinical_ts.models.s4_model import S4Model
from clinical_ts.utils.schedulers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_invsqrt_decay_schedule_with_warmup,
)
from clinical_ts.models.ecg_foundation_models.ecg_founder import Net1D
from clinical_ts.models.fm_ecg import (
    ECGFounderWrapper,
    ECGJEPAWrapper,
    StMemWrapper,
    MerlWrapper,
    EcgFmKEDWrapper,
    HubertEcgWrapper,
    CPCWrapper,
    ECG_JEPA_Scratch,
)
from clinical_ts.models.fm_ecg import RelayWrapper
from clinical_ts.utils.stratify import stratified_subset


class Main_Lite_ECG(Main_Lite):
    def __init__(self, hparams):
        super().__init__(hparams)

        if hparams.finetune_dataset == "ptb":
            num_classes = 18
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "ningbo":
            num_classes = 68
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "cpsc2018":
            num_classes = 9
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "cpsc_extra":
            num_classes = 33
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "georgia":
            num_classes = 50
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "chapman":
            num_classes = 42
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "sph":
            num_classes = 35
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "code15_diag":
            num_classes = 6
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "code_test":
            num_classes = 6
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "ptbxl_super":
            num_classes = 5
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "ptbxl_sub":
            num_classes = 23
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "ptbxl_all":
            num_classes = 71
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "echonext":
            num_classes = 11
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "samitrop":
            num_classes = 1
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "mimic":
            num_classes = 1127
            self.task = "classification_and_regression"
            self.n_regression = 35
        elif hparams.finetune_dataset == "mimic_diagnostic":
            num_classes = 1076
            self.task = "classification_and_regression"
            self.n_regression = 0
        elif hparams.finetune_dataset == "mimic_cardiac":
            num_classes = 158
            self.task = "classification_and_regression"
            self.n_regression = 0
        elif hparams.finetune_dataset == "mimic_noncardiac":
            num_classes = 918
            self.task = "classification_and_regression"
            self.n_regression = 0
        elif hparams.finetune_dataset == "mimic_deterioration":
            num_classes = 15
            self.task = "classification_and_regression"
            self.n_regression = 0
        elif hparams.finetune_dataset == "mimic_sex":
            num_classes = 1
            self.task = "classification_and_regression"
            self.n_regression = 0
        elif hparams.finetune_dataset == "mimic_metadata":
            num_classes = 35
            self.task = "classification_and_regression"
            self.n_regression = 35
        elif hparams.finetune_dataset == "mimic_age":
            num_classes = 1
            self.task = "classification_and_regression"
            self.n_regression = 1
        elif hparams.finetune_dataset == "mimic_biometrics":
            num_classes = 3
            self.task = "classification_and_regression"
            self.n_regression = 3
        elif hparams.finetune_dataset == "mimic_ecg_features":
            num_classes = 7
            self.task = "classification_and_regression"
            self.n_regression = 7
        elif hparams.finetune_dataset == "mimic_lab_values":
            num_classes = 18
            self.task = "classification_and_regression"
            self.n_regression = 18
        elif hparams.finetune_dataset == "mimic_vital_signs":
            num_classes = 6
            self.task = "classification_and_regression"
            self.n_regression = 6
        elif hparams.finetune_dataset == "zzu_pecg":
            num_classes = 58
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "hefei":
            num_classes = 25
            self.task = "classification_multi"
        elif hparams.finetune_dataset in ("cinc2017", "icentia11k"):
            # zero-shot test sets for the PTB-XL All models: labels in PTB-XL All's 71-label order
            num_classes = 71
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "heart_sound_mc":
            num_classes = 5
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "heart_sound_binary":
            num_classes = 1
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "ptbxl_label_efficiency":
            num_classes = 5
            self.task = "classification_multi"
        elif hparams.finetune_dataset == "echonext_label_efficiency":
            num_classes = 6
            self.task = "classification_multi"
        elif self.hparams.finetune_dataset == "ptbxl_all_label_efficiency":
            num_classes = 13
            self.task = "classification_multi"
        elif self.hparams.finetune_dataset == "clinical_data":
            num_classes = 12
            self.task = "classification_multi"

        # elif(hparams.finetune_dataset.startswith("mimic")): #default setting: mimic_ed_all_edfirst_all_2000_5A
        # prefetch dataframe
        #    df_mapped, lbl_itos,  mean, std = load_dataset(Path(list(hparams.data.split(","))[0]))
        #    _, lbl_itos, _, _ = self.setup_dataset(hparams.data, df_mapped, lbl_itos, mean, std)
        #    num_classes = len(lbl_itos)
        #    self.task = "classification_multi"

        ###################################################################################################
        # prepare loss
        ###################################################################################################
        if self.task == "classification_single":
            self.criterion = F.cross_entropy
        elif self.task == "classification_multi":
            self.criterion = F.binary_cross_entropy_with_logits
        elif self.task == "regression":
            self.criterion = F.l1_loss
        elif self.task == "classification_and_regression":
            # Last `n_regression` output dims are regression (L1) targets; the rest
            # are binary-classification (BCE-with-logits). Both branches mask NaN
            # so partially-labeled samples don't poison the gradient.
            n_reg = getattr(self, "n_regression", 35)

            def classification_and_regression_loss(preds, targs):
                if n_reg > 0:
                    cls_preds, reg_preds = preds[:, :-n_reg], preds[:, -n_reg:]
                    cls_targs, reg_targs = targs[:, :-n_reg], targs[:, -n_reg:]
                else:
                    cls_preds, cls_targs = preds, targs
                    reg_preds = reg_targs = preds[:, :0]

                losses = []
                for i in range(cls_targs.size(1)):
                    ti = cls_targs[:, i]
                    pi = cls_preds[:, i]
                    mi = ~torch.isnan(ti)
                    if torch.any(mi):
                        losses.append(F.binary_cross_entropy_with_logits(pi[mi], ti[mi]))
                for i in range(reg_targs.size(1)):
                    ti = reg_targs[:, i]
                    pi = reg_preds[:, i]
                    mi = ~torch.isnan(ti)
                    if torch.any(mi):
                        losses.append(F.l1_loss(pi[mi], ti[mi]))

                if not losses:
                    return torch.zeros((), device=preds.device, requires_grad=True)
                return torch.stack(losses).mean()

            self.criterion = classification_and_regression_loss

        ###################################################################################################
        # prepare model
        ###################################################################################################
        input_size_model = int(self.hparams.input_size * self.hparams.fs_model)
        if hparams.architecture == "ecg_founder":
            self.model = ECGFounderWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
            )
        elif hparams.architecture == "ecg_jepa":
            self.model = ECGJEPAWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
            )
        elif hparams.architecture == "st_mem":
            self.model = StMemWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
                native_leads=getattr(hparams, "st_mem_native_leads", False),
            )
        elif hparams.architecture == "merl":
            self.model = MerlWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                backbone=hparams.merl_backbone,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
            )
        elif hparams.architecture == "ecgfm_ked":
            self.model = EcgFmKEDWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
            )
        elif hparams.architecture == "s4":
            self.model = S4Model(
                d_input=hparams.input_channels,
                d_output=num_classes,
                l_max=input_size_model,
                d_state=self.hparams.s4_n,
                d_model=self.hparams.s4_h,
                n_layers=self.hparams.s4_layers,
                bidirectional=True,
            )
        elif hparams.architecture == "net1d":
            self.model = Net1D(
                in_channels=hparams.input_channels,
                base_filters=64,
                ratio=1,
                filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                kernel_size=16,
                stride=2,
                groups_width=16,
                verbose=False,
                use_bn=False,
                use_do=False,
                n_classes=num_classes,
            )
        elif hparams.architecture == "cpc":
            self.model = CPCWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                config_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
            )
        elif hparams.architecture == "hubert_ecg":
            self.model = HubertEcgWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
                variant=getattr(hparams, "hubert_variant", "base"),
            )
        elif hparams.architecture == "ecg_jepa_scratch":
            self.model = ECG_JEPA_Scratch(num_classes=num_classes)
        elif hparams.architecture == "relay":
            self.model = RelayWrapper(
                num_classes=num_classes,
                num_output_tokens=input_size_model,
                relay_ecg_code_path=hparams.relay_ecg_code_path,
                config_path=hparams.relay_config_path,
                pretrained_path=hparams.pretrained,
                eval_mode=hparams.eval_mode,
                lr=self.lr,
                discriminative_lr_factor=hparams.discriminative_lr_factor,
                num_leads=hparams.input_channels,
                nodes_per_lead=hparams.relay_nodes_per_lead,
                edge_strategy=hparams.relay_edge_strategy,
            )
        else:
            assert False

    def setup_dataset(self, target_folder, df_mapped, lbl_itos, mean, std):
        if self.hparams.finetune_dataset == "ptb":
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "ningbo":
            lbl_itos = np.array(lbl_itos["label_filtered"])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "cpsc2018":
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            lbl_itos = np.array(lbl_itos["label_filtered"])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "cpsc_extra":
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            lbl_itos = np.array(lbl_itos["label_filtered"])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "georgia":
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            lbl_itos = np.array(lbl_itos["label_filtered"])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "chapman":
            lbl_itos = np.array(lbl_itos["label_all_filtered"])
            df_mapped["label"] = df_mapped["label_all_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "sph":
            lbl_itos = np.array(lbl_itos["label_primary_filtered"])
            df_mapped["label"] = df_mapped["label_primary_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "code15_diag":
            df_mapped = df_mapped[df_mapped.strat_fold >= 0].copy()
            df_mapped = df_mapped[df_mapped["data_length"] >= 4000]
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "code_test":
            df_mapped = df_mapped[df_mapped["data_length"] >= 4000]
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "ptbxl_super":
            lbl_itos = np.array(lbl_itos["label_diag_superclass"])
            df_mapped["label"] = df_mapped[
                "label_diag_superclass_filtered_numeric"
            ].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif self.hparams.finetune_dataset == "ptbxl_sub":
            lbl_itos = np.array(lbl_itos["label_diag_subclass"])
            df_mapped["label"] = df_mapped[
                "label_diag_subclass_filtered_numeric"
            ].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif self.hparams.finetune_dataset == "ptbxl_all":
            lbl_itos = np.array(lbl_itos["label_all"])
            df_mapped["label"] = df_mapped["label_all_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "echonext":
            df_mapped = df_mapped[df_mapped["split"].isin(["train", "val", "test"])]
            df_mapped.loc[df_mapped["split"] == "val", "strat_fold"] = 8
            df_mapped.loc[df_mapped["split"] == "test", "strat_fold"] = 9
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "samitrop":
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset.startswith("mimic"):
            # The default loader reads df_memmap.pkl (no labels). Mimic labels
            # live in df_mimic_benchmark.pkl; swap and pick the right vocab/column.
            target_folder = Path(target_folder)
            df_mapped = pd.read_pickle(target_folder / "df_mimic_benchmark.pkl")

            def _load_lbl(name):
                p1 = target_folder / f"{name}.npy"
                p2 = target_folder / "data" / f"{name}.npy"
                src = p1 if p1.exists() else p2
                return np.load(src, allow_pickle=True)

            # Whole-column tasks: (vocab_file_or_None, label_column_name)
            whole_col_tasks = {
                "mimic":               ("lbl_itos_mimic",         "label_all"),
                "mimic_diagnostic":    ("lbl_itos_diagnostic",    "label_diagnostic"),
                "mimic_deterioration": ("lbl_itos_deterioration", "label_deterioration"),
                "mimic_sex":           (None,                     "label_sex"),
                "mimic_metadata":      ("lbl_itos_metadata",      "label_metadata"),
            }
            # Metadata-slice tasks: ranges into label_metadata / lbl_itos_metadata.
            #   age=[0], biometrics=[1:4], ecg_features=[4:11],
            #   lab_values=[11:29], vital_signs=[29:35]
            metadata_slices = {
                "mimic_age":          slice(0, 1),
                "mimic_biometrics":   slice(1, 4),
                "mimic_ecg_features": slice(4, 11),
                "mimic_lab_values":   slice(11, 29),
                "mimic_vital_signs":  slice(29, 35),
            }
            task = self.hparams.finetune_dataset

            if task in whole_col_tasks:
                vocab_name, label_col = whole_col_tasks[task]
                lbl_itos = np.array(["sex"]) if vocab_name is None else _load_lbl(vocab_name)
                df_mapped["label"] = df_mapped[label_col].apply(
                    lambda x: np.array(x, dtype=np.float32)
                )
            elif task in metadata_slices:
                # Slice label_metadata + lbl_itos_metadata by column range.
                sl = metadata_slices[task]
                meta_vocab = _load_lbl("lbl_itos_metadata")
                lbl_itos = meta_vocab[sl]
                df_mapped["label"] = df_mapped["label_metadata"].apply(
                    lambda x: np.array(x, dtype=np.float32)[sl]
                )
            elif task in ("mimic_cardiac", "mimic_noncardiac"):
                # ICD-10 chapter IX = circulatory system = codes starting with I00-I99.
                diag_vocab = _load_lbl("lbl_itos_diagnostic")
                is_cardiac = np.array(
                    [str(c).startswith("I") and len(str(c)) > 1 and str(c)[1].isdigit()
                     for c in diag_vocab]
                )
                mask = is_cardiac if task == "mimic_cardiac" else ~is_cardiac
                lbl_itos = diag_vocab[mask]
                df_mapped["label"] = df_mapped["label_diagnostic"].apply(
                    lambda x: np.array(x, dtype=np.float32)[mask]
                )
            else:
                raise ValueError(f"Unknown mimic finetune-dataset: {task}")
        elif self.hparams.finetune_dataset == "zzu_pecg":
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            lbl_itos = np.array(lbl_itos["aha_description_filtered"])
            df_mapped["label"] = df_mapped["aha_description_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "hefei":
            df_mapped = df_mapped[df_mapped["data_length"] >= 5000]
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset in ("cinc2017", "icentia11k"):
            # single-channel zero-shot sets: lbl_itos.npy holds PTB-XL All's 71 labels, "label" the
            # positive label indices; no dataset statistics (stats of 1 channel are meaningless here)
            lbl_itos = np.array(lbl_itos)
            assert len(lbl_itos) == 71, f"expected PTB-XL All's 71 labels, got {len(lbl_itos)}"
            mean = np.zeros(12, dtype=np.float32)
            std = np.ones(12, dtype=np.float32)
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
            # test-only sets: every record is in the test fold, but the train and val datasets are
            # still built (and never evaluated with --eval-only); one placeholder row each keeps them non-empty
            test_fold = df_mapped.strat_fold.max()
            placeholders = pd.concat([df_mapped.iloc[[0]]] * 2, ignore_index=True)
            placeholders["strat_fold"] = [test_fold - 2, test_fold - 1]
            df_mapped = pd.concat([df_mapped, placeholders], ignore_index=True)
        elif self.hparams.finetune_dataset in ("heart_sound_mc", "heart_sound_binary"):
            # Per-channel stats are meaningless on an 11-of-12-zero dataset and
            # would shift the zero leads into nonzero territory
            # Disable dataset normalization and let each model's own transforms
            # handle per-sample scaling.
            mean = np.zeros(12, dtype=np.float32)
            std = np.ones(12, dtype=np.float32)
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "ptbxl_label_efficiency":
            lbl_itos = np.array(lbl_itos["label_diag_superclass"])

            df_train = df_mapped[df_mapped["strat_fold"] <= 8]
            df_val = df_mapped[df_mapped["strat_fold"] == 9]
            df_test = df_mapped[df_mapped["strat_fold"] == 10]

            df_train_sub = stratified_subset(df_train, self.hparams.label_ratio)
            df_val_sub = stratified_subset(df_val, self.hparams.label_ratio)

            df_mapped = pd.concat([df_train_sub, df_val_sub, df_test])
            df_mapped["label"] = df_mapped[
                "label_diag_superclass_filtered_numeric"
            ].apply(lambda x: multihot_encode(x, len(lbl_itos)))
        elif self.hparams.finetune_dataset == "echonext_label_efficiency":
            label_filtered = np.array(
                [
                    "lvef_lte_45_flag",
                    "lvwt_gte_13_flag",
                    "tricuspid_regurgitation_moderate_or_greater_flag",
                    "rv_systolic_dysfunction_moderate_or_greater_flag",
                    "pasp_gte_45_flag",
                    "tr_max_gte_32_flag",
                ]
            )
            mapping = {
                i: np.where(label_filtered == lbl)[0][0]
                for i, lbl in enumerate(lbl_itos)
                if lbl in label_filtered
            }
            df_mapped["label_filtered_numeric"] = df_mapped["label"].apply(
                lambda x: [mapping[idx] for idx in x if idx in mapping]
            )
            idx_to_label = {i: lbl for i, lbl in enumerate(label_filtered)}
            df_mapped["label_filtered"] = df_mapped["label_filtered_numeric"].apply(
                lambda x: [idx_to_label[idx] for idx in x]
            )

            df_mapped = df_mapped[df_mapped["split"].isin(["train", "val", "test"])]
            df_mapped.loc[df_mapped["split"] == "val", "strat_fold"] = 8
            df_mapped.loc[df_mapped["split"] == "test", "strat_fold"] = 9

            df_train = df_mapped[df_mapped["strat_fold"] < 8]
            df_val = df_mapped[df_mapped["strat_fold"] == 8]
            df_test = df_mapped[df_mapped["strat_fold"] == 9]

            df_train_sub = stratified_subset(
                df=df_train,
                ratio=self.hparams.label_ratio,
                col_label="label_filtered",
                col_age="age_at_ecg",
                value_male="male",
                col_patient="patient_key",
            )
            df_val_sub = stratified_subset(
                df=df_val,
                ratio=self.hparams.label_ratio,
                col_label="label_filtered",
                col_age="age_at_ecg",
                value_male="male",
                col_patient="patient_key",
            )
            lbl_itos = [label for label in label_filtered]
            df_mapped = pd.concat([df_train_sub, df_val_sub, df_test])
            df_mapped["label"] = df_mapped["label_filtered_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "ptbxl_all_label_efficiency":
            labels = [
                "SR",
                "NORM",
                "ABQRS",
                "IMI",
                "ASMI",
                "LVH",
                "NDT",
                "LAFB",
                "AFIB",
                "ISC_",
                "PVC",
                "IRBBB",
                "STD_",
            ]
            df_mapped["label"] = df_mapped["label_all_filtered"].apply(
                lambda x: [item for item in x if item in labels]
            )
            df_mapped["label_numeric"] = df_mapped["label"].apply(
                lambda x: [labels.index(item) for item in x]
            )
            df_mapped = df_mapped[df_mapped["label"].apply(len) != 0]

            df_train = df_mapped[df_mapped["strat_fold"] < 9]
            df_val = df_mapped[df_mapped["strat_fold"] == 9]
            df_test = df_mapped[df_mapped["strat_fold"] == 10]

            df_train_sub = stratified_subset(
                df=df_train, ratio=self.hparams.label_ratio, col_label="label"
            )
            df_val_sub = stratified_subset(
                df=df_val, ratio=self.hparams.label_ratio, col_label="label"
            )

            lbl_itos = np.array(labels)
            df_mapped = pd.concat([df_train_sub, df_val_sub, df_test])
            df_mapped["label"] = df_mapped["label_numeric"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )
        elif self.hparams.finetune_dataset == "clinical_data":
            df_mapped["label"] = df_mapped["label"].apply(
                lambda x: multihot_encode(x, len(lbl_itos))
            )

        # RELAY always drops invalid ECGs (it cannot build a graph from an all-zero
        # input). Other models only with --drop-invalid-ecgs, so that existing
        # baseline checkpoints / protocols are unchanged by default.
        if self.hparams.architecture == "relay" or getattr(self.hparams, "drop_invalid_ecgs", False):
            df_mapped = self._filter_invalid_relay_samples(df_mapped, target_folder)

        return df_mapped, lbl_itos, mean, std

    @staticmethod
    def compute_ecg_valid_mask(target_folder, window):
        """Boolean mask over ALL memmap records: True = the record is usable.

        A record is invalid when some input window of `window` raw samples can be
        all-zero on every lead, i.e. its longest all-lead-zero run (NaN counted
        as 0, as main_lite_base.forward applies nan_to_num) is >= min(window,
        record length). This covers entirely-zero records as well as records
        whose signal stops partway (a random crop / test chunk would be empty).
        NaN alone does not invalidate a record.

        Multi-file memmaps (e.g. MIMIC: memmap_0..4, CODE-15%: 2 files) are
        handled through meta["file_idx"]; `start` is relative to each file. The
        previous implementation read every record from filenames[0], so the
        mask was wrong for every record outside the first file.
        """
        target_folder = Path(target_folder)
        meta = np.load(target_folder / "memmap_meta.npz", allow_pickle=True)
        dtype = np.dtype(str(meta["dtype"]))
        start = meta["start"].astype(np.int64)
        length = meta["length"].astype(np.int64)
        n_total = int(len(start))
        filenames = [str(f) for f in meta["filenames"]]
        file_idx = meta["file_idx"].astype(np.int64) if "file_idx" in meta.files else np.zeros(n_total, np.int64)
        shapes = meta["shape"] if meta["shape"].ndim > 1 else meta["shape"][None]

        max_zero_run = np.zeros(n_total, dtype=np.int64)
        chunk = 2000
        for f, fname in enumerate(filenames):
            mem = np.memmap(str(target_folder / fname), dtype, "r", shape=tuple(shapes[f].astype(np.int64)))
            idx = np.flatnonzero(file_idx == f)
            idx = idx[np.argsort(start[idx])]
            for c in range(0, len(idx), chunk):
                ii = idx[c:c + chunk]
                s0 = int(start[ii].min())
                block = np.asarray(mem[s0:int((start[ii] + length[ii]).max())])
                for i in ii:
                    x = block[start[i] - s0:start[i] - s0 + length[i]]
                    zero_t = ~(np.nan_to_num(x) != 0).reshape(len(x), -1).any(axis=1)
                    d = np.diff(np.concatenate(([0], zero_t.astype(np.int8), [0])))
                    runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
                    max_zero_run[i] = runs.max() if len(runs) else 0
                del block
            del mem
            print(f"[relay-filter]   scanned {fname}")
        return max_zero_run < np.minimum(int(window), length)

    def _filter_invalid_relay_samples(self, df_mapped, target_folder):
        """Drop samples with no usable signal in some input window (see
        compute_ecg_valid_mask). RELAY cannot build a graph from an all-zero
        input (RelayWrapper asserts on it). The mask is computed once per
        (dataset, window length), cached next to the memmap, and df_mapped is
        pruned before the TimeSeriesDataset is built so indexing stays consistent.
        """
        target_folder = Path(target_folder)
        meta_path = target_folder / "memmap_meta.npz"
        if not meta_path.exists():
            return df_mapped

        n_total = int(len(np.load(meta_path, allow_pickle=True)["start"]))
        window = int(round(self.hparams.input_size * self.hparams.fs_data))

        # v2: multi-file aware + window criterion. The old relay_valid_mask.npy is
        # wrong for multi-file memmaps and is deliberately no longer read.
        cache_path = target_folder / f"relay_valid_mask_v2_w{window}.npy"
        valid_mask = None
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (n_total,) and cached.dtype == np.bool_:
                valid_mask = cached

        if valid_mask is None:
            print(f"[relay-filter] scanning {n_total} samples in {target_folder} (window={window} samples)...")
            valid_mask = self.compute_ecg_valid_mask(target_folder, window)
            try:
                np.save(cache_path, valid_mask)
            except OSError as e:
                print(f"[relay-filter] could not cache mask to {cache_path}: {e}")

        n_invalid = int((~valid_mask).sum())
        if n_invalid == 0:
            return df_mapped

        bad_rows = np.flatnonzero(~valid_mask).astype(np.int64)
        n_before = len(df_mapped)
        df_mapped = df_mapped[~df_mapped["data"].isin(bad_rows)].copy()
        print(
            f"[relay-filter] {target_folder}: dropped {n_before - len(df_mapped)} of {n_before} rows "
            f"({n_invalid} invalid memmap samples detected)"
        )
        return df_mapped
        # elif(self.hparams.finetune_dataset.startswith("mimic")):
        # now request the actual labels (will load records_w_diag_icd10.csv to fetch the full labels)
        #    df_mapped, lbl_itos = prepare_mimic_ecg(self.hparams.finetune_dataset, Path(target_folder), df_mapped=df_mapped)

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            opt = torch.optim.SGD
        elif self.hparams.optimizer == "adam":
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")

        if hasattr(self.model, "get_params"):
            params = self.model.get_params()
        else:
            params = self.parameters()
        optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_schedule == "const":
            scheduler = get_constant_schedule(optimizer)
        elif self.hparams.lr_schedule == "warmup-const":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, self.hparams.lr_num_warmup_steps
            )
        elif self.hparams.lr_schedule == "warmup-cos":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                self.hparams.lr_num_warmup_steps,
                self.hparams.epochs * len(self.train_dataloader()),
                num_cycles=0.5,
            )
        elif self.hparams.lr_schedule == "warmup-cos-restart":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                self.hparams.lr_num_warmup_steps,
                self.hparams.epochs * len(self.train_dataloader()),
                num_cycles=self.hparams.epochs - 1,
            )
        elif self.hparams.lr_schedule == "warmup-poly":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                self.hparams.lr_num_warmup_steps,
                self.hparams.epochs * len(self.train_dataloader()),
                num_cycles=self.hparams.epochs - 1,
            )
        elif self.hparams.lr_schedule == "warmup-invsqrt":
            scheduler = get_invsqrt_decay_schedule_with_warmup(
                optimizer, self.hparams.lr_num_warmup_steps
            )
        elif self.hparams.lr_schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 0, self.hparams.epochs * len(self.train_dataloader())
            )
        else:
            assert False
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            ],
        )
