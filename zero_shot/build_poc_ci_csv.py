"""One CSV of zero-shot AUROC with 95% bootstrap CIs on CinC 2017 and Icentia11k, all models, per label and macro.

Same predictions, heads, checks and bootstrap as eval_poc_zero_shot.py (1000 resamples, seed 0, shared by every model,
slot and head of a dataset; record-level, i.e. windows for Icentia11k, as the pipeline's --bootstrap). MACRO = mean of the dataset's
head AUROCs, averaged inside each resample. Rows:
    slot = I, II, ...      one lead slot (CinC 2017: lead I only)
    slot = MEAN            Icentia11k: mean over the model's applicable slots inside each resample (the POC table's
                           value; ECG-JEPA: its 8 slots)
applicable = False: ECG-JEPA in III, aVR, aVL, aVF, where the lead never reaches the encoder (kept, not in MEAN).
st_mem = zero-padded ST-MEM, st_mem_native = ST-MEM on the present lead only (the paper's ST-MEM).

Usage:  python zero_shot/build_poc_ci_csv.py     # writes zero_shot/results/poc_auroc_ci.csv
"""
import multiprocessing as mp

import numpy as np
import pandas as pd

from eval_poc_zero_shot import (HEADS, MODEL_ORDER, NOT_APPLICABLE, RESULTS, bootstrap_weights, ci, load_dataset,
                                load_predictions, weighted_auroc)

N_BOOT, SEED = 1000, 0
SLOT_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CTX = {}


def score(task):
    """{head: (1 + B,)} AUROC of one model in one slot; first entry = the full sample."""
    dataset, slot, model = task
    pred_file = RESULTS / dataset / f"slot_{slot}" / model / "predictions" / f"{dataset}_version_0" / "agg" / "test_0_epoch_0_agg.npz"
    itos, y, W = CTX[dataset]["itos"], CTX[dataset]["y"], CTX[dataset]["W"]
    preds = load_predictions(pred_file, y, itos)
    out = {}
    for head in HEADS[dataset]:
        j = itos.index(head)
        out[head] = weighted_auroc(preds[:, j].astype(np.float64), y[:, j].astype(np.float64), W)
    return task, out


def main():
    tasks = []
    for dataset in HEADS:
        df, itos, y = load_dataset(dataset)
        groups = np.arange(len(df))  # records, as the pipeline's --bootstrap
        W = np.vstack([np.ones((1, len(df)), dtype=np.float32), bootstrap_weights(groups, N_BOOT, SEED)])
        CTX[dataset] = {"itos": itos, "y": y, "W": W, "unit": "record", "n": len(df),
                        "n_patients": df.patient_id.nunique() if dataset == "icentia11k" else None}
        slots = [s for s in SLOT_ORDER if (RESULTS / dataset / f"slot_{s}").exists()]
        tasks += [(dataset, s, m) for s in slots for m in MODEL_ORDER]
    res = {}
    with mp.Pool(min(12, len(tasks))) as pool:
        for i, (task, out) in enumerate(pool.imap_unordered(score, tasks)):
            res[task] = out
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(tasks)}", flush=True)

    rows = []

    def add(dataset, slot, model, applicable, n_slots, per_head):
        c, y, itos = CTX[dataset], CTX[dataset]["y"], CTX[dataset]["itos"]
        per_head = dict(per_head)
        per_head["MACRO"] = np.mean([per_head[h] for h in HEADS[dataset]], axis=0)
        for label, v in per_head.items():
            n_pos = int(y[:, itos.index(label)].sum()) if label != "MACRO" else None
            lo, hi = ci(v[1:])
            rows.append({"dataset": dataset, "model": model, "slot": slot, "applicable": applicable, "n_slots": n_slots,
                         "label": label, "auroc": float(v[0]), "ci_lo": lo, "ci_hi": hi, "n_pos": n_pos,
                         "n_neg": c["n"] - n_pos if n_pos is not None else None, "n_records": c["n"],
                         "n_patients": c["n_patients"], "bootstrap_unit": c["unit"]})

    for dataset in HEADS:
        slots = [s for s in SLOT_ORDER if (dataset, s, MODEL_ORDER[0]) in res]
        for model in MODEL_ORDER:
            ok = [s for s in slots if s not in NOT_APPLICABLE.get(model, ())]
            for s in slots:
                add(dataset, s, model, s in ok, 1, res[(dataset, s, model)])
            if len(slots) > 1:
                add(dataset, "MEAN", model, True, len(ok),
                    {h: np.mean([res[(dataset, s, model)][h] for s in ok], axis=0) for h in HEADS[dataset]})
    out = RESULTS / "poc_auroc_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
