"""Scores the zero-shot POC runs (run_poc_zero_shot.sh): per-head AUROC / AUPRC of the PTB-XL All heads
that map to each dataset's labels, with bootstrap CIs and paired comparisons against RELAY.

  cinc2017   : AFIB head (AF vs Normal), primary; SR head (Normal vs AF) also reported.
               Bootstrap unit: record.
  Main-paper metric: MACRO = mean of the dataset's head AUROCs (bootstrapped per resample); per-head
  AUROC / AUPRC go to the supplement.
  icentia11k : SR, AFIB, AFLT, PVC, PAC heads, each against all other selected windows.
               Bootstrap unit: window (record), as the pipeline's --bootstrap (empirical_bootstrap).
               AUPRC depends on the class mix of the selected sample, not on real prevalence.

Checks: every model must have exactly the dataset's records, in order, with targets equal to the
dataset labels (a dropped or reordered record fails loudly).
Paired comparisons use the same bootstrap resamples for every model: difference RELAY - model, 95% CI,
two-sided bootstrap p-value. The "oracle" slot (icentia11k, best slot chosen on test per model and head)
is reported separately and labelled as baseline-favouring.

Usage: python eval_poc_zero_shot.py [--n-boot 1000] [--seed 0] [--reference relay]
Output: zero_shot/results/summary_<dataset>.csv (+ printed tables)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "zero_shot" / "results"
HEADS = {"cinc2017": ["AFIB", "SR"], "icentia11k": ["SR", "AFIB", "AFLT", "PVC", "PAC"]}
PRIMARY_SLOT = {"cinc2017": "I", "icentia11k": "II"}
# ECG-JEPA encodes only the 8 independent leads (I, II, V1-V6): a lead placed in these slots never reaches it
NOT_APPLICABLE = {"ecg_jepa_multiblock": {"III", "aVR", "aVL", "aVF"}}
MODEL_ORDER = ["ecg_founder", "ecg_jepa_multiblock", "st_mem", "st_mem_native", "merl_resnet", "ecgfm_ked",
               "hubert_ecg_base", "cpc", "relay"]


def load_dataset(name):
    df = pd.read_pickle(ROOT / name / "df_memmap.pkl")
    df = df[df.strat_fold == df.strat_fold.max()].reset_index(drop=True)
    itos = [str(x) for x in np.load(ROOT / name / "lbl_itos.npy")]
    y = np.zeros((len(df), len(itos)), dtype=np.float32)
    for i, labels in enumerate(df.label):
        y[i, labels] = 1
    return df, itos, y


def load_predictions(path, y_true, itos):
    z = np.load(path, allow_pickle=True)
    preds, targs = z["preds"], z["targs"]
    assert [str(x) for x in z["lbl_itos"]] == itos, f"{path}: label order differs from the dataset"
    assert preds.shape == y_true.shape, f"{path}: {preds.shape[0]} records, dataset has {y_true.shape[0]}"
    assert np.array_equal(targs, y_true), f"{path}: targets do not match the dataset rows (order or content)"
    return preds


def bootstrap_weights(groups, n_boot, seed):
    """(n_boot, n) multiplicity weights: resample the unique groups with replacement; every sample
    of a drawn group gets the group's draw count."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(n_boot)])
    return counts[:, inv].astype(np.float32)


def weighted_auroc(scores, y, W):
    """AUROC (ties count 1/2) for each weight row of W (B, n); returns (B,). Same as sklearn for W = 1.
    A row without a positive or a negative gets 0.5, as the original implementation (multiclass_roc_curve)."""
    order = np.argsort(scores, kind="mergesort")
    s, yo, Wo = scores[order], y[order], W[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    pos = np.add.reduceat(Wo * yo, starts, axis=1)
    neg = np.add.reduceat(Wo * (1 - yo), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    P, N = pos.sum(1), neg.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((P > 0) & (N > 0), (pos * (below + 0.5 * neg)).sum(1) / (P * N), 0.5)


def ci(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def vs_best(dataset, res, boot):
    """Paired bootstrap of each model against the best model of the same row (same resamples): row = slot x head,
    plus, for icentia11k, the mean over a model's applicable slots per head. A model whose CI of (best - model)
    includes 0 is not significantly worse than the best (marked * in the tables). Not-applicable slots are left
    out. Output: summary_<dataset>_vs_best.csv"""
    ok = lambda m, s: s not in NOT_APPLICABLE.get(m, ())
    res = res[res.model != "st_mem"]  # the tables show ST-MEM native; zero-padded ST-MEM is not a table column
    rows = []

    def compare(slot, head, points, boots):
        best = max(points, key=points.get)
        for m in points:
            d = boots[best] - boots[m]
            lo, hi = ci(d)
            p = 1.0 if m == best else min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            rows.append({"dataset": dataset, "slot": slot, "head": head, "model": m, "auroc": points[m], "best": best,
                         "best_minus": points[best] - points[m], "diff_lo": lo, "diff_hi": hi, "p_boot": p,
                         "not_worse": bool(m == best or lo <= 0)})

    for (slot, head), sub in res.groupby(["slot", "head"]):
        sub = sub[[ok(m, slot) for m in sub.model]]
        compare(slot, head, dict(zip(sub.model, sub.auroc)), {m: boot[(slot, m, head)] for m in sub.model})
    if res.slot.nunique() > 1:
        for head, sub in res.groupby("head"):
            points, boots = {}, {}
            for m, sm in sub.groupby("model"):
                slots = [s for s in sm.slot if ok(m, s)]
                points[m] = float(sm[sm.slot.isin(slots)].auroc.mean())
                boots[m] = np.mean([boot[(s, m, head)] for s in slots], axis=0)
            compare("MEAN", head, points, boots)
    pd.DataFrame(rows).to_csv(RESULTS / f"summary_{dataset}_vs_best.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", default="relay")
    args = parser.parse_args()

    for dataset, heads in HEADS.items():
        if not (RESULTS / dataset).exists():
            continue
        df, itos, y_true = load_dataset(dataset)
        groups = np.arange(len(df))  # records, as the pipeline's --bootstrap
        W = bootstrap_weights(groups, args.n_boot, args.seed)
        ones = np.ones((1, len(df)), dtype=np.float32)

        rows, boot = [], {}  # boot[(slot, model, head)] -> (B,) bootstrap AUROCs
        for slot_dir in sorted((RESULTS / dataset).glob("slot_*")):
            slot = slot_dir.name[len("slot_"):]
            for model_dir in sorted(slot_dir.iterdir()):
                pred_file = model_dir / "predictions" / f"{dataset}_version_0" / "agg" / "test_0_epoch_0_agg.npz"
                if not pred_file.exists():
                    continue
                preds = load_predictions(pred_file, y_true, itos)
                head_points, head_boots = [], []
                for head in heads:
                    j = itos.index(head)
                    yj, pj = y_true[:, j], preds[:, j]
                    point = float(weighted_auroc(pj, yj, ones)[0])
                    b = weighted_auroc(pj, yj, W)
                    boot[(slot, model_dir.name, head)] = b
                    head_points.append(point)
                    head_boots.append(b)
                    lo, hi = ci(b)
                    rows.append({"dataset": dataset, "slot": slot, "model": model_dir.name, "head": head,
                                 "n_pos": int(yj.sum()), "n_neg": int((1 - yj).sum()),
                                 "auroc": point, "auroc_lo": lo, "auroc_hi": hi,
                                 "auprc": float(average_precision_score(yj, pj)), "prevalence": float(yj.mean())})
                # macro AUROC over the dataset's heads (main-paper metric); per resample, so CIs and paired
                # differences are those of the macro itself
                b = np.mean(head_boots, axis=0)
                boot[(slot, model_dir.name, "MACRO")] = b
                lo, hi = ci(b)
                rows.append({"dataset": dataset, "slot": slot, "model": model_dir.name, "head": "MACRO",
                             "n_pos": np.nan, "n_neg": np.nan, "auroc": float(np.mean(head_points)),
                             "auroc_lo": lo, "auroc_hi": hi, "auprc": np.nan, "prevalence": np.nan})
        if not rows:
            continue
        res = pd.DataFrame(rows)

        # paired differences vs the reference model, same slot and head
        diffs = []
        for r in res.itertuples():
            ref = boot.get((r.slot, args.reference, r.head))
            if ref is None or r.model == args.reference:
                diffs.append((np.nan, np.nan, np.nan, np.nan))
                continue
            d = ref - boot[(r.slot, r.model, r.head)]
            ref_point = res[(res.slot == r.slot) & (res.model == args.reference) & (res["head"] == r.head)].auroc.iloc[0]
            lo, hi = ci(d)
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            diffs.append((ref_point - r.auroc, lo, hi, p))
        res[[f"{args.reference}_minus", "diff_lo", "diff_hi", "p_boot"]] = diffs
        out = RESULTS / f"summary_{dataset}.csv"
        res.to_csv(out, index=False)
        vs_best(dataset, res, boot)

        def show(sub, title):
            print(f"\n=== {dataset}: {title} ===")
            table = sub.assign(cell=[f"{100*a:.1f} [{100*l:.1f}, {100*h:.1f}]" for a, l, h in zip(sub.auroc, sub.auroc_lo, sub.auroc_hi)])
            table = table.pivot(index="model", columns="head", values="cell")
            table = table.reindex([m for m in MODEL_ORDER if m in table.index] + [m for m in table.index if m not in MODEL_ORDER])
            print(table[["MACRO"] + [h for h in heads if h in table.columns]].to_string())

        for slot in sorted(res.slot.unique(), key=lambda s: s != PRIMARY_SLOT[dataset]):
            tag = "primary" if slot == PRIMARY_SLOT[dataset] else "secondary"
            show(res[res.slot == slot], f"lead {slot} slot ({tag}), AUROC x100 [95% CI]")
        if dataset == "icentia11k" and res.slot.nunique() > 1:
            macro = res[res["head"] == "MACRO"].pivot(index="model", columns="slot", values="auroc")
            spread = pd.DataFrame({"slots": macro.notna().sum(1), "mean": 100 * macro.mean(1), "std": 100 * macro.std(1),
                                   "min": 100 * macro.min(1), "max": 100 * macro.max(1)}).round(2)
            spread.to_csv(RESULTS / "slot_spread_icentia11k.csv")
            print(f"\n=== icentia11k: macro AUROC x100 across lead slots ===\n{spread.to_string()}")
        if dataset == "icentia11k" and res.slot.nunique() > 2:
            best = res.loc[res.groupby(["model", "head"]).auroc.idxmax()]
            show(best, "ORACLE slot chosen on test per model and head (baseline-favouring)")
        print(f"\nn = {len(df):,} {'windows' if dataset == 'icentia11k' else 'records'}, "
              f"{len(np.unique(groups)):,} bootstrap units, {args.n_boot} resamples -> {out}")


if __name__ == "__main__":
    main()
