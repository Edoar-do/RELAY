"""LaTeX tables for the zero-shot POC evaluation, from zero_shot/results/summary_<dataset>.csv
(eval_poc_zero_shot.py). Single runs: values are point estimates (x100); CIs stay in the CSVs.

  correct_tables/table_poc.tex            main paper: macro AUROC
      CinC 2017   : lead I slot (AliveCor single lead = lead I)
      Icentia11k  : one row per lead slot the single patch lead is placed in (its identity is unknown),
                    plus mean (std) across the slots
  correct_tables/table_poc_per_label.tex  supplement: per-label AUROC
      CinC 2017 (lead I slot): AFIB, SR; Icentia11k: SR, AFIB, AFLT, PVC, PAC, mean across slots

ST-MEM is ST-MEM native (--st-mem-native-leads), as in the other tables.
Marks: \\uno / \\due / \\tre by dense ranking of the displayed values within each row; * = not significantly
worse than the best model of the row (paired bootstrap, 95% CI of best - model includes 0;
summary_<dataset>_vs_best.csv from eval_poc_zero_shot.py).
Usage: python make_poc_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "zero_shot" / "results"
OUT = ROOT / "correct_tables"
COLUMNS = [("ecg_founder", "ECGFounder", "33.8M"), ("ecg_jepa_multiblock", "ECG-JEPA", "87.2M"),
           ("st_mem_native", "ST-MEM", "90.3M"), ("merl_resnet", "MERL", "4.6M"),
           ("ecgfm_ked", "ECGFM-KED", "9.7M"), ("hubert_ecg_base", "HuBERT-ECG B.", "97.2M"),
           ("cpc", "ECG-CPC", "3.8M"), ("relay", "\\textbf{RELAY}", "7.1M")]
SLOTS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
HEADS = {"cinc2017": ["AFIB", "SR"], "icentia11k": ["SR", "AFIB", "AFLT", "PVC", "PAC"]}
# ECG-JEPA encodes only the 8 independent leads (I, II, V1-V6; EcgJepaWrapper selected_indices): the 4 limb
# leads that are linear combinations of I and II are discarded, so a lead placed there never reaches it.
NOT_APPLICABLE = {"ecg_jepa_multiblock": {"III", "aVR", "aVL", "aVF"}}
NA = "N/A"


def marked(values):
    """values: list of display strings or None -> cells with dense-rank marks on the numeric part."""
    nums = [float(v.rstrip("*").split("(")[0]) for v in values if v not in (None, NA)]
    distinct = sorted(set(nums), reverse=True)
    out = []
    for v in values:
        if v is None or v == NA:
            out.append("--" if v is None else NA)
            continue
        star = "*" if v.endswith("*") else ""
        v = v.rstrip("*")
        r = distinct.index(float(v.split("(")[0]))
        out.append((f"\\{['uno', 'due', 'tre'][r]}{{{v}}}" if r < 3 else v) + star)
    return out


def header(ncols_extra=1):
    names = " & ".join(n for _, n, _ in COLUMNS)
    sizes = " & ".join(f"\\textbf{{({s})}}" if "RELAY" in n else f"({s})" for _, n, s in COLUMNS)
    return [f"\\multirow{{2}}{{*}}{{Dataset}} & \\multirow{{2}}{{*}}{{Lead slot}} & {names} \\\\",
            f"        &     & {sizes} \\\\", "        \\midrule"]


def load(dataset):
    f = RESULTS / f"summary_{dataset}.csv"
    return pd.read_csv(f) if f.exists() else None


def applicable(model, slot):
    return slot not in NOT_APPLICABLE.get(model, ())


def load_vs_best(dataset):
    f = RESULTS / f"summary_{dataset}_vs_best.csv"
    return pd.read_csv(f) if f.exists() else None


def star(vs, slot, model, head):
    """'*' if the model is not significantly worse than the row's best (and is not the best itself)."""
    if vs is None:
        return ""
    r = vs[(vs.slot == slot) & (vs.model == model) & (vs["head"] == head)]
    return "*" if (not r.empty and bool(r.not_worse.iloc[0]) and r.best.iloc[0] != model) else ""


def cell(df, slot, model, head, vs=None):
    if not applicable(model, slot):
        return NA
    r = df[(df.slot == slot) & (df.model == model) & (df["head"] == head)]
    return None if r.empty else f"{100 * r.auroc.iloc[0]:.1f}" + star(vs, slot, model, head)


def main_table():
    cinc, ic = load("cinc2017"), load("icentia11k")
    vc, vi = load_vs_best("cinc2017"), load_vs_best("icentia11k")
    lines = ["\\begin{table*}", "\\centering", "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{0.9}",
             "\\resizebox{\\textwidth}{!}{", "\\begin{tabular}{ll cccccccc}", "\\toprule"] + header()
    if cinc is not None:
        vals = [cell(cinc, "I", m, "MACRO", vc) for m, _, _ in COLUMNS]
        lines += ["CinC2017 & lead I & " + " & ".join(marked(vals)) + " \\\\", "\\midrule"]
    if ic is not None:
        slots = [s for s in SLOTS if (ic.slot == s).any()]
        lines.append(f"\\multirow{{{len(slots) + (len(slots) > 1)}}}{{*}}{{Icentia11k}}")
        for s in slots:
            vals = [cell(ic, s, m, "MACRO", vi) for m, _, _ in COLUMNS]
            lines.append(f" & {s} & " + " & ".join(marked(vals)) + " \\\\")
        macro = ic[ic["head"] == "MACRO"].pivot(index="model", columns="slot", values="auroc")
        means = []
        for m, _, _ in COLUMNS:
            ok_slots = [s for s in slots if applicable(m, s)]
            if m in macro.index and macro.loc[m, ok_slots].notna().all():
                v = macro.loc[m, ok_slots]
                means.append(f"{100 * v.mean():.1f}({100 * v.std():.1f})" + star(vi, "MEAN", m, "MACRO"))
            else:
                means.append(None)
        if len(slots) > 1:
            lines += ["\\cmidrule{2-10}", " & mean (std) & " + " & ".join(marked(means)) + " \\\\"]
    lines += ["\\bottomrule", "\\end{tabular}", "}",
              "\\caption{Zero-shot macro AUROC on real point-of-care recordings, with the models finetuned on "
              "12-lead PTB-XL All (heads mapped to the dataset labels: CinC2017 AFIB, SR; Icentia11k SR, AFIB, "
              "AFLT, PVC, PAC). The single recorded lead is placed in one lead slot of a zero-filled 12-lead input: "
              "lead I for CinC2017 (AliveCor); every slot for Icentia11k, whose patch lead has no standard identity. "
              "RELAY uses no lead identity, so its output does not depend on the slot. N/A: ECG-JEPA uses only the 8 "
              "independent leads (I, II, V1--V6), so a lead placed in III, aVR, aVL or aVF never reaches it; its mean "
              "is over its 8 slots. *: not significantly worse than the best model of the row (paired bootstrap of the "
              "difference, 1000 resamples, 95\\% CI includes 0; record-level for CinC2017, patient-level for Icentia11k; "
              "for the mean row, the difference of the slot means).}",
              "\\label{tab: poc}", "\\end{table*}"]
    (OUT / "table_poc.tex").write_text("\n".join(lines) + "\n")


def per_label_table():
    cinc, ic = load("cinc2017"), load("icentia11k")
    vc, vi = load_vs_best("cinc2017"), load_vs_best("icentia11k")
    lines = ["\\begin{table*}", "\\centering", "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{0.9}",
             "\\resizebox{\\textwidth}{!}{", "\\begin{tabular}{ll cccccccc}", "\\toprule"] + header()
    if cinc is not None:
        lines.append(f"\\multirow{{{len(HEADS['cinc2017'])}}}{{*}}{{CinC2017 (lead I)}}")
        for h in HEADS["cinc2017"]:
            lines.append(f" & {h} & " + " & ".join(marked([cell(cinc, 'I', m, h, vc) for m, _, _ in COLUMNS])) + " \\\\")
        lines.append("\\midrule")
    if ic is not None:
        lines.append(f"\\multirow{{{len(HEADS['icentia11k'])}}}{{*}}{{Icentia11k (mean over slots)}}")
        for h in HEADS["icentia11k"]:
            sub = ic[(ic["head"] == h) & ic.apply(lambda r: applicable(r.model, r.slot), axis=1)].groupby("model").auroc.mean()
            vals = [f"{100 * sub[m]:.1f}" + star(vi, "MEAN", m, h) if m in sub.index else None for m, _, _ in COLUMNS]
            lines.append(f" & {h} & " + " & ".join(marked(vals)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}",
              "\\caption{Zero-shot per-label AUROC on the point-of-care datasets (models finetuned on 12-lead "
              "PTB-XL All). Icentia11k: mean over the lead slots of Table~\\ref{tab: poc} (ECG-JEPA: its 8 applicable slots); per-slot values, "
              "95\\% bootstrap CIs (record-level for CinC2017, patient-level for Icentia11k) and AUPRC are in "
              "the released results. *: not significantly worse than the best model of the row (paired bootstrap of the "
              "difference, 95\\% CI includes 0).}",
              "\\label{tab: poc per label}", "\\end{table*}"]
    (OUT / "table_poc_per_label.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main_table()
    per_label_table()
    print("written:", OUT / "table_poc.tex", OUT / "table_poc_per_label.tex")
