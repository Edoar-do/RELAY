"""Linear CKA between full-lead and reduced-lead representations (dropped vs zero-padded leads).

Reads cka/embeddings_<arm>.npz (cka_extract.py) and writes, in cka/:
  cka_to_full.csv / .pdf / .svg
      per arm and setting: CKA(12-lead embedding, reduced-lead embedding) with the missing leads
      dropped vs zero-padded; K1 / K2 averaged over the 3 lead-draw seeds (mean, std, n).
  cka_matrix_<setting>.csv, cka_matrix.pdf / .svg
      CKA between every pair of (arm x {12 leads, dropped, zero-padded}) representations of the
      same records; K1 / K2 = mean of the per-seed matrices.

Linear CKA (Kornblith et al., 2019) on column-centred features, in float64:
  CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
Usage:  python cka_plot.py      (any env with numpy, pandas, matplotlib)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent / "cka"
ARMS = ["rls", "norls_pm0217", "norls_tok", "norls"]
ARM_LABEL = {"rls": "RLS", "norls_pm0217": "noRLS\nmask", "norls_tok": "noRLS\ntoken", "norls": "noRLS\nstep"}
ARM_SHORT = {"rls": "RLS", "norls_pm0217": "noRLS-mask", "norls_tok": "noRLS-tok", "norls": "noRLS-step"}
SETTINGS = {"K1": "1 random lead", "K2": "2 random leads", "leadI": "Lead I only", "leadII": "Lead II only"}
SEEDS = {"K1": [0, 43, 44], "K2": [0, 43, 44], "leadI": [None], "leadII": [None]}

# Reference palette (dataviz skill, references/palette.md): categorical slots 1-2, light mode
DROP_C, PAD_C = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # blue 100 -> 700


_PREP = {}  # id(array) -> (centred float64 features, ||X^T X||_F)


def _prep(X: np.ndarray):
    if id(X) not in _PREP:
        Xc = X.astype(np.float64); Xc -= Xc.mean(0)
        _PREP[id(X)] = (Xc, np.linalg.norm(Xc.T @ Xc, "fro"), X)  # keep X alive so id() stays unique
    return _PREP[id(X)][:2]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    (Xc, nx), (Yc, ny) = _prep(X), _prep(Y)
    return float(np.linalg.norm(Yc.T @ Xc, "fro") ** 2 / (nx * ny))


def key(setting, seed, rep):
    return f"{setting}{'' if seed is None else f'_s{seed}'}_{rep}"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2); ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelcolor=INK, width=0.6, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(D / f"{name}.{ext}", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "text.color": INK, "axes.labelcolor": INK})
    # load into memory once: indexing an NpzFile re-reads the array from disk on every access
    E = {}
    for arm in ARMS:
        if (D / f"embeddings_{arm}.npz").exists():
            with np.load(D / f"embeddings_{arm}.npz") as z:
                E[arm] = {k: z[k] for k in z.files}
    arms = [a for a in ARMS if a in E]

    # ---------------- CKA to the arm's own 12-lead representation ----------------
    rows = []
    for arm in arms:
        full = E[arm]["K12"]
        for s_name, seeds in SEEDS.items():
            for rep in ("drop", "pad"):
                vals = [linear_cka(full, E[arm][key(s_name, s, rep)]) for s in seeds]
                rows.append(dict(arm=arm, setting=s_name, missing_leads="dropped" if rep == "drop" else "zero-padded",
                                 cka_mean=np.mean(vals), cka_std=np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                                 n_seeds=len(vals), per_seed=";".join(f"{v:.4f}" for v in vals)))
    to_full = pd.DataFrame(rows)
    to_full.to_csv(D / "cka_to_full.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    x = np.arange(len(arms)); w = 0.36
    for ax, (s_name, title) in zip(axes, SETTINGS.items()):
        style_axes(ax)
        for j, (rep, color, dx) in enumerate((("dropped", DROP_C, -w / 2), ("zero-padded", PAD_C, w / 2))):
            sub = to_full[(to_full.setting == s_name) & (to_full.missing_leads == rep)].set_index("arm").loc[arms]
            err = sub.cka_std.to_numpy() if s_name in ("K1", "K2") else None
            # width w - 0.02 leaves a surface gap between the paired bars
            ax.bar(x + dx, sub.cka_mean, width=w - 0.02, color=color, label=f"missing leads {rep}",
                   yerr=err, ecolor=INK2, capsize=2, error_kw=dict(linewidth=0.8), zorder=3)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=7.5)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Linear CKA with the 12-lead representation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.08, "Ribeiro-dev validation (n = 4,819), frozen Stage-1 encoders, no training on reduced or padded "
             "inputs. noRLS mask / token / step = masked-target-, token- and step-matched arms.\n"
             "Error bars: std over 3 lead draws.", ha="center", fontsize=7.5, color=INK2)
    save(fig, "cka_to_full")

    # ---------------- cross-representation CKA matrices ----------------
    labels = [f"{ARM_SHORT[a]} · {r}" for a in arms for r in ("12 leads", "dropped", "zero-padded")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor=SURFACE, gridspec_kw=dict(wspace=0.04, hspace=0.32))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
    for ax, (s_name, title) in zip(axes.ravel(), SETTINGS.items()):
        mats = []
        for s in SEEDS[s_name]:
            reps = [E[a]["K12"] if r == "full" else E[a][key(s_name, s, r)] for a in arms for r in ("full", "drop", "pad")]
            n = len(reps)
            M = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    M[i, j] = M[j, i] = linear_cka(reps[i], reps[j])
            mats.append(M)
        M = np.mean(mats, 0)
        pd.DataFrame(M, index=labels, columns=labels).to_csv(D / f"cka_matrix_{s_name}.csv")
        im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="#ffffff" if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        # all panels share the row order: label rows on the left column only
        ax.set_yticklabels(labels if ax in axes[:, 0] else [], fontsize=7)
        for b in range(3, n, 3):  # separate arms with surface-coloured gaps
            ax.axhline(b - 0.5, color=SURFACE, linewidth=2); ax.axvline(b - 0.5, color=SURFACE, linewidth=2)
        ax.set_title(title + (" (mean of 3 lead draws)" if s_name in ("K1", "K2") else ""), fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0, labelcolor=INK)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Linear CKA"); cbar.outline.set_visible(False)
    save(fig, "cka_matrix")

    print(to_full[["arm", "setting", "missing_leads", "cka_mean", "cka_std"]].round(4).to_string(index=False))
    print(f"written to {D}")


if __name__ == "__main__":
    main()
