"""Selects the Icentia11k zero-shot windows from the scan tables (scan_icentia11k.py). Rules fixed before
looking at any model output:

  valid window : 10 s, fully inside one annotated NSR / AFIB / AFL episode (no rhythm change),
                 >= 1 annotated beat, <= 50% undefined (Q) beats
  classes      : NSR (no S/V beat), AFIB, AFL, PVC (>= 1 V beat, NSR), PAC (>= 1 S beat, NSR)
  sampling     : every patient; up to WINDOWS_PER_CLASS windows per class per patient,
                 uniformly at random with a per-patient seed (SEED, patient)
  labels       : from the window content, not from the class it was drawn for:
                 SR (NSR episode), AFIB, AFLT, PVC (>= 1 V beat), PAC (>= 1 S beat)

Outputs (icentia11k/):
  selected_windows.csv   one row per window: patient, segment, window, start sample, rhythm, beat counts,
                         drawn_for (class), labels
  selected_segments.txt  record paths whose signals must be downloaded
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "icentia11k"
WIN = 2500
SEED = 0
WINDOWS_PER_CLASS = 2
KEYS = ("patient", "segment", "window", "rhythm", "n_N", "n_S", "n_V", "n_Q")
RHYTHM_NAME = {1: "NSR", 2: "AFIB", 3: "AFL"}


def load_windows():
    parts = {k: [] for k in KEYS}
    for f in sorted(glob.glob(str(DIR / "windows" / "shard_*.npz"))):
        z = np.load(f)
        for k in KEYS:
            parts[k].append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def main():
    d = load_windows()
    total = d["n_N"].astype(np.int32) + d["n_S"] + d["n_V"] + d["n_Q"]
    r = d["rhythm"]
    valid = (r >= 1) & (r <= 3) & (total > 0) & (d["n_Q"] * 2 <= total)
    classes = {
        "NSR": valid & (r == 1) & (d["n_S"] == 0) & (d["n_V"] == 0),
        "AFIB": valid & (r == 2),
        "AFL": valid & (r == 3),
        "PVC": valid & (r == 1) & (d["n_V"] >= 1),
        "PAC": valid & (r == 1) & (d["n_S"] >= 1),
    }

    chosen = {}  # row index -> class it was first drawn for
    for name, mask in classes.items():
        idx = np.flatnonzero(mask)
        pats = d["patient"][idx]
        order = np.argsort(pats, kind="stable")
        idx, pats = idx[order], pats[order]
        bounds = np.flatnonzero(np.diff(pats)) + 1
        for group in np.split(idx, bounds):
            rng = np.random.default_rng([SEED, int(d["patient"][group[0]])])
            take = group if len(group) <= WINDOWS_PER_CLASS else rng.choice(group, WINDOWS_PER_CLASS, replace=False)
            for i in take:
                chosen.setdefault(int(i), name)

    rows = np.array(sorted(chosen))
    df = pd.DataFrame({k: d[k][rows] for k in KEYS})
    df["drawn_for"] = [chosen[i] for i in rows]
    df["start"] = df.window.astype(np.int64) * WIN
    df["record"] = [f"p{p // 1000:02d}/p{p:05d}/p{p:05d}_s{s:02d}" for p, s in zip(df.patient, df.segment)]
    df["rhythm_name"] = df.rhythm.map(RHYTHM_NAME)
    df["labels"] = [
        ";".join(l for l, on in (("SR", rh == 1), ("AFIB", rh == 2), ("AFLT", rh == 3), ("PVC", v >= 1), ("PAC", s >= 1)) if on)
        for rh, v, s in zip(df.rhythm, df.n_V, df.n_S)
    ]
    df.to_csv(DIR / "selected_windows.csv", index=False)
    segments = sorted(df.record.unique())
    (DIR / "selected_segments.txt").write_text("\n".join(segments) + "\n")

    print(f"selected {len(df):,} windows from {df.patient.nunique():,} patients, {len(segments):,} segments")
    print("drawn for:", df.drawn_for.value_counts().to_dict())
    for lab in ("SR", "AFIB", "AFLT", "PVC", "PAC"):
        has = df.labels.str.split(";").apply(lambda ls: lab in ls)
        print(f"  label {lab:5s}: {has.sum():>7,} windows, {df.patient[has].nunique():>6,} patients")


if __name__ == "__main__":
    main()
