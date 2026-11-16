"""Builds the Icentia11k zero-shot test set in the benchmark's memmap format -> icentia11k/

One sample per window selected by select_icentia11k.py: 10 s (2,500 samples) of the single patch
lead at 250 Hz in mV, stored as ONE channel and unmodified (the patch lead is not a standard lead; the
lead slot it occupies in the 12-lead input is chosen at test time). Label vectors are in PTB-XL All's
71-label order: SR, AFIB, AFLT, PVC, PAC as listed in selected_windows.csv. Every window is in the
test fold; patient_id is kept for the patient-level bootstrap.

Usage: python build_icentia11k.py [--procs 16]
"""
import argparse
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from clinical_ts.data.time_series_dataset_utils import reformat_as_memmap  # noqa: E402

DIR = ROOT / "icentia11k"
RAW = DIR / "raw"
FS = 250
WIN = 2500
TEST_FOLD = 10


def extract_segment(args):
    """Reads all selected windows of one segment; writes one (2500, 1) npy per window."""
    record, rows, npy_dir = args
    sig = wfdb.rdrecord(str(RAW / record), physical=True)
    assert sig.fs == FS and sig.units == ["mV"], record
    x = sig.p_signal[:, 0].astype(np.float32)
    out = []
    for name, start in rows:
        w = x[start:start + WIN]
        assert len(w) == WIN and np.isfinite(w).all(), (record, start)
        np.save(npy_dir / name, w[:, None])
        out.append(name)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=16)
    args = parser.parse_args()

    with open(ROOT / "ptb-xl" / "records500" / "lbl_itos.pkl", "rb") as f:
        itos = [str(x) for x in pickle.load(f)["label_all"]]
    stoi = {s: i for i, s in enumerate(itos)}

    sel = pd.read_csv(DIR / "selected_windows.csv")
    sel["data"] = [f"{r.split('/')[-1]}_w{w:03d}.npy" for r, w in zip(sel.record, sel.window)]

    npy_dir = DIR / "npy_tmp"
    npy_dir.mkdir(exist_ok=True)
    jobs = [(rec, list(zip(g.data, g.start)), npy_dir) for rec, g in sel.groupby("record")]
    done = 0
    with ProcessPoolExecutor(args.procs) as pool:
        for out in pool.map(extract_segment, jobs, chunksize=16):
            done += len(out)
    assert done == len(sel), (done, len(sel))

    df = pd.DataFrame({
        "data": sel.data,
        "patient_id": sel.patient,
        "record": sel.record,
        "window": sel.window,
        "start": sel.start,
        "rhythm": sel.rhythm_name,
        "n_N": sel.n_N, "n_S": sel.n_S, "n_V": sel.n_V, "n_Q": sel.n_Q,
        "drawn_for": sel.drawn_for,
        "labels_text": sel.labels,
        "label": [[stoi[l] for l in s.split(";")] for s in sel.labels],
        "strat_fold": TEST_FOLD,
        "data_length": WIN,
    })
    reformat_as_memmap(df, DIR / "memmap.npy", fs=FS, data_folder=npy_dir, delete_npys=True)
    shutil.rmtree(npy_dir, ignore_errors=True)
    df.to_pickle(DIR / "df.pkl")
    np.save(DIR / "lbl_itos.npy", np.array(itos))
    print(f"Icentia11k zero-shot set: {len(df):,} windows, {df.patient_id.nunique():,} patients -> {DIR}")


if __name__ == "__main__":
    main()
