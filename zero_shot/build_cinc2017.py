"""Builds the CinC 2017 zero-shot test set in the benchmark's memmap format -> cinc2017/

Task: AF vs Normal (the 12-lead heads AFIB and SR of the PTB-XL All models).
  - records labelled N (Normal) or A (AF) only; O (other rhythm) and ~ (noisy) are dropped:
    they have no counterpart among the 12-lead labels.
  - label vectors are in PTB-XL All's 71-label order (lbl_itos["label_all"]): N -> SR, A -> AFIB.
    SR, not NORM: NORM is a whole-ECG morphology statement, SR is the rhythm.
  - signal: the single AliveCor lead (approximates lead I), 300 Hz, mV, stored as ONE channel
    and unmodified (no filtering, no polarity correction). The lead slot it occupies in the
    12-lead input is chosen at test time.
  - records shorter than 10 s (11 of them) are zero-padded at the tail to exactly 10.0 s, identically for
    every model: the benchmark drops records shorter than a model's input window (ECG-JEPA: 10 s), and all
    models must be scored on the same records. `padded_s` records how much was added.
  - every record is in the test fold.
  - polarity_inverted: heuristic flag, recorded for reporting only (see estimate_inversion).

Usage: python build_cinc2017.py
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import wfdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from clinical_ts.data.time_series_dataset_utils import reformat_as_memmap  # noqa: E402

RAW = ROOT / "cinc2017" / "raw"
OUT = ROOT / "cinc2017"
FS = 300
LABEL_MAP = {"N": "SR", "A": "AFIB"}
MIN_LEN = 10 * FS  # longest input window among the benchmarked models (ECG-JEPA)
TEST_FOLD = 10


def ptbxl_all_labels():
    with open(ROOT / "ptb-xl" / "records500" / "lbl_itos.pkl", "rb") as f:
        return [str(x) for x in pickle.load(f)["label_all"]]


def estimate_inversion(sig, fs=FS):
    """True if the dominant QRS deflection is negative: compares the 99.5th percentile with the
    magnitude of the 0.5th percentile of the 5-25 Hz band-passed signal."""
    b, a = scipy.signal.butter(2, [5, 25], btype="bandpass", fs=fs)
    x = scipy.signal.filtfilt(b, a, sig)
    return bool(abs(np.percentile(x, 0.5)) > 1.5 * np.percentile(x, 99.5))


def main():
    itos = ptbxl_all_labels()
    assert len(itos) == 71, len(itos)
    stoi = {s: i for i, s in enumerate(itos)}

    ref = pd.read_csv(RAW / "REFERENCE-v3.csv", header=None, names=["record", "label_cinc"])
    ref = ref[ref.label_cinc.isin(LABEL_MAP)].reset_index(drop=True)

    npy_dir = OUT / "npy_tmp"
    npy_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec, lab in zip(ref.record, ref.label_cinc):
        r = wfdb.rdrecord(str(RAW / "training2017" / rec))
        assert r.fs == FS and r.sig_name == ["ECG"] and r.units == ["mV"], rec
        sig = r.p_signal[:, 0].astype(np.float32)
        assert np.isfinite(sig).all(), rec
        original_len = len(sig)
        polarity_inverted = estimate_inversion(sig)
        if original_len < MIN_LEN:
            sig = np.concatenate([sig, np.zeros(MIN_LEN - original_len, dtype=np.float32)])
        np.save(npy_dir / f"{rec}.npy", sig[:, None])  # (T, 1)
        rows.append({
            "data": f"{rec}.npy",
            "record": rec,
            "patient_id": rec,  # CinC 2017 gives no patient ids: one record per subject assumed
            "label_cinc": lab,
            "label": [stoi[LABEL_MAP[lab]]],
            "strat_fold": TEST_FOLD,
            "data_length": len(sig),
            "duration_s": original_len / FS,
            "padded_s": (len(sig) - original_len) / FS,
            "polarity_inverted": polarity_inverted,
        })
    df = pd.DataFrame(rows)

    reformat_as_memmap(df, OUT / "memmap.npy", fs=FS, data_folder=npy_dir, delete_npys=True)
    shutil.rmtree(npy_dir, ignore_errors=True)
    df.to_pickle(OUT / "df.pkl")
    np.save(OUT / "lbl_itos.npy", np.array(itos))

    print(f"CinC 2017 zero-shot set: {len(df)} records -> {OUT}")
    print(df.label_cinc.value_counts().to_dict())
    print(f"duration: {df.duration_s.min():.1f}-{df.duration_s.max():.1f} s, median {df.duration_s.median():.1f} s; "
          f"< 10 s: {(df.duration_s < 10).sum()}")
    print(f"tail-padded to 10 s: {(df.padded_s > 0).sum()} records (max {df.padded_s.max():.2f} s)")
    print("polarity flagged inverted:", df.groupby("label_cinc").polarity_inverted.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
