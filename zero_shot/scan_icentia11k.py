"""Scans every Icentia11k annotation file and summarises each segment as non-overlapping 10-s windows
(2,500 samples at 250 Hz, aligned to the segment start, matching PTB-XL's record length).

For every window it stores:
  rhythm : 0 none (not fully inside an annotated rhythm episode, or spanning a rhythm change)
           1 NSR   2 AFIB   3 AFL   4 other episode type
  n_N, n_S, n_V, n_Q : beats of each type whose annotation falls inside the window
                       (N normal, S supraventricular ectopic = PAC, V ventricular ectopic = PVC,
                        Q undefined / unclassifiable)
Output: icentia11k/windows/shard_XXX.npz (100 patients each) with arrays
        patient (int16), segment (int8), window (int16), rhythm (uint8), n_N/n_S/n_V/n_Q (uint8)
Nothing is selected here; selection is done by select_icentia11k.py.

Usage: python scan_icentia11k.py [--procs 8]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "icentia11k" / "raw"
OUT = ROOT / "icentia11k" / "windows"
FS = 250
WIN = 10 * FS
RHYTHM_CODES = {"(N": 1, "(AFIB": 2, "(AFL": 3}
BEATS = ("N", "S", "V", "Q")


def rhythm_episodes(ann):
    """[(start_sample, end_sample, code)] from '+' annotations: '(X' opens an episode, ')' closes it."""
    episodes, current = [], None
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        if sym != "+":
            continue
        aux = (aux or "").strip()
        if aux.startswith("(") and not aux.endswith(")"):
            if current is not None:
                episodes.append((current[0], int(s), current[1]))
            current = (int(s), RHYTHM_CODES.get(aux, 4))
        elif aux == ")" and current is not None:
            episodes.append((current[0], int(s), current[1]))
            current = None
    return episodes


def scan_segment(record_path, n_samples):
    ann = wfdb.rdann(str(record_path), "atr")
    n_win = n_samples // WIN
    rhythm = np.zeros(n_win, dtype=np.uint8)
    for start, end, code in rhythm_episodes(ann):
        first = -(-start // WIN)          # first window starting at or after the episode start
        last = end // WIN                 # windows ending at or before the episode end
        rhythm[first:min(last, n_win)] = code
    counts = {}
    sym = np.asarray(ann.symbol)
    win_of = ann.sample // WIN
    for b in BEATS:
        idx = win_of[(sym == b) & (win_of < n_win)]
        counts[b] = np.minimum(np.bincount(idx, minlength=n_win), 255).astype(np.uint8)
    return rhythm, counts


def scan_shard(shard, segments, n_samples=1048577):
    rows = {k: [] for k in ("patient", "segment", "window", "rhythm", "n_N", "n_S", "n_V", "n_Q")}
    missing = 0
    for seg in segments:  # e.g. p00/p00000/p00000_s00
        path = RAW / seg
        if not path.with_suffix(".atr").exists():
            missing += 1
            continue
        rhythm, counts = scan_segment(path, n_samples)
        n = len(rhythm)
        name = Path(seg).name
        rows["patient"].append(np.full(n, int(name[1:6]), np.int16))
        rows["segment"].append(np.full(n, int(name.split("_s")[1]), np.int8))
        rows["window"].append(np.arange(n, dtype=np.int16))
        rows["rhythm"].append(rhythm)
        for b in BEATS:
            rows[f"n_{b}"].append(counts[b])
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}
    np.savez(OUT / f"shard_{shard:03d}.npz", **out)
    return shard, len(segments), missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    segments = (RAW / "RECORDS_segments.txt").read_text().split()
    by_shard = {}
    for s in segments:
        by_shard.setdefault(int(Path(s).name[1:6]) // 100, []).append(s)
    todo = {k: v for k, v in by_shard.items() if not (OUT / f"shard_{k:03d}.npz").exists()}
    print(f"{len(by_shard)} shards, {len(todo)} to scan", flush=True)
    with ProcessPoolExecutor(args.procs) as pool:
        for shard, n, missing in pool.map(scan_shard, todo.keys(), todo.values()):
            print(f"shard {shard:03d}: {n} segments, {missing} missing annotations", flush=True)


if __name__ == "__main__":
    main()
