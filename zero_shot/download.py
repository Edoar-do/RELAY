"""Downloads the two zero-shot POC datasets from PhysioNet's open-data S3 mirror (no credentials needed).

  cinc2017       : training2017.zip + REFERENCE-v3.csv (1.4 GB)            -> cinc2017/raw/
  icentia11k_atr : the beat/rhythm annotation (.atr) files of all 11,000 patients
                   (~540k segments, ~30 GB), no signals                    -> icentia11k/raw/
  icentia11k_dat : the header + signal (.hea/.dat) files of the segments listed in a file
                   (one record path per line, e.g. p00/p00000/p00000_s00) -> icentia11k/raw/

Resumable: files already on disk with the expected size are skipped.
Usage:  python download.py cinc2017
        python download.py icentia11k_atr [--workers 32]
        python download.py icentia11k_dat --records icentia11k/selected_segments.txt
"""
import argparse
import sys
import threading
import time
import zipfile

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root
S3 = "https://physionet-open.s3.amazonaws.com"
CINC_BASE = f"{S3}/challenge-2017/1.0.0"
ICENTIA_BASE = f"{S3}/icentia11k-continuous-ecg/1.0"


_local = threading.local()


def session():
    """One keep-alive session per thread: one DNS lookup per thread instead of one per file
    (the cluster DNS fails under many parallel lookups)."""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def get_text(url, retries=8):
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=120)
            r.raise_for_status()
            return r.text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch(url, dest, retries=8):
    """Downloads url to dest unless dest already exists (complete files only: written via .part). Returns bytes written."""
    if dest.exists():
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with session().get(url, timeout=120, stream=True) as r:
                r.raise_for_status()
                size = int(r.headers.get("Content-Length", -1))
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                if size >= 0 and tmp.stat().st_size != size:
                    raise IOError(f"size mismatch for {url}")
                tmp.rename(dest)
                return size
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(2 ** attempt)


def fetch_many(jobs, workers, label):
    done, failed, nbytes, t0 = 0, [], 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, url, dest): url for url, dest in jobs}
        for fut in as_completed(futures):
            try:
                nbytes += fut.result()
            except Exception as e:
                failed.append(str(e))
            done += 1
            if done % 5000 == 0 or done == len(jobs):
                rate = done / (time.time() - t0)
                print(f"[{label}] {done}/{len(jobs)} files, {nbytes / 2**30:.1f} GiB new, "
                      f"{rate:.0f} files/s, {len(failed)} failed", flush=True)
    if failed:
        print(f"[{label}] {len(failed)} failures, first: {failed[:3]}", file=sys.stderr)
    return failed


def cinc2017():
    raw = ROOT / "cinc2017" / "raw"
    for name in ("training2017.zip", "REFERENCE-v3.csv"):
        fetch(f"{CINC_BASE}/{name}", raw / name)
    if not (raw / "training2017").exists():
        with zipfile.ZipFile(raw / "training2017.zip") as z:
            z.extractall(raw)
    print("CinC 2017 records:", len(list((raw / "training2017").glob("*.hea"))))


def icentia_segments():
    listing = ROOT / "icentia11k" / "raw" / "RECORDS_segments.txt"
    if listing.exists():
        return listing.read_text().split()
    patients = get_text(f"{ICENTIA_BASE}/RECORDS").split()

    def patient_records(p):
        return [p + r for r in get_text(f"{ICENTIA_BASE}/{p}RECORDS").split()]

    segments = []
    with ThreadPoolExecutor(8) as pool:
        for i, recs in enumerate(pool.map(patient_records, patients), 1):
            segments += recs
            if i % 1000 == 0:
                print(f"[listing] {i}/{len(patients)} patients, {len(segments)} segments", flush=True)
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("\n".join(segments) + "\n")
    return segments


def icentia11k_atr(workers):
    raw = ROOT / "icentia11k" / "raw"
    segments = icentia_segments()
    print(f"Icentia11k: {len(segments)} segments")
    # headers are not needed to read annotations (all segments: 250 Hz, ~1,048,577 samples);
    # they are fetched with the signals of the selected segments
    jobs = [(f"{ICENTIA_BASE}/{s}.atr", raw / f"{s}.atr") for s in segments]
    return fetch_many(jobs, workers, "icentia11k atr")


def icentia11k_dat(records_file, workers):
    raw = ROOT / "icentia11k" / "raw"
    segments = Path(records_file).read_text().split()
    jobs = [(f"{ICENTIA_BASE}/{s}.{ext}", raw / f"{s}.{ext}") for s in segments for ext in ("hea", "dat")]
    return fetch_many(jobs, workers, "icentia11k dat")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["cinc2017", "icentia11k_atr", "icentia11k_dat"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--records", help="for icentia11k_dat: file listing the segments to fetch")
    args = parser.parse_args()
    if args.what == "cinc2017":
        cinc2017()
    elif args.what == "icentia11k_atr":
        sys.exit(1 if icentia11k_atr(args.workers) else 0)
    else:
        sys.exit(1 if icentia11k_dat(args.records, args.workers) else 0)
