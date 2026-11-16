"""Builds the RLS x lin-eval-K table from the lin-eval logs: best validation macro AUROC over all
validations (Ribeiro-dev has no test split), plus the step at which it was reached."""
import re
import sys
from pathlib import Path

LOG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "logs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"Validating the model at step (\d+)")
AUROC = re.compile(r"Validation auroc \(macro\) = ([0-9.]+)")


def best_auroc(log_file):
    best, best_step, step, n_val = None, None, None, 0
    for line in log_file.read_text(errors="ignore").splitlines():
        line = ANSI.sub("", line)
        if m := STEP.search(line):
            step = int(m.group(1))
        elif m := AUROC.search(line):
            n_val += 1
            if best is None or float(m.group(1)) > best:
                best, best_step = float(m.group(1)), step
    return best, best_step, n_val


CONFIGS = ("K12", "K1", "K2", "leadI", "leadII")
print(f"{'pretraining':<12}" + "".join(f"{c:>22}" for c in CONFIGS))
for arm in ("rls", "rls_zeropad", "norls", "norls_tok", "norls_pm0217"):
    row = f"{arm:<12}"
    for c in CONFIGS:
        log_file = LOG_DIR / f"lineval_stage1_{arm}_{c}.log"
        if not log_file.exists():
            row += f"{'—':>22}"
            continue
        best, step, n_val = best_auroc(log_file)
        row += f"{'n/a' if best is None else f'{best:.4f} @{step} ({n_val}v)':>22}"
    print(row)

# --- lin-eval seeds: seed 42 = the runs above (validation lead seed 0); others from run_seed_queue.sh ---
# Only finished runs (14 validations) count. Mean +- sample std over seeds; the seeds are paired across
# arms (same seed -> same validation lead draws), so arm differences are also given per seed.
ARMS = ("rls", "rls_zeropad", "norls", "norls_tok", "norls_pm0217")
seed_files = sorted(LOG_DIR.glob("lineval_stage1_*_s[0-9]*.log"))
if seed_files:
    import statistics
    print("\nfinished runs per seed (best val macro AUROC)")
    res = {}
    for arm in ARMS:
        for c in CONFIGS:
            files = {42: LOG_DIR / f"lineval_stage1_{arm}_{c}.log"}
            files.update({int(f.stem.rsplit("_s", 1)[1]): f for f in LOG_DIR.glob(f"lineval_stage1_{arm}_{c}_s[0-9]*.log")})
            for seed, f in files.items():
                if f.exists():
                    best, _, n_val = best_auroc(f)
                    if best is not None and n_val == 14:
                        res[arm, c, seed] = best
    for arm in ARMS:
        for c in CONFIGS:
            seeds = sorted(s for (a, cc, s) in res if a == arm and cc == c)
            if len(seeds) < 2:
                continue
            vals = [res[arm, c, s] for s in seeds]
            print(f"{arm:<13}{c:<7}" + "  ".join(f"s{s}={v:.4f}" for s, v in zip(seeds, vals))
                  + f"   mean {statistics.mean(vals):.4f} +- {statistics.stdev(vals):.4f} (n={len(vals)})")
    print("\npaired differences (arm - rls), same seed")
    for arm in ARMS[1:]:
        for c in CONFIGS:
            seeds = sorted(s for (a, cc, s) in res if a == arm and cc == c and ("rls", c, s) in res)
            if len(seeds) < 2:
                continue
            d = [res[arm, c, s] - res["rls", c, s] for s in seeds]
            print(f"{arm:<13}{c:<7}" + "  ".join(f"s{s}={v:+.4f}" for s, v in zip(seeds, d))
                  + f"   mean {statistics.mean(d):+.4f} +- {statistics.stdev(d):.4f} (n={len(d)})")
