"""Zero-shot reduced-lead evaluation of the 12-lead linear probes (Stage-1 RLS ablation).

The original ablation (lineval_ribeiro.sh) trains AND validates the probe on K leads. The
paper's claim is zero-shot transfer: train the probe once on 12 leads, then evaluate it on
fewer leads. This script takes the saved K=12 probes
    $FINETUNED_DIR/gat_best_val_loss_lineval_stage1_<arm>_K12.pt
(best validation loss at 12 leads = model selection at L=12 only, no reduced-lead
information) and evaluates each, unchanged, on the Ribeiro-dev validation split at:
    K12                 all 12 leads (sanity check: must reproduce the logged value)
    K1, K2              K random leads, drawn once per record with --val_lead_seed
                        0 / 43 / 44 = the seeds of the original lin-eval runs 42 / 43 / 44
    leadI, leadII       that lead only
Validation set, graph construction, loss and AUROC metric are exactly finetune.py's
(same ECGDataset arguments, same validate(), macro = mean of per-class AUROC).

With --pad, the leads that are not selected are zero-padded at inference time (full 12-lead
graph, ECGDataset(pad_unsampled_leads=True)) instead of dropped from the graph. At K12 the two
are identical, so the same 12-lead probe is tested under both representations of missing
leads without any training on padded inputs (no adaptation). Output: zeroshot_pad/.

With --probes, probes trained on other lead configurations are evaluated too, e.g. the
K1 / K2 / leadI / leadII probes of lineval_ribeiro.sh
    $FINETUNED_DIR/gat_best_val_loss_lineval_stage1_<arm>_<probe>.pt
on every test setting (a probe tested on a setting it was not trained on is zero-shot). With
--modes drop pad, both representations of missing leads are evaluated in one call. The
rls_zeropad arm (pretrained and probed with zero-padded unsampled leads, i.e. always the full
12-subgraph topology with "null" subgraphs) is tested this way by run_zeropad_ablation.sh.

Usage (from any directory; uses ${HOME}/relay_env like the rest of rls_stage1):
    python eval_zeroshot.py                       # all arms, all settings
    python eval_zeroshot.py --arms rls --settings K12 K1
    python eval_zeroshot.py --arms rls rls_zeropad --probes K12 K1 K2 leadI leadII \
        --modes drop pad --out zeroshot_zeropad
Output: zeroshot/results.csv (one row per arm x probe x missing-lead mode x setting x lead seed),
zeroshot/table.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = str(Path(__file__).resolve().parents[2] / "relay_code")
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)  # finetune.py is always run from here (ecg_val_dir_path="./")

from torch.utils.data import DataLoader  # noqa: E402
from dataset import ECGDataset  # noqa: E402
from modelling_graph_ecg import MODEL_NAME_2_FINETUNING_MODEL  # noqa: E402
from utils import get_criterions, get_metrics  # noqa: E402
from finetune import validate, worker_init_fn  # noqa: E402

ABL_DIR = Path(__file__).resolve().parent
FINETUNED_DIR = Path(os.environ.get("DATA", "data")) / "ECG_pretraining/models/any_lead/finetuned"
VAL_CSV = os.path.join(os.environ.get("DATA", "data"), "ribeiro_dev_val.csv")
ARMS = ["rls", "rls_zeropad", "norls_pm0217", "norls_tok", "norls"]
LEAD_SEEDS = [0, 43, 44]
SETTINGS = {  # name -> (sampling, how_many_leads, random?)
    "K12": ("random", 12, True),
    "K1": ("random", 1, True),
    "K2": ("random", 2, True),
    "leadI": (["I"], None, False),
    "leadII": (["II"], None, False),
}


def build_val_set(model_config, sampling, k, lead_seed, pad=False):
    # identical to the val_set of finetune.py (CSV branch) with lineval_ribeiro.sh's arguments
    return ECGDataset(
        path_to_dataset_csv=VAL_CSV, ecg_dir_path="./", features_path=None,
        downsampling_factor=5, label_start_index=3,
        nodes_per_lead=model_config.nodes_per_lead, pretrain=False, random_crop=False,
        return_full_length=False, create_edge_index=True, edge_strategy="spatiotemporal",
        sampling=sampling, how_many_leads=k,
        use_temporal_encoding=model_config.use_temporal_encoding,
        use_lead_encoding=model_config.use_lead_encoding,
        fixed_lead_seed=lead_seed,
        pad_unsampled_leads=pad,
    )


def logged_val_auroc(arm: str, step: int, probe: str = "K12") -> float:
    """Validation macro AUROC printed by finetune.py at `step` in the lin-eval run of `probe`."""
    log = ABL_DIR / "logs" / f"lineval_stage1_{arm}_{probe}.log"
    if not log.exists():
        return float("nan")
    cur, out = None, {}
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"Validating the model at step (\d+)", line)
        if m:
            cur = int(m.group(1))
        m = re.search(r"Validation auroc \(macro\) = (?:tensor\()?([0-9.]+)", line)
        if m and cur is not None:
            out[cur] = float(m.group(1))
    return out.get(step, float("nan"))


@torch.no_grad()
def evaluate(model, model_config, setting, lead_seed, device, batch_size, num_workers, pad=False):
    sampling, k, _ = SETTINGS[setting]
    torch.manual_seed(42); np.random.seed(42)
    val_set = build_val_set(model_config, sampling, k, lead_seed, pad)
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, collate_fn=val_set.collate_fn, worker_init_fn=worker_init_fn)
    _, val_criterion = get_criterions("multi_label", train_weigths=val_set.weights,
                                      val_weights=val_set.weights, label_smoothing=0.0)
    val_criterion.to(device)
    metrics = get_metrics("multi_label", num_classes=model_config.num_classes)
    for m in metrics.values():
        m.to(device); m.reset()
    losses = validate(argparse.Namespace(batch_size=batch_size), device, model, loader,
                      val_criterion, metrics, [])
    per_class = metrics["auroc"].compute().detach().cpu().numpy()
    return float(per_class.mean()), per_class, float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--probes", nargs="+", default=["K12"],
                    help="lin-eval runs whose probes are evaluated, e.g. K12 K1 K2 leadI leadII K1_s43")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pad", action="store_true", help="zero-pad unselected leads at inference instead of dropping them")
    ap.add_argument("--modes", nargs="+", choices=["drop", "pad"], default=None,
                    help="missing-lead representations to evaluate (default: pad if --pad else drop)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.modes is None:
        args.modes = ["pad"] if args.pad else ["drop"]
    if args.out is None:
        args.out = ABL_DIR / ("zeroshot_pad" if args.modes == ["pad"] else "zeroshot")
    if not args.out.is_absolute():
        args.out = ABL_DIR / args.out  # cwd is CODE_DIR (see top of file)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    classes = list(pd.read_csv(VAL_CSV, nrows=0).columns[3:])

    rows = []
    for arm in args.arms:
        for probe in args.probes:
            ckpt_path = FINETUNED_DIR / f"gat_best_val_loss_lineval_stage1_{arm}_{probe}.pt"
            if not ckpt_path.exists():
                print(f"[{arm}] {ckpt_path.name}: not found, skipped", flush=True)
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt["model_config"]
            model = MODEL_NAME_2_FINETUNING_MODEL["gat"](cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            step = int(ckpt["global_step"])
            # the probe's own lin-eval validation (same setting, same missing-lead mode, lead seed 0)
            # must be reproduced exactly; only checked for the seed-42 runs
            trained_mode = "pad" if arm == "rls_zeropad" else "drop"
            print(f"[{arm}] {ckpt_path.name}: probe from step {step}", flush=True)
            for mode in args.modes:
                pad = mode == "pad"
                for setting in args.settings:
                    seeds = LEAD_SEEDS if setting in ("K1", "K2") else ([LEAD_SEEDS[0]] if setting == "K12" else [None])
                    for s in seeds:
                        auroc, per_class, loss = evaluate(model, cfg, setting, s, device, args.batch_size, args.num_workers, pad)
                        row = dict(arm=arm, probe=probe, setting=setting, lead_seed=s,
                                   missing_leads="zero-padded" if pad else "dropped",
                                   macro_auroc=auroc, val_loss=loss,
                                   probe_step=step, ckpt=ckpt_path.name,
                                   **{f"auroc_{c}": v for c, v in zip(classes, per_class)})
                        check = setting == probe and s in (0, None) and (mode == trained_mode or setting == "K12")
                        if check:
                            row["logged_auroc_at_step"] = logged_val_auroc(arm, step, probe)
                        rows.append(row)
                        print(f"  {mode:4s} {setting:7s} lead_seed={s}: macro AUROC {auroc:.4f}"
                              + (f"  (logged at step {step}: {row['logged_auroc_at_step']:.4f})" if check else ""),
                              flush=True)
            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "results.csv", index=False)

    keys = ["arm", "probe", "missing_leads", "setting"]
    agg = df.groupby(keys).macro_auroc.agg(["mean", "std", "count"]).reset_index()
    order = [s for s in SETTINGS if s in set(df.setting)]
    lines = ["K1/K2: mean ± sample std over lead-sampling seeds " + ", ".join(map(str, LEAD_SEEDS))
             + "; other columns are deterministic. probe = lead configuration the linear probe was trained on.",
             "",
             "| arm | probe | missing leads | " + " | ".join(order) + " |",
             "|---" * (len(order) + 3) + "|"]
    for arm in [a for a in ARMS if a in set(df.arm)] + sorted(set(df.arm) - set(ARMS)):
        for probe in args.probes:
            for mode in ("dropped", "zero-padded"):
                sub = agg[(agg.arm == arm) & (agg.probe == probe) & (agg.missing_leads == mode)]
                if sub.empty:
                    continue
                cells = []
                for s in order:
                    r = sub[sub.setting == s]
                    if r.empty:
                        cells.append("–"); continue
                    r = r.iloc[0]
                    cells.append(f"{r['mean']:.4f}" + (f" ± {r['std']:.4f}" if r["count"] > 1 else ""))
                lines.append(f"| {arm} | {probe} | {mode} | " + " | ".join(cells) + " |")
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
