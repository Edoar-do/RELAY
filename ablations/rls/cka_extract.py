"""Extract pooled graph embeddings (the linear probe's input) of the Stage-1 encoders on the
Ribeiro-dev validation split, for CKA between full-lead and reduced-lead representations.

For every arm, the encoder (frozen; weights from the saved 12-lead probe checkpoint, identical
to the pretrained encoder) embeds the same 4,819 records under:
    K12                                   all leads (reference; drop == pad at 12 leads)
    K1 / K2  x lead seeds 0, 43, 44       random leads, drawn once per record (as eval_zeroshot.py)
    leadI / leadII
each reduced setting with the missing leads either DROPPED from the graph (variable graph)
or ZERO-PADDED (full 12-lead graph, all-zero signals; ECGDataset(pad_unsampled_leads=True)).
No training of any kind. The embedding is the input of the probe's output layer (graph
pooling of the encoder output), captured with a forward hook.

Usage:  python cka_extract.py [--arms rls norls ...]
Output: cka/embeddings_<arm>.npz  (keys: K12, K1_s0_drop, K1_s0_pad, ..., leadII_pad)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import eval_zeroshot as ez  # chdirs into the code dir and imports the RELAY modules
from torch.utils.data import DataLoader

OUT = ez.ABL_DIR / "cka"


@torch.no_grad()
def embed(model, cfg, setting, lead_seed, pad, device, batch_size=64, num_workers=4):
    sampling, k, _ = ez.SETTINGS[setting]
    torch.manual_seed(42); np.random.seed(42)
    val_set = ez.build_val_set(cfg, sampling, k, lead_seed, pad)
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, collate_fn=val_set.collate_fn, worker_init_fn=ez.worker_init_fn)
    _, crit = ez.get_criterions("multi_label", train_weigths=val_set.weights, val_weights=val_set.weights)
    crit.to(device)
    metrics = ez.get_metrics("multi_label", num_classes=cfg.num_classes)
    for m in metrics.values():
        m.to(device); m.reset()
    feats = []
    hook = model.output_layer.register_forward_hook(lambda mod, inp, out: feats.append(inp[0].detach().float().cpu()))
    try:
        ez.validate(argparse.Namespace(batch_size=batch_size), device, model, loader, crit, metrics, [])
    finally:
        hook.remove()
    X = torch.cat(feats).numpy()
    assert X.shape[0] == len(val_set), (X.shape, len(val_set))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ez.ARMS)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda")
    for arm in args.arms:
        out_path = OUT / f"embeddings_{arm}.npz"
        if out_path.exists():
            print(f"[{arm}] exists, skipping: {out_path}"); continue
        ckpt = torch.load(ez.FINETUNED_DIR / f"gat_best_val_loss_lineval_stage1_{arm}_K12.pt", map_location="cpu")
        cfg = ckpt["model_config"]
        model = ez.MODEL_NAME_2_FINETUNING_MODEL["gat"](cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        E = {"K12": embed(model, cfg, "K12", 0, False, device)}
        print(f"[{arm}] K12 {E['K12'].shape}", flush=True)
        for setting in ("K1", "K2", "leadI", "leadII"):
            for s in (ez.LEAD_SEEDS if setting in ("K1", "K2") else [None]):
                for pad in (False, True):
                    key = f"{setting}{'' if s is None else f'_s{s}'}_{'pad' if pad else 'drop'}"
                    E[key] = embed(model, cfg, setting, s, pad, device)
                    print(f"[{arm}] {key} {E[key].shape}", flush=True)
        np.savez(out_path, **E)
        print(f"[{arm}] saved {out_path}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
