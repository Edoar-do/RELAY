"""Checks that the container can run the ablation: packages, GPU, codebook, data paths, model build."""
import os
import sys
from pathlib import Path

CODE_DIR = str(Path(__file__).resolve().parents[2] / "relay_code")
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

import json
import torch
import torch_geometric
import torch_scatter
import sklearn
import transformers
import wandb

print(f"python {sys.version.split()[0]} | torch {torch.__version__} | pyg {torch_geometric.__version__} "
      f"| sklearn {sklearn.__version__} | transformers {transformers.__version__} | wandb {wandb.__version__}")

assert torch.cuda.is_available(), "CUDA not available"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 2**30:.0f} GiB")

from utils import retrieve_kmeans_models
from config import MODEL_NAME_2_CONFIG
from modelling_graph_ecg import MODEL_NAME_2_PRETRAINING_MODEL, MODEL_NAME_2_FINETUNING_MODEL

kmeans = retrieve_kmeans_models(os.path.join(CODE_DIR, "kmeans_path.txt"))
assert [k.n_clusters for k in kmeans] == [50], f"expected the C=50 Stage-1 codebook, got {[k.n_clusters for k in kmeans]}"
print("codebook: C=50 loaded")

with open(Path(__file__).resolve().parents[2] / "ablations" / "config_gat.json") as f:
    config = MODEL_NAME_2_CONFIG["gat"](**json.load(f))
model = MODEL_NAME_2_PRETRAINING_MODEL["gat"](config)
n_params = sum(p.numel() for p in model.parameters())
print(f"pretraining model: {n_params} params (original run: 7140096)")
MODEL_NAME_2_FINETUNING_MODEL["gat"](config)

DATA = os.environ.get("DATA", "data")
for path in [DATA + "/ssl_train.csv", DATA + "/ssl_val.csv",
             DATA + "/grouped_features_ssl_train", DATA + "/grouped_features_ssl_val",
             DATA + "/ribeiro_dev_train.csv", DATA + "/ribeiro_dev_val.csv",
             DATA + "/ECG_pretraining/models/any_lead/pretrained",
             DATA + "/ECG_pretraining/models/any_lead/finetuned"]:
    assert os.path.exists(path), f"missing: {path}"
print("data paths: ok")

if not (os.environ.get("WANDB_API_KEY") or os.path.exists(os.path.expanduser("~/.netrc"))):
    print("WARNING: no wandb credentials (set WANDB_API_KEY or run `wandb login`)")

print("ENV OK")
