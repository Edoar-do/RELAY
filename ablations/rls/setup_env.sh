#!/bin/bash
# Creates the RELAY training environment inside a container (local disk), pinned to the versions
# recorded by wandb for the original Stage-1 run (Python 3.8, torch 2.0.1+cu118, PyG 2.6.1, sklearn 1.3.0).
# Download caches live on the shared FS, so the 2nd and 3rd containers install much faster.
#
# Usage:  bash setup_env.sh            # env goes to ${HOME}/relay_env (override with ENV_DIR=...)

set -euo pipefail

ENV_DIR=${ENV_DIR:-${HOME}/relay_env}
CACHE=${CACHE:-${HOME}/.cache}
export PIP_CACHE_DIR=$CACHE/pip
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}
export CONDA_PKGS_DIRS=$CACHE/conda_pkgs
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

# CA bundle on the shared FS, for images without system certificates (pip ships its own)
CA_BUNDLE=${CA_BUNDLE:-$CACHE/certs/cacert.pem}
export SSL_CERT_FILE=$CA_BUNDLE

if [[ -x "$ENV_DIR/bin/python" ]]; then
    echo "Env already exists at $ENV_DIR"
else
    # --- Python 3.8: use conda if the image has it, otherwise fetch a standalone micromamba ---
    if command -v conda >/dev/null 2>&1; then
        conda create -y -p "$ENV_DIR" python=3.8
    else
        # a static micromamba binary is cached on the shared FS (no python3/curl/bzip2 needed)
        if [[ ! -x "$MAMBA_ROOT_PREFIX/bin/micromamba" && -x "$CACHE/micromamba/bin/micromamba" ]]; then
            mkdir -p "$MAMBA_ROOT_PREFIX/bin"
            cp "$CACHE/micromamba/bin/micromamba" "$MAMBA_ROOT_PREFIX/bin/"
        fi
        if [[ ! -x "$MAMBA_ROOT_PREFIX/bin/micromamba" ]]; then
            mkdir -p "$MAMBA_ROOT_PREFIX"
            # python instead of curl/bzip2, which minimal images often lack
            python3 -c "
import io, sys, tarfile, urllib.request
data = urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest').read()
tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2').extract('bin/micromamba', sys.argv[1])
" "$MAMBA_ROOT_PREFIX"
        fi
        "$MAMBA_ROOT_PREFIX/bin/micromamba" create -y -p "$ENV_DIR" -c conda-forge --cacert-path "$CA_BUNDLE" python=3.8 pip
    fi
fi

PIP="$ENV_DIR/bin/python -m pip"
$PIP install --upgrade "pip<24.1"

# --- PyTorch (CUDA 11.8 wheels) ---
$PIP install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# --- PyG compiled extensions, matched to torch 2.0.1+cu118 ---
$PIP install torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# --- everything else (scikit-learn must stay 1.3.0: the k-means codebooks are pickled with it) ---
$PIP install \
    torch-geometric==2.6.1 \
    torchmetrics==1.5.2 \
    torcheval==0.0.7 \
    transformers==4.33.0 \
    numpy==1.24.4 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    joblib==1.3.2 \
    threadpoolctl==3.2.0 \
    loguru==0.7.0 \
    tqdm==4.66.1 \
    wandb==0.15.8 \
    sqlalchemy==2.0.20

echo
echo "Done. Verify with:  $ENV_DIR/bin/python $(dirname "$(readlink -f "$0")")/check_env.py"
