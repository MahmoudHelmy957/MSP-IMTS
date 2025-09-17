#!/usr/bin/env bash
set -euo pipefail

# Paths
REPO="$HOME/Test/t-PatchGNN"
CODE="$REPO/tPatchGNN"
DATA="$REPO/data/physionet"

# Activate your venv
source "$HOME/venv310/bin/activate"

# Make sure local imports (lib/, model/) work
export PYTHONPATH="$REPO:$CODE:${PYTHONPATH-}"

# Create data dirs if missing
mkdir -p "$DATA"

# 1) Download + preprocess -> writes processed/set-{a,b,c}_1.0.pt
python - <<'PY'
import torch, os
from lib.physionet import PhysioNet

root = "../data/physionet"   # relative to tPatchGNN/
print("Preprocessing PhysioNet into:", os.path.abspath(os.path.join("..","data","physionet","processed")))
PhysioNet(root, download=True, quantization=1.0, device=torch.device("cpu"))
print("Done preprocessing.")
PY

# 2) Quick sanity check that the processed files exist
ls -lh "$DATA/processed"/set-*_1.0.pt
