#!/usr/bin/env python
"""
Quick sparsity check for all IMTS datasets.

Run from project root:
    source venv310/bin/activate
    PYTHONPATH="$PWD:$PWD/tPatchGNN" python scripts/check_sparsity.py
"""

import os
import sys
import numpy as np
import torch

# -------------------------------------------------------------------------
# Make sure we can import lib.*
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.physionet import PhysioNet
from lib.mimic import MIMIC
from lib.person_activity import PersonActivity
from lib.ushcn import USHCN


# -------------------------------------------------------------------------
# Generic sparsity computation on a list of (record_id, tt, vals, mask)
# -------------------------------------------------------------------------
def compute_sparsity_from_records(records, name: str) -> None:
    """
    records: list of (record_id, tt, vals, mask)
      tt   : (T,)
      vals : (T, D)
      mask : (T, D) in {0,1}
    """
    n_series = len(records)
    if n_series == 0:
        print(f"=== {name} ===")
        print("No records found.\n")
        return

    _, tt0, vals0, mask0 = records[0]
    n_dim = vals0.shape[1]

    total_elems = 0.0
    total_obs = 0.0
    total_timesteps = 0.0

    obs_per_dim = torch.zeros(n_dim, dtype=torch.float64)
    total_per_dim = torch.zeros(n_dim, dtype=torch.float64)

    for _, tt, vals, mask in records:
        m = mask.to(torch.float64)

        total_elems += m.numel()
        total_obs += m.sum().item()
        total_timesteps += tt.numel()

        # per-dimension stats
        obs_per_dim += m.sum(dim=0).cpu()
        total_per_dim += torch.ones_like(m, dtype=torch.float64).sum(dim=0).cpu()

    overall_obs_ratio = total_obs / total_elems
    overall_sparsity = 1.0 - overall_obs_ratio
    avg_len = total_timesteps / n_series

    per_dim_sparsity = 1.0 - (obs_per_dim.numpy() / total_per_dim.numpy())
    pd = per_dim_sparsity

    print(f"=== {name} ===")
    print(f"# series: {n_series}")
    print(f"Avg length (time points per series): {avg_len:.2f}")
    print(f"Overall observed ratio: {overall_obs_ratio:.4f}")
    print(f"Overall sparsity (missing ratio): {overall_sparsity:.4f}")
    print(
        "Per-variable sparsity: "
        f"min={pd.min():.4f}, median={np.median(pd):.4f}, max={pd.max():.4f}"
    )
    print()


# -------------------------------------------------------------------------
# Dataset-specific wrappers
# -------------------------------------------------------------------------
def physionet_stats():
    root = os.path.join(PROJECT_ROOT, "data", "physionet")
    # IMPORTANT: quantization must match your processed filenames (set-a_1.0.pt, etc.)
    ds = PhysioNet(
        root,
        quantization=1.0,
        download=False,          # you already have the .pt files
        n_samples=None,
        device=torch.device("cpu"),
    )
    compute_sparsity_from_records(ds.data, "PhysioNet")


def mimic_stats():
    root = os.path.join(PROJECT_ROOT, "data", "mimic")
    ds = MIMIC(
        root,
        n_samples=None,
        device=torch.device("cpu"),
    )
    compute_sparsity_from_records(ds.data, "MIMIC")


def activity_stats():
    root = os.path.join(PROJECT_ROOT, "data", "activity")
    ds = PersonActivity(
        root,
        n_samples=None,
        download=False,          # set True only if you really need to re-download
        device=torch.device("cpu"),
    )
    compute_sparsity_from_records(ds.data, "Human Activity")


def ushcn_stats():
    root = os.path.join(PROJECT_ROOT, "data", "ushcn")
    ds = USHCN(
        root,
        n_samples=None,
        device=torch.device("cpu"),
    )
    compute_sparsity_from_records(ds.data, "USHCN")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in [physionet_stats, mimic_stats, activity_stats, ushcn_stats]:
        try:
            fn()
        except Exception as e:
            print(f"Skipping {fn.__name__} due to error: {e}\n")
