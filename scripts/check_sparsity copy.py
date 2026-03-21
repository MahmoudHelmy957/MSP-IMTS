# scripts/check_sparsity.py

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "tPatchGNN"))
import torch
from sklearn import model_selection

from lib.physionet import PhysioNet
from lib.mimic import MIMIC
from lib.person_activity import PersonActivity
from lib.ushcn import USHCN


device = torch.device("cpu")


def compute_sparsity(dataset):
    """
    dataset: iterable of (record_id, tt, vals, mask)
    returns:
      global_density, global_sparsity, per_dim_density (1D tensor)
    """
    total_obs = 0
    total_entries = 0

    per_dim_obs = None
    per_dim_tot = None

    for _, _, vals, mask in dataset:
        # mask shape: (T, D), 0/1 or float. Convert to bool.
        m = (mask > 0).to(torch.bool)

        # global counts
        total_obs += m.sum().item()
        total_entries += m.numel()

        # per-dimension counts
        # obs_per_dim: how many times this variable is present over time
        obs_per_dim = m.sum(dim=0)                 # (D,)
        entries_per_dim = torch.full_like(obs_per_dim, m.shape[0])

        if per_dim_obs is None:
            per_dim_obs = obs_per_dim.clone()
            per_dim_tot = entries_per_dim.clone()
        else:
            per_dim_obs += obs_per_dim
            per_dim_tot += entries_per_dim

    global_density = total_obs / total_entries
    global_sparsity = 1.0 - global_density
    per_dim_density = (per_dim_obs / per_dim_tot).cpu()

    return global_density, global_sparsity, per_dim_density


def report_for_split(name, split_name, dataset):
    dens, spars, per_dim = compute_sparsity(dataset)
    print(f"[{name} | {split_name}]")
    print(f"  Global density : {dens:.4f}")
    print(f"  Global sparsity: {spars:.4f}")
    print(f"  Per-dim density (first 10 dims): {per_dim[:10].tolist()}")
    print()


def physionet_stats():
    print("=== PhysioNet ===")
    ds = PhysioNet("../data/physionet", quantization=1.0,
                   download=False, n_samples=None, device=device)
    seen, test = model_selection.train_test_split(
        ds, train_size=0.8, random_state=42, shuffle=True
    )
    train, val = model_selection.train_test_split(
        seen, train_size=0.75, random_state=42, shuffle=False
    )
    report_for_split("PhysioNet", "train", train)
    report_for_split("PhysioNet", "val",   val)
    report_for_split("PhysioNet", "test",  test)


def mimic_stats():
    print("=== MIMIC ===")
    ds = MIMIC("../data/mimic/", n_samples=None, device=device)
    seen, test = model_selection.train_test_split(
        ds, train_size=0.8, random_state=42, shuffle=True
    )
    train, val = model_selection.train_test_split(
        seen, train_size=0.75, random_state=42, shuffle=False
    )
    report_for_split("MIMIC", "train", train)
    report_for_split("MIMIC", "val",   val)
    report_for_split("MIMIC", "test",  test)


def activity_stats():
    print("=== Human Activity ===")
    ds = PersonActivity("../data/activity/", n_samples=None,
                        download=True, device=device)
    seen, test = model_selection.train_test_split(
        ds, train_size=0.8, random_state=42, shuffle=True
    )
    train, val = model_selection.train_test_split(
        seen, train_size=0.75, random_state=42, shuffle=False
    )
    report_for_split("Activity", "train", train)
    report_for_split("Activity", "val",   val)
    report_for_split("Activity", "test",  test)


def ushcn_stats():
    print("=== USHCN ===")
    ds = USHCN("../data/ushcn/", n_samples=None, device=device)
    seen, test = model_selection.train_test_split(
        ds, train_size=0.8, random_state=42, shuffle=True
    )
    train, val = model_selection.train_test_split(
        seen, train_size=0.75, random_state=42, shuffle=False
    )
    report_for_split("USHCN", "train", train)
    report_for_split("USHCN", "val",   val)
    report_for_split("USHCN", "test",  test)


if __name__ == "__main__":
    physionet_stats()
    mimic_stats()
    activity_stats()
    ushcn_stats()
