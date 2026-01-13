import torch
import numpy as np


def _init_channel_arrays(C):
    return {
        "min": torch.full((C,), float("inf")),
        "max": torch.full((C,), float("-inf")),
        "sum": torch.zeros(C),
        "sqsum": torch.zeros(C),
        "count": torch.zeros(C),
    }


def analyze_activity_pt_varlen(path="data.pt", assume_tuple_order=True):
    """
    Supports Activity-style processed .pt where obj is a list of variable-length samples.

    Expected each item:
      (id, tt, vals, mask)  OR  (tt, vals, mask) depending on your preprocessing.

    vals: (T, C)
    mask: (T, C) with {0,1} or bool
    tt:   (T,) or (T,1)
    """

    obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, (list, tuple)):
        raise ValueError(f"Expected list/tuple of samples, got {type(obj)}")

    # We will infer C from first sample
    first = obj[0]
    if isinstance(first, (list, tuple)):
        if len(first) >= 4:
            _, tt0, v0, m0 = first[:4]
        elif len(first) == 3:
            tt0, v0, m0 = first
        else:
            raise ValueError(f"Unexpected sample tuple length: {len(first)}")
    else:
        raise ValueError("Samples are not tuples/lists; please adapt loader.")

    C = int(v0.shape[-1])

    ch = _init_channel_arrays(C)

    # global accumulators
    g_min = float("inf")
    g_max = float("-inf")
    g_sum = 0.0
    g_sqsum = 0.0
    g_count = 0

    mask_total = 0
    mask_obs = 0

    lengths = []

    t_min = float("inf")
    t_max = float("-inf")
    t_sum = 0.0
    t_count = 0

    for item in obj:
        if len(item) >= 4:
            _, tt, vals, mask = item[:4]
        elif len(item) == 3:
            tt, vals, mask = item
        else:
            raise ValueError(f"Unexpected sample tuple length: {len(item)}")

        vals = vals.float()              # (T,C)
        mask = mask.bool() if mask.dtype != torch.bool else mask

        T = vals.shape[0]
        lengths.append(T)

        # ----- mask stats -----
        mask_total += mask.numel()
        mask_obs += int(mask.sum().item())

        # ----- global stats over observed values -----
        obs_vals = vals[mask]
        if obs_vals.numel() > 0:
            vmin = float(obs_vals.min().item())
            vmax = float(obs_vals.max().item())
            g_min = min(g_min, vmin)
            g_max = max(g_max, vmax)

            s = float(obs_vals.sum().item())
            ss = float((obs_vals ** 2).sum().item())
            c = int(obs_vals.numel())

            g_sum += s
            g_sqsum += ss
            g_count += c

        # ----- per-channel stats over observed values -----
        for c in range(C):
            vc = vals[:, c][mask[:, c]]
            if vc.numel() == 0:
                continue
            ch["min"][c] = torch.minimum(ch["min"][c], vc.min())
            ch["max"][c] = torch.maximum(ch["max"][c], vc.max())
            ch["sum"][c] += vc.sum()
            ch["sqsum"][c] += (vc ** 2).sum()
            ch["count"][c] += vc.numel()

        # ----- time stats (optional) -----
        if tt is not None:
            tt = tt.float().reshape(-1)
            t_min = min(t_min, float(tt.min().item()))
            t_max = max(t_max, float(tt.max().item()))
            t_sum += float(tt.sum().item())
            t_count += int(tt.numel())

    # finalize global
    g_mean = g_sum / max(g_count, 1)
    g_var = g_sqsum / max(g_count, 1) - (g_mean ** 2)
    g_std = float(np.sqrt(max(g_var, 0.0)))

    # finalize per-channel
    ch_mean = ch["sum"] / torch.clamp(ch["count"], min=1)
    ch_var = ch["sqsum"] / torch.clamp(ch["count"], min=1) - ch_mean ** 2
    ch_std = torch.sqrt(torch.clamp(ch_var, min=0.0))

    # output dict
    out = {
        "global": {
            "min": g_min,
            "max": g_max,
            "mean": float(g_mean),
            "std": float(g_std),
            "count_observed": int(g_count),
        },
        "mask": {
            "mask_ratio": float(mask_obs / max(mask_total, 1)),
            "sparsity": float(1.0 - (mask_obs / max(mask_total, 1))),
            "observed_points": int(mask_obs),
            "total_points": int(mask_total),
        },
        "sequence_length": {
            "num_sequences": int(len(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": float(np.mean(lengths)),
        },
        "channels": {
            "min": ch["min"].tolist(),
            "max": ch["max"].tolist(),
            "mean": ch_mean.tolist(),
            "std": ch_std.tolist(),
            "count_observed": ch["count"].tolist(),
        },
    }

    if t_count > 0:
        out["time"] = {
            "min": float(t_min),
            "max": float(t_max),
            "mean": float(t_sum / t_count),
            "count": int(t_count),
        }
    else:
        out["time"] = None

    return out


if __name__ == "__main__":
    stats = analyze_activity_pt_varlen("dataIR/activity/processed/data.pt")

    print("\n=== GLOBAL ===")
    for k, v in stats["global"].items():
        print(f"{k:>15}: {v}")

    print("\n=== MASK ===")
    for k, v in stats["mask"].items():
        print(f"{k:>15}: {v}")

    print("\n=== SEQ LENGTH ===")
    for k, v in stats["sequence_length"].items():
        print(f"{k:>15}: {v}")

    print("\n=== CHANNELS (first 5) ===")
    for c in range(min(5, len(stats["channels"]["min"]))):
        print(
            f"ch{c:02d}: "
            f"min={stats['channels']['min'][c]:.6f} "
            f"max={stats['channels']['max'][c]:.6f} "
            f"mean={stats['channels']['mean'][c]:.6f} "
            f"std={stats['channels']['std'][c]:.6f} "
            f"count={int(stats['channels']['count_observed'][c])}"
        )

    if stats["time"] is not None:
        print("\n=== TIME ===")
        for k, v in stats["time"].items():
            print(f"{k:>15}: {v}")
