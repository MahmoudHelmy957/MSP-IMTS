#!/usr/bin/env python3
import re, sys, glob
from math import fsum, sqrt
from statistics import stdev

NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

# Detect a new seed starting (handles: "--seed 5", "seed=5", "Seed 5", "Namespace(..., seed=5, ...)")
SEED_RE = re.compile(r'(?i)(?:--seed\s+|seed\s*=\s*|Seed[:\s]+)(\d+)')

# Capture everything on the "Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE:" line
TEST_RE = re.compile(r'Test\s*-\s*Best epoch.*?:\s*(.*)$', re.I)

# Extract all numbers from that line
NUMBERS_RE = re.compile(NUM)


def iter_lines(paths):
    for p in paths:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line.rstrip('\n')


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_err.py <log_path_or_glob ...>")
        sys.exit(1)

    # Expand globs
    paths = []
    for arg in sys.argv[1:]:
        matches = glob.glob(arg)
        paths.extend(matches if matches else [arg])

    current_seed = None
    last_test_for_seed = None  # (mse, mae)
    per_seed = {}              # seed -> (mse, mae)

    def flush_current():
        nonlocal current_seed, last_test_for_seed, per_seed
        if current_seed is not None and last_test_for_seed is not None:
            per_seed[current_seed] = last_test_for_seed
        last_test_for_seed = None

    for line in iter_lines(paths):
        # New seed?
        s = SEED_RE.search(line)
        if s:
            flush_current()
            current_seed = int(s.group(1))
            continue

        # Test-Best line?
        m = TEST_RE.search(line)
        if m and current_seed is not None:
            nums = [float(x) for x in NUMBERS_RE.findall(m.group(1))]
            # Expected: [epoch, Loss, MSE, RMSE, MAE, MAPE]
            if len(nums) >= 5:
                mse = nums[2]
                mae = nums[4]
                last_test_for_seed = (mse, mae)

    # flush last seed
    flush_current()

    if not per_seed:
        print("No per-seed results found. Make sure your logs contain seed markers and 'Test - Best epoch' lines.")
        sys.exit(2)

    # Collect & stats
    seeds = sorted(per_seed.keys())
    mses = [per_seed[s][0] for s in seeds]
    maes = [per_seed[s][1] for s in seeds]
    n = len(seeds)

    avg_mse = fsum(mses) / n
    avg_mae = fsum(maes) / n

    # Sample std (ddof=1) if n>=2, else 0.0
    std_mse = stdev(mses) if n >= 2 else 0.0
    std_mae = stdev(maes) if n >= 2 else 0.0

    print("Per-seed (last Test line before next seed):")
    for s in seeds:
        mse, mae = per_seed[s]
        print(f"  seed={s}  mse={mse:.10g}  mae={mae:.10g}")

    print("\nAverages across seeds (raw):")
    print(f"  MSE: mean={avg_mse:.10g}  std={std_mse:.10g}")
    print(f"  MAE: mean={avg_mae:.10g}  std={std_mae:.10g}")

    # Paper-style scaling: MSE×1e3, MAE×1e2
    mse_scaled_mean = avg_mse * 1e3
    mse_scaled_std  = std_mse * 1e3
    mae_scaled_mean = avg_mae * 1e2
    mae_scaled_std  = std_mae * 1e2

    print("\n   MSE×10^-3    MAE×10^-2")
    print(f"  {mse_scaled_mean:.2f} ± {mse_scaled_std:.2f}    {mae_scaled_mean:.2f} ± {mae_scaled_std:.2f}")

if __name__ == "__main__":
    main()
