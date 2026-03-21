#!/usr/bin/env python3
import re, sys, glob, os
from math import fsum
from statistics import stdev

NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

SEED_RE = re.compile(r'(?i)(?:--seed\s+|seed\s*=\s*|Seed[:\s]+)(\d+)')
TEST_RE = re.compile(r'Test\s*-\s*Best epoch.*?:\s*(.*)$', re.I)
NUMBERS_RE = re.compile(NUM)


def parse_file(path):
    current_seed = None
    last_test_for_seed = None
    per_seed = {}

    def flush_current():
        nonlocal current_seed, last_test_for_seed
        if current_seed is not None and last_test_for_seed is not None:
            per_seed[current_seed] = last_test_for_seed
        last_test_for_seed = None

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')

            s = SEED_RE.search(line)
            if s:
                flush_current()
                current_seed = int(s.group(1))
                continue

            m = TEST_RE.search(line)
            if m and current_seed is not None:
                nums = [float(x) for x in NUMBERS_RE.findall(m.group(1))]
                # Expected: [epoch, Loss, MSE, RMSE, MAE, MAPE]
                if len(nums) >= 5:
                    mse = nums[2]
                    mae = nums[4]
                    last_test_for_seed = (mse, mae)

    flush_current()
    return per_seed


def mean_std(values):
    n = len(values)
    mean = fsum(values) / n
    std = stdev(values) if n >= 2 else 0.0
    return mean, std


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_err.py <log_path_or_glob ...>")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        matches = sorted(glob.glob(arg))
        paths.extend(matches if matches else [arg])

    if not paths:
        print("No files matched.")
        sys.exit(1)

    print("Matched files:")
    for p in paths:
        print(" ", p)

    #overall_mses = []
    #overall_maes = []

    found_any = False

    for path in paths:
        per_seed = parse_file(path)

        if not per_seed:
            print(f"\n=== {os.path.basename(path)} ===")
            print("No per-seed results found.")
            continue

        found_any = True
        seeds = sorted(per_seed.keys())
        mses = [per_seed[s][0] for s in seeds]
        maes = [per_seed[s][1] for s in seeds]

        #overall_mses.extend(mses)
        #overall_maes.extend(maes)

        avg_mse, std_mse = mean_std(mses)
        avg_mae, std_mae = mean_std(maes)

        print(f"\n=== {os.path.basename(path)} ===")
        print("Per-seed:")
        for s in seeds:
            mse, mae = per_seed[s]
            print(f"  seed={s}  mse={mse:.10g}  mae={mae:.10g}")

        print("\nAverages across seeds (raw):")
        print(f"  MSE: mean={avg_mse:.10g}  std={std_mse:.10g}")
        print(f"  MAE: mean={avg_mae:.10g}  std={std_mae:.10g}")

        print("\n   MSE×10^-2    MAE×10^-2")
        print(f"  {avg_mse * 1e2:.2f} ± {std_mse * 1e2:.2f}    {avg_mae * 1e2:.2f} ± {std_mae * 1e2:.2f}")
    
    if not found_any:
        print("\nNo per-seed results found in any matched file.")
        sys.exit(2)

    # overall_avg_mse, overall_std_mse = mean_std(overall_mses)
    # overall_avg_mae, overall_std_mae = mean_std(overall_maes)

    # print("\n================ OVERALL ACROSS ALL FILES ================")
    # print(f"  MSE: mean={overall_avg_mse:.10g}  std={overall_std_mse:.10g}")
    # print(f"  MAE: mean={overall_avg_mae:.10g}  std={overall_std_mae:.10g}")
    # print("\n   MSE×10^-3    MAE×10^-2")
    # print(f"  {overall_avg_mse * 1e3:.2f} ± {overall_std_mse * 1e3:.2f}    {overall_avg_mae * 1e2:.2f} ± {overall_std_mae * 1e2:.2f}")


if __name__ == "__main__":
    main()