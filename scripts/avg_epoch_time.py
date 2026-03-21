#!/usr/bin/env python
import argparse
import pathlib
import re
import statistics as stats

TIME_RE = re.compile(r"Time spent:\s*([0-9.]+)s")

def extract_times(path: pathlib.Path):
    times = []
    with path.open() as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                times.append(float(m.group(1)))
    return times

def main():
    parser = argparse.ArgumentParser(
        description="Compute average training time per epoch from tPatchGNN log files."
    )
    parser.add_argument("logs", nargs="+",
                        help="One or more log files from logs/*.log")
    parser.add_argument(
        "--drop-first", action="store_true",
        help="Drop first epoch (warm-up) when computing averages."
    )
    args = parser.parse_args()

    all_means = []

    for log_path in args.logs:
        p = pathlib.Path(log_path)
        times = extract_times(p)
        if not times:
            print(f"{p}: no 'Time spent:' lines found.")
            continue

        if args.drop_first and len(times) > 1:
            times = times[1:]

        mean_t = sum(times) / len(times)
        std_t = stats.pstdev(times) if len(times) > 1 else 0.0

        print(f"{p.name}:")
        print(f"  epochs counted   : {len(times)}")
        print(f"  avg time / epoch : {mean_t:.2f} s")
        print(f"  std time / epoch : {std_t:.2f} s")
        all_means.append(mean_t)

    if len(all_means) > 1:
        overall = sum(all_means) / len(all_means)
        print(f"\nMean over {len(all_means)} runs: {overall:.2f} s / epoch")

if __name__ == "__main__":
    main()
