# lib/log_summary.py

import re
import glob
import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np

NUM = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"

# Example:
# INFO | Test - Best epoch 2, Loss, MSE, RMSE, MAE: 0.08482, 0.08482, 0.28930, 0.19062
TEST_LINE_RE_RMSE = re.compile(
    r"Test\s*-\s*Best epoch\s+(?P<best_epoch>\d+)\s*,\s*"
    r"Loss\s*,\s*MSE\s*,\s*RMSE\s*,\s*MAE:\s*"
    rf"(?P<loss>{NUM})\s*,\s*(?P<mse>{NUM})\s*,\s*(?P<rmse>{NUM})\s*,\s*(?P<mae>{NUM})"
)

# Epoch line:
# INFO | - Epoch 002, ExpID 12623
EPOCH_LINE_RE = re.compile(r"-\s*Epoch\s+(?P<epoch>\d+)\b")


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()


def extract_final_best_from_file(train_log_path: str) -> Optional[Dict[str, float]]:
    """
    Returns the final best metrics for ONE file using rule:
      - only accept test line when current_epoch == best_epoch
      - keep overwriting -> last accepted = final best
    """
    lines = _read_lines(train_log_path)

    current_epoch: Optional[int] = None
    best: Optional[Dict[str, float]] = None

    for line in lines:
        m_epoch = EPOCH_LINE_RE.search(line)
        if m_epoch:
            current_epoch = int(m_epoch.group("epoch"))

        m_test = TEST_LINE_RE_RMSE.search(line)
        if m_test:
            best_epoch = int(m_test.group("best_epoch"))
            if current_epoch is None or current_epoch != best_epoch:
                continue  # ignore "best epoch refers to earlier epoch" lines

            # update best (overwrite old)
            best = {
                "best_epoch": best_epoch,

                "loss": float(m_test.group("loss")),
                "mse": float(m_test.group("mse")),
                "rmse": float(m_test.group("rmse")),
                "mae": float(m_test.group("mae")),
            }

    return best


def print_per_file_results(files: List[str], scaled: bool = True) -> None:
    """
    Print one final result per file.
    If scaled=True prints paper format:
      - MSE (×1e-3)
      - MAE (×1e-2)
    """
    results = []
    missing = []

    for fp in files:
        d = extract_final_best_from_file(fp)
        if d is None:
            missing.append(fp)
            continue
        results.append((fp, d))

    print(f"\nFound .train.log files: {len(files)}")
    print(f"Valid results: {len(results)}")
    if missing:
        print(f"[WARN] No valid (epoch==best_epoch) record found in {len(missing)} file(s).")

    print("\n--- Final best per file ---")
    for fp, d in results:
        rel = os.path.relpath(fp)

        if scaled:
            mse_s = d["mse"] * 1e3
            mae_s = d["mae"] * 1e2
            print(
                f"{rel} | "
                f"MSE (×1e-3)={mse_s:.3f} | MAE (×1e-2)={mae_s:.3f}"
            )
        else:
            print(
                f"{rel} | best_epoch={d['best_epoch']:4d} | "
                f"mse={d['mse']:.6f} | rmse={d['rmse']:.6f} | mae={d['mae']:.6f} | loss={d['loss']:.6f}"
            )


def main():
    ap = argparse.ArgumentParser("Print final best metrics per .train.log file (no global averaging)")
    ap.add_argument(
        "--glob",
        required=True,
        help='Glob for logs, e.g. "**/*.train.log" or "analyzelogs/**/*.train.log"',
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Print raw mse/mae/rmse/loss instead of scaled paper format.",
    )
    args = ap.parse_args()

    matched = sorted(glob.glob(args.glob, recursive=True))

    # hard filter: ONLY *.train.log
    files = [f for f in matched if f.endswith(".train.log")]

    if not files:
        raise ValueError(f"No .train.log files matched: {args.glob}")

    print_per_file_results(files, scaled=(not args.raw))


if __name__ == "__main__":
    main()