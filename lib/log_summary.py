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


# -------------------------------
# NEW: dataset-aware scaling
# -------------------------------
def _get_scaling_from_filename(path: str) -> Tuple[float, float, str, str]:
    """
    Decide scaling factors based on filename/path substring match (case-insensitive).

    Returns:
      (mse_factor, mae_factor, mse_label, mae_label)

    Labels are formatted like: "MSE (×1e-3)" for printing.
    """
    p = path.lower()

    if "mimic" in p:
        mse_factor = 1e2   # show as ×1e-2 => multiply by 1e2
        mae_factor = 1e2   # show as ×1e-2 => multiply by 1e2
        return mse_factor, mae_factor, "MSE (×1e-2)", "MAE (×1e-2)"

    if "physionet" in p:
        mse_factor = 1e3   # show as ×1e-3 => multiply by 1e3
        mae_factor = 1e2   # show as ×1e-2 => multiply by 1e2
        return mse_factor, mae_factor, "MSE (×1e-3)", "MAE (×1e-2)"

    if "ushcn" in p:
        mse_factor = 1e1   # show as ×1e-1 => multiply by 1e1
        mae_factor = 1e1   # show as ×1e-1 => multiply by 1e1
        return mse_factor, mae_factor, "MSE (×1e-1)", "MAE (×1e-1)"

    # default (your current "paper scaling")
    mse_factor = 1e3
    mae_factor = 1e2
    return mse_factor, mae_factor, "MSE (×1e-3)", "MAE (×1e-2)"


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


def print_seed_summary_for_file(train_log_path: str, scaled: bool = True) -> None:
    seeds = extract_best_per_seed_from_file(train_log_path)

    rel = os.path.relpath(train_log_path)
    print(f"\n=== {rel} ===")
    print(f"Valid seeds found: {len(seeds)}")

    mse_factor, mae_factor, mse_label, mae_label = _get_scaling_from_filename(train_log_path)

    for s in seeds:
        if scaled:
            print(
                f"seed={int(s['seed_idx']):02d} | "
                f"{mse_label}={(s['mse']*mse_factor):.2f} | "
                f"{mae_label}={(s['mae']*mae_factor):.2f} | "
                f"best_epoch={int(s['best_epoch'])}"
            )
        else:
            print(
                f"seed={int(s['seed_idx']):02d} | "
                f"mse={s['mse']:.6f} | mae={s['mae']:.6f} | rmse={s['rmse']:.6f} | "
                f"loss={s['loss']:.6f} | best_epoch={int(s['best_epoch'])}"
            )

    summ = summarize_seeds(seeds)

    if scaled:
        print("\n--- Mean ± Std over seeds (dataset-aware scaling) ---")
        print(f"{mse_label}: {(summ['mse_mean']*mse_factor):.2f} ± {(summ['mse_std']*mse_factor):.2f}")
        print(f"{mae_label}: {(summ['mae_mean']*mae_factor):.2f} ± {(summ['mae_std']*mae_factor):.2f}")
    else:
        print("\n--- Mean ± Std over seeds (raw) ---")
        print(f"MSE: {summ['mse_mean']:.6f} ± {summ['mse_std']:.6f}")
        print(f"MAE: {summ['mae_mean']:.6f} ± {summ['mae_std']:.6f}")


def print_per_file_results(files: List[str], scaled: bool = True) -> None:
    """
    Print one final result per file.
    If scaled=True prints dataset-aware paper-like format.
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
            mse_factor, mae_factor, mse_label, mae_label = _get_scaling_from_filename(fp)
            mse_s = d["mse"] * mse_factor
            mae_s = d["mae"] * mae_factor
            print(f"{rel} | {mse_label}={mse_s:.3f} | {mae_label}={mae_s:.3f}")
        else:
            print(
                f"{rel} | best_epoch={d['best_epoch']:4d} | "
                f"mse={d['mse']:.6f} | rmse={d['rmse']:.6f} | mae={d['mae']:.6f} | loss={d['loss']:.6f}"
            )


def main():
    ap = argparse.ArgumentParser("Summarize per-seed results inside .train.log files")
    ap.add_argument(
        "--glob",
        required=True,
        help='Glob for logs, e.g. "**/*.train.log"',
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Print raw mse/mae instead of scaled format.",
    )
    ap.add_argument(
        "--per_seed",
        action="store_true",
        help="Compute mean±std across seeds inside each file.",
    )
    args = ap.parse_args()

    matched = sorted(glob.glob(args.glob, recursive=True))
    files = [f for f in matched if f.endswith(".train.log")]

    if not files:
        raise ValueError(f"No .train.log files matched: {args.glob}")

    for fp in files:
        if args.per_seed:
            print_seed_summary_for_file(fp, scaled=(not args.raw))
        else:
            print_per_file_results([fp], scaled=(not args.raw))


def extract_best_per_seed_from_file(train_log_path: str) -> List[Dict[str, float]]:
    """
    Parse ONE .train.log that contains multiple seeds/runs.
    Rule per seed:
      - seed boundary detected when epoch == 0 (Epoch 000)
      - only accept a test line when current_epoch == best_epoch
      - keep overwriting *within the seed* -> last accepted is the seed's final best
    Returns: list of dicts, one per seed (only seeds that have a valid accepted record).
    """
    lines = _read_lines(train_log_path)

    seeds: List[Dict[str, float]] = []

    current_seed_best: Optional[Dict[str, float]] = None
    current_epoch: Optional[int] = None
    seed_idx = -1

    def finalize_seed():
        nonlocal current_seed_best
        if current_seed_best is not None:
            seeds.append(current_seed_best)
        current_seed_best = None

    for line in lines:
        m_epoch = EPOCH_LINE_RE.search(line)
        if m_epoch:
            ep = int(m_epoch.group("epoch"))

            # new seed starts at Epoch 000
            if ep == 0:
                # finalize previous seed (if any)
                if seed_idx >= 0:
                    finalize_seed()
                seed_idx += 1
                current_epoch = 0
            else:
                current_epoch = ep

        m_test = TEST_LINE_RE_RMSE.search(line)
        if m_test:
            best_epoch = int(m_test.group("best_epoch"))
            if current_epoch is None or current_epoch != best_epoch:
                continue

            current_seed_best = {
                "seed_idx": seed_idx,
                "best_epoch": best_epoch,
                "loss": float(m_test.group("loss")),
                "mse": float(m_test.group("mse")),
                "rmse": float(m_test.group("rmse")),
                "mae": float(m_test.group("mae")),
            }

    # finalize last seed
    finalize_seed()
    return seeds


def summarize_seeds(seeds: List[Dict[str, float]]) -> Dict[str, float]:
    if not seeds:
        raise ValueError("No valid per-seed results found (no accepted epoch==best_epoch test lines).")

    mse = np.array([s["mse"] for s in seeds], dtype=float)
    mae = np.array([s["mae"] for s in seeds], dtype=float)

    return {
        "n_seeds": float(len(seeds)),
        "mse_mean": float(mse.mean()),
        "mse_std": float(mse.std(ddof=0)),
        "mae_mean": float(mae.mean()),
        "mae_std": float(mae.std(ddof=0)),
    }


if __name__ == "__main__":
    main()