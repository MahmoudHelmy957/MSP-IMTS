import os
import re
import glob
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class ExpSummary:
    file: str
    exp_id: Optional[int]
    best_epoch: Optional[int]
    test_loss: Optional[float]
    test_mse: Optional[float]
    test_mae: Optional[float]
    total_masked: Optional[int]
    correct: Optional[int]
    right_rate_pct: Optional[float]  # percent, e.g. 60.0899


# ---- regex patterns (based on your log format) ----
RE_EPOCH_LINE = re.compile(r"-\s*Epoch\s+\d+,\s*ExpID\s+(\d+)\s+best_epoch=(\d+)\s+improved=\d+")
RE_TEST_LINE = re.compile(
    r"Test\s*-\s*Best epoch,\s*Loss,\s*MSE,\s*MAE:\s*(\d+),\s*([0-9.eE+-]+),\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)"
)
RE_POINTS_LINE = re.compile(
    r"Test\s*-\s*Points\s*\(EXACT\):\s*total_masked=(\d+)\s+correct=(\d+)\s+right_rate=([0-9.]+)%"
)


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def parse_train_log(path: str) -> ExpSummary:
    """
    Parse one *.train.log and return the BEST (min) MSE summary found in the file.
    This is robust to appended logs and repeated "Test - Best epoch" lines.
    """
    exp_id = None

    best_epoch = None
    best_loss = None
    best_mse = None
    best_mae = None

    total_masked = None
    correct = None
    right_rate_pct = None

    # Track best by MSE (lower is better)
    best_mse_val = float("inf")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_EPOCH_LINE.search(line)
            if m:
                exp_id = _safe_int(m.group(1))
                # this is "best_epoch" as the training script claims; we don't trust it fully
                continue

            m = RE_TEST_LINE.search(line)
            if m:
                ep = _safe_int(m.group(1))
                loss = _safe_float(m.group(2))
                mse = _safe_float(m.group(3))
                mae = _safe_float(m.group(4))

                if mse is not None and mse < best_mse_val:
                    best_mse_val = mse
                    best_epoch = ep
                    best_loss = loss
                    best_mse = mse
                    best_mae = mae
                continue

            m = RE_POINTS_LINE.search(line)
            if m:
                # These "points" lines might correspond to the last printed evaluation,
                # not necessarily the epoch with best MSE. We keep the last-seen.
                total_masked = _safe_int(m.group(1))
                correct = _safe_int(m.group(2))
                right_rate_pct = _safe_float(m.group(3))
                continue

    return ExpSummary(
        file=path,
        exp_id=exp_id,
        best_epoch=best_epoch,
        test_loss=best_loss,
        test_mse=best_mse,
        test_mae=best_mae,
        total_masked=total_masked,
        correct=correct,
        right_rate_pct=right_rate_pct,
    )

def find_best_experiment(
    folder: str,
    pattern: str = "*.train.log",
    metric: str = "test_mse",  # "test_mse" | "test_loss" | "test_mae" | "right_rate_pct"
    recursive: bool = True,
) -> Tuple[Optional[ExpSummary], List[ExpSummary]]:
    """
    Returns (best_summary, all_summaries_sorted).

    For loss/mse/mae: smaller is better.
    For right_rate_pct: larger is better.
    """
    folder = os.path.abspath(folder)
    glob_pat = os.path.join(folder, "**", pattern) if recursive else os.path.join(folder, pattern)
    files = sorted(glob.glob(glob_pat, recursive=recursive))

    summaries: List[ExpSummary] = [parse_train_log(p) for p in files]

    def get_metric(s: ExpSummary) -> Optional[float]:
        return getattr(s, metric, None)

    valid = [s for s in summaries if get_metric(s) is not None]
    if not valid:
        return None, summaries

    reverse = metric == "right_rate_pct"
    valid_sorted = sorted(valid, key=lambda s: get_metric(s), reverse=reverse)

    return valid_sorted[0], valid_sorted


def pretty_print(best: Optional[ExpSummary], ranked: List[ExpSummary], top_k: int = 10) -> None:
    def fmt(x, nd=6):
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    print("\nTop experiments:")
    print(
        f"{'rank':>4}  {'exp_id':>8}  {'best_ep':>7}  {'test_mse':>12}  {'test_mae':>10}  {'right%':>8}  file"
    )
    print("-" * 110)
    for i, s in enumerate(ranked[:top_k], start=1):
        print(
            f"{i:>4}  {fmt(s.exp_id):>8}  {fmt(s.best_epoch):>7}  {fmt(s.test_mse, 8):>12}  "
            f"{fmt(s.test_mae, 6):>10}  {fmt(s.right_rate_pct, 4):>8}  {os.path.basename(s.file)}"
        )

    if best is not None:
        print("\nBEST:")
        print(best)
    else:
        print("\nNo valid experiments found (could not parse metric from logs).")


if __name__ == "__main__":
    # Example:
    #   python best_experiment_from_logs.py /path/to/logs --metric test_mse
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str, help="Folder containing *.train.log files")
    ap.add_argument("--metric", type=str, default="test_mse",
                    choices=["test_mse", "test_loss", "test_mae", "right_rate_pct"])
    ap.add_argument("--pattern", type=str, default="*.train.log")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    best, ranked = find_best_experiment(
        folder=args.folder,
        pattern=args.pattern,
        metric=args.metric,
        recursive=(not args.no_recursive),
    )
    pretty_print(best, ranked, top_k=args.topk)
