# lib/plot_val_test_loss.py
# Parse train.log and plot Val vs Test loss (single run or mean±std across runs)
# Supports axis limits via: --xmin --xmax --ymin --ymax

import re
import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Regex patterns for your logs
# ----------------------------
# Example:
# INFO | - Epoch 109, ExpID 63707 best_epoch=109 improved=1
EPOCH_LINE_RE = re.compile(
    r"-\s*Epoch\s+(?P<epoch>\d+)\s*,\s*ExpID\s+(?P<expid>\d+)"
)

# Example:
# INFO | Val - Loss, MSE, MAE: 0.00093, 0.00093, 0.01801
VAL_LINE_RE = re.compile(
    r"Val\s*-\s*Loss,\s*MSE,\s*MAE\s*:\s*"
    r"(?P<val_loss>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*,\s*"
    r"(?P<val_mse>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*,\s*"
    r"(?P<val_mae>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
)

# Example:
# INFO | Test - Best epoch, Loss, MSE, MAE: 109, 0.00090, 0.00090, 0.01808
TEST_LINE_RE = re.compile(
    r"Test\s*-\s*Best epoch,\s*Loss,\s*MSE,\s*MAE\s*:\s*"
    r"(?P<best_epoch>-?\d+)\s*,\s*"
    r"(?P<test_loss>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*,\s*"
    r"(?P<test_mse>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s*,\s*"
    r"(?P<test_mae>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
)


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()


def parse_log_file(path: str) -> Dict[str, List[dict]]:
    """
    Parses a single log file that may contain one or multiple ExpIDs.
    Returns:
        runs[expid] = list of records, each record:
            {
              "epoch": int,
              "val_loss": float or np.nan,
              "val_mse": float or np.nan,
              "val_mae": float or np.nan,
              "test_loss": float or np.nan,   # best-so-far test loss at that epoch
              "test_mse": float or np.nan,
              "test_mae": float or np.nan
            }
    """
    lines = _read_lines(path)

    runs: Dict[str, List[dict]] = {}

    current_expid: Optional[str] = None
    current_epoch: Optional[int] = None
    current_val: Optional[Tuple[float, float, float]] = None  # (val_loss, val_mse, val_mae)
    current_test: Optional[Tuple[float, float, float]] = None  # (test_loss, test_mse, test_mae)

    def _flush_if_ready():
        nonlocal current_expid, current_epoch, current_val, current_test
        if current_expid is None or current_epoch is None:
            return

        rec = {
            "epoch": int(current_epoch),
            "val_loss": float(current_val[0]) if current_val is not None else np.nan,
            "val_mse": float(current_val[1]) if current_val is not None else np.nan,
            "val_mae": float(current_val[2]) if current_val is not None else np.nan,
            "test_loss": float(current_test[0]) if current_test is not None else np.nan,
            "test_mse": float(current_test[1]) if current_test is not None else np.nan,
            "test_mae": float(current_test[2]) if current_test is not None else np.nan,
        }
        runs.setdefault(current_expid, []).append(rec)

        # reset epoch-scoped values
        current_epoch = None
        current_val = None
        current_test = None

    for line in lines:
        m_epoch = EPOCH_LINE_RE.search(line)
        if m_epoch:
            # new epoch encountered -> flush previous epoch (if any)
            if current_epoch is not None:
                _flush_if_ready()

            current_expid = m_epoch.group("expid")
            current_epoch = int(m_epoch.group("epoch"))
            current_val = None
            current_test = None
            continue

        m_val = VAL_LINE_RE.search(line)
        if m_val and current_epoch is not None:
            current_val = (
                float(m_val.group("val_loss")),
                float(m_val.group("val_mse")),
                float(m_val.group("val_mae")),
            )
            continue

        m_test = TEST_LINE_RE.search(line)
        if m_test and current_epoch is not None:
            current_test = (
                float(m_test.group("test_loss")),
                float(m_test.group("test_mse")),
                float(m_test.group("test_mae")),
            )
            continue

    # flush last epoch
    if current_epoch is not None:
        _flush_if_ready()

    # sort each run by epoch and deduplicate by epoch (keep last)
    for expid, recs in runs.items():
        by_epoch = {}
        for r in recs:
            by_epoch[r["epoch"]] = r
        runs[expid] = [by_epoch[e] for e in sorted(by_epoch.keys())]

    return runs


def _align_runs_for_mean(records_by_run: Dict[str, List[dict]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns multiple runs by epoch index and truncates to the minimum length.
    Returns:
      epochs (min_len,), values (n_runs, min_len)
    """
    expids = sorted(records_by_run.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    series = []
    epochs_list = []

    for expid in expids:
        recs = records_by_run[expid]
        epochs = np.array([r["epoch"] for r in recs], dtype=int)
        vals = np.array([r[key] for r in recs], dtype=float)
        epochs_list.append(epochs)
        series.append(vals)

    min_len = min(len(s) for s in series)
    epochs0 = epochs_list[0][:min_len]
    vals_mat = np.stack([s[:min_len] for s in series], axis=0)
    return epochs0, vals_mat


def _apply_axis_limits(xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin is not None or xmax is not None:
        plt.xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin, top=ymax)


def plot_single_run(
    run_id: str,
    recs: List[dict],
    title: str,
    outpath: Optional[str] = None,
    show: bool = True,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    epochs = [r["epoch"] for r in recs]
    val_loss = [r["val_loss"] for r in recs]
    test_loss = [r["test_loss"] for r in recs]

    plt.figure()
    plt.plot(epochs, val_loss, label="Val loss")
    plt.plot(epochs, test_loss, label="Test loss (best-so-far)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} | {run_id}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    _apply_axis_limits(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    if outpath:
        if os.path.dirname(outpath):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def plot_mean_over_runs(
    records_by_run: Dict[str, List[dict]],
    title: str,
    outpath: Optional[str] = None,
    show: bool = True,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    epochs, val_mat = _align_runs_for_mean(records_by_run, "val_loss")
    _, test_mat = _align_runs_for_mean(records_by_run, "test_loss")

    val_mean = np.nanmean(val_mat, axis=0)
    val_std = np.nanstd(val_mat, axis=0, ddof=1) if val_mat.shape[0] > 1 else np.zeros_like(val_mean)

    test_mean = np.nanmean(test_mat, axis=0)
    test_std = np.nanstd(test_mat, axis=0, ddof=1) if test_mat.shape[0] > 1 else np.zeros_like(test_mean)

    plt.figure()
    plt.plot(epochs, val_mean, label="Val loss (mean)")
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.plot(epochs, test_mean, label="Test loss (best-so-far, mean)")
    plt.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} | mean ± std over {val_mat.shape[0]} Seeds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    _apply_axis_limits(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    if outpath:
        if os.path.dirname(outpath):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser("Plot Val/Test loss from train.log files")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="Path to a log file (may contain multiple ExpIDs).")
    src.add_argument("--glob", type=str, help="Glob to many train.log files (each file = one run).")

    # axis limits
    ap.add_argument("--xmin", type=float, default=None, help="X-axis min (epoch).")
    ap.add_argument("--xmax", type=float, default=None, help="X-axis max (epoch).")
    ap.add_argument("--ymin", type=float, default=None, help="Y-axis min (loss).")
    ap.add_argument("--ymax", type=float, default=None, help="Y-axis max (loss).")

    ap.add_argument("--expid", type=str, default=None, help="If --file has multiple ExpIDs, plot only this ExpID.")
    ap.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "mean"],
        help="single: plot one run; mean: average over multiple runs (mean ± std).",
    )
    ap.add_argument("--out", type=str, default=None, help="Output .png path (optional).")
    ap.add_argument("--title", type=str, default="Val vs Test Loss", help="Plot title prefix.")
    ap.add_argument("--no_show", action="store_true", help="Do not display window (useful on servers).")
    ap.add_argument("--out_dir", type=str, default=None, help="When plotting multiple runs, save plots into this dir.")

    args = ap.parse_args()
    show = not args.no_show

    if args.file:
        runs = parse_log_file(args.file)

        if args.mode == "mean":
            if len(runs) < 2:
                raise SystemExit("Need >=2 ExpIDs in the file to use --mode mean.")
            plot_mean_over_runs(
                runs,
                title=args.title,
                outpath=args.out,
                show=show,
                xmin=args.xmin,
                xmax=args.xmax,
                ymin=args.ymin,
                ymax=args.ymax,
            )
            return

        # mode == single
        if args.expid is None:
            if len(runs) == 1:
                expid = next(iter(runs.keys()))
            else:
                expids = ", ".join(sorted(runs.keys(), key=lambda x: int(x)))
                raise SystemExit(
                    f"--file contains multiple ExpIDs ({expids}). Please specify --expid <one of them> "
                    f"or use --mode mean."
                )
        else:
            expid = args.expid
            if expid not in runs:
                expids = ", ".join(sorted(runs.keys(), key=lambda x: int(x)))
                raise SystemExit(f"ExpID={expid} not found in file. Available: {expids}")

        plot_single_run(
            run_id=f"ExpID={expid}",
            recs=runs[expid],
            title=args.title,
            outpath=args.out,
            show=show,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
        )
        return

    # args.glob
    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        raise SystemExit(f"No files matched glob: {args.glob}")

    runs_by_file: Dict[str, List[dict]] = {}
    for fp in files:
        parsed = parse_log_file(fp)
        if not parsed:
            continue
        if len(parsed) == 1:
            expid = next(iter(parsed.keys()))
            runs_by_file[f"{os.path.basename(fp)}::ExpID={expid}"] = parsed[expid]
        else:
            for expid, recs in parsed.items():
                runs_by_file[f"{os.path.basename(fp)}::ExpID={expid}"] = recs

    if not runs_by_file:
        raise SystemExit("No valid runs found in the matched files.")

    if args.mode == "mean":
        plot_mean_over_runs(
            runs_by_file,
            title=args.title,
            outpath=args.out,
            show=show,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
        )
    else:
        out_dir = args.out_dir
        if out_dir is None and args.out is not None:
            out_dir = os.path.dirname(args.out) if os.path.dirname(args.out) else "plots"

        for run_name, recs in runs_by_file.items():
            if out_dir:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
                outpath = os.path.join(out_dir, f"{safe_name}.png")
            else:
                outpath = None

            plot_single_run(
                run_id=run_name,
                recs=recs,
                title=args.title,
                outpath=outpath,
                show=show,
                xmin=args.xmin,
                xmax=args.xmax,
                ymin=args.ymin,
                ymax=args.ymax,
            )


if __name__ == "__main__":
    main()