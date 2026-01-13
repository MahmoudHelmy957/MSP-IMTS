# lib/analyzelogs.py
# Fixed + consistent version
#  - File name matches: `from lib.analyzelogs import *`
#  - No placeholder code in sample_data_sanity
#  - No hidden dependence on utils; caller passes a batch_fetch_fn
#  - Safe logger creation (no duplicate handlers, no propagation)
#  - Functions used by RunModelsLogs.py are present:
#       ensure_dir, build_file_logger, add_stdout_handler,
#       safe_float, compute_grad_norm, cuda_mem_stats,
#       summarize_mask, try_get_timestamp,
#       count_right_points_tol, log_topk_errors, log_topk_best

import os
import sys
import logging
import numpy as np
import torch


# ------------------------- paths -------------------------

def project_root() -> str:
    """
    Assumes this file is at: <root>/lib/analyzelogs.py
    Returns: <root>
    """
    lib_dir = os.path.dirname(os.path.abspath(__file__))     # <root>/lib
    return os.path.abspath(os.path.join(lib_dir, ".."))      # <root>


def ensure_dir(rel_path: str) -> str:
    """
    Create a directory under project root and return absolute path.
    Example: ensure_dir("analyzelogs") -> <root>/analyzelogs
    """
    root = project_root()
    target = os.path.join(root, rel_path)
    os.makedirs(target, exist_ok=True)
    return target


# ------------------------- logger creation -------------------------

def build_file_logger(name: str, filepath: str, mode: str = "a", level=logging.INFO) -> logging.Logger:
    """
    Dedicated file logger. No propagation, no duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # clear handlers (important if you rerun in same process)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(fmt="%(levelname)s | %(message)s")

    fh = logging.FileHandler(filepath, mode=mode)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def add_stdout_handler(logger: logging.Logger, level=logging.INFO) -> None:
    # avoid duplicating stdout handlers if called twice
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout:
            return

    fmt = logging.Formatter(fmt=" %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def setup_loggers(
    *,
    log_dir_rel: str,
    run_tag: str,
    logmode: str = "a",
    stdout_run: bool = True,
    stdout_train: bool = True,
):
    """
    Creates your standard set of loggers under <root>/<log_dir_rel>.

    Files created:
      <base>.run.log
      <base>.train.log
      <base>.system.log
      <base>.toperr.log
      <base>.error.log

    Returns:
      (paths_dict, loggers_dict)
    """
    log_dir_abs = ensure_dir(log_dir_rel)
    base = os.path.join(log_dir_abs, run_tag)

    paths = {
        "log_dir": log_dir_abs,
        "base": base,
        "run": base + ".run.log",
        "train": base + ".train.log",
        "system": base + ".system.log",
        "toperr": base + ".toperr.log",
        "error": base + ".error.log",
    }

    loggers = {
        "run": build_file_logger("run_logger", paths["run"], mode=logmode, level=logging.INFO),
        "train": build_file_logger("train_logger", paths["train"], mode=logmode, level=logging.INFO),
        "system": build_file_logger("sys_logger", paths["system"], mode=logmode, level=logging.INFO),
        "toperr": build_file_logger("top_logger", paths["toperr"], mode=logmode, level=logging.INFO),
        "error": build_file_logger("err_logger", paths["error"], mode=logmode, level=logging.INFO),
    }

    if stdout_run:
        add_stdout_handler(loggers["run"], level=logging.INFO)
    if stdout_train:
        add_stdout_handler(loggers["train"], level=logging.INFO)

    return paths, loggers


# ------------------------- numeric helpers -------------------------

def safe_float(x, default=np.nan) -> float:
    try:
        if isinstance(x, torch.Tensor):
            # accept scalars or 0-dim tensors
            return float(x.detach().cpu().item())
        return float(x)
    except Exception:
        return float(default)


def compute_grad_norm(model: torch.nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if torch.isfinite(g).all():
            total_sq += g.float().pow(2).sum().item()
        else:
            return float("nan")
    return float(np.sqrt(total_sq))


def cuda_mem_stats():
    if torch.cuda.is_available():
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return alloc, reserved, max_alloc
        except Exception:
            return np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan


def summarize_mask(mask_tensor: torch.Tensor) -> dict:
    if mask_tensor is None:
        return {"mask_ratio": np.nan, "mask_count": 0, "mask_total": 0}
    m = mask_tensor.detach()
    mb = (m > 0.5) if m.dtype != torch.bool else m
    cnt = int(mb.sum().item())
    tot = int(mb.numel())
    ratio = cnt / max(tot, 1)
    return {"mask_ratio": float(ratio), "mask_count": cnt, "mask_total": tot}


def try_get_timestamp(tp_to_predict, b_idx: int, t_idx: int):
    """
    Best-effort extractor for a timestamp index from tp_to_predict.
    Supports Tensor [B,Lp], Tensor [Lp], list-of-lists, etc.
    """
    try:
        if isinstance(tp_to_predict, torch.Tensor):
            if tp_to_predict.dim() == 2:
                return safe_float(tp_to_predict[b_idx, t_idx])
            if tp_to_predict.dim() == 1:
                return safe_float(tp_to_predict[t_idx])
        if isinstance(tp_to_predict, (list, tuple, np.ndarray)):
            if len(tp_to_predict) == 0:
                return None
            if isinstance(tp_to_predict[0], (list, tuple, np.ndarray)):
                return float(tp_to_predict[b_idx][t_idx])
            return float(tp_to_predict[t_idx])
    except Exception:
        pass
    return None


# ------------------------- "right points" metric -------------------------

def count_right_points_tol(pred, tgt, msk, abs_tol: float, rel_tol: float):
    """
    Count "right" if:
      |pred-true| <= abs_tol  OR  |pred-true| <= rel_tol * |true|
    under mask.
    """
    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
    total = int(mb.sum().item())
    if total == 0:
        return 0, 0

    diff = (pred - tgt).abs()
    tgt_abs = tgt.abs().clamp(min=1e-8)
    correct = ((diff <= abs_tol) | (diff <= rel_tol * tgt_abs)) & mb
    return total, int(correct.sum().item())


# ------------------------- top-k logging -------------------------

def log_topk_errors(
    logger_top: logging.Logger,
    split: str,
    epoch: int,
    pred: torch.Tensor,
    tgt: torch.Tensor,
    msk: torch.Tensor,
    tp_to_predict=None,
    topk: int = 10,
):
    """
    TOP-K largest abs error among masked points.
    pred/tgt/msk expected shapes: (B, Lp, N)
    """
    if pred is None or tgt is None or msk is None:
        logger_top.warning(f"TOP_ERR | split={split} epoch={epoch} skipped (missing tensors)")
        return

    with torch.no_grad():
        pred_ = pred.detach()
        tgt_ = tgt.detach()
        m = msk.detach()
        mb = (m > 0.5) if m.dtype != torch.bool else m

        if mb.sum().item() == 0:
            logger_top.warning(f"TOP_ERR | split={split} epoch={epoch} skipped (mask empty)")
            return

        if pred_.dim() != 3:
            logger_top.warning(f"TOP_ERR | split={split} epoch={epoch} skipped (pred shape={tuple(pred_.shape)})")
            return

        err_abs = (pred_ - tgt_).abs()

        # mask out invalid points
        err_abs_masked = err_abs.clone()
        err_abs_masked[~mb] = -1.0

        flat = err_abs_masked.reshape(-1)
        valid = int((flat >= 0).sum().item())
        k = min(int(topk), valid)
        if k <= 0:
            logger_top.warning(f"TOP_ERR | split={split} epoch={epoch} skipped (no valid entries)")
            return

        vals, idxs = torch.topk(flat, k=k, largest=True)

        B, Lp, N = pred_.shape
        for rank in range(k):
            flat_i = int(idxs[rank].item())
            abs_e = float(vals[rank].item())

            b = flat_i // (Lp * N)
            rem = flat_i % (Lp * N)
            t = rem // N
            v = rem % N

            y_pred = safe_float(pred_[b, t, v])
            y_true = safe_float(tgt_[b, t, v])
            diff = y_pred - y_true
            ts = try_get_timestamp(tp_to_predict, b, t) if tp_to_predict is not None else None

            if ts is None:
                logger_top.info(
                    f"TOP_ERR | split={split} epoch={epoch} rank={rank+1}/{k} "
                    f"b={b} t={t} var_idx={v} "
                    f"y_true={y_true:.6f} y_pred={y_pred:.6f} diff={diff:.6f} abs_err={abs_e:.6f}"
                )
            else:
                logger_top.info(
                    f"TOP_ERR | split={split} epoch={epoch} rank={rank+1}/{k} "
                    f"b={b} t={t} var_idx={v} tp={ts:.6f} "
                    f"y_true={y_true:.6f} y_pred={y_pred:.6f} diff={diff:.6f} abs_err={abs_e:.6f}"
                )


def log_topk_best(
    logger_top: logging.Logger,
    split: str,
    epoch: int,
    pred: torch.Tensor,
    tgt: torch.Tensor,
    msk: torch.Tensor,
    tp_to_predict=None,
    topk: int = 3,
):
    """
    TOP-K smallest abs error among masked points ("best corrected").
    pred/tgt/msk expected shapes: (B, Lp, N)
    """
    if pred is None or tgt is None or msk is None:
        logger_top.warning(f"TOP_BEST | split={split} epoch={epoch} skipped (missing tensors)")
        return

    with torch.no_grad():
        pred_ = pred.detach()
        tgt_ = tgt.detach()
        m = msk.detach()
        mb = (m > 0.5) if m.dtype != torch.bool else m

        if mb.sum().item() == 0:
            logger_top.warning(f"TOP_BEST | split={split} epoch={epoch} skipped (mask empty)")
            return

        if pred_.dim() != 3:
            logger_top.warning(f"TOP_BEST | split={split} epoch={epoch} skipped (pred shape={tuple(pred_.shape)})")
            return

        err_abs = (pred_ - tgt_).abs()

        err_abs_masked = err_abs.clone()
        err_abs_masked[~mb] = float("inf")

        flat = err_abs_masked.reshape(-1)
        valid = int(torch.isfinite(flat).sum().item())
        k = min(int(topk), valid)
        if k <= 0:
            logger_top.warning(f"TOP_BEST | split={split} epoch={epoch} skipped (no valid entries)")
            return

        vals, idxs = torch.topk(flat, k=k, largest=False)

        B, Lp, N = pred_.shape
        for rank in range(k):
            flat_i = int(idxs[rank].item())
            abs_e = float(vals[rank].item())

            b = flat_i // (Lp * N)
            rem = flat_i % (Lp * N)
            t = rem // N
            v = rem % N

            y_pred = safe_float(pred_[b, t, v])
            y_true = safe_float(tgt_[b, t, v])
            diff = y_pred - y_true
            ts = try_get_timestamp(tp_to_predict, b, t) if tp_to_predict is not None else None

            if ts is None:
                logger_top.info(
                    f"TOP_BEST | split={split} epoch={epoch} rank={rank+1}/{k} "
                    f"b={b} t={t} var_idx={v} "
                    f"y_true={y_true:.6f} y_pred={y_pred:.6f} diff={diff:.6f} abs_err={abs_e:.6f}"
                )
            else:
                logger_top.info(
                    f"TOP_BEST | split={split} epoch={epoch} rank={rank+1}/{k} "
                    f"b={b} t={t} var_idx={v} tp={ts:.6f} "
                    f"y_true={y_true:.6f} y_pred={y_pred:.6f} diff={diff:.6f} abs_err={abs_e:.6f}"
                )


# ------------------------- data sanity (optional helper) -------------------------

def sample_data_sanity(
    run_logger: logging.Logger,
    data_obj: dict,
    split_name: str,
    dl_key: str,
    n_key: str,
    max_batches: int,
    batch_fetch_fn,
):
    """
    Logs mask ratio stats for a few batches into run.log.

    You MUST pass:
      batch_fetch_fn = lambda data_obj, dl_key: <batch_dict>

    In your RunModelsLogs you already use:
      batch_fetch_fn = lambda dobj, key: utils.get_next_batch(dobj[key])

    This keeps analyzelogs.py independent from lib.utils (no circular imports).
    """
    if dl_key not in data_obj or n_key not in data_obj:
        run_logger.warning(f"DATA_SANITY | split={split_name} missing {dl_key}/{n_key}")
        return

    n_batches = int(data_obj[n_key])
    k = min(int(max_batches), n_batches)
    if k <= 0:
        run_logger.warning(f"DATA_SANITY | split={split_name} has 0 batches")
        return

    mask_ratios = []
    for _ in range(k):
        b = batch_fetch_fn(data_obj, dl_key)
        if isinstance(b, dict) and ("mask_predicted_data" in b) and (b["mask_predicted_data"] is not None):
            mask_ratios.append(summarize_mask(b["mask_predicted_data"])["mask_ratio"])

    if len(mask_ratios) > 0:
        run_logger.info(
            f"DATA_SANITY | split={split_name} sampled_batches={k} "
            f"mask_ratio_mean={float(np.mean(mask_ratios)):.6f} "
            f"mask_ratio_min={float(np.min(mask_ratios)):.6f} "
            f"mask_ratio_max={float(np.max(mask_ratios)):.6f}"
        )
    else:
        run_logger.info(f"DATA_SANITY | split={split_name} sampled_batches={k} (no mask_predicted_data found)")

