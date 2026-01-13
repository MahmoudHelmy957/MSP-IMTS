# RunModelsLogs.py
# Fixed version + LAZY INIT FIX (MultiScaleTPatchGNN) + CHANNEL INFLUENCE (per-channel TEST error ranking)
# + OPTIONAL: train/eval only on ONE target channel (e.g., USHCN temperature ch=0)
#
# Key additions:
#  - --target_channel (int, default=-1): if >=0, loss/metrics computed ONLY on that channel
#    (model still outputs all channels; we just slice pred/tgt/mask for supervision).
#  - _masked_metrics_per_channel + _aggregate_per_channel
#  - TEST per-channel ranking logged to top_logger + worst channel summary logged to train_logger

import os
import sys

sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
from random import SystemRandom
import socket
import traceback
import logging

import torch
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import tPatchGNN
from lib.analyzelogs import *


parser = argparse.ArgumentParser("IMTS Forecasting")

############################# multi scale ########################
parser.add_argument(
    "--multi_scales",
    type=str,
    default="",
    help='Comma list of patch sizes in hours, e.g. "2,8,24". Empty = single-scale.',
)
parser.add_argument(
    "--multi_strides",
    type=str,
    default="",
    help="Comma list of strides in hours. Empty = same as multi_scales.",
)
parser.add_argument(
    "--fusion",
    type=str,
    default="concat",
    choices=["concat", "scale_attn"],
    help="Fusion method for multi-scale.",
)
################################################

parser.add_argument("--state", type=str, default="def")
parser.add_argument("-n", type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument("--hop", type=int, default=1, help="hops in GNN")
parser.add_argument("--nhead", type=int, default=1, help="heads in Transformer")
parser.add_argument("--tf_layer", type=int, default=1, help="# of layer in Transformer")
parser.add_argument("--nlayer", type=int, default=1, help="# of layer in TSmodel")
parser.add_argument("--epoch", type=int, default=1000, help="training epochs")
parser.add_argument("--patience", type=int, default=10, help="patience for early stop")
parser.add_argument(
    "--history",
    type=int,
    default=24,
    help="number of hours (months for ushcn and ms for activity) as historical window",
)
parser.add_argument("-ps", "--patch_size", type=float, default=24, help="window size for a patch")
parser.add_argument("--stride", type=float, default=24, help="period stride for patch sliding")
parser.add_argument("--logmode", type=str, default="a", help="File mode of logging.")

parser.add_argument("--lr", type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument("--w_decay", type=float, default=0.0, help="weight decay.")
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("--normalization",type=int,default=0, help="0 = per-channel (default), 1 = global scalar normalization (Activity only).")

parser.add_argument("--save", type=str, default="experiments/", help="Path for save checkpoints")
parser.add_argument(
    "--load",
    type=str,
    default=None,
    help="ID of the experiment to load for evaluation. If None, run a new experiment.",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument(
    "--dataset", type=str, default="physionet", help="Dataset to load. Available: physionet, mimic, ushcn"
)

# value 0 means using original time granularity, Value 1 means quantization by 1 hour,
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument("--quantization", type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument("--model", type=str, default="tPatchGNN", help="Model name")
parser.add_argument("--outlayer", type=str, default="Linear", help="Output layer name")
parser.add_argument("-hd", "--hid_dim", type=int, default=64, help="Number of units per hidden layer")
parser.add_argument("-td", "--te_dim", type=int, default=10, help="Number of units for time encoding")
parser.add_argument("-nd", "--node_dim", type=int, default=10, help="Number of units for node vectors")
parser.add_argument("--gpu", type=str, default="0", help="which gpu to use.")

# top error logging
parser.add_argument("--topk_err", type=int, default=10, help="Top-K largest TEST forecast errors to log.")

# data sanity
parser.add_argument(
    "--data_sanity_batches",
    type=int,
    default=3,
    help="How many batches to sample per split for data sanity logs.",
)

# OPTIONAL: supervise only one target channel (e.g. USHCN temperature channel=0)
parser.add_argument(
    "--target_channel",
    type=int,
    default=-1,
    help="If >=0, compute loss/metrics ONLY on this channel index (e.g. 0 for temperature).",
)

args = parser.parse_args()
args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

# SLURM identifiers (for log naming)
slurm_job_id = os.environ.get("SLURM_JOB_ID", str(args.PID))
slurm_job_name = os.environ.get("SLURM_JOB_NAME", "local")


# ------------------------- helpers: channel slicing -------------------------
def _slice_target_channel(pred, tgt, msk, ch: int):
    """
    Slice pred/tgt/msk to a single channel.
    Input shapes expected: (B, L, C).
    Returns (pred1, tgt1, msk1) with shape (B, L, 1).
    """
    if ch is None or ch < 0:
        return pred, tgt, msk
    return pred[..., ch : ch + 1], tgt[..., ch : ch + 1], msk[..., ch : ch + 1]


# ------------------------- helpers: collect exact matches -------------------------
def _collect_correct_examples_exact(pred, tgt, msk, tp_to_predict=None, max_k: int = 3):
    """
    Collect up to max_k examples where pred == tgt exactly (after masking).
    NOTE: exact float equality is rare; this is mostly for debugging.
    Returns list of dicts: {b,t,var_idx,tp,y_true,y_pred}
    """
    if pred is None or tgt is None or msk is None:
        return []

    with torch.no_grad():
        mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
        if mb.sum().item() == 0:
            return []

        eq = (pred == tgt) & mb
        idxs = eq.nonzero(as_tuple=False)
        if idxs.numel() == 0:
            return []

        out = []
        take = min(max_k, idxs.shape[0])
        for i in range(take):
            b, t, v = [int(x.item()) for x in idxs[i]]
            y_pred = safe_float(pred[b, t, v])
            y_true = safe_float(tgt[b, t, v])
            ts = try_get_timestamp(tp_to_predict, b, t) if tp_to_predict is not None else None
            out.append(dict(b=b, t=t, var_idx=v, tp=ts, y_true=y_true, y_pred=y_pred))
        return out


# ------------------------- data sanity -------------------------
def _sample_data_sanity_main(run_logger, data_obj, split_name: str, dl_key: str, n_key: str, max_batches: int):
    if dl_key not in data_obj or n_key not in data_obj:
        run_logger.warning(f"DATA_SANITY | split={split_name} missing {dl_key}/{n_key}")
        return

    n_batches = int(data_obj[n_key])
    k = min(max_batches, n_batches)
    if k <= 0:
        run_logger.warning(f"DATA_SANITY | split={split_name} has 0 batches")
        return

    mask_ratios = []
    for _ in range(k):
        b = utils.get_next_batch(data_obj[dl_key])
        if "mask_predicted_data" in b:
            ms = summarize_mask(b["mask_predicted_data"])
            mask_ratios.append(ms["mask_ratio"])

    if len(mask_ratios) > 0:
        run_logger.info(
            f"DATA_SANITY | split={split_name} sampled_batches={k} "
            f"mask_ratio_mean={float(np.mean(mask_ratios)):.6f} "
            f"mask_ratio_min={float(np.min(mask_ratios)):.6f} "
            f"mask_ratio_max={float(np.max(mask_ratios)):.6f}"
        )
    else:
        run_logger.info(f"DATA_SANITY | split={split_name} sampled_batches={k} (no mask_predicted_data found)")


def _debug_value_ranges(run_logger, data_obj, split_name: str, dl_key: str, n_batches: int = 1):
    for bi in range(n_batches):
        b = utils.get_next_batch(data_obj[dl_key])

        if "X_list" in b:
            # Multi-scale
            for k, x in enumerate(b["X_list"]):
                run_logger.info(
                    f"[VAL_DEBUG] split={split_name} batch={bi} X_list[{k}] "
                    f"min={float(x.min()):.6f} max={float(x.max()):.6f} mean={float(x.mean()):.6f}"
                )
            y = b["data_to_predict"]
            run_logger.info(
                f"[VAL_DEBUG] split={split_name} batch={bi} data_to_predict "
                f"min={float(y.min()):.6f} max={float(y.max()):.6f} mean={float(y.mean()):.6f}"
            )
        else:
            # Single-scale
            x = b["observed_data"]
            y = b["data_to_predict"]
            run_logger.info(
                f"[VAL_DEBUG] split={split_name} batch={bi} observed_data "
                f"min={float(x.min()):.6f} max={float(x.max()):.6f} mean={float(x.mean()):.6f}"
            )
            run_logger.info(
                f"[VAL_DEBUG] split={split_name} batch={bi} data_to_predict "
                f"min={float(y.min()):.6f} max={float(y.max()):.6f} mean={float(y.mean()):.6f}"
            )


# ------------------------- time/scale debug -------------------------
def _debug_time_scale_consistency(run_logger, data_obj, split_name: str, dl_key: str, n_batches: int = 2):
    """
    Expected (after the MS time-fix):
      - tt_list values are normalized to [0,1] over the HISTORY window -> max should be near 1.0
      - tp_to_predict used for decoding can be:
           * history-normalized => future often > 1.0
           * time_max-normalized (0..1) => future <= 1.0
    We flag cases that look like double-normalization:
      - tt_list max extremely small (e.g. 0.02)
    """

    def _parse_list(s):
        import re

        if s in (None, "", []):
            return []
        return [float(x) for x in re.split(r"[,\s]+", str(s).strip()) if x]

    scales = _parse_list(getattr(args, "multi_scales", ""))
    strides = _parse_list(getattr(args, "multi_strides", "")) or scales
    history = float(getattr(args, "history", 0.0))

    run_logger.info(
        f"[TS_DEBUG] split={split_name} use_ms={bool(scales)} "
        f"history={history} scales={scales} strides={strides} time_max={safe_float(data_obj.get('time_max', np.nan))}"
    )

    for bi in range(n_batches):
        b = utils.get_next_batch(data_obj[dl_key])

        # ---- Multi-scale path ----
        if "X_list" in b:
            tt_list = b.get("tt_list", None)
            tp = b.get("tp_to_predict", None)

            run_logger.info(f"[TS_DEBUG] split={split_name} batch={bi} MS keys={list(b.keys())}")
            run_logger.info(
                f"[TS_DEBUG] split={split_name} batch={bi} "
                f"#scales={len(b['X_list'])} X_shapes={[tuple(x.shape) for x in b['X_list']]}"
            )

            if tt_list is not None:
                tt_ranges = [(float(t.min().item()), float(t.max().item())) for t in tt_list]
                run_logger.info(f"[TS_DEBUG] split={split_name} batch={bi} tt_list_minmax={tt_ranges}")

                tt0_max = float(tt_list[0].max().item())
                if tt0_max < 0.2:
                    run_logger.warning(
                        f"[TS_DEBUG][WARNING] tt_list[0]_max≈{tt0_max:.4f} is very small. "
                        f"Likely time got normalized by time_max and then treated as history-normalized again. "
                        f"Expected tt_list max near 1.0."
                    )

            if tp is not None:
                run_logger.info(
                    f"[TS_DEBUG] split={split_name} batch={bi} "
                    f"tp_to_predict_minmax=({float(tp.min().item()):.6f},{float(tp.max().item()):.6f}) "
                    f"tp_shape={tuple(tp.shape)}"
                )

                tp_max = float(tp.max().item())
                if tp_max <= 1.05:
                    run_logger.warning(
                        f"[TS_DEBUG][NOTE] tp_to_predict_max≈{tp_max:.4f} is not > 1. "
                        f"If you intended history-normalized tp_to_predict, this is suspicious. "
                        f"If you intentionally keep tp normalized by time_max, ignore this."
                    )

            if "mask_predicted_data" in b:
                m = b["mask_predicted_data"]
                run_logger.info(
                    f"[TS_DEBUG] split={split_name} batch={bi} "
                    f"mask_sum={float(m.sum().item())} mask_ratio={summarize_mask(m)['mask_ratio']:.6f} "
                    f"mask_dtype={m.dtype} mask_shape={tuple(m.shape)}"
                )

        # ---- Single-scale path ----
        else:
            run_logger.info(f"[TS_DEBUG] split={split_name} batch={bi} SS keys={list(b.keys())}")

            for k in ["observed_tp", "tp_to_predict"]:
                if k in b:
                    t = b[k]
                    run_logger.info(
                        f"[TS_DEBUG] split={split_name} batch={bi} {k}_minmax="
                        f"({float(t.min().item()):.6f},{float(t.max().item()):.6f}) shape={tuple(t.shape)}"
                    )

            if "mask_predicted_data" in b:
                m = b["mask_predicted_data"]
                run_logger.info(
                    f"[TS_DEBUG] split={split_name} batch={bi} "
                    f"mask_sum={float(m.sum().item())} mask_ratio={summarize_mask(m)['mask_ratio']:.6f} "
                    f"mask_dtype={m.dtype} mask_shape={tuple(m.shape)}"
                )


# ------------------------- per-channel influence helpers -------------------------
def _masked_metrics(pred, tgt, msk):
    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
    diff = (pred - tgt)[mb]
    if diff.numel() == 0:
        return dict(loss=np.nan, mse=np.nan, rmse=np.nan, mae=np.nan, mape=np.nan)
    mse = (diff**2).mean().item()
    mae = diff.abs().mean().item()
    rmse = float(np.sqrt(mse))
    tgt_safe = tgt[mb].abs()
    mape = float(torch.mean((diff.abs() / torch.clamp(tgt_safe, min=1e-8))).item())
    return dict(loss=mse, mse=mse, rmse=rmse, mae=mae, mape=mape)


def _masked_metrics_per_channel(pred, tgt, msk):
    """
    pred,tgt,msk: (B, L, C)
    Returns: dict(overall={mse,mae}, per_ch=[{ch,n,mse,mae,rmse}...])
    """
    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
    if mb.sum().item() == 0:
        return dict(overall=dict(mse=np.nan, mae=np.nan), per_ch=[])

    diff = pred - tgt

    d_all = diff[mb]
    overall_mse = (d_all**2).mean().item()
    overall_mae = d_all.abs().mean().item()

    C = pred.shape[-1]
    per = []
    for c in range(C):
        mb_c = mb[..., c]
        if mb_c.sum().item() == 0:
            per.append(dict(ch=c, n=0, mse=np.nan, mae=np.nan, rmse=np.nan))
            continue
        d = diff[..., c][mb_c]
        mse = (d**2).mean().item()
        mae = d.abs().mean().item()
        rmse = float(np.sqrt(mse))
        per.append(dict(ch=c, n=int(mb_c.sum().item()), mse=mse, mae=mae, rmse=rmse))

    return dict(overall=dict(mse=overall_mse, mae=overall_mae), per_ch=per)


def _aggregate_per_channel(per_batch_list):
    """
    Weighted average by n (masked points) per channel.
    """
    if len(per_batch_list) == 0:
        return None
    C = len(per_batch_list[0]["per_ch"])
    acc = [{"n": 0, "mse_sum": 0.0, "mae_sum": 0.0} for _ in range(C)]

    for d in per_batch_list:
        for ci in d["per_ch"]:
            c = ci["ch"]
            n = ci["n"]
            if n <= 0:
                continue
            acc[c]["n"] += n
            acc[c]["mse_sum"] += ci["mse"] * n
            acc[c]["mae_sum"] += ci["mae"] * n

    out = []
    for c in range(C):
        n = acc[c]["n"]
        if n == 0:
            out.append(dict(ch=c, n=0, mse=np.nan, mae=np.nan, rmse=np.nan))
        else:
            mse = acc[c]["mse_sum"] / n
            mae = acc[c]["mae_sum"] / n
            out.append(dict(ch=c, n=n, mse=float(mse), mae=float(mae), rmse=float(np.sqrt(mse))))
    return out


# ------------------------- main -------------------------
if __name__ == "__main__":
    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    ckpt_path = os.path.join(args.save, f"experiment_{experimentID}.ckpt")

    # rebuild command string without --load <id>
    input_command = sys.argv[:]
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        i = ind[0]
        input_command = input_command[:i] + input_command[i + 2 :]
    input_command = " ".join(input_command)

    LOG_DIR = ensure_dir("analyzelogs")

    run_tag = f"{args.dataset}_{slurm_job_name}_job{slurm_job_id}"
    base = os.path.join(LOG_DIR, run_tag)

    run_logger = build_file_logger("run_logger", base + ".run.log", mode=args.logmode, level=logging.INFO)
    train_logger = build_file_logger("train_logger", base + ".train.log", mode=args.logmode, level=logging.INFO)
    sys_logger = build_file_logger("sys_logger", base + ".system.log", mode=args.logmode, level=logging.INFO)
    top_logger = build_file_logger("top_logger", base + ".toperr.log", mode=args.logmode, level=logging.INFO)
    err_logger = build_file_logger("err_logger", base + ".error.log", mode=args.logmode, level=logging.INFO)

    add_stdout_handler(run_logger, level=logging.INFO)
    add_stdout_handler(train_logger, level=logging.INFO)

    run_logger.info(f"TARGET_CHANNEL | target_channel={args.target_channel} (-1 means all channels)")

    # ------------------------- data loading -------------------------
    try:
        data_obj = parse_datasets(args, patch_ts=True)
        try:
            dmin = data_obj.get("data_min", None)
            dmax = data_obj.get("data_max", None)
            tmax = data_obj.get("time_max", None)

            def _tensor_stats(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    xx = x.detach().cpu().float()
                    return dict(
                        shape=tuple(xx.shape),
                        min=float(xx.min().item()),
                        max=float(xx.max().item()),
                        mean=float(xx.mean().item()),
                    )
                return dict(shape=(), min=float(x), max=float(x), mean=float(x))

            smin = _tensor_stats(dmin)
            smax = _tensor_stats(dmax)
            stmax = _tensor_stats(tmax)

            run_logger.info(f"DATA_STATS | dataset={args.dataset}")
            run_logger.info(f"DATA_STATS | data_min: {smin}")
            run_logger.info(f"DATA_STATS | data_max: {smax}")
            run_logger.info(f"DATA_STATS | time_max: {stmax}")

            if isinstance(dmin, torch.Tensor) and dmin.numel() > 1:
                N = min(10, dmin.numel())
                run_logger.info(
                    f"DATA_STATS | first_{N}_dims data_min={dmin[:N].detach().cpu().tolist()} "
                    f"data_max={dmax[:N].detach().cpu().tolist()}"
                )

        except Exception as e:
            err_logger.warning(f"DATA_STATS_LOG_FAIL: {repr(e)}")
    except Exception:
        err_logger.error("FAILED during parse_datasets()")
        err_logger.error(traceback.format_exc())
        raise

    input_dim = data_obj["input_dim"]
    args.ndim = input_dim

    # ------------------------- model setup -------------------------
    from copy import deepcopy

    use_ms = args.multi_scales not in (None, "", [])

    try:
        if use_ms:
            from model.multiscale_tpatchgnn import MultiScaleTPatchGNN

            # IMPORTANT: take first_batch BEFORE TS_DEBUG sampling
            first_batch = utils.get_next_batch(data_obj["train_dataloader"])
            run_logger.info(
                f"[MS] use_ms={use_ms} fusion={args.fusion} "
                f"scales={args.multi_scales or 'single-scale'} "
                f"strides={(args.multi_strides or args.multi_scales) or 'same-as-scales'} "
                f"npatches_per_scale={list(map(int, first_batch['npatches']))}"
            )

            submodels = []
            for M_k in first_batch["npatches"]:
                sub_args = deepcopy(args)
                sub_args.npatch = int(M_k)
                submodels.append(tPatchGNN(sub_args, supports=None, dropout=0).to(args.device))

            model = MultiScaleTPatchGNN(
                submodels=submodels,
                te_dim=args.te_dim,
                proj_dim=args.hid_dim,
                fusion=args.fusion,
            ).to(args.device)

            # ------------------------- LAZY INIT FIX: force-build lazy modules BEFORE optimizer -------------------------
            model.train()
            with torch.no_grad():
                _ = model(
                    first_batch["X_list"],
                    first_batch["tt_list"],
                    first_batch["mk_list"],
                    first_batch["tp_to_predict"],
                )
            run_logger.info(
                f"[MS_INIT] forced forward done. decoder_is_none={getattr(model, 'decoder', None) is None}"
            )

        else:
            model = tPatchGNN(args).to(args.device)

    except Exception:
        err_logger.error("FAILED during model construction")
        err_logger.error(traceback.format_exc())
        raise

    # ------------------------- TS_DEBUG (after model construction) -------------------------
    _debug_time_scale_consistency(run_logger, data_obj, "train", "train_dataloader", n_batches=2)
    _debug_time_scale_consistency(run_logger, data_obj, "val", "val_dataloader", n_batches=2)
    _debug_time_scale_consistency(run_logger, data_obj, "test", "test_dataloader", n_batches=2)
    _debug_value_ranges(run_logger, data_obj, "train", "train_dataloader", n_batches=1)
    _debug_value_ranges(run_logger, data_obj, "val", "val_dataloader", n_batches=1)
    _debug_value_ranges(run_logger, data_obj, "test", "test_dataloader", n_batches=1)

    # ------------------------- RUN header logs -------------------------
    host = socket.gethostname()
    run_logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run_logger.info(f"command: {input_command}")
    run_logger.info(
        f"ExpID={experimentID} PID={args.PID} host={host} device={args.device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    run_logger.info(f"SLURM_JOB_NAME={slurm_job_name} SLURM_JOB_ID={slurm_job_id}")
    run_logger.info(f"ckpt_path={ckpt_path}")
    run_logger.info(f"args={args}")

    run_logger.info(
        f"data: input_dim={input_dim} "
        f"n_train_batches={data_obj.get('n_train_batches')} "
        f"n_val_batches={data_obj.get('n_val_batches')} "
        f"n_test_batches={data_obj.get('n_test_batches')} "
        f"quantization={args.quantization} history={args.history} patch_size={args.patch_size} stride={args.stride} npatch={args.npatch}"
    )

    # data sanity
    _sample_data_sanity_main(
        run_logger, data_obj, "train", "train_dataloader", "n_train_batches", args.data_sanity_batches
    )
    _sample_data_sanity_main(run_logger, data_obj, "val", "val_dataloader", "n_val_batches", args.data_sanity_batches)
    _sample_data_sanity_main(
        run_logger, data_obj, "test", "test_dataloader", "n_test_batches", args.data_sanity_batches
    )

    # ------------------------- optimizer + scheduler -------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    # ------------------------- OPT CHECK: ensure decoder params are included -------------------------
    if use_ms and hasattr(model, "decoder") and (model.decoder is not None):
        dec_params = set(id(p) for p in model.decoder.parameters())
        opt_params = set(id(p) for g in optimizer.param_groups for p in g["params"])
        missing = dec_params - opt_params
        run_logger.info(
            f"[OPT_CHECK] decoder_params={len(dec_params)} in_optimizer={len(dec_params) - len(missing)} missing={len(missing)}"
        )
        if len(missing) > 0:
            run_logger.warning("[OPT_CHECK][WARNING] decoder params are NOT in optimizer! (should be missing=0)")

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        cooldown=1,
        min_lr=1e-6,
        verbose=True,
    )

    num_batches = int(data_obj["n_train_batches"])
    print("n_train_batches:", num_batches)

    best_val_mse = np.inf
    test_res = None
    best_iter = 0

    last_test_total_points = 0
    last_test_correct_points = 0
    last_test_right_rate = np.nan

    # ------------------------- training loop -------------------------
    for itr in range(args.epoch):
        st_epoch = time.time()
        nan_loss_flag = False
        nan_pred_flag = False
        nan_grad_flag = False

        # ---- TRAIN ----
        model.train()
        last_train_loss = None
        last_mask_ratio = np.nan
        grad_norm = np.nan

        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

            try:
                if use_ms:
                    out = model(
                        batch_dict["X_list"],
                        batch_dict["tt_list"],
                        batch_dict["mk_list"],
                        batch_dict["tp_to_predict"],
                    )
                    pred = out[0]  # (B, Lp, C)
                    tgt = batch_dict["data_to_predict"]
                    msk = batch_dict["mask_predicted_data"]

                    # OPTIONAL: supervise only one channel (e.g., temperature ch=0)
                    pred_s, tgt_s, msk_s = _slice_target_channel(pred, tgt, msk, args.target_channel)

                    mb = msk_s.bool() if msk_s.dtype == torch.bool else (msk_s > 0.5)
                    if mb.sum().item() == 0:
                        err_logger.warning(f"MASK_EMPTY | train | epoch={itr} (skipping loss update for this batch)")
                        continue

                    loss = ((pred_s - tgt_s)[mb] ** 2).mean()
                    if not torch.isfinite(loss):
                        nan_loss_flag = True
                        err_logger.error(f"LOSS_NAN_INF | train | epoch={itr} loss={safe_float(loss)}")
                        continue

                    loss.backward()

                    grad_norm = compute_grad_norm(model)
                    if not np.isfinite(grad_norm):
                        nan_grad_flag = True
                        err_logger.error(f"GRAD_NAN_INF | train | epoch={itr}")

                    optimizer.step()
                    last_train_loss = loss.detach()
                    last_mask_ratio = summarize_mask(msk_s)["mask_ratio"]

                    if not torch.isfinite(pred_s).all():
                        nan_pred_flag = True
                        err_logger.warning(f"PRED_NAN_INF | train | epoch={itr}")

                else:
                    from lib.utils import compute_all_losses  # adjust if needed

                    # NOTE: single-scale path: if you need target_channel here too, you must modify compute_all_losses
                    # to slice pred/tgt/mask internally. For now we keep SS as-is.
                    train_res = compute_all_losses(model, batch_dict)
                    loss = train_res["loss"]

                    if not torch.isfinite(loss):
                        nan_loss_flag = True
                        err_logger.error(f"LOSS_NAN_INF | train | epoch={itr} loss={safe_float(loss)}")
                        continue

                    loss.backward()
                    grad_norm = compute_grad_norm(model)
                    if not np.isfinite(grad_norm):
                        nan_grad_flag = True
                        err_logger.error(f"GRAD_NAN_INF | train | epoch={itr}")

                    optimizer.step()
                    last_train_loss = loss.detach()

            except Exception:
                err_logger.error(f"EXCEPTION | train | epoch={itr}")
                err_logger.error(traceback.format_exc())
                raise

        # ---- VAL / TEST ----
        model.eval()
        improved = False

        try:
            with torch.no_grad():
                if use_ms:
                    # VAL aggregate
                    val_logs = []
                    for _ in range(int(data_obj["n_val_batches"])):
                        b = utils.get_next_batch(data_obj["val_dataloader"])
                        out = model(b["X_list"], b["tt_list"], b["mk_list"], b["tp_to_predict"])
                        pred = out[0]
                        tgt = b["data_to_predict"]
                        msk = b["mask_predicted_data"]

                        pred_s, tgt_s, msk_s = _slice_target_channel(pred, tgt, msk, args.target_channel)
                        val_logs.append(_masked_metrics(pred_s, tgt_s, msk_s))

                    val_res = (
                        {k: float(np.mean([d[k] for d in val_logs])) for k in val_logs[0].keys()}
                        if len(val_logs) > 0
                        else dict(loss=np.nan, mse=np.nan, rmse=np.nan, mae=np.nan, mape=np.nan)
                    )

                    # TEST when improved
                    if np.isfinite(val_res["mse"]) and val_res["mse"] < best_val_mse:
                        improved = True
                        best_val_mse = val_res["mse"]
                        best_iter = itr

                        test_logs = []
                        test_ch_logs = []  # for channel influence
                        best_test_batch_for_top = None

                        test_total_points = 0
                        test_correct_points = 0
                        correct_examples = []

                        for _ in range(int(data_obj["n_test_batches"])):
                            b = utils.get_next_batch(data_obj["test_dataloader"])
                            out = model(b["X_list"], b["tt_list"], b["mk_list"], b["tp_to_predict"])
                            pred = out[0]
                            tgt = b["data_to_predict"]
                            msk = b["mask_predicted_data"]

                            # keep full tensors for influence ranking (unless you force single-channel)
                            pred_s, tgt_s, msk_s = _slice_target_channel(pred, tgt, msk, args.target_channel)

                            if best_test_batch_for_top is None:
                                best_test_batch_for_top = (pred_s, tgt_s, msk_s, b.get("tp_to_predict", None))

                            tot, cor = count_right_points_tol(pred_s, tgt_s, msk_s, abs_tol=1e-5, rel_tol=0.05)
                            test_total_points += tot
                            test_correct_points += cor

                            if len(correct_examples) < 3:
                                correct_examples.extend(
                                    _collect_correct_examples_exact(
                                        pred=pred_s,
                                        tgt=tgt_s,
                                        msk=msk_s,
                                        tp_to_predict=b.get("tp_to_predict", None),
                                        max_k=(3 - len(correct_examples)),
                                    )
                                )

                            test_logs.append(_masked_metrics(pred_s, tgt_s, msk_s))

                            # channel influence only makes sense when NOT forcing a single channel
                            if args.target_channel < 0:
                                test_ch_logs.append(_masked_metrics_per_channel(pred, tgt, msk))

                        test_res = (
                            {k: float(np.mean([d[k] for d in test_logs])) for k in test_logs[0].keys()}
                            if len(test_logs) > 0
                            else None
                        )

                        last_test_total_points = test_total_points
                        last_test_correct_points = test_correct_points
                        last_test_right_rate = test_correct_points / max(test_total_points, 1)

                        # log top errors on TEST only (on supervised tensor)
                        if best_test_batch_for_top is not None:
                            p, t, m, tp = best_test_batch_for_top
                            log_topk_errors(top_logger, "test", itr, p, t, m, tp_to_predict=tp, topk=args.topk_err)
                            log_topk_best(top_logger, "test", itr, p, t, m, tp_to_predict=tp, topk=3)

                        # log top 3 exact matches
                        if len(correct_examples) == 0:
                            top_logger.info(f"TOP_OK | split=test epoch={itr} none_found (no exact matches)")
                        else:
                            for rank, ex in enumerate(correct_examples, start=1):
                                if ex["tp"] is None:
                                    top_logger.info(
                                        f"TOP_OK | split=test epoch={itr} rank={rank}/3 "
                                        f"b={ex['b']} t={ex['t']} var_idx={ex['var_idx']} "
                                        f"y_true={ex['y_true']:.6f} y_pred={ex['y_pred']:.6f}"
                                    )
                                else:
                                    top_logger.info(
                                        f"TOP_OK | split=test epoch={itr} rank={rank}/3 "
                                        f"b={ex['b']} t={ex['t']} var_idx={ex['var_idx']} tp={ex['tp']:.6f} "
                                        f"y_true={ex['y_true']:.6f} y_pred={ex['y_pred']:.6f}"
                                    )

                        # -------- Channel influence (which channel contributes most to error) --------
                        if args.target_channel < 0 and len(test_ch_logs) > 0:
                            per_ch = _aggregate_per_channel(test_ch_logs)
                            if per_ch is not None:
                                per_ch_sorted = sorted(
                                    [x for x in per_ch if np.isfinite(x["mae"])],
                                    key=lambda z: z["mae"],
                                    reverse=True,
                                )
                                top_logger.info(
                                    f"[CH_INFLUENCE] TEST per-channel masked error ranked by MAE (epoch={itr})"
                                )
                                for r, x in enumerate(per_ch_sorted, start=1):
                                    top_logger.info(
                                        f"[CH_INFLUENCE] rank={r}/{len(per_ch_sorted)} ch={x['ch']} "
                                        f"n={x['n']} mae={x['mae']:.6f} rmse={x['rmse']:.6f} mse={x['mse']:.6f}"
                                    )
                                worst = per_ch_sorted[0] if len(per_ch_sorted) else None
                                if worst is not None:
                                    train_logger.info(
                                        f"Test - WorstChannelByMAE: ch={worst['ch']} mae={worst['mae']:.6f} (n={worst['n']})"
                                    )

                else:
                    from lib.utils import evaluation  # adjust if needed

                    val_res = evaluation(model, data_obj["val_dataloader"], int(data_obj["n_val_batches"]))
                    if np.isfinite(val_res["mse"]) and val_res["mse"] < best_val_mse:
                        improved = True
                        best_val_mse = val_res["mse"]
                        best_iter = itr
                        test_res = evaluation(model, data_obj["test_dataloader"], int(data_obj["n_test_batches"]))

        except Exception:
            err_logger.error(f"EXCEPTION | eval | epoch={itr}")
            err_logger.error(traceback.format_exc())
            raise

        # ------------------------- LR scheduler step -------------------------
        metric_for_lr = float(val_res["mse"])
        if np.isfinite(metric_for_lr):
            lr_scheduler.step(metric_for_lr)
        else:
            err_logger.warning(f"LR_SCHED_SKIP | epoch={itr} metric_for_lr={metric_for_lr} (non-finite)")

        # ---- LOGGING: training category file ----
        lr_now = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else np.nan
        epoch_time = time.time() - st_epoch
        train_loss_val = safe_float(last_train_loss)

        train_logger.info(f"- Epoch {itr:03d}, ExpID {experimentID} best_epoch={best_iter} improved={int(improved)}")
        train_logger.info(f"Train - Loss (last batch): {train_loss_val:.5f}")

        train_logger.info(
            "Val - Loss, MSE, MAE: {:.5f}, {:.5f}, {:.5f}".format(
                float(val_res["loss"]), float(val_res["mse"]), float(val_res["mae"])
            )
        )

        if test_res is not None:
            train_logger.info(
                "Test - Best epoch, Loss, MSE, MAE: {}, {:.5f}, {:.5f}, {:.5f}".format(
                    best_iter, float(test_res["loss"]), float(test_res["mse"]), float(test_res["mae"])
                )
            )
            train_logger.info(
                f"Test - Points (EXACT): total_masked={last_test_total_points} correct={last_test_correct_points} "
                f"right_rate={last_test_right_rate*100:.4f}%"
            )

        # ---- LOGGING: system/perf category file ----
        alloc, reserved, max_alloc = cuda_mem_stats()
        sys_logger.info(
            f"SYSTEM | epoch={itr} time={epoch_time:.2f}s lr={lr_now:.6g} grad_norm={grad_norm:.6f} "
            f"mask_ratio_lastbatch={last_mask_ratio if np.isfinite(last_mask_ratio) else np.nan:.6f} "
            f"cuda_mem_alloc_mb={alloc:.2f} cuda_mem_reserved_mb={reserved:.2f} cuda_mem_max_alloc_mb={max_alloc:.2f}"
        )

        # ---- LOGGING: errors/anomalies category file ----
        if nan_loss_flag or nan_pred_flag or nan_grad_flag:
            err_logger.warning(
                f"ANOMALY_FLAGS | epoch={itr} loss_nan={int(nan_loss_flag)} pred_nan={int(nan_pred_flag)} grad_nan={int(nan_grad_flag)}"
            )

        if (itr - best_iter) >= args.patience:
            run_logger.info(f"EARLY_STOP | epoch={itr} best_epoch={best_iter} patience={args.patience}")
            print("Exp has been early stopped!")
            sys.exit(0)
