import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import time
import datetime
import numpy as np
from random import SystemRandom
import socket
import traceback
import logging
import torch
import torch.optim as optim
from copy import deepcopy
import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import tPatchGNN
from lib.analyzelogs import *
from lib.cli_args import build_multi_scale_parser

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


def _rebuild_command_without_load(argv):
    argv = list(argv)
    idx = [i for i in range(len(argv)) if argv[i] == "--load"]
    if len(idx) == 1:
        i = idx[0]
        argv = argv[:i] + argv[i + 2:]
    return " ".join(argv)

# ------------------------- main -------------------------
if __name__ == "__main__":

    parser = build_multi_scale_parser()
    args = parser.parse_args()

    utils.setup_seed(args.seed)

    # compute npatch for single-scale fallback (kept; does not change algorithm)
    args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1

    # device + pid (kept; your script uses args.device / args.PID later)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.PID = os.getpid()
    print("PID, device:", args.PID, args.device)

    # SLURM identifiers (for log naming)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", str(args.PID))
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "local")

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    ckpt_path = os.path.join(args.save, f"experiment_{experimentID}.ckpt")

    # rebuild command string without --load <id>
    input_command = _rebuild_command_without_load(sys.argv)

    LOG_DIR = ensure_dir("analyzelogs")

    run_tag = f"{args.dataset}_{slurm_job_name}_job{slurm_job_id}"
    base = os.path.join(LOG_DIR, run_tag)

    run_logger = build_file_logger("run_logger", base + ".run.log", mode=args.logmode, level=logging.INFO)
    train_logger = build_file_logger("train_logger", base + ".train.log", mode=args.logmode, level=logging.INFO)
    sys_logger = build_file_logger("sys_logger", base + ".system.log", mode=args.logmode, level=logging.INFO)
    err_logger = build_file_logger("err_logger", base + ".error.log", mode=args.logmode, level=logging.INFO)

    # NOTE: removed top_logger + TOP error logging + data sanity checks (algorithm unchanged)

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

                    if np.isfinite(val_res["mse"]) and val_res["mse"] < best_val_mse:
                        improved = True
                        best_val_mse = val_res["mse"]
                        best_iter = itr

                        test_logs = []
                        test_ch_logs = []
                        test_total_points = 0
                        test_correct_points = 0

                        for _ in range(int(data_obj["n_test_batches"])):
                            b = utils.get_next_batch(data_obj["test_dataloader"])
                            out = model(b["X_list"], b["tt_list"], b["mk_list"], b["tp_to_predict"])
                            pred = out[0]
                            tgt = b["data_to_predict"]
                            msk = b["mask_predicted_data"]

                            pred_s, tgt_s, msk_s = _slice_target_channel(pred, tgt, msk, args.target_channel)

                            tot, cor = count_right_points_tol(
                                pred_s, tgt_s, msk_s, abs_tol=args.exact_abs_tol, rel_tol=args.exact_rel_tol
                            )
                            test_total_points += tot
                            test_correct_points += cor

                            test_logs.append(_masked_metrics(pred_s, tgt_s, msk_s))

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

                        # NOTE: removed TOP error logs + exact-match example logs only (algorithm unchanged)

                        if args.target_channel < 0 and len(test_ch_logs) > 0:
                            per_ch = _aggregate_per_channel(test_ch_logs)
                            if per_ch is not None:
                                per_ch_sorted = sorted(
                                    [x for x in per_ch if np.isfinite(x["mae"])],
                                    key=lambda z: z["mae"],
                                    reverse=True,
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

        # ---- LOGGING: errors/anomalies category file ----
        if nan_loss_flag or nan_pred_flag or nan_grad_flag:
            err_logger.warning(
                f"ANOMALY_FLAGS | epoch={itr} loss_nan={int(nan_loss_flag)} pred_nan={int(nan_pred_flag)} grad_nan={int(nan_grad_flag)}"
            )

        if (itr - best_iter) >= args.patience:
            run_logger.info(f"EARLY_STOP | epoch={itr} best_epoch={best_iter} patience={args.patience}")
            print("Exp has been early stopped!")
            sys.exit(0)