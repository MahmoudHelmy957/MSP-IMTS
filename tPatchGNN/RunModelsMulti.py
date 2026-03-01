import os
import sys
from turtle import st

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
from lib.evaluation import compute_all_losses, evaluation  



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
    top_logger = build_file_logger("top_logger", base + ".toperr.log", mode=args.logmode, level=logging.INFO)

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
    try:
        from model.multiscale_tpatchgnn import MultiScaleTPatchGNN

        # IMPORTANT: take first_batch BEFORE TS_DEBUG sampling
        first_batch = utils.get_next_batch(data_obj["train_dataloader"])
        run_logger.info(
            f"[MS] fusion={args.fusion} "
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
    if hasattr(model, "decoder") and (model.decoder is not None):
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
                # ------------------------------------------------------------------
                # Use compute_all_losses (from lib/likelihood_eval.py) which calls
                # model.forecasting() internally and returns mse/rmse/mae + loss.
                # ------------------------------------------------------------------
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

                # mask_ratio and pred NaN checks kept for diagnostics;
                # we re-run a cheap forward only if needed for logging flags.
                # pred NaN check: inspect model output on the same batch
                with torch.no_grad():
                    out = model(
                        batch_dict["X_list"],
                        batch_dict["tt_list"],
                        batch_dict["mk_list"],
                        batch_dict["tp_to_predict"],
                    )
                    pred = out[0]
                    msk_s = batch_dict["mask_predicted_data"]
                    last_mask_ratio = summarize_mask(msk_s)["mask_ratio"]

                    if not torch.isfinite(pred).all():
                        nan_pred_flag = True
                        err_logger.warning(f"PRED_NAN_INF | train | epoch={itr}")

            except Exception:
                err_logger.error(f"EXCEPTION | train | epoch={itr}")
                err_logger.error(traceback.format_exc())
                raise

        # ---- VAL ----
        # Use evaluation() which iterates the full dataloader for n_val_batches
        # and returns aggregated mse/rmse/mae/mape.
        model.eval()
        improved = False

        try:
            with torch.no_grad():
                val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])

            if np.isfinite(val_res["mse"]) and val_res["mse"] < best_val_mse:
                improved = True
                best_val_mse = val_res["mse"]
                best_iter = itr

                # ---- TEST ----
                # Also use evaluation() for test set.
                with torch.no_grad():
                    test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])

                # Per-channel and exact-point stats still require the raw forward pass;
                # keep that secondary loop unchanged in logic but now separate from
                # the primary metric computation above.
                test_total_points = 0
                test_correct_points = 0

                with torch.no_grad():
                    for _ in range(int(data_obj["n_test_batches"])):
                        b = utils.get_next_batch(data_obj["test_dataloader"])
                        out = model(b["X_list"], b["tt_list"], b["mk_list"], b["tp_to_predict"])
                        pred = out[0]
                        tgt = b["data_to_predict"]
                        msk = b["mask_predicted_data"]

                        tot, cor = count_right_points_tol(
                            pred, tgt, msk, abs_tol=args.exact_abs_tol, rel_tol=args.exact_rel_tol
                        )
                        test_total_points += tot
                        test_correct_points += cor

                last_test_total_points = test_total_points
                last_test_correct_points = test_correct_points
                last_test_right_rate = test_correct_points / max(test_total_points, 1)

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

        train_logger.info(
            "Train - Loss (one batch): {:.5f}".format(train_loss_val))

        train_logger.info(
            "Val - Loss, MSE, RMSE, MAE: {:.5f}, {:.5f}, {:.5f}, {:.5f}"
            .format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"])
        )

        if test_res is not None:
            train_logger.info(
                "Test - Best epoch {}, Loss, MSE, RMSE, MAE: {:.5f}, {:.5f}, {:.5f}, {:.5f}"
                .format(best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"])
            )

        train_logger.info("Time spent: {:.2f}s".format(time.time() - st_epoch))

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