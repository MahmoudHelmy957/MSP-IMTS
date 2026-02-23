# tpatchGNN/RunModelsSingle.py

import os
import sys

# Ensure we can import sibling folder "lib" and the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import time
import datetime
import numpy as np
from random import SystemRandom
import socket
import traceback

import torch
import torch.optim as optim

import lib.utils as utils
from lib.cli_args import build_single_scale_parser
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *  # tPatchGNN + evaluation (we won't use evaluation here)

from lib.analyzelogs import (
    setup_loggers,
    safe_float,
    compute_grad_norm,
    cuda_mem_stats,
    summarize_mask,
    sample_data_sanity,
    log_topk_errors,
    log_topk_best,
    count_right_points_tol,
)

from lib.ss_forecast import get_pred_ss, ensure_3d_B_Lp_N, move_like
from lib.train_losses import masked_global_mse_loss
from lib.exact_metrics import collect_topk_exact_examples


# ------------------------- helpers -------------------------
def denorm_minmax(x, data_min, data_max):
    # x: (B, Lp, D)
    # data_min/max: (D,)
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    dmin = data_min.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
    dmax = data_max.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
    return x * (dmax - dmin) + dmin


def masked_global_mse_value(pred, tgt, msk):
    """
    pred/tgt/msk: (B, Lp, D)
    returns float (global masked MSE) or np.nan if empty mask
    """
    pred = ensure_3d_B_Lp_N(pred)
    tgt = ensure_3d_B_Lp_N(tgt)
    msk = ensure_3d_B_Lp_N(msk)

    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
    cnt = int(mb.sum().item())
    if cnt <= 0:
        return np.nan
    diff2 = (pred - tgt) ** 2
    return float(diff2[mb].mean().item())


def _rebuild_command_without_load(argv):
    argv = list(argv)
    idx = [i for i in range(len(argv)) if argv[i] == "--load"]
    if len(idx) == 1:
        i = idx[0]
        argv = argv[:i] + argv[i + 2:]
    return " ".join(argv)


# ------------------------- main -------------------------
if __name__ == "__main__":
    # ------------------------- CLI -------------------------
    parser = build_single_scale_parser()
    args = parser.parse_args()

    # Derived
    args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.PID = os.getpid()
    print("PID, device:", args.PID, args.device)

    # SLURM identifiers (for log naming)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", str(args.PID))
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "local")

    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    ckpt_path = os.path.join(args.save, f"experiment_{experimentID}.ckpt")
    input_command = _rebuild_command_without_load(sys.argv)

    # ------------------------- Loggers -------------------------
    run_tag = f"{args.dataset}_{slurm_job_name}_job{slurm_job_id}"
    _, loggers = setup_loggers(
        log_dir_rel="analyzelogs",
        run_tag=run_tag,
        logmode=args.logmode,
        stdout_run=True,
        stdout_train=True,
    )
    run_logger = loggers["run"]
    train_logger = loggers["train"]
    sys_logger = loggers["system"]
    err_logger = loggers["error"]
    top_logger = loggers["toperr"]

    host = socket.gethostname()
    run_logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run_logger.info(f"command: {input_command}")
    run_logger.info(
        f"ExpID={experimentID} PID={args.PID} host={host} device={args.device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    run_logger.info(f"SLURM_JOB_NAME={slurm_job_name} SLURM_JOB_ID={slurm_job_id}")
    run_logger.info(f"ckpt_path={ckpt_path}")
    run_logger.info(f"args={args}")
    run_logger.info("LOSS_MODE | TRAIN/VAL/TEST uses GLOBAL masked MSE: mean((pred-true)^2 over ALL masked points)")
    run_logger.info("METRICS | We report BOTH scaled (normalized) and real-unit (denormalized) MSE for val/test.")

    # ------------------------- Data -------------------------
    try:
        data_obj = parse_datasets(args, patch_ts=True)
        data_min = data_obj["data_min"]
        data_max = data_obj["data_max"]
    except Exception:
        err_logger.error("FAILED during parse_datasets()")
        err_logger.error(traceback.format_exc())
        raise

    input_dim = data_obj["input_dim"]
    args.ndim = input_dim

    run_logger.info(
        f"data: input_dim={input_dim} "
        f"n_train_batches={data_obj.get('n_train_batches')} "
        f"n_val_batches={data_obj.get('n_val_batches')} "
        f"n_test_batches={data_obj.get('n_test_batches')} "
        f"quantization={args.quantization} history={args.history} patch_size={args.patch_size} "
        f"stride={args.stride} npatch={args.npatch}"
    )

    # ------------------------- Data sanity -------------------------
    try:
        batch_fetch_fn = lambda dobj, key: utils.get_next_batch(dobj[key])
        sample_data_sanity(
            run_logger, data_obj, "train", "train_dataloader", "n_train_batches",
            args.data_sanity_batches, batch_fetch_fn
        )
        sample_data_sanity(
            run_logger, data_obj, "val", "val_dataloader", "n_val_batches",
            args.data_sanity_batches, batch_fetch_fn
        )
        sample_data_sanity(
            run_logger, data_obj, "test", "test_dataloader", "n_test_batches",
            args.data_sanity_batches, batch_fetch_fn
        )
    except Exception:
        err_logger.warning("DATA_SANITY failed (continuing)")
        err_logger.warning(traceback.format_exc())

    # ------------------------- Model -------------------------
    try:
        model = tPatchGNN(args).to(args.device)
    except Exception:
        err_logger.error("FAILED during model construction")
        err_logger.error(traceback.format_exc())
        raise

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    num_batches = int(data_obj["n_train_batches"])
    run_logger.info(f"n_train_batches: {num_batches}")

    best_val_mse_real = np.inf
    best_iter = 0

    # cached best test results (real units)
    best_test_mse_real = np.nan
    best_test_mae_real = np.nan

    last_test_total_points = 0
    last_test_correct_points = 0
    last_test_right_rate = np.nan

    # ------------------------- training loop -------------------------
    for itr in range(args.epoch):
        st_epoch = time.time()
        nan_loss_flag = False
        nan_grad_flag = False

        model.train()
        last_train_loss = None
        last_mask_ratio = np.nan
        grad_norm = np.nan

        # ---- train batches ----
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

            try:
                pred = get_pred_ss(model, batch_dict, args.device)
                tgt = batch_dict["data_to_predict"]
                msk = batch_dict["mask_predicted_data"]

                pred = ensure_3d_B_Lp_N(pred)
                tgt = ensure_3d_B_Lp_N(tgt)
                msk = ensure_3d_B_Lp_N(msk)

                tgt = move_like(tgt, pred)
                msk = move_like(msk, pred)

                # TRAIN LOSS (scaled / normalized) = global masked MSE
                loss = masked_global_mse_loss(pred, tgt, msk)
                if loss is None:
                    err_logger.warning(f"EMPTY_MASK | train | epoch={itr} skipping batch")
                    continue

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
                if msk is not None:
                    last_mask_ratio = summarize_mask(msk)["mask_ratio"]

            except Exception:
                err_logger.error(f"EXCEPTION | train | epoch={itr}")
                err_logger.error(traceback.format_exc())
                raise

        # ------------------------- VAL / TEST (NO evaluation()) -------------------------
        model.eval()

        # val metrics (scaled + real)
        val_mses_scaled = []
        val_mses_real = []
        val_maes_real = []

        try:
            with torch.no_grad():
                # --- VAL LOOP ---
                n_val_batches = int(data_obj["n_val_batches"])
                for _ in range(n_val_batches):
                    b = utils.get_next_batch(data_obj["val_dataloader"])

                    pred = get_pred_ss(model, b, args.device)
                    tgt = b["data_to_predict"]
                    msk = b["mask_predicted_data"]

                    pred = ensure_3d_B_Lp_N(pred)
                    tgt = ensure_3d_B_Lp_N(tgt)
                    msk = ensure_3d_B_Lp_N(msk)

                    tgt = move_like(tgt, pred)
                    msk = move_like(msk, pred)

                    # scaled MSE
                    v_mse_scaled = masked_global_mse_value(pred, tgt, msk)
                    if np.isfinite(v_mse_scaled):
                        val_mses_scaled.append(v_mse_scaled)

                    # real-unit metrics (denormalize)
                    pred_o = denorm_minmax(pred, data_min, data_max)
                    tgt_o  = denorm_minmax(tgt,  data_min, data_max)

                    v_mse_real = masked_global_mse_value(pred_o, tgt_o, msk)
                    if np.isfinite(v_mse_real):
                        val_mses_real.append(v_mse_real)

                    # real-unit MAE (global masked)
                    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
                    if int(mb.sum().item()) > 0:
                        val_maes_real.append(float((pred_o - tgt_o).abs()[mb].mean().item()))

                val_mse_scaled = float(np.mean(val_mses_scaled)) if len(val_mses_scaled) else np.nan
                val_mse_real   = float(np.mean(val_mses_real))   if len(val_mses_real)   else np.nan
                val_mae_real   = float(np.mean(val_maes_real))   if len(val_maes_real)   else np.nan

                improved = np.isfinite(val_mse_real) and (val_mse_real < best_val_mse_real)
                if improved:
                    best_val_mse_real = float(val_mse_real)
                    best_iter = itr

                    # --- TEST LOOP (only when improved) ---
                    test_mses_scaled = []
                    test_mses_real = []
                    test_maes_real = []

                    best_test_batch_for_top = None
                    test_total_points = 0
                    test_correct_points = 0
                    correct_examples = []

                    n_test_batches = int(data_obj["n_test_batches"])
                    for _ in range(n_test_batches):
                        b = utils.get_next_batch(data_obj["test_dataloader"])

                        pred = get_pred_ss(model, b, args.device)
                        tgt = b["data_to_predict"]
                        msk = b["mask_predicted_data"]
                        tp  = b.get("tp_to_predict", None)

                        pred = ensure_3d_B_Lp_N(pred)
                        tgt = ensure_3d_B_Lp_N(tgt)
                        msk = ensure_3d_B_Lp_N(msk)

                        tgt = move_like(tgt, pred)
                        msk = move_like(msk, pred)

                        # scaled test MSE
                        t_mse_scaled = masked_global_mse_value(pred, tgt, msk)
                        if np.isfinite(t_mse_scaled):
                            test_mses_scaled.append(t_mse_scaled)

                        # denormalize for real-unit test metrics + logging
                        pred_o = denorm_minmax(pred, data_min, data_max)
                        tgt_o  = denorm_minmax(tgt,  data_min, data_max)

                        t_mse_real = masked_global_mse_value(pred_o, tgt_o, msk)
                        if np.isfinite(t_mse_real):
                            test_mses_real.append(t_mse_real)

                        mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
                        if int(mb.sum().item()) > 0:
                            test_maes_real.append(float((pred_o - tgt_o).abs()[mb].mean().item()))

                        # Store denormalized batch for top-err logging
                        if best_test_batch_for_top is None:
                            best_test_batch_for_top = (pred_o, tgt_o, msk, tp)

                        # EXACT / TopK on DENORMALIZED values
                        tot, cor = count_right_points_tol(
                            pred_o, tgt_o, msk,
                            abs_tol=args.exact_abs_tol,
                            rel_tol=args.exact_rel_tol
                        )
                        test_total_points += tot
                        test_correct_points += cor

                        if len(correct_examples) < 3:
                            correct_examples.extend(
                                collect_topk_exact_examples(
                                    pred_o, tgt_o, msk,
                                    tp_to_predict=tp,
                                    abs_tol=args.exact_abs_tol,
                                    rel_tol=args.exact_rel_tol,
                                    k=3 - len(correct_examples),
                                )
                            )

                    last_test_total_points = int(test_total_points)
                    last_test_correct_points = int(test_correct_points)
                    last_test_right_rate = float(test_correct_points / max(test_total_points, 1))

                    best_test_mse_scaled = float(np.mean(test_mses_scaled)) if len(test_mses_scaled) else np.nan
                    best_test_mse_real   = float(np.mean(test_mses_real))   if len(test_mses_real)   else np.nan
                    best_test_mae_real   = float(np.mean(test_maes_real))   if len(test_maes_real)   else np.nan

                    # cache best test real-unit metrics
                    best_test_mse_real = best_test_mse_real
                    best_test_mae_real = best_test_mae_real

                    # TOPERR logging on denormalized
                    if best_test_batch_for_top is not None:
                        p, t, m, tp = best_test_batch_for_top
                        log_topk_errors(top_logger, "test", itr, p, t, m, tp_to_predict=tp, topk=args.topk_err)
                        log_topk_best(top_logger, "test", itr, p, t, m, tp_to_predict=tp, topk=3)
                    else:
                        top_logger.warning(f"TOP_ERR | split=test epoch={itr} skipped (no valid test batch)")

                    if len(correct_examples) == 0:
                        top_logger.info(f"TOP_OK | split=test epoch={itr} none_found (no exact matches)")
                    else:
                        for rank, ex in enumerate(correct_examples[:3], start=1):
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

        except Exception:
            err_logger.error(f"EXCEPTION | val/test loop | epoch={itr}")
            err_logger.error(traceback.format_exc())
            raise

        # ------------------------- LOGGING -------------------------
        train_loss_val = safe_float(last_train_loss)

        train_logger.info(f"- Epoch {itr:03d}, ExpID {experimentID} best_epoch={best_iter} improved={int(itr == best_iter)}")
        train_logger.info(f"Train - Loss (last batch, scaled): {train_loss_val:.6f}")

        train_logger.info(
            "Val - GLOBAL masked MSE scaled: {:.6f} | real: {:.6f} | real MAE: {:.6f}".format(
                val_mse_scaled if np.isfinite(val_mse_scaled) else np.nan,
                val_mse_real if np.isfinite(val_mse_real) else np.nan,
                val_mae_real if np.isfinite(val_mae_real) else np.nan,
            )
        )

        # if we improved, we computed test above
        if np.isfinite(best_val_mse_real) and (best_iter == itr):
            train_logger.info(
                "Test - Best epoch, GLOBAL masked MSE scaled: {:.6f} | real: {:.6f} | real MAE: {:.6f}".format(
                    best_test_mse_scaled if "best_test_mse_scaled" in locals() else np.nan,
                    best_test_mse_real if np.isfinite(best_test_mse_real) else np.nan,
                    best_test_mae_real if np.isfinite(best_test_mae_real) else np.nan,
                )
            )
            train_logger.info(
                f"Test - Points (EXACT, real-units): total_masked={last_test_total_points} "
                f"correct={last_test_correct_points} right_rate={last_test_right_rate * 100:.4f}%"
            )

        # ---- system.log ----
        lr_now = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else np.nan
        epoch_time = time.time() - st_epoch
        alloc, reserved, max_alloc = cuda_mem_stats()
        sys_logger.info(
            f"SYSTEM | epoch={itr} time={epoch_time:.2f}s lr={lr_now:.6g} grad_norm={grad_norm:.6f} "
            f"mask_ratio_lastbatch={last_mask_ratio if np.isfinite(last_mask_ratio) else np.nan:.6f} "
            f"cuda_mem_alloc_mb={alloc:.2f} cuda_mem_reserved_mb={reserved:.2f} cuda_mem_max_alloc_mb={max_alloc:.2f}"
        )

        # ---- error.log ----
        if nan_loss_flag or nan_grad_flag:
            err_logger.warning(
                f"ANOMALY_FLAGS | epoch={itr} loss_nan={int(nan_loss_flag)} grad_nan={int(nan_grad_flag)}"
            )
            

        # ---- early stop (based on BEST REAL-UNIT VAL MSE) ----
        if (itr - best_iter) >= args.patience:
            run_logger.info(f"EARLY_STOP | epoch={itr} best_epoch={best_iter} patience={args.patience}")
            print("Exp has been early stopped!")
            sys.exit(0)

        # ------------------------- (kept as requested) evaluation lines commented -------------------------
        # val_res = evaluation(model, data_obj["val_dataloader"], int(data_obj["n_val_batches"]))
        # test_res = evaluation(model, data_obj["test_dataloader"], int(data_obj["n_test_batches"]))