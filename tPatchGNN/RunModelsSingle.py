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
from model.tPatchGNN import *  # tPatchGNN + evaluation

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
def _masked_metrics(pred, tgt, msk):
    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
    if mb.sum().item() == 0:
        return dict(loss=np.nan, mse=np.nan, rmse=np.nan, mae=np.nan, mape=np.nan)

    diff = (pred - tgt)[mb]

    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    rmse = np.sqrt(mse)

    tgt_safe = tgt[mb].abs()
    mape = float(torch.mean(diff.abs() / torch.clamp(tgt_safe, min=1e-8)).item())

    return dict(loss=mse, mse=mse, rmse=rmse, mae=mae)

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


def masked_repo_mse_loss(pred, tgt, msk):
    """
    Uses the repo's compute_error() logic (same style as your compute_all_losses/evaluation).
    Returns torch scalar suitable for backward().
    """
    pred = ensure_3d_B_Lp_N(pred)
    tgt = ensure_3d_B_Lp_N(tgt)
    msk = ensure_3d_B_Lp_N(msk)

    # compute_error(true, pred, mask=..., func="MSE", reduce="mean") -> torch scalar
    mse = compute_error(tgt, pred, mask=msk, func="MSE", reduce="mean")
    return mse


def evaluate_global_mse(model, dataloader, n_batches, device):
    """
    Run model on dataloader and compute GLOBAL masked MSE (scaled/normalized space).
    Returns dict compatible with evaluation(): {"loss": mse, "mse": mse, "mae": np.nan}
    """
    mses = []
    for _ in range(int(n_batches)):
        b = utils.get_next_batch(dataloader)

        pred = get_pred_ss(model, b, device)
        tgt = move_like(ensure_3d_B_Lp_N(b["data_to_predict"]), pred)
        msk = move_like(ensure_3d_B_Lp_N(b["mask_predicted_data"]), pred)

        v = masked_global_mse_value(pred, tgt, msk)
        if np.isfinite(v):
            mses.append(v)

    mse = float(np.mean(mses)) if len(mses) > 0 else np.nan
    return {"loss": mse, "mse": mse, "mae": np.nan}


def _rebuild_command_without_load(argv):
    argv = list(argv)
    idx = [i for i in range(len(argv)) if argv[i] == "--load"]
    if len(idx) == 1:
        i = idx[0]
        argv = argv[:i] + argv[i + 2:]
    return " ".join(argv)

    
if __name__ == "__main__":
    # ------------------------- CLI -------------------------
    parser = build_single_scale_parser()
    args = parser.parse_args()

    # Flags
    use_global_loss = int(getattr(args, "global_loss", 1)) == 1
    use_denorm_test_pred = int(getattr(args, "denorm_test_pred", 0)) == 1

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
    run_tag = f"{args.dataset}_{slurm_job_name}"
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

    run_logger.info(
        f"FLAGS | global_loss={int(use_global_loss)} (1=global masked MSE train/val, 0=repo MSE + evaluation()) | "
        f"denorm_test_pred={int(use_denorm_test_pred)} (1=denorm for TOPERR/EXACT)"
    )
    run_logger.info("EVAL_MODE | VAL uses: _masked_metrics() if global_loss=1 else evaluation().")
    run_logger.info("TOPERR/EXACT | computed on (denorm_test_pred? denorm : scaled) values.")

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

    best_val_mse = np.inf
    best_iter = 0
    test_res = None

    for itr in range(args.epoch):
        st = time.time()
        model.train()

        # -------------------- TRAIN --------------------
        for _ in range(num_batches):
            optimizer.zero_grad()
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

            pred = get_pred_ss(model, batch_dict, args.device)
            tgt  = batch_dict["data_to_predict"]
            msk  = batch_dict["mask_predicted_data"]

            pred = ensure_3d_B_Lp_N(pred)
            tgt  = ensure_3d_B_Lp_N(tgt)
            msk  = ensure_3d_B_Lp_N(msk)

            tgt = move_like(tgt, pred)
            msk = move_like(msk, pred)

            # ---- Apply denorm if activated ----
            if use_denorm_test_pred:
                pred = denorm_minmax(pred, data_min, data_max)
                tgt  = denorm_minmax(tgt,  data_min, data_max)

            mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)

            if mb.sum() == 0:
                continue

            diff = (pred - tgt)[mb]

            mse = (diff ** 2).mean()
            mae = diff.abs().mean()

            loss = mse
            loss.backward()
            optimizer.step()

            train_res = {
                "loss": loss.detach().item(),
                "mse": mse.detach().item(),
                "rmse": float(torch.sqrt(mse).item()),
                "mae": mae.detach().item(),
            }

        # ---- VAL / TEST ----
        model.eval()
        val_res = None

        try:
            with torch.no_grad():

                # -------------------- VALIDATION (always) --------------------
                if use_global_loss:
                    val_logs = []
                    for _ in range(int(data_obj["n_val_batches"])):
                        b = utils.get_next_batch(data_obj["val_dataloader"])

                        pred = get_pred_ss(model, b, args.device)
                        tgt  = b["data_to_predict"]
                        msk  = b["mask_predicted_data"]

                        pred = ensure_3d_B_Lp_N(pred)
                        tgt  = ensure_3d_B_Lp_N(tgt)
                        msk  = ensure_3d_B_Lp_N(msk)

                        tgt = move_like(tgt, pred)
                        msk = move_like(msk, pred)

                        # IMPORTANT: denorm flag only for logging/toperr typically,
                        # but if you want VAL in real units, keep this.
                        if use_denorm_test_pred:
                            pred = denorm_minmax(pred, data_min, data_max)
                            tgt  = denorm_minmax(tgt,  data_min, data_max)

                        mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
                        if mb.sum().item() == 0:
                            continue

                        diff = (pred - tgt)[mb]
                        mse  = (diff ** 2).mean().item()
                        mae  = diff.abs().mean().item()
                        rmse = float(np.sqrt(mse))

                        val_logs.append({"loss": mse, "mse": mse, "rmse": rmse, "mae": mae, "mape": 0.0})

                    if len(val_logs) > 0:
                        val_res = {k: float(np.mean([d[k] for d in val_logs])) for k in val_logs[0].keys()}
                    else:
                        val_res = {"loss": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}

                else:
                    # repo evaluation (always)
                    val_res = evaluation(model, data_obj["val_dataloader"], int(data_obj["n_val_batches"]))

                # -------------------- TESTING (only when VAL improves) --------------------
                if np.isfinite(val_res.get("mse", np.nan)) and (val_res["mse"] < best_val_mse):
                    best_val_mse = float(val_res["mse"])
                    best_iter = itr

                    if use_global_loss:
                        test_logs = []
                        for _ in range(int(data_obj["n_test_batches"])):
                            b = utils.get_next_batch(data_obj["test_dataloader"])

                            pred = get_pred_ss(model, b, args.device)
                            tgt  = b["data_to_predict"]
                            msk  = b["mask_predicted_data"]

                            pred = ensure_3d_B_Lp_N(pred)
                            tgt  = ensure_3d_B_Lp_N(tgt)
                            msk  = ensure_3d_B_Lp_N(msk)

                            tgt = move_like(tgt, pred)
                            msk = move_like(msk, pred)

                            # same note as above: keep if you want test in real units
                            if use_denorm_test_pred:
                                pred = denorm_minmax(pred, data_min, data_max)
                                tgt  = denorm_minmax(tgt,  data_min, data_max)

                            mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
                            if mb.sum().item() == 0:
                                continue

                            diff = (pred - tgt)[mb]
                            mse  = (diff ** 2).mean().item()
                            mae  = diff.abs().mean().item()
                            rmse = float(np.sqrt(mse))

                            test_logs.append({"loss": mse, "mse": mse, "rmse": rmse, "mae": mae, "mape": 0.0})

                        if len(test_logs) > 0:
                            test_res = {k: float(np.mean([d[k] for d in test_logs])) for k in test_logs[0].keys()}
                        else:
                            test_res = {"loss": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}

                    else:
                        test_res = evaluation(model, data_obj["test_dataloader"], int(data_obj["n_test_batches"]))

        except Exception:
            err_logger.error(f"EXCEPTION | eval | epoch={itr}")
            err_logger.error(traceback.format_exc())
            raise

        # -------------------- LOGGING (exact same style as your example) --------------------

        train_logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))

        train_logger.info(
            "Train - Loss (one batch): {:.5f}".format(train_res["loss"])
        )

        train_logger.info(
            "Val - Loss, MSE, RMSE, MAE: {:.5f}, {:.5f}, {:.5f}, {:.5f}"
            .format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"])
        )

        if test_res is not None:
            train_logger.info(
                "Test - Best epoch {}, Loss, MSE, RMSE, MAE: {:.5f}, {:.5f}, {:.5f}, {:.5f}"
                .format(best_iter, test_res["loss"], test_res["mse"], test_res["rmse"], test_res["mae"])
            )

        train_logger.info("Time spent: {:.2f}s".format(time.time() - st))

        # flush to avoid empty train.log
        for h in train_logger.handlers:
            try:
                h.flush()
            except:
                pass