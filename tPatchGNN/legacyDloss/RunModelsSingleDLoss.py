# RunModelsSingle.py (single-scale) — RunModelsLogs-style logging + TOPERR + TOP_OK(EXACT tol)
# FIXED for your repo:
#   - In your version, compute_all_losses(...) returns ONLY metrics: ['loss','mse','rmse','mae']
#     so we CANNOT get predictions from it.
#   - For SS we must run the model prediction using model.forecasting(...)
#     because your tPatchGNN.py implements forecasting(...) (not forward()).

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

import torch
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.tPatchGNN import *  # tPatchGNN + compute_all_losses + evaluation

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

# ------------------------- CLI -------------------------

parser = argparse.ArgumentParser("IMTS Forecasting (Single-Scale)")

parser.add_argument("--state", type=str, default="def")
parser.add_argument("-n", type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument("--hop", type=int, default=1, help="hops in GNN")
parser.add_argument("--nhead", type=int, default=1, help="heads in Transformer")
parser.add_argument("--tf_layer", type=int, default=1, help="# of layer in Transformer")
parser.add_argument("--nlayer", type=int, default=1, help="# of layer in TSmodel")
parser.add_argument("--epoch", type=int, default=1000, help="training epochs")
parser.add_argument("--patience", type=int, default=10, help="patience for early stop")

parser.add_argument("--history", type=int, default=24, help="historical window")
parser.add_argument("-ps", "--patch_size", type=float, default=24, help="window size for a patch")
parser.add_argument("--stride", type=float, default=24, help="period stride for patch sliding")

parser.add_argument("--logmode", type=str, default="a", help="File mode of logging (a/w).")
parser.add_argument("--normalization",type=int,default=0, help="0 = per-channel (default), 1 = global scalar normalization (Activity only).")

parser.add_argument("--lr", type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument("--w_decay", type=float, default=0.0, help="weight decay.")
parser.add_argument("-b", "--batch_size", type=int, default=32)

parser.add_argument("--save", type=str, default="experiments/", help="Path for save checkpoints")
parser.add_argument("--load", type=str, default=None, help="Experiment ID to load; if None, create new.")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--dataset", type=str, default="physionet", help="Dataset to load.")

parser.add_argument("--quantization", type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument("--model", type=str, default="tPatchGNN", help="Model name")
parser.add_argument("--outlayer", type=str, default="Linear", help="Output layer name")
parser.add_argument("-hd", "--hid_dim", type=int, default=64, help="Hidden dim")
parser.add_argument("-td", "--te_dim", type=int, default=10, help="Time enc dim")
parser.add_argument("-nd", "--node_dim", type=int, default=10, help="Node dim")
parser.add_argument("--gpu", type=str, default="0", help="which gpu to use.")

parser.add_argument("--data_sanity_batches", type=int, default=3, help="batches to sample per split for sanity logs.")

# Top-K error logging
parser.add_argument("--topk_err", type=int, default=10, help="Top-K largest abs errors to log in toperr.log on TEST.")

# EXACT tolerance (defaults match your previous logs)
parser.add_argument("--exact_abs_tol", type=float, default=1e-5, help="Absolute tolerance for EXACT points.")
parser.add_argument("--exact_rel_tol", type=float, default=0.05, help="Relative tolerance for EXACT points.")

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


# ------------------------- helpers -------------------------

def _rebuild_command_without_load(argv):
    argv = list(argv)
    idx = [i for i in range(len(argv)) if argv[i] == "--load"]
    if len(idx) == 1:
        i = idx[0]
        argv = argv[:i] + argv[i + 2 :]
    return " ".join(argv)


def _ensure_3d_B_Lp_N(x):
    # squeeze leading (1, B, Lp, N) -> (B, Lp, N)
    if torch.is_tensor(x) and x.dim() == 4 and x.size(0) == 1:
        x = x.squeeze(0)
    return x


def _move_like(x, ref):
    if (x is None) or (not torch.is_tensor(x)) or (not torch.is_tensor(ref)):
        return x
    return x.to(device=ref.device)


def _collect_topk_exact_examples(pred, tgt, msk, tp_to_predict=None, abs_tol=1e-5, rel_tol=0.05, k=3):
    """
    Return up to k examples of exact matches under tolerance.
    pred/tgt/msk expected shape (B, Lp, N).
    """
    if pred is None or tgt is None or msk is None:
        return []
    if (not torch.is_tensor(pred)) or (not torch.is_tensor(tgt)) or (not torch.is_tensor(msk)):
        return []

    pred = pred.detach()
    tgt = tgt.detach()
    msk = msk.detach()

    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
    if mb.sum().item() == 0:
        return []

    diff = (pred - tgt).abs()
    tgt_abs = tgt.abs().clamp(min=1e-8)
    ok = ((diff <= abs_tol) | (diff <= rel_tol * tgt_abs)) & mb

    idxs = ok.nonzero(as_tuple=False)
    if idxs.numel() == 0:
        return []

    ex = []
    for i in range(min(k, idxs.size(0))):
        b_i, t_i, v_i = idxs[i].tolist()
        one = {
            "b": b_i,
            "t": t_i,
            "var_idx": v_i,
            "y_true": float(tgt[b_i, t_i, v_i].item()),
            "y_pred": float(pred[b_i, t_i, v_i].item()),
            "tp": None,
        }
        if tp_to_predict is not None and torch.is_tensor(tp_to_predict):
            try:
                if tp_to_predict.dim() == 2:
                    one["tp"] = float(tp_to_predict[b_i, t_i].item())
                elif tp_to_predict.dim() == 1:
                    one["tp"] = float(tp_to_predict[t_i].item())
            except Exception:
                one["tp"] = None
        ex.append(one)
    return ex


def _pick_first_key(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _get_pred_ss(model, batch_dict):
    """
    Single-scale prediction getter for YOUR repo.

    Your tPatchGNN implements:
        forecasting(time_steps_to_predict, X, truth_time_steps, mask=None)

    So we map:
        time_steps_to_predict  <- batch['tp_to_predict'] (or similar)
        X                     <- batch['observed_data']  (or similar)
        truth_time_steps      <- batch['observed_tp']    (or similar)
        mask                  <- batch['observed_mask'] or batch['mask_observed_data'] (if exists)

    Returns pred with shape (B, Lp, N) (after squeezing if needed).
    """
    # common SS keys across IMTS repos
    tp_to_predict = _pick_first_key(batch_dict, ["tp_to_predict", "time_steps_to_predict", "tp_pred"])
    X = _pick_first_key(batch_dict, ["observed_data", "X", "data", "observations"])
    tt = _pick_first_key(batch_dict, ["observed_tp", "truth_time_steps", "tp", "time_steps"])
    mask_obs = _pick_first_key(batch_dict, ["observed_mask", "mask_observed_data", "mask", "mk"])

    if tp_to_predict is None or X is None or tt is None:
        raise KeyError(
            "SS batch missing required keys for forecasting(). "
            f"Have keys={list(batch_dict.keys())} "
            f"(need tp_to_predict + observed_data + observed_tp)"
        )

    # Ensure on device
    tp_to_predict = tp_to_predict.to(args.device) if torch.is_tensor(tp_to_predict) else tp_to_predict
    X = X.to(args.device) if torch.is_tensor(X) else X
    tt = tt.to(args.device) if torch.is_tensor(tt) else tt
    if torch.is_tensor(mask_obs):
        mask_obs = mask_obs.to(args.device)

    # Call forecasting
    if hasattr(model, "forecasting"):
        pred = model.forecasting(tp_to_predict, X, tt, mask=mask_obs)
    else:
        # fallback (rare): some variants implement forward for SS
        try:
            pred = model(tp_to_predict, X, tt, mask_obs)
        except Exception as e:
            raise RuntimeError("Model has no forecasting() and SS forward fallback failed.") from e

    pred = _ensure_3d_B_Lp_N(pred)
    return pred


# ------------------------- main -------------------------

if __name__ == "__main__":
    utils.setup_seed(args.seed)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    ckpt_path = os.path.join(args.save, f"experiment_{experimentID}.ckpt")
    input_command = _rebuild_command_without_load(sys.argv)

    # Loggers (RunModelsLogs style)
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

    # Header logs
    host = socket.gethostname()
    run_logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run_logger.info(f"command: {input_command}")
    run_logger.info(
        f"ExpID={experimentID} PID={args.PID} host={host} device={args.device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    run_logger.info(f"SLURM_JOB_NAME={slurm_job_name} SLURM_JOB_ID={slurm_job_id}")
    run_logger.info(f"ckpt_path={ckpt_path}")
    run_logger.info(f"args={args}")

    # Data
    try:
        data_obj = parse_datasets(args, patch_ts=True)
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
        f"quantization={args.quantization} history={args.history} patch_size={args.patch_size} stride={args.stride} npatch={args.npatch}"
    )

    # Data sanity
    try:
        batch_fetch_fn = lambda dobj, key: utils.get_next_batch(dobj[key])
        sample_data_sanity(
            run_logger,
            data_obj,
            split_name="train",
            dl_key="train_dataloader",
            n_key="n_train_batches",
            max_batches=args.data_sanity_batches,
            batch_fetch_fn=batch_fetch_fn,
        )
        sample_data_sanity(
            run_logger,
            data_obj,
            split_name="val",
            dl_key="val_dataloader",
            n_key="n_val_batches",
            max_batches=args.data_sanity_batches,
            batch_fetch_fn=batch_fetch_fn,
        )
        sample_data_sanity(
            run_logger,
            data_obj,
            split_name="test",
            dl_key="test_dataloader",
            n_key="n_test_batches",
            max_batches=args.data_sanity_batches,
            batch_fetch_fn=batch_fetch_fn,
        )
    except Exception:
        err_logger.warning("DATA_SANITY failed (continuing)")
        err_logger.warning(traceback.format_exc())

    # Model
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

                if "mask_predicted_data" in batch_dict and batch_dict["mask_predicted_data"] is not None:
                    last_mask_ratio = summarize_mask(batch_dict["mask_predicted_data"])["mask_ratio"]

            except Exception:
                err_logger.error(f"EXCEPTION | train | epoch={itr}")
                err_logger.error(traceback.format_exc())
                raise

        # ---- VAL / TEST ----
        model.eval()
        improved = False
        val_res = None

        try:
            with torch.no_grad():
                val_res = evaluation(model, data_obj["val_dataloader"], int(data_obj["n_val_batches"]))

                if np.isfinite(val_res.get("mse", np.nan)) and (val_res["mse"] < best_val_mse):
                    improved = True
                    best_val_mse = float(val_res["mse"])
                    best_iter = itr

                    # aggregated test metrics
                    test_res = evaluation(model, data_obj["test_dataloader"], int(data_obj["n_test_batches"]))

                    # ------------------ TOPERR + EXACT ------------------
                    best_test_batch_for_top = None
                    test_total_points = 0
                    test_correct_points = 0
                    correct_examples = []

                    n_test_batches = int(data_obj["n_test_batches"])
                    for _ in range(n_test_batches):
                        b = utils.get_next_batch(data_obj["test_dataloader"])

                        # SS prediction using forecasting(...)
                        pred = _get_pred_ss(model, b)

                        tgt = b["data_to_predict"]
                        msk = b["mask_predicted_data"]
                        tp = b.get("tp_to_predict", None)

                        # normalize shapes
                        pred = _ensure_3d_B_Lp_N(pred)
                        tgt = _ensure_3d_B_Lp_N(tgt)
                        msk = _ensure_3d_B_Lp_N(msk)

                        # ensure same device
                        tgt = _move_like(tgt, pred)
                        msk = _move_like(msk, pred)

                        # skip invalid
                        if pred is None or tgt is None or msk is None:
                            continue
                        if (not torch.is_tensor(pred)) or (not torch.is_tensor(tgt)) or (not torch.is_tensor(msk)):
                            continue
                        if pred.dim() != 3 or tgt.dim() != 3 or msk.dim() != 3:
                            err_logger.warning(
                                f"SHAPE_SKIP | epoch={itr} pred={tuple(pred.shape)} tgt={tuple(tgt.shape)} msk={tuple(msk.shape)}"
                            )
                            continue

                        if best_test_batch_for_top is None:
                            best_test_batch_for_top = (pred, tgt, msk, tp)

                        tot, cor = count_right_points_tol(
                            pred, tgt, msk, abs_tol=args.exact_abs_tol, rel_tol=args.exact_rel_tol
                        )
                        test_total_points += tot
                        test_correct_points += cor

                        if len(correct_examples) < 3:
                            correct_examples.extend(
                                _collect_topk_exact_examples(
                                    pred, tgt, msk,
                                    tp_to_predict=tp,
                                    abs_tol=args.exact_abs_tol,
                                    rel_tol=args.exact_rel_tol,
                                    k=3 - len(correct_examples),
                                )
                            )

                    last_test_total_points = int(test_total_points)
                    last_test_correct_points = int(test_correct_points)
                    last_test_right_rate = float(test_correct_points / max(test_total_points, 1))

                    # Top errors on TEST
                    if best_test_batch_for_top is not None:
                        p, t, m, tp = best_test_batch_for_top
                        log_topk_errors(top_logger, "test", itr, p, t, m, tp_to_predict=tp, topk=args.topk_err)
                        log_topk_best(top_logger, "test", itr, p, t, m, tp_to_predict=tp, topk=3)
                    else:
                        top_logger.warning(f"TOP_ERR | split=test epoch={itr} skipped (no valid test batch)")

                    # TOP_OK (exact matches under tolerance)
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
            err_logger.error(f"EXCEPTION | eval | epoch={itr}")
            err_logger.error(traceback.format_exc())
            raise

        # ---- LOGGING: train.log ----
        train_loss_val = safe_float(last_train_loss)
        train_logger.info(f"- Epoch {itr:03d}, ExpID {experimentID} best_epoch={best_iter} improved={int(improved)}")
        train_logger.info(f"Train - Loss (last batch): {train_loss_val:.5f}")

        train_logger.info(
            "Val - Loss, MSE, MAE: {:.5f}, {:.5f}, {:.5f}".format(
                float(val_res.get("loss", np.nan)),
                float(val_res.get("mse", np.nan)),
                float(val_res.get("mae", np.nan)),
            )
        )

        if test_res is not None:
            train_logger.info(
                "Test - Best epoch, Loss, MSE, MAE: {}, {:.5f}, {:.5f}, {:.5f}".format(
                    best_iter,
                    float(test_res.get("loss", np.nan)),
                    float(test_res.get("mse", np.nan)),
                    float(test_res.get("mae", np.nan)),
                )
            )
            train_logger.info(
                f"Test - Points (EXACT): total_masked={last_test_total_points} correct={last_test_correct_points} "
                f"right_rate={last_test_right_rate*100:.4f}%"
            )

        # ---- LOGGING: system.log ----
        lr_now = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else np.nan
        epoch_time = time.time() - st_epoch
        alloc, reserved, max_alloc = cuda_mem_stats()
        sys_logger.info(
            f"SYSTEM | epoch={itr} time={epoch_time:.2f}s lr={lr_now:.6g} grad_norm={grad_norm:.6f} "
            f"mask_ratio_lastbatch={last_mask_ratio if np.isfinite(last_mask_ratio) else np.nan:.6f} "
            f"cuda_mem_alloc_mb={alloc:.2f} cuda_mem_reserved_mb={reserved:.2f} cuda_mem_max_alloc_mb={max_alloc:.2f}"
        )

        # ---- LOGGING: error.log ----
        if nan_loss_flag or nan_grad_flag:
            err_logger.warning(
                f"ANOMALY_FLAGS | epoch={itr} loss_nan={int(nan_loss_flag)} grad_nan={int(nan_grad_flag)}"
            )

        # ---- early stop ----
        if (itr - best_iter) >= args.patience:
            run_logger.info(f"EARLY_STOP | epoch={itr} best_epoch={best_iter} patience={args.patience}")
            print("Exp has been early stopped!")
            sys.exit(0)
