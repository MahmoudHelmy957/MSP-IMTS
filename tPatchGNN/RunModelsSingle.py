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

from lib.analyzelogs import *

from lib.ss_forecast import get_pred_ss, ensure_3d_B_Lp_N, move_like
from lib.train_losses import masked_global_mse_loss
from lib.exact_metrics import collect_topk_exact_examples

# ------------------------- helpers -------------------------
# ------------------------- EXTRA LOGGING (INSERT-ONLY) -------------------------
def _collect_masked_points_for_logging(model, dataloader, device, data_min, data_max, denorm_flag, max_batches=2):
    """
    Collect masked per-element points from up to `max_batches` batches.
    Returns dict with tensors on CPU:
      - idx: (M, 3) integer indices [b, t, d]
      - pred: (M,) float
      - tgt: (M,) float
      - diff: (M,) float (pred - tgt)
      - abs_err: (M,) float
    """
    model.eval()
    all_idx = []
    all_pred = []
    all_tgt = []

    with torch.no_grad():
        for _ in range(int(max_batches)):
            bdict = utils.get_next_batch(dataloader)

            pred = get_pred_ss(model, bdict, device)
            tgt  = ensure_3d_B_Lp_N(bdict["data_to_predict"])
            msk  = ensure_3d_B_Lp_N(bdict["mask_predicted_data"])

            pred = ensure_3d_B_Lp_N(pred)

            tgt = move_like(tgt, pred)
            msk = move_like(msk, pred)

            # Denorm ONLY for logging if requested (your run_logger message already states this behavior)
            if denorm_flag:
                pred_eval = denorm_minmax(pred, data_min, data_max)
                tgt_eval  = denorm_minmax(tgt,  data_min, data_max)
            else:
                pred_eval = pred
                tgt_eval  = tgt

            mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
            if mb.sum().item() == 0:
                continue

            nz = torch.nonzero(mb, as_tuple=False)  # (M, 3) with columns [B, Lp, D]
            pv = pred_eval[mb].detach()
            tv = tgt_eval[mb].detach()

            all_idx.append(nz)
            all_pred.append(pv)
            all_tgt.append(tv)

    if len(all_idx) == 0:
        return None

    idx = torch.cat(all_idx, dim=0)
    pred_v = torch.cat(all_pred, dim=0)
    tgt_v  = torch.cat(all_tgt, dim=0)

    diff = pred_v - tgt_v
    abs_err = diff.abs()

    return {
        "idx": idx.cpu(),
        "pred": pred_v.cpu(),
        "tgt": tgt_v.cpu(),
        "diff": diff.cpu(),
        "abs_err": abs_err.cpu(),
    }


def _log_points_summary(top_logger, tag, pack, k_random=15, k_best=5, k_worst=5, seed=1234):
    """
    Logs:
      - random K points (true/pred/diff)
      - top K worst abs errors
      - top K best (closest / "correct") points
    """
    if pack is None:
        top_logger.info(f"{tag} | No masked points found (empty mask).")
        return

    idx = pack["idx"]
    pred = pack["pred"]
    tgt  = pack["tgt"]
    diff = pack["diff"]
    abs_err = pack["abs_err"]

    M = int(abs_err.shape[0])
    if M <= 0:
        top_logger.info(f"{tag} | No masked points found (M=0).")
        return

    # Random selection (deterministic per epoch/tag if you pass a changing seed)
    g = torch.Generator()
    g.manual_seed(int(seed) & 0x7FFFFFFF)

    k_random = min(int(k_random), M)
    perm = torch.randperm(M, generator=g)
    ridx = perm[:k_random]

    top_logger.info(f"{tag} | RANDOM {k_random} masked points (true, pred, diff, abs_err):")
    for j in ridx.tolist():
        b, t, d = idx[j].tolist()
        top_logger.info(
            f"{tag} | [B={b}, T={t}, D={d}] true={float(tgt[j]):.6f} pred={float(pred[j]):.6f} "
            f"diff={float(diff[j]):+.6f} abs_err={float(abs_err[j]):.6f}"
        )

    # Worst / Best
    k_worst = min(int(k_worst), M)
    k_best  = min(int(k_best),  M)

    order = torch.argsort(abs_err)  # ascending
    best_idx = order[:k_best]
    worst_idx = order[-k_worst:].flip(0)  # descending

    top_logger.info(f"{tag} | TOP {k_worst} WORST masked points (largest abs_err):")
    for j in worst_idx.tolist():
        b, t, d = idx[j].tolist()
        top_logger.info(
            f"{tag} | [B={b}, T={t}, D={d}] true={float(tgt[j]):.6f} pred={float(pred[j]):.6f} "
            f"diff={float(diff[j]):+.6f} abs_err={float(abs_err[j]):.6f}"
        )

    top_logger.info(f"{tag} | TOP {k_best} BEST/CORRECT masked points (smallest abs_err):")
    for j in best_idx.tolist():
        b, t, d = idx[j].tolist()
        top_logger.info(
            f"{tag} | [B={b}, T={t}, D={d}] true={float(tgt[j]):.6f} pred={float(pred[j]):.6f} "
            f"diff={float(diff[j]):+.6f} abs_err={float(abs_err[j]):.6f}"
        )

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
    # D: normalization mode (dataset preprocessing)
    # D=0 -> per-dim stats, D=1 -> global stats
    D_norm_mode = int(getattr(args, "normalization", 0))  # 0/1
    use_global_loss = int(getattr(args, "global_loss", 1)) == 1  # G flag
    use_denorm_eval = int(getattr(args, "denorm_test_pred", 0)) == 1  # N flag (denorm val/test metrics)

    run_logger.info(f"FLAGS | D(normalization)={D_norm_mode} (0=per-dim,1=global) "
                    f"G(global_loss)={int(use_global_loss)} "
                    f"N(denorm_eval)={int(use_denorm_eval)}")

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

            # NOTE: DO NOT denormalize during training.
            # Train must remain in the same scale as model outputs (normalized space).

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

                        # ✅ Validation ALWAYS in normalized space (no denorm here)
                        pred_eval = pred
                        tgt_eval  = tgt

                        mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
                        if mb.sum().item() == 0:
                            continue

                        diff = (pred_eval - tgt_eval)[mb]
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
                # -------------------- TESTING (only when VAL improves) --------------------
                if np.isfinite(val_res.get("mse", np.nan)) and (val_res["mse"] < best_val_mse):
                    best_val_mse = float(val_res["mse"])
                    best_iter = itr

                    if use_global_loss:
                        # --------- TEST (manual metrics path) ---------
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

                            # ---- Denorm ONLY for TEST metrics if activated ----
                            if use_denorm_test_pred:
                                pred_eval = denorm_minmax(pred, data_min, data_max)
                                tgt_eval  = denorm_minmax(tgt,  data_min, data_max)
                            else:
                                pred_eval = pred
                                tgt_eval  = tgt

                            mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
                            if mb.sum().item() == 0:
                                continue

                            diff = (pred_eval - tgt_eval)[mb]
                            mse  = (diff ** 2).mean().item()
                            mae  = diff.abs().mean().item()
                            rmse = float(np.sqrt(mse))

                            test_logs.append({"loss": mse, "mse": mse, "rmse": rmse, "mae": mae, "mape": 0.0})

                        if len(test_logs) > 0:
                            test_res = {k: float(np.mean([d[k] for d in test_logs])) for k in test_logs[0].keys()}
                        else:
                            test_res = {"loss": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}

                    else:
                        # --------- TEST (repo evaluation path) ---------
                        # Always compute the repo metric (normalized space)
                        test_res = evaluation(model, data_obj["test_dataloader"], int(data_obj["n_test_batches"]))

                        # If denorm flag is ON, also compute real-unit metrics manually and overwrite test_res
                        # so denorm works even when using evaluation()
                        if use_denorm_test_pred:
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

                                # denorm for real-unit metrics
                                pred_eval = denorm_minmax(pred, data_min, data_max)
                                tgt_eval  = denorm_minmax(tgt,  data_min, data_max)

                                mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
                                if mb.sum().item() == 0:
                                    continue

                                diff = (pred_eval - tgt_eval)[mb]
                                mse  = (diff ** 2).mean().item()
                                mae  = diff.abs().mean().item()
                                rmse = float(np.sqrt(mse))

                                test_logs.append({"loss": mse, "mse": mse, "rmse": rmse, "mae": mae, "mape": 0.0})

                            if len(test_logs) > 0:
                                test_res = {k: float(np.mean([d[k] for d in test_logs])) for k in test_logs[0].keys()}
                            else:
                                test_res = {"loss": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan}
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

        # -------------------- EXTRA TOPERR LOGS (every 10 epochs) --------------------
        if ((itr + 1) % 10) == 0:
            try:
                # Use TEST dataloader (your request), log in (denorm_test_pred ? denorm : scaled)
                pack = _collect_masked_points_for_logging(
                    model=model,
                    dataloader=data_obj["test_dataloader"],
                    device=args.device,
                    data_min=data_min,
                    data_max=data_max,
                    denorm_flag=use_denorm_test_pred,
                    max_batches=2,  # increase if you want a wider pool
                )
                _log_points_summary(
                    top_logger=top_logger,
                    tag=f"EPOCH_{itr:03d}",
                    pack=pack,
                    k_random=15,
                    k_best=5,
                    k_worst=5,
                    seed=args.seed + itr,
                )
            except Exception:
                err_logger.error(f"EXCEPTION | TOPERR_LOGGING | epoch={itr}")
                err_logger.error(traceback.format_exc())

        # flush to avoid empty train.log
        for h in train_logger.handlers:
            try:
                h.flush()
            except:
                pass