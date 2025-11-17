import torch
import torch.nn as nn
import lib.utils as utils
from lib.utils import get_device


############################################################
# compute_error (unchanged, working version)
############################################################
def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None):

    if len(pred_y.shape) == 3:
        pred_y = pred_y.unsqueeze(0)

    n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()

    truth_repeated = truth.repeat(n_traj_samples, 1, 1, 1)
    mask = mask.repeat(n_traj_samples, 1, 1, 1)

    if func == "MSE":
        error = ((truth_repeated - pred_y) ** 2) * mask

    elif func == "MAE":
        error = torch.abs(truth_repeated - pred_y) * mask

    elif func == "MAPE":
        mask = (truth_repeated != 0) * mask
        truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
        error = torch.abs(truth_repeated - pred_y) / truth_div * mask

    else:
        raise Exception("Unknown error function.")

    error_var_sum = error.reshape(-1, n_dim).sum(dim=0)
    mask_count = mask.reshape(-1, n_dim).sum(dim=0)

    if reduce == "mean":
        err = error_var_sum / (mask_count + 1e-8)
        n_avai = torch.count_nonzero(mask_count)
        return err.sum() / n_avai

    return error_var_sum, mask_count


############################################################
# compute_all_losses — CLEAN VERSION, NO observed_data
############################################################
def compute_all_losses(model, batch_dict):
    """
    Unified loss function for:
      - single-scale (observed_data style)
      - single-scale + NodeMixer
      - multi-scale (X_list style)
    """

    # ------------------------------------------------------------
    # 1. Forecasting step — handle all possible batch formats
    # ------------------------------------------------------------

    if "X_list" in batch_dict:
        # Multi-scale format
        pred_y = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["X_list"],
            batch_dict["tt_list"],
            batch_dict["mk_list"],
        )
        truth = batch_dict["data_to_predict"]
        mask  = batch_dict["mask_predicted_data"]

    elif "observed_data" in batch_dict:
        # Classic single-scale tPatchGNN format
        pred_y = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["observed_data"],
            batch_dict["observed_tp"],
            batch_dict["observed_mask"],     # IMPORTANT FIX
        )
        truth = batch_dict["data_to_predict"]
        mask  = batch_dict["mask_predicted_data"]

    elif "X" in batch_dict:
        # Simple patched format (X, tt, mk)
        pred_y = model.forecasting(
            batch_dict["tp_to_predict"],
            batch_dict["X"],
            batch_dict["tt"],
            batch_dict["mk"],
        )
        truth = batch_dict["data_to_predict"]
        mask  = batch_dict["mask_predicted_data"]

    else:
        raise KeyError(f"compute_all_losses: unknown batch format, keys = {list(batch_dict.keys())}")

    # ------------------------------------------------------------
    # 2. Compute losses
    # ------------------------------------------------------------
    mse_val  = compute_error(truth, pred_y, mask=mask, func="MSE", reduce="mean")
    rmse_val = torch.sqrt(mse_val)
    mae_val  = compute_error(truth, pred_y, mask=mask, func="MAE", reduce="mean")

    return {
        "loss": mse_val,
        "mse":  mse_val.item(),
        "rmse": rmse_val.item(),
        "mae":  mae_val.item(),
    }


############################################################
# evaluation — CLEAN VERSION, NO observed_data
############################################################
def evaluation(model, dataloader, n_batches):

    total = {"loss": 0, "mse": 0, "mae": 0, "rmse": 0, "mape": 0}

    n_eval_samples = 0
    n_eval_samples_mape = 0

    for _ in range(n_batches):
        batch_dict = utils.get_next_batch(dataloader)

        # ------------------ unified model call ---------------------
        if "X_list" in batch_dict:
            pred_y = model.forecasting(
                batch_dict["tp_to_predict"],
                batch_dict["X_list"],
                batch_dict["tt_list"],
                batch_dict["mk_list"],
            )
            truth = batch_dict["data_to_predict"]
            mask  = batch_dict["mask_predicted_data"]

        elif "observed_data" in batch_dict:
            pred_y = model.forecasting(
                batch_dict["tp_to_predict"],
                batch_dict["observed_data"],
                batch_dict["observed_tp"],
                batch_dict["observed_mask"],
            )
            truth = batch_dict["data_to_predict"]
            mask  = batch_dict["mask_predicted_data"]

        elif "X" in batch_dict:
            pred_y = model.forecasting(
                batch_dict["tp_to_predict"],
                batch_dict["X"],
                batch_dict["tt"],
                batch_dict["mk"],
            )
            truth = batch_dict["data_to_predict"]
            mask  = batch_dict["mask_predicted_data"]

        else:
            raise KeyError(f"evaluation: unknown batch format {list(batch_dict.keys())}")

        # ------------------ compute error vectors -------------------
        se_var_sum, mask_count = compute_error(truth, pred_y, mask, "MSE", "sum")
        ae_var_sum, _          = compute_error(truth, pred_y, mask, "MAE", "sum")
        ape_var_sum, mask_count_mape = compute_error(truth, pred_y, mask, "MAPE", "sum")

        # accumulate
        total["loss"] += se_var_sum
        total["mse"]  += se_var_sum
        total["mae"]  += ae_var_sum
        total["mape"] += ape_var_sum

        n_eval_samples      += mask_count
        n_eval_samples_mape += mask_count_mape

    # ------------------ aggregate averages -------------------------
    n_avai      = torch.count_nonzero(n_eval_samples)
    n_avai_mape = torch.count_nonzero(n_eval_samples_mape)

    total["loss"] = (total["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai
    total["mse"]  = total["loss"]
    total["rmse"] = torch.sqrt(total["mse"])
    total["mae"]  = (total["mae"]  / (n_eval_samples + 1e-8)).sum() / n_avai
    total["mape"] = (total["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_mape

    # convert tensors to floats
    for k in total:
        if isinstance(total[k], torch.Tensor):
            total[k] = total[k].item()

    return total
