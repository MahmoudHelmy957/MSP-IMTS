# lib/ss_forecast.py

import torch


def ensure_3d_B_Lp_N(x):
    """
    Some models may return (1, B, Lp, N) -> squeeze the leading 1.
    Expect final shape (B, Lp, N).
    """
    if torch.is_tensor(x) and x.dim() == 4 and x.size(0) == 1:
        x = x.squeeze(0)
    return x


def move_like(x, ref):
    """
    Move tensor x to same device as ref (if both tensors).
    """
    if (x is None) or (not torch.is_tensor(x)) or (not torch.is_tensor(ref)):
        return x
    return x.to(device=ref.device)


def pick_first_key(d, keys):
    """
    Return d[k] for the first key k in keys that exists and is not None.
    """
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def get_pred_ss(model, batch_dict, device):
    """
    Single-scale prediction via:
        model.forecasting(time_steps_to_predict, X, truth_time_steps, mask=mask_obs)
    """
    tp_to_predict = pick_first_key(batch_dict, ["tp_to_predict", "time_steps_to_predict", "tp_pred"])
    X = pick_first_key(batch_dict, ["observed_data", "X", "data", "observations"])
    tt = pick_first_key(batch_dict, ["observed_tp", "truth_time_steps", "tp", "time_steps"])
    mask_obs = pick_first_key(batch_dict, ["observed_mask", "mask_observed_data", "mask", "mk"])

    if tp_to_predict is None or X is None or tt is None:
        raise KeyError(
            "SS batch missing required keys for forecasting(). "
            f"Have keys={list(batch_dict.keys())} "
            f"(need tp_to_predict + observed_data + observed_tp)"
        )

    tp_to_predict = tp_to_predict.to(device) if torch.is_tensor(tp_to_predict) else tp_to_predict
    X = X.to(device) if torch.is_tensor(X) else X
    tt = tt.to(device) if torch.is_tensor(tt) else tt
    if torch.is_tensor(mask_obs):
        mask_obs = mask_obs.to(device)

    if hasattr(model, "forecasting"):
        pred = model.forecasting(tp_to_predict, X, tt, mask=mask_obs)
    else:
        # rare fallback
        pred = model(tp_to_predict, X, tt, mask_obs)

    return ensure_3d_B_Lp_N(pred)