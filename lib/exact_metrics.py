# lib/exact_metrics.py

import torch


def collect_topk_exact_examples(pred, tgt, msk, tp_to_predict=None, abs_tol=1e-5, rel_tol=0.05, k=3):
    """
    Collect up to k examples where prediction is "EXACT" under:
      |pred-true| <= abs_tol OR |pred-true| <= rel_tol*|true|
    among masked points only.

    Returns list of dicts:
      {b, t, var_idx, y_true, y_pred, tp(optional)}
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