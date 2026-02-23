# lib/train_losses.py

import torch
from .ss_forecast import ensure_3d_B_Lp_N


def masked_global_mse_loss(pred, tgt, msk):
    """
    GLOBAL masked MSE (NOT per-dim):
        sum((pred-tgt)^2 * mask) / sum(mask)
    pred/tgt/msk: (B, Lp, N)
    """
    pred = ensure_3d_B_Lp_N(pred)
    tgt = ensure_3d_B_Lp_N(tgt)
    msk = ensure_3d_B_Lp_N(msk)

    mb = msk.bool() if msk.dtype == torch.bool else (msk > 0.5)
    cnt = mb.sum()
    if cnt.item() <= 0:
        return None  # caller decides what to do

    diff2 = (pred - tgt) ** 2
    # Mean over ALL masked points globally
    return diff2[mb].mean()