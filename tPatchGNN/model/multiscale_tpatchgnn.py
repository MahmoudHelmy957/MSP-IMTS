# model/multiscale_tpatchgnn.py
import numpy as np
import torch
import torch.nn as nn


# ======================================================================================
# Debug helpers (safe stats for tensors)  [same idea as your tPatchGNN debug_internal]
# ======================================================================================

def _safe_stats(x: torch.Tensor):
    x = x.detach()
    if x.numel() == 0:
        return dict(shape=tuple(x.shape), min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    return dict(
        shape=tuple(x.shape),
        min=float(x.min().item()),
        max=float(x.max().item()),
        mean=float(x.mean().item()),
        std=float(x.std(unbiased=False).item()),
    )

def _masked_stats(x: torch.Tensor, mask: torch.Tensor):
    """
    x: tensor
    mask: broadcastable to x, 1 where valid
    """
    mb = mask.bool() if mask.dtype == torch.bool else (mask > 0.5)
    if mb.sum().item() == 0:
        return dict(shape=tuple(x.shape), count=0, min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    v = x.detach()[mb]
    return dict(
        shape=tuple(x.shape),
        count=int(mb.sum().item()),
        min=float(v.min().item()),
        max=float(v.max().item()),
        mean=float(v.mean().item()),
        std=float(v.std(unbiased=False).item()),
    )


class MultiScaleTPatchGNN(nn.Module):
    """
    Wraps K single-scale tPatchGNN encoders and fuses their representations.

    Forward expects:
      - X_list, tt_list, mk_list: lists of length K
        each element shaped (B, M_k, L, N)
      - time_steps_to_predict: (B, Lp)

    Returns:
      - out: (1, B, Lp, N)  # same shape convention as original tPatchGNN forward
    """
    def __init__(
        self,
        submodels,
        te_dim: int = 10,
        proj_dim: int | None = None,
        fusion: str = "concat",
        debug_internal: bool = False,   # <--- NEW
    ):
        super().__init__()
        assert len(submodels) >= 2, "Use >= 2 scales."
        self.submodels = nn.ModuleList(submodels)
        self._te_dim   = te_dim
        self._proj_dim = proj_dim
        self._fusion   = fusion

        # ----------------------- Debug flag & store -----------------------
        self.debug_internal = bool(debug_internal)
        self.last_debug: dict = {}

        # Lazily built on first forward (so we know fused_dim)
        self.fuse_proj: nn.Linear | None = None
        self.decoder: nn.Sequential | None = None

    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

    # ------------------------------------------------------------------
    # Debug capture (same pattern as your tPatchGNN) :contentReference[oaicite:3]{index=3}
    # ------------------------------------------------------------------
    def _dbg(self, key: str, tensor: torch.Tensor, mask: torch.Tensor = None):
        if not self.debug_internal:
            return
        try:
            if tensor is None:
                self.last_debug[key] = {"note": "None"}
                return
            if mask is None:
                self.last_debug[key] = _safe_stats(tensor)
            else:
                self.last_debug[key] = _masked_stats(tensor, mask)
        except Exception:
            # never break training due to debug
            pass

    def _build_heads_if_needed(self, fused_dim: int, device: torch.device):
        """
        Builds (optionally) a projection from fused_dim -> proj_dim
        and a small MLP decoder that combines hidden state with TE and
        predicts per-node values at future time steps.
        """
        final_dim = fused_dim
        if self._proj_dim is not None and fused_dim != self._proj_dim:
            self.fuse_proj = nn.Linear(fused_dim, self._proj_dim, device=device)
            final_dim = self._proj_dim

        self.decoder = nn.Sequential(
            nn.Linear(final_dim + self._te_dim, final_dim, device=device),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim, final_dim, device=device),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim, 1, device=device),
        )

    def extra_loss(self) -> torch.Tensor:
        """
        Compatibility hook for training loops that expect an auxiliary regularizer.
        For 'concat' fusion this is zero.
        """
        dev = self._device()
        return torch.tensor(0.0, device=dev)

    def forward(self, X_list, tt_list, mk_list, time_steps_to_predict):
        assert len(X_list) == len(tt_list) == len(mk_list) == len(self.submodels), \
            "Lists must have same length as number of submodels"

        device = self._device()

        # ---- Debug: high-level inputs ----
        # time_steps_to_predict is the key thing that exposed your normalization issue before,
        # so it’s useful to capture its range each forward. :contentReference[oaicite:4]{index=4}
        self._dbg("ms_tp_to_predict", time_steps_to_predict)
        # Capture per-scale time ranges to quickly spot “tt max too small” issues
        for i, tt in enumerate(tt_list):
            self._dbg(f"ms_tt_list[{i}]", tt)

        # Encode each scale with its own single-scale tPatchGNN
        reps = []
        for i, (mdl, X, tt, mk) in enumerate(zip(self.submodels, X_list, tt_list, mk_list)):
            Xd  = X.to(device)
            ttd = tt.to(device)
            mkd = mk.to(device)

            # Debug per-scale tensors
            self._dbg(f"ms_X_list[{i}]", Xd)
            self._dbg(f"ms_mk_list[{i}]", mkd)

            h_i = mdl.encode_from_patched(Xd, ttd, mkd)  # (B, N, D_k)
            reps.append(h_i)
            self._dbg(f"ms_rep[{i}]", h_i)

        # Only concat fusion supported in this file
        if self._fusion != "concat":
            raise NotImplementedError("Only 'concat' fusion is implemented in this version.")
        H = torch.cat(reps, dim=-1)  # (B, N, sum_k D_k)
        self._dbg("ms_fused_H_preproj", H)

        # Build heads lazily on first forward
        if self.decoder is None:
            self._build_heads_if_needed(H.shape[-1], device)

        # Optional projection back to a common hidden size
        if self.fuse_proj is not None:
            H = self.fuse_proj(H)  # (B, N, hid_dim)
            self._dbg("ms_fused_H_postproj", H)

        B, N, F = H.shape
        Lp = time_steps_to_predict.shape[-1]

        # Tile hidden features over prediction horizon
        H_rep = H.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, N, Lp, F)
        self._dbg("ms_H_rep", H_rep)

        # Use the TE module from the first submodel (shared weights assumed across submodels)
        te_pred = self.submodels[0].LearnableTE(
            time_steps_to_predict.view(B, 1, Lp, 1).repeat(1, N, 1, 1).to(device)
        )  # (B, N, Lp, te_dim)
        self._dbg("ms_te_pred", te_pred)

        dec_in = torch.cat([H_rep, te_pred], dim=-1)  # (B, N, Lp, F + te_dim)
        self._dbg("ms_dec_in", dec_in)

        out = self.decoder(dec_in).squeeze(-1)        # (B, N, Lp)
        self._dbg("ms_dec_out_raw_BNLp", out)
        if self.debug_internal:
            try:
                last_linear = self.decoder[-1]   # nn.Linear(final_dim, 1)

                self.last_debug["decoder_last_weight"] = {
                    "shape": tuple(last_linear.weight.shape),
                    "abs_mean": float(last_linear.weight.abs().mean().item()),
                    "abs_max":  float(last_linear.weight.abs().max().item()),
                }

                if last_linear.bias is not None:
                    self.last_debug["decoder_last_bias"] = {
                        "mean": float(last_linear.bias.mean().item()),
                        "abs_max": float(last_linear.bias.abs().max().item()),
                    }
            except Exception:
                pass


        out = out.permute(0, 2, 1).unsqueeze(0)       # (1, B, Lp, N)
        self._dbg("ms_outputs_1BLpN", out)

        return out
