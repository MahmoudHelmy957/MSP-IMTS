# model/multiscale_tpatchgnn.py
import torch
import torch.nn as nn


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
    def __init__(self, submodels, te_dim: int = 10, proj_dim: int | None = None, fusion: str = "concat"):
        super().__init__()
        assert len(submodels) >= 2, "Use >= 2 scales."
        self.submodels = nn.ModuleList(submodels)
        self._te_dim   = te_dim
        self._proj_dim = proj_dim     # if not None, project fused features back to this dim (usually = hid_dim)
        self._fusion   = fusion

        # Lazily built on first forward (so we know fused_dim)
        self.fuse_proj: nn.Linear | None = None
        self.decoder: nn.Sequential | None = None

    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

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

        # Encode each scale with its own single-scale tPatchGNN
        reps = []
        for mdl, X, tt, mk in zip(self.submodels, X_list, tt_list, mk_list):
            reps.append(mdl.encode_from_patched(X.to(device), tt.to(device), mk.to(device)))  # (B, N, D_k)

        # Only concat fusion supported in this file
        if self._fusion != "concat":
            raise NotImplementedError("Only 'concat' fusion is implemented in this version.")
        H = torch.cat(reps, dim=-1)  # (B, N, sum_k D_k)

        # Build heads lazily on first forward
        if self.decoder is None:
            self._build_heads_if_needed(H.shape[-1], device)

        # Optional projection back to a common hidden size
        if self.fuse_proj is not None:
            H = self.fuse_proj(H)  # (B, N, hid_dim)

        B, N, F = H.shape
        Lp = time_steps_to_predict.shape[-1]

        # Tile hidden features over prediction horizon
        H_rep = H.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B, N, Lp, F)

        # Use the TE module from the first submodel (shared weights assumed across submodels)
        te_pred = self.submodels[0].LearnableTE(
            time_steps_to_predict.view(B, 1, Lp, 1).repeat(1, N, 1, 1).to(device)
        )  # (B, N, Lp, te_dim)

        dec_in = torch.cat([H_rep, te_pred], dim=-1)  # (B, N, Lp, F + te_dim)
        out = self.decoder(dec_in).squeeze(-1)        # (B, N, Lp)
        out = out.permute(0, 2, 1).unsqueeze(0)       # (1, B, Lp, N) to match original API
        return out
