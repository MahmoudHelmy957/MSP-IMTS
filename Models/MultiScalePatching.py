import torch
import torch.nn as nn

class MultiScalePatching(nn.Module):
    """
    Wraps K single-scale tPatchGNN encoders.
    Forward expects:
      - X_list, tt_list, mk_list: lists of length K, each (B, M_k, L, N)
      - time_steps_to_predict: (B, Lp)
    Returns: (1, B, Lp, N)
    """
    def __init__(self, submodels, te_dim=10, proj_dim=None, fusion="concat"):
        super().__init__()
        assert len(submodels) >= 2, "Use >=2 scales."
        self.submodels = nn.ModuleList(submodels)
        self._te_dim = te_dim
        self._proj_dim = proj_dim   # usually = hid_dim
        self._fusion = fusion
        self.fuse_proj = None       # lazy init
        self.decoder  = None

    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

    def _build_heads_if_needed(self, fused_dim, device):
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

    def forward(self, X_list, tt_list, mk_list, time_steps_to_predict):
        assert len(X_list) == len(tt_list) == len(mk_list) == len(self.submodels)
        device = self._device()

        reps = []
        for mdl, X, tt, mk in zip(self.submodels, X_list, tt_list, mk_list):
            reps.append(mdl.encode_from_patched(X.to(device), tt.to(device), mk.to(device)))  # (B,N,D)

        if self._fusion != "concat":
            raise NotImplementedError("Only 'concat' fusion implemented.")
        H = torch.cat(reps, dim=-1)  # (B,N,sum D)

        if self.decoder is None:
            self._build_heads_if_needed(H.shape[-1], device)
        if self.fuse_proj is not None:
            H = self.fuse_proj(H)    # (B,N,hid_dim)

        B, N, F = H.shape
        Lp = time_steps_to_predict.shape[-1]
        H_rep = H.unsqueeze(2).repeat(1, 1, Lp, 1)  # (B,N,Lp,F)

        te_pred = self.submodels[0].LearnableTE(
            time_steps_to_predict.view(B,1,Lp,1).repeat(1,N,1,1).to(device)
        )  # (B,N,Lp,te_dim)

        dec_in = torch.cat([H_rep, te_pred], dim=-1)  # (B,N,Lp,F+te_dim)
        out = self.decoder(dec_in).squeeze(-1).permute(0,2,1).unsqueeze(0)  # (1,B,Lp,N)
        return out
