import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTPatchGNN(nn.Module):
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
        assert fusion in ("concat", "scale_attn")
        self.submodels = nn.ModuleList(submodels)
        self._te_dim = te_dim
        self._proj_dim = proj_dim   # usually = hid_dim
        self._fusion = fusion

        # lazy-built heads
        self.fuse_proj = None
        self.decoder  = None
        self.attn_mlp = None    # for scale_attn
        self.ln       = None    # for scale_attn (post-proj LayerNorm)

    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

    def _build_heads_if_needed(self, fused_dim, device):
        final_dim = fused_dim
        if self._proj_dim is not None and fused_dim != self._proj_dim:
            self.fuse_proj = nn.Linear(fused_dim, self._proj_dim, device=device)
            final_dim = self._proj_dim

        if self._fusion == "scale_attn" and self.ln is None:
            self.ln = nn.LayerNorm(final_dim, device=device)

        if self.decoder is None:
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

        # encode each scale -> (B, N, H)
        reps = []
        for mdl, X, tt, mk in zip(self.submodels, X_list, tt_list, mk_list):
            reps.append(mdl.encode_from_patched(X.to(device), tt.to(device), mk.to(device)))  # (B,N,H)

        B, N, H = reps[0].shape
        S = len(reps)

        # fuse across scales
        if self._fusion == "concat":
            Hf = torch.cat(reps, dim=-1)                                  # (B, N, H*S)
        else:  # "scale_attn"
            Hstk = torch.stack(reps, dim=2)                                # (B, N, S, H)
            if self.attn_mlp is None:
                hidden = max(8, H // 2)
                self.attn_mlp = nn.Sequential(
                    nn.Linear(H, hidden, device=device),
                    nn.ReLU(),
                    nn.Linear(hidden, 1, device=device),
                )
            scores = self.attn_mlp(Hstk)                                   # (B, N, S, 1)
            alphas = torch.softmax(scores, dim=2)                          # softmax over scales S
            Hf = (alphas * Hstk).sum(dim=2)                                # (B, N, H)

        # projection + decoder construction
        self._build_heads_if_needed(Hf.shape[-1], device)
        if self.fuse_proj is not None:
            Hf = self.fuse_proj(Hf)                                        # (B, N, hid_dim)
        if self._fusion == "scale_attn" and self.ln is not None:
            Hf = self.ln(Hf)

        # decode exactly like tPatchGNN.forecasting
        Lp = time_steps_to_predict.shape[-1]
        H_rep = Hf.unsqueeze(2).repeat(1, 1, Lp, 1)                        # (B, N, Lp, F)
        te_pred = self.submodels[0].LearnableTE(
            time_steps_to_predict.view(B, 1, Lp, 1).repeat(1, N, 1, 1).to(device)
        )                                                                   # (B, N, Lp, te_dim)
        dec_in = torch.cat([H_rep, te_pred], dim=-1)                        # (B, N, Lp, F+te)
        out = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1).unsqueeze(0)
        return out  # (1, B, Lp, N)
