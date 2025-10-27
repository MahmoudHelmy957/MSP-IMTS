import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTPatchGNN(nn.Module):
    """
    Wraps K single-scale tPatchGNN encoders.

    Forward expects:
      - X_list, tt_list, mk_list: lists of length K, each shaped (B, M_k, L, N)
      - time_steps_to_predict: (B, Lp)
    Returns:
      - (1, B, Lp, N)
    """

    def __init__(
        self,
        submodels,
        te_dim=10,
        proj_dim=None,             # usually = hid_dim
        fusion="concat",           # 'concat' | 'scale_attn'
        attn_hidden=32,            # only for 'scale_attn'
        attn_temp=1.0,
        attn_dropout=0.0,
        attn_norm=False,
        attn_reg=0.0,              # entropy reg weight on attention
    ):
        super().__init__()
        assert len(submodels) >= 2, "Use >= 2 scales."
        assert fusion in ("concat", "scale_attn")
        self.submodels   = nn.ModuleList(submodels)
        self._te_dim     = te_dim
        self._proj_dim   = proj_dim
        self._fusion     = fusion

        # --- scale-attn hyperparams ---
        self._attn_hidden  = attn_hidden
        self._attn_temp    = attn_temp
        self._attn_dropout = attn_dropout
        self._attn_norm    = attn_norm
        self._attn_reg     = attn_reg

        # lazy-built heads (need to know fused_dim first)
        self.fuse_proj = None
        self.decoder   = None

        # for scale-attn (built lazily because we need H)
        self.attn_mlp  = None
        self.ln        = None

        # for logging / regularization
        self._last_alphas = None

    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

    def _build_dec_if_needed(self, fused_dim, device):
        """Build projection + decoder heads once we know the fused feature size."""
        final_dim = fused_dim
        if self._proj_dim is not None and fused_dim != self._proj_dim:
            self.fuse_proj = nn.Linear(fused_dim, self._proj_dim, device=device)
            final_dim = self._proj_dim

        if self._fusion == "scale_attn" and self._attn_norm and self.ln is None:
            self.ln = nn.LayerNorm(final_dim, device=device)

        if self.decoder is None:
            self.decoder = nn.Sequential(
                nn.Linear(final_dim + self._te_dim, final_dim, device=device),
                nn.ReLU(inplace=True),
                nn.Linear(final_dim, final_dim, device=device),
                nn.ReLU(inplace=True),
                nn.Linear(final_dim, 1, device=device),
            )

    def _build_attn_if_needed(self, H_dim, device):
        """Build attention MLP for scale_attn only (maps R_{H} -> R_{1})."""
        if self.attn_mlp is None:
            hidden = max(8, self._attn_hidden)
            self.attn_mlp = nn.Sequential(
                nn.Linear(H_dim, hidden, device=device),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1, device=device),
            )
            if self._attn_dropout > 0:
                self.attn_drop = nn.Dropout(self._attn_dropout)
            else:
                self.attn_drop = nn.Identity()

    def extra_loss(self):
        """
        Optional regularization loss (entropy on scale weights).
        Returns a scalar tensor (0 for concat).
        """
        if self._fusion != "scale_attn" or self._attn_reg <= 0.0 or self._last_alphas is None:
            return torch.tensor(0.0, device=self._device())
        # self._last_alphas: (B, N, S, 1), sum over S = 1
        p = self._last_alphas.clamp(min=1e-8)
        ent = -(p * torch.log(p)).sum(dim=2).mean()  # mean over B,N (keep dim=1 for channel)
        # We *subtract* entropy to penalize peaky attention (or add -ent),
        # but many prefer adding negative entropy (i.e., ent_reg = -ent) to *encourage* high entropy.
        # Here we add (-ent) so larger entropy reduces loss:
        return (-ent) * self._attn_reg

    def forward(self, X_list, tt_list, mk_list, time_steps_to_predict):
        assert len(X_list) == len(tt_list) == len(mk_list) == len(self.submodels)
        device = self._device()
        self._last_alphas = None  # reset per forward

        # ----- Encode each scale -> (B, N, H) -----
        reps = []
        for mdl, X, tt, mk in zip(self.submodels, X_list, tt_list, mk_list):
            reps.append(mdl.encode_from_patched(X.to(device), tt.to(device), mk.to(device)))  # (B, N, H)

        B, N, H = reps[0].shape
        S = len(reps)

        # ----- Fuse across scales -----
        if self._fusion == "concat":
            Hf = torch.cat(reps, dim=-1)  # (B, N, S*H)
        else:
            # scale-attn: get per-(B,N) weights across S encoders
            self._build_attn_if_needed(H_dim=H, device=device)
            Hstk = torch.stack(reps, dim=2)                      # (B, N, S, H)
            scores = self.attn_mlp(Hstk)                         # (B, N, S, 1)
            if self._attn_temp != 1.0:
                scores = scores / float(self._attn_temp)
            alphas = torch.softmax(scores, dim=2)                # sum over S = 1
            alphas = self.attn_drop(alphas)
            self._last_alphas = alphas                           # for logging / reg

            Hf = (alphas * Hstk).sum(dim=2)                      # (B, N, H)

        # ----- Projection + decoder -----
        self._build_dec_if_needed(Hf.shape[-1], device)
        if self.fuse_proj is not None:
            Hf = self.fuse_proj(Hf)                              # (B, N, hid_dim)
        if self._fusion == "scale_attn" and self.ln is not None:
            Hf = self.ln(Hf)

        # Decode exactly like tPatchGNN.forecasting()
        Lp = time_steps_to_predict.shape[-1]
        H_rep = Hf.unsqueeze(2).repeat(1, 1, Lp, 1)              # (B, N, Lp, F)

        te_pred = self.submodels[0].LearnableTE(
            time_steps_to_predict.view(B, 1, Lp, 1).repeat(1, N, 1, 1).to(device)
        )                                                         # (B, N, Lp, te_dim)

        dec_in = torch.cat([H_rep, te_pred], dim=-1)             # (B, N, Lp, F+te_dim)
        out = self.decoder(dec_in).squeeze(-1).permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, N)
        return out
