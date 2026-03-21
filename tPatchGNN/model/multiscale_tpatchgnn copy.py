# model/multiscale_tpatchgnn.py
import torch
import torch.nn as nn


class MultiScaleTPatchGNN(nn.Module):
    """
    Wraps K single-scale tPatchGNN encoders and fuses their representations
    (or predictions, depending on fusion mode).

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
    ):
        super().__init__()
        assert len(submodels) >= 2, "Use >= 2 scales."
        self.submodels = nn.ModuleList(submodels)
        self._K = len(submodels)

        self._te_dim = te_dim
        # if not None, project fused features back to this dim (usually = hid_dim)
        self._proj_dim = proj_dim
        # fusion: "concat", "attn", "gated", "cs_attn"
        self._fusion = fusion

        # Lazily built on first forward (so we know fused_dim)
        self.fuse_proj: nn.Linear | None = None
        self.decoder: nn.Sequential | None = None

        # --- shared attention parameters (for attn / cs_attn) ---
        self.attn_V: nn.Linear | None = None   # V in R^{dxd}
        self.attn_u: nn.Parameter | None = None  # u in R^{d}

        # --- gating MLP (for "gated") ---
        self.gate_mlp: nn.Sequential | None = None

        # --- per-scale W^(k) (for "cs_attn") ---
        self.cs_W: nn.ModuleList | None = None
        # optional dropout for cs_attn fused reps
        self.cs_dropout: nn.Dropout | None = None


    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

    # ------------------------------------------------------------------
    # Decoder head
    # ------------------------------------------------------------------
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
        For now, returns 0 for all fusion modes.
        """
        dev = self._device()
        return torch.tensor(0.0, device=dev)

    # ------------------------------------------------------------------
    # Helpers for feature-level fusion
    # ------------------------------------------------------------------
    def _ensure_attn_params(self, D: int, device: torch.device):
        if self.attn_V is None:
            self.attn_V = nn.Linear(D, D, device=device)
        if self.attn_u is None:
            self.attn_u = nn.Parameter(torch.zeros(D, device=device))

    def _ensure_gating_mlp(self, D: int, device: torch.device):
        if self.gate_mlp is None:
            self.gate_mlp = nn.Sequential(
                nn.Linear(D, D, device=device),
                nn.Tanh(),
                nn.Linear(D, 1, device=device),
            )

    def _ensure_cs_params(self, D: int, device: torch.device):
        # attention params shared with simple attn
        self._ensure_attn_params(D, device)
        if self.cs_W is None:
            self.cs_W = nn.ModuleList(
                [nn.Linear(D, D, device=device) for _ in range(self._K)]
            )
        if self.cs_dropout is None:
            #tweak p 
            self.cs_dropout = nn.Dropout(p=0.1)

    
    # ---------- simple attention over scales (old "attn") ----------
    def _fuse_attn(self, reps, device: torch.device) -> torch.Tensor:
        """
        reps: list of K tensors, each of shape (B, N, D)
        returns: H of shape (B, N, D)
        """
        d0 = reps[0].shape[-1]
        for r in reps[1:]:
            if r.shape[-1] != d0:
                raise ValueError(
                    "All submodels must have same hidden dim for 'attn' fusion."
                )
        H_stack = torch.stack(reps, dim=2)  # (B, N, K, D)
        B, N, K, D = H_stack.shape

        self._ensure_attn_params(D, device)

        Vh = torch.tanh(self.attn_V(H_stack))          # (B, N, K, D)
        scores = torch.einsum("bnkd,d->bnk", Vh, self.attn_u)  # (B, N, K)
        alpha = torch.softmax(scores, dim=2)           # (B, N, K)
        H = (alpha.unsqueeze(-1) * H_stack).sum(dim=2) # (B, N, D)
        return H

    # ---------- gated fusion over scales ----------
    def _fuse_gated(self, reps, device: torch.device) -> torch.Tensor:
        """
        reps: list of K tensors, each of shape (B, N, D)
        returns: H of shape (B, N, D)
        """
        d0 = reps[0].shape[-1]
        for r in reps[1:]:
            if r.shape[-1] != d0:
                raise ValueError(
                    "All submodels must have same hidden dim for 'gated' fusion."
                )
        H_stack = torch.stack(reps, dim=2)  # (B, N, K, D)
        B, N, K, D = H_stack.shape

        self._ensure_gating_mlp(D, device)

        scores = self.gate_mlp(H_stack).squeeze(-1)    # (B, N, K)
        alpha = torch.softmax(scores, dim=2)           # (B, N, K)
        H = (alpha.unsqueeze(-1) * H_stack).sum(dim=2) # (B, N, D)
        return H

    # ---------- Cross-Scale Attention Fusion (your formula) ----------
    def _fuse_cs_attn(self, reps, device: torch.device) -> torch.Tensor:
        """
        Cross-Scale Attention Fusion, roughly:

          e_m^{(k)} = u^T tanh(V h_m^{(k)})
          α_m^{(k)} = softmax_k e_m^{(k)}
          z_m       = Σ_k α_m^{(k)} W^{(k)} h_m^{(k)}

        Here "m" corresponds to node index (per node fusion).
        reps: list of K tensors, each (B, N, D)
        returns: H fused, (B, N, D)
        """
        d0 = reps[0].shape[-1]
        for r in reps[1:]:
            if r.shape[-1] != d0:
                raise ValueError(
                    "All submodels must have same hidden dim for 'cs_attn' fusion."
                )

        H_stack = torch.stack(reps, dim=2)        # (B, N, K, D)
        B, N, K, D = H_stack.shape
        assert K == self._K

        # ensure params (V, u, and W^(k))
        self._ensure_cs_params(D, device)

        # 1. scores per scale
        Vh = torch.tanh(self.attn_V(H_stack))     # (B, N, K, D)
        scores = torch.einsum("bnkd,d->bnk", Vh, self.attn_u)  # (B, N, K)

        # 2. attention weights over scales
        alpha = torch.softmax(scores, dim=2)      # (B, N, K)

        # 3. fused representation using scale-specific W^(k)
        # Apply W^(k) to each rep_k -> H_W in (B, N, K, D)
        H_W_list = []
        for k, r in enumerate(reps):
            h_k = self.cs_W[k](r)                 # (B, N, D)
            if self.cs_dropout is not None:
                h_k = self.cs_dropout(h_k)
            H_W_list.append(h_k)
        H_W = torch.stack(H_W_list, dim=2)        # (B, N, K, D)


        H = (alpha.unsqueeze(-1) * H_W).sum(dim=2)  # (B, N, D)
        return H

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, X_list, tt_list, mk_list, time_steps_to_predict):
        assert len(X_list) == len(tt_list) == len(mk_list) == len(
            self.submodels
        ), "Lists must have same length as number of submodels"

        device = self._device()

        # Encode each scale with its own single-scale tPatchGNN
        reps = []
        for mdl, X, tt, mk in zip(self.submodels, X_list, tt_list, mk_list):
            reps.append(
                mdl.encode_from_patched(
                    X.to(device), tt.to(device), mk.to(device)
                )
            )  # (B, N, D_k)

        # ---------- FEATURE-LEVEL FUSION ----------
        if self._fusion == "concat":
            H = torch.cat(reps, dim=-1)  # (B, N, sum_k D_k)

        elif self._fusion == "attn":
            H = self._fuse_attn(reps, device=device)  # (B, N, D)

        elif self._fusion == "gated":
            H = self._fuse_gated(reps, device=device)  # (B, N, D)

        elif self._fusion == "cs_attn":
            H = self._fuse_cs_attn(reps, device=device)  # (B, N, D)

        else:
            raise NotImplementedError(
                f"Fusion mode '{self._fusion}' is not implemented. "
                "Supported: 'concat', 'attn', 'gated', 'cs_attn'."
            )

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

        # Use the TE module from the first submodel (shared weights assumed)
        te_pred = self.submodels[0].LearnableTE(
            time_steps_to_predict.view(B, 1, Lp, 1)
            .repeat(1, N, 1, 1)
            .to(device)
        )  # (B, N, Lp, te_dim)

        dec_in = torch.cat([H_rep, te_pred], dim=-1)  # (B, N, Lp, F + te_dim)
        out = self.decoder(dec_in).squeeze(-1)        # (B, N, Lp)
        out = out.permute(0, 2, 1).unsqueeze(0)       # (1, B, Lp, N)
        return out
