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
        self._te_dim = te_dim
        # if not None, project fused features back to this dim (usually = hid_dim)
        self._proj_dim = proj_dim
        # fusion mode: "concat" (original), "attn", or "gated"
        self._fusion = fusion

        # Lazily built on first forward (so we know fused_dim)
        self.fuse_proj: nn.Linear | None = None
        self.decoder: nn.Sequential | None = None

        # --- For "attn" fusion (lazy init) ---
        self.attn_V: nn.Linear | None = None
        self.attn_u: nn.Parameter | None = None

        # --- For "gated" fusion (lazy init) ---
        self.gate_mlp: nn.Sequential | None = None

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
        For now, returns 0 for all fusion modes.
        """
        dev = self._device()
        return torch.tensor(0.0, device=dev)

    # ------------ Attention-over-scales fusion ------------

    def _fuse_attn(self, reps, device: torch.device) -> torch.Tensor:
        """
        reps: list of K tensors, each of shape (B, N, D)
        returns: H of shape (B, N, D)

        We learn attention weights over scales and take a weighted sum.
        All submodels must share the same hidden dim for this mode.
        """
        # Ensure all hidden dims match
        d0 = reps[0].shape[-1]
        for r in reps[1:]:
            if r.shape[-1] != d0:
                raise ValueError(
                    "All submodels must have same hidden dim for 'attn' fusion."
                )

        # Stack over a new "scale" dimension K
        # H_stack: (B, N, K, D)
        H_stack = torch.stack(reps, dim=2)
        B, N, K, D = H_stack.shape

        # Lazily create attention parameters
        if self.attn_V is None:
            self.attn_V = nn.Linear(D, D, device=device)
        if self.attn_u is None:
            self.attn_u = nn.Parameter(torch.randn(D, device=device))

        # Apply V and tanh on the last dimension D
        # Vh: (B, N, K, D)
        Vh = torch.tanh(self.attn_V(H_stack))
        # scores: (B, N, K)
        scores = torch.einsum("bnkd,d->bnk", Vh, self.attn_u)
        # attention over scales
        alpha = torch.softmax(scores, dim=2)  # (B, N, K)
        # weighted concat instead of weighted sum
        H_weighted = alpha.unsqueeze(-1) * H_stack   # (B, N, K, D)
        H = H_weighted.reshape(B, N, K * D)          # (B, N, K*D)
        return H

    # ------------ Gated fusion over scales ------------

    def _fuse_gated(self, reps, device: torch.device) -> torch.Tensor:
        """
        reps: list of K tensors, each of shape (B, N, D)
        returns: H of shape (B, N, D)

        Learn a gating weight per scale for each (B, N) using a small MLP.
        """
        d0 = reps[0].shape[-1]
        for r in reps[1:]:
            if r.shape[-1] != d0:
                raise ValueError(
                    "All submodels must have same hidden dim for 'gated' fusion."
                )

        # Stack: (B, N, K, D)
        H_stack = torch.stack(reps, dim=2)
        B, N, K, D = H_stack.shape

        # Flatten node + batch to apply MLP: (B*N, K, D)
        H_flat = H_stack.view(B * N, K, D)

        # Lazy build gate MLP
        if self.gate_mlp is None:
            # MLP processes each scale's D-dim vector -> 1 logit
            self.gate_mlp = nn.Sequential(
                nn.Linear(D, D, device=device),
                nn.ReLU(inplace=True),
                nn.Linear(D, 1, device=device),
            )

        # gate_logits: (B*N, K, 1) -> (B*N, K)
        gate_logits = self.gate_mlp(H_flat).squeeze(-1)
        gates = torch.softmax(gate_logits, dim=1)  # (B*N, K)

        # Reshape gates to (B, N, K, 1)
        gates = gates.view(B, N, K, 1)

        # weighted concat instead of weighted sum
        H_weighted = gates * H_stack      # (B, N, K, D)
        H = H_weighted.reshape(B, N, K * D)   # (B, N, K*D)
        return H

    # ----------------------------------------------------------------------

    def _fuse_gated_feat_wconcat(self, reps, device: torch.device) -> torch.Tensor:
        """
        reps: list of K tensors, each of shape (B, N, D)
        returns: H of shape (B, N, K*D)

        Feature-wise gating:
        for each feature d, learn separate weights across scales.
        """
        d0 = reps[0].shape[-1]
        for r in reps[1:]:
            if r.shape[-1] != d0:
                raise ValueError(
                    "All submodels must have same hidden dim for 'gated_feat_wconcat' fusion."
                )

        H_stack = torch.stack(reps, dim=2)   # (B, N, K, D)
        B, N, K, D = H_stack.shape

        # Lazy build feature-wise gate MLP
        if not hasattr(self, "gate_mlp_feat") or self.gate_mlp_feat is None:
            self.gate_mlp_feat = nn.Sequential(
                nn.Linear(D, D, device=device),
                nn.ReLU(inplace=True),
                nn.Linear(D, D, device=device),
            )

        # feature-wise logits: (B, N, K, D)
        gate_logits = self.gate_mlp_feat(H_stack)

        # softmax over scales K, separately for each feature d
        gates = torch.softmax(gate_logits, dim=2)   # (B, N, K, D)

        # weighted concat
        H_weighted = gates * H_stack                # (B, N, K, D)
        H = H_weighted.reshape(B, N, K * D)         # (B, N, K*D)
        return H
    

    # ----------------------------------------------------------------------
    def _fuse_gated_feat(self, reps, device: torch.device) -> torch.Tensor:
        """
        reps: list of K tensors, each of shape (B, N, D)
        returns: H of shape (B, N, D)

        Feature-wise gating with weighted sum over scales.
        """
        d0 = reps[0].shape[-1]
        for r in reps[1:]:
            if r.shape[-1] != d0:
                raise ValueError(
                    "All submodels must have same hidden dim for 'gated_feat' fusion."
                )

        H_stack = torch.stack(reps, dim=2)   # (B, N, K, D)
        B, N, K, D = H_stack.shape

        if not hasattr(self, "gate_mlp_feat") or self.gate_mlp_feat is None:
            self.gate_mlp_feat = nn.Sequential(
                nn.Linear(D, D, device=device),
                nn.ReLU(inplace=True),
                nn.Linear(D, D, device=device),
            )

        gate_logits = self.gate_mlp_feat(H_stack)   # (B, N, K, D)
        gates = torch.softmax(gate_logits, dim=2)   # softmax over K

        H = (gates * H_stack).sum(dim=2)            # (B, N, D)
        return H
    # ----------------------------------------------------------------------

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

        # ---------- FUSION ----------
        if self._fusion == "concat":
            # ORIGINAL BEHAVIOUR (unchanged)
            H = torch.cat(reps, dim=-1)  # (B, N, sum_k D_k)

        elif self._fusion == "attn":
            # attention over scales
            H = self._fuse_attn(reps, device=device)  # (B, N, K*D)

        elif self._fusion == "gated":
            # gating over scales
            H = self._fuse_gated(reps, device=device)  # (B, N, K*D)

        elif self._fusion == "gated_feat_wconcat":
            H = self._fuse_gated_feat_wconcat(reps, device=device)

        elif self._fusion == "gated_feat":
            H = self._fuse_gated_feat(reps, device=device)

        else:
            # Keep old error pattern for unknown modes
            raise NotImplementedError(
                f"Fusion mode '{self._fusion}' is not implemented. "
                "Supported: 'concat', 'attn', 'gated', 'gated_feat', 'gated_feat_wconcat'."
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
        out = self.decoder(dec_in).squeeze(-1)  # (B, N, Lp)
        out = out.permute(0, 2, 1).unsqueeze(0)  # (1, B, Lp, N)
        return out
