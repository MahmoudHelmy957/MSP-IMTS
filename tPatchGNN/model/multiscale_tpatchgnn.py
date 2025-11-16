
# model/multiscale_tpatchgnn.py
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from model.fusion_blocks import NodeMixerBlock


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
        *,
        use_node_mixer: bool = True,
        node_mixer_kwargs: Optional[Mapping[str, float]] = None,
        track_fusion_stats: bool = False,
        fusion_block: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert len(submodels) >= 2, "Use >= 2 scales."
        self.submodels = nn.ModuleList(submodels)
        self._te_dim   = te_dim
        self._proj_dim = proj_dim     # if not None, project fused features back to this dim (usually = hid_dim)
        self._fusion   = fusion
        self._fused_dim: int | None = None

        # Infer shared structural information so fusion layers can be
        # instantiated up front (and therefore registered with the optimizer).
        with torch.no_grad():
            n_tokens = getattr(self.submodels[0], "N", None)
            assert n_tokens is not None, "Submodels must expose attribute 'N' for number of nodes."
            self._n_tokens = n_tokens

            fused_dim = 0
            for mdl in self.submodels:
                mdl_dim = getattr(mdl, "hid_dim", None)
                assert mdl_dim is not None, "Submodels must expose attribute 'hid_dim'."
                fused_dim += mdl_dim
                assert getattr(mdl, "N", n_tokens) == n_tokens, "All submodels must share the same number of nodes."

        self._fused_dim = fused_dim

        if fusion_block is not None:
            self.fusion_block: nn.Module = fusion_block
        else:
            if use_node_mixer:
                nm_kwargs: Dict[str, float] = {"hidden_mult": 2.0, "drop": 0.1}
                if node_mixer_kwargs is not None:
                    nm_kwargs.update(dict(node_mixer_kwargs))
                self.fusion_block = NodeMixerBlock(
                    n_tokens=self._n_tokens,
                    d_model=fused_dim,
                    hidden_mult=float(nm_kwargs.get("hidden_mult", 2.0)),
                    drop=float(nm_kwargs.get("drop", 0.1)),
                )
            else:
                self.fusion_block = nn.Identity()

        # Backwards-compatibility attribute (e.g. checkpoints looking for "node_mixer")
        self.node_mixer = self.fusion_block

        self._track_fusion_stats = track_fusion_stats and not isinstance(self.fusion_block, nn.Identity)
        if self._track_fusion_stats:
            self.register_buffer("_fusion_batch_count", torch.tensor(0, dtype=torch.long))
            self.register_buffer("_fusion_delta_rms_sum", torch.tensor(0.0))
            self.register_buffer("_fusion_input_rms_sum", torch.tensor(0.0))
            self.register_buffer("_fusion_ratio_sum", torch.tensor(0.0))
            self.register_buffer("_fusion_max_abs", torch.tensor(0.0))

        # Lazily built on first forward (so we know fused_dim)
        self.fuse_proj: nn.Linear | None = None
        self.decoder: nn.Sequential | None = None

    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

    def _build_heads_if_needed(self, fused_dim: int, device: torch.device):
        """
        Builds a small MLP decoder that combines hidden state with TE and
        predicts per-node values at future time steps.
        We skip any extra projection and just use fused_dim directly.
        """
        final_dim = fused_dim

        # No projection: fuse_proj is identity
        self.fuse_proj = nn.Identity()

        self.decoder = nn.Sequential(
            nn.Linear(final_dim + self._te_dim, final_dim),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim, final_dim),
            nn.ReLU(inplace=True),
            nn.Linear(final_dim, 1),
        )

    @torch.no_grad()
    def _device(self):
        return next(self.submodels[0].parameters()).device

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
            r = mdl.encode_from_patched(X.to(device), tt.to(device), mk.to(device))
            # FIX: flatten patch dimension M
            if r.dim() == 4:   # (B, N, M, D)
                r = r.mean(dim=2)   # -> (B, N, D)
            reps.append(r)
        # Only concat fusion supported in this file
        if self._fusion != "concat":
            raise NotImplementedError("Only 'concat' fusion is implemented in this version.")
        H = torch.cat(reps, dim=-1)  # (B, N, sum_k D_k)
        # --- Node Mixer (cross-node interactions) ---
        if getattr(self, "node_mixer", None) is None:
            self.node_mixer = NodeMixerBlock(
                n_tokens=H.shape[1],
                d_model=H.shape[-1],
                hidden_mult=2,
                drop=0.1
            ).to(device)
        
        H = self.node_mixer(H)
        # --- end Mixer block ---


        # Build heads lazily on first forward
        if self.decoder is None:
            self._build_heads_if_needed(H.shape[-1], device)
            # Fix: move newly-created modules to the correct device
            self.fuse_proj = self.fuse_proj.to(device)
            self.decoder = self.decoder.to(device)

        H_in = H
        H = self.fusion_block(H)

        if self._track_fusion_stats:
            with torch.no_grad():
                diff = H - H_in
                delta_rms = diff.pow(2).mean().sqrt()
                input_rms = H_in.pow(2).mean().sqrt()
                ratio = delta_rms / (input_rms + 1e-12)
                self._fusion_delta_rms_sum += delta_rms
                self._fusion_input_rms_sum += input_rms
                self._fusion_ratio_sum += ratio
                current_max = diff.abs().max()
                if current_max.item() > self._fusion_max_abs.item():
                    self._fusion_max_abs.copy_(current_max)
                self._fusion_batch_count += 1

        # No projection: fuse_proj is Identity
        H = self.fuse_proj(H)  # (B, N, fused_dim)
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
    def forecasting(self, tp_to_predict, X_list, tt_list, mk_list, **kwargs):
        """
        Wrapper for compatibility with evaluation.compute_all_losses().
        It simply calls forward().
        """
        out = self.forward(X_list, tt_list, mk_list, tp_to_predict)
        if isinstance(out, torch.Tensor) and out.dim() == 4 and out.shape[0] == 1:
            out = out[0]
        return out
    @property
    def fused_dim(self) -> int:
        assert self._fused_dim is not None
        return self._fused_dim

    def fusion_stats(self, reset: bool = False) -> Dict[str, float]:
        stats: Dict[str, float] = {
            "fusion_type": type(self.fusion_block).__name__,
            "fused_dim": float(self.fused_dim) if self._fused_dim is not None else float("nan"),
            "tracked_batches": 0.0,
        }
        if self._track_fusion_stats:
            batch_count = int(self._fusion_batch_count.item())
            stats["tracked_batches"] = float(batch_count)
            if batch_count > 0:
                stats.update(
                    {
                        "delta_rms_mean": float((self._fusion_delta_rms_sum / batch_count).item()),
                        "input_rms_mean": float((self._fusion_input_rms_sum / batch_count).item()),
                        "delta_to_input_ratio": float((self._fusion_ratio_sum / batch_count).item()),
                        "delta_abs_max": float(self._fusion_max_abs.item()),
                    }
                )
            else:
                stats.update(
                    {
                        "delta_rms_mean": 0.0,
                        "input_rms_mean": 0.0,
                        "delta_to_input_ratio": 0.0,
                        "delta_abs_max": 0.0,
                    }
                )
            if reset:
                self.reset_fusion_stats()
        return stats

    def reset_fusion_stats(self) -> None:
        if not self._track_fusion_stats:
            return
        self._fusion_batch_count.zero_()
        self._fusion_delta_rms_sum.zero_()
        self._fusion_input_rms_sum.zero_()
        self._fusion_ratio_sum.zero_()
        self._fusion_max_abs.zero_()

    def fusion_parameter_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {
            "fusion_type": type(self.fusion_block).__name__,
            "fused_dim": float(self._fused_dim or 0),
            "n_tokens": float(self._n_tokens),
        }
    
        fb = self.fusion_block
        if isinstance(fb, nn.Identity):
            return stats
    
        # Handle both simple Sequential and GraFITi-style NodeMixerBlock
        try:
            token_mixer = getattr(fb.token_mixer, "fn", fb.token_mixer)
            channel_mixer = getattr(fb.channel_mixer, "fn", fb.channel_mixer)
        except AttributeError:
            return stats  # fallback: nothing to extract
    
        # Try to gather parameter norms if available
        for name, module in {"token_mixer": token_mixer, "channel_mixer": channel_mixer}.items():
            for param_name, param in module.named_parameters(recurse=True):
                stats[f"{name}_{param_name}_fro"] = float(param.norm().item())
    
        return stats

