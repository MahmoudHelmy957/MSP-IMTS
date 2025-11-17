import torch
import torch.nn as nn
from model.fusion_blocks import NodeMixerBlock


class tPatchGNN_WithMixer(nn.Module):
    """
    Wraps a single-scale tPatchGNN with a NodeMixer fusion block on top.
    """

    def __init__(
        self,
        base_model,        # a tPatchGNN instance
        hidden_mult=2.0,
        drop=0.1
    ):
        super().__init__()
        self.base = base_model
        self.N = base_model.N
        self.hid_dim = base_model.hid_dim

        # Add NodeMixer after hidden representation
        self.mixer = NodeMixerBlock(
            n_tokens=self.N,
            d_model=self.hid_dim,
            hidden_mult=hidden_mult,
            drop=drop,
        )

        # Reuse decoder from tPatchGNN
        self.decoder = base_model.decoder
        self.LearnableTE = base_model.LearnableTE

    def forward(self, X, tt, mk, tp_to_predict):

        # ---- FIX 1: unwrap lists (single-scale datasets produce lists of len=1) ----
        if isinstance(X, (list, tuple)):
            X = X[0]
        if isinstance(tt, (list, tuple)):
            tt = tt[0]
        if isinstance(mk, (list, tuple)):
            mk = mk[0]
    
        # ---- 1. Encode with base single-scale tPatchGNN ----
        h = self.base.encode_from_patched(X, tt, mk)   # (B, N, M, D) for activity
    
        # ---- FIX 2: reduce patch dimension M → mean pooling ----
        if h.dim() == 4:   # (B, N, M, D)
            h = h.mean(dim=2)    # -> (B, N, D)
    
        # ---- 2. NodeMixer fusion ----
        h = self.mixer(h)       # (B, N, D)
    
        # ---- 3. Standard tPatchGNN decoding ----
        B, N, F = h.shape
        Lp = tp_to_predict.shape[-1]
    
        # ---- FIX 3: repeat along Lp only (NOT along N!) ----
        h_rep = h.unsqueeze(2).repeat(1, 1, Lp, 1)      # (B, N, Lp, D)
    
        # temporal encodings for future times
        te_pred = self.LearnableTE(
            tp_to_predict.view(B, 1, Lp, 1).repeat(1, N, 1, 1)
        )                                               # (B, N, Lp, te_dim)
    
        # concat decoder input
        dec_in = torch.cat([h_rep, te_pred], dim=-1)    # (B, N, Lp, D + te_dim)
    
        # decode forecasts
        out = self.decoder(dec_in).squeeze(-1)          # (B, N, Lp)
        out = out.permute(0, 2, 1).unsqueeze(0)         # (1, B, Lp, N)
    
        return out

    def forecasting(self, tp_to_predict, X, tt, mk):
        return self.forward(X, tt, mk, tp_to_predict)
