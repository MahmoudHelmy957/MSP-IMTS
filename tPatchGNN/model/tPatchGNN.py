import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer_EncDec import Encoder, EncoderLayer
from model.SelfAttention_Family import FullAttention, AttentionLayer

import lib.utils as utils
from lib.evaluation import *

# ======================================================================================
# Debug helpers (safe stats for tensors)
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

# ======================================================================================

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x (B, F, N, M)
        # A (B, M, N, N)
        x = torch.einsum('bfnm,bmnv->bfvm', (x, A))  # (B, F, N, M)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        # x (B, F, N, M)
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x (B, F, N, M)
        # support: list of A (B, M, N, N)
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.relu(h)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class tPatchGNN(nn.Module):
    def __init__(self, args, supports=None, dropout=0):
        super(tPatchGNN, self).__init__()
        self.device = args.device
        self.hid_dim = args.hid_dim
        self.N = args.ndim
        self.M = args.npatch
        self.batch_size = None
        self.supports = supports
        self.n_layer = args.nlayer

        # ----------------------- Debug flag & store -----------------------
        # Enable by adding args.debug_internal=1 in your training script parser
        self.debug_internal = bool(getattr(args, "debug_internal", False))
        self.last_debug = {}  # latest forward stats for external logging

        # ----------------------- Intra-time series modeling -----------------------
        # Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim - 1)

        # TTCN
        input_dim = 1 + args.te_dim
        ttcn_dim = args.hid_dim - 1
        self.ttcn_dim = ttcn_dim
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(ttcn_dim, input_dim * ttcn_dim, bias=True),
        )
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))

        d_model = args.hid_dim

        # Transformer
        self.ADD_PE = PositionalEncoding(d_model)
        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=args.nhead, batch_first=True
            )
            self.transformer_encoder.append(
                nn.TransformerEncoder(encoder_layer, num_layers=args.tf_layer)
            )

        # ----------------------- Inter-time series modeling -----------------------
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        nodevec_dim = args.node_dim
        self.nodevec_dim = nodevec_dim
        if supports is None:
            self.supports = []

        # Multi-scale safe device allocation
        self.nodevec1 = nn.Parameter(torch.randn(self.N, nodevec_dim, device=self.device), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(nodevec_dim, self.N, device=self.device), requires_grad=True)

        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        for _ in range(self.n_layer):
            self.nodevec_linear1.append(nn.Linear(args.hid_dim, nodevec_dim))
            self.nodevec_linear2.append(nn.Linear(args.hid_dim, nodevec_dim))
            self.nodevec_gate1.append(nn.Sequential(
                nn.Linear(args.hid_dim + nodevec_dim, 1),
                nn.Tanh(),
                nn.ReLU()
            ))
            self.nodevec_gate2.append(nn.Sequential(
                nn.Linear(args.hid_dim + nodevec_dim, 1),
                nn.Tanh(),
                nn.ReLU()
            ))

        self.supports_len += 1

        self.gconv = nn.ModuleList()
        for _ in range(self.n_layer):
            self.gconv.append(
                gcn(d_model, d_model, dropout, support_len=self.supports_len, order=args.hop)
            )

        # ----------------------- Encoder output layer -----------------------
        self.outlayer = args.outlayer
        enc_dim = args.hid_dim
        if self.outlayer == "Linear":
            self.temporal_agg = nn.Sequential(nn.Linear(args.hid_dim * self.M, enc_dim))
        elif self.outlayer == "CNN":
            self.temporal_agg = nn.Sequential(nn.Conv1d(d_model, enc_dim, kernel_size=self.M))

        # ----------------------- Decoder -----------------------
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim + args.te_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, 1),
        )

    # ------------------------------------------------------------------
    # Debug capture
    # ------------------------------------------------------------------
    def _dbg(self, key: str, tensor: torch.Tensor, mask: torch.Tensor = None):
        if not self.debug_internal:
            return
        try:
            if mask is None:
                self.last_debug[key] = _safe_stats(tensor)
            else:
                self.last_debug[key] = _masked_stats(tensor, mask)
        except Exception:
            # Never break training due to debug
            pass

    def LearnableTE(self, tt):
        # tt: (..., 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def TTCN(self, X_int, mask_X):
        """
        X_int:  (B*N*M, L, F_in)
        mask_X: (B*N*M, L, 1)
        returns: (B*N*M, ttcn_dim)
        """
        N_, Lx, _ = mask_X.shape

        # Per-time filters
        Filter = self.Filter_Generators(X_int)  # (N_, L, F_in*ttcn_dim)

        # Mask invalid time steps before softmax (sequence dim = -2)
        Filter_mask = Filter * mask_X + (1.0 - mask_X) * (-1e8)
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N_, L, F_in*ttcn_dim)

        # Separate (ttcn_dim, F_in): (N_, L, Q, F_in)
        Filter_seqnorm = Filter_seqnorm.view(N_, Lx, self.ttcn_dim, -1).contiguous()

        # Contract: (N_, L, F_in) x (N_, L, Q, F_in) -> (N_, Q)
        ttcn_out = torch.einsum('nlf,nlqf->nq', X_int, Filter_seqnorm)

        h_t = torch.relu(ttcn_out + self.T_bias)  # (N_, ttcn_dim)
        return h_t

    def IMTS_Model(self, x, mask_X):
        """
        x (B*N*M, L, F)
        mask_X (B*N*M, L, 1)
        returns (B, N, hid_dim)
        """
        # mask for the patch
        mask_patch = (mask_X.sum(dim=1) > 0)  # (B*N*M, 1)

        # Debug: input to TTCN
        self._dbg("imts_in_X", x, mask_X.expand_as(x) if x.shape == mask_X.expand_as(x).shape else None)
        self._dbg("imts_in_mask", mask_X)

        # TTCN for patch modeling
        x_patch = self.TTCN(x, mask_X)  # (B*N*M, hid_dim-1)
        self._dbg("ttcn_out", x_patch)

        x_patch = torch.cat([x_patch, mask_patch], dim=-1)  # (B*N*M, hid_dim)
        x_patch = x_patch.view(self.batch_size, self.N, self.M, -1)  # (B, N, M, hid_dim)
        B, N, M, D = x_patch.shape

        x = x_patch
        for layer in range(self.n_layer):
            if layer > 0:  # residual
                x_last = x.clone()

            # Transformer for temporal modeling
            x = x.reshape(B * N, M, -1)  # (B*N, M, F)
            x = self.ADD_PE(x)
            x = self.transformer_encoder[layer](x).view(x_patch.shape)  # (B, N, M, F)

            # Graph structure learning
            nodevec1 = self.nodevec1.view(1, 1, N, self.nodevec_dim).repeat(B, M, 1, 1)
            nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N).repeat(B, M, 1, 1)

            x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))

            x_p1 = x_gate1 * self.nodevec_linear1[layer](x)  # (B, N, M, nodevec_dim)
            x_p2 = x_gate2 * self.nodevec_linear2[layer](x)  # (B, N, M, nodevec_dim)

            nodevec1 = nodevec1 + x_p1.permute(0, 2, 1, 3)  # (B, M, N, nodevec_dim)
            nodevec2 = nodevec2 + x_p2.permute(0, 2, 3, 1)  # (B, M, nodevec_dim, N)

            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1)  # (B, M, N, N)
            new_supports = self.supports + [adp]

            # Graph conv
            x = self.gconv[layer](x.permute(0, 3, 1, 2), new_supports)  # (B, F, N, M)
            x = x.permute(0, 2, 3, 1)  # (B, N, M, F)

            if layer > 0:
                x = x_last + x

        # Output layer
        if self.outlayer == "CNN":
            x = x.reshape(self.batch_size * self.N, self.M, -1).permute(0, 2, 1)  # (B*N, F, M)
            x = self.temporal_agg(x)  # -> (B*N, F, 1)
            x = x.view(self.batch_size, self.N, -1)  # (B, N, F)
        elif self.outlayer == "Linear":
            x = x.reshape(self.batch_size, self.N, -1)  # (B, N, M*F)
            x = self.temporal_agg(x)  # (B, N, hid_dim)

        # Debug: encoder output right before decoding
        self._dbg("enc_out_h", x)
        return x

    # ======================================================================================
    # Multi-scale: encode directly from patched tensors (B, M, L, N)
    # ======================================================================================
    def encode_from_patched(self, X, truth_time_steps, mask):
        """
        X, truth_time_steps, mask: (B, M, L, N)
        returns h: (B, N, hid_dim)
        """
        B, M, L_in, N = X.shape
        self.batch_size = B
        X_ = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        tt_ = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        mk_ = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)

        te_his = self.LearnableTE(tt_)
        X_ = torch.cat([X_, te_his], dim=-1)

        # Debug inputs
        self._dbg("enc_from_patched_Xcat", X_)
        self._dbg("enc_from_patched_mask", mk_)

        h = self.IMTS_Model(X_, mk_)  # (B, N, hid_dim)
        self._dbg("enc_from_patched_h", h)
        return h

    # ======================================================================================
    # Forecasting (single-scale) path
    # ======================================================================================
    def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask=None):
        """
        time_steps_to_predict (B, Lp)
        X (B, M, L, N)
        truth_time_steps (B, M, L, N)
        mask (B, M, L, N)
        returns outputs (1, B, Lp, N)
        """
        B, M, L_in, N = X.shape
        self.batch_size = B

        Xr = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        ttr = truth_time_steps.permute(0, 3, 1, 2).reshape(-1, L_in, 1)
        mkr = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)

        te_his = self.LearnableTE(ttr)
        Xcat = torch.cat([Xr, te_his], dim=-1)

        # Debug: inputs
        self._dbg("forecast_in_Xcat", Xcat)
        self._dbg("forecast_in_mask", mkr)

        # Encoder
        h = self.IMTS_Model(Xcat, mkr)  # (B, N, hid_dim)
        self._dbg("forecast_h_before_repeat", h)

        # Decoder prep
        L_pred = time_steps_to_predict.shape[-1]
        h_rep = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)  # (B, N, Lp, hid_dim)

        ttp = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N, 1, 1)  # (B, N, Lp, 1)
        te_pred = self.LearnableTE(ttp)  # (B, N, Lp, te_dim)
        self._dbg("forecast_te_pred", te_pred)

        dec_in = torch.cat([h_rep, te_pred], dim=-1)  # (B, N, Lp, hid_dim+te_dim)
        self._dbg("forecast_dec_in", dec_in)

        # Decoder
        dec_raw = self.decoder(dec_in).squeeze(dim=-1)  # (B, N, Lp)
        self._dbg("forecast_dec_raw", dec_raw)

        outputs = dec_raw.permute(0, 2, 1).unsqueeze(dim=0)  # (1, B, Lp, N)
        self._dbg("forecast_outputs", outputs)

        return outputs
