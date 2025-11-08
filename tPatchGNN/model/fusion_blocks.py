import torch
import torch.nn as nn

class ScaleAttentionFusion(nn.Module):
    def __init__(self, d_in, d_hid=128, temperature_init=1.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hid)
        self.att  = nn.Linear(d_hid, 1, bias=False)
        self.tau  = nn.Parameter(torch.tensor(float(temperature_init)))  # learnable temperature

    @staticmethod
    def masked_gap(x, m, eps=1e-6):
        num = (x * m).sum(dim=(1, 2))             # (B, D)
        den = m.sum(dim=(1, 2)).clamp_min(eps)    # (B, D)
        return num / den

    def forward(self, H_list, M_list):
        logits, zs = [], []
        for Hk, Mk in zip(H_list, M_list):
            pooled = self.masked_gap(Hk, Mk)
            z = torch.tanh(self.proj(pooled))
            zs.append(z)
            logits.append(self.att(z))

        logits = torch.cat(logits, dim=1)           # (B, K)
        tau = self.tau.clamp_min(0.1)
        attw = torch.softmax(logits / tau, dim=1)   # (B, K)

        fused = torch.zeros_like(zs[0])
        for k, zk in enumerate(zs):
            fused = fused + attw[:, k:k+1] * zk
        return fused, attw
