import torch
import torch.nn as nn

class ScaleAttentionFusion(nn.Module):
    def __init__(self, d_in, d_hid=128, temperature_init=1.0):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hid)
        self.att  = nn.Linear(d_hid, 1, bias=False)
        self.tau  = nn.Parameter(torch.tensor(float(temperature_init)))

    def forward(self, Z_list):
        """
        Z_list: list of K tensors, each (B, D)
        Returns fused (B, d_hid), attw (B, K)
        """
        logits, zs = [], []
        for Z in Z_list:
            z = torch.tanh(self.proj(Z))     # (B, d_hid)
            zs.append(z)
            logits.append(self.att(z))       # (B, 1)

        logits = torch.cat(logits, dim=1)    # (B, K)
        attw = torch.softmax(logits / self.tau.clamp_min(0.1), dim=1)  # (B, K)

        fused = torch.zeros_like(zs[0])
        for k, zk in enumerate(zs):
            fused = fused + attw[:, k:k+1] * zk
        return fused, attw
