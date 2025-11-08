import torch
import torch.nn as nn

# --- Simple Mixer components ---

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_mult=4, drop=0.1):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )
    def forward(self, x):
        return self.net(x)

class TokenMixer(nn.Module):
    """Mix across sensor nodes/tokens."""
    def __init__(self, n_tokens, drop=0.1):
        super().__init__()
        self.proj1 = nn.Linear(n_tokens, n_tokens)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.proj2 = nn.Linear(n_tokens, n_tokens)
    def forward(self, x):  # x: (B, N, D)
        y = x.transpose(1, 2)               # (B, D, N)
        y = self.proj2(self.drop(self.act(self.proj1(y))))
        return y.transpose(1, 2)            # (B, N, D)

class ChannelMixer(nn.Module):
    """Mix across feature channels (D)."""
    def __init__(self, dim, hidden_mult=4, drop=0.1):
        super().__init__()
        self.ff = FeedForward(dim, hidden_mult=hidden_mult, drop=drop)
    def forward(self, x):
        return self.ff(x)

class NodeMixerBlock(nn.Module):
    """Full Mixer block (nodes × channels)."""
    def __init__(self, n_tokens, d_model, hidden_mult=4, drop=0.1):
        super().__init__()
        self.token_mixer = PreNorm(d_model, TokenMixer(n_tokens, drop))
        self.channel_mixer = PreNorm(d_model, ChannelMixer(d_model, hidden_mult, drop))
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x):  # (B, N, D)
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return self.norm_out(x)
