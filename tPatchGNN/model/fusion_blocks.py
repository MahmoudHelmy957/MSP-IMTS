import torch
import torch.nn as nn


class NodeMixerBlock(nn.Module):
    """
    A simple node-wise fusion block that mixes features across nodes and channels.

    Args:
        n_tokens (int): number of nodes
        d_model (int): feature dimension
        hidden_mult (float): expansion factor for MLP width
        drop (float): dropout rate
    """

    def __init__(self, n_tokens: int, d_model: int, hidden_mult: float = 2.0, drop: float = 0.1):
        super().__init__()
        hidden_dim = int(d_model * hidden_mult)

        # Token mixer (mix information across nodes)
        self.token_mixer = nn.Sequential(
            nn.Linear(n_tokens, n_tokens),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        # Channel mixer (mix across feature channels per node)
        self.channel_mixer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x = x + self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixer(x)
        return x
