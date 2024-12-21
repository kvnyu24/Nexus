import torch
import torch.nn as nn
from typing import Tuple
from nexus.core.base import NexusModule


class RotaryEmbedding(NexusModule):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        sinusoid = torch.einsum('i,j->ij', position, inv_freq)
        self.register_buffer('sin', sinusoid.sin())
        self.register_buffer('cos', sinusoid.cos())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        return self.sin[:seq_len], self.cos[:seq_len]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
