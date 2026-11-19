import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:pe[0, :, 1::2].shape[-1]])
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x, return_attention=False):
        z = self.norm1(x)
        a, weights = self.attn(z, z, z, need_weights=return_attention,
                               average_attn_weights=False)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, weights


class ClimateTransformer(nn.Module):
    def __init__(self, token_size=84, tokens_out=1, dim=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError('A dimensão do embedding precisa ser divisível pelo número de cabeças.')
        self.tokens_out = tokens_out
        self.token_size = token_size
        self.embed = nn.Linear(token_size, dim)
        self.position = PositionalEncoding(dim)
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, tokens_out * token_size)

    def forward(self, x, return_attention=False):
        x = self.position(self.embed(x))
        weights = None
        for i, block in enumerate(self.blocks):
            x, weights = block(x, return_attention=return_attention and i == len(self.blocks) - 1)
        last = self.norm(x[:, -1, :])
        pred = self.output(last).reshape(-1, self.tokens_out, self.token_size)
        return (pred, weights) if return_attention else pred
