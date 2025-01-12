import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x: torch.Tensor):
        # x.shape = (N, B, D)
        self.pe = self.pe.to(x.device)
        return x + self.pe[:x.size(0), :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super().__init__()

        self.positional_encoding = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.positional_encoding.weight, -0.1, 0.1)

    def forward(self, x):
        # x.shape = (N, B, D)
        N, _, _ = x.size()
        positions = torch.arange(N, device=x.device).unsqueeze(1)
        pos_encoding = self.positional_encoding(positions).transpose(0, 1)
        return x + pos_encoding

