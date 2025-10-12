import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class LLinear(nn.Module):
    def __init__(self, llm_hidden_dim, seg_hidden_dim, from_llm_to_seg=True, dropout=0.0, bias=True):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(llm_hidden_dim, seg_hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(seg_hidden_dim, seg_hidden_dim, bias=bias),
        ) if from_llm_to_seg else nn.Sequential(
            nn.Linear(seg_hidden_dim, llm_hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim, bias=bias),
        )
        self.dropout = nn.Dropout(dropout)
        self.from_llm_to_seg = from_llm_to_seg

    def forward(self, x, hidden_feat_shape: tuple[int, int]|None=None):
        if self.from_llm_to_seg:
            # (B, N, D) -> (B, H, W, D)
            x = self.proj(x)
            if self.training:
                x = self.dropout(x)
            if hidden_feat_shape is not None:
                x = rearrange(x, 'b (h w) d -> b h w d', h=hidden_feat_shape[0], w=hidden_feat_shape[1])
            return x
        else:
            x = rearrange(x, 'b h w d -> b (h w) d')
            x = self.proj(x)
            if self.training:
                x = self.dropout(x)
            return x
