# TODO: Test
from torch import nn

from .variant_scale_attention import VariantScaleImageAttention


class FeedForward(nn.Module):
    def __init__(self, n_channels, inner_channels, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(n_channels, inner_channels, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inner_channels, n_channels, bias=True)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class VariantVisionTransformerEncoder(nn.Module):
    def __init__(self, n_channels, hidden_dim, qkv_channels, patch_size, num_head, dropout, inner_channels):
        super(VariantVisionTransformerEncoder, self).__init__()

        self.image_attention = VariantScaleImageAttention(
            n_channels, hidden_dim, qkv_channels, patch_size, num_head, dropout
        )
        self.linear = FeedForward(hidden_dim, inner_channels, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        a = self.image_attention(x)
        a += x
        a = self.norm(a)
        o = self.linear(a)
        o += a
        o = self.norm(o)
        return o
