import torch
from torch import nn
from ..attention.MultiHeadAttention import MultiHeadAttention, MultiHeadCrossAttention, AttentionType

from ..attention.attention_mask import get_attn_mask
from ..norm.RMSNorm import RMSNorm

class _MLP(nn.Module):
    def __init__(self, embed_dim: int, r: int, mlp_dropout: float=0.2, mlp_act=nn.LeakyReLU):
        super(_MLP, self).__init__()

        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * r),
            nn.Dropout(mlp_dropout),
            mlp_act(inplace=True),
            nn.Linear(embed_dim * r, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, embed_dim: int, num_heads: int, r: int, attn_dropout: float=0.2, attn_out_dropout: float=0.0, mlp_dropout: float=0.3, mlp_out_dropout: float=0.3, mlp_act=nn.LeakyReLU, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, rms_norm=False, *, share_kv=False, rope=False, pre_norm=True):
        super(TransformerEncoder, self).__init__()
        self.pre_norm = pre_norm

        self.n_layers = n_layers
        self.attn = MultiHeadAttention(embed_dim, num_heads, num_groups=num_groups, attn_type=attn_type, attn_dropout=attn_dropout, qkv_bias=qkv_bias, share_kv=share_kv, rope=rope)
        self.attn_dropout = nn.Dropout(attn_out_dropout)
        self.mlp = _MLP(embed_dim=embed_dim, r=r, mlp_dropout=mlp_dropout, mlp_act=mlp_act)
        self.mlp_dropout = nn.Dropout(mlp_out_dropout)
        self.norm1 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_weights = []
        if self.pre_norm:
            for _ in range(self.n_layers):
                x = self.norm1(x)
                x, attn_weight = self.attn(x)
                x = self.attn_dropout(x) + x
                x = self.norm2(x)
                x = self.mlp(x)
                x = self.mlp_dropout(x) + x

                attn_weights.append(attn_weight)
        else:
            for _ in range(self.n_layers):
                x, attn_weight = self.attn(x)
                x = self.attn_dropout(x) + x
                x = self.norm1(x)
                x = self.mlp(x)
                x = self.mlp_dropout(x) + x
                x = self.norm2(x)

                attn_weights.append(attn_weight)

        return x, attn_weights

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers: int, embed_dim: int, num_heads: int, r: int, attn_dropout: float=0.2, attn_out_dropout: float=0.0, mlp_dropout: float=0.3, mlp_out_dropout: float=0.3, mlp_act=nn.LeakyReLU, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, rms_norm=False, *, share_kv=False, rope=False, pre_norm=True):
        super(TransformerDecoder, self).__init__()

        self.pre_norm = pre_norm

        self.n_layers = n_layers
        self.attn = MultiHeadAttention(embed_dim, num_heads, num_groups=num_groups, attn_type=attn_type, attn_dropout=attn_dropout, qkv_bias=qkv_bias, share_kv=share_kv, rope=rope)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, num_groups=num_groups, attn_type=attn_type, attn_dropout=attn_dropout, qkv_bias=qkv_bias, rope=rope)
        self.attn_dropout = nn.Dropout(attn_out_dropout)
        self.mlp = _MLP(embed_dim=embed_dim, r=r, mlp_dropout=mlp_dropout, mlp_act=mlp_act)
        self.mlp_dropout = nn.Dropout(mlp_out_dropout)
        self.norm1 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim) if not rms_norm else RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        attn_weights = []
        if self.pre_norm:
            for _ in range(self.n_layers):
                x = self.norm1(x)
                x, attn_weight = self.attn(x)
                x = self.attn_dropout(x) + x
                x = self.norm2(x)
                x, attn_weight_cross = self.attn(x, mask=get_attn_mask(x.shape[1], 'all'))
                x = self.attn_dropout(x) + x
                x = self.norm3(x)
                x = self.mlp(x)
                x = self.mlp_dropout(x) + x

                attn_weights.append((attn_weight, attn_weight_cross))
        else:
            for _ in range(self.n_layers):
                x, attn_weight = self.attn(x)
                x = self.attn_dropout(x) + x
                x = self.norm1(x)
                x, attn_weight_cross = self.attn(x, mask=get_attn_mask(x.shape[1], 'all'))
                x = self.attn_dropout(x) + x
                x = self.norm2(x)
                x = self.mlp(x)
                x = self.mlp_dropout(x) + x
                x = self.norm3(x)

                attn_weights.append((attn_weight, attn_weight_cross))

        return x, attn_weights
