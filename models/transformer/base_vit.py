import torch
from torch import nn
from ..embedding import PatchEmbeddingWithPE, PatchEmbedding
from ..position_encoding import PositionalEncoding, LearnablePositionalEncoding
from ..transformer.base_transformers import TransformerEncoder, TransformerDecoder
from ..attention.MultiHeadAttention import AttentionType 

def pair(x: int|tuple[int, int]):
    if isinstance(x, int):
        x = (x, x) 
    return x


class BaseViT(nn.Module):
    def __init__(self, n_channels: int, patch_size: int|tuple[int, int], image_size: int|tuple[int, int], n_layers: int, embed_dim: int, num_heads: int, r: int, attn_dropout: float=0.2, attn_out_dropout: float=0.0, mlp_dropout: float=0.3, mlp_out_dropout: float=0.3, mlp_act=nn.LeakyReLU, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, rms_norm=False, *, use_learnable=False):
        super(BaseViT, self).__init__()

        patch_size = pair(patch_size)
        image_size = pair(image_size)

        max_num_patch = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) 
        self.pe = PatchEmbeddingWithPE(n_channels, embed_dim, patch_size, max_num_patch, use_learnable=use_learnable)
        self.blocks = TransformerEncoder(n_layers, embed_dim, num_heads, r, attn_dropout, attn_out_dropout, mlp_dropout, mlp_out_dropout, mlp_act, num_groups, attn_type, qkv_bias, rms_norm)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.pe(x)
        return self.blocks(x)

class CrossViT(nn.Module):
    def __init__(self, n_channels: int, patch_size: int|tuple[int, int], image_size: int|tuple[int, int], n_layers: int, embed_dim: int, num_heads: int, r: int, attn_dropout: float=0.2, attn_out_dropout: float=0.0, mlp_dropout: float=0.3, mlp_out_dropout: float=0.3, mlp_act=nn.LeakyReLU, num_groups: int|None=None, attn_type: AttentionType='mha', qkv_bias=True, rms_norm=False, *, use_learnable=False):
        super(CrossViT, self).__init__()

        patch_size = pair(patch_size)
        image_size = pair(image_size)

        max_num_patch = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) 
        self.pe = PatchEmbeddingWithPE(n_channels, embed_dim, patch_size, max_num_patch, use_learnable=use_learnable)
        self.blocks = TransformerDecoder(n_layers, embed_dim, num_heads, r, attn_dropout, attn_out_dropout, mlp_dropout, mlp_out_dropout, mlp_act, num_groups, attn_type, qkv_bias, rms_norm)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x, y, z = self.pe(x), self.pe(y), self.pe(z)
        return self.blocks(x, y, z)
