import torch
from torch import nn
from axial_attention import AxialAttention, AxialImageTransformer, AxialPositionalEmbedding

def build_axial_pe_3d(dim: int, shape: tuple[int, ...]):
    return AxialPositionalEmbedding(dim=dim, shape=shape)

def build_axial_attention_3d(dim: int, num_heads: int=8):
    return AxialAttention(
        dim=dim, heads=num_heads, dim_index=2, num_dimensions=3)

def build_axial_transformer_3d(dim: int, depth: int, shape: tuple[int, ...], num_heads: int=8, reversible: bool=True):
    return AxialImageTransformer(dim=dim, depth=depth, heads=num_heads, reversible=reversible, axial_pos_emb_shape=shape)
