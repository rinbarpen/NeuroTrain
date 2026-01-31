"""
ViT with Grid and Fixed patch size, retaining spatial info in the transformer.
Uses 2D positional encoding and optional 2D feature map output for downstream tasks.
"""
from __future__ import annotations

import torch
from torch import nn

from ..embedding import PatchEmbedding


def _pair(x):
    return (x, x) if isinstance(x, int) else x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if (heads != 1 or dim_head != dim)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).flatten(-2)
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ])
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTGrid(nn.Module):
    """
    Vision Transformer with fixed patch size and 2D grid structure.
    Retains spatial info via 2D positional encoding; can output 2D feature map for segmentation/detection.
    """

    def __init__(
        self,
        *,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        image_h, image_w = _pair(image_size)
        patch_h, patch_w = _pair(patch_size)
        assert image_h % patch_h == 0 and image_w % patch_w == 0
        self.num_patches_h = image_h // patch_h
        self.num_patches_w = image_w // patch_w
        num_patches = self.num_patches_h * self.num_patches_w

        self.patch_embed = PatchEmbedding(channels, dim, patch_size=(patch_h, patch_w), use_cls_token=False)
        self.pos_embed_2d = nn.Parameter(torch.randn(1, self.num_patches_h, self.num_patches_w, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img: torch.Tensor, return_2d: bool = False) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W)
            return_2d: If True, return (B, H', W', dim) feature map; else return (B, num_classes) logits.
        """
        x = self.patch_embed(img)
        B, N, D = x.shape
        x = x.view(B, self.num_patches_h, self.num_patches_w, D)
        x = x + self.pos_embed_2d
        x = self.dropout(x)
        x = x.view(B, N, D)
        x = self.transformer(x)
        if return_2d:
            return x.view(B, self.num_patches_h, self.num_patches_w, D)
        x = x.mean(dim=1)
        return self.mlp_head(x)


def vit_grid_tiny_patch16_224(num_classes: int, **kwargs) -> ViTGrid:
    return ViTGrid(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        **kwargs,
    )


def vit_grid_small_patch16_224(num_classes: int, **kwargs) -> ViTGrid:
    return ViTGrid(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        **kwargs,
    )


def vit_grid_base_patch16_224(num_classes: int, **kwargs) -> ViTGrid:
    return ViTGrid(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs,
    )


def vit_grid_base_patch32_224(num_classes: int, **kwargs) -> ViTGrid:
    return ViTGrid(
        image_size=224,
        patch_size=32,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs,
    )
