import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

from models.conv.DWConv import get_dwconv_layer2d
from models.transformer.base_transformers import TransformerDecoder

class MaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_masks: int,
        window_size: int,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
    ):
        super(MaskDecoder, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_image_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.is_multitask = n_masks > 1
        self.n_masks = n_masks

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )

        self.proj = nn.Linear(embed_dim * window_size * 2, embed_dim)

        self.transformer = TransformerDecoder(n_layers=1, embed_dim=embed_dim, num_heads=12, attn_type='gqa', num_groups=2)
        self.seg_head = nn.Linear(embed_dim, n_masks * patch_size[0] * patch_size[1])

    def forward(
        self,
        x: torch.Tensor,
        sparse_prompt_tokens: torch.Tensor,
        dense_prompt_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # (B, N, D), (B, H, W, D)

        masks, attn_scores = self.predict_masks(
            x, sparse_prompt_tokens, dense_prompt_tokens
        )

        if self.is_multitask:
            masks = masks[:, 1:, :, :]

        return masks, attn_scores

    def predict_masks(
        self,
        x: torch.Tensor,
        sparse_prompt_tokens: torch.Tensor,
        dense_prompt_tokens: torch.Tensor,
    ):
        depth_tokens = rearrange(dense_prompt_tokens, 'b d h w ws -> b (d ws) h w')
        spatial_tokens = rearrange(dense_prompt_tokens, 'b d h w ws -> b (d ws) h w')
        spatial_tokens = F.adaptive_avg_pool2d(spatial_tokens, 1).expand_as(depth_tokens)

        tokens = self.proj(torch.cat([depth_tokens, spatial_tokens], dim=1))
        depth_tokens, spatial_tokens = torch.chunk(tokens, 2, dim=1)

        x += depth_tokens
        x *= spatial_tokens

        x, attn_scores = self.transformer(x, sparse_prompt_tokens)
        x = self.seg_head(x)
        x = rearrange(x, 'b n (c h w) -> b n c h w', c=self.n_masks, h=self.patch_size[0], w=self.patch_size[1])
        
        N_h, N_w = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

        x = rearrange(x, 'b (nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=N_h, nw=N_w)
        return x, attn_scores
