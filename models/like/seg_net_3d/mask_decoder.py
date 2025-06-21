import torch
from torch import nn
from einops import rearrange

from models.conv.DWConv import get_dwconv_layer2d
from models.transformer.base_transformers import TransformerDecoder

class MaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        depth_proj_lite: bool=True, 
    ):
        super(MaskDecoder, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_image_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.is_multitask = n_classes > 1
        if n_classes > 1:
            n_classes += 1
        self.n_classes = n_classes

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )

        self.transformer = TransformerDecoder(n_layers=1, embed_dim=embed_dim, num_heads=12, attn_type='gqa', num_groups=2)

        self.seg_head = nn.Linear(embed_dim, n_classes * patch_size[0] * patch_size[1])

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
        h, w = dense_prompt_tokens.shape[-2], dense_prompt_tokens.shape[-1]
        _sparse_prompt_tokens = sparse_prompt_tokens
        _dense_prompt_tokens = rearrange(dense_prompt_tokens, 'b d h w -> b (h w) d')
        _n_spt = _sparse_prompt_tokens.shape[1]
        _n_dpt = _dense_prompt_tokens.shape[1]

        y = self.mlp(torch.cat([_sparse_prompt_tokens, _dense_prompt_tokens], dim=1))
        _sparse_prompt_tokens, _dense_prompt_tokens = y[:_n_spt], y[_n_spt:]
        sparse_prompt_tokens = _sparse_prompt_tokens
        dense_prompt_tokens = rearrange(_dense_prompt_tokens, 'b n d -> b d (h w)', h=h, w=w)

        x += dense_prompt_tokens
        x = x.permute(0, 3, 1, 2)
        x = rearrange(x, 'b d h w -> b (h w) d')

        x, attn_scores = self.transformer(x, sparse_prompt_tokens)
        x = self.seg_head(x)
        x = rearrange(x, 'b n (c h w) -> b n c h w', c=self.n_classes, h=self.patch_size[0], w=self.patch_size[1])
        
        N_h, N_w = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

        x = rearrange(x, 'b (nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=N_h, nw=N_w)
        return x, attn_scores
