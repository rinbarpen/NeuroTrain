import torch
from torch import nn
from einops import rearrange

def pair(x: int|tuple[int, int]):
    if isinstance(x, int):
        x = (x, x) 
    return x

class ImageReconstruction(nn.Module):
    def __init__(self, embed_dim: int, n_channels: int, patch_size: int|tuple[int, int]):
        super(ImageReconstruction, self).__init__()

        self.patch_h, self.patch_w = pair(patch_size)
        self.n_channels = n_channels
        self.embed_dim = embed_dim

        self.proj = nn.Linear(embed_dim, n_channels * self.patch_h * self.patch_w)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]):
        B, N, D = x.shape
        H_out, W_out = output_size

        x = self.proj(x)

        x = x.view(B, N, self.n_channels, self.patch_h, self.patch_w)
        N_h, N_w = H_out // self.patch_h, W_out // self.patch_w

        if N != N_h * N_w:
            raise ValueError(f"{N} != {N_h} * {N_w}")

        x = rearrange(x, 'b (nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=N_h, nw=N_w)
        return x
