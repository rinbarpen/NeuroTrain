import torch
from torch import nn

from .utils import get_model_config
from .image_encoder import ImageEncoder
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder

class Net(nn.Module):
    # def __init__(self, n_classes: int, embed_dim: int, n_layers: int, num_heads: int, image_size: int|tuple[int, int], patch_size: int|tuple[int, int]=14, window_size: int=1):
    def __init__(self, n_classes: int, model_name: str='sam_vit_b', image_size: int|tuple[int, int]=224, patch_size: int|tuple[int, int]=14, window_size: int=1, n_tokens_per_mask: int=1):
        super(Net, self).__init__()

        model_config = get_model_config(model_name)
        embed_dim = model_config['embed_dim']

        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.n_classes = n_classes
        self.n_masks = n_classes + 1 if n_classes > 1 else 1

        self.windows_size = window_size

        self.image_encoder = ImageEncoder(image_size=image_size, patch_size=patch_size, n_channels=1, n_classes=n_classes, model_config=model_config)

        patched_image_size=(self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.prompt_encoder = PromptEncoder(embed_dim=embed_dim, image_size=self.image_size, patched_image_size=patched_image_size, mask_channels=embed_dim, n_masks=self.n_classes, n_tokens_per_mask=n_tokens_per_mask)

        self.mask_decoder = MaskDecoder(n_masks=self.n_masks, window_size=window_size, image_size=self.image_size, embed_dim=embed_dim, patch_size=self.patch_size)

    def forward(self, depth_x: torch.Tensor, points: torch.Tensor|None=None, bboxes: torch.Tensor|None=None):
        # (B, C, H, W, D)
        if depth_x.dim() >= 5:
            prev_x = []
            for x in depth_x.unbind(dim=-1):
                x = self.image_encoder(x)
                sparse_tokens, dense_tokens = self.prompt_encoder(masks=torch.stack(prev_x[len(prev_x)-self.window_size if len(prev_x)>=self.window_size else 0:], dim=-1))
                x = self.mask_decoder(x, sparse_tokens, dense_tokens)
                prev_x.append(x)
            return torch.stack(prev_x, dim=-1)
        else:
            x = depth_x
            x = self.image_encoder(x)
            
            sparse_tokens, dense_tokens = self.prompt_encoder()
            x = self.mask_decoder(x, sparse_tokens, dense_tokens)
            return x

# try to disable sparse_tokens while points or bboxes prompt is provided
