import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional

from models.attention.ECAModule import ECAModule

from lib.SAM_Med2D.segment_anything.modeling import image_encoder

class ImageEncoder(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, n_classes: int, embed_dim: int, n_layer: int, num_heads: int, mlp_ratio: float):
        self.encoder = image_encoder.ImageEncoderViT(img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim, depth=n_layer, num_heads=num_heads, mlp_ratio=mlp_ratio, out_chans=n_classes)
