import torch
from torch import nn
import torch.nn.functional as F
import timm
from typing import Optional
from peft import LoraModel, LoraConfig, LoraRuntimeConfig, get_peft_model

from .utils import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from models.position_encoding import PositionalEncoding, SpatialPositionEmbedding
from segment_anything.modeling.image_encoder import PatchEmbed as SAMPatchEmbedding


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        n_classes: int,
        n_channels: int = 1,
        out_channels: int = 256,
        model_config: dict = {
            "embed_dim": 768,
            "build_model_fn": build_sam_vit_b,
        },
    ):
        super(ImageEncoder, self).__init__()

        image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

        self.image_size = image_size
        self.n_classes = n_classes

        self.embed_dim = embed_dim = model_config["embed_dim"]
        sam_model = model_config["build_model_fn"]().image_encoder
        self.patch_embed = SAMPatchEmbedding(kernel_size=self.patch_size, stride=self.patch_size, in_chans=n_channels, embed_dim=embed_dim)
        self.blocks = sam_model.blocks
        self.neck = sam_model.neck

        self.pos_embed = nn.Parameter(torch.zeros(1, self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1], embed_dim))

        self.adapter = Adapter_Layer(embed_dim=embed_dim)

        # self.lora_finetune()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x += self.pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.neck(x)
        x = self.adapter(x)
        return x

    # def lora_finetune(self):
    #     lora_config = LoraConfig(
    #         r=4,
    #         lora_alpha=16,
    #         target_modules=["qkv", "proj", "fc1", "fc2"],
    #         lora_dropout=0.05,
    #         bias="none",
    #         task_type="IMAGE_SEGMENTATION",
    #     )

    #     self.encoder = get_peft_model(self.encoder, lora_config)

class Adapter_Layer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer = nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim, bias=False),
                nn.Sigmoid()
        )

        self.spatial = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, x):
        #x -> （B, H, W, C）-> （B, C, H, W）
        x = x.permute(0,3,1,2)
        B, C, _, _ = x.size()
        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x
        x_spatial = self.spatial(x_channel)
        
        if self.skip_connect:
            x = x + x_spatial
        else:
            x = x_spatial
        #（B, C, H, W） -> (B, H, W, C)
        x = x.permute(0,2,3,1)
        return self.norm(x)
