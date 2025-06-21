import torch
from torch import nn
import torch.nn.functional as F
import timm
from typing import Optional
from peft import LoraModel, LoraConfig, LoraRuntimeConfig, get_peft_model

from .utils import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from models.embedding import PatchEmbeddingWithPE

class ImageEncoder(nn.Module):
    def __init__(self, image_size: int|tuple[int, int], patch_size: int|tuple[int, int], n_classes: int, n_channels: int=1, model_config: dict={
        'embed_dim': 768,
        'build_model_fn': build_sam_vit_b,
    }):
        super(ImageEncoder, self).__init__()
        
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        self.image_size = image_size
        self.n_classes = n_classes
        
        self.embed_dim = embed_dim = model_config['embed_dim']
        self.encoder = model_config['build_model_fn']()
        self.encoder = self.encoder.image_encoder.blocks

        # self.encoder = timm.create_model(model_name, pretrained=True, cache_dir=f"data/cache/pretrained/{model_name}")
        self.patch_embed = PatchEmbeddingWithPE(n_channels, embed_dim, patch_size, max_num_patch=(image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]))
        # self.encoder = self.encoder.blocks

        self.lora_finetune()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.encoder(x)
        return x

    def lora_finetune(self):
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
            task_type="IMAGE_SEGMENTATION",
        )

        self.encoder = get_peft_model(self.encoder, lora_config)
