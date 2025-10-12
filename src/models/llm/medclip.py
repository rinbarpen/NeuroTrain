import pdb
import os
import copy
from collections import defaultdict
import requests

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from torchvision import transforms as mtf
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

from libs.MedCLIP.medclip import *

def init_medclip(model: MedCLIPModel, checkpoint_dir: str|None = None):
    model.from_pretrained(checkpoint_dir)

def process(images: np.ndarray|list[Image.Image], text: str|list[str]):
    processor = MedCLIPProcessor()
    inputs = processor(
        text=text, 
        images=images, 
        return_tensors="pt", 
        padding=True
    )
    return inputs

IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

def norm_input_image(img: np.ndarray|Image.Image):
    if isinstance(img, np.ndarray):
        x = Image.fromarray(img)

    x = torch.from_numpy(x)  # torch.Tensor (H, W)

    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    x = F.interpolate(
        x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=True
    )  # (1,1,224,224)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)  # (1,3,224,224)
    x = (x - IMG_MEAN) / IMG_STD
    return x

def norm_input_text(model: MedCLIPModel, texts: str|list[str]):
    return model.text_model.tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    )["input_ids"]

def similarity(model: MedCLIPModel, feats1: torch.Tensor, feats2: torch.Tensor):
    return model.compute_logits(feats1, feats2)

# encode_text
# encode_image
# forward(text_feats, image_feats, return_loss=True)
# ->
# {'loss_value': loss, 'text_embeds': text_embeds, 'img_embeds': image_embeds, 'logits': logits_per_image, 'logits_per_text': logits_per_text}
# 
def build_medclip(checkpoint: str|None = None, vision_checkpoint: str|None=None):
    model = MedCLIPModel(MedCLIPVisionModelViT, checkpoint=checkpoint, vision_checkpoint=vision_checkpoint)
    return model

def build_medclip_text():
    model = MedCLIPTextModel()
    return model

def build_medclip_vision(checkpoint: str|None = None):
    model = MedCLIPVisionModel(checkpoint)
    return model

def build_medclip_vision_vit(checkpoint: str|None = None, medclip_checkpoint: str|None=None):
    model = MedCLIPVisionModelViT(checkpoint, medclip_checkpoint)
    return model
