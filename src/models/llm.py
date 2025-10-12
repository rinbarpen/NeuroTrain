from transformers import AutoTokenizer, CLIPVisionModel, LlavaForConditionalGeneration, LlavaProcessor
from transformers import SamModel, SamProcessor
from transformers.models.sam.modeling_sam import SamAttention

from peft import get_peft_model, LoraConfig
from einops import rearrange, einsum

import torch

LLAVA_MODELS = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "liuhaotian/llava-v1.5-7b",
    "liuhaotian/llava-v1.5-13b",
    "liuhaotian/llava-llama-2-13b-chat-lightning-preview",
    "liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview",
    "liuhaotian/LLaVA-Lightning-7B-delta-v1-1",
    "liuhaotian/LLaVA-7b-delta-v0",
]
VISION_MODELS = [
    "openai/clip-vit-large-patch14",
]
SAM_MODELS = [
    "facebook/sam-vit-base",
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2-hiera-large",
    "facebook/sam2-hiera-small",
    "facebook/sam2-hiera-tiny",
    "facebook/sam2.1-hiera-large",
    "facebook/sam2.1-hiera-small",
    "sam_vit_l_0b3195.pth",
    "sam_vit_h_4b8939.pth",
    "sam_vit_b_01ec64.pth",
]

def get_llm_model(model_id: str, cache_dir: str='./cache/models/pretrained', dtype=torch.float16, device_map: str='auto', force_download: bool=False, **kwargs):
    model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, 
    trust_remote_code=True, torch_dtype=dtype, device_map=device_map, force_download=force_download, **kwargs)
    processor = LlavaProcessor.from_pretrained(model_id, cache_dir=cache_dir, 
    trust_remote_code=True, torch_dtype=dtype, device_map=device_map, force_download=force_download, **kwargs)
    return model, processor

def get_vision_model(model_id: str, cache_dir: str='./cache/models/pretrained', dtype=torch.float16, device_map: str='auto', force_download: bool=False, **kwargs):
    model = CLIPVisionModel.from_pretrained(model_id, cache_dir=cache_dir, 
    trust_remote_code=True, torch_dtype=dtype, device_map=device_map, force_download=force_download, **kwargs)
    return model
