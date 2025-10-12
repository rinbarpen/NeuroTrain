from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from typing import Sequence
from pathlib import Path
import cv2
import numpy as np
import os
from PIL import Image

from src.utils.llm.llm_utils import image_to_PIL
from src.utils.llm.chat_history import ChatHistory
from src.config import PRETRAINED_MODEL_DIR

LLAVA_MODEL_ID_DEFAULT = 'llava-hf/llava-1.5-7b-hf'

CLIP_MODEL_ID_DEFAULT = 'openai/clip-vit-base-patch32'
MEDCLIP_MODEL_ID_DEFAULT = 'medclip/medclip-vit-base-patch32'

DINO_MODEL_ID_DEFAULT = 'facebook/dino-vitb16'
SAM_MODEL_ID_DEFAULT = 'facebook/sam-vit-base'
SAM2_MODEL_ID_DEFAULT = 'facebook/sam-vit-huge'

# -------------- Additional common model IDs --------------
# LLaVA family
LLAVA_MODEL_ID_13B = 'llava-hf/llava-1.5-13b-hf'

# CLIP family (OpenAI)
CLIP_MODEL_ID_VIT_BASE_PATCH16 = 'openai/clip-vit-base-patch16'
CLIP_MODEL_ID_VIT_LARGE_PATCH14 = 'openai/clip-vit-large-patch14'

# OpenCLIP and SigLIP variants
OPENCLIP_MODEL_ID_BIGG_14 = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
SIGLIP_MODEL_ID_SO400M_384 = 'google/siglip-so400m-patch14-384'

# Biomedical CLIP
BIOMEDCLIP_MODEL_ID_BASE_16_224 = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

# DINOv2 family
DINOV2_MODEL_ID_BASE = 'facebook/dinov2-base'
DINOV2_MODEL_ID_LARGE = 'facebook/dinov2-large'

# DINOv3 family
DINOV3_MODEL_ID_VITS16_PRETRAIN_LVD1689M = 'facebook/dinov3-vits16-pretrain-lvd1689m'
DINOV3_MODEL_ID_VITB16_PRETRAIN_LVD1689M = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
DINOV3_MODEL_ID_VITL16_PRETRAIN_LVD1689M = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
DINOV3_MODEL_ID_VIT7B16_PRETRAIN_LVD1689M = 'facebook/dinov3-vit7b16-pretrain-lvd1689m'
DINOV3_MODEL_ID_CONVNEXT_TINY_PRETRAIN_LVD1689M = 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'

# SAM 2.1 (Hiera) variants
SAM21_MODEL_ID_HIERA_BASE_PLUS = 'facebook/sam2.1-hiera-base-plus'
SAM21_MODEL_ID_HIERA_LARGE = 'facebook/sam2.1-hiera-large'
SAM2_MODEL_ID_HIERA_LARGE = 'facebook/sam2-hiera-large'

# Grounded detection / open-vocabulary detection
GROUNDING_DINO_MODEL_ID_BASE = 'IDEA-Research/grounding-dino-base'
OWL_VIT_MODEL_ID_BASE_PATCH32 = 'google/owlvit-base-patch32'

# BLIP / BLIP-2
BLIP_MODEL_ID_CAPTION_BASE = 'Salesforce/blip-image-captioning-base'
BLIP2_MODEL_ID_OPT_2_7B = 'Salesforce/blip2-opt-2.7b'

# Florence-2 (multimodal)
FLORENCE2_MODEL_ID_LARGE = 'microsoft/Florence-2-large'

# Qwen2-VL (multimodal)
QWEN2_VL_MODEL_ID_2B_INSTRUCT = 'Qwen/Qwen2-VL-2B-Instruct'

# Phi-3.5 Vision (multimodal)
PHI35_VISION_MODEL_ID_INSTRUCT = 'microsoft/Phi-3.5-vision-instruct'

# LLaMA family (Meta)
LLAMA2_MODEL_ID_7B = 'meta-llama/Llama-2-7b-hf'
LLAMA2_MODEL_ID_7B_CHAT = 'meta-llama/Llama-2-7b-chat-hf'
LLAMA2_MODEL_ID_13B = 'meta-llama/Llama-2-13b-hf'
LLAMA2_MODEL_ID_13B_CHAT = 'meta-llama/Llama-2-13b-chat-hf'
LLAMA2_MODEL_ID_70B = 'meta-llama/Llama-2-70b-hf'
LLAMA2_MODEL_ID_70B_CHAT = 'meta-llama/Llama-2-70b-chat-hf'

# LLaMA 3 family
LLAMA3_MODEL_ID_8B = 'meta-llama/Meta-Llama-3-8B'
LLAMA3_MODEL_ID_8B_INSTRUCT = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLAMA3_MODEL_ID_70B = 'meta-llama/Meta-Llama-3-70B'
LLAMA3_MODEL_ID_70B_INSTRUCT = 'meta-llama/Meta-Llama-3-70B-Instruct'

# LLaMA 3.1 family
LLAMA31_MODEL_ID_8B = 'meta-llama/Llama-3.1-8B'
LLAMA31_MODEL_ID_8B_INSTRUCT = 'meta-llama/Llama-3.1-8B-Instruct'
LLAMA31_MODEL_ID_70B = 'meta-llama/Llama-3.1-70B'
LLAMA31_MODEL_ID_70B_INSTRUCT = 'meta-llama/Llama-3.1-70B-Instruct'
LLAMA31_MODEL_ID_405B = 'meta-llama/Llama-3.1-405B'
LLAMA31_MODEL_ID_405B_INSTRUCT = 'meta-llama/Llama-3.1-405B-Instruct'

# LLaMA 3.2 family
LLAMA32_MODEL_ID_1B = 'meta-llama/Llama-3.2-1B'
LLAMA32_MODEL_ID_1B_INSTRUCT = 'meta-llama/Llama-3.2-1B-Instruct'
LLAMA32_MODEL_ID_3B = 'meta-llama/Llama-3.2-3B'
LLAMA32_MODEL_ID_3B_INSTRUCT = 'meta-llama/Llama-3.2-3B-Instruct'

# -------- Model ID catalog (grouped by family/task) --------
# 说明：
# - 统一将已有的 model_id 常量按"任务大类"和"模型家族/子类"进行归档；
# - 仅做归类聚合，不改变原有常量命名，避免对外部引用造成破坏；
# - 你可以用 MODEL_ID_CATALOG['segmentation']['sam2.1'][0] 这样的方式快速取模型 ID；
MODEL_ID_CATALOG = {
    'llm': {  # Large Language Models / Text Generation
        'llama2': [LLAMA2_MODEL_ID_7B, LLAMA2_MODEL_ID_7B_CHAT, LLAMA2_MODEL_ID_13B, LLAMA2_MODEL_ID_13B_CHAT, LLAMA2_MODEL_ID_70B, LLAMA2_MODEL_ID_70B_CHAT],
        'llama3': [LLAMA3_MODEL_ID_8B, LLAMA3_MODEL_ID_8B_INSTRUCT, LLAMA3_MODEL_ID_70B, LLAMA3_MODEL_ID_70B_INSTRUCT],
        'llama3.1': [LLAMA31_MODEL_ID_8B, LLAMA31_MODEL_ID_8B_INSTRUCT, LLAMA31_MODEL_ID_70B, LLAMA31_MODEL_ID_70B_INSTRUCT, LLAMA31_MODEL_ID_405B, LLAMA31_MODEL_ID_405B_INSTRUCT],
        'llama3.2': [LLAMA32_MODEL_ID_1B, LLAMA32_MODEL_ID_1B_INSTRUCT, LLAMA32_MODEL_ID_3B, LLAMA32_MODEL_ID_3B_INSTRUCT],
    },
    'vlm': {  # Visual Language Models / Multimodal
        'llava': [LLAVA_MODEL_ID_DEFAULT, LLAVA_MODEL_ID_13B],
        'qwen2_vl': [QWEN2_VL_MODEL_ID_2B_INSTRUCT],
        'phi_vision': [PHI35_VISION_MODEL_ID_INSTRUCT],
        'florence2': [FLORENCE2_MODEL_ID_LARGE],
    },
    'image_encoder': {  # 图像编码/自监督表征
        'clip': [CLIP_MODEL_ID_DEFAULT, CLIP_MODEL_ID_VIT_BASE_PATCH16, CLIP_MODEL_ID_VIT_LARGE_PATCH14],
        'openclip': [OPENCLIP_MODEL_ID_BIGG_14],
        'siglip': [SIGLIP_MODEL_ID_SO400M_384],
        'dino': [DINO_MODEL_ID_DEFAULT],
        'dinov2': [DINOV2_MODEL_ID_BASE, DINOV2_MODEL_ID_LARGE],
        'dinov3': [
            DINOV3_MODEL_ID_VITS16_PRETRAIN_LVD1689M,
            DINOV3_MODEL_ID_VITB16_PRETRAIN_LVD1689M,
            DINOV3_MODEL_ID_VITL16_PRETRAIN_LVD1689M,
            DINOV3_MODEL_ID_VIT7B16_PRETRAIN_LVD1689M,
            DINOV3_MODEL_ID_CONVNEXT_TINY_PRETRAIN_LVD1689M,
        ],
    },
    'segmentation': {  # 分割/交互式分割
        'sam_v1': [SAM_MODEL_ID_DEFAULT],
        'sam2': [SAM2_MODEL_ID_HIERA_LARGE],
        'sam2.1': [SAM21_MODEL_ID_HIERA_BASE_PLUS, SAM21_MODEL_ID_HIERA_LARGE],
    },
    'detection': {  # 开放词汇检测/检测
        'grounding_dino': [GROUNDING_DINO_MODEL_ID_BASE],
        'owl_vit': [OWL_VIT_MODEL_ID_BASE_PATCH32],
    },
    'captioning': {  # 图像描述
        'blip': [BLIP_MODEL_ID_CAPTION_BASE],
        'blip2': [BLIP2_MODEL_ID_OPT_2_7B],
    },
    'biomedical': {  # 医学/生物医学领域模型
        'medclip': [MEDCLIP_MODEL_ID_DEFAULT],
        'biomedclip': [BIOMEDCLIP_MODEL_ID_BASE_16_224],
    },
}

# 常用默认选择（按任务大类挑一个默认 ID）
DEFAULT_MODEL_IDS = {
    'llm': LLAMA31_MODEL_ID_8B_INSTRUCT,  # 使用 LLaMA 3.1 8B Instruct 作为默认 LLM
    'vlm': LLAVA_MODEL_ID_DEFAULT,
    'image_encoder': DINO_MODEL_ID_DEFAULT,
    'segmentation': SAM_MODEL_ID_DEFAULT,
    'detection': GROUNDING_DINO_MODEL_ID_BASE,
    'captioning': BLIP_MODEL_ID_CAPTION_BASE,
    'biomedical': MEDCLIP_MODEL_ID_DEFAULT,
}

def build_transformers(model_id: str = LLAVA_MODEL_ID_DEFAULT, cache_dir: str=PRETRAINED_MODEL_DIR, device: str='auto', proxies: dict|None=None, **kwargs): 
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, device_map=device, proxies=proxies, **kwargs)
    except:
        model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, device_map=device, proxies=proxies, **kwargs)
    except:
        tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, device_map=device, proxies=proxies, **kwargs)
    except:
        processor = None
    return model, tokenizer, processor

def infer_vlm(model, tokenizer, processor, images: Image.Image|Path|str|cv2.Mat|Sequence[Image.Image|Path|str|cv2.Mat], text: str|ChatHistory, max_new_tokens: int=128, padding: str='max_length', truncation: bool=True, **kwargs):
    if isinstance(images, (str, Path, Image.Image, cv2.Mat)):
        images = [images]
    images = [image_to_PIL(img) for img in images]

    if isinstance(text, ChatHistory):
        text = processor.apply_chat_template(text.get_history(), add_generation_prompt=True)
    else:
        text = [text]

    inputs = processor(text=text, images=images, padding=padding, truncation=truncation, max_length=max_new_tokens, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def infer_llm(model, tokenizer, text: str, max_new_tokens: int=128, padding: str='max_length', truncation: bool=True, **kwargs):
    if isinstance(text, ChatHistory):
        t = ""
        for message in text.get_history():
            if message['role'] == 'user':
                t += "<s>[INST]{}[/INST]".format(message['content'][0]['text'])
            else:
                t += "{}</s>".format(message['content'][0]['text'])
        text = t
    else:
        text = [text]

    inputs = tokenizer(text=text, padding=padding, truncation=truncation, max_length=max_new_tokens, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
