import logging
import torch
from pathlib import Path
from typing import Optional
import time
from functools import wraps

from torchvision import models
import timm
from huggingface_hub import snapshot_download as hf_download

from src.constants import PRETRAINED_MODEL_DIR
from src.utils.annotation import retry


# 支持的torchvision模型
TORCHVISION_MODEL_MAP = {
    # AlexNet
    'alexnet': models.alexnet,
    # VGG
    'vgg11': models.vgg11,
    'vgg11_bn': models.vgg11_bn,
    'vgg13': models.vgg13,
    'vgg13_bn': models.vgg13_bn,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'vgg19': models.vgg19,
    'vgg19_bn': models.vgg19_bn,
    # ResNet
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
    # SqueezeNet
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
    # DenseNet
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    # Inception
    'inception_v3': models.inception_v3,
    'googlenet': models.googlenet,
    # ShuffleNet V2
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    # MobileNet
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    # MNASNet
    'mnasnet0_5': models.mnasnet0_5,
    'mnasnet1_0': models.mnasnet1_0,
    # EfficientNet
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5,
    'efficientnet_b6': models.efficientnet_b6,
    'efficientnet_b7': models.efficientnet_b7,
    # RegNet
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    'regnet_y_3_2gf': models.regnet_y_3_2gf,
    'regnet_y_8gf': models.regnet_y_8gf,
    'regnet_y_16gf': models.regnet_y_16gf,
    'regnet_y_32gf': models.regnet_y_32gf,
    'regnet_x_400mf': models.regnet_x_400mf,
    'regnet_x_800mf': models.regnet_x_800mf,
    'regnet_x_1_6gf': models.regnet_x_1_6gf,
    'regnet_x_3_2gf': models.regnet_x_3_2gf,
    'regnet_x_8gf': models.regnet_x_8gf,
    'regnet_x_16gf': models.regnet_x_16gf,
    'regnet_x_32gf': models.regnet_x_32gf,
    # Vision Transformer
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_l_16': models.vit_l_16,
    'vit_l_32': models.vit_l_32,
    'vit_h_14': models.vit_h_14,
}


def download_and_save(model_name: str, desired_provider: Optional[str] = None):
    CACHE_DIR = Path(PRETRAINED_MODEL_DIR)
    logger = logging.getLogger('downloader')

    @retry()
    def from_torchvision():
        if model_name in TORCHVISION_MODEL_MAP:
            model_fn = TORCHVISION_MODEL_MAP[model_name]
            logger.info(f"Downloading torchvision model: {model_name}...")
            save_dir = CACHE_DIR / model_name
            save_dir.mkdir(parents=True, exist_ok=True)
            model_fn(pretrained=True, cache_dir=save_dir)
            logger.info(f"Saved {model_name} weights to {save_dir}")
            return save_dir
        raise ValueError(f"Model not found in torchvision: {model_name}")

    @retry()
    def from_timm():
        logger.info(f"Downloading timm model: {model_name}...")
        save_dir = CACHE_DIR / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        timm.create_model(model_name, pretrained=True, cache_dir=save_dir)
        logger.info(f"Saved {model_name} weights to {save_dir}")
        return save_dir

    @retry()
    def from_huggingface():
        logger.info(f"Downloading huggingface model: {model_name}...")
        save_dir = CACHE_DIR / model_name.replace('/', '／')
        save_dir.mkdir(parents=True, exist_ok=True)
        hf_download(
            repo_id=model_name,
            local_dir=save_dir,
            cache_dir=save_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"Saved {model_name} files to {save_dir}")
        return save_dir

    provider_map = {
        'torchvision': from_torchvision,
        'timm': from_timm,
        'huggingface': from_huggingface,
    }

    if desired_provider:
        if desired_provider not in provider_map:
            raise ValueError(f"Unsupported provider: {desired_provider}")
        try:
            return provider_map[desired_provider]()
        except Exception as e:
            logger.error(f"Failed to download {model_name} from {desired_provider}: {e}")
            raise
    else:
        for provider, download_func in provider_map.items():
            try:
                logger.info(f"Trying provider: {provider}")
                return download_func()
            except Exception as e:
                logger.warning(f"Failed with provider {provider}: {e}")

    raise ValueError(f"Failed to download {model_name} from any provider.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import argparse
    parser = argparse.ArgumentParser(description='Download pretrained backbone weights')
    parser.add_argument('--model', type=str, required=True, help='Model name from torchvision, timm or huggingface')
    parser.add_argument('--provider', type=str, choices=['torchvision', 'timm', 'huggingface'], default=None, help='The desired provider to download from')
    args = parser.parse_args()
    download_and_save(args.model, args.provider)
