# REF
from pathlib import Path
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget, SoftmaxOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, vgg19
from timm import get_pretrained_cfg, create_model
import torch
from torch import nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal

from utils.transform import build_image_transforms
from models.sample.unet import UNet
from models.transformer.deit_vit import vit_base_patch16_224

def unet_check(image: Path, mask: np.ndarray, is_rgb: bool, *, target_category: int=0) -> None:
    # Load a pre-trained ResNet50 model
    model = UNet(1, 1, True)
    model.load_state_dict(torch.load(r'..\results\train\unet\weights\best_model.pth')['model'])
    model.eval()

    # Choose a target layer (e.g., the last convolutional layer)
    target_layers = [model.layer4[-1]]

    # Load and preprocess an image
    image = Image.open(image)
    image = image.convert('RGB') if is_rgb else image.convert('L')

    transform = build_image_transforms(resize=(224, 224), is_pil_image=True, is_rgb=is_rgb)
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the model and input tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    image_np = np.array(image) / 255.0

    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    
    targets = [SemanticSegmentationTarget(0, mask)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

def resnet50_check(image: Path, is_rgb: bool, *, target_category: int=0) -> None:
    # Load a pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    model.eval()

    # Choose a target layer (e.g., the last convolutional layer)
    target_layers = [model.layer4[-1]]

    # Load and preprocess an image
    image = Image.open(image)
    image = image.convert('RGB') if is_rgb else image.convert('L')

    transform = build_image_transforms(resize=(224, 224), is_pil_image=True, is_rgb=is_rgb)
    input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Move the model and input tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    image_np = np.array(image) / 255.0

    # image_np: (H, W, C) format
    H, W = image_np.shape[0], image_np.shape[1]
    targets = [ClassifierOutputTarget(target_category)]
    
    cam = GradCAM(model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor, targets=targets, eigen_smooth=False)[0, :]
    
    # 确保grayscale_cam的大小与image_np匹配
    grayscale_cam = cv2.resize(grayscale_cam, (W, H))  # 注意这里是 (W, H)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)

    return Image.fromarray(cam_image, mode='RGB')


class ImageHeatMapGenerator():
    def __init__(self, model: nn.Module, target_layers: list[nn.Module], targets: list, model_params_file: str|Path|None=None):
        self.model = model
        if model_params_file:
            self.model.load_state_dict(torch.load(model_params_file))
        self.model.eval()
        self.target_layers = target_layers
        self.targets = targets

    def check(self, image: Image.Image, transform_fn: transforms.Compose, is_rgb=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if is_rgb:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        H, W = image.size

        input_tensor = transform_fn(image).unsqueeze(0)
        image_np = np.array(image).astype(np.float32)

        self.model = self.model.to(device)
        input_tensor = input_tensor.to(device)

        with GradCAMPlusPlus(self.model, target_layers=self.target_layers) as cam: 
            grayscale_cam = cam(input_tensor, targets=self.targets, eigen_smooth=False)[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (W, H))

            cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=is_rgb)
            heatmap_image = Image.fromarray(cam_image, mode='RGB' if is_rgb else 'L')
        return heatmap_image

    def check_batch(self, images: list[Image.Image], transform_fn: transforms.Compose, is_rgb=True):
        return [self.check(image, transform_fn, is_rgb=is_rgb) for image in images]

    # image = Image.open(img_path).convert('L' or 'RGB')
    def check_transformer_block_attention_heatmap(self, image: Image.Image, attn_scores: torch.Tensor, patch_size: tuple[int, int]|int, alpha: float = 0.5, head_type: Literal['mean', 'max']='mean'):
        original_image_size = image.size # (H, W)
        # attention
        # attn_scores is (B, H, N, N)
        if head_type == 'mean':
            attn_scores = attn_scores.mean(dim=(0, 1))
        elif head_type == 'max':
            attn_scores = attn_scores.max(dim=(0, 1))

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size) # (h, w)

        attn_map = attn_scores.reshape(patch_size)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        attn_map = cv2.resize(attn_map, original_image_size, interpolation=cv2.INTER_LINEAR)

        # heatmap
        colormap = plt.get_cmap("viridis")
        heatmap = colormap(attn_map)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        image = np.array(image, dtype=np.uint8)
        fused_image = cv2.addWeighted(heatmap, alpha, image, 1.0 - alpha, 0, dtype=np.uint8)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(image)
        # plt.imshow(attn_map, cmap="viridis", alpha=alpha)
        # plt.axis("off")
        # plt.title("Attention Map Overlay")
        # plt.show()

        return fused_image, heatmap # (fused, attn_heatmap) all with uint8
